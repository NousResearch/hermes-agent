"""Relay Phase 3 interactive tests — prompt op egress, prompt_response
consumption, and the react ack lifecycle.

Covers:
  - send_exec_approval / send_slash_confirm / send_clarify render through ONE
    `prompt` op with the right option sets, honoring op gating (legacy
    connectors get the base/text behaviour or a structured failure that
    triggers run.py's text fallback);
  - the pending-prompt registry: mint → consume-once → expiry;
  - _consume_prompt_response routes answers to the approval / slash-confirm /
    clarify resolvers and CONSUMES the event; unknown/expired ids fall
    through to normal dispatch;
  - the Discord type-3 hp1 decode (structured prompt_response replacing the
    bare-custom_id stub; foreign custom_ids keep the legacy text shape);
  - on_processing_start/complete drive react ops (👀 → ✅/❌), op-gated and
    best-effort.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource

from tests.gateway.relay.stub_connector import StubConnector

FULL_OPS = (
    "send",
    "edit",
    "typing",
    "get_chat_info",
    "send_media",
    "prompt",
    "react",
)


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        supported_ops=FULL_OPS,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _adapter(**desc_kw) -> tuple[RelayAdapter, StubConnector]:
    stub = StubConnector(make_desc(**desc_kw))
    adapter = RelayAdapter(PlatformConfig(), make_desc(**desc_kw), transport=stub)
    return adapter, stub


def _event(
    prompt_response: Optional[Dict[str, Any]] = None,
    text: str = "/once",
    chat_id: str = "c1",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform="telegram", chat_id=chat_id, chat_type="dm", user_id="u1"
        ),
        prompt_response=prompt_response,
    )


# ── egress: the three prompt surfaces ────────────────────────────────────


@pytest.mark.asyncio
async def test_exec_approval_renders_full_option_set():
    adapter, stub = _adapter()
    result = await adapter.send_exec_approval(
        "c1", "rm -rf /tmp/x", "sess:1", description="deletes files"
    )
    assert result.success is True
    assert result.message_id == "pm1"
    action = stub.sent[-1]
    assert action["op"] == "prompt"
    assert action["prompt_kind"] == "approval"
    ids = [o["id"] for o in action["options"]]
    assert ids == ["once", "session", "always", "deny"]
    assert "rm -rf /tmp/x" in action["content"]
    assert "deletes files" in action["content"]
    # The registry holds the pending prompt keyed by the wire's prompt_id.
    assert action["prompt_id"] in adapter._pending_prompts
    state = adapter._pending_prompts[action["prompt_id"]]
    assert state["kind"] == "exec_approval"
    assert state["session_key"] == "sess:1"


@pytest.mark.asyncio
async def test_exec_approval_smart_denied_and_flag_gating():
    adapter, stub = _adapter()
    await adapter.send_exec_approval(
        "c1", "cmd", "s", smart_denied=True, allow_permanent=True, allow_session=True
    )
    ids = [o["id"] for o in stub.sent[-1]["options"]]
    assert ids == ["once", "deny"]  # smart-deny: no session/always
    await adapter.send_exec_approval(
        "c1", "cmd", "s", allow_session=True, allow_permanent=False
    )
    ids = [o["id"] for o in stub.sent[-1]["options"]]
    assert ids == ["once", "session", "deny"]


@pytest.mark.asyncio
async def test_slash_confirm_renders_three_options():
    adapter, stub = _adapter()
    result = await adapter.send_slash_confirm(
        "c1", "Reload MCP", "This invalidates the prompt cache.", "sess:1", "cf-9"
    )
    assert result.success is True
    action = stub.sent[-1]
    ids = [o["id"] for o in action["options"]]
    assert ids == ["once", "always", "cancel"]
    assert "Reload MCP" in action["content"]
    state = adapter._pending_prompts[action["prompt_id"]]
    assert state == {
        **state,
        "kind": "slash_confirm",
        "confirm_id": "cf-9",
        "session_key": "sess:1",
    }


@pytest.mark.asyncio
async def test_clarify_renders_choices_plus_other_with_positional_ids():
    adapter, stub = _adapter()
    result = await adapter.send_clarify(
        "c1",
        "Which environment?",
        ["staging — the safe one", "production"],
        "cl-1",
        "sess:1",
    )
    assert result.success is True
    action = stub.sent[-1]
    assert action["prompt_kind"] == "clarify"
    ids = [o["id"] for o in action["options"]]
    # Positional ids (choice text is arbitrary UTF-8; ids must be callback-safe).
    assert ids == ["c0", "c1", "other"]
    labels = [o["label"] for o in action["options"]]
    assert labels[0].startswith("staging")
    state = adapter._pending_prompts[action["prompt_id"]]
    assert state["choices"] == ["staging — the safe one", "production"]


# ── the pending-prompt registry ──────────────────────────────────────────


# ── inbound consumption ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_response_resolves_clarify_choice_and_other(monkeypatch):
    adapter, stub = _adapter()
    await adapter.send_clarify("c1", "Which?", ["alpha", "beta"], "cl-9", "s")
    prompt_id = stub.sent[-1]["prompt_id"]

    resolved: list[tuple] = []
    marked: list[str] = []
    monkeypatch.setattr(
        "tools.clarify_gateway.resolve_gateway_clarify",
        lambda cid, resp: resolved.append((cid, resp)) or True,
    )
    monkeypatch.setattr(
        "tools.clarify_gateway.mark_awaiting_text", lambda cid: marked.append(cid)
    )
    # Positional id maps back to the REAL choice text.
    event = _event({"prompt_id": prompt_id, "option_id": "c1"})
    assert await adapter._consume_prompt_response(event) is True
    assert resolved == [("cl-9", "beta")]

    # "Other" flips to text capture.
    await adapter.send_clarify("c1", "Which?", ["a"], "cl-10", "s")
    prompt_id2 = stub.sent[-1]["prompt_id"]
    event2 = _event({"prompt_id": prompt_id2, "option_id": "other"})
    assert await adapter._consume_prompt_response(event2) is True
    assert marked == ["cl-10"]


# ── Discord type-3 hp1 decode ────────────────────────────────────────────


def test_discord_component_interaction_decodes_prompt_token():
    adapter, _stub = _adapter()

    class Forward:
        platform = "discord"
        method = "POST"
        path = "/interactions/bot1"
        body = (
            b'{"type": 3, "id": "i1", "channel_id": "ch1", "guild_id": "g1",'
            b' "message": {"id": "pm55"},'
            b' "member": {"user": {"id": "u1", "username": "ben"}},'
            b' "data": {"custom_id": "hp1:a1b2c3d4:deny"}}'
        )

    event = adapter._discord_interaction_to_event(Forward())
    assert event is not None
    assert event.prompt_response == {
        "prompt_id": "a1b2c3d4",
        "option_id": "deny",
        "prompt_message_id": "pm55",
    }
    assert event.text == "/deny"
    assert event.message_type == MessageType.COMMAND


# ── react ack lifecycle ──────────────────────────────────────────────────


def _reactable_event() -> MessageEvent:
    return MessageEvent(
        text="do something",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform="discord",
            chat_id="ch1",
            chat_type="channel",
            user_id="u1",
            message_id="m42",
        ),
        message_id="m42",
    )


@pytest.mark.asyncio
async def test_processing_lifecycle_reacts_eyes_then_check():
    adapter, stub = _adapter()
    event = _reactable_event()
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    reacts = [a for a in stub.sent if a["op"] == "react"]
    assert [(r["emoji"], r.get("remove", False)) for r in reacts] == [
        ("👀", False),
        ("👀", True),
        ("✅", False),
    ]
    assert all(r["message_id"] == "m42" and r["chat_id"] == "ch1" for r in reacts)


# ── fanned-out prompt answers (one press, many gateways) ─────────────────
#
# The connector delivers a passthrough forward (a Discord button press) to
# EVERY live gateway session of the tenant, unlike a message, which it narrows
# to the admitted instance set. So one press reaches every sibling gateway
# while only the minting one can resolve it. These pin that a non-owner stays
# silent and that the owner still answers exactly once.


@pytest.mark.asyncio
async def test_sibling_gateway_ignores_another_instances_prompt_answer(monkeypatch):
    """A press for a prompt this process didn't mint is consumed silently."""
    owner, owner_stub = _adapter()
    sibling, sibling_stub = _adapter()
    await owner.send_clarify("c1", "Which?", ["alpha", "beta"], "cl-1", "s")
    prompt_id = owner_stub.sent[-1]["prompt_id"]

    resolved: list[tuple] = []
    monkeypatch.setattr(
        "tools.clarify_gateway.resolve_gateway_clarify",
        lambda cid, resp: resolved.append((cid, resp)) or True,
    )
    monkeypatch.setattr("tools.clarify_gateway.mark_awaiting_text", lambda cid: None)

    event = _event({"prompt_id": prompt_id, "option_id": "c1"})
    # Consumed (True) so the "/c1"-shaped text is never dispatched as chat --
    # that fall-through is what produced one "Unknown command `/c1`" per
    # sibling gateway. And the sibling neither resolves nor says anything.
    assert await sibling._consume_prompt_response(event) is True
    assert resolved == []
    assert sibling_stub.sent == []

    # The owner still resolves the same press normally.
    assert await owner._consume_prompt_response(event) is True
    assert resolved == [("cl-1", "beta")]


@pytest.mark.asyncio
async def test_repeat_answer_for_resolved_prompt_is_ignored(monkeypatch):
    """A double tap / redelivered forward must not resolve twice."""
    adapter, stub = _adapter()
    await adapter.send_clarify("c1", "Which?", ["alpha", "beta"], "cl-2", "s")
    prompt_id = stub.sent[-1]["prompt_id"]

    resolved: list[tuple] = []
    monkeypatch.setattr(
        "tools.clarify_gateway.resolve_gateway_clarify",
        lambda cid, resp: resolved.append((cid, resp)) or True,
    )
    event = _event({"prompt_id": prompt_id, "option_id": "c0"})
    assert await adapter._consume_prompt_response(event) is True
    assert resolved == [("cl-2", "alpha")]

    sent_after_first = len(stub.sent)
    assert await adapter._consume_prompt_response(event) is True
    assert resolved == [("cl-2", "alpha")]  # not resolved a second time
    assert len(stub.sent) == sent_after_first  # and no second ack / notice


@pytest.mark.asyncio
async def test_expired_own_prompt_notifies_instead_of_unknown_command():
    """An expired prompt of OURS gets an expiry notice, not chat dispatch.

    Falling through would hand run.py a command-shaped "/c1", which is not a
    real command, so the user got "Unknown command `/c1`".
    """
    adapter, stub = _adapter()
    prompt_id = adapter._mint_prompt("clarify", {"chat_id": "c1"}, timeout_s=-1.0)

    event = _event({"prompt_id": prompt_id, "option_id": "c1"})
    assert await adapter._consume_prompt_response(event) is True
    # The notice is fire-and-forget now (read-loop self-deadlock fix:
    # awaiting a send from _consume_prompt_response blocks the very read
    # loop that resolves the send's result future). Yield so the
    # background ack task runs before asserting egress.
    await asyncio.sleep(0.05)
    notices = [a for a in stub.sent if a["op"] == "send"]
    assert len(notices) == 1
    assert "no longer waiting" in notices[0]["content"]


def test_minted_prompt_ids_are_instance_scoped_and_callback_safe():
    """Ids carry the minting process's nonce and stay codec-legal.

    The connector's promptCodec validates each id as [A-Za-z0-9_.-]{1,32} and
    caps "hp1:<prompt_id>:<option_id>" at Telegram's 64-byte callback budget.
    """
    import re

    a, _ = _adapter()
    b, _ = _adapter()
    id_a = a._mint_prompt("clarify", {"chat_id": "c1"})
    id_b = b._mint_prompt("clarify", {"chat_id": "c1"})

    assert re.fullmatch(r"[A-Za-z0-9_.\-]{1,32}", id_a)
    assert len(f"hp1:{id_a}:option_id_up_to_32_chars_here") <= 64
    assert a._minted_here(id_a) is True
    assert b._minted_here(id_a) is False
    assert a._minted_here(id_b) is False
    # A legacy id minted before the nonce existed (no "." segment) is still
    # treated as ours, so a prompt in flight across an upgrade resolves.
    assert a._minted_here("a1b2c3d4") is True
