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
async def test_exec_approval_without_prompt_op_fails_for_text_fallback():
    adapter, stub = _adapter(supported_ops=("send", "edit", "typing"))
    result = await adapter.send_exec_approval("c1", "cmd", "s")
    # success=False → gateway/run.py falls back to the text approval prompt
    # (same contract as a failed native button send).
    assert result.success is False
    assert all(a["op"] != "prompt" for a in stub.sent)
    assert adapter._pending_prompts == {}  # nothing left pending


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


@pytest.mark.asyncio
async def test_clarify_open_ended_uses_base_text_path(monkeypatch):
    adapter, stub = _adapter()
    # No choices → base class question-only text send (no prompt op).
    result = await adapter.send_clarify("c1", "What now?", None, "cl-2", "sess:1")
    assert result.success is True
    assert all(a["op"] != "prompt" for a in stub.sent)


@pytest.mark.asyncio
async def test_prompt_decline_degrades_clarify_to_numbered_text(monkeypatch):
    adapter, stub = _adapter()
    stub.next_prompt_result = {"success": False, "error": "nope"}
    marked: list[str] = []
    monkeypatch.setattr(
        "tools.clarify_gateway.mark_awaiting_text", lambda cid: marked.append(cid)
    )
    result = await adapter.send_clarify(
        "c1", "Which?", ["a", "b"], "cl-3", "sess:1"
    )
    # Falls back to the base numbered-text clarify (a plain send).
    assert result.success is True
    assert stub.sent[-1]["op"] == "send"
    assert "1. a" in stub.sent[-1]["content"]
    assert marked == ["cl-3"]
    assert adapter._pending_prompts == {}


# ── the pending-prompt registry ──────────────────────────────────────────


def test_registry_mint_consume_once_and_expiry():
    adapter, _stub = _adapter()
    pid = adapter._mint_prompt("exec_approval", {"session_key": "s"}, timeout_s=3600)
    assert adapter._pop_prompt(pid) is not None
    assert adapter._pop_prompt(pid) is None  # one answer wins
    stale = adapter._mint_prompt("exec_approval", {"session_key": "s"}, timeout_s=-1)
    assert adapter._pop_prompt(stale) is None  # expired misses


# ── inbound consumption ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_response_resolves_exec_approval(monkeypatch):
    adapter, stub = _adapter()
    await adapter.send_exec_approval("c1", "cmd", "sess:9")
    prompt_id = stub.sent[-1]["prompt_id"]

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "tools.approval.resolve_gateway_approval",
        lambda sk, choice, **kw: calls.append((sk, choice)) or 1,
    )
    event = _event({"prompt_id": prompt_id, "option_id": "session"})
    consumed = await adapter._consume_prompt_response(event)
    assert consumed is True
    assert calls == [("sess:9", "session")]
    # Consumed prompts leave the registry; the ack landed as a plain send.
    assert prompt_id not in adapter._pending_prompts
    assert stub.sent[-1]["op"] == "send"
    assert "session" in stub.sent[-1]["content"].lower()


@pytest.mark.asyncio
async def test_prompt_response_resolves_slash_confirm(monkeypatch):
    adapter, stub = _adapter()
    await adapter.send_slash_confirm("c1", "T", "msg", "sess:9", "cf-1")
    prompt_id = stub.sent[-1]["prompt_id"]

    resolved: list[tuple] = []

    async def fake_resolve(session_key, confirm_id, choice, **kw):
        resolved.append((session_key, confirm_id, choice))
        return "done!"

    monkeypatch.setattr("tools.slash_confirm.resolve", fake_resolve)
    event = _event({"prompt_id": prompt_id, "option_id": "always"})
    assert await adapter._consume_prompt_response(event) is True
    assert resolved == [("sess:9", "cf-1", "always")]
    # The handler's result text went out as a follow-up send.
    sends = [a for a in stub.sent if a["op"] == "send"]
    assert any("done!" in a["content"] for a in sends)


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


@pytest.mark.asyncio
async def test_unknown_or_expired_prompt_falls_through():
    adapter, _stub = _adapter()
    event = _event({"prompt_id": "deadbeef", "option_id": "once"})
    assert await adapter._consume_prompt_response(event) is False
    assert await adapter._consume_prompt_response(_event(None)) is False
    # Malformed shapes never consume.
    assert await adapter._consume_prompt_response(_event({"prompt_id": ""})) is False


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


def test_discord_foreign_custom_id_keeps_legacy_text_shape():
    adapter, _stub = _adapter()

    class Forward:
        platform = "discord"
        method = "POST"
        path = "/interactions/bot1"
        body = (
            b'{"type": 3, "id": "i1", "channel_id": "ch1", "guild_id": "g1",'
            b' "data": {"custom_id": "someones_button"}}'
        )

    event = adapter._discord_interaction_to_event(Forward())
    assert event is not None
    assert event.prompt_response is None
    assert event.text == "someones_button"
    assert event.message_type == MessageType.TEXT


def test_decode_prompt_token_matches_connector_codec():
    adapter, _stub = _adapter()
    assert adapter._decode_prompt_token("hp1:p1:deny") == ("p1", "deny")
    assert adapter._decode_prompt_token("ea:once:3") is None
    assert adapter._decode_prompt_token("hp1:p1") is None
    assert adapter._decode_prompt_token("hp1:bad id:x") is None
    assert adapter._decode_prompt_token("") is None


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


@pytest.mark.asyncio
async def test_processing_failure_reacts_cross():
    adapter, stub = _adapter()
    event = _reactable_event()
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    emojis = [a["emoji"] for a in stub.sent if a["op"] == "react"]
    assert emojis[-1] == "❌"


@pytest.mark.asyncio
async def test_react_is_op_gated_and_best_effort():
    adapter, stub = _adapter(supported_ops=("send", "edit", "typing"))
    event = _reactable_event()
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert all(a["op"] != "react" for a in stub.sent)  # never hit the wire
    # And a connector decline never raises.
    adapter2, stub2 = _adapter()
    stub2.next_react_result = {"success": False, "error": "nope"}
    await adapter2.on_processing_start(event)  # must not raise


@pytest.mark.asyncio
async def test_cancelled_outcome_removes_eyes_without_verdict():
    adapter, stub = _adapter()
    event = _reactable_event()
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)
    reacts = [(a["emoji"], a.get("remove", False)) for a in stub.sent if a["op"] == "react"]
    assert reacts == [("👀", True)]  # eyes removed, no ✅/❌
