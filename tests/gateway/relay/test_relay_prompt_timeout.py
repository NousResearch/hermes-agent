"""Relay prompt lifetimes: a prompt carries the wait its caller actually applies.

A connector may close a ``prompt`` that has no ``timeout_s`` when the turn ends. A
slash confirm is asked outside any agent turn and waits
``tools.slash_confirm.DEFAULT_TIMEOUT_SECONDS``, so its prompt must say so on the
wire, and the gateway's own pending-prompt expiry must agree. Other prompt kinds
keep their current shape.
"""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command
from tools import slash_confirm
from tools.slash_confirm import DEFAULT_TIMEOUT_SECONDS

from tests.gateway.relay.stub_connector import StubConnector
from tests.waiting import async_wait_until

_UNTIMED_PROMPT_TTL_S = 3600.0


def _desc() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="telegram", label="Telegram",
        max_message_length=4096, supports_draft_streaming=False, supports_edit=True,
        supports_threads=True, markdown_dialect="markdown_v2", len_unit="utf16",
        supported_ops=("send", "edit", "typing", "get_chat_info", "prompt", "react"),
    )


def _adapter() -> tuple[RelayAdapter, StubConnector]:
    stub = StubConnector(_desc())
    return RelayAdapter(PlatformConfig(), _desc(), transport=stub), stub


def _prompt_ops(stub: StubConnector) -> list[dict]:
    return [a for a in stub.sent if a.get("op") == "prompt"]


def _assert_local_expiry(adapter: RelayAdapter, prompt_id: str, ttl: float, before: float, after: float) -> None:
    expires_at = adapter._pending_prompts[prompt_id]["expires_at"]
    assert before + ttl <= expires_at <= after + ttl


@pytest.mark.asyncio
async def test_slash_confirm_prompt_carries_the_confirm_wait_on_the_wire_and_locally():
    adapter, stub = _adapter()
    before = time.time()
    result = await adapter.send_slash_confirm("c1", "/new", "Discards history.", "sess:1", "7")
    after = time.time()

    assert result.success is True
    (action,) = _prompt_ops(stub)
    assert action["timeout_s"] == DEFAULT_TIMEOUT_SECONDS
    _assert_local_expiry(adapter, action["prompt_id"], DEFAULT_TIMEOUT_SECONDS, before, after)


@pytest.mark.asyncio
async def test_exec_approval_and_clarify_prompts_keep_their_shape():
    adapter, stub = _adapter()
    before = time.time()
    await adapter.send_exec_approval("c1", "rm -rf /tmp/x", "sess:1")
    await adapter.send_clarify("c1", "Which one?", ["a", "b"], "cl-1", "sess:1")
    after = time.time()

    approval, clarify = _prompt_ops(stub)
    assert (approval["prompt_kind"], clarify["prompt_kind"]) == ("approval", "clarify")
    for action in (approval, clarify):
        assert "timeout_s" not in action
        _assert_local_expiry(adapter, action["prompt_id"], _UNTIMED_PROMPT_TTL_S, before, after)


@pytest.mark.asyncio
async def test_a_press_after_the_confirm_wait_gets_the_expiry_notice_not_an_approval(monkeypatch):
    clock = [1000.0]
    # Both the adapter's prompt registry and tools.slash_confirm read the wall clock.
    monkeypatch.setattr(time, "time", lambda: clock[0])
    adapter, stub = _adapter()
    ran: list[str] = []

    async def _handler(choice: str) -> str:
        ran.append(choice)
        return "session reset"

    slash_confirm.clear("sess:1")
    slash_confirm.register("sess:1", "7", "new", _handler)
    try:
        await adapter.send_slash_confirm("c1", "/new", "Discards history.", "sess:1", "7")
        (prompt,) = _prompt_ops(stub)
        clock[0] += DEFAULT_TIMEOUT_SECONDS + 1
        press = MessageEvent(
            text="/once", message_type=MessageType.COMMAND,
            source=SessionSource(platform="telegram", chat_id="c1", chat_type="dm", user_id="u1"),
            prompt_response={"prompt_id": prompt["prompt_id"], "option_id": "once"},
        )
        assert await adapter._consume_prompt_response(press) is True
        acks = await async_wait_until(
            lambda: [a["content"] for a in stub.sent if a.get("op") == "send"], what="the press to be acked"
        )
    finally:
        slash_confirm.clear("sess:1")

    assert ran == []
    assert len(acks) == 1 and "no longer waiting" in acks[0]


# ── every gateway entry into a slash confirm reaches the wire timed ───────


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.COMMAND,
        source=SessionSource(platform="telegram", chat_id="c1", chat_type="dm", user_id="u1"),
    )


def _runner(adapter: RelayAdapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._delivery_adapter_for = lambda _source: adapter
    runner._thread_metadata_for_source = lambda _source, _anchor: None
    runner._reply_anchor_for_event = lambda _event: None
    runner._session_key_for_source = lambda _source: "sess:1"
    runner._read_user_config = dict
    runner._async_profile_scope_for_source = lambda _source: contextlib.nullcontext()
    runner._is_telegram_topic_root_lobby = lambda _source: False
    return runner


async def _dispatch(runner, text: str) -> None:
    event = _event(text)
    canonical = resolve_command(event.get_command()).name
    handled, _reply = await runner._hm_dispatch_canonical_command(event, event.source, "sess:1", canonical)
    assert handled is True


async def _model_guard(runner, monkeypatch) -> None:
    from gateway.slash_commands_model import _ModelSwitchContext

    monkeypatch.setattr(
        "hermes_cli.model_selection_guards.combined_selection_warning",
        lambda *a, **k: SimpleNamespace(title="Expensive model", message="Costs a lot."),
    )
    monkeypatch.setattr("hermes_cli.model_selection_guards.selection_context_for_agent", lambda _agent: None)
    runner._cached_agent_for = lambda _key: None
    event = _event("/model pricey")
    ctx = _ModelSwitchContext(session_key="sess:1", source=event.source, config_path=None, persist_global=False)
    switch = SimpleNamespace(new_model="pricey", target_provider="openrouter", base_url="", api_key="", model_info=None)
    fired, _reply = await runner._model_selection_guard_reply(event, ctx, switch)
    assert fired is True


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/new", "/reset", "/undo", "/reload-mcp", "model-guard"])
async def test_each_slash_confirm_entry_sends_a_timed_prompt(text, monkeypatch):
    adapter, stub = _adapter()
    runner = _runner(adapter)
    slash_confirm.clear("sess:1")
    try:
        if text == "model-guard":
            await _model_guard(runner, monkeypatch)
        else:
            await _dispatch(runner, text)
        assert slash_confirm.get_pending("sess:1") is not None
    finally:
        slash_confirm.clear("sess:1")

    (action,) = _prompt_ops(stub)
    assert adapter._pending_prompts[action["prompt_id"]]["kind"] == "slash_confirm"
    assert action["timeout_s"] == DEFAULT_TIMEOUT_SECONDS
