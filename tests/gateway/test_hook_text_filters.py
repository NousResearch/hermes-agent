"""Contract tests for the gateway's collectable text-filter hooks.

A subscribed hook may replace the user text the agent sees
(``agent:message:filter``) or the assistant text that goes out
(``agent:response:filter``). Both call sites funnel through the shared
``apply_collectable_text_filter`` helper, so these tests pin its contract:
the first valid replacement wins, malformed returns are ignored, and a
subscriber that raises never breaks the turn (fail-open).
"""

import contextlib
from typing import Any, cast

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn import apply_collectable_text_filter
from gateway.session import SessionSource

# Reused harness: it drives the REAL in-band queued (/queue) drain through
# ``GatewayRunner._run_agent`` with a fake AIAgent — the exact path whose coverage the review
# asked to confirm.
from tests.gateway.test_queued_followup_processing_hooks import (
    SESSION_KEY,
    HookRecordingAdapter,
    _install_fake_agent,
    _make_runner,
    _source,
    _TwoTurnAgent,
)


class _FakeHooks:
    """Minimal stand-in for the hook registry: only ``emit_collect`` is used."""

    def __init__(self, results=(), raises=None):
        self._results = results
        self._raises = raises
        self.calls = []

    async def emit_collect(self, event_type, context):
        self.calls.append((event_type, context))
        if self._raises is not None:
            raise self._raises
        return self._results


@pytest.mark.asyncio
async def test_no_hooks_leaves_text_untouched():
    """No subscribers (empty result list) must change neither text path."""
    hooks = _FakeHooks(results=[])

    message = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {"platform": "telegram"}, "message", "original message",
    )
    response = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {"platform": "telegram"}, "response", "original response",
    )

    assert message == "original message"
    assert response == "original response"


@pytest.mark.asyncio
async def test_message_filter_replacement_is_applied():
    """A ``{"message": ...}`` result replaces the text the agent sees."""
    hooks = _FakeHooks(results=[{"message": "redacted message"}])

    result = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {"session_id": "s1"}, "message", "secret message",
    )

    assert result == "redacted message"


@pytest.mark.asyncio
async def test_first_valid_result_wins():
    """With several subscribers, the first valid replacement is the one applied."""
    hooks = _FakeHooks(results=[{"response": "first"}, {"response": "second"}])

    result = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {"session_id": "s1"}, "response", "original response",
    )

    assert result == "first"


@pytest.mark.asyncio
async def test_malformed_hook_results_are_ignored():
    """Non-dict returns, wrong value types, and missing keys are skipped silently."""
    hooks = _FakeHooks(results=[
        "not a dict",
        {"response": 123},          # present but not a string
        {"other_key": "value"},     # key absent
        None,
    ])

    result = await apply_collectable_text_filter(
        hooks, "agent:response:filter", {}, "response", "untouched",
    )

    assert result == "untouched"


@pytest.mark.asyncio
async def test_raising_hook_is_fail_open():
    """A subscriber blowing up returns the original text instead of propagating."""
    hooks = _FakeHooks(raises=RuntimeError("plugin exploded"))

    result = await apply_collectable_text_filter(
        hooks, "agent:message:filter", {}, "message", "untouched",
    )

    assert result == "untouched"


# ── Where the filters are emitted: the single turn funnel ────────────────────────────────
# Both turn entry points pass through ``GatewayRunner._run_agent``: the idle-message handler
# (``_handle_message_with_agent``) and the in-band drain of a message that arrived while a turn
# was still running (``_run_agent_queued_followup``). An emission placed in one caller silently
# skips the other — and the /queue message is precisely the one a user typed while the agent was
# busy, which is the PII case this hook pair exists for.


class _RecordingFilterHooks:
    """Hook registry stand-in: records every emission and rewrites both texts."""

    loaded_hooks = True

    def __init__(self, message_replacement=None, response_prefix=None, raises=None):
        self.events: list = []
        self._message = message_replacement
        self._prefix = response_prefix
        self._raises = raises

    async def emit(self, event_type, context):  # only recorded; nothing subscribes to it here
        self.events.append((event_type, dict(context)))

    async def emit_collect(self, event_type, context):
        self.events.append((event_type, dict(context)))
        if self._raises is not None:
            raise self._raises
        if event_type == "agent:message:filter" and self._message is not None:
            return [{"message": self._message}]
        if event_type == "agent:response:filter" and self._prefix is not None:
            return [{"response": self._prefix + (context.get("response") or "")}]
        return []


# The funnel is called unbound so a stub can stand in for the runner: cast keeps the type
# checker out of the way without weakening the production signature.
_run_agent_unbound = cast(Any, GatewayRunner._run_agent)


class _FunnelStub:
    """Just enough runner for the real ``GatewayRunner._run_agent`` (called unbound) to run."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.seen: list = []

    def _profile_scope_for_source(self, source):
        return contextlib.nullcontext()

    async def _run_agent_inner(self, message, context_prompt, history, source, session_id, **turn_kwargs):
        self.seen.append(message)
        return {"final_response": f"done:{message}"}


@pytest.mark.asyncio
async def test_the_turn_funnel_filters_both_directions():
    """The model reads the inbound filter's output; the funnel returns the outbound one's."""
    hooks = _RecordingFilterHooks(message_replacement="[redactado]", response_prefix="revelado:")
    stub = _FunnelStub(hooks)

    result = await _run_agent_unbound(
        stub, message="mi dni es 12345678Z", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        session_id="s-filtros",
    )

    assert stub.seen == ["[redactado]"]
    assert result["final_response"] == "revelado:done:[redactado]"

    # One emission per direction, per turn: the move must not double-filter.
    assert [event for event, _ in hooks.events] == [
        "agent:message:filter", "agent:response:filter",
    ]
    context = hooks.events[0][1]
    assert context["session_id"] == "s-filtros"
    assert context["platform"] == "telegram"


@pytest.mark.asyncio
async def test_the_turn_funnel_tolerates_a_runner_without_a_hook_registry():
    """Proxy dispatch builds a runner with no hook registry at all: the turn must still run.

    Regression: emitting the filters on the funnel put a ``self.hooks`` read on the path every
    turn takes, including runners that never carry a registry (``test_proxy_mode``).
    """
    stub = _FunnelStub(_RecordingFilterHooks())
    del stub.hooks

    result = await _run_agent_unbound(
        stub, message="hola", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        session_id="s-sin-registro",
    )

    assert stub.seen == ["hola"]
    assert result["final_response"] == "done:hola"


@pytest.mark.asyncio
async def test_the_turn_funnel_is_fail_open():
    """A raising subscriber leaves both texts untouched instead of breaking the turn."""
    hooks = _RecordingFilterHooks(raises=RuntimeError("plugin exploded"))
    stub = _FunnelStub(hooks)

    result = await _run_agent_unbound(
        stub, message="intacto", context_prompt="", history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm"),
        session_id="s-fail-open",
    )

    assert stub.seen == ["intacto"]
    assert result["final_response"] == "done:intacto"


@pytest.mark.asyncio
async def test_a_queued_followup_is_filtered_too(monkeypatch, tmp_path):
    """The review's ask: a /queue message (drained in-band, mid-turn) must be filtered as well.

    Driven through the real drain — the follow-up re-enters ``_run_agent`` and never touches the
    hook block in ``_handle_message_with_agent``, so before the filters moved to the funnel this
    second message reached the model raw.
    """
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    runner.hooks = _RecordingFilterHooks(message_replacement="[ofuscado]", response_prefix="salida:")

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="el seguimiento", message_type=MessageType.TEXT, source=_source(), message_id="queued-f",
    )

    result = await runner._run_agent(
        message="el primer turno", context_prompt="", history=[], source=_source(),
        session_id="s-cola", session_key=SESSION_KEY,
    )

    # Both the opening turn AND the queued follow-up reached the model filtered.
    assert _TwoTurnAgent.calls == ["[ofuscado]", "[ofuscado]"]
    assert result["final_response"] == "salida:done-2"

    events = [event for event, _ in runner.hooks.events]
    # One inbound filter per turn (two turns ran)...
    assert events.count("agent:message:filter") == 2
    # ...and exactly one outbound pass over the delivered reply: the terminal turn's text is
    # filtered by the frame that opened the chain, never twice by the nested one.
    assert events.count("agent:response:filter") == 1
