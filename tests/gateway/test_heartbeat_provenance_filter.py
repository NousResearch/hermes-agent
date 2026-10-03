"""Heartbeat activity-provenance filtering + unified status sanitization.

Internal maintenance activities (context compression, its timeout/cooldown/turnhold
variants) must never leak their diagnostic descriptions into user-visible gateway
heartbeats. The heartbeat is emitted by
``GatewayTurnMixin._run_agent_notify_long_running`` (gateway/run_turn.py, split out of
``gateway/run.py`` in the run_* mixin refactor), so the behavior tests below drive that
real coroutine directly and assert on what the adapter actually received.
"""

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.conversation_compression import COMPACTION_STATUS
from agent.session_activity import ActivityProvenance
from gateway.config import Platform
from gateway.run import (
    _INTERNAL_ACTIVITY_PROVENANCES,
    _is_internal_activity_provenance,
    _prepare_gateway_status_message,
)
from gateway.run_turn import GatewayTurnMixin
from gateway.turn_context import TurnContext

# A recognizable credential shape (Synthetic from #23810 — placeholder gibberish, never
# a real token) that the shared outbound redactor must mask before chat delivery.
_SYNTHETIC_GITHUB_PAT = "ghp_" + "Ab3Cd4Ef5Gh6Ij7Kl8Mn9Op0Qr1St2Uv3Wx"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _CaptureAdapter:
    """Records heartbeat sends/edits so assertions read what the user would receive."""

    def __init__(self):
        self.sent = []
        self.edits = []

    async def send(self, chat_id, content, metadata=None):
        mid = f"hb-{len(self.sent) + 1}"
        self.sent.append({"chat_id": chat_id, "content": content, "message_id": mid})
        return SimpleNamespace(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SimpleNamespace(success=True, message_id=message_id)

    @property
    def messages(self):
        return [s["content"] for s in self.sent] + [e["content"] for e in self.edits]


class _ActivityAgent:
    """Minimal ``get_activity_summary`` provider — shape consumed by the heartbeat."""

    def __init__(self, *, current_tool=None, desc=None, provenance=None,
                 api_call_count=1, max_iterations=10):
        self._summary = {
            "current_tool": current_tool,
            "last_activity_desc": desc,
            "last_activity_provenance": provenance,
            "api_call_count": api_call_count,
            "max_iterations": max_iterations,
        }

    def get_activity_summary(self):
        return dict(self._summary)


def _make_disp(*, mode="on", generic_phrase="still on it"):
    disp = SimpleNamespace()
    disp._display_surface_mode = lambda *a, **k: mode
    disp.resolve_display_setting = lambda *a, **k: True
    disp.user_config = {}
    disp.platform_key = "telegram"
    disp._generic_status_phrase = lambda kind, **k: generic_phrase
    return disp


def _make_mixin(adapter, *, should_emit):
    """A ``GatewayTurnMixin`` with only the two seams the heartbeat reads overridden.

    Both seams are ``GatewayRunner`` attributes the mixin reads through ``self`` (the
    methods are split out of the god-file and bound via the MRO), so they are set on the
    bare mixin instance here.
    """
    mixin = GatewayTurnMixin()
    mixin._adapter_for_source = lambda source: adapter  # type: ignore[method-assign]
    mixin._should_emit_long_running_notification = should_emit  # type: ignore[method-assign]
    return mixin


def _make_ctx(agent, *, platform=Platform.TELEGRAM, executor_task_holder=None):
    return TurnContext(
        source=SimpleNamespace(chat_id="chat-1", chat_type="group", platform=platform),
        session_key="agent:main:telegram:group:chat-1",
        agent_holder=[agent],
    ), (executor_task_holder if executor_task_holder is not None else [object()])


async def _run_heartbeat(mixin, disp, ctx, executor_holder) -> None:
    # One emit tick, then the emit gate closes (mirrors a finished turn).
    await mixin._run_agent_notify_long_running(disp, ctx, executor_holder)


@pytest.fixture(autouse=True)
def _fast_notify_interval(monkeypatch):
    """Collapse the heartbeat interval so each tick costs ~10ms, not 180s."""
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_float_env",
        lambda name, default: 0.01 if name == "HERMES_AGENT_NOTIFY_INTERVAL" else default,
    )


# ---------------------------------------------------------------------------
# Helper unit coverage
# ---------------------------------------------------------------------------

def test_is_internal_activity_provenance_constants():
    """Internal maintenance provenances classify as internal; user/tool ones do not."""
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION) is True
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION_TIMEOUT) is True
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION_COOLDOWN) is True
    assert _is_internal_activity_provenance(ActivityProvenance.AGENT_COMPRESSION_TURNHOLD) is True

    assert _is_internal_activity_provenance("agent.compression") is True
    assert _is_internal_activity_provenance("agent.compression_timeout") is True
    assert _is_internal_activity_provenance("AGENT.COMPRESSION") is True
    # Future variants must be covered without touching the frozenset — the prefix rule
    # is what keeps the filter from silently missing a new ``agent.compression*`` value.
    assert _is_internal_activity_provenance("agent.compression.future_variant") is True

    assert _is_internal_activity_provenance(ActivityProvenance.UNKNOWN) is False
    assert _is_internal_activity_provenance("unknown") is False
    assert _is_internal_activity_provenance(None) is False
    assert _is_internal_activity_provenance("") is False
    assert _is_internal_activity_provenance("user") is False
    assert _is_internal_activity_provenance("tool.call") is False


def test_internal_provenance_set_covers_compression_activity_values():
    """The frozenset stays in lockstep with the ActivityProvenance compressions."""
    declared = {
        p.value for p in ActivityProvenance
        if str(p.value).startswith("agent.compression")
    }
    assert declared, "ActivityProvenance must declare at least one compression value"
    assert declared <= _INTERNAL_ACTIVITY_PROVENANCES


def test_prepare_gateway_status_message_sanitizes_heartbeat_content():
    """Legitimate heartbeats survive; routine compression chatter is suppressed."""
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "⏳ Working — 3 min") == "⏳ Working — 3 min"
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "⏳ Working — 3 min — web_search") == "⏳ Working — 3 min — web_search"
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "still on it") == "still on it"

    # Routine compression noise is suppressed (returns None → caller falls back).
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM, "heartbeat", f"⏳ Working — 3 min — {COMPACTION_STATUS}"
    ) is None
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "heartbeat", "⏳ Working — 3 min — preflight compression") is None

    # Secrets riding a heartbeat are redacted, not delivered verbatim.
    sanitized = _prepare_gateway_status_message(
        Platform.TELEGRAM, "heartbeat", f"⏳ Working — 1 min — {_SYNTHETIC_GITHUB_PAT}"
    )
    assert sanitized is not None
    assert _SYNTHETIC_GITHUB_PAT not in sanitized


# ---------------------------------------------------------------------------
# Behavioral coverage for the real heartbeat coroutine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heartbeat_suppresses_compression_provenance():
    """agent.compression provenance → terse liveness only, no internal wording."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(
        desc="context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(), ctx, exec_holder)

    assert adapter.messages, "the heartbeat must still fire while compression runs"
    for msg in adapter.messages:
        assert "compression" not in msg
        assert "context compression in progress" not in msg
        assert "⏳ Working" in msg


@pytest.mark.asyncio
async def test_heartbeat_suppresses_compression_provenance_but_keeps_iteration_detail():
    """Only the description is dropped — the gated iteration counter still shows."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(
        desc="context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
        api_call_count=3,
        max_iterations=12,
    )
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(), ctx, exec_holder)

    assert adapter.messages
    joined = " | ".join(adapter.messages)
    assert "compression" not in joined
    assert "iteration 3/12" in joined


@pytest.mark.asyncio
async def test_heartbeat_preserves_active_tool_over_internal_provenance():
    """A live current_tool wins; a stale compression stamp must not hide it."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(
        current_tool="web_search",
        desc="context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(), ctx, exec_holder)

    assert any("web_search" in msg for msg in adapter.messages)
    assert not any("context compression in progress" in msg for msg in adapter.messages)


@pytest.mark.asyncio
async def test_heartbeat_preserves_user_facing_activity_desc():
    """A user-facing (non-internal) activity description is preserved."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(
        desc="synthesizing search results",
        provenance=ActivityProvenance.UNKNOWN,
    )
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(), ctx, exec_holder)

    assert any("synthesizing search results" in msg for msg in adapter.messages)


@pytest.mark.asyncio
async def test_heartbeat_falls_back_to_terse_when_noisy_detail_filtered():
    """Unknown provenance + routine compression wording → filtered detail, terse fallback.

    Exercises the layered defense: the provenance filter cannot catch a description
    whose provenance is UNKNOWN, so the shared status sanitizer must suppress it and
    the heartbeat must still reach the user as the terse liveness line.
    """
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(desc=COMPACTION_STATUS, provenance=ActivityProvenance.UNKNOWN)
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(), ctx, exec_holder)

    assert adapter.messages, "the terse fallback must still reach the user"
    for msg in adapter.messages:
        assert COMPACTION_STATUS not in msg
        assert "⏳ Working" in msg


@pytest.mark.asyncio
async def test_heartbeat_redacts_secret_in_activity_desc():
    """A secret echoed into an activity description never reaches the chat surface."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(
        desc=f"tool_call(api_key={_SYNTHETIC_GITHUB_PAT})",
        provenance=ActivityProvenance.UNKNOWN,
    )
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(), ctx, exec_holder)

    assert adapter.messages
    for msg in adapter.messages:
        assert _SYNTHETIC_GITHUB_PAT not in msg


@pytest.mark.asyncio
async def test_heartbeat_generic_mode_uses_phrase_and_never_leaks_detail():
    """Generic mode replaces the wording wholesale — internal detail still never leaks."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(
        desc="context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(side_effect=[True, False]))

    await _run_heartbeat(mixin, _make_disp(mode="generic", generic_phrase="still on it"), ctx, exec_holder)

    assert adapter.messages
    for msg in adapter.messages:
        assert "compression" not in msg
        assert "still on it" in msg


@pytest.mark.asyncio
async def test_heartbeat_waits_through_startup_window_before_agent_binds():
    """Startup tolerance: a heartbeating run with no agent bound yet must not terminate.

    An early tick can read ``agent_holder[0] is None`` while the session slot is not
    registered yet — the emit gate reports False on a session this run is about to own.
    Breaking there ends the heartbeat before it can ever fire. The loop must keep the
    heartbeat alive until the agent binds, then emit normally.
    """
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(desc="working on it", provenance=ActivityProvenance.UNKNOWN)
    ctx, exec_holder = _make_ctx(None)

    async def _pending_turn():
        await asyncio.sleep(3600)

    exec_holder[0] = asyncio.ensure_future(_pending_turn())

    def _emit(*_a, **_k):
        return True  # the run still owns the heartbeat; only the agent is not bound yet

    mixin = _make_mixin(adapter, should_emit=_emit)

    async def _bind_agent_after_a_few_ticks():
        # A real turn binds the agent from the executor thread a few heartbeats in; before
        # that the session slot is not registered either, so an early tick sees an unbound
        # agent. Without the startup tolerance the loop would have no agent to gate on and
        # would break before any heartbeat could reach the adapter.
        await asyncio.sleep(0.03)
        ctx.agent_holder[0] = agent

    heartbeat = asyncio.ensure_future(
        mixin._run_agent_notify_long_running(_make_disp(), ctx, exec_holder)
    )
    try:
        await asyncio.wait_for(_bind_agent_after_a_few_ticks(), timeout=5.0)
        # Let the next heartbeat tick observe the bound agent and emit.
        await asyncio.sleep(0.1)
    finally:
        # The heartbeat loops for the life of the run; this "run" ends here.
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        exec_holder[0].cancel()
        with suppress(asyncio.CancelledError):
            await exec_holder[0]

    assert adapter.messages, "the heartbeat must survive the startup window and fire"
    assert any("working on it" in msg for msg in adapter.messages)


@pytest.mark.asyncio
async def test_heartbeat_stops_when_executor_finished_without_agent():
    """A finished executor with no agent is a gone run — the loop terminates."""
    adapter = _CaptureAdapter()
    ctx, exec_holder = _make_ctx(None)
    exec_holder[0] = asyncio.get_running_loop().create_future()
    exec_holder[0].set_result(None)  # executor already finished, agent never bound

    mixin = _make_mixin(adapter, should_emit=MagicMock(return_value=True))

    await asyncio.wait_for(
        mixin._run_agent_notify_long_running(_make_disp(), ctx, exec_holder), timeout=5.0
    )

    assert mixin._should_emit_long_running_notification.call_count == 0
    assert adapter.sent == [] and adapter.edits == []


@pytest.mark.asyncio
async def test_heartbeat_stops_when_run_no_longer_owns_session(monkeypatch):
    """Post-startup, a False gate still terminates the loop (no stale heartbeat)."""
    adapter = _CaptureAdapter()
    agent = _ActivityAgent(desc="working on it", provenance=ActivityProvenance.UNKNOWN)
    ctx, exec_holder = _make_ctx(agent)
    mixin = _make_mixin(adapter, should_emit=MagicMock(return_value=False))

    await asyncio.wait_for(
        mixin._run_agent_notify_long_running(_make_disp(), ctx, exec_holder), timeout=2.0
    )

    assert mixin._should_emit_long_running_notification.call_count == 1
    assert adapter.sent == [] and adapter.edits == []
