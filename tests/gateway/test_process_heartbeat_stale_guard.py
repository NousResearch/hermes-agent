"""#120334 regression: a queued background-process heartbeat for a process that
has already exited must NOT start a fresh agent turn.

The fix lives at two levels:
  * ``tools.process_registry`` — ``_heartbeat_loop`` and ``_emit_heartbeat``
    re-validate liveness atomically under the registry lock right before
    enqueueing. A heartbeat emitted while the process was running can be
    stale by the time the gateway consumes it, so the gateway revalidates
    again at admission time.
  * ``gateway.run_heartbeat_acceptance.process_heartbeat_still_alive`` — the
    synthetic ``MessageEvent`` carries ``_process_heartbeat_session_id`` +
    ``_process_heartbeat_started_at`` stamped at ``_inject_watch_notification``
    time; the liveness helper looks the session up in the live process
    registry and rejects events whose process has exited or whose id has been
    recycled (started_at mismatch).

These tests assert the helper directly (unit) and the injection path +
helper combined (integration). Pre-fix: the synthetic event carries no
identity, the helper has no logic, and the stale event is admitted. Post-fix:
the event is dropped, no turn starts, and the helper returns False.
"""
import logging
from types import SimpleNamespace
from unittest.mock import patch

from gateway.platforms.event import MessageType


# ── unit: process_heartbeat_still_alive ───────────────────────────────────


def _stub_session(*, exited, started_at=100.0, session_id="proc_x"):
    """Build a ProcessSession-like object compatible with the helper's reads."""
    return SimpleNamespace(id=session_id, started_at=started_at, exited=exited)


def test_process_heartbeat_still_alive_returns_true_for_non_heartbeat_event():
    """A normal user turn or scheduled /heartbeat has no process provenance;
    the helper must NOT block it. The scheduled /heartbeat path keeps its own
    provenance check (heartbeat_owner_is_current)."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive

    event = SimpleNamespace()  # no _process_heartbeat_session_id
    assert process_heartbeat_still_alive(event) is True


def test_process_heartbeat_still_alive_returns_true_for_running_session(monkeypatch):
    """A still-running session in the live process registry is alive — the
    helper must return True (admit the heartbeat)."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    live = _stub_session(exited=False, started_at=100.0)
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: live)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_x",
        _process_heartbeat_started_at=100.0,
    )
    assert process_heartbeat_still_alive(event) is True


def test_process_heartbeat_still_alive_returns_false_for_exited_session(monkeypatch, caplog):
    """The session exited between emission and delivery — pre-fix this would
    start a turn with a 'still running' message after the user saw the
    completion. Post-fix the helper returns False and the gateway drops it."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    dead = _stub_session(exited=True, started_at=100.0)
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: dead)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_x",
        _process_heartbeat_started_at=100.0,
    )
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        assert process_heartbeat_still_alive(event) is False


def test_process_heartbeat_still_alive_returns_false_for_missing_session(monkeypatch):
    """The session is no longer in the registry at all (reaped, evicted, or
    from a different gateway lifetime). Helper must return False."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: None)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_missing",
        _process_heartbeat_started_at=100.0,
    )
    assert process_heartbeat_still_alive(event) is False


def test_process_heartbeat_still_alive_returns_false_for_started_at_mismatch(monkeypatch):
    """The session id was recycled by a fresh spawn (different started_at).
    The old heartbeat must be rejected — the original incarnation finished."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    fresh = _stub_session(exited=False, started_at=200.0)  # NEW incarnation
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: fresh)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_x",
        _process_heartbeat_started_at=100.0,  # OLD incarnation's epoch
    )
    assert process_heartbeat_still_alive(event) is False


# ── integration: _inject_watch_notification stamps the provenance ──────────


def test_inject_watch_notification_stamps_process_heartbeat_identity(monkeypatch):
    """``_inject_watch_notification`` must stamp the synthetic event with the
    heartbeat session identity so the gateway can revalidate at admission.
    Pre-fix: no attributes set, the staleness check has nothing to look at."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform

    # Stand up a minimal GatewayRunner with the run_notifications mixin.
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)

    # Minimal adapter that records the synth event and accepts it.
    captured = {}

    class _Adapter:
        name = "fake"

        def fronts_platform(self, platform):
            return True

        def prime_routing_cache(self, event):
            pass

        async def handle_message(self, event):
            captured["event"] = event
            event._gateway_accepted = True

    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner.config = SimpleNamespace(platforms={})

    # Pre-build the event source.
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm",
    )
    runner._build_process_event_source = lambda evt: source

    def _resolve(name, source=None):
        return runner.adapters[Platform.TELEGRAM]
    runner._resolve_injection_adapter = _resolve

    evt = {
        "type": "heartbeat",
        "session_id": "proc_stale_inject",
        "started_at": 999.0,
        "session_key": "agent:telegram:dm:42",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "42",
        "command": "sleep 60",
    }

    import asyncio
    asyncio.run(runner._inject_watch_notification("still working", evt))

    injected = captured.get("event")
    assert injected is not None, "adapter never received the synthetic event"
    assert isinstance(injected, MessageEvent)
    assert getattr(injected, "_process_heartbeat_session_id", None) == "proc_stale_inject"
    assert getattr(injected, "_process_heartbeat_started_at", None) == 999.0


def test_inject_watch_notification_does_not_stamp_non_heartbeat_events(monkeypatch):
    """A watch_match / completion / async_delegation event must NOT receive
    heartbeat provenance attributes — the helper would (correctly) treat it
    as a heartbeat and look up a process session that doesn't exist."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)

    captured = {}

    class _Adapter:
        name = "fake"

        def fronts_platform(self, platform):
            return True

        def prime_routing_cache(self, event):
            pass

        async def handle_message(self, event):
            captured["event"] = event
            event._gateway_accepted = True

    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner.config = SimpleNamespace(platforms={})
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm",
    )
    runner._build_process_event_source = lambda evt: source

    def _resolve(name, source=None):
        return runner.adapters[Platform.TELEGRAM]
    runner._resolve_injection_adapter = _resolve

    evt = {
        "type": "watch_match",
        "session_id": "proc_w",
        "session_key": "agent:telegram:dm:42",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "42",
        "command": "echo done",
        "pattern": "DONE",
        "output": "DONE",
    }

    import asyncio
    asyncio.run(runner._inject_watch_notification("matched", evt))

    injected = captured["event"]
    assert not hasattr(injected, "_process_heartbeat_session_id") \
        or getattr(injected, "_process_heartbeat_session_id", None) is None


# ── integration: end-to-end stale heartbeat does NOT start a turn ──────────


def test_stale_queued_heartbeat_does_not_start_turn(monkeypatch, caplog):
    """End-to-end #120334. Build a synthetic MessageEvent that looks like
    a queued background-process heartbeat for a session that has already
    exited, then call the liveness helper the gateway uses. Pre-fix: the
    helper does not exist / returns True / a turn starts. Post-fix: the
    helper returns False, the caller drops the event, no turn starts."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    # The session is GONE (process exited long ago).
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: None)

    event = MessageEvent(
        text="heartbeat #1 — still running after 1m",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_120334"
    event._process_heartbeat_started_at = 1700000000.0

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = process_heartbeat_still_alive(event)
    assert result is False, "Stale heartbeat must be rejected (#120334)"


def test_fresh_queued_heartbeat_admitted(monkeypatch):
    """Sanity: a still-running process admits its heartbeat. Regression guard
    against an over-eager helper that would silence valid heartbeats."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    monkeypatch.setattr(
        pr_mod.process_registry, "get",
        lambda sid: SimpleNamespace(id=sid, started_at=1700000000.0, exited=False),
    )

    event = MessageEvent(
        text="heartbeat #1",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_alive"
    event._process_heartbeat_started_at = 1700000000.0

    assert process_heartbeat_still_alive(event) is True
