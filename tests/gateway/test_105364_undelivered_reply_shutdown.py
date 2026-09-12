"""Regression: a reply that is mid-delivery when the gateway stops must not vanish silently.

A turn's agent loop ends inside ``_run_agent``, which releases the session's ``_running_agents``
slot (``gateway/run_turn.py::_run_agent_cleanup_turn_tasks``) as soon as the agent returns. The
reply itself is shaped, persisted and handed to the adapter *after* that, in
``_handle_message_with_agent``. The shutdown drain only counts ``_running_agents`` / cron / api /
deferred work, so a reply in that window is invisible: the drain reports no work, the adapter is
disconnected underneath the send, and because ``ctx.timed_out`` stayed False the shutdown writes
``.clean_shutdown`` — which makes the next boot call ``discard_active_turn_markers()`` and throw the
session's durable active-turn marker away instead of recovering the interrupted turn. The reply is
lost with no user-visible notice. See #105364.

``active_turn_token`` is the accurate \"this session still owes the user a reply\" signal:
``_mark_durable_active_turn`` sets it before the turn runs and ``_clear_durable_active_turn`` clears
it only once delivery is done.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner


def _runner_with_active_turn(tmp_path):
    """Runner whose only in-flight work is a turn that has not delivered its reply yet."""
    runner, _adapter = make_restart_runner()
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
    entry = store.get_or_create_session(source)
    entry.active_turn_token = "tok-1"
    entry.active_turn_started_at = datetime.now()
    runner.session_store = store
    return runner, entry


def _ctx():
    ctx = GatewayRunner._StopContext(deferred_count=lambda: 0)
    ctx.started_at = time.monotonic()
    return ctx


@pytest.mark.asyncio
async def test_drain_waits_for_a_reply_that_is_mid_delivery(tmp_path, monkeypatch):
    """The shutdown must hold the transport open until the in-flight reply lands.

    The delivery task is only spawned from the drain's *first* probe, so it can only complete if the
    drain yields — a drain that returns without waiting never sees the reply land.
    """
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner, entry = _runner_with_active_turn(tmp_path)
    delivered: list[str] = []

    async def _finish_delivery() -> None:
        # Stands in for the turn's adapter send, still in flight when the drain starts.
        delivered.append(entry.session_key)
        entry.active_turn_token = None  # runner._clear_durable_active_turn

    probes = {"n": 0}

    def _probe() -> list:
        probes["n"] += 1
        if probes["n"] == 1:
            asyncio.get_running_loop().create_task(_finish_delivery())
            return [entry.session_key]
        return [entry.session_key] if entry.active_turn_token else []

    monkeypatch.setattr(runner, "_undelivered_reply_session_keys", _probe)

    await runner._stop_drain_active_work(0.0, _ctx())

    assert probes["n"] > 1, (
        "the drain returned without waiting for a reply that was still mid-delivery"
    )
    assert delivered == [entry.session_key]
    assert entry.active_turn_token is None


@pytest.mark.asyncio
async def test_reply_that_never_lands_arms_recovery_instead_of_a_clean_shutdown(tmp_path, monkeypatch):
    """A reply that cannot be delivered must be recoverable, never a silent loss."""
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run_shutdown._REPLY_DELIVERY_GRACE_S", 0.3)
    runner, entry = _runner_with_active_turn(tmp_path)
    ctx = _ctx()

    await runner._stop_drain_active_work(0.0, ctx)

    # No delivery happened, so the turn must be resumable on the next boot ...
    assert entry.resume_pending is True
    assert entry.resume_reason == "shutdown_timeout"
    # ... and the drain must report the timeout so the shutdown does not write .clean_shutdown:
    # a clean receipt makes the next boot discard the active-turn marker
    # (discard_active_turn_markers) instead of recovering the turn.
    assert ctx.timed_out is True

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.release_gateway_runtime_lock"), \
            patch("gateway.run._shutdown_gateway_health_export"):
        runner._update_runtime_status = MagicMock()
        runner._stop_persist_exit_state(ctx)

    assert not (tmp_path / ".clean_shutdown").exists(), (
        "a reply was still undelivered, yet the shutdown wrote .clean_shutdown — the next boot will "
        "discard the active-turn marker and never recover the turn"
    )
