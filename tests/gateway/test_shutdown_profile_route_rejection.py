"""Regression tests for the per-session error guard around the profile-runtime scope that
wraps each active-session shutdown notice in ``_notify_active_sessions_of_shutdown``.

The documented invariant of that method (``gateway/run_shutdown.py``) is "send failures never
block shutdown". Commit ``cd3de040ab`` wrapped each per-session notice body in
``async with _async_profile_runtime_scope(self._resolve_profile_home_for_source(source))`` but
placed that call *below* the existing per-session ``try: ... except Exception: continue`` guard,
so a ``ProfileRouteRejected`` raised by ``_resolve_profile_home_for_source`` (a route to a
now-unserved / tombstoned profile) escaped the loop and aborted ``stop()`` mid-drain — leaving
adapters connected, the PID/lock held, and every later teardown phase skipped until the
OS-level supervisor SIGKILLed the process.

The fix wraps the route-resolution + scope-entry + present-notification in the same per-session
guard so one session's routing/scope failure skips just that session, matching the stall-watcher
shape in ``gateway/run_watchers.py`` and the sibling regression
``test_cron_interrupt_notification.py::test_a_raising_adapter_cannot_block_shutdown``.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.profile_routing import ProfileRouteRejected
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _wire_active_session(runner, source) -> str:
    """Register ``source`` with an active agent so the shutdown-notice loop targets it."""
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()
    runner._cache_session_source(session_key, source)
    return session_key


@pytest.mark.asyncio
async def test_profile_route_rejection_for_one_session_does_not_block_shutdown():
    """A ``ProfileRouteRejected`` from ``_resolve_profile_home_for_source`` for one active
    session is caught by the per-session guard: the bad session is skipped (no notice sent),
    a healthy sibling session still receives its notice, and the call returns cleanly.

    On the base commit (pre-fix) this test is RED: the raise from the bad session escapes
    ``_notify_active_sessions_of_shutdown`` before the good session is even visited.
    """
    runner, adapter = make_restart_runner()
    bad_source = make_restart_source(chat_id="bad-chat")
    good_source = make_restart_source(chat_id="good-chat")
    _wire_active_session(runner, bad_source)
    _wire_active_session(runner, good_source)

    real_resolver = gateway_run.GatewayRunner._resolve_profile_home_for_source.__get__(
        runner, gateway_run.GatewayRunner
    )

    def _reject_for_bad(source):
        if source.chat_id == bad_source.chat_id:
            raise ProfileRouteRejected("tombstoned")
        return real_resolver(source)

    runner._resolve_profile_home_for_source = _reject_for_bad

    await runner._notify_active_sessions_of_shutdown()  # MUST NOT raise

    sent_chats = {call[0] for call in adapter.sent_calls}
    assert bad_source.chat_id not in sent_chats, (
        "rejected-route session must not receive a notice"
    )
    assert good_source.chat_id in sent_chats, (
        "healthy sibling must still receive its notice (per-session skip, not total abort)"
    )


@pytest.mark.asyncio
async def test_profile_scope_entry_failure_for_one_session_does_not_block_shutdown(monkeypatch):
    """A raise from ``_async_profile_runtime_scope`` entry (defensive coverage for any future
    regression that lets a scope-entry exception escape — no current code path raises here,
    since ``_load_profile_secret_scope`` fails open on missing/unreadable ``.env``) is likewise
    per-session skipped: the bad session gets no notice, a healthy sibling still does.

    On the base commit (pre-fix) this test is RED: the unguarded ``async with scope:`` line
    propagates the raise, aborting the loop before the good session is visited.
    """
    runner, adapter = make_restart_runner()
    bad_source = make_restart_source(chat_id="bad-chat")
    good_source = make_restart_source(chat_id="good-chat")
    _wire_active_session(runner, bad_source)
    _wire_active_session(runner, good_source)

    bad_home = Path("/tmp/opencode/hermes-bad-profile-home-scope-entry")
    good_home = Path("/tmp/opencode/hermes-good-profile-home-scope-entry")
    runner._resolve_profile_home_for_source = lambda source: (
        bad_home if source.chat_id == bad_source.chat_id else good_home
    )

    @asynccontextmanager
    async def _raising_scope_for_bad(home):
        if Path(home) == bad_home:
            raise RuntimeError("scope entry exploded")
        yield

    monkeypatch.setattr("gateway.run._async_profile_runtime_scope", _raising_scope_for_bad)

    await runner._notify_active_sessions_of_shutdown()  # MUST NOT raise

    sent_chats = {call[0] for call in adapter.sent_calls}
    assert bad_source.chat_id not in sent_chats, (
        "scope-failure session must not receive a notice"
    )
    assert good_source.chat_id in sent_chats, (
        "healthy sibling must still receive its notice (per-session skip, not total abort)"
    )
