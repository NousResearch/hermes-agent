"""The expiry watcher never finalizes a mid-turn session.

A turn outlasting the idle window must survive the watcher pass unfinalized
(same skip-not-kill norm as the agent-cache cap/idle sweeps); expiry lands
after turn end via the normal path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.run_watchers import GatewaySessionWatchersMixin


def _entry(session_id):
    return SimpleNamespace(session_id=session_id, expiry_finalized=False)


def _runner(entries, running_keys):
    finalized = []

    async def _finalize(expired, failures):
        finalized.append([key for key, _ in expired])
        runner._running = False

    runner = SimpleNamespace(
        _running=True,
        session_store=SimpleNamespace(_entries=dict(entries)),
        async_session_store=SimpleNamespace(
            _ensure_loaded=AsyncMock(),
            _is_session_expired=AsyncMock(return_value=True),
        ),
        _is_session_running=lambda key: key in running_keys,
        _finalize_expired_sessions=_finalize,
        _expiry_housekeeping=AsyncMock(),
    )
    return runner, finalized


async def _run_watcher(runner):
    with (
        patch("asyncio.sleep", new=AsyncMock()),
        patch("gateway.run_watchers._interruptible_sleep", new=AsyncMock()),
    ):
        await GatewaySessionWatchersMixin._session_expiry_watcher(runner)


class TestExpiryLiveTurnGuard:
    @pytest.mark.asyncio
    async def test_live_turn_session_skipped(self):
        runner, finalized = _runner(
            {"k-live": _entry("s-live"), "k-idle": _entry("s-idle")}, {"k-live"})
        await _run_watcher(runner)
        assert finalized == [["k-idle"]]

    @pytest.mark.asyncio
    async def test_idle_only_session_still_finalized(self):
        runner, finalized = _runner({"k-idle": _entry("s-idle")}, set())
        await _run_watcher(runner)
        assert finalized == [["k-idle"]]
