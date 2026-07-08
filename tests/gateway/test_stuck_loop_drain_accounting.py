"""Integration tests for stuck-loop restart-count accounting across the REAL
gateway teardown (#7536 false-suspend fix).

Background: the stuck-loop guard auto-suspends a session (clearing its history)
after 3 consecutive restarts where it was "active".  The bug was that the
counter incremented on EVERY restart where a session was active at drain-START,
ignoring whether the drain completed cleanly — so 3 clean deploy-restarts of a
resident session falsely tripped the guard and wiped the user's conversation.

The fix gates the increment on ``timed_out`` AND consumes a set of
genuinely-interrupted sessions CAPTURED at the pre-interrupt read of
``self._running_agents`` — because the interrupt path drains that dict before
the counter site runs.  A fresh read at the counter site would be empty and a
real stuck loop would never accumulate (the CB-B1 hole).

These tests drive the real ``stop()`` teardown with the orthogonal subsystems
(adapters, DB close, memory-provider cleanup, finalize) mocked, while keeping
the load-bearing sequence real: capture → interrupt (empties _running_agents)
→ counter site.  They must NEVER inject ``_interrupted_keys`` at the counter
site — that would re-hide the exact bug.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as run_mod
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner


def _make_agent():
    """A mock running agent.  Its removal from _running_agents is driven by the
    fixture's _interrupt side-effect (timed-out) or the fake drain (clean),
    mirroring how a live agent pops itself once cancelled/finished.
    """
    return MagicMock()


def _register_session(runner, chat_id):
    """Create a real session entry so the pre-drain resume-pending marking path
    (which builds _pre_drain_keys) runs against a genuine SessionEntry.
    Returns the session_key.
    """
    src = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1")
    entry = runner.session_store.get_or_create_session(src)
    return entry.session_key


@pytest.fixture
def teardown_runner(tmp_path, monkeypatch):
    """A runner wired so stop() can run its teardown with orthogonal subsystems
    neutralized but the stuck-loop accounting path fully real.
    """
    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    # Real session store so pre-drain marking (→ _pre_drain_keys) is faithful.
    runner.session_store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())

    # --- Neutralize orthogonal teardown steps (unrelated to stuck-loop counts) ---
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._agent_cache = None
    runner._agent_cache_lock = None
    runner._finalize_shutdown_agents = AsyncMock()
    runner._bounded_adapter_teardown = AsyncMock()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._launch_detached_restart_command = AsyncMock()
    runner._update_runtime_status = MagicMock()
    runner._notify_restart_loop_suspended = AsyncMock()
    runner._notify_active_sessions_of_shutdown = AsyncMock()

    # --- Make the interrupt path EMPTY _running_agents, as it does live ---
    def _interrupt(reason):
        for _sk, _ag in list(runner._running_agents.items()):
            if _ag is _AGENT_PENDING_SENTINEL:
                continue
            runner._running_agents.pop(_sk, None)
    runner._interrupt_running_agents = MagicMock(side_effect=_interrupt)

    return runner, tmp_path


async def _run_stop_with_drain(runner, *, running_at_start, timed_out):
    """Drive runner.stop() faithfully.

    ``running_at_start`` is the {session_key: agent} dict resident when stop()
    begins — the pre-drain marking loop reads it to build _pre_drain_keys, then
    the (mocked) drain resolves.  For a CLEAN drain the mock empties
    _running_agents (all sessions finished); for a TIMED-OUT drain it leaves
    them resident so the real interrupt path (wired in the fixture) empties them
    AFTER the pre-interrupt capture.
    """
    runner._running_agents = dict(running_at_start)
    snapshot = dict(running_at_start)

    async def _fake_drain(timeout):
        if not timed_out:
            # Clean drain: every active session finished during the window.
            runner._running_agents.clear()
        # Timed-out: leave _running_agents populated (real interrupt empties it).
        return snapshot, timed_out

    runner._drain_active_agents = _fake_drain
    runner._stop_task = None  # allow re-entry across simulated restarts
    await runner.stop()


def _counts(home, runner):
    path = home / runner._STUCK_LOOP_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@pytest.mark.asyncio
async def test_clean_drain_does_not_count(teardown_runner):
    """AB1: a clean drain (timed_out=False) must NOT increment — the resident
    session finished its turn; nothing was stuck.
    """
    runner, home = teardown_runner
    sk = _register_session(runner, "123")
    await _run_stop_with_drain(runner, running_at_start={sk: _make_agent()}, timed_out=False)
    assert sk not in _counts(home, runner), (
        "clean drain wrongly incremented the stuck-loop counter"
    )


@pytest.mark.asyncio
async def test_clean_drain_resets_prior_count(teardown_runner):
    """AB1 (mixed sequence): a session with a count from prior real interruptions
    that then drains cleanly has its count RESET to 0 — it proved it can finish.
    """
    runner, home = teardown_runner
    sk = _register_session(runner, "123")
    # Seed a prior count (as if 2 earlier timed-out restarts interrupted it).
    runner._increment_restart_failure_counts({sk})
    runner._increment_restart_failure_counts({sk})
    assert _counts(home, runner)[sk] == 2
    # This restart: the session is running at drain-start (so it's pre-drain
    # marked → in _pre_drain_keys) and finishes cleanly during the drain.
    await _run_stop_with_drain(runner, running_at_start={sk: _make_agent()}, timed_out=False)
    assert sk not in _counts(home, runner), (
        "clean drain did not reset a stale accumulated count (mixed timeout->clean)"
    )


@pytest.mark.asyncio
async def test_timed_out_counts_captured_not_fresh_read(teardown_runner):
    """AB2 / CB-B1 regression: on a timed-out drain the counter must consume the
    PRE-INTERRUPT captured set, not a fresh _running_agents read.  The fixture's
    interrupt path EMPTIES _running_agents before the counter site — so if the
    counter re-read it fresh, the genuine session would NOT be counted (the bug).
    This test proves it IS counted.
    """
    runner, home = teardown_runner
    sk = _register_session(runner, "stuck")
    await _run_stop_with_drain(runner, running_at_start={sk: _make_agent()}, timed_out=True)
    # The interrupt path emptied _running_agents; the count must still be 1,
    # proving the captured (pre-interrupt) set was used.
    assert runner._running_agents == {}, "fixture precondition: interrupt emptied the dict"
    assert _counts(home, runner).get(sk) == 1, (
        "genuinely-interrupted session was NOT counted — counter used a stale "
        "fresh read instead of the pre-interrupt capture (CB-B1 regression)"
    )


@pytest.mark.asyncio
async def test_timed_out_skips_pending_sentinel(teardown_runner):
    """A _AGENT_PENDING_SENTINEL entry (agent not started) is not a genuine
    interruption and must not be counted.
    """
    runner, home = teardown_runner
    sk_real = _register_session(runner, "real")
    sk_pending = _register_session(runner, "pending")
    await _run_stop_with_drain(
        runner,
        running_at_start={sk_real: _make_agent(), sk_pending: _AGENT_PENDING_SENTINEL},
        timed_out=True,
    )
    counts = _counts(home, runner)
    assert counts.get(sk_real) == 1
    assert sk_pending not in counts


@pytest.mark.asyncio
async def test_genuine_stuck_loop_suspends_through_teardown(teardown_runner):
    """AB3: 3 consecutive timed-out drains (full drain->interrupt->count path)
    with the same session still-running each time → it reaches threshold 3 and
    is suspended on the next startup.  Proves the fix does NOT neuter genuine
    protection even though the interrupt path empties _running_agents each time.
    """
    runner, home = teardown_runner
    sk = _register_session(runner, "stuck")

    for _ in range(3):
        await _run_stop_with_drain(runner, running_at_start={sk: _make_agent()}, timed_out=True)

    assert _counts(home, runner).get(sk) == 3, (
        "genuine stuck loop did not accumulate to threshold through the real teardown"
    )

    # Now the startup-side suspend must fire for it.
    mock_entry = MagicMock()
    mock_entry.suspended = False
    runner.session_store._entries = {sk: mock_entry}
    runner.session_store._save = MagicMock()
    suspended = runner._suspend_stuck_loop_sessions()
    assert suspended == 1
    assert mock_entry.suspended is True


@pytest.mark.asyncio
async def test_three_clean_restarts_do_not_suspend(teardown_runner):
    """AB4 (the Ace regression): 3 clean deploy-restarts of a resident session →
    it never accumulates → never suspended → history preserved.
    """
    runner, home = teardown_runner
    sk = _register_session(runner, "live")

    for _ in range(3):
        await _run_stop_with_drain(runner, running_at_start={sk: _make_agent()}, timed_out=False)

    assert sk not in _counts(home, runner), (
        "clean deploy-restarts accumulated toward false suspension (the Ace bug)"
    )

    # And the startup suspend must NOT fire.
    mock_entry = MagicMock()
    mock_entry.suspended = False
    runner.session_store._entries = {sk: mock_entry}
    assert runner._suspend_stuck_loop_sessions() == 0
    assert mock_entry.suspended is False
