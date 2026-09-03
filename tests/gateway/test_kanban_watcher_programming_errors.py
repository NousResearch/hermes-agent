"""Programming errors inside a kanban watcher tick must SURFACE, not be masked.

BUI-938: the fail-closed watcher handlers caught a broad ``Exception`` around
each tick and downgraded everything — including programming errors
(``NameError`` / ``AttributeError`` / ``TypeError``) — to a swallowed
"tick failed" / "unexpected watcher error" log line while the loop kept
spinning. BUI-936 CI confirmed a real ``NameError`` inside
``_auto_decompose_tick`` (an undefined name referenced when querying eligible
candidates) was silently swallowed and quietly changed corrupt-board behavior.

A typo, a bad attribute access, or a wrong call signature is a *bug in the
watcher*, not an operational failure of the board / DB / network. These tests
pin two contracts:

  * a ``NameError`` raised inside a dispatcher **or** notifier tick propagates
    out of the watcher (surfaced), and
  * genuine operational failures (I/O, SQLite) remain fail-closed — logged and
    swallowed so one bad tick never wedges the loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import asyncio
import sqlite3

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


def _make_runner(with_adapter=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()} if with_adapter else {}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = None
    # Skip _active_profile_name() lookup in the notifier watcher.
    runner._kanban_notifier_profile = "default"
    return runner


def _dispatcher_config():
    """A dispatch-owning config with the aux paths (auto-decompose, Linear
    bridge) OFF so the tick under test is purely ``dispatch_once``."""
    return {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "linear_bridge": {"enabled": False},
        }
    }


def _auto_decompose_config():
    """A dispatch-owning config with auto-decompose ON (Linear bridge OFF) so
    the tick under test runs ``_auto_decompose_tick`` — the exact BUI-936
    path where a ``NameError`` was raised while calling
    ``list_auto_decompose_ids(limit=<undefined name>)``."""
    return {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": True,
            "auto_decompose_per_tick": 3,
            "linear_bridge": {"enabled": False},
        }
    }


class _DummyConn:
    def close(self):
        pass


async def _passthru_to_thread(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _stop_after(runner, n):
    """A fake ``asyncio.sleep`` that flips ``_running`` false after N calls so a
    swallowing (buggy) watcher terminates instead of looping forever."""
    calls = []

    async def fake_sleep(delay):
        calls.append(delay)
        if len(calls) >= n:
            runner._running = False

    return fake_sleep


def test_nameerror_in_dispatch_tick_surfaces_out_of_watcher(tmp_path):
    """A ``NameError`` from within a dispatcher tick re-raises out of the
    watcher rather than being downgraded to a swallowed 'tick failed' log."""
    runner = _make_runner()

    def boom_dispatch(*args, **kwargs):
        raise NameError("name '_auto_decompose_per_tick' is not defined")

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_dispatcher_config()):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(_kb, "kanban_db_path", return_value=tmp_path / "kanban.db"):
                with patch.object(_kb, "list_boards", return_value=[{"slug": "default"}]):
                    with patch.object(_kb, "connect", return_value=_DummyConn()):
                        with patch.object(_kb, "dispatch_once", side_effect=boom_dispatch):
                            with patch.object(_kb, "write_dispatcher_heartbeat"):
                                with patch.object(_kb, "reap_worker_zombies", return_value=[]):
                                    with patch("asyncio.sleep", side_effect=_stop_after(runner, 3)):
                                        with patch("asyncio.to_thread", side_effect=_passthru_to_thread):
                                            with pytest.raises(NameError):
                                                asyncio.run(runner._kanban_dispatcher_watcher())


def test_operational_error_in_dispatch_tick_stays_failclosed(tmp_path):
    """A genuine I/O failure from ``dispatch_once`` remains fail-closed: it is
    logged and swallowed, the loop keeps ticking, and the watcher exits
    cleanly without propagating the error."""
    runner = _make_runner()
    dispatch_calls = []

    def flaky_dispatch(*args, **kwargs):
        dispatch_calls.append(True)
        raise OSError("transient disk read failure")

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_dispatcher_config()):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(_kb, "kanban_db_path", return_value=tmp_path / "kanban.db"):
                with patch.object(_kb, "list_boards", return_value=[{"slug": "default"}]):
                    with patch.object(_kb, "connect", return_value=_DummyConn()):
                        with patch.object(_kb, "dispatch_once", side_effect=flaky_dispatch):
                            with patch.object(_kb, "has_spawnable_ready", return_value=False):
                                with patch.object(_kb, "has_spawnable_review", return_value=False):
                                    with patch.object(_kb, "write_dispatcher_heartbeat"):
                                        with patch.object(_kb, "reap_worker_zombies", return_value=[]):
                                            with patch("asyncio.sleep", side_effect=_stop_after(runner, 3)):
                                                with patch("asyncio.to_thread", side_effect=_passthru_to_thread):
                                                    # Must NOT raise — operational failures stay fail-closed.
                                                    asyncio.run(runner._kanban_dispatcher_watcher())

    assert dispatch_calls, "dispatch tick should have run at least once"


def test_nameerror_in_auto_decompose_tick_surfaces_out_of_watcher(tmp_path):
    """The confirmed BUI-936 failure path: a ``NameError`` while calling
    ``list_auto_decompose_ids`` inside ``_auto_decompose_tick`` must surface
    out of the dispatcher watcher, NOT be swallowed by the tick's local
    ``except Exception`` (which used to downgrade it to ``triage_ids = []``)."""
    runner = _make_runner()

    def boom_list_ids(*args, **kwargs):
        raise NameError("name 'auto_decompose_limit' is not defined")

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_auto_decompose_config()):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(_kb, "kanban_db_path", return_value=tmp_path / "kanban.db"):
                with patch.object(_kb, "list_boards", return_value=[{"slug": "default"}]):
                    with patch(
                        "hermes_cli.kanban_decompose.list_auto_decompose_ids",
                        side_effect=boom_list_ids,
                    ):
                        with patch.object(_kb, "write_dispatcher_heartbeat"):
                            with patch.object(_kb, "reap_worker_zombies", return_value=[]):
                                with patch("asyncio.sleep", side_effect=_stop_after(runner, 3)):
                                    with patch("asyncio.to_thread", side_effect=_passthru_to_thread):
                                        with pytest.raises(NameError):
                                            asyncio.run(runner._kanban_dispatcher_watcher())


def test_operational_error_in_auto_decompose_tick_stays_failclosed(tmp_path):
    """A genuine operational failure from ``list_auto_decompose_ids`` (e.g. a
    transient DB read error) remains fail-closed: it is logged, the board is
    skipped, and the dispatcher keeps ticking without propagating."""
    runner = _make_runner()
    list_calls = []

    def flaky_list_ids(*args, **kwargs):
        list_calls.append(True)
        raise sqlite3.OperationalError("database is locked")

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_auto_decompose_config()):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(_kb, "kanban_db_path", return_value=tmp_path / "kanban.db"):
                with patch.object(_kb, "list_boards", return_value=[{"slug": "default"}]):
                    with patch(
                        "hermes_cli.kanban_decompose.list_auto_decompose_ids",
                        side_effect=flaky_list_ids,
                    ):
                        with patch.object(_kb, "connect", return_value=_DummyConn()):
                            with patch.object(_kb, "dispatch_once", return_value=None):
                                with patch.object(_kb, "has_spawnable_ready", return_value=False):
                                    with patch.object(_kb, "has_spawnable_review", return_value=False):
                                        with patch.object(_kb, "write_dispatcher_heartbeat"):
                                            with patch.object(_kb, "reap_worker_zombies", return_value=[]):
                                                with patch("asyncio.sleep", side_effect=_stop_after(runner, 3)):
                                                    with patch("asyncio.to_thread", side_effect=_passthru_to_thread):
                                                        # Must NOT raise — operational failures stay fail-closed.
                                                        asyncio.run(runner._kanban_dispatcher_watcher())

    assert list_calls, "auto-decompose tick should have called list_auto_decompose_ids"


def test_nameerror_in_notifier_tick_surfaces_out_of_watcher(tmp_path):
    """The sibling notifier watcher must likewise surface a programming error
    from its tick instead of masking it as 'kanban notifier tick failed'."""
    runner = _make_runner(with_adapter=True)

    def boom_list_subs(*args, **kwargs):
        raise NameError("name 'sub_row' is not defined")

    import hermes_cli.kanban_db as _kb

    with patch("hermes_cli.config.load_config", return_value=_dispatcher_config()):
        with patch.object(
            _kb,
            "list_boards",
            return_value=[{"slug": "default", "db_path": str(tmp_path / "kanban.db")}],
        ):
            with patch.object(_kb, "count_notify_subs", return_value=1):
                with patch.object(_kb, "connect", return_value=_DummyConn()):
                    with patch.object(_kb, "list_notify_subs", side_effect=boom_list_subs):
                        with patch("asyncio.sleep", side_effect=_stop_after(runner, 3)):
                            with patch("asyncio.to_thread", side_effect=_passthru_to_thread):
                                with pytest.raises(NameError):
                                    asyncio.run(runner._kanban_notifier_watcher())
