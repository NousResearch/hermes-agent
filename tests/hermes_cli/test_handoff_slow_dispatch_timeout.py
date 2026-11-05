"""Regression tests: a slow gateway handoff dispatch must not be misreported
as "gateway not running" — and no waiter may stomp a claimed (running) row.

Bug shape (live-reproduced on main @1c5ee5815f): /handoff poll-waited a flat
60s for a TERMINAL state. The gateway watcher claims within seconds, but the
dispatch is a FULL synthetic agent turn (whole transcript replay + delivery)
that routinely exceeds 60s. The CLI then printed "Timed out waiting for the
gateway. Is `hermes gateway` running?" (false diagnosis), called
fail_handoff() on the RUNNING row (stomping the gateway's claim), and claimed
"Your CLI session is intact" after switch_session had already re-pointed the
session. The gateway later overwrote failed -> completed: split-brain.

Fix under test:
  1. coordination.fail_handoff(only_states=...) is a CAS: waiters can only fail
     rows still in the given states.
  2. CLI _handle_handoff_command: 60s deadline applies only to PENDING;
     a RUNNING row gets a long (15 min) wait and is never failed by the CLI.
  3. tui_gateway handoff.fail: only pending rows can be failed by Desktop.
"""

from __future__ import annotations

import time
import types
from unittest.mock import MagicMock, patch

import pytest

from agent.i18n import t
from hermes_cli.gateway_session_store import GatewaySessionStore, SurfaceLabelReader
from persistence.authority.coordination import HandoffStatus
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.fixture()
def store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return storage_handle(home)


def _state(store, session_id):
    return store.coordination.get_handoff(session_id=session_id).state.value


# ---------------------------------------------------------------------------
# 1. State layer: fail_handoff CAS
# ---------------------------------------------------------------------------

class TestFailHandoffCAS:
    def test_cas_fails_pending_row(self, store):
        seed_session(store, "s1", "cli")
        assert store.coordination.request_handoff(session_id="s1", platform="discord")
        assert store.coordination.fail_handoff(
            session_id="s1", error="timed out", only_states=(HandoffStatus.PENDING,)) is True
        assert _state(store, "s1") == "failed"

    def test_cas_refuses_running_row(self, store):
        """A waiter timeout must NOT stomp a row the gateway has claimed."""
        seed_session(store, "s2", "cli")
        assert store.coordination.request_handoff(session_id="s2", platform="discord")
        assert store.coordination.claim_handoff(session_id="s2")  # gateway claimed: pending -> running
        assert store.coordination.fail_handoff(
            session_id="s2", error="timed out", only_states=(HandoffStatus.PENDING,)) is False
        assert _state(store, "s2") == "running"
        # gateway still reaches its own terminal state
        store.coordination.complete_handoff(session_id="s2")
        assert _state(store, "s2") == "completed"

    def test_unconditional_fail_still_available_to_owner(self, store):
        """The gateway watcher (owner of a claimed row) fails unconditionally."""
        seed_session(store, "s3", "cli")
        assert store.coordination.request_handoff(session_id="s3", platform="discord")
        assert store.coordination.claim_handoff(session_id="s3")
        assert store.coordination.fail_handoff(session_id="s3", error="dispatch raised") is True
        assert _state(store, "s3") == "failed"

    def test_cas_refuses_terminal_rows(self, store):
        seed_session(store, "s4", "cli")
        assert store.coordination.request_handoff(session_id="s4", platform="discord")
        assert store.coordination.claim_handoff(session_id="s4")
        store.coordination.complete_handoff(session_id="s4")
        assert store.coordination.fail_handoff(
            session_id="s4", error="late timeout", only_states=(HandoffStatus.PENDING,)) is False
        assert _state(store, "s4") == "completed"


# ---------------------------------------------------------------------------
# 2. CLI wait loop
# ---------------------------------------------------------------------------

def _offline_store():
    """The classic CLI's session store once its gateway connection is final: falsy, so the handoff
    runs on a local writer handle on the current home's store."""
    return GatewaySessionStore(types.SimpleNamespace(final=True))


def _run_handoff(session_db, session_id, monkeypatch, time_budget=30.0):
    """Drive the real _handle_handoff_command with ``session_db`` as the CLI's session store.

    Gateway config / platform plumbing is stubbed; the poll loop, state
    transitions, and fail semantics are exercised for real. time.sleep is
    compressed so a simulated 60s pending deadline elapses in well under a
    second of wall clock.
    """
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    printed: list[str] = []

    class Host(CLICommandsMixin):
        def __init__(self):
            self.session_id = session_id
            self._session_db = session_db
            self._agent_running = False
            self._should_exit = False

    host = Host.__new__(Host)
    Host.__init__(host)

    home = types.SimpleNamespace(chat_id="123", name="home", thread_id=None)
    gw_config = MagicMock()
    gw_config.platforms = {}
    gw_config.get_home_channel.return_value = home

    import gateway.config as gwc

    platform_obj = gwc.Platform("discord")
    pcfg = types.SimpleNamespace(enabled=True, extra={})
    gw_config.platforms = {platform_obj: pcfg}

    # Compress time: each sleep(0.5) advances a fake clock by 2.0s.
    clock = {"t": time.time()}

    def fake_time():
        return clock["t"]

    def fake_sleep(secs):
        clock["t"] += max(secs * 4, 2.0)

    import cli as cli_mod

    with patch.object(gwc, "load_gateway_config", return_value=gw_config), \
         patch("time.time", side_effect=fake_time), \
         patch("time.sleep", side_effect=fake_sleep), \
         patch.object(cli_mod, "_cprint", side_effect=lambda s="": printed.append(s)):
        keep_going = host._handle_handoff_command("/handoff discord")
    return keep_going, printed, host


class TestCLIWaitLoop:
    def test_pending_timeout_still_reports_gateway_down(self, store, monkeypatch):
        """No watcher ever claims the row -> 60s pending timeout, row failed."""
        seed_session(store, "cli-sess-a", "cli")
        keep, printed, _ = _run_handoff(_offline_store(), "cli-sess-a", monkeypatch)
        out = "\n".join(printed)
        assert keep is True
        assert t("cli.commands.handoff.timed_out") in out
        assert _state(store, "cli-sess-a") == "failed"

    def test_without_a_session_store_the_handoff_runs_on_the_home_store(self, tmp_path, monkeypatch):
        """No session store on the CLI: the handoff is armed and CAS-failed on the current home's store."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        local = storage_handle(home)
        seed_session(local, "cli-sess-local", "cli")
        keep, printed, _ = _run_handoff(None, "cli-sess-local", monkeypatch)
        assert keep is True
        assert t("cli.commands.handoff.timed_out") in "\n".join(printed)
        assert _state(local, "cli-sess-local") == "failed"

    def test_running_row_is_never_failed_by_cli(self, store, monkeypatch):
        """Row claimed (running) and never finishing: CLI gives up eventually
        but must NOT fail the row and must NOT print the gateway-down line."""
        seed_session(store, "cli-sess-b", "cli")
        # Pre-claim: by the time the CLI polls, the gateway owns the row.
        # request_handoff happens inside the command; claim it from a fake
        # watcher the instant it lands via a get_handoff_state side hook.
        real_get = SurfaceLabelReader.get_handoff_state

        def claiming_get(reader, sid):
            row = real_get(reader, sid)
            if row and row.get("state") == "pending":
                store.coordination.claim_handoff(session_id=sid)
                row = real_get(reader, sid)
            return row

        monkeypatch.setattr(SurfaceLabelReader, "get_handoff_state", claiming_get)

        keep, printed, _ = _run_handoff(_offline_store(), "cli-sess-b", monkeypatch)
        out = "\n".join(printed)
        assert keep is True
        assert t("cli.commands.handoff.timed_out") not in out
        assert t("cli.commands.handoff.taking_long") in out
        # The row is still owned by the gateway — untouched by the CLI.
        assert _state(store, "cli-sess-b") == "running"
        # Late gateway completion wins cleanly (no split-brain).
        store.coordination.complete_handoff(session_id="cli-sess-b")
        assert _state(store, "cli-sess-b") == "completed"

    def test_slow_dispatch_beyond_60s_completes(self, store, monkeypatch):
        """The exact live-repro shape: claim @~5s, complete @~80s simulated.
        Old code timed out at 60s with the false 'gateway running?' message;
        new code waits through the running phase and exits on completed."""
        seed_session(store, "cli-sess-c", "cli")
        t0 = {"polls": 0}
        real_get = SurfaceLabelReader.get_handoff_state

        def slow_gateway(reader, sid):
            t0["polls"] += 1
            row = real_get(reader, sid)
            state = (row or {}).get("state")
            if state == "pending" and t0["polls"] >= 2:
                store.coordination.claim_handoff(session_id=sid)
            # each poll ~2s simulated; complete after ~40 polls (~80s+)
            if state == "running" and t0["polls"] >= 45:
                store.coordination.complete_handoff(session_id=sid)
            return real_get(reader, sid)

        monkeypatch.setattr(SurfaceLabelReader, "get_handoff_state", slow_gateway)

        keep, printed, host = _run_handoff(_offline_store(), "cli-sess-c", monkeypatch)
        out = "\n".join(printed)
        assert keep is False  # completed -> CLI exits like /quit
        assert t("cli.commands.handoff.complete", platform="discord") in out
        assert t("cli.commands.handoff.timed_out") not in out
        assert host._should_exit is True
        assert _state(store, "cli-sess-c") == "completed"


# ---------------------------------------------------------------------------
# 3. Desktop handoff.fail RPC
# ---------------------------------------------------------------------------

class TestDesktopHandoffFail:
    def _call(self, db, session_key):
        """Invoke the handoff.fail handler body with server-global stand-ins."""
        import contextlib

        from tui_gateway import methods_session as ms
        from tui_gateway import server

        handler = None
        for name, fn in ms._registry._pending:
            if name == "handoff.fail":
                handler = fn
                break
        assert handler is not None, "handoff.fail handler not registered"

        session = {"session_key": session_key}

        @contextlib.contextmanager
        def fake_session_db(_s):
            yield db

        globs = dict(handler.__globals__)
        globs["_sess_nowait"] = lambda p, r: (session, None)
        globs["_session_db"] = fake_session_db
        globs["_persistence_for"] = server._persistence_for
        globs["_ok"] = lambda rid, result: {"ok": True, **result}
        globs["_db_unavailable_error"] = lambda rid, code: {"error": code}
        rebound = types.FunctionType(
            handler.__code__, globs, handler.__name__,
            handler.__defaults__, handler.__closure__,
        )
        return rebound("rid", {"error": "poll timeout"})

    def test_desktop_fail_refuses_running_row(self, store):
        seed_session(store, "d1", "desktop")
        assert store.coordination.request_handoff(session_id="d1", platform="discord")
        assert store.coordination.claim_handoff(session_id="d1")
        res = self._call(store, "d1")
        assert res["failed"] is False
        assert res["state"] == "running"
        assert _state(store, "d1") == "running"

    def test_desktop_fail_still_fails_pending_row(self, store):
        seed_session(store, "d2", "desktop")
        assert store.coordination.request_handoff(session_id="d2", platform="discord")
        res = self._call(store, "d2")
        assert res["failed"] is True
        assert _state(store, "d2") == "failed"
