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
  1. SessionDB.fail_handoff(only_states=...) — CAS: waiters can only fail
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

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


# ---------------------------------------------------------------------------
# 1. State layer: fail_handoff CAS
# ---------------------------------------------------------------------------

class TestFailHandoffCAS:
    def test_cas_fails_pending_row(self, db):
        db.ensure_session("s1", "cli")
        assert db.request_handoff("s1", "discord")
        assert db.fail_handoff("s1", "timed out", only_states=("pending",)) is True
        assert db.get_handoff_state("s1")["state"] == "failed"

    def test_cas_refuses_running_row(self, db):
        """A waiter timeout must NOT stomp a row the gateway has claimed."""
        db.ensure_session("s2", "cli")
        assert db.request_handoff("s2", "discord")
        assert db.claim_handoff("s2")  # gateway claimed: pending -> running
        assert db.fail_handoff("s2", "timed out", only_states=("pending",)) is False
        assert db.get_handoff_state("s2")["state"] == "running"
        # gateway still reaches its own terminal state
        db.complete_handoff("s2")
        assert db.get_handoff_state("s2")["state"] == "completed"

    def test_unconditional_fail_still_available_to_owner(self, db):
        """The gateway watcher (owner of a claimed row) fails unconditionally."""
        db.ensure_session("s3", "cli")
        assert db.request_handoff("s3", "discord")
        assert db.claim_handoff("s3")
        assert db.fail_handoff("s3", "dispatch raised") is True
        assert db.get_handoff_state("s3")["state"] == "failed"

    def test_cas_refuses_terminal_rows(self, db):
        db.ensure_session("s4", "cli")
        assert db.request_handoff("s4", "discord")
        assert db.claim_handoff("s4")
        db.complete_handoff("s4")
        assert db.fail_handoff("s4", "late timeout", only_states=("pending",)) is False
        assert db.get_handoff_state("s4")["state"] == "completed"


# ---------------------------------------------------------------------------
# 2. CLI wait loop
# ---------------------------------------------------------------------------

def _run_handoff(db, session_id, monkeypatch, time_budget=30.0):
    """Drive the real _handle_handoff_command against a real SessionDB.

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
            self._session_db = db
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
    def test_pending_timeout_still_reports_gateway_down(self, db, monkeypatch):
        """No watcher ever claims the row -> 60s pending timeout, row failed."""
        db.ensure_session("cli-sess-a", "cli")
        keep, printed, _ = _run_handoff(db, "cli-sess-a", monkeypatch)
        out = "\n".join(printed)
        assert keep is True
        assert "Timed out waiting for the gateway" in out
        assert db.get_handoff_state("cli-sess-a")["state"] == "failed"

    def test_running_row_is_never_failed_by_cli(self, db, monkeypatch):
        """Row claimed (running) and never finishing: CLI gives up eventually
        but must NOT fail the row and must NOT print the gateway-down line."""
        db.ensure_session("cli-sess-b", "cli")
        # Pre-claim: by the time the CLI polls, the gateway owns the row.
        # request_handoff happens inside the command; claim it from a fake
        # watcher the instant it lands via a get_handoff_state side hook.
        real_get = db.get_handoff_state

        def claiming_get(sid):
            row = real_get(sid)
            if row and row.get("state") == "pending":
                db.claim_handoff(sid)
                row = real_get(sid)
            return row

        db_proxy = MagicMock(wraps=db)
        db_proxy.get_handoff_state.side_effect = claiming_get
        db_proxy.request_handoff.side_effect = db.request_handoff
        db_proxy.fail_handoff.side_effect = db.fail_handoff

        keep, printed, _ = _run_handoff(db_proxy, "cli-sess-b", monkeypatch)
        out = "\n".join(printed)
        assert keep is True
        assert "Is `hermes gateway` running?" not in out
        assert "taking unusually long" in out
        # The row is still owned by the gateway — untouched by the CLI.
        assert db.get_handoff_state("cli-sess-b")["state"] == "running"
        # Late gateway completion wins cleanly (no split-brain).
        db.complete_handoff("cli-sess-b")
        assert db.get_handoff_state("cli-sess-b")["state"] == "completed"

    def test_slow_dispatch_beyond_60s_completes(self, db, monkeypatch):
        """The exact live-repro shape: claim @~5s, complete @~80s simulated.
        Old code timed out at 60s with the false 'gateway running?' message;
        new code waits through the running phase and exits on completed."""
        db.ensure_session("cli-sess-c", "cli")
        t0 = {"polls": 0}
        real_get = db.get_handoff_state

        def slow_gateway(sid):
            t0["polls"] += 1
            row = real_get(sid)
            state = (row or {}).get("state")
            if state == "pending" and t0["polls"] >= 2:
                db.claim_handoff(sid)
            # each poll ~2s simulated; complete after ~40 polls (~80s+)
            if state == "running" and t0["polls"] >= 45:
                db.complete_handoff(sid)
            return real_get(sid)

        db_proxy = MagicMock(wraps=db)
        db_proxy.get_handoff_state.side_effect = slow_gateway
        db_proxy.request_handoff.side_effect = db.request_handoff
        db_proxy.fail_handoff.side_effect = db.fail_handoff

        keep, printed, host = _run_handoff(db_proxy, "cli-sess-c", monkeypatch)
        out = "\n".join(printed)
        assert keep is False  # completed -> CLI exits like /quit
        assert "Handoff complete" in out
        assert "Timed out waiting for the gateway" not in out
        assert host._should_exit is True
        assert db.get_handoff_state("cli-sess-c")["state"] == "completed"


# ---------------------------------------------------------------------------
# 3. Desktop handoff.fail RPC
# ---------------------------------------------------------------------------

class TestDesktopHandoffFail:
    def _call(self, db, session_key):
        """Invoke the handoff.fail handler body with server-global stand-ins."""
        import contextlib

        from tui_gateway import methods_session as ms

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
        globs["_ok"] = lambda rid, result: {"ok": True, **result}
        globs["_db_unavailable_error"] = lambda rid, code: {"error": code}
        rebound = types.FunctionType(
            handler.__code__, globs, handler.__name__,
            handler.__defaults__, handler.__closure__,
        )
        return rebound("rid", {"error": "poll timeout"})

    def test_desktop_fail_refuses_running_row(self, db):
        db.ensure_session("d1", "desktop")
        assert db.request_handoff("d1", "discord")
        assert db.claim_handoff("d1")
        res = self._call(db, "d1")
        assert res["failed"] is False
        assert res["state"] == "running"
        assert db.get_handoff_state("d1")["state"] == "running"

    def test_desktop_fail_still_fails_pending_row(self, db):
        db.ensure_session("d2", "desktop")
        assert db.request_handoff("d2", "discord")
        res = self._call(db, "d2")
        assert res["failed"] is True
        assert db.get_handoff_state("d2")["state"] == "failed"
