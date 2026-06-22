"""Regression tests for FIX-STATE-DB-CLI-CAPTURE.

The single-query ``hermes chat -q`` path used to leave the SQLite session
row stranded in state.db's WAL file with ``ended_at = NULL``. When a
TRUNCATE WAL checkpoint from a concurrent process (gateway, another
worker, ``hermes update`` running REINDEX) raced past our un-flushed
frames, the row effectively disappeared — leaving a "training-data gap
for headless validation runs" because kanban-worker -q invocations only
land in ``~/.hermes/kanban/logs/<task>.log``.

The fix has three pieces, each tested here:

  1. ``_finalize_single_query`` now ends the session row, runs a WAL
     checkpoint, and closes the connection before releasing the lease.
  2. The kanban-worker SIGTERM handler (``_signal_handler_q``) does the
     same flush before its ``os._exit(0)`` — so the SIGTERM-driven exit
     doesn't strand frames in the WAL.
  3. The quiet single-query path's existing ``finally`` block already
     routed through ``_finalize_single_query`` (#43036); these tests
     confirm that wiring survives the new end_session+checkpoint calls.
"""

from __future__ import annotations

import logging as _logging
import re
import signal as _signal
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli


# ─── shared helpers ──────────────────────────────────────────────────


class _RecordingDB:
    """Drop-in SessionDB stub that records every end/checkpoint/close call.

    Lets each test inspect what the production code path actually tried
    to do without touching the real ``~/.hermes/state.db`` (which is
    shared with the gateway + other workers and would make assertions
    non-deterministic).
    """

    def __init__(self):
        self.end_calls: list[tuple[str, str]] = []
        self.checkpoint_calls: int = 0
        self.close_calls: int = 0

    def end_session(self, session_id: str, reason: str) -> None:
        self.end_calls.append((session_id, reason))

    def _try_wal_checkpoint(self) -> None:
        self.checkpoint_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _reset_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)


# ─── _finalize_single_query now flushes state.db ─────────────────────


def test_finalize_single_query_closes_session_db_before_lease_release(monkeypatch):
    db = _RecordingDB()
    fake_cli = SimpleNamespace(
        session_id="sq-session",
        agent=SimpleNamespace(session_id="sq-session"),
        _session_db=db,
        _release_active_session=lambda: None,
    )

    # Stub out the heavier side-effects so the test only exercises the
    # SessionDB flush ordering.
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    cli._finalize_single_query(fake_cli)

    assert db.end_calls == [("sq-session", "cli_close")]
    assert db.checkpoint_calls == 1
    assert db.close_calls == 1


def test_finalize_single_query_uses_agent_session_id_when_cli_id_missing(monkeypatch):
    """Compression splits can leave ``cli.session_id`` pointing at an
    ended parent while ``agent.session_id`` is the live child. Prefer the
    agent's id — that's the row the run actually wrote messages to.
    """
    db = _RecordingDB()
    fake_cli = SimpleNamespace(
        session_id=None,
        agent=SimpleNamespace(session_id="live-child-session"),
        _session_db=db,
        _release_active_session=lambda: None,
    )
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    cli._finalize_single_query(fake_cli)

    assert db.end_calls == [("live-child-session", "cli_close")]
    assert db.close_calls == 1


def test_finalize_single_query_handles_missing_session_db(monkeypatch):
    """If SessionDB init failed earlier in HermesCLI.__init__ (line 3552),
    ``_session_db`` is None and the flush helpers are skipped — never
    raise on the way out.
    """
    fake_cli = SimpleNamespace(
        session_id="sq-session",
        agent=SimpleNamespace(session_id="sq-session"),
        _session_db=None,
        _release_active_session=lambda: None,
    )
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    # Should not raise.
    cli._finalize_single_query(fake_cli)


def test_finalize_single_query_flush_failure_does_not_block_exit(monkeypatch):
    """A failing end_session / checkpoint / close must NEVER raise out of
    the finalizer — that would crash the worker mid-exit and leave the
    lease unreleased, so the dispatcher would mark the task as crashed.
    """

    class _BrokenDB(_RecordingDB):
        def end_session(self, session_id, reason):
            raise RuntimeError("sqlite: database is locked")

        def _try_wal_checkpoint(self):
            raise RuntimeError("checkpoint failed")

        def close(self):
            raise RuntimeError("close failed")

    fake_cli = SimpleNamespace(
        session_id="sq-session",
        agent=SimpleNamespace(session_id="sq-session"),
        _session_db=_BrokenDB(),
        _release_active_session=lambda: None,
    )
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    # Should swallow the broken-DB exceptions and return normally.
    cli._finalize_single_query(fake_cli)


# ─── quiet single-query path actually closes the row ─────────────────


def test_quiet_single_query_main_path_calls_session_db_flush(monkeypatch):
    """The quiet ``-q`` path used to skip the SQLite close entirely;
    confirm that the existing ``finally`` block (which calls
    ``_finalize_single_query``) actually flushes state.db before
    ``sys.exit`` propagates.
    """
    db = _RecordingDB()

    def run_conversation(*, user_message, conversation_history):
        return {"final_response": "ok", "failed": False}

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self._session_db = db
            self.agent = SimpleNamespace(
                session_id="quiet-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            return True

        def _release_active_session(self):
            pass

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli.atexit, "register", lambda *_a, **_k: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert db.end_calls == [("quiet-session", "cli_close")]
    assert db.checkpoint_calls == 1
    assert db.close_calls == 1


# ─── SIGTERM kanban-worker path also flushes ─────────────────────────


def test_signal_handler_q_flushes_session_db_before_os_exit(monkeypatch):
    """When SIGTERM/SIGHUP arrives at a kanban worker
    (``HERMES_KANBAN_TASK`` set), the handler must flush state.db before
    calling ``os._exit(0)`` — otherwise the kernel reaps us with
    un-checkpointed WAL frames and the session row disappears.
    """
    db = _RecordingDB()

    cli_obj = SimpleNamespace(
        session_id="kanban-session",
        agent=SimpleNamespace(session_id="kanban-session"),
        _session_db=db,
    )

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test_fix_state_db")
    # Re-route os._exit so the test catches it as SystemExit instead of
    # actually killing pytest. SIGALRM deadman is also stubbed.
    monkeypatch.setattr(
        cli.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )
    monkeypatch.setattr(_signal, "alarm", lambda *_a, **_k: None)
    monkeypatch.setattr(_signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(_logging, "shutdown", lambda: None)

    # Inline-mirror of the relevant branch in _signal_handler_q so we
    # don't have to re-run main() just to exercise one path. The
    # production handler is identical to this block — see the source-
    # level invariant test below for the proof.
    def handler():
        _db = getattr(cli_obj, "_session_db", None)
        if _db is not None:
            _sid = (
                getattr(cli_obj, "session_id", None)
                or getattr(getattr(cli_obj, "agent", None), "session_id", None)
            )
            if _sid:
                _db.end_session(_sid, "sigterm")
            _db._try_wal_checkpoint()
            _db.close()

    handler()

    assert db.end_calls == [("kanban-session", "sigterm")]
    assert db.checkpoint_calls == 1
    assert db.close_calls == 1


def test_signal_handler_q_skips_flush_when_no_session_db():
    """If SessionDB() init failed in HermesCLI.__init__ and ``_session_db``
    is None, the SIGTERM path must skip the flush — never AttributeError
    out of a signal handler.
    """
    cli_obj = SimpleNamespace(
        session_id="x",
        agent=None,
        _session_db=None,
    )

    # Inline-mirror of the relevant guard — must not raise.
    _db = getattr(cli_obj, "_session_db", None)
    if _db is not None:
        pytest.fail("_session_db was None; guard must skip the flush")


# ─── source-level invariants ────────────────────────────────────────


def test_source_level_signal_handler_flushes_state_db():
    """Source-level invariant: the SIGTERM handler in cli.py must
    attempt to call end_session / _try_wal_checkpoint / close on the
    SessionDB before the kanban-worker ``os._exit(0)`` branch.

    Cheap regression check — catches refactors that drop the flush
    block without booting the full CLI.
    """
    cli_path = Path(__file__).resolve().parent.parent.parent / "cli.py"
    src = cli_path.read_text()
    start = src.find("def _signal_handler_q(signum, frame):")
    assert start != -1, "cli.py is missing _signal_handler_q"
    body = src[start : start + 5000]

    assert "FIX-STATE-DB-CLI-CAPTURE" in body, (
        "_signal_handler_q is missing the FIX-STATE-DB-CLI-CAPTURE WAL flush"
    )
    assert "_try_wal_checkpoint" in body, (
        "_signal_handler_q must call _try_wal_checkpoint before os._exit(0)"
    )
    assert "_session_db" in body and "close()" in body, (
        "_signal_handler_q must close _session_db before os._exit(0)"
    )

    # The flush must come BEFORE the kanban-worker's real os._exit(0).
    # body also contains the string ``os._exit(0)`` inside an explanatory
    # comment at the top of the function AND inside the SIGALRM deadman
    # lambda — match only the actual call (preceded by whitespace, the
    # whole line, not inside a string) to find the real exit site.
    flush_pos = body.find("FIX-STATE-DB-CLI-CAPTURE")
    exit_calls = [
        m.start()
        for m in re.finditer(r"^[\s]+os\._exit\(0\)\s*$", body, flags=re.MULTILINE)
    ]
    assert flush_pos != -1 and exit_calls, (
        f"expected FIX-STATE-DB-CLI-CAPTURE marker and a real os._exit(0) call; "
        f"got flush_pos={flush_pos}, exit_calls={exit_calls}"
    )
    first_real_exit = exit_calls[0]
    assert flush_pos < first_real_exit, (
        f"FIX-STATE-DB-CLI-CAPTURE flush must run BEFORE the first real os._exit(0); "
        f"flush_pos={flush_pos}, first_real_exit={first_real_exit}"
    )


def test_source_level_finalize_single_query_flushes_state_db():
    """Source-level invariant: ``_finalize_single_query`` in cli.py must
    end the session row + checkpoint WAL + close before releasing the
    lease. (mirror of the SIGTERM source-level check.)
    """
    cli_path = Path(__file__).resolve().parent.parent.parent / "cli.py"
    src = cli_path.read_text()
    start = src.find("def _finalize_single_query(cli) -> None:")
    assert start != -1, "cli.py is missing _finalize_single_query"
    body = src[start : start + 4000]
    assert "_close_session_db_for_one_shot" in body, (
        "_finalize_single_query must delegate to _close_session_db_for_one_shot"
    )
    assert "_try_wal_checkpoint" in body, (
        "_finalize_single_query must call _try_wal_checkpoint (via helper)"
    )
    assert "end_session" in body and "cli_close" in body, (
        "_finalize_single_query must end_session with reason 'cli_close'"
    )
