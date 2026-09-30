"""Provider reset-time plumbing for kanban rate-limit requeues (#127495).

A worker exiting ``KANBAN_RATE_LIMIT_EXIT_CODE`` was re-spawned on a fixed
300 s cooldown forever, whatever the provider said: a multi-day usage-limit
wall turned into ~240 doomed full-agent spawns per card per day. The worker
now carries the provider's reset epoch to the dispatcher (a ``reset=``
trailer next to the ``rc=`` trailer, stashed from the turn result's
``failure_resets_at``), the dead-worker sweep records it in the rate-limited
run's metadata, and ``check_respawn_guard`` holds the card until that moment
instead of probing on cooldown alone.
"""

import os

import pytest

import hermes_cli.kanban_db as _kb
import hermes_cli.kanban_db_dispatch as _kbd
import hermes_cli.quiet_single_query as _qsq
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc


@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban"))
    return tmp_path


def _seed_rate_limited_run(conn, tid, *, ended_at, metadata=None):
    import json

    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    conn.execute(
        "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
        "ended_at=?, metadata=? WHERE id=?",
        (ended_at, json.dumps(metadata, ensure_ascii=False) if metadata else None, run_id),
    )
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, "
        "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
        "last_failure_error=? WHERE id=?",
        ("pid 1 exited rate-limited (quota wall) — requeued", tid),
    )
    conn.commit()


class TestWorkerResetTrailer:
    def test_exit_single_query_writes_reset_trailer(self, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_RESET_AT", "5000.5")
        with pytest.raises(SystemExit) as excinfo:
            _qsq.exit_single_query(_kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        assert excinfo.value.code == _kb.KANBAN_RATE_LIMIT_EXIT_CODE
        err = capsys.readouterr().err
        assert _qsq.KANBAN_WORKER_EXIT_TRAILER + "75" in err
        assert _qsq.KANBAN_WORKER_RESET_TRAILER + "5000.5" in err

    def test_reset_trailer_only_written_when_stashed(self, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_RESET_AT", raising=False)
        with pytest.raises(SystemExit):
            _qsq.exit_single_query(_kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        err = capsys.readouterr().err
        assert _qsq.KANBAN_WORKER_RESET_TRAILER not in err

    def test_rc_trailer_regex_unaffected_by_extra_line(self):
        log = (
            "some worker output\n"
            "[kanban-worker-exit] rc=75\n"
            "[kanban-worker-exit] reset=4999\n"
        )
        assert _kbd._EXIT_TRAILER_RE.findall(log) == ["75"]
        assert _kbd._RESET_TRAILER_RE.findall(log) == ["4999"]

    def test_worker_log_reset_reader(self, monkeypatch, tmp_path):
        """The dispatcher reads the reset epoch from the worker log tail."""
        tid = "task-reset-1"
        log_dir = tmp_path / "kanban" / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / f"{tid}.log").write_text(
            "output\n[kanban-worker-exit] rc=75\n[kanban-worker-exit] reset=5100.5\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_kb, "read_worker_log", lambda task_id, tail_bytes=0, board=None: (
            (log_dir / f"{task_id}.log").read_text(encoding="utf-8")))
        assert _kbd._worker_log_rate_limit_reset_at(tid) == 5100.5
        assert _kbd._worker_log_rate_limit_reset_at("nope") is None


class TestNoteResetFromTurnResult:
    def test_failure_resets_at_is_stashed(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_RESET_AT", raising=False)
        from hermes_cli.cli_single_query import _note_rate_limit_reset_for_dispatcher

        _note_rate_limit_reset_for_dispatcher({"failure_resets_at": 4800.0})
        assert os.environ["HERMES_KANBAN_RATE_LIMIT_RESET_AT"] == "4800.0"

    def test_ignored_outside_kanban(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_RESET_AT", raising=False)
        from hermes_cli.cli_single_query import _note_rate_limit_reset_for_dispatcher

        _note_rate_limit_reset_for_dispatcher({"failure_resets_at": 4800.0})
        assert "HERMES_KANBAN_RATE_LIMIT_RESET_AT" not in os.environ

    def test_non_numeric_resets_ignored(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
        monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_RESET_AT", raising=False)
        from hermes_cli.cli_single_query import _note_rate_limit_reset_for_dispatcher

        for bad in ("soon", True, None, {"at": 5}):
            _note_rate_limit_reset_for_dispatcher({"failure_resets_at": bad})
            assert "HERMES_KANBAN_RATE_LIMIT_RESET_AT" not in os.environ


class TestRespawnGuardResetWindow:
    def test_guard_holds_until_provider_reset(self, kanban_home, monkeypatch):
        """Past the fixed cooldown but before the provider's reset: hold."""
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        now = 5_000_000
        reset_at = now + 4 * 86_400  # a 4-day wall

        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="rl-reset", assignee="a")
            _seed_rate_limited_run(
                conn, tid, ended_at=now,
                metadata={"rate_limit_reset_at": float(reset_at)})

            monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
            assert _kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

    def test_guard_releases_after_reset_passes(self, kanban_home, monkeypatch):
        """A reset in the past must not hold the card."""
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        now = 5_000_000

        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="rl-reset-past", assignee="a")
            _seed_rate_limited_run(
                conn, tid, ended_at=now,
                metadata={"rate_limit_reset_at": float(now + 100)})

            monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
            assert _kbd.check_respawn_guard(conn, tid) is None

    def test_no_reset_metadata_keeps_cooldown_behavior(self, kanban_home, monkeypatch):
        """Backward compat: metadata without the marker behaves exactly as before."""
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        now = 5_000_000

        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="rl-legacy", assignee="a")
            _seed_rate_limited_run(conn, tid, ended_at=now, metadata={"pid": 7})

            monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
            assert _kbd.check_respawn_guard(conn, tid) is None

            monkeypatch.setattr(_kb.time, "time", lambda: now + 100)
            assert _kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

    def test_malformed_reset_metadata_ignored(self, kanban_home, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        now = 5_000_000

        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="rl-bad", assignee="a")
            _seed_rate_limited_run(
                conn, tid, ended_at=now, metadata={"rate_limit_reset_at": "tomorrow"})

            monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
            assert _kbd.check_respawn_guard(conn, tid) is None
