"""Tests for the per-task model-override write path.

Covers the WRITE side that was the only gap (the column, read, and the
`_default_spawn` `-m` append already existed):

  * ``create_task(model_override=...)`` persists; no-override → NULL.
  * ``set_task_model`` clears to NULL and signals rows-affected.
  * CLI ``--model`` / ``--clear-model`` / sentinel-untouched.
  * argv-injection round-trip (write → SQLite → reload → spawn argv).
  * retry/re-read path: a model changed between spawns is re-read.
  * the structured spawn log line (emitted only when set).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Legacy crash-grace semantics are irrelevant here; keep instant reclaim
    # so the retry path doesn't wait on a grace window.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Phase 1 — DB write path
# ---------------------------------------------------------------------------

def test_create_task_persists_model_override(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="opus task", assignee="worker",
            model_override="claude-opus-4-8",
        )
        task = kb.get_task(conn, tid)
    assert task.model_override == "claude-opus-4-8"


def test_create_task_without_override_is_null(kanban_home):
    """No-override create → column is SQL NULL, not '' and not a default."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
        task = kb.get_task(conn, tid)
        raw = conn.execute(
            "SELECT model_override FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert task.model_override is None
    assert raw["model_override"] is None  # genuine NULL at the storage layer


def test_set_task_model_sets_and_clears(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # set
        affected = kb.set_task_model(conn, tid, "claude-opus-4-8")
        assert affected == 1
        assert kb.get_task(conn, tid).model_override == "claude-opus-4-8"
        # clear → NULL
        affected = kb.set_task_model(conn, tid, None)
        assert affected == 1
        assert kb.get_task(conn, tid).model_override is None
        raw = conn.execute(
            "SELECT model_override FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert raw["model_override"] is None


def test_set_task_model_literal_empty_string(kanban_home):
    """The DB layer takes str|None literally — it does NOT interpret ''."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        affected = kb.set_task_model(conn, tid, "")
        assert affected == 1
        raw = conn.execute(
            "SELECT model_override FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        # Stored verbatim as empty string, not coerced to NULL.
        assert raw["model_override"] == ""


def test_set_task_model_nonexistent_id_rows_affected_zero(kanban_home):
    with kb.connect() as conn:
        affected = kb.set_task_model(conn, "t_doesnotexist", "claude-opus-4-8")
    assert affected == 0


# ---------------------------------------------------------------------------
# Phase 2 — CLI flags (via run_slash, the same entry CLI + gateway use)
# ---------------------------------------------------------------------------

def _created_id(out: str) -> str:
    m = re.search(r"(t_[a-f0-9]+)", out)
    assert m, f"no task id in output: {out!r}"
    return m.group(1)


def test_cli_create_model_shows_in_show(kanban_home):
    out = kc.run_slash("create 'opus task' --assignee alice --model claude-opus-4-8")
    tid = _created_id(out)
    show = kc.run_slash(f"show {tid}")
    assert "model:" in show
    assert "claude-opus-4-8" in show


def test_cli_create_no_model_no_model_line(kanban_home):
    out = kc.run_slash("create 'plain' --assignee alice")
    tid = _created_id(out)
    show = kc.run_slash(f"show {tid}")
    assert "model:" not in show


def test_cli_edit_clear_model_removes_line(kanban_home):
    out = kc.run_slash("create 'opus' --assignee alice --model claude-opus-4-8")
    tid = _created_id(out)
    assert "claude-opus-4-8" in kc.run_slash(f"show {tid}")
    edited = kc.run_slash(f"edit {tid} --clear-model")
    assert "Cleared model override" in edited
    show = kc.run_slash(f"show {tid}")
    assert "model:" not in show


def test_cli_edit_set_model(kanban_home):
    out = kc.run_slash("create 'plain' --assignee alice")
    tid = _created_id(out)
    edited = kc.run_slash(f"edit {tid} --model claude-opus-4-8")
    assert "Set model override" in edited
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).model_override == "claude-opus-4-8"


def test_cli_edit_unrelated_field_leaves_model_untouched(kanban_home):
    """Editing without --model must leave an existing override intact
    (the None sentinel means 'unchanged', not 'clear')."""
    out = kc.run_slash("create 'opus' --assignee alice --model claude-opus-4-8")
    tid = _created_id(out)
    # An edit that touches nothing model-related: pass only --result.
    # (result-backfill needs a done task; we don't assert it succeeds — we
    # assert the model override is left alone regardless.)
    kc.run_slash(f"edit {tid} --result 'some backfill'")
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).model_override == "claude-opus-4-8"


def test_cli_edit_model_and_clear_model_mutually_exclusive(kanban_home):
    out = kc.run_slash("create 'x' --assignee alice")
    tid = _created_id(out)
    res = kc.run_slash(f"edit {tid} --model claude-opus-4-8 --clear-model")
    assert "mutually exclusive" in res


def test_cli_edit_nonexistent_id_reports_failure(kanban_home):
    res = kc.run_slash("edit t_nope --model claude-opus-4-8")
    assert "cannot set model" in res


# ---------------------------------------------------------------------------
# Phase 3 — argv-injection round-trip (write → store → reload → spawn argv)
# ---------------------------------------------------------------------------

def _spawn_argv_for(monkeypatch, task) -> list:
    """Drive _default_spawn with Popen stubbed; return the captured argv."""
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)
    # Pin the hermes invocation to a single bare token so the dispatch argv is
    # deterministic across hosts. Without this, a box where `hermes` is not a
    # console-script on PATH falls back to `python -m hermes_cli.main`, which
    # injects a spurious first `-m` and defeats the single-`-m` round-trip
    # assertions (env-dependent test artifact, not a dispatch bug — the model
    # override is still added exactly once after the entrypoint). (2026-06-29)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    workspace = kb.resolve_workspace(task)
    pid = kb._default_spawn(task, str(workspace))
    assert pid == 4242
    return captured["cmd"]


@pytest.mark.parametrize("injected", ["x ; y", "a\nb", "--provider evil", "$(rm -rf /)"])
def test_argv_injection_single_m_token_round_trip(kanban_home, monkeypatch, injected):
    """A metacharacter/newline/flag-like model string set via the CLI must
    survive write → SQLite → reload and land as EXACTLY ONE -m value token,
    byte-for-byte equal to the stored string."""
    # Write via the CLI write path using set_task_model semantics (the CLI
    # edit path). Use the DB setter directly to avoid shell-quoting in the
    # test harness masking the round-trip — the point is the store→reload→argv
    # leg, and the CLI uses the same setter.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="inject", assignee="worker")
        affected = kb.set_task_model(conn, tid, injected)
        assert affected == 1

    # Reload from a fresh connection — proves persistence, not in-memory state.
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.model_override == injected  # byte-for-byte after reload

    argv = _spawn_argv_for(monkeypatch, task)
    # Exactly one -m, and its value is the stored string verbatim.
    assert argv.count("-m") == 1
    idx = argv.index("-m")
    assert argv[idx + 1] == injected
    # Nothing split it into a second argv token.
    assert injected in argv
    assert argv.count(injected) == 1


def test_argv_injection_via_cli_create(kanban_home, monkeypatch):
    """Same round-trip but the write enters through the CLI create flag,
    proving CLI arg parsing doesn't split the value either."""
    # shlex in run_slash handles quoting; embed a metachar string.
    out = kc.run_slash("create 'inj' --assignee worker --model 'x ; y'")
    tid = _created_id(out)
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.model_override == "x ; y"
    argv = _spawn_argv_for(monkeypatch, task)
    assert argv.count("-m") == 1
    assert argv[argv.index("-m") + 1] == "x ; y"


# ---------------------------------------------------------------------------
# Phase 3 — retry / re-read path: model changed between spawns is re-read
# ---------------------------------------------------------------------------

def test_model_reread_on_retry_spawn(kanban_home, monkeypatch, all_assignees_spawnable):
    """dispatch → worker exits → set_task_model to a NEW value between
    attempts → the SECOND spawn actually occurs and carries the new model."""
    spawns: list[list] = []

    def _stub_spawn(task, ws, *, board=None):
        # Capture the argv the real _default_spawn WOULD build, so we assert
        # the model the dispatcher would pass on this attempt. We rebuild the
        # -m portion from the task the dispatcher handed us (which it re-read
        # from the DB at claim time).
        argv = ["hermes", "-p", task.assignee or ""]
        if task.model_override:
            argv += ["-m", task.model_override]
        spawns.append(argv)
        # Return a pid that is already dead so the next dispatch reclaims it.
        return 2  # init; effectively never our child → treated as crashed

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="retry", assignee="worker",
            model_override="claude-sonnet-4-5",
        )
        # First dispatch → spawn #1 with the original model.
        kb.dispatch_once(conn, spawn_fn=_stub_spawn)
        assert len(spawns) == 1
        assert spawns[0][-2:] == ["-m", "claude-sonnet-4-5"]

        # Worker "exits": clear the pid + return the task to ready so the
        # next tick re-dispatches it. detect_crashed_workers handles the
        # dead pid; force the task back to ready directly to keep the test
        # deterministic regardless of crash-detection timing.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (tid,),
            )

        # Change the model BETWEEN attempts.
        assert kb.set_task_model(conn, tid, "claude-opus-4-8") == 1

        # Second dispatch → spawn #2 must occur AND carry the new model.
        kb.dispatch_once(conn, spawn_fn=_stub_spawn)
        assert len(spawns) == 2, "retry must re-invoke the spawn fn"
        assert spawns[1][-2:] == ["-m", "claude-opus-4-8"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 3 — observability: spawn log line emitted ONLY when set
# ---------------------------------------------------------------------------

def test_spawn_logs_override_when_set(kanban_home, monkeypatch, caplog):
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)

    class FakeProc:
        pid = 7

    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: FakeProc())

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="x", assignee="worker",
            model_override="claude-opus-4-8",
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)

    with caplog.at_level(logging.INFO, logger="hermes_cli.kanban_db"):
        kb._default_spawn(task, str(workspace))

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "kanban spawn" in m and f"task={tid}" in m
        and "model_override=claude-opus-4-8" in m
        for m in msgs
    ), f"expected structured spawn line, got: {msgs}"


def test_spawn_no_log_when_no_override(kanban_home, monkeypatch, caplog):
    monkeypatch.setattr(kb, "_kanban_worker_skill_available", lambda _h: False)

    class FakeProc:
        pid = 8

    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: FakeProc())

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)

    with caplog.at_level(logging.INFO, logger="hermes_cli.kanban_db"):
        kb._default_spawn(task, str(workspace))

    msgs = [r.getMessage() for r in caplog.records]
    assert not any("model_override=" in m for m in msgs), (
        f"no override → must emit no model_override spawn line, got: {msgs}"
    )
