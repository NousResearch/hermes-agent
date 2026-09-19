"""Regression: kanban worker handoff stays lean (task-scoped, bounded).

Task t_4bf79a41 — enforces the audit verdict that worker spawn carries no
full-session context. Guards against future regressions reintroducing context
bloat in the kanban handoff.

Two invariants are enforced:

  1. ``build_worker_context`` output stays BOUNDED under pathological inputs
     (huge body, comment storm, oversized metadata) and per-field caps
     truncate. A future change that removes the caps and lets one giant field
     dominate the prompt fails here.
  2. The spawned worker env contains NO content/session blob and the spawn
     prompt is exactly the fixed, minimal ``work kanban task <id>`` string.
     A future change that starts passing full-session history on spawn fails
     here.

Run:  pytest tests/plugins/test_kanban_context_leanness.py
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures (mirror the kanban_db fixtures used by the other plugin tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb.Path, "home", lambda: tmp_path)
    # Force re-init against the isolated home.
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    _conn = kb.connect()
    try:
        yield _conn
    finally:
        _conn.close()


# A generous but hard ceiling. Worst realistic render with every cap maxed
# stays far under this; a regression that drops the caps would blow past it.
_CTX_CEILING = 512 * 1024  # 512 KB


# ---------------------------------------------------------------------------
# Invariant 1: bounded, capped rendering
# ---------------------------------------------------------------------------


def test_worker_context_bounded_under_pathological_inputs(conn):
    """Huge body + comment storm + oversized metadata must stay bounded."""
    task_id = kb.create_task(
        conn,
        title="pathological",
        assignee="architect",
        body="x" * (64 * 1024),           # 64 KB body (> 8 KB cap)
    )
    for i in range(200):                  # comment storm (> 30 shown, > 2 KB each)
        kb.add_comment(conn, task_id, author=f"w{i}", body="y" * (16 * 1024))

    ctx = kb.build_worker_context(conn, task_id)

    assert len(ctx) < _CTX_CEILING, (
        f"worker context unexpectedly large: {len(ctx)} chars"
    )
    # The 64 KB body must be capped (8 KB cap) -> truncation marker present.
    assert "… [truncated" in ctx
    # Comment storm must be collapsed: an "earlier comments omitted" marker.
    assert "earlier comment" in ctx


def test_worker_context_caps_per_field(conn):
    """A single oversized summary/result must be ellipsized, not dominant."""
    task_id = kb.create_task(conn, title="caps", assignee="architect")

    # Record an oversized summary via a completed run so it flows into context.
    # claim + complete writes a run; synthesize one directly is simpler but the
    # public path keeps the test honest with the real lifecycle.
    kb.claim_task(conn, task_id)
    # complete_task requires a summary; provide an oversized one.
    kb.complete_task(
        conn, task_id,
        expected_run_id=kb._current_run_id(conn, task_id),
        summary="z" * (128 * 1024),       # 128 KB summary (> 4 KB cap)
    )

    ctx = kb.build_worker_context(conn, task_id)
    assert "… [truncated" in ctx, "oversized summary must be ellipsized"
    assert len(ctx) < _CTX_CEILING


def test_worker_context_no_full_session_history(conn):
    """Only the task's own + recent assignee handoff summaries appear; the
    worker context must never contain full prior conversation transcripts."""
    task_id = kb.create_task(
        conn, title="lean", assignee="architect", body="A short lean body."
    )
    ctx = kb.build_worker_context(conn, task_id)
    # The context is a bounded composition; assert structural markers.
    assert ctx.startswith("# Kanban task")
    assert "## Body" in ctx
    # No section that would smuggle in arbitrary session history.
    for forbidden in ("## Session history", "## Conversation", "prior messages"):
        assert forbidden not in ctx


# ---------------------------------------------------------------------------
# Invariant 2: spawn carries task identity only, never content
# ---------------------------------------------------------------------------


def test_spawn_prompt_and_env_are_lean(conn, monkeypatch, tmp_path):
    """The spawned worker's argv is the fixed minimal prompt and its env holds
    only task-scoped identity vars — no context/session blob.

    The spawn path lives in ``hermes_cli.kanban_db_dispatch._default_spawn``
    (the kanban_db + dispatch split). Patching ``subprocess.Popen`` here mirrors
    the established `test_kanban_worker_session_source` fixture, which drives
    the real ``_default_spawn`` so the leak assertions exercise the actual
    dispatch code.
    """
    import subprocess

    from hermes_cli import kanban_db_dispatch as kbd

    task_id = kb.create_task(conn, title="spawn-lean", assignee="architect")
    kb.claim_task(conn, task_id)
    task = kb.get_task(conn, task_id)
    assert task is not None

    captured = {}

    class _Proc:
        pid = 424242
        returncode = None

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env", {}))
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    # Dispatch-only plumbing; keep the test on the leak + prompt assertions.
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(
        kbd, "_restart_safe_worker_argv", lambda task, command: command
    )
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    ws = str(tmp_path / "ws")
    import os as _os
    _os.makedirs(ws, exist_ok=True)
    kbd._default_spawn(task, ws, board="default")

    assert captured["cmd"], "spawn argv was not captured"
    # The prompt is exactly the fixed minimal string and the argv lands the
    # fixed `chat -q "work kanban task <id>"` tail (cmd[-1] is the prompt).
    assert captured["cmd"][-1] == f"work kanban task {task_id}"
    assert "chat" in captured["cmd"] and "-q" in captured["cmd"]
    env = captured["env"]
    # Task identity must be present.
    assert env.get("HERMES_KANBAN_TASK") == task_id
    assert env.get("HERMES_KANBAN_WORKSPACE") == ws
    # No content/session/history blob key is set on the child env.
    blob_like = [
        k for k in env
        if "CONTEXT" in k.upper() or "SESSION_HISTORY" in k.upper()
        or "HISTORY" in k.upper() and k != "HERMES_SESSION_SOURCE"
    ]
    assert blob_like == [], f"child env carries context/session blob keys: {blob_like}"