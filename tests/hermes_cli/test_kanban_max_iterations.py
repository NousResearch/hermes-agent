"""Per-card iteration budget (Fix B) + iteration-cap-routes-to-blocked (retry-fix).

Covers:
  * ``max_iterations`` round-trips through create_task → persist → get_task.
  * When set, ``_default_spawn`` exports HERMES_MAX_ITERATIONS into the worker
    env; when unset it injects nothing (worker resolves the global default).
  * Iteration-cap exhaustion routes the task to ``blocked`` (force_trip),
    NOT the below-threshold auto-retry ``ready`` phase, and preserves the
    reviewer-run guard.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home



# ---------------------------------------------------------------------------
# Fix B — persistence round-trip
# ---------------------------------------------------------------------------


def test_max_iterations_round_trips_create_persist_show(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="big task", assignee="gohanlite", max_iterations=80,
        )
        task = kb.get_task(conn, t)
        assert task.max_iterations == 80


def test_max_iterations_unset_defaults_to_none(kanban_home):
    """A card created without the field must persist NULL, so the worker
    falls through to the global default (existing behaviour unchanged)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="normal task", assignee="gohanlite")
        task = kb.get_task(conn, t)
        assert task.max_iterations is None


# ---------------------------------------------------------------------------
# Fix B — spawn-env injection
# ---------------------------------------------------------------------------


def _make_task(kb_mod, *, max_iterations):
    return kb_mod.Task(
        id="t_iter_budget",
        title="iter budget",
        body=None,
        assignee="gohanlite",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        max_iterations=max_iterations,
    )


def _capture_spawn_env(monkeypatch, tmp_path, *, max_iterations):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "gohanlite"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "toolsets:\n  - hermes-cli\n", encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb._default_spawn(
        _make_task(kb, max_iterations=max_iterations), str(workspace),
    )
    return captured["env"]


def test_default_spawn_exports_max_iterations_when_set(monkeypatch, tmp_path):
    env = _capture_spawn_env(monkeypatch, tmp_path, max_iterations=80)
    assert env["HERMES_MAX_ITERATIONS"] == "80"


def test_default_spawn_omits_max_iterations_when_unset(monkeypatch, tmp_path):
    """Unset card must NOT pin HERMES_MAX_ITERATIONS, so the worker resolves
    the global agent.max_turns default exactly as before this change.

    The dispatcher scrubs session-routing ContextVars but otherwise inherits
    os.environ; assert the card injected nothing rather than that the key is
    globally absent.
    """
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    env = _capture_spawn_env(monkeypatch, tmp_path, max_iterations=None)
    assert "HERMES_MAX_ITERATIONS" not in env


# ---------------------------------------------------------------------------
# Retry-fix — iteration-cap exhaustion routes to blocked, not auto-retry
# ---------------------------------------------------------------------------


def test_iteration_exhaustion_routes_to_blocked_not_ready(kanban_home):
    """A running task whose worker exhausts its iteration budget must land in
    ``blocked`` (human decision: resize/split), NOT be re-queued ``ready`` to
    blind-retry into the same wall."""
    from agent.turn_finalizer import _record_kanban_budget_exhausted
    import logging

    with kb.connect() as conn:
        t = kb.create_task(conn, title="too big", assignee="gohanlite")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")

    # The worker path connects on its own; point it at this board's DB.
    _record_kanban_budget_exhausted(
        t, api_call_count=45, max_iterations=45,
        logger=logging.getLogger("test"),
    )

    with kb.connect() as conn:
        task = kb.get_task(conn, t)
        assert task.status == "blocked", (
            f"iteration-cap exhaustion should block, got {task.status!r}"
        )
        # A gave_up event should have been emitted with the block cause.
        events = kb.list_events(conn, t)
        gave_up = [e for e in events if e.kind == "gave_up"]
        assert gave_up, "expected a gave_up event on iteration-cap block"
        payload = gave_up[-1].payload or {}
        assert payload.get("block_cause") == "iteration_budget_exhausted"
        # Reason must be structured + human-readable, no secrets.
        assert "resize max_iterations" in (task.last_failure_error or "")


# ---------------------------------------------------------------------------
# #2 — CLI + agent-tool surface expose max_iterations
# ---------------------------------------------------------------------------


def test_cli_create_accepts_max_iterations(kanban_home):
    """`kanban create --max-iterations N` persists the budget, and it shows
    up in both `show` text and `list --json`."""
    from hermes_cli import kanban as kc

    out = kc.run_slash(
        "create 'big cli task' --assignee gohanlite "
        "--max-iterations 80 --json"
    )
    import json as _json
    created = _json.loads(out)
    assert created["max_iterations"] == 80

    with kb.connect() as conn:
        task = kb.get_task(conn, created["id"])
        assert task.max_iterations == 80

    show = kc.run_slash(f"show {created['id']}")
    assert "max-iterations: 80 (task)" in show


def test_cli_create_rejects_nonpositive_max_iterations(kanban_home):
    from hermes_cli import kanban as kc

    rc = kc.run_slash("create x --assignee a --max-iterations 0")
    # run_slash returns the printed output; the arg-validation path prints an
    # error to stderr and returns exit code 2 — assert nothing was created.
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
        assert not any(t.title == "x" for t in tasks)


def test_kanban_create_tool_schema_exposes_max_iterations():
    """The agent-facing tool schema must advertise the new field so the
    leader can set it from a kanban_create call."""
    from tools import kanban_tools

    spec = kanban_tools.KANBAN_CREATE_SCHEMA
    props = spec["parameters"]["properties"]
    assert "max_iterations" in props
    assert props["max_iterations"]["type"] == "integer"

