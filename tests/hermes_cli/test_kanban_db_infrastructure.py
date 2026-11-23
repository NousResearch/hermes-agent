"""Linux systemd worker placement contract (moved out of the oversized DB test file)."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home



@pytest.mark.platforms("linux")
def test_infrastructure_spawn_refusal_never_charges_the_card(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """The host refusing to place a worker (managed gateway, user bus gone —
    #114720) is not a card failure: through the REAL spawn boundary and the
    real dispatcher accounting, ``consecutive_failures`` stays put, the breaker
    never parks the card as a bare ``blocked``, the run is tagged
    ``infrastructure`` and the guard spaces the retries. A control spawn
    failure on the same card still counts."""
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway")
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    def spawn_via_real_boundary(task, workspace, board=None):
        kbd._restart_safe_worker_argv(task, ["hermes", "chat"])  # raises: real probe verdict, real _degrade()
        raise AssertionError("unreachable")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="bus is down", assignee="a")
        for _ in range(3):
            res = kbd.dispatch_once(conn, spawn_fn=spawn_via_real_boundary, failure_limit=2)
            assert res.auto_blocked == []
        row = conn.execute(
            "SELECT status, block_kind, consecutive_failures, last_failure_error FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert (row["status"], row["block_kind"], row["consecutive_failures"]) == ("ready", None, 0)
        assert "enable-linger" in row["last_failure_error"]
        runs = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["spawn_failed"] * 3
        assert all(json.loads(r["metadata"])["infrastructure"] is True for r in runs)

        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"

        # Control: an ordinary spawn failure on the same card still spends budget.
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

        def spawn_broken(task, workspace, board=None):
            raise RuntimeError("profile launcher exploded")

        kbd.dispatch_once(conn, spawn_fn=spawn_broken, failure_limit=2)
        assert conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,),
        ).fetchone()[0] == 1

