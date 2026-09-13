from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban as kc
from hermes_cli.kanban_lease_spec import claim_lease_exec_spec


CAPABILITY = {
    "provider": "openai-codex",
    "model_id": "gpt-5.6-sol",
    "effort": "low",
}


def _routed_task(conn, title: str, *, priority: int = 0) -> str:
    kb.put_route_policy(
        conn, policy_ref="sunny.worker", policy_version=1,
        provider_ref=CAPABILITY["provider"], model_ref=CAPABILITY["model_id"],
        effort_ref=CAPABILITY["effort"],
    )
    return kb.create_task(
        conn, title=title, priority=priority, tenant="business-a",
        workspace_kind="dir", workspace_path=f"/workspaces/{title}",
        bucket_key="coding", route_policy_ref="sunny.worker", route_policy_version=1,
    )


def test_lease_spec_selects_one_task_and_returns_the_complete_identity(tmp_path):
    path = tmp_path / "kanban.db"
    with kbc.connect_closing(db_path=path) as conn:
        wanted = _routed_task(conn, "high", priority=9)
        _routed_task(conn, "low", priority=1)
        result = claim_lease_exec_spec(
            conn, worker_identity="worker-7", worker_capabilities=[CAPABILITY],
            bucket_key="coding", ttl_seconds=120, board="default",
        )
        assert result["ok"] is True
        assert result["task_id"] == wanted
        assert result["claim_lock"] == result["claim_token"]
        assert result["claim_lock"].startswith("worker-7:")
        assert result["scope"] == {"tenant": "business-a", "business": "business-a"}
        assert result["workspace"] == "/workspaces/high"
        assert result["route"] == {
            "provider": "openai-codex", "model": "gpt-5.6-sol",
            "requested_effort": "low", "applied_effort": "low",
        }
        assert result["claim_expires"] - result["heartbeat_at"] == 120
        assert len(kb.list_runs(conn, wanted)) == 1


def test_lease_claim_is_atomic_and_failed_readback_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / "kanban.db"
    with kbc.connect_closing(db_path=path) as conn:
        task_id = _routed_task(conn, "only")

    def claim(worker):
        with kbc.connect_closing(db_path=path) as conn:
            return claim_lease_exec_spec(
                conn, worker_identity=worker, worker_capabilities=[CAPABILITY],
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("worker-a", "worker-b")))
    assert sum(result["ok"] for result in results) == 1
    with kbc.connect_closing(db_path=path) as conn:
        winner = next(result for result in results if result["ok"])
        guards = {
            "expected_run_id": winner["run_id"],
            "expected_claim_lock": winner["claim_lock"],
            "expected_tenant": winner["tenant"],
            "expected_workspace_path": winner["workspace"],
        }
        assert kb.reclaim_task(conn, task_id, **guards)

        original = kb.get_task
        monkeypatch.setattr(kb, "get_task", lambda *_args, **_kwargs: None)
        failed = claim_lease_exec_spec(
            conn, worker_identity="worker-c", worker_capabilities=[CAPABILITY],
        )
        monkeypatch.setattr(kb, "get_task", original)
        assert failed["code"] == "lease_readback_failed"
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.claim_lock is None
        assert len(kb.list_runs(conn, task_id)) == 1


def test_lease_spec_cli_emits_the_same_exact_contract(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with kbc.connect_closing() as conn:
        task_id = _routed_task(conn, "cli")
    result = json.loads(kc.run_slash(
        "lease-spec --worker-id fm-1 "
        "--capability openai-codex:gpt-5.6-sol:low --json"
    ))
    assert result["ok"] is True
    assert result["task_id"] == task_id
    assert result["board"] == "default"
    assert result["worker_identity"] == "fm-1"
