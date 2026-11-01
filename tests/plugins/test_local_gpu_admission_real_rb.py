"""Exercise the real Resource Balancer GPU admission API against a temporary queue only."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


def test_real_gpu_admission_queue_holds_and_releases_owner_bound_lease(tmp_path):
    tools_root = Path(os.environ.get(
        "LOCAL_GPU_RESOURCE_BALANCER_TOOLS", "/mnt/MiroT20/synapsium/tools",
    )).resolve()
    if not (tools_root / "resource_balancer" / "admission.py").is_file():
        pytest.skip("Resource Balancer source is not available in this checkout")

    sys.path.insert(0, str(tools_root))
    from resource_balancer.core import Controller, Store, configuration
    from plugins.local_gpu_admission import ResourceBalancerController
    from hermes_cli.kanban_local_model_scope import task_run_contract_sha256

    state_dir = tmp_path / "rb-state"
    store = Store(state_dir)
    store.status({
        "mode": "manage",
        "pressure": "normal",
        "blocked": False,
        "resource_gates": {
            "disk": {"blocked": False, "reason": "test"},
            "gpu": {"blocked": False, "reason": "test"},
        },
        "updated": time.time(),
    })
    store.db.close()

    processes = {}
    errors = []
    completed = threading.Event()
    competing_blocked = threading.Event()
    competing_id = {}

    class LeaseOnlyBackend:
        def launch(self, job, _directory):
            process = subprocess.Popen(
                json.loads(job["argv"]),
                cwd=job["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes[job["id"]] = process

    def schedule_temp_queue():
        worker_store = Store(state_dir)
        try:
            controller = Controller(worker_store, LeaseOnlyBackend(), configuration(), manage=True)
            controller.resource_gates = {
                "disk": {"blocked": False},
                "gpu": {"blocked": False},
            }
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                controller.admit()
                if competing_id.get("value"):
                    competing = worker_store.db.execute(
                        "SELECT state FROM jobs WHERE id=?", (competing_id["value"],),
                    ).fetchone()
                    if competing and competing["state"] == "queued":
                        competing_blocked.set()
                for job_id, process in tuple(processes.items()):
                    return_code = process.poll()
                    if return_code is not None:
                        worker_store.update(job_id, "succeeded" if return_code == 0 else "failed")
                        completed.set()
                if completed.is_set():
                    return
                time.sleep(0.01)
            errors.append(TimeoutError("temporary RB queue did not settle"))
        except Exception as exc:  # surface scheduler-thread errors in the owning test
            errors.append(exc)
        finally:
            worker_store.db.close()

    scheduler = threading.Thread(target=schedule_temp_queue, daemon=True)
    scheduler.start()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}", encoding="utf-8")
    task = SimpleNamespace(
        id="rb-task-1", current_run_id=1, title="test", body="capability",
        assignee="test-profile", workspace_kind="scratch", workspace_path=str(tmp_path),
        branch_name="", model_override="fixture-model", provider_override="ollama",
        reasoning_effort="", completion_contract="local-only",
        skills=["local-gpu-model-task-gate"], max_runtime_seconds=300, max_retries=1,
        goal_mode=False, goal_max_turns=None, claim_lock="claim-1",
    )
    scope = {
        "task_id": task.id, "kanban_task_id": task.id, "task_run_id": "1",
        "kanban_board": "default", "kanban_db": str(tmp_path / "kanban.db"),
        "kanban_workspace": task.workspace_path, "kanban_claim_lock": task.claim_lock,
        "capability_id": "fixed-json.v1", "task_contract_sha256": task_run_contract_sha256(task),
        "api_request_id": "request-1", "context_length": "4096", "ollama_num_ctx": "4096",
        "session_id": "session-1",
        "model": "fixture-model", "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1/",
        "streaming": False, "has_tools": False, "api_call_count": 1,
        "existing_response_format": False,
    }
    controller = ResourceBalancerController(
        profile="test-profile",
        config={
            "resource_balancer_python_path": str(tools_root),
            "resource_balancer_state_dir": str(state_dir),
            "capability_registry_path": str(registry_path),
            "capability_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
    )
    controller._read_task = lambda _context: task
    controller._load_oracle = lambda _task, _scope: (
        hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "v" * 64, tmp_path / "verify.py", "f" * 64,
        {"digest": "d" * 64, "quantization": "Q4_K_M", "runtime_version": "0.34.2"},
        {"answer": "42"},
        {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    )
    controller._live_model_evidence = lambda _scope: {
        "digest": "d" * 64,
        "quantization": "Q4_K_M",
        "runtime_version": "0.34.2",
    }

    lease_id = None
    try:
        acquired = controller(action="acquire", **scope)
        assert acquired["decision"] == "allow", acquired
        lease_id = acquired["lease_ids"][0]
        validated = controller(action="validate", lease_ids=(lease_id,), **scope)
        assert validated["decision"] == "allow", validated
        writer = Store(state_dir)
        try:
            competing_id["value"] = writer.submit(
                "gpu", str(tmp_path), ["/usr/bin/true"],
            )
        finally:
            writer.db.close()
        assert competing_blocked.wait(timeout=3), "second GPU admission was not observed"
        reader = Store(state_dir)
        try:
            competing_row = reader.db.execute(
                "SELECT state FROM jobs WHERE id=?", (competing_id["value"],),
            ).fetchone()
            assert competing_row["state"] == "queued"
        finally:
            reader.db.close()
        released = controller(action="release", lease_ids=(lease_id,), **scope)
        assert released["decision"] == "released", released
    finally:
        scheduler.join(timeout=10)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)

    assert not scheduler.is_alive(), "temporary RB scheduler did not stop"
    assert not errors, errors
    assert lease_id
    reader = Store(state_dir)
    try:
        row = reader.db.execute("SELECT class,state FROM jobs WHERE id=?", (lease_id,)).fetchone()
        assert row["class"] == "gpu"
        assert row["state"] == "succeeded"
        competing_row = reader.db.execute(
            "SELECT state FROM jobs WHERE id=?", (competing_id["value"],),
        ).fetchone()
        assert competing_row["state"] == "queued"
        reader.cancel(competing_id["value"])
    finally:
        reader.db.close()
