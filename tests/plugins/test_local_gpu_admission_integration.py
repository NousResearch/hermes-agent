"""End-to-end admission lifecycle with real Kanban, registry, artifact, and verifier data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import ModuleType, SimpleNamespace

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_local_model_scope import CONTRACT_SCHEMA, task_run_contract_sha256
from hermes_cli.plugins import PluginManager
from plugins.local_gpu_admission import ResourceBalancerController


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_admission_lifecycle_reads_task_verifies_artifact_and_releases_lease(tmp_path, monkeypatch):
    home = tmp_path / ".hermes" / "profiles" / "test-profile"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    workspace = home / "kanban" / "workspaces" / "fixed-json-task"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-profile")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.init_db()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    expected = {"answer": "42"}
    expected_bytes = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    fixture_path = tmp_path / "expected.json"
    fixture_path.write_bytes(expected_bytes)
    verifier_path = tmp_path / "verify.py"
    verifier_path.write_text(
        "import json, sys\n"
        "assert json.load(open(sys.argv[1], encoding='utf-8')) == {'answer': '42'}\n",
        encoding="utf-8",
    )
    artifact_path = workspace / "result.json"

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="fixed JSON fixture",
            body=("Local-GPU-Capability: fixed-json.v1\n"
                  "Given facts {\"value\":\"42\"}, return an object with an answer field "
                  "containing the value of the value field."),
            assignee="test-profile",
            workspace_kind="scratch",
            workspace_path=str(workspace),
            skills=["local-gpu-model-task-gate"],
            max_runtime_seconds=300,
            max_retries=1,
            model_override="fixture-model",
            provider_override="ollama",
            completion_contract="local-only",
        )
        task = kb.claim_task(conn, task_id, claimer="claim-1")
        assert task is not None
        assert '{"answer":"42"}' not in task.body
        conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (os.getpid(), task.current_run_id))
        conn.commit()

    registry = {
        "task_run_binding": {"schema": CONTRACT_SCHEMA},
        "capabilities": [{
            "id": "fixed-json.v1",
            "allow": True,
            "status": "narrowly_supported",
            "model_tag": "fixture-model",
            "model_digest": "d" * 64,
            "request_schema_policy": "json_schema_from_fixture_types_v1",
            "trial_binding": {
                "admission_capability_id": "fixed-json.v1",
                "trial_task_class": "fixed-json.v1.trial",
                "mapping_review": {
                    "status": "approved", "reviewer": "test-reviewer",
                    "reviewed_at_utc": "2026-10-28T00:00:00Z",
                },
            },
            "profile_route_evidence": {"task_class": "fixed-json.v1.trial"},
            "observed_runtime": {
                "backend": "ollama",
                "base_url": "http://127.0.0.1:11434/v1/",
                "context_length": 4096,
                "ollama_num_ctx": 4096,
                "quantization": "Q4_K_M",
                "runtime_version": "0.34.2",
                "profile": "test-profile",
            },
            "prepared_external_verifier": {
                "task_id": task_id,
                "verifier_path": "verify.py",
                "verifier_sha256": _sha256(verifier_path),
                "expected_fixture_path": "expected.json",
                "expected_fixture_sha256": _sha256(fixture_path),
            },
        }],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    registry_hash = _sha256(registry_path)

    rb_root = tmp_path / "rb-python"
    (rb_root / "resource_balancer").mkdir(parents=True)
    (rb_root / "resource_balancer" / "admission.py").write_text("# test package marker", encoding="utf-8")
    state_dir = tmp_path / "rb-state"
    state_dir.mkdir()
    queue_path = state_dir / "queue.sqlite3"
    conn = sqlite3.connect(queue_path)
    conn.execute("CREATE TABLE controller (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, class TEXT NOT NULL, state TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO controller VALUES (1, ?)",
        (json.dumps({
            "mode": "manage", "blocked": False, "updated": time.time(),
            "resource_gates": {"gpu": {"blocked": False}},
        }),),
    )
    conn.commit()
    conn.close()

    lease_id = "lease-test-1"

    class FakeLease:
        def __enter__(self):
            with sqlite3.connect(queue_path) as queue:
                queue.execute("INSERT INTO jobs VALUES (?, 'gpu', 'running')", (lease_id,))
            return lease_id

        def __exit__(self, *_exc):
            with sqlite3.connect(queue_path) as queue:
                queue.execute("UPDATE jobs SET state='succeeded' WHERE id=?", (lease_id,))

    rb_package = ModuleType("resource_balancer")
    rb_package.__path__ = [str(rb_root / "resource_balancer")]
    rb_admission = ModuleType("resource_balancer.admission")
    rb_admission.admission = lambda *args, **kwargs: FakeLease()
    rb_integration = ModuleType("resource_balancer.integration")
    rb_integration.process_identity = lambda _pid: "test-start"
    monkeypatch.setitem(sys.modules, "resource_balancer", rb_package)
    monkeypatch.setitem(sys.modules, "resource_balancer.admission", rb_admission)
    monkeypatch.setitem(sys.modules, "resource_balancer.integration", rb_integration)

    (home / "config.yaml").write_text(json.dumps({
        "plugins": {
            "enabled": ["local_gpu_admission"],
            "entries": {"local_gpu_admission": {"settings": {
                "resource_balancer_python_path": str(rb_root),
                "resource_balancer_state_dir": str(state_dir),
                "capability_registry_path": str(registry_path),
                "capability_registry_sha256": registry_hash,
            }}},
        },
    }), encoding="utf-8")
    manager = PluginManager(scope_key=str(home))
    manager.discover_and_load()
    controllers = manager._middleware.get("llm_admission", [])
    assert len(controllers) == 1
    controller = controllers[0]
    assert type(controller).__name__ == "ResourceBalancerController"
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(plugins, "has_middleware", lambda kind: bool(manager._middleware.get(kind)))
    controller._live_model_evidence = lambda _scope: {
        "digest": "d" * 64,
        "quantization": "Q4_K_M",
        "runtime_version": "0.34.2",
    }
    contract_hash = task_run_contract_sha256(task)
    scope = {
        "task_id": task_id,
        "kanban_task_id": task_id,
        "task_run_id": str(task.current_run_id),
        "kanban_board": "default",
        "kanban_db": str(db_path),
        "kanban_workspace": str(workspace),
        "kanban_claim_lock": task.claim_lock,
        "capability_id": "fixed-json.v1",
        "task_contract_sha256": contract_hash,
        "api_request_id": "request-1",
        "context_length": "4096",
        "ollama_num_ctx": "4096",
        "session_id": "session-1",
        "model": "fixture-model",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1/",
        "streaming": "false",
        "has_tools": "false",
        "api_call_count": "1",
        "existing_response_format": "false",
    }

    wrong_provider = controller(action="acquire", **{**scope, "provider": "not-ollama"})
    assert wrong_provider["decision"] == "block", wrong_provider
    assert wrong_provider["policy_id"] == "acquire_refused"
    assert "provider/context differs" in wrong_provider["reason"]

    wrong_task_override = SimpleNamespace(
        id=task.id, model_override=task.model_override, provider_override="not-ollama",
    )
    try:
        controller._load_oracle(wrong_task_override, scope)
    except ValueError as exc:
        assert "task provider override differs" in str(exc)
    else:
        raise AssertionError("non-exact task provider override must be refused")

    acquired = controller(action="acquire", **scope)
    assert acquired["decision"] == "allow", acquired
    assert acquired["lease_ids"] == [lease_id]
    assert acquired["response_format"] == {
        "type": "json_schema",
        "json_schema": {"schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        }},
    }

    validated = controller(action="validate", lease_ids=(lease_id,), **scope)
    assert validated["decision"] == "allow", validated

    retry = controller(
        action="validate", lease_ids=(lease_id,),
        **{**scope, "api_call_count": 2, "api_request_id": "retry-request"},
    )
    assert retry["decision"] == "block"
    assert retry["policy_id"] == "capability_request_limit"

    changed_context = controller(
        action="validate", lease_ids=(lease_id,), **{**scope, "ollama_num_ctx": "8192"}
    )
    assert changed_context["decision"] == "block", changed_context
    assert changed_context["policy_id"] == "lease_scope_changed"

    controller._live_model_evidence = lambda _scope: {
        "digest": "d" * 64,
        "quantization": "Q8_0",
        "runtime_version": "0.34.2",
    }
    changed_runtime = controller(action="validate", lease_ids=(lease_id,), **scope)
    assert changed_runtime["decision"] == "block", changed_runtime
    assert changed_runtime["policy_id"] == "lease_validation_refused"
    controller._live_model_evidence = lambda _scope: {
        "digest": "d" * 64,
        "quantization": "Q4_K_M",
        "runtime_version": "0.34.2",
    }

    # Exercise the loaded controller again at the real sync provider seam. The fake provider
    # callback records the exact request but performs no network/model I/O.
    import agent.chat_completion_helpers as helpers

    for key, env_key in {
        "task_id": "HERMES_KANBAN_TASK",
        "task_run_id": "HERMES_KANBAN_RUN_ID",
        "kanban_board": "HERMES_KANBAN_BOARD",
        "kanban_db": "HERMES_KANBAN_DB",
        "kanban_workspace": "HERMES_KANBAN_WORKSPACE",
        "kanban_claim_lock": "HERMES_KANBAN_CLAIM_LOCK",
    }.items():
        monkeypatch.setenv(env_key, str(scope[key]))
    from agent.delegation_context import set_owned_local_gpu_task_context
    set_owned_local_gpu_task_context({
        "capability_id": scope["capability_id"],
        "task_contract_sha256": scope["task_contract_sha256"],
    })
    fake_provider_requests = []
    fake_provider_response = expected_bytes.decode("utf-8")
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda _agent: True)
    monkeypatch.setattr(
        helpers, "direct_api_call",
        lambda _agent, request: (fake_provider_requests.append(request.copy()), fake_provider_response)[1],
    )
    agent = SimpleNamespace(
        provider="ollama", base_url=scope["base_url"], model="fixture-model",
        session_id="session-1", api_mode="chat_completions", _api_call_count=1,
        context_compressor=SimpleNamespace(context_length=4096), _ollama_num_ctx=4096,
        _local_gpu_admission_lease_ids=(lease_id,),
    )
    from agent.chat_completion_helpers import interruptible_api_call
    response = interruptible_api_call(agent, {
        "model": "fixture-model", "response_format": acquired["response_format"],
    })
    assert response == fake_provider_response
    assert len(fake_provider_requests) == 1
    assert fake_provider_requests[0]["response_format"] == acquired["response_format"]
    artifact_path.write_text(response, encoding="utf-8")

    completion = {
        key: scope[key]
        for key in (
            "task_id", "kanban_task_id", "task_run_id", "kanban_claim_lock", "kanban_board",
            "kanban_db", "kanban_workspace", "capability_id", "task_contract_sha256",
        )
    }
    completion.update({
        "workspace_path": str(workspace),
        "assignee": "test-profile",
        "model_override": "fixture-model",
        "provider_override": "ollama",
        "completion_contract": "local-only",
        "artifacts": [str(artifact_path)],
    })
    verified = controller(action="verify_completion", **completion)
    assert verified["decision"] == "verified", verified
    assert verified["artifact_sha256"] == _sha256(artifact_path)
    assert verified["verifier_sha256"] == _sha256(verifier_path)

    released = controller(action="release", lease_ids=(lease_id,), **scope)
    assert released["decision"] == "released", released
    with sqlite3.connect(queue_path) as queue:
        assert queue.execute("SELECT state FROM jobs WHERE id=?", (lease_id,)).fetchone()[0] == "succeeded"
