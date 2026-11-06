"""Fail-closed task/run and lease checks for the local GPU plugin controller."""

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import PluginManager
from hermes_cli.kanban_local_model_scope import (
    capability_id_from_body,
    task_run_contract_sha256,
)
from plugins import local_gpu_admission
from plugins.local_gpu_admission import ResourceBalancerController


def test_capability_marker_must_be_unique_and_nonempty():
    assert capability_id_from_body("Local-GPU-Capability: gemma.fixed-json.v1") == "gemma.fixed-json.v1"
    assert capability_id_from_body("Local-GPU-Capability: one\nLocal-GPU-Capability: two") == ""
    assert capability_id_from_body("Local-GPU-Capability: two words") == ""


def test_task_run_contract_digest_changes_with_claim_identity():
    task = {
        "id": "task-1", "title": "Extract fields", "body": "fixed schema",
        "assignee": "profile", "workspace_kind": "scratch", "workspace_path": "/tmp/task",
        "completion_contract": "local-only", "skills": ["local-gpu-model-task-gate"],
        "max_runtime_seconds": 300, "max_retries": 1, "current_run_id": 7,
        "claim_lock": "claim-a",
    }

    first = task_run_contract_sha256(task)
    changed_claim = {**task, "claim_lock": "claim-b"}

    assert first != task_run_contract_sha256(changed_claim)


def test_stale_task_run_lease_is_refused():
    controller = ResourceBalancerController(profile="profile", config={})

    decision = controller(
        action="validate",
        lease_ids=("expired-lease",),
        kanban_task_id="task-1",
        task_run_id="7",
        kanban_claim_lock="claim-a",
    )

    assert decision["decision"] == "block"
    assert decision["policy_id"] == "lease_missing"


def test_fresh_resource_balancer_global_block_refuses_admission(tmp_path):
    state_dir = tmp_path / "rb-state"
    state_dir.mkdir()
    with sqlite3.connect(state_dir / "queue.sqlite3") as conn:
        conn.execute("CREATE TABLE controller (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO controller VALUES (1, ?)",
            (json.dumps({
                "mode": "manage", "blocked": True, "updated": time.time(),
                "resource_gates": {"gpu": {"blocked": False}},
            }),),
        )

    controller = ResourceBalancerController(profile="profile", config={})

    with pytest.raises(ValueError, match="stale, blocked, or GPU-gated"):
        controller._resource_balancer_status(state_dir)


def test_pre_spawn_reservation_must_match_exact_task_run_contract(tmp_path):
    state_dir = tmp_path / "rb-state"
    state_dir.mkdir()
    scope = {
        "kanban_task_id": "task-1", "task_run_id": "7", "kanban_claim_lock": "claim-a",
        "task_contract_sha256": "a" * 64, "capability_id": "capability-1",
    }
    with sqlite3.connect(state_dir / "queue.sqlite3") as conn:
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, class TEXT, state TEXT, context TEXT)")
        conn.execute("CREATE TABLE hermes_local_gpu_reservations "
                     "(job_id TEXT PRIMARY KEY, context TEXT NOT NULL, created_at REAL NOT NULL)")
        conn.execute("INSERT INTO jobs VALUES (?,?,?,NULL)", ("gpu-job-1", "gpu", "running"))
        conn.execute(
            "INSERT INTO hermes_local_gpu_reservations VALUES (?,?,?)",
            ("gpu-job-1", json.dumps({
                "task_id": "task-1", "run_id": "7", "claim_lock": "claim-a",
                "task_contract_sha256": "a" * 64, "capability_id": "capability-1",
            }), time.time()),
        )

    ResourceBalancerController._validate_external_reservation(state_dir, "gpu-job-1", scope)
    with pytest.raises(ValueError, match="does not match"):
        ResourceBalancerController._validate_external_reservation(
            state_dir, "gpu-job-1", {**scope, "task_run_id": "8"},
        )


def test_controller_rejects_task_run_id_that_differs_from_live_database(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.init_db()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="fixed JSON fixture",
            body="Local-GPU-Capability: capability-1",
            assignee="profile",
            workspace_kind="scratch",
            workspace_path=str(tmp_path / "workspace"),
            skills=["local-gpu-model-task-gate"],
            max_runtime_seconds=300,
            max_retries=1,
            completion_contract="local-only",
        )
        claimed = kb.claim_task(conn, task_id, claimer="claim-1")
        assert claimed is not None

    controller = ResourceBalancerController(profile="profile", config={})
    context = {
        "task_id": task_id,
        "kanban_task_id": task_id,
        "task_run_id": str(claimed.current_run_id + 1),
        "kanban_db": str(db_path),
        "kanban_claim_lock": claimed.claim_lock,
        "kanban_workspace": claimed.workspace_path,
        "capability_id": "capability-1",
        "task_contract_sha256": task_run_contract_sha256(claimed),
    }

    with pytest.raises(ValueError, match="task/run/claim/capability contract"):
        controller._read_task(context)


def test_controller_refuses_unmanaged_scratch_workspace_before_admission(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    workspace = tmp_path / 'unmanaged-workspace'
    workspace.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACES_ROOT', str(tmp_path / 'managed-workspaces'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    db_path = kb.init_db()
    monkeypatch.setenv('HERMES_KANBAN_DB', str(db_path))
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title='fixed JSON', body='Local-GPU-Capability: capability-1',
            assignee='profile', workspace_kind='scratch', workspace_path=str(workspace),
            skills=['local-gpu-model-task-gate'], max_runtime_seconds=300, max_retries=1,
            model_override='gemma4:12b-64k', provider_override='ollama',
            completion_contract='local-only',
        )
        claimed = kb.claim_task(conn, task_id, claimer='claim-1')
        assert claimed is not None
        conn.execute('UPDATE task_runs SET worker_pid=? WHERE id=?',
                     (__import__('os').getpid(), claimed.current_run_id))
        conn.commit()
    context = {
        'kanban_db': str(db_path), 'kanban_task_id': task_id,
        'task_run_id': str(claimed.current_run_id), 'kanban_claim_lock': claimed.claim_lock,
        'kanban_workspace': claimed.workspace_path, 'capability_id': 'capability-1',
        'task_contract_sha256': task_run_contract_sha256(claimed),
    }
    controller = ResourceBalancerController(profile='profile', config={})
    with pytest.raises(ValueError, match='task workspace is not a managed scratch directory'):
        controller._read_task(context)


def test_admission_refusal_includes_bounded_actionable_reason():
    controller = ResourceBalancerController(profile="profile", config={})

    def reject_task_read(_context):
        raise ValueError("Resource Balancer refused admission: pressure/recovery")

    controller._read_task = reject_task_read
    scope = {
        "task_id": "task-1", "kanban_task_id": "task-1", "task_run_id": "7",
        "kanban_board": "default", "kanban_db": "/tmp/kanban.sqlite3",
        "kanban_workspace": "/tmp/workspace", "kanban_claim_lock": "claim-1",
        "capability_id": "cap-1", "task_contract_sha256": "a" * 64,
        "api_request_id": "request-1", "context_length": "4096",
        "ollama_num_ctx": "4096", "session_id": "session-1", "model": "fixture-model",
        "provider": "ollama", "base_url": "http://127.0.0.1:11434/v1/",
        "streaming": False, "has_tools": False, "api_call_count": 1,
        "existing_response_format": False,
    }

    decision = controller(action="acquire", **scope)

    assert decision["decision"] == "block"
    assert "pressure/recovery" in decision["reason"]
    assert "ValueError" not in decision["reason"]


def test_completion_without_active_task_run_lease_is_refused():
    controller = ResourceBalancerController(profile="profile", config={})

    decision = controller(
        action="verify_completion",
        kanban_task_id="task-1",
        task_run_id="7",
        kanban_claim_lock="claim-a",
        artifacts=["result.json"],
    )

    assert decision["decision"] == "block"
    assert decision["policy_id"] == "completion_lease_missing"


@pytest.mark.parametrize(
    "allow, status, digest, mapping_review_status, expected_error",
    [
        (True, "narrowly_supported", None,
         "approved", "historical capability evidence has no pinned model digest"),
        (True, "narrowly_supported", "d" * 64,
         "approved", "capability evidence is missing exact quantization or runtime version"),
        (False, "historical_success_pending_exact_digest_trials", "d" * 64,
         "approved", "model/capability pair is outside the proven envelope"),
        (True, "narrowly_supported", "d" * 64,
         "pending", "trial task-class mapping is missing or not independently reviewed"),
    ],
)
def test_missing_trial_evidence_is_refused_before_live_model_lookup(
    tmp_path, monkeypatch, allow, status, digest, mapping_review_status, expected_error,
):
    rb_root = tmp_path / "tools"
    (rb_root / "resource_balancer").mkdir(parents=True)
    (rb_root / "resource_balancer" / "admission.py").write_text("# fixture", encoding="utf-8")
    state_dir = tmp_path / "rb-state"
    state_dir.mkdir()
    (state_dir / "queue.sqlite3").touch()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "task_run_binding": {"schema": "hermes.local-model-task-contract.v1"},
        "capabilities": [{
            "id": "cap-1", "allow": allow, "status": status,
            "model_tag": "fixture-model", "model_digest": digest,
            "request_schema_policy": "json_schema_from_fixture_types_v1",
            "trial_binding": {
                "admission_capability_id": "cap-1",
                "trial_task_class": "cap-1.trial.v1",
                "mapping_review": {
                    "status": mapping_review_status, "reviewer": "test-reviewer",
                    "reviewed_at_utc": "2026-10-28T00:00:00Z",
                },
            },
            "profile_route_evidence": {"task_class": "cap-1.trial.v1"},
            "observed_runtime": {"context_length": 4096, "backend": "ollama"},
        }],
    }), encoding="utf-8")
    controller = ResourceBalancerController(
        profile="profile",
        config={
            "resource_balancer_python_path": str(rb_root),
            "resource_balancer_state_dir": str(state_dir),
            "capability_registry_path": str(registry_path),
            "capability_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(
        controller, "_live_model_evidence",
        lambda _scope: pytest.fail("live model lookup must not replace missing trial evidence"),
    )

    with pytest.raises(ValueError, match=expected_error):
        controller._load_oracle(
            SimpleNamespace(id="task-1", model_override="", provider_override=""),
            {
                "capability_id": "cap-1", "model": "fixture-model", "streaming": "false",
                "has_tools": "false", "api_call_count": "1", "existing_response_format": "false",
            },
        )


def test_pinned_verifier_nonzero_exit_refuses_artifact(tmp_path, monkeypatch):
    verifier = tmp_path / "verify.py"
    artifact = tmp_path / "result.json"
    verifier.write_text("# pinned verifier fixture", encoding="utf-8")
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        local_gpu_admission.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(ValueError, match="independent verifier rejected"):
        local_gpu_admission._run_pinned_verifier(verifier, artifact)


def test_live_model_evidence_reads_digest_quantization_and_runtime_version(monkeypatch):
    import json

    responses = iter([
        SimpleNamespace(read=lambda _limit: json.dumps({
            "models": [{
                "name": "fixture-model", "digest": "d" * 64,
                "details": {"quantization_level": "Q4_K_M"},
            }],
        }).encode()),
        SimpleNamespace(read=lambda _limit: b'{"version":"0.34.2"}'),
    ])
    urls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit):
            return self.payload.read(limit)

    def fake_urlopen(url, timeout):
        urls.append((url, timeout))
        return Response(next(responses))

    monkeypatch.setattr(local_gpu_admission, "urlopen", fake_urlopen)
    evidence = ResourceBalancerController._live_model_evidence({
        "model": "fixture-model", "base_url": "http://127.0.0.1:11434/v1/",
    })

    assert evidence == {
        "digest": "d" * 64,
        "quantization": "Q4_K_M",
        "runtime_version": "0.34.2",
    }
    assert [url for url, _timeout in urls] == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/api/version",
    ]


def test_bundled_plugin_registers_admission_in_an_isolated_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [local_gpu_admission]\n"
        "  entries:\n"
        "    local_gpu_admission:\n"
        "      settings:\n"
        f"        resource_balancer_python_path: {tmp_path / 'tools'}\n"
        f"        resource_balancer_state_dir: {tmp_path / 'rb-state'}\n"
        f"        capability_registry_path: {tmp_path / 'registry.json'}\n"
        f"        capability_registry_sha256: {'a' * 64}\n",
        encoding="utf-8",
    )

    manager = PluginManager()
    manager.discover_and_load()

    callbacks = manager._middleware["llm_admission"]
    assert len(callbacks) == 1
    assert type(callbacks[0]).__name__ == "ResourceBalancerController"
    assert type(callbacks[0]).__module__.startswith("hermes_plugins.local_gpu_admission")
    assert manager.has_middleware("llm_admission")

    import agent.chat_completion_helpers as helpers
    from hermes_cli.middleware import LocalModelAdmissionBlockedError

    for key, value in {
        "HERMES_KANBAN_TASK": "task-1", "HERMES_KANBAN_RUN_ID": "1",
        "HERMES_KANBAN_BOARD": "default", "HERMES_KANBAN_DB": str(tmp_path / "missing.db"),
        "HERMES_KANBAN_WORKSPACE": str(tmp_path), "HERMES_KANBAN_CLAIM_LOCK": "claim-1",
    }.items():
        monkeypatch.setenv(key, value)
    from agent.delegation_context import set_owned_local_gpu_task_context
    set_owned_local_gpu_task_context({"capability_id": "fixed-json.v1", "task_contract_sha256": "a" * 64})
    provider_calls = []
    monkeypatch.setattr(helpers, "direct_api_call", lambda *_args, **_kwargs: provider_calls.append("called"))
    agent = SimpleNamespace(
        provider="ollama", base_url="http://127.0.0.1:11434/v1/", model="fixture-model",
        session_id="session-1", api_mode="chat_completions", _api_call_count=1,
        context_compressor=SimpleNamespace(context_length=4096), _ollama_num_ctx=4096,
    )

    with pytest.raises(LocalModelAdmissionBlockedError, match="controller_block"):
        helpers.interruptible_api_call(agent, {"model": "fixture-model"})
    assert provider_calls == []
