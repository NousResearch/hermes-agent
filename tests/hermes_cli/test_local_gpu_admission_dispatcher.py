from contextlib import nullcontext
import hashlib
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db_dispatch as dispatcher


def test_capability_worker_is_wrapped_with_task_bound_rb_exec(monkeypatch, tmp_path):
    rb_root = tmp_path / "tools"
    (rb_root / "resource_balancer").mkdir(parents=True)
    (rb_root / "resource_balancer" / "admission.py").write_text("# fixture", encoding="utf-8")
    state_dir = tmp_path / "rb-state"
    state_dir.mkdir()
    (state_dir / "queue.sqlite3").touch()
    board_db = tmp_path / "kanban.db"
    board_db.touch()
    profile_home = tmp_path / "profile"
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"plugins": {"entries": {"local_gpu_admission": {
            "enabled": True,
            "settings": {
                "resource_balancer_python_path": str(rb_root),
                "resource_balancer_state_dir": str(state_dir),
            },
        }}}},
    )
    monkeypatch.setattr(dispatcher._kb, "kanban_db_path", lambda board=None: board_db)
    monkeypatch.setattr(
        "hermes_cli.kanban_local_model_scope.task_run_contract_sha256",
        lambda _task: "a" * 64,
    )
    task = SimpleNamespace(
        id="task-1", current_run_id=7, claim_lock="claim-a",
        body="Local-GPU-Capability: capability-1",
    )
    env = {"PYTHONPATH": "/hermes/source"}

    wrapped = dispatcher._wrap_capability_worker_with_gpu_reservation(
        task, ["hermes", "worker"], profile_home=str(profile_home), env=env, board="default",
    )

    assert wrapped[:3] == [
        dispatcher.sys.executable, "-m", "plugins.local_gpu_admission.rb_exec_entrypoint",
    ]
    assert wrapped[-3:] == ["--", "hermes", "worker"]
    assert env["PYTHONPATH"] == f"/hermes/source{dispatcher.os.pathsep}{rb_root}"
    context = json.loads(wrapped[wrapped.index("--context-json") + 1])
    assert context == {
        "task_id": "task-1", "run_id": "7", "claim_lock": "claim-a",
        "capability_id": "capability-1", "task_contract_sha256": "a" * 64,
    }


def test_capability_worker_refuses_spawn_without_profile_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"plugins": {}})
    task = SimpleNamespace(
        id="task-1", current_run_id=7, claim_lock="claim-a",
        body="Local-GPU-Capability: capability-1",
    )
    with pytest.raises(RuntimeError, match="profile gate is not enabled"):
        dispatcher._wrap_capability_worker_with_gpu_reservation(
            task, ["hermes", "worker"], profile_home=str(tmp_path), env={}, board=None,
        )


def test_dispatch_preflight_refuses_disabled_capability_before_claim(monkeypatch, tmp_path):
    profile_home = tmp_path / "gemma4-12b-64k-canary"
    profile_home.mkdir()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"capabilities": [{
        "id": "capability-1", "allow": False,
    }]}), encoding="utf-8")
    rb_state = tmp_path / "rb-state"
    rb_state.mkdir()
    config = {
        "model": {"provider": "ollama", "default": "gemma4:12b-64k", "ollama_num_ctx": 64000},
        "providers": {"ollama": {"base_url": "http://127.0.0.1:11434/v1"}},
        "plugins": {"entries": {"local_gpu_admission": {
            "enabled": True,
            "settings": {
                "capability_registry_path": str(registry_path),
                "capability_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                "resource_balancer_state_dir": str(rb_state),
            },
        }}},
    }
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _name: str(profile_home))
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    row = {
        "id": "task-1", "body": "Local-GPU-Capability: capability-1",
        "model_override": "gemma4:12b-64k", "provider_override": "ollama",
    }

    reason = dispatcher._local_gpu_dispatch_preflight(row, "gemma4-12b-64k-canary")

    assert reason == "pinned capability is not authorized (allow is false)"


def test_dispatch_preflight_checks_exact_runtime_before_allowing_claim(monkeypatch, tmp_path):
    profile_home = tmp_path / "profile-canary"
    profile_home.mkdir()
    registry_path = tmp_path / "registry.json"
    verifier_path = tmp_path / "verify.py"
    fixture_path = tmp_path / "expected.json"
    verifier_path.write_text("pass\n", encoding="utf-8")
    fixture_path.write_text("{}\n", encoding="utf-8")
    base_url = "http://127.0.0.1:11434/v1"
    model_cfg = {
        "provider": "ollama", "default": "model:tag", "ollama_num_ctx": 64000,
        "context_length": 64000,
    }
    provider_cfg = {"base_url": base_url}
    route = {
        "profile": "profile-canary", "profile_default_model": model_cfg["default"],
        "provider": model_cfg["provider"], "ollama_num_ctx": model_cfg["ollama_num_ctx"],
        "api_base": provider_cfg["base_url"],
    }
    route_hash = hashlib.sha256(json.dumps(route, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    verifier_hash = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    registry = {"capabilities": [{
        "id": "capability-1", "allow": True, "status": "narrowly_supported_test",
        "request_schema_policy": "json_schema_from_fixture_types_v1",
        "model_tag": "model:tag", "model_digest": "d" * 64,
        "observed_runtime": {
            "profile": "profile-canary", "backend": "ollama", "base_url": base_url,
            "context_length": 64000, "ollama_num_ctx": 64000,
            "quantization": "Q4_K_M", "runtime_version": "0.34.2",
            "profile_route_sha256": route_hash,
        },
        "profile_route_evidence": {"task_class": "fixed-json.v1"},
        "trial_binding": {
            "admission_capability_id": "capability-1", "trial_task_class": "fixed-json.v1",
            "mapping_review": {"status": "approved", "reviewer": "reviewer",
                                "reviewed_at_utc": "2026-10-28T00:00:00Z"},
        },
        "prepared_external_verifier": {
            "task_id": "task-1", "state": "verified", "verifier_path": "verify.py",
            "verifier_sha256": verifier_hash, "expected_fixture_path": "expected.json",
            "expected_fixture_sha256": fixture_hash,
        },
    }]}
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    rb_state = tmp_path / "rb-state"
    rb_state.mkdir()
    with sqlite3.connect(rb_state / "queue.sqlite3") as conn:
        conn.execute("CREATE TABLE controller (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
        conn.execute("CREATE TABLE jobs (id TEXT, class TEXT, state TEXT)")
        conn.execute("INSERT INTO controller VALUES (1,?)", (json.dumps({
            "mode": "manage", "blocked": False, "updated": time.time(),
            "resource_gates": {"gpu": {"blocked": False}},
        }),))
    config = {
        "model": model_cfg, "providers": {"ollama": provider_cfg},
        "plugins": {"entries": {"local_gpu_admission": {
            "enabled": True,
            "settings": {
                "capability_registry_path": str(registry_path),
                "capability_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                "resource_balancer_state_dir": str(rb_state),
            },
        }}},
    }
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _name: str(profile_home))
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)

    class Response:
        def __init__(self, value):
            self.value = json.dumps(value).encode()
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self, _limit): return self.value

    answers = iter([
        {"models": [{"name": "model:tag", "digest": "d" * 64,
                     "details": {"quantization_level": "Q4_K_M"}}]},
        {"version": "0.34.2"},
    ])
    monkeypatch.setattr(dispatcher, "urlopen", lambda _url, timeout: Response(next(answers)))
    row = {"id": "task-1", "body": "Local-GPU-Capability: capability-1",
           "model_override": "model:tag", "provider_override": "ollama"}

    assert dispatcher._local_gpu_dispatch_preflight(row, "profile-canary") is None


def test_dispatch_preflight_ignores_explicit_remote_task(monkeypatch, tmp_path):
    profile_home = tmp_path / "remote-profile"
    profile_home.mkdir()
    config = {
        "model": {"provider": "openai", "default": "gpt-5.6-sol"},
        "providers": {"openai": {"base_url": "https://api.openai.com/v1"}},
    }
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _name: str(profile_home))
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    row = {"id": "task-remote", "body": "ordinary remote task", "model_override": None,
           "provider_override": None}
    assert dispatcher._local_gpu_dispatch_preflight(row, "remote-profile") is None


def test_dispatch_preflight_refuses_custom_fqdn_that_resolves_private(monkeypatch, tmp_path):
    from hermes_cli import local_model_routes

    profile_home = tmp_path / "custom-private-fqdn"
    profile_home.mkdir()
    config = {
        "model": {"provider": "custom", "default": "local-model"},
        "providers": {"custom": {"base_url": "https://gpu.region.example/v1"}},
    }
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _name: str(profile_home))
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr(local_model_routes, "_is_known_official_provider_host", lambda *_args: False)
    monkeypatch.setattr(local_model_routes, "_host_resolves_to_local_address", lambda _host: True)
    row = {"id": "task-private-fqdn", "body": "ordinary task", "model_override": None,
           "provider_override": None}

    reason = dispatcher._local_gpu_dispatch_preflight(row, "custom-private-fqdn")

    assert reason == "effective profile route is local but task has no Local-GPU-Capability marker"


def test_dispatch_refusal_persists_reason_without_claim_or_spawn(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "local-profile"
    profile_home.mkdir(parents=True)
    profile_home.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb.init_db()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"capabilities": [{"id": "cap-1", "allow": False}]}),
                             encoding="utf-8")
    rb_state = tmp_path / "rb-state"
    rb_state.mkdir()
    config = {
        "model": {"provider": "ollama", "default": "model:tag", "ollama_num_ctx": 64000},
        "providers": {"ollama": {"base_url": "http://127.0.0.1:11434/v1"}},
        "plugins": {"entries": {"local_gpu_admission": {
            "enabled": True,
            "settings": {
                "capability_registry_path": str(registry_path),
                "capability_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                "resource_balancer_state_dir": str(rb_state),
            },
        }}},
    }
    monkeypatch.setattr(dispatcher, "_worker_profile_scope", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr(dispatcher, "_profile_exists_fn", lambda: lambda _name: True)
    spawn_calls = []
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="must remain unclaimed", body="Local-GPU-Capability: cap-1",
            assignee="local-profile",
        )
        row = conn.execute(
            "SELECT id,assignee,body,model_override,provider_override FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        result = dispatcher.DispatchResult()
        dispatched = dispatcher._dispatch_lane_task(
            conn, row, "local-profile", result, lane="ready", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=2,
            spawn_fn=lambda *args, **kwargs: spawn_calls.append((args, kwargs)),
            per_profile_cap=None, per_profile_running={},
        )
        task = kb.get_task(conn, task_id)
        events = conn.execute(
            "SELECT kind,payload FROM task_events WHERE task_id=? ORDER BY id", (task_id,),
        ).fetchall()

    assert not dispatched
    assert not spawn_calls
    assert task.status == "ready"
    assert task.current_run_id is None
    assert result.skipped_local_admission == [
        (task_id, "pinned capability is not authorized (allow is false)"),
    ]
    assert events[-1]["kind"] == "local_gpu_admission_refused"
    assert json.loads(events[-1]["payload"])["reason"] == "pinned capability is not authorized (allow is false)"
