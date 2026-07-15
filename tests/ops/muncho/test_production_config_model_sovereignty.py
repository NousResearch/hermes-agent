from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).parents[3]
MODULE_PATH = (
    ROOT
    / "ops"
    / "muncho"
    / "runtime"
    / "production_config_model_sovereignty.py"
)


def _load():
    name = "production_config_model_sovereignty_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> bytes:
    return b"""model:
  default: gpt-5.6-sol
  provider: openai-codex
  base_url: https://chatgpt.com/backend-api/codex
agent:
  max_turns: 90
  tool_use_enforcement: auto
  task_completion_guidance: true
  parallel_tool_call_guidance: true
  environment_hint: 'Use OpenAI Codex OAuth/openai-codex gpt-5.6-sol; do not route GPT-5.5 through OPENAI_API_KEY. Keep working.'
  reasoning_effort: high
compression:
  enabled: true
  threshold: 0.5
  target_ratio: 0.2
  abort_on_summary_failure: false
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    api_key: ''
    timeout: 120
curator:
  enabled: false
  consolidate: false
  prune_builtins: false
tool_loop_guardrails:
  warnings_enabled: true
  hard_stop_enabled: false
kanban:
  dispatch_in_gateway: true
  auto_decompose: false
  failure_limit: 2
hooks_auto_accept: false
"""


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load()
    path = tmp_path / "config.yaml"
    path.write_bytes(_config())
    path.chmod(0o600)
    metadata = path.stat()
    monkeypatch.setattr(module, "CONFIG_PATH", path)
    monkeypatch.setattr(module, "EXPECTED_UID", metadata.st_uid)
    monkeypatch.setattr(module, "EXPECTED_GID", metadata.st_gid)
    monkeypatch.setattr(
        module,
        "_require_gateway_stopped",
        lambda: {
            "unit": module.GATEWAY_UNIT,
            "load_state": "loaded",
            "active_state": "inactive",
            "main_pid": 0,
        },
    )
    return module, path


def test_plan_is_read_only_and_binds_exact_target(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before = path.read_bytes()
    plan = module.build_plan(
        expected_before_sha256=hashlib.sha256(before).hexdigest()
    )

    assert path.read_bytes() == before
    assert plan["before_sha256"] == hashlib.sha256(before).hexdigest()
    assert plan["after_sha256"] != plan["before_sha256"]
    assert plan["service_restart_performed"] is False
    assert plan["mutations"] == list(module.MUTATIONS)


def test_apply_requires_plan_and_writes_exact_backup(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before = path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)

    receipt = module.apply_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    retry_receipt = module.apply_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )

    assert receipt["ok"] is True
    assert retry_receipt == receipt
    assert receipt["after_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert Path(receipt["backup_path"]).read_bytes() == before
    effective = yaml.safe_load(path.read_text())
    assert effective["agent"]["adaptive_reasoning"] == {
        "enabled": True,
        "max_effort": "max",
    }
    assert effective["agent"]["background_review_enabled"] is False
    assert effective["agent"]["tool_use_enforcement"] is True
    assert effective["agent"]["verify_on_stop"] is False
    assert effective["agent"]["verification_ledger_enabled"] is False
    assert effective["agent"]["task_completion_guidance"] is True
    assert effective["agent"]["parallel_tool_call_guidance"] is True
    assert "gpt-5.5" not in effective["agent"]["environment_hint"].casefold()
    assert effective["kanban"]["auxiliary_planning_enabled"] is False
    assert effective["kanban"]["auto_decompose"] is False
    assert effective["kanban"]["dispatch_in_gateway"] is False
    assert effective["compression"]["abort_on_summary_failure"] is True
    assert effective["auxiliary"]["compression"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "base_url": "",
        "api_key": "",
        "timeout": 120,
    }
    assert effective["plugins"] == {"enabled": [], "disabled": []}
    assert effective["hooks"] == {}
    assert effective["hooks_auto_accept"] is False
    assert effective["curator"]["enabled"] is False
    assert effective["curator"]["consolidate"] is False
    assert effective["curator"]["prune_builtins"] is False
    assert effective["tool_loop_guardrails"]["warnings_enabled"] is True
    assert effective["tool_loop_guardrails"]["hard_stop_enabled"] is False
    assert effective["tools"]["tool_search"] == {"enabled": "off"}
    assert effective["approvals"]["plan_owner_user_ids"] == [
        "1279454038731264061"
    ]
    assert effective["approvals"]["gateway_authorized_user_ids"] == [
        "1279454038731264061"
    ]
    assert effective["approvals"]["gateway_authorized_user_names"] == []
    assert effective["approvals"]["gateway_authorized_labels"] == ["Емо"]
    assert effective["approvals"]["gateway_owner_escalation"] == {
        "enabled": True,
        "owner_user_id": "1279454038731264061",
        "owner_guild_id": "1282725267068157972",
        "owner_channel_id": "1504852355588423801",
        "owner_target_type": "guild_channel",
    }
    assert effective["goals"] == {"max_turns": 0}
    assert effective["command_allowlist"] == []

    rollback = module.rollback_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    retry_rollback = module.rollback_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    assert rollback["ok"] is True
    assert retry_rollback == rollback
    assert rollback["to_sha256"] == before_sha
    assert path.read_bytes() == before
    assert Path(rollback["recovery_backup_path"]).read_bytes() != before


def test_wrong_before_or_plan_digest_fails_closed(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)

    with pytest.raises(module.ConfigGateError, match="before_sha256_mismatch"):
        module.build_plan(expected_before_sha256="0" * 64)
    with pytest.raises(module.ConfigGateError, match="approved_plan_sha256_mismatch"):
        module.apply_plan(
            expected_before_sha256=before_sha,
            approved_plan_sha256="1" * 64,
        )
    assert not Path(plan["backup_path"]).exists()


def test_rollback_requires_exact_applied_state(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)
    module.apply_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    path.write_bytes(path.read_bytes() + b"# drift\n")

    with pytest.raises(module.ConfigGateError, match="rollback_source_mismatch"):
        module.rollback_plan(
            expected_before_sha256=before_sha,
            approved_plan_sha256=plan["plan_sha256"],
        )


def test_plan_rejects_partial_or_already_applied_source(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    raw = path.read_bytes()
    path.write_bytes(
        raw.replace(
            b"  reasoning_effort: high\n",
            b"  reasoning_effort: high\n"
            b"  adaptive_reasoning:\n"
            b"    enabled: true\n"
            b"    max_effort: max\n",
        )
    )

    with pytest.raises(module.ConfigGateError, match="adaptive_source_drifted"):
        module.build_plan(
            expected_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_apply_rechecks_source_after_backup_publish(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before = path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)
    publish = module._publish_backup

    def publish_then_drift(backup, raw):
        publish(backup, raw)
        path.write_bytes(raw + b"# concurrent operator change\n")

    monkeypatch.setattr(module, "_publish_backup", publish_then_drift)
    with pytest.raises(module.ConfigGateError, match="changed_after_plan"):
        module.apply_plan(
            expected_before_sha256=before_sha,
            approved_plan_sha256=plan["plan_sha256"],
        )
    assert path.read_bytes().endswith(b"# concurrent operator change\n")


def test_backup_publication_failure_leaves_no_final_file(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    target = path.with_name("exact-backup.bak")

    def fail_publish(_source, _target):
        raise OSError("injected publish interruption")

    monkeypatch.setattr(module, "_rename_noreplace", fail_publish)
    with pytest.raises(OSError, match="injected publish interruption"):
        module._publish_backup(target, path.read_bytes())
    assert not target.exists()
    assert not list(tmp_path.glob(".exact-backup.bak.publish.*"))


def test_gateway_stop_gate_requires_loaded_inactive_zero_pid(monkeypatch):
    module = _load()
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=inactive\nMainPID=0\n",
        ),
    )
    assert module._require_gateway_stopped()["active_state"] == "inactive"

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=active\nMainPID=42\n",
        ),
    )
    with pytest.raises(module.ConfigGateError, match="gateway_not_stopped"):
        module._require_gateway_stopped()


def test_apply_rechecks_stopped_gateway_at_replace_boundary(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)
    checks = 0

    def stopped():
        nonlocal checks
        checks += 1
        return {
            "unit": module.GATEWAY_UNIT,
            "load_state": "loaded",
            "active_state": "inactive",
            "main_pid": 0,
        }

    monkeypatch.setattr(module, "_require_gateway_stopped", stopped)
    module.apply_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    assert checks == 2


def test_linux_exchange_path_applies_and_rolls_back_exactly(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before = path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)

    def noreplace(source, target):
        if target.exists():
            raise FileExistsError(target)
        source.replace(target)

    def exchange(source, target):
        holding = source.with_name(f".{source.name}.exchange")
        target.replace(holding)
        source.replace(target)
        holding.replace(source)

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "_rename_noreplace", noreplace)
    monkeypatch.setattr(module, "_rename_exchange", exchange)
    applied = module.apply_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == applied["after_sha256"]

    rolled_back = module.rollback_plan(
        expected_before_sha256=before_sha,
        approved_plan_sha256=plan["plan_sha256"],
    )
    assert rolled_back["to_sha256"] == before_sha
    assert path.read_bytes() == before


def test_linux_exchange_detects_and_restores_last_moment_writer(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = module.build_plan(expected_before_sha256=before_sha)
    concurrent = path.read_bytes() + b"# last-moment writer\n"
    exchanges = 0

    def noreplace(source, target):
        if target.exists():
            raise FileExistsError(target)
        source.replace(target)

    def exchange(source, target):
        nonlocal exchanges
        exchanges += 1
        if exchanges == 1:
            target.write_bytes(concurrent)
        holding = source.with_name(f".{source.name}.exchange")
        target.replace(holding)
        source.replace(target)
        holding.replace(source)

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "_rename_noreplace", noreplace)
    monkeypatch.setattr(module, "_rename_exchange", exchange)
    with pytest.raises(module.ConfigGateError, match="changed_after_plan"):
        module.apply_plan(
            expected_before_sha256=before_sha,
            approved_plan_sha256=plan["plan_sha256"],
        )
    assert exchanges == 2
    assert path.read_bytes() == concurrent


def test_semantic_drift_and_yaml_aliases_are_rejected(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    path.write_bytes(_config().replace(b"reasoning_effort: high", b"reasoning_effort: low"))
    with pytest.raises(module.ConfigGateError, match="agent_baseline_drifted"):
        module.build_plan(
            expected_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    path.write_text("model: &model {}\ncopy: *model\n")
    with pytest.raises(module.ConfigGateError, match="yaml_alias_forbidden"):
        module.build_plan(
            expected_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_plan_requires_curator_semantic_automation_to_remain_disabled(
    tmp_path, monkeypatch
):
    module, path = _prepare(tmp_path, monkeypatch)
    path.write_bytes(_config().replace(b"  consolidate: false", b"  consolidate: true"))

    with pytest.raises(module.ConfigGateError, match="curator_policy_drifted"):
        module.build_plan(
            expected_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_plan_requires_tool_loop_hard_stop_to_remain_disabled(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    path.write_bytes(
        _config().replace(b"  hard_stop_enabled: false", b"  hard_stop_enabled: true")
    )

    with pytest.raises(module.ConfigGateError, match="tool_loop_policy_drifted"):
        module.build_plan(
            expected_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_plan_requires_completion_and_parallel_guidance(tmp_path, monkeypatch):
    module, path = _prepare(tmp_path, monkeypatch)
    path.write_bytes(
        _config().replace(
            b"  task_completion_guidance: true",
            b"  task_completion_guidance: false",
        )
    )

    with pytest.raises(module.ConfigGateError, match="execution_policy_drifted"):
        module.build_plan(
            expected_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
