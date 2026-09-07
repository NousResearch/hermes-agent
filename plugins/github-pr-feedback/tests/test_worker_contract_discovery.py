"""Worker configuration must admit the policy through real plugin discovery."""
import importlib.metadata
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from github_pr_feedback.cli import DoctorProbe
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import FeedbackReceipt


@pytest.mark.parametrize("plugins, expected", [
    ({"enabled": []}, False),
    ({"enabled": ["github-pr-feedback"]}, True),
    ({"enabled": ["github-pr-feedback"], "disabled": ["github-pr-feedback"]}, False),
    ({"enabled": "github-pr-feedback"}, False),
    ({"enabled": ["github-pr-feedback", {}]}, False),
])
def test_doctor_checks_worker_plugin_opt_in_without_changing_profile(tmp_path, plugins, expected):
    profile = tmp_path / "profiles/worker/config.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump({"plugins": plugins, "model": {"name": "keep-this-model"}}))
    before = profile.read_bytes()
    policy = SimpleNamespace(assignee="worker", assignee_rules=(), routing_rules=(),
                             local_ci_audit=None, repair_steward=None, targets={}, board="repairs",
                             merge_policies=lambda: (), release_policies=lambda: ())
    runner = SimpleNamespace(which=lambda name: None)
    checks = DoctorProbe(tmp_path, runner).checks(policy, tmp_path / "ledger.sqlite3")
    assert checks.get("worker_completion_policy") == ("ok" if expected else "failed")
    assert profile.read_bytes() == before


def test_doctor_checks_worker_plugin_opt_in_after_env_expansion(tmp_path, monkeypatch):
    profile = tmp_path / "profiles/worker/config.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"],
        "disabled": ["${WORKER_DISABLED_PLUGIN}"],
    }}))
    monkeypatch.setenv("WORKER_DISABLED_PLUGIN", "github-pr-feedback")
    policy = SimpleNamespace(assignee="worker", assignee_rules=(), routing_rules=(),
                             local_ci_audit=None, repair_steward=None, targets={}, board="repairs",
                             merge_policies=lambda: (), release_policies=lambda: ())
    runner = SimpleNamespace(which=lambda name: None)

    checks = DoctorProbe(tmp_path, runner).checks(policy, tmp_path / "ledger.sqlite3")

    assert checks.get("worker_completion_policy") == "failed"


@pytest.mark.parametrize("enabled", [False, True])
def test_real_worker_discovery_enforces_control_home_receipt(tmp_path, monkeypatch, enabled):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, plugins
    from tools import kanban_tools

    control = tmp_path / "control"
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"] if enabled else [], "disabled": []}}))
    monkeypatch.setenv("HERMES_HOME", str(worker))
    monkeypatch.setenv("HERMES_CONTROL_HOME", str(control))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as connection:
        tid = kb.create_task(connection, title="Repair feedback", assignee="worker")
        kb.claim_task(connection, tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(kb.get_task(connection, tid).current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    ledger = FeedbackLedger(control / "github-pr-feedback/ledger.sqlite3")
    now = datetime.now(UTC)
    receipt = FeedbackReceipt("acme/repo", 1, "review_comment", "comment-1", "a" * 40)
    claim = ledger.claim(receipt, owner="test", claimed_at=now, stale_before=now-timedelta(minutes=5))
    ledger.finalize(receipt, tid, claim)
    try:
        # No synthetic manager or manually registered hooks: load the real plugin
        # from its manifest using this worker's actual opt-in configuration.
        plugins.discover_plugins(force=True)
        result = json.loads(kanban_tools._handle_complete({"task_id": tid, "summary": "Tests passed"}))
        assert (result.get("ok") is True) is (not enabled)
        with kbc.connect() as connection:
            assert (kb.get_task(connection, tid).status == "done") is (not enabled)
        assert ledger._connection.execute("SELECT action_status FROM feedback_receipts").fetchone()[0] == "pending"
    finally:
        ledger.close()


def test_worker_readiness_rejects_user_override_without_completion_hooks(tmp_path, monkeypatch):
    from github_pr_feedback.worker_contract import worker_contract_enabled

    worker = tmp_path / "profiles/worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"], "disabled": []}}))
    plugin = worker / "plugins/github-pr-feedback"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: github-pr-feedback\ndescription: stale user override\n"
    )
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('executed').write_text('unsafe')\n"
        "def register(ctx):\n    return None\n"
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert worker_contract_enabled(tmp_path, "worker") is False
    assert not (plugin / "executed").exists()


def test_worker_readiness_rejects_portable_manifest_hooks(tmp_path, monkeypatch):
    import github_pr_feedback.worker_contract as worker_contract

    worker = tmp_path / "profiles/worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"], "disabled": []}}))
    plugin = worker / "plugins/github-pr-feedback"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        "name": "github-pr-feedback",
        "provides_hooks": ["pre_tool_call", "pre_kanban_complete"],
    }))
    monkeypatch.setattr(worker_contract, "_entrypoint_override_present", lambda: False)

    assert worker_contract.worker_contract_enabled(tmp_path, "worker") is False


@pytest.mark.parametrize(("project_enabled", "expected"), [(False, True), (True, False)])
def test_worker_readiness_applies_project_plugin_opt_in(
    tmp_path, monkeypatch, project_enabled, expected
):
    import github_pr_feedback.worker_contract as worker_contract

    worker = tmp_path / "profiles/worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"], "disabled": []}}))
    project_plugin = tmp_path / "project/.hermes/plugins/github-pr-feedback"
    project_plugin.mkdir(parents=True)
    (project_plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "github-pr-feedback",
        "provides_hooks": ["pre_tool_call", "pre_kanban_complete"],
    }))
    monkeypatch.setattr(worker_contract, "_entrypoint_override_present", lambda: False)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1" if project_enabled else "")

    assert worker_contract.worker_contract_enabled(
        tmp_path, "worker", project_root=tmp_path / "project"
    ) is expected


def test_worker_readiness_rejects_entrypoint_override_without_completion_hooks(tmp_path, monkeypatch):
    from github_pr_feedback.worker_contract import worker_contract_enabled

    worker = tmp_path / "profiles/worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"], "disabled": []}}))
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: [
        SimpleNamespace(group="hermes_agent.plugins", name="github-pr-feedback")
    ])

    assert worker_contract_enabled(tmp_path, "worker") is False


def test_worker_readiness_rejects_manifest_declared_hooks_without_importing_worker(tmp_path):
    from github_pr_feedback.worker_contract import worker_contract_enabled

    worker = tmp_path / "profiles/worker"
    plugin = worker / "plugins/github-pr-feedback"
    plugin.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["github-pr-feedback"], "disabled": []}}))
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "github-pr-feedback", "provides_hooks": ["pre_tool_call", "pre_kanban_complete"]
    }))
    (plugin / "__init__.py").write_text("raise AssertionError('worker code imported')\n")

    assert worker_contract_enabled(tmp_path, "worker") is False


@pytest.mark.parametrize("enabled", [["github-pr-feedback"], ["category/github-pr-feedback"]])
def test_worker_readiness_rejects_untrusted_bare_and_canonical_manifest_keys(tmp_path, enabled):
    from github_pr_feedback.worker_contract import worker_contract_enabled

    worker = tmp_path / "profiles/worker"
    plugin = worker / "plugins/category/github-pr-feedback"
    plugin.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": enabled, "disabled": []}}))
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "github-pr-feedback", "provides_hooks": ["pre_tool_call", "pre_kanban_complete"]
    }))

    assert worker_contract_enabled(tmp_path, "worker") is False


@pytest.mark.parametrize("managed,raw,expected", [
    ({"plugins": {"disabled": ["github-pr-feedback"]}}, b"plugins:\n  enabled: [github-pr-feedback]\n", False),
    ({}, b"\xff", False),
])
def test_worker_readiness_respects_managed_policy_and_invalid_encoding(tmp_path, monkeypatch, managed, raw, expected):
    from github_pr_feedback.worker_contract import worker_contract_enabled
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(yaml.safe_dump(managed))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    profile = tmp_path / "profiles/worker/config.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_bytes(raw)
    assert worker_contract_enabled(tmp_path, "worker") is expected
    assert profile.read_bytes() == raw
