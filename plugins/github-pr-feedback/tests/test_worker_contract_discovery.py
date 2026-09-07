"""Worker configuration must admit the policy through real plugin discovery."""
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
