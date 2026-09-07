from datetime import UTC, datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from github_pr_feedback.ci_runner import CIAuditIdentity, CIAuditReceipt, CommandEvidence, _receipt_id
from github_pr_feedback.github_client import CheckState
from github_pr_feedback.controller import _local_ci_feedback_id
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import FeedbackReceipt


def _registered_guard(enabled=True):
    spec = spec_from_file_location("feedback_guard_entry", Path(__file__).parents[1] / "__init__.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    hooks = {}
    ctx = SimpleNamespace(get_config=lambda key, default=None: enabled if key == "enabled" else default,
                          register_cli_command=lambda **kwargs: None,
                          register_hook=lambda name, callback: hooks.setdefault(name, callback))
    module.register(ctx)
    return hooks["pre_tool_call"]


@pytest.mark.parametrize("case", ["none", "wrong_head", "wrong_base", "stale", "valid", "non_ci", "running", "failed"])
def test_registered_ci_completion_gate_uses_durable_exact_dispatch(tmp_path, monkeypatch, case):
    from hermes_cli.plugins import get_pre_tool_call_block_message
    import hermes_cli.lifecycle as lifecycle

    control = tmp_path / "control"
    monkeypatch.setenv("HERMES_CONTROL_HOME", str(control))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "pr-local-ci-auditor"))
    ledger = FeedbackLedger(control / "github-pr-feedback" / "ledger.sqlite3")
    now = datetime.now(UTC)
    identity = CIAuditIdentity("acme/widgets", 17, "a" * 40, "b" * 40)
    kind = "review_comment" if case == "non_ci" else "pr_local_ci"
    dispatch = FeedbackReceipt(identity.repository, identity.pr_number, kind,
                               _local_ci_feedback_id(identity), identity.head_sha)
    try:
        claim = ledger.claim(dispatch, owner="test", claimed_at=now, stale_before=now - timedelta(minutes=5))
        ledger.finalize(dispatch, "audit-task", claim)
        if case == "running":
            assert ledger.claim_ci_run(identity.repository, identity.pr_number, identity.base_sha,
                                       identity.head_sha, "f" * 64, supervisor_pid=12345,
                                       claimed_at=now, stale_before=now - timedelta(minutes=5),
                                       pid_is_alive=lambda pid: True) is not None
        if case not in {"none", "non_ci", "running"}:
            actual = CIAuditIdentity(identity.repository, identity.pr_number,
                                     "c" * 40 if case == "wrong_base" else identity.base_sha,
                                     "c" * 40 if case == "wrong_head" else identity.head_sha)
            started = now - timedelta(minutes=2) if case == "stale" else now
            completed = started + timedelta(seconds=1)
            commands = (CommandEvidence(("scripts/run_tests.sh",), str(tmp_path), 0, 1, False,
                                        "d" * 64, "e" * 64, "passed"),)
            status = "failed" if case == "failed" else "passed"
            receipt = CIAuditReceipt(_receipt_id(actual, "f" * 64, status, completed, commands),
                                    actual, "f" * 64, status, started, completed,
                                    CheckState(False, True, 0), commands)
            ledger.record_ci_receipt(receipt)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "audit-task")
        hook = _registered_guard(enabled=False)  # worker settings do not own the control ledger
        monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **kw: [hook(**kw)])
        rejection = get_pre_tool_call_block_message("kanban_complete", {
            "summary": "Tests passed, invented command output", "metadata": {"receipt_id": "fabricated"}})
        assert (rejection is None) is (case in {"valid", "non_ci"})
        assert get_pre_tool_call_block_message("kanban_block", {}) is None
    finally:
        ledger.close()


def test_ledger_unavailable_blocks_completion_without_creating_it(tmp_path, monkeypatch):
    target = tmp_path / "github-pr-feedback" / "ledger.sqlite3"
    monkeypatch.setenv("HERMES_CONTROL_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "audit-task")
    result = _registered_guard()(tool_name="kanban_complete", args={})
    assert result["action"] == "block"
    assert not target.exists()
