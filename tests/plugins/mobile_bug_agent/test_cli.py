from __future__ import annotations

import copy
import json
import re
from argparse import ArgumentParser
from types import SimpleNamespace

from plugins.mobile_bug_agent import cli
from plugins.mobile_bug_agent.cli import (
    _open_state,
    run_approve_command,
    run_configure_approved_pr_command,
    run_configure_linear_only_command,
    run_configure_local_fix_only_command,
    run_doctor_command,
    run_linear_metadata_command,
    run_retry_command,
    run_setup_plan_command,
    run_simulate_command,
    run_show_command,
    run_slack_manifest_command,
    run_slack_metadata_command,
    run_status_command,
)
from plugins.mobile_bug_agent.config import (
    LinearConfig,
    LoopConfig,
    MonicaConfig,
    ProofConfig,
    RepoConfig,
    RuntimeConfig,
    SlackConfig,
    VerificationConfig,
    WorkerConfig,
    config_from_mapping,
)
from plugins.mobile_bug_agent.state import MonicaState


def test_status_lists_recent_runs(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )

    exit_code = run_status_command(state=state, limit=5)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "checkout crash" in out


def test_status_lists_rollout_context_for_recent_runs(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        linear_identifier="MOB-123",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
    )

    exit_code = run_status_command(state=state, limit=5)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "slack:C123/1710000000.000100" in out
    assert "linear:MOB-123" in out
    assert "branch:monica/MOB-123-checkout-crash" in out
    assert "url:https://linear.app/acme/issue/MOB-123" in out


def test_status_json_lists_recent_runs(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="awaiting_fix_approval",
        linear_identifier="MOB-123",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
    )

    exit_code = run_status_command(state=state, limit=5, json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["count"] == 1
    assert payload["runs"][0]["id"] == run.id
    assert payload["runs"][0]["status"] == "awaiting_fix_approval"
    assert payload["runs"][0]["slack"]["channel_id"] == "C123"
    assert payload["runs"][0]["linear"]["identifier"] == "MOB-123"
    assert payload["runs"][0]["branch_name"] == "monica/MOB-123-checkout-crash"
    assert payload["runs"][0]["request_text"] == "@monica checkout crash"


def test_show_missing_run_returns_error(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")

    exit_code = run_show_command(state=state, run_id="missing")

    assert exit_code == 1
    assert "missing" in capsys.readouterr().out


def test_show_json_outputs_full_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
        raw_event={"type": "app_mention", "event_id": "E123"},
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="verification failed",
        approved_by_user_id="U_APPROVER",
        pr_url="https://github.com/acme/mobile/pull/123",
    )

    exit_code = run_show_command(state=state, run_id=run.id, json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["id"] == run.id
    assert payload["status"] == "failed"
    assert payload["failure_reason"] == "verification failed"
    assert payload["approved_by_user_id"] == "U_APPROVER"
    assert payload["pr_url"] == "https://github.com/acme/mobile/pull/123"
    assert payload["raw_event"] == {"type": "app_mention", "event_id": "E123"}


def test_retry_refuses_completed_runs(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        pr_url="https://github.com/acme/mobile/pull/123",
        approved_by_user_id="U1",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "done"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert launched == []
    assert "not retryable" in capsys.readouterr().out


def test_retry_refuses_cancelled_runs(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="blocked",
        failure_reason="cancelled by U_APPROVER",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "blocked"
    assert updated.failure_reason == "cancelled by U_APPROVER"
    assert launched == []
    assert "was cancelled from Slack" in capsys.readouterr().out


def test_retry_failed_approved_run_resumes_from_approval_gate(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="opening_pr_failed: gh pr create failed",
        approved_by_user_id="U_APPROVER",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 0
    assert updated.status == "approved"
    assert updated.failure_reason == ""
    assert updated.approved_by_user_id == "U_APPROVER"
    assert launched == [run.id]
    assert f"Retried Monica run {run.id}" in capsys.readouterr().out


def test_retry_proof_blocked_run_resumes_from_proof_gate(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="proof_blocked",
        failure_reason="proof_unavailable: simulator not configured",
        branch_name="monica/MOB-123-checkout-crash",
        approved_by_user_id="U_APPROVER",
    )
    launched: list[str] = []

    def fake_loop(run_id: str, *, state: MonicaState) -> None:
        relaunched = state.get_run(run_id)
        assert relaunched is not None
        assert relaunched.status == "proof_blocked"
        assert relaunched.failure_reason == ""
        assert relaunched.branch_name == "monica/MOB-123-checkout-crash"
        assert relaunched.pr_url == ""
        launched.append(run_id)

    monkeypatch.setattr(cli, "_run_loop", fake_loop)

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 0
    assert updated.status == "proof_blocked"
    assert updated.failure_reason == ""
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert launched == [run.id]
    assert f"Retried Monica run {run.id}" in capsys.readouterr().out


def test_retry_interrupted_proofing_run_resumes_from_proof_gate(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="proofing",
        branch_name="monica/MOB-123-checkout-crash",
        approved_by_user_id="U_APPROVER",
    )
    launched: list[str] = []

    def fake_loop(run_id: str, *, state: MonicaState) -> None:
        relaunched = state.get_run(run_id)
        assert relaunched is not None
        assert relaunched.status == "proof_blocked"
        assert relaunched.failure_reason == ""
        assert relaunched.branch_name == "monica/MOB-123-checkout-crash"
        launched.append(run_id)

    monkeypatch.setattr(cli, "_run_loop", fake_loop)

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 0
    assert updated.status == "proof_blocked"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert launched == [run.id]
    assert f"Retried Monica run {run.id}" in capsys.readouterr().out


def test_retry_json_success_outputs_updated_run(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="opening_pr_failed: transient gh error",
        approved_by_user_id="U_APPROVER",
    )
    launched: list[str] = []

    def fake_loop(run_id: str, *, state: MonicaState) -> None:
        launched.append(run_id)
        state.update_run(run_id, status="done", pr_url="https://github.com/acme/mobile/pull/123")

    monkeypatch.setattr(cli, "_run_loop", fake_loop)

    exit_code = run_retry_command(state=state, run_id=run.id, json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert launched == [run.id]
    assert payload["ok"] is True
    assert payload["action"] == "retry"
    assert payload["run"]["id"] == run.id
    assert payload["run"]["status"] == "done"
    assert payload["run"]["pr_url"] == "https://github.com/acme/mobile/pull/123"
    assert payload["run"]["approved_by_user_id"] == "U_APPROVER"


def test_retry_json_missing_run_outputs_error(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(state=state, run_id="missing", json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert launched == []
    assert payload["ok"] is False
    assert payload["action"] == "retry"
    assert payload["error"]["code"] == "not_found"
    assert "missing" in payload["error"]["message"]


def test_retry_approved_run_refuses_unconfigured_stored_approver(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="opening_pr_failed: gh pr create failed",
        approved_by_user_id="U_OLD_APPROVER",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(
        state=state,
        run_id=run.id,
        config=MonicaConfig(slack=SlackConfig(approver_user_ids=("U_CURRENT_APPROVER",))),
    )

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "failed"
    assert updated.failure_reason == "opening_pr_failed: gh pr create failed"
    assert updated.approved_by_user_id == "U_OLD_APPROVER"
    assert launched == []
    assert "stored approver U_OLD_APPROVER is not configured" in capsys.readouterr().out


def test_retry_approved_run_refuses_when_readiness_check_fails(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="opening_pr_failed: gh pr create failed",
        approved_by_user_id="U_APPROVER",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(
        state=state,
        run_id=run.id,
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(approver_user_ids=("U_APPROVER",)),
        ),
        readiness_checker=lambda: 1,
    )

    updated = state.get_run(run.id)
    out = capsys.readouterr().out
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "failed"
    assert updated.failure_reason == "opening_pr_failed: gh pr create failed"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert launched == []
    assert "Monica readiness check failed" in out


def test_retry_refuses_run_that_already_has_pr_url(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="opening_pr_failed: transient Slack post failed",
        branch_name="monica/MOB-123-checkout-crash",
        pr_url="https://github.com/acme/mobile/pull/123",
        approved_by_user_id="U_APPROVER",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "failed"
    assert updated.failure_reason == "opening_pr_failed: transient Slack post failed"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert launched == []
    assert "already has a draft PR" in capsys.readouterr().out


def test_retry_clears_stale_branch_metadata_before_relaunch(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="failed",
        failure_reason="opening_pr_failed: stale gh failure",
        branch_name="monica/old-branch",
    )
    relaunched_state: list[tuple[str, str, str, str]] = []

    def fake_run_loop(run_id: str, *, state: MonicaState) -> None:
        current = state.get_run(run_id)
        assert current is not None
        relaunched_state.append(
            (current.status, current.failure_reason, current.branch_name, current.pr_url)
        )

    monkeypatch.setattr(cli, "_run_loop", fake_run_loop)

    exit_code = run_retry_command(state=state, run_id=run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 0
    assert updated.status == "queued"
    assert updated.failure_reason == ""
    assert updated.branch_name == ""
    assert updated.pr_url == ""
    assert relaunched_state == [("queued", "", "", "")]
    assert f"Retried Monica run {run.id}" in capsys.readouterr().out


def test_approve_refuses_completed_runs(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        pr_url="https://github.com/acme/mobile/pull/123",
    )
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_approve_command(state=state, run_id=run.id, user_id="local-operator")

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "done"
    assert updated.approved_by_user_id == ""
    assert launched == []
    assert "not awaiting fix approval" in capsys.readouterr().out


def test_approve_waiting_run_marks_approved_and_invokes_loop(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_approve_command(state=state, run_id=run.id, user_id="local-operator")

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 0
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "local-operator"
    assert launched == [run.id]
    assert f"Approved Monica run {run.id}" in capsys.readouterr().out


def test_approve_json_success_outputs_updated_run(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    launched: list[str] = []

    def fake_loop(run_id: str, *, state: MonicaState) -> None:
        launched.append(run_id)
        state.update_run(run_id, status="fixing")

    monkeypatch.setattr(cli, "_run_loop", fake_loop)

    exit_code = run_approve_command(
        state=state,
        run_id=run.id,
        user_id="local-operator",
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert launched == [run.id]
    assert payload["ok"] is True
    assert payload["action"] == "approve"
    assert payload["run"]["id"] == run.id
    assert payload["run"]["status"] == "fixing"
    assert payload["run"]["approved_by_user_id"] == "local-operator"


def test_approve_json_unconfigured_user_outputs_error(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_approve_command(
        state=state,
        run_id=run.id,
        user_id="local-operator",
        config=MonicaConfig(slack=SlackConfig(approver_user_ids=("U_APPROVER",))),
        json_output=True,
    )

    updated = state.get_run(run.id)
    payload = json.loads(capsys.readouterr().out)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []
    assert payload["ok"] is False
    assert payload["action"] == "approve"
    assert payload["error"]["code"] == "approver_not_allowed"
    assert payload["run"]["id"] == run.id
    assert payload["run"]["status"] == "awaiting_fix_approval"


def test_approve_refuses_user_outside_configured_approver_allowlist(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_approve_command(
        state=state,
        run_id=run.id,
        user_id="local-operator",
        config=MonicaConfig(slack=SlackConfig(approver_user_ids=("U_APPROVER",))),
    )

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []
    assert "not configured as a Monica code approver" in capsys.readouterr().out


def test_approve_refuses_approved_pr_when_readiness_check_fails(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    exit_code = run_approve_command(
        state=state,
        run_id=run.id,
        user_id="U_APPROVER",
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(approver_user_ids=("U_APPROVER",)),
        ),
        readiness_checker=lambda: 1,
    )

    updated = state.get_run(run.id)
    out = capsys.readouterr().out
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []
    assert "Monica readiness check failed" in out


def test_approve_does_not_launch_when_atomic_approval_already_changed(tmp_path, capsys, monkeypatch):
    state = MonicaState.open(tmp_path / "state.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C123",
        thread_ts="T1",
        message_ts="T1",
        user_id="U1",
        request_text="@monica checkout crash",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    launched: list[str] = []
    monkeypatch.setattr(cli, "_run_loop", lambda run_id, *, state: launched.append(run_id))

    def approve_once(run_id: str, *, approved_by_user_id: str):
        approved = state.update_run(run_id, status="approved", approved_by_user_id="U_OTHER")
        return approved, False

    monkeypatch.setattr(state, "approve_fix_once", approve_once)

    exit_code = run_approve_command(state=state, run_id=run.id, user_id="local-operator")

    updated = state.get_run(run.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "U_OTHER"
    assert launched == []
    assert "already approved or no longer awaiting fix approval" in capsys.readouterr().out


def test_simulate_requires_enabled_config(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=False),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
    )

    assert exit_code == 1
    assert "Monica is disabled" in capsys.readouterr().out
    assert state.list_runs() == []


def test_simulate_json_reports_disabled_config_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=False),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "dry_run"
    assert payload["error"]["code"] == "config_disabled"
    assert "mobile_bug_agent.enabled" in payload["error"]["message"]


def test_simulate_creates_local_slack_shaped_run_and_invokes_loop(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    def fake_loop(run_id: str, *, state: MonicaState) -> None:
        launched.append(run_id)
        state.update_run(run_id, status="done", linear_identifier="DRY-RUN")

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        thread_ts="sim-thread",
        loop_runner=fake_loop,
    )

    runs = state.list_runs()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert runs[0].platform == "slack"
    assert runs[0].channel_id == "LOCAL"
    assert runs[0].thread_ts == "sim-thread"
    assert runs[0].raw_event is not None
    assert runs[0].raw_event["monica_simulated"] is True
    assert runs[0].raw_event["type"] == "app_mention"
    assert runs[0].raw_event["permalink"] == "local://monica/simulated/sim-thread"
    assert "Simulated Monica run" in out
    assert "DRY-RUN" in out


def test_simulate_json_outputs_created_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    def fake_loop(run_id: str, *, state: MonicaState) -> None:
        launched.append(run_id)
        state.update_run(
            run_id,
            status="done",
            linear_identifier="DRY-RUN",
            linear_url="https://linear.app/acme/issue/DRY-RUN",
        )

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        thread_ts="sim-thread",
        loop_runner=fake_loop,
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    runs = state.list_runs()
    assert exit_code == 0
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert payload["run"]["id"] == runs[0].id
    assert payload["run"]["status"] == "done"
    assert payload["run"]["slack"]["channel_id"] == "LOCAL"
    assert payload["run"]["linear"]["identifier"] == "DRY-RUN"
    assert payload["created"] is True
    assert payload["allow_side_effects"] is False
    assert payload["rollout_mode"] == "dry_run"


def test_simulate_default_thread_ids_do_not_collide_when_clock_repeats(
    tmp_path,
    capsys,
    monkeypatch,
):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []
    monkeypatch.setattr(cli.time, "time", lambda: 1710000000.0)

    first = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )
    second = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="checkout still crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    runs = state.list_runs()
    assert first == 0
    assert second == 0
    assert len(runs) == 2
    assert len({run.thread_ts for run in runs}) == 2
    assert set(launched) == {run.id for run in runs}
    assert all(re.fullmatch(r"\d+\.\d{6}", run.thread_ts) for run in runs)
    assert "already exists" not in capsys.readouterr().out


def test_simulate_refuses_side_effect_rollout_without_explicit_flag(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="linear_only", dry_run=False),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        thread_ts="sim-thread",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert "--allow-side-effects" in out


def test_simulate_json_reports_empty_text_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="   ",
        channel_id="LOCAL",
        user_id="local-operator",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "dry_run"
    assert payload["allow_side_effects"] is False
    assert payload["error"]["code"] == "empty_text"
    assert "text is required" in payload["error"]["message"]


def test_simulate_json_reports_side_effect_rollout_without_explicit_flag(
    tmp_path,
    capsys,
):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="linear_only", dry_run=False),
        state=state,
        text="checkout crashes on Android",
        channel_id="LOCAL",
        user_id="local-operator",
        thread_ts="sim-thread",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["allow_side_effects"] is False
    assert payload["error"]["code"] == "side_effects_not_allowed"
    assert "--allow-side-effects" in payload["error"]["message"]


def test_simulate_allows_side_effect_rollout_with_explicit_flag(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C_SIM",),
                bot_user_ids=("U_MONICA",),
            ),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        readiness_checker=lambda: 0,
    )

    runs = state.list_runs()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert "Simulated Monica run" in out


def test_simulate_refuses_side_effect_rollout_when_readiness_check_fails(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C_SIM",),
                bot_user_ids=("U_MONICA",),
            ),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        readiness_checker=lambda: 1,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert "Monica readiness check failed" in out


def test_simulate_json_reports_readiness_failure_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C_SIM",),
                bot_user_ids=("U_MONICA",),
            ),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        json_output=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        readiness_checker=lambda: 1,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["allow_side_effects"] is True
    assert payload["error"]["code"] == "readiness_failed"


def test_simulate_json_reports_readiness_exception_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    def fail_readiness() -> int:
        raise RuntimeError("doctor exploded")

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C_SIM",),
                bot_user_ids=("U_MONICA",),
            ),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        json_output=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        readiness_checker=fail_readiness,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["allow_side_effects"] is True
    assert payload["error"]["code"] == "readiness_exception"
    assert "doctor exploded" in payload["error"]["message"]


def test_simulate_refuses_side_effect_rollout_without_bot_user_ids(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(allowed_channels=("C_SIM",)),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert "mobile_bug_agent.slack.bot_user_ids" in out


def test_simulate_json_reports_missing_bot_user_ids_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(allowed_channels=("C_SIM",)),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["allow_side_effects"] is True
    assert payload["error"]["code"] == "bot_user_ids_empty"
    assert "mobile_bug_agent.slack.bot_user_ids" in payload["error"]["message"]


def test_simulate_refuses_approved_pr_side_effects_without_configured_approvers(
    tmp_path,
    capsys,
):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            dry_run=False,
            slack=SlackConfig(allowed_channels=("C_SIM",), bot_user_ids=("U_MONICA",)),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert "mobile_bug_agent.slack.approver_user_ids" in out


def test_simulate_json_reports_missing_approvers_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            dry_run=False,
            slack=SlackConfig(allowed_channels=("C_SIM",), bot_user_ids=("U_MONICA",)),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "approved_pr"
    assert payload["allow_side_effects"] is True
    assert payload["error"]["code"] == "approver_user_ids_empty"
    assert "mobile_bug_agent.slack.approver_user_ids" in payload["error"]["message"]


def test_simulate_refuses_side_effect_rollout_when_allowed_channels_are_empty(
    tmp_path,
    capsys,
):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="linear_only", dry_run=False),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert "mobile_bug_agent.slack.allowed_channels" in out


def test_simulate_json_reports_missing_allowed_channels_without_creating_run(
    tmp_path,
    capsys,
):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="linear_only", dry_run=False),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_SIM",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["allow_side_effects"] is True
    assert payload["error"]["code"] == "allowed_channels_empty"
    assert "mobile_bug_agent.slack.allowed_channels" in payload["error"]["message"]


def test_simulate_refuses_side_effect_rollout_outside_allowed_channel(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(allowed_channels=("C_MOBILE",), bot_user_ids=("U_MONICA",)),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_OTHER",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert "C_OTHER is not in mobile_bug_agent.slack.allowed_channels" in out


def test_simulate_json_reports_disallowed_channel_without_creating_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(allowed_channels=("C_MOBILE",), bot_user_ids=("U_MONICA",)),
        ),
        state=state,
        text="checkout crashes on Android",
        channel_id="C_OTHER",
        user_id="local-operator",
        thread_ts="sim-thread",
        allow_side_effects=True,
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert state.list_runs() == []
    assert launched == []
    assert payload["created"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["allow_side_effects"] is True
    assert payload["error"]["code"] == "channel_not_allowed"
    assert "C_OTHER" in payload["error"]["message"]


def test_simulate_refuses_duplicate_thread_without_relaunching(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    existing = state.create_run(
        platform="slack",
        channel_id="LOCAL",
        thread_ts="sim-thread",
        message_ts="sim-message",
        user_id="local-operator",
        request_text="old checkout crash simulation",
        raw_event={"monica_simulated": True},
    )
    state.update_run(existing.id, status="done", linear_identifier="DRY-RUN")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="new checkout crash simulation",
        channel_id="LOCAL",
        user_id="local-operator",
        thread_ts="sim-thread",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
    )

    updated = state.get_run(existing.id)
    assert updated is not None
    assert exit_code == 1
    assert updated.status == "done"
    assert updated.request_text == "old checkout crash simulation"
    assert launched == []
    assert "already exists" in capsys.readouterr().out


def test_simulate_json_duplicate_thread_returns_existing_run(tmp_path, capsys):
    state = MonicaState.open(tmp_path / "state.sqlite")
    existing = state.create_run(
        platform="slack",
        channel_id="LOCAL",
        thread_ts="sim-thread",
        message_ts="sim-message",
        user_id="local-operator",
        request_text="old checkout crash simulation",
        raw_event={"monica_simulated": True},
    )
    state.update_run(existing.id, status="done", linear_identifier="DRY-RUN")
    launched: list[str] = []

    exit_code = run_simulate_command(
        config=MonicaConfig(enabled=True, rollout_mode="dry_run"),
        state=state,
        text="new checkout crash simulation",
        channel_id="LOCAL",
        user_id="local-operator",
        thread_ts="sim-thread",
        loop_runner=lambda run_id, *, state: launched.append(run_id),
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert launched == []
    assert payload["created"] is False
    assert payload["error"]["code"] == "duplicate_thread"
    assert payload["run"]["id"] == existing.id
    assert payload["run"]["status"] == "done"
    assert payload["run"]["linear"]["identifier"] == "DRY-RUN"


def test_open_state_uses_configured_runtime_root(tmp_path):
    config = config_from_mapping({"runtime": {"home_subdir": str(tmp_path / "monica")}})

    state = _open_state(config)

    assert state.path == tmp_path / "monica" / "state.sqlite"


def test_slack_manifest_outputs_required_monica_app_manifest(capsys):
    exit_code = run_slack_manifest_command(
        config=MonicaConfig(),
        app_name="Monica Staging",
        bot_display_name="monica",
    )

    manifest = json.loads(capsys.readouterr().out)
    scopes = manifest["oauth_config"]["scopes"]["bot"]
    assert exit_code == 0
    assert manifest["display_information"]["name"] == "Monica Staging"
    assert manifest["features"]["app_home"] == {
        "home_tab_enabled": False,
        "messages_tab_enabled": True,
        "messages_tab_read_only_enabled": False,
    }
    assert manifest["features"]["bot_user"]["display_name"] == "monica"
    assert manifest["features"]["bot_user"]["always_online"] is True
    assert manifest["settings"]["socket_mode_enabled"] is True
    assert manifest["settings"]["event_subscriptions"]["bot_events"] == [
        "app_mention",
        "message.im",
    ]
    assert scopes == [
        "app_mentions:read",
        "channels:history",
        "groups:history",
        "im:history",
        "im:read",
        "chat:write",
        "files:read",
    ]


def test_slack_manifest_omits_files_scope_when_attachment_downloads_disabled(capsys):
    exit_code = run_slack_manifest_command(
        config=MonicaConfig(slack=SlackConfig(download_attachments=False)),
        app_name="",
        bot_display_name="",
    )

    manifest = json.loads(capsys.readouterr().out)
    scopes = manifest["oauth_config"]["scopes"]["bot"]
    assert exit_code == 0
    assert manifest["display_information"]["name"] == "Monica"
    assert manifest["features"]["bot_user"]["display_name"] == "monica"
    assert "files:read" not in scopes


def test_linear_metadata_json_outputs_teams_projects_and_labels(capsys):
    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key

        def list_workspace_metadata(self):
            return SimpleNamespace(
                teams=(SimpleNamespace(id="team-mobile", key="MOB", name="Mobile"),),
                projects=(SimpleNamespace(id="project-app", name="Mobile App", state="started"),),
                labels=(
                    SimpleNamespace(id="label-bug", name="Bug", color="#e5484d"),
                    SimpleNamespace(id="label-mobile", name="Mobile", color="#3b82f6"),
                ),
            )

    exit_code = run_linear_metadata_command(
        api_key="lin_api_key",
        json_output=True,
        client_factory=FakeClient,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["teams"][0] == {"id": "team-mobile", "key": "MOB", "name": "Mobile"}
    assert payload["projects"][0] == {
        "id": "project-app",
        "name": "Mobile App",
        "state": "started",
    }
    assert payload["labels"][1] == {
        "id": "label-mobile",
        "name": "Mobile",
        "color": "#3b82f6",
    }


def test_linear_metadata_text_outputs_configure_hint(capsys):
    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key

        def list_workspace_metadata(self):
            return SimpleNamespace(
                teams=(SimpleNamespace(id="team-mobile", key="MOB", name="Mobile"),),
                projects=(),
                labels=(),
            )

    exit_code = run_linear_metadata_command(
        api_key="lin_api_key",
        json_output=False,
        client_factory=FakeClient,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Teams:" in out
    assert "team-mobile" in out
    assert "configure-linear-only" in out


def test_linear_metadata_refuses_missing_api_key_without_contacting_linear(capsys):
    contacted = False

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            nonlocal contacted
            contacted = True

    exit_code = run_linear_metadata_command(
        api_key="",
        json_output=True,
        client_factory=FakeClient,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert contacted is False
    assert payload["ok"] is False
    assert payload["error"]["code"] == "linear_api_key_missing"


def test_slack_metadata_json_outputs_bot_user_and_channels(capsys):
    class FakeClient:
        def __init__(self, *, token: str) -> None:
            self.token = token

        def list_workspace_metadata(self):
            return SimpleNamespace(
                auth=SimpleNamespace(
                    bot_user_id="U_MONICA",
                    bot_id="B_MONICA",
                    team_id="T123",
                    team_name="Acme",
                    team_url="https://acme.slack.com/",
                ),
                channels=(
                    SimpleNamespace(
                        id="C_MOBILE",
                        name="mobile-bugs",
                        is_private=False,
                        is_member=True,
                        is_archived=False,
                    ),
                    SimpleNamespace(
                        id="G_TRIAGE",
                        name="app-triage",
                        is_private=True,
                        is_member=False,
                        is_archived=False,
                    ),
                ),
            )

    exit_code = run_slack_metadata_command(
        token="xoxb-token",
        json_output=True,
        client_factory=FakeClient,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["auth"]["bot_user_id"] == "U_MONICA"
    assert payload["auth"]["bot_id"] == "B_MONICA"
    assert payload["auth"]["team_id"] == "T123"
    assert payload["channels"][0] == {
        "id": "C_MOBILE",
        "name": "mobile-bugs",
        "is_private": False,
        "is_member": True,
        "is_archived": False,
    }


def test_slack_metadata_text_outputs_configure_hint(capsys):
    class FakeClient:
        def __init__(self, *, token: str) -> None:
            self.token = token

        def list_workspace_metadata(self):
            return SimpleNamespace(
                auth=SimpleNamespace(
                    bot_user_id="U_MONICA",
                    bot_id="B_MONICA",
                    team_id="T123",
                    team_name="Acme",
                    team_url="https://acme.slack.com/",
                ),
                channels=(
                    SimpleNamespace(
                        id="C_MOBILE",
                        name="mobile-bugs",
                        is_private=False,
                        is_member=True,
                        is_archived=False,
                    ),
                ),
            )

    exit_code = run_slack_metadata_command(
        token="xoxb-token",
        json_output=False,
        client_factory=FakeClient,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Bot user ID: U_MONICA" in out
    assert "C_MOBILE" in out
    assert "configure-linear-only" in out


def test_slack_metadata_refuses_missing_token_without_contacting_slack(capsys):
    contacted = False

    class FakeClient:
        def __init__(self, *, token: str) -> None:
            nonlocal contacted
            contacted = True

    exit_code = run_slack_metadata_command(
        token="",
        json_output=True,
        client_factory=FakeClient,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert contacted is False
    assert payload["ok"] is False
    assert payload["error"]["code"] == "slack_bot_token_missing"


def test_slack_metadata_uses_monica_token_instead_of_shared_slack_token(capsys, monkeypatch):
    seen: list[str] = []

    class FakeClient:
        def __init__(self, *, token: str) -> None:
            seen.append(token)

        def list_workspace_metadata(self):
            return SimpleNamespace(auth=SimpleNamespace(), channels=())

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-chandler")
    monkeypatch.setenv("MONICA_SLACK_BOT_TOKEN", "xoxb-monica")

    exit_code = run_slack_metadata_command(
        json_output=True,
        client_factory=FakeClient,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert seen == ["xoxb-monica"]


def test_setup_plan_json_lists_ordered_linear_only_rollout_steps(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(enabled=False),
        target_rollout_mode="linear_only",
        json_output=True,
        environ={},
        which=lambda command: f"/usr/bin/{command}",
        module_available=lambda name: True,
    )

    payload = json.loads(capsys.readouterr().out)
    step_ids = [step["id"] for step in payload["steps"]]
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["rollout_mode"] == "linear_only"
    assert payload["failures"][0]["code"] == "config_enabled"
    assert step_ids == [
        "create_slack_app",
        "set_slack_bot_token",
        "discover_slack_ids",
        "set_linear_api_key",
        "discover_linear_ids",
        "configure_linear_only",
        "rerun_doctor",
    ]
    assert payload["steps"][0]["command"] == "hermes mobile-bug-agent slack-manifest"
    assert payload["steps"][2]["command"] == "hermes mobile-bug-agent slack-metadata --json"
    assert payload["steps"][4]["command"] == "hermes mobile-bug-agent linear-metadata --json"
    assert "configure-linear-only" in payload["steps"][5]["command"]


def test_setup_plan_json_lists_ordered_dry_run_bootstrap_steps(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(enabled=False),
        target_rollout_mode="dry_run",
        json_output=True,
        environ={},
        which=lambda command: f"/usr/bin/{command}",
        module_available=lambda name: True,
    )

    payload = json.loads(capsys.readouterr().out)
    step_ids = [step["id"] for step in payload["steps"]]
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["rollout_mode"] == "dry_run"
    assert payload["failures"][0]["code"] == "config_enabled"
    assert step_ids == [
        "configure_dry_run",
        "create_slack_app",
        "set_slack_bot_token",
        "rerun_doctor",
    ]
    assert payload["steps"][0]["command"] == "hermes mobile-bug-agent configure-dry-run"


def test_setup_plan_text_lists_next_commands(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(enabled=False),
        target_rollout_mode="linear_only",
        json_output=False,
        environ={},
        which=lambda command: f"/usr/bin/{command}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Monica setup plan: linear_only" in out
    assert "hermes mobile-bug-agent slack-manifest" in out
    assert "hermes mobile-bug-agent configure-linear-only" in out


def test_setup_plan_json_has_no_steps_when_rollout_is_ready(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C123MOBILE",),
                bot_user_ids=("U123MONICA",),
            ),
            linear=LinearConfig(team_id="team-mobile"),
        ),
        target_rollout_mode="linear_only",
        json_output=True,
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "LINEAR_API_KEY": "lin_api_key",
        },
        which=lambda command: f"/usr/bin/{command}",
        module_available=lambda name: True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["steps"] == []


def test_setup_plan_json_lists_proof_warning_steps_when_rollout_is_ready(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C123MOBILE",),
                bot_user_ids=("U123MONICA",),
                approver_user_ids=("U123APPROVER",),
            ),
            linear=LinearConfig(team_id="team-mobile"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
            proof=ProofConfig(
                enabled=True,
                required_for_done=True,
                commands=("uv run --project \"$MONICA_HERMES_AGENT_ROOT\" python -m plugins.mobile_bug_agent.simulator_proof",),
                platform_order=("ios", "android"),
            ),
        ),
        target_rollout_mode="approved_pr",
        json_output=True,
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin_api_key",
            "GITHUB_TOKEN": "gh_token",
        },
        which=lambda command: None
        if command in {"xcrun", "xcodebuild", "adb", "emulator"}
        else f"/usr/bin/{command}",
        module_available=lambda name: True,
    )

    payload = json.loads(capsys.readouterr().out)
    step_ids = [step["id"] for step in payload["steps"]]
    assert exit_code == 0
    assert payload["ready"] is True
    assert step_ids == ["prepare_ios_simulator", "prepare_android_emulator", "rerun_doctor"]
    assert "Install full Xcode" in payload["steps"][0]["why"]
    assert "Install Android SDK" in payload["steps"][1]["why"]


def test_setup_plan_json_lists_approved_pr_rollout_steps(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C123MOBILE",),
                bot_user_ids=("U123MONICA",),
            ),
            linear=LinearConfig(team_id="team-mobile"),
        ),
        target_rollout_mode="approved_pr",
        json_output=True,
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "LINEAR_API_KEY": "lin_api_key",
        },
        which=lambda command: None,
        module_available=lambda name: True,
    )

    payload = json.loads(capsys.readouterr().out)
    step_ids = [step["id"] for step in payload["steps"]]
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["rollout_mode"] == "approved_pr"
    assert "configure_approved_pr" in step_ids
    assert "prepare_pr_tools" in step_ids
    assert payload["steps"][-1]["id"] == "rerun_doctor"
    assert "configure-approved-pr" in next(
        step["command"] for step in payload["steps"] if step["id"] == "configure_approved_pr"
    )


def test_mobile_bug_agent_command_reports_invalid_runtime_root_for_state_actions(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-home"))
    monkeypatch.setattr(
        cli,
        "load_monica_config",
        lambda: config_from_mapping({"runtime": {"home_subdir": "../outside-runtime"}}),
    )

    exit_code = cli.mobile_bug_agent_command(
        SimpleNamespace(mobile_bug_agent_action="status", limit=5),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Monica runtime is not configured correctly" in out
    assert "mobile_bug_agent.runtime.home_subdir must stay inside HERMES_HOME." in out


def test_doctor_parser_accepts_target_rollout_mode():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(["doctor", "--rollout-mode", "linear_only", "--json"])

    assert args.mobile_bug_agent_action == "doctor"
    assert args.rollout_mode == "linear_only"
    assert args.json is True


def test_setup_plan_parser_accepts_target_rollout_mode_and_json():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(["setup-plan", "--rollout-mode", "linear_only", "--json"])

    assert args.mobile_bug_agent_action == "setup_plan"
    assert args.rollout_mode == "linear_only"
    assert args.json is True


def test_slack_manifest_parser_accepts_app_and_bot_names():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(
        ["slack-manifest", "--app-name", "Monica Staging", "--bot-display-name", "monica"]
    )

    assert args.mobile_bug_agent_action == "slack_manifest"
    assert args.app_name == "Monica Staging"
    assert args.bot_display_name == "monica"


def test_linear_metadata_parser_accepts_json_flag():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(["linear-metadata", "--json"])

    assert args.mobile_bug_agent_action == "linear_metadata"
    assert args.json is True


def test_slack_metadata_parser_accepts_json_flag():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(["slack-metadata", "--json"])

    assert args.mobile_bug_agent_action == "slack_metadata"
    assert args.json is True


def test_status_and_show_parsers_accept_json_flags():
    parser = ArgumentParser()

    cli.register_cli(parser)
    status_args = parser.parse_args(["status", "--json", "--limit", "3"])
    show_args = parser.parse_args(["show", "run-id", "--json"])

    assert status_args.mobile_bug_agent_action == "status"
    assert status_args.json is True
    assert status_args.limit == 3
    assert show_args.mobile_bug_agent_action == "show"
    assert show_args.json is True
    assert show_args.run_id == "run-id"


def test_retry_and_approve_parsers_accept_json_flags():
    parser = ArgumentParser()

    cli.register_cli(parser)
    retry_args = parser.parse_args(["retry", "run-id", "--json"])
    approve_args = parser.parse_args(["approve", "run-id", "--user-id", "U_APPROVER", "--json"])

    assert retry_args.mobile_bug_agent_action == "retry"
    assert retry_args.run_id == "run-id"
    assert retry_args.json is True
    assert approve_args.mobile_bug_agent_action == "approve"
    assert approve_args.run_id == "run-id"
    assert approve_args.user_id == "U_APPROVER"
    assert approve_args.json is True


def test_simulate_parser_accepts_json_flag():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(["simulate", "checkout", "crash", "--json"])

    assert args.mobile_bug_agent_action == "simulate"
    assert args.text == ["checkout", "crash"]
    assert args.json is True


def test_configure_linear_only_parser_accepts_non_secret_rollout_values():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(
        [
            "configure-linear-only",
            "--bot-user-id",
            "U123MONICA",
            "--channel-id",
            "C123MOBILE",
            "--linear-team-id",
            "team-id",
            "--linear-project-id",
            "project-id",
            "--linear-label-id",
            "label-one",
            "--linear-label-id",
            "label-two",
        ]
    )

    assert args.mobile_bug_agent_action == "configure_linear_only"
    assert args.bot_user_ids == ["U123MONICA"]
    assert args.channel_ids == ["C123MOBILE"]
    assert args.linear_team_id == "team-id"
    assert args.linear_project_id == "project-id"
    assert args.linear_label_ids == ["label-one", "label-two"]


def test_configure_dry_run_parser_accepts_optional_slack_scope_values():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(
        [
            "configure-dry-run",
            "--bot-user-id",
            "U123MONICA",
            "--channel-id",
            "C123PRIVATE",
            "--channel-id",
            "G456PRIVATE",
        ]
    )

    assert args.mobile_bug_agent_action == "configure_dry_run"
    assert args.bot_user_ids == ["U123MONICA"]
    assert args.channel_ids == ["C123PRIVATE", "G456PRIVATE"]


def test_configure_approved_pr_parser_accepts_non_secret_rollout_values():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(
        [
            "configure-approved-pr",
            "--approver-user-id",
            "U123APPROVER",
            "--repo-url",
            "git@github.com:acme/mobile-app.git",
            "--verification-command",
            "npm test",
            "--verification-command",
            "npm run lint",
            "--repo-local-name",
            "mobile-app",
            "--default-branch",
            "main",
            "--branch-prefix",
            "monica",
        ]
    )

    assert args.mobile_bug_agent_action == "configure_approved_pr"
    assert args.approver_user_ids == ["U123APPROVER"]
    assert args.repo_url == "git@github.com:acme/mobile-app.git"
    assert args.verification_commands == ["npm test", "npm run lint"]
    assert args.repo_local_name == "mobile-app"
    assert args.default_branch == "main"
    assert args.branch_prefix == "monica"


def test_configure_local_fix_only_parser_accepts_non_secret_rollout_values():
    parser = ArgumentParser()

    cli.register_cli(parser)
    args = parser.parse_args(
        [
            "configure-local-fix-only",
            "--approver-user-id",
            "U123APPROVER",
            "--repo-url",
            "git@github.com:acme/mobile-app.git",
            "--verification-command",
            "npm test",
            "--repo-local-name",
            "mobile-app",
            "--default-branch",
            "main",
            "--branch-prefix",
            "monica",
        ]
    )

    assert args.mobile_bug_agent_action == "configure_local_fix_only"
    assert args.approver_user_ids == ["U123APPROVER"]
    assert args.repo_url == "git@github.com:acme/mobile-app.git"
    assert args.verification_commands == ["npm test"]
    assert args.repo_local_name == "mobile-app"
    assert args.default_branch == "main"
    assert args.branch_prefix == "monica"


def test_configure_dry_run_persists_non_secret_bootstrap_config(capsys):
    existing_config = {
        "plugins": {
            "enabled": ["other-plugin"],
            "disabled": ["mobile-bug-agent", "legacy-plugin"],
        },
        "mobile_bug_agent": {
            "enabled": False,
            "rollout_mode": "linear_only",
            "dry_run": False,
            "slack": {
                "download_attachments": False,
                "allowed_channels": ["C123EXISTING"],
            },
            "runtime": {
                "worker_session_prefix": "not-monica",
                "skip_memory": False,
            },
        },
    }
    saved: list[dict] = []

    exit_code = cli.run_configure_dry_run_command(
        bot_user_ids=("U123MONICA",),
        channel_ids=("C123PRIVATE", "G456PRIVATE"),
        load_config_fn=lambda: copy.deepcopy(existing_config),
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Configured Monica for dry_run rollout" in out
    assert len(saved) == 1
    saved_config = saved[0]
    assert saved_config["plugins"]["enabled"] == ["mobile-bug-agent", "other-plugin"]
    assert saved_config["plugins"]["disabled"] == ["legacy-plugin"]
    monica = saved_config["mobile_bug_agent"]
    assert monica["enabled"] is True
    assert monica["rollout_mode"] == "dry_run"
    assert monica["dry_run"] is True
    assert monica["slack"]["download_attachments"] is False
    assert monica["slack"]["bot_user_ids"] == ["U123MONICA"]
    assert monica["slack"]["allowed_channels"] == ["C123PRIVATE", "G456PRIVATE"]
    assert monica["loop"]["create_linear"] is True
    assert monica["loop"]["require_fix_approval"] is True
    assert monica["runtime"]["home_subdir"] == "agents/monica"
    assert monica["runtime"]["worker_session_prefix"] == "monica"
    assert monica["runtime"]["skip_memory"] is True
    assert "SLACK_BOT_TOKEN" not in str(saved_config)
    assert "LINEAR_API_KEY" not in str(saved_config)


def test_configure_dry_run_keeps_existing_slack_scope_when_not_provided(capsys):
    existing_config = {
        "plugins": {
            "enabled": ["mobile-bug-agent"],
        },
        "mobile_bug_agent": {
            "slack": {
                "bot_user_ids": ["U123EXISTING"],
                "allowed_channels": ["C123EXISTING"],
            },
        },
    }
    saved: list[dict] = []

    exit_code = cli.run_configure_dry_run_command(
        load_config_fn=lambda: copy.deepcopy(existing_config),
        save_config_fn=lambda config: saved.append(config),
    )

    assert exit_code == 0
    monica = saved[0]["mobile_bug_agent"]
    assert monica["slack"]["bot_user_ids"] == ["U123EXISTING"]
    assert monica["slack"]["allowed_channels"] == ["C123EXISTING"]


def test_configure_dry_run_rejects_handles_and_channel_names_without_saving(capsys):
    saved: list[dict] = []

    exit_code = cli.run_configure_dry_run_command(
        bot_user_ids=("@monica",),
        channel_ids=("#mobile-bugs",),
        load_config_fn=lambda: {},
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert saved == []
    assert "Slack mention user IDs like U012ABCDEF, not handles like @monica" in out
    assert "Slack channel IDs like C123 or G123, not names like #mobile-bugs" in out
    assert "Monica dry_run configuration was not saved." in out


def test_configure_linear_only_persists_non_secret_rollout_config(capsys):
    existing_config = {
        "plugins": {
            "enabled": ["other-plugin"],
            "disabled": ["mobile-bug-agent", "legacy-plugin"],
        },
        "mobile_bug_agent": {
            "slack": {
                "download_attachments": False,
            },
            "loop": {
                "max_thread_messages": 12,
            },
        },
    }
    saved: list[dict] = []

    exit_code = run_configure_linear_only_command(
        bot_user_ids=("U123MONICA",),
        channel_ids=("C123MOBILE", "G456PRIVATE"),
        linear_team_id="team-id",
        linear_project_id="project-id",
        linear_label_ids=("label-one", "label-two"),
        load_config_fn=lambda: copy.deepcopy(existing_config),
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Configured Monica for linear_only rollout" in out
    assert len(saved) == 1
    saved_config = saved[0]
    assert saved_config["plugins"]["enabled"] == ["mobile-bug-agent", "other-plugin"]
    assert saved_config["plugins"]["disabled"] == ["legacy-plugin"]
    monica = saved_config["mobile_bug_agent"]
    assert monica["enabled"] is True
    assert monica["rollout_mode"] == "linear_only"
    assert monica["dry_run"] is False
    assert monica["slack"]["bot_user_ids"] == ["U123MONICA"]
    assert monica["slack"]["allowed_channels"] == ["C123MOBILE", "G456PRIVATE"]
    assert monica["slack"]["download_attachments"] is False
    assert monica["loop"]["create_linear"] is True
    assert monica["loop"]["max_thread_messages"] == 12
    assert monica["linear"]["team_id"] == "team-id"
    assert monica["linear"]["project_id"] == "project-id"
    assert monica["linear"]["label_ids"] == ["label-one", "label-two"]
    assert "SLACK_BOT_TOKEN" not in str(saved_config)
    assert "LINEAR_API_KEY" not in str(saved_config)


def test_configure_approved_pr_persists_non_secret_full_loop_config(capsys):
    existing_config = {
        "plugins": {
            "enabled": ["mobile-bug-agent"],
        },
        "mobile_bug_agent": {
            "enabled": True,
            "rollout_mode": "linear_only",
            "slack": {
                "bot_user_ids": ["U123MONICA"],
                "allowed_channels": ["C123MOBILE"],
            },
            "linear": {
                "team_id": "team-id",
            },
            "runtime": {
                "home_subdir": "agents/monica",
                "worker_session_prefix": "not-monica",
                "skip_memory": False,
            },
            "worker": {
                "backend": "codex_cli",
                "codex_command": "codex",
                "codex_model": "gpt-5-codex",
                "codex_profile": "monica",
                "codex_sandbox": "danger-full-access",
                "codex_approval_policy": "on-request",
                "timeout_minutes": 99,
            },
        },
    }
    saved: list[dict] = []

    exit_code = run_configure_approved_pr_command(
        approver_user_ids=("U123APPROVER",),
        repo_url="git@github.com:acme/mobile-app.git",
        verification_commands=("npm test", "npm run lint"),
        repo_local_name="mobile-app",
        default_branch="main",
        branch_prefix="monica",
        load_config_fn=lambda: copy.deepcopy(existing_config),
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Configured Monica for approved_pr rollout" in out
    assert len(saved) == 1
    monica = saved[0]["mobile_bug_agent"]
    assert monica["enabled"] is True
    assert monica["rollout_mode"] == "approved_pr"
    assert monica["dry_run"] is False
    assert monica["loop"]["create_linear"] is True
    assert monica["loop"]["require_fix_approval"] is True
    assert monica["slack"]["approver_user_ids"] == ["U123APPROVER"]
    assert monica["slack"]["bot_user_ids"] == ["U123MONICA"]
    assert monica["linear"]["team_id"] == "team-id"
    assert monica["repo"]["url"] == "git@github.com:acme/mobile-app.git"
    assert monica["repo"]["local_name"] == "mobile-app"
    assert monica["repo"]["default_branch"] == "main"
    assert monica["repo"]["branch_prefix"] == "monica"
    assert monica["verification"]["commands"] == ["npm test", "npm run lint"]
    assert monica["runtime"]["worker_session_prefix"] == "monica"
    assert monica["runtime"]["skip_memory"] is True
    assert monica["worker"]["backend"] == "codex_cli"
    assert monica["worker"]["codex_command"] == "codex"
    assert monica["worker"]["codex_model"] == "gpt-5-codex"
    assert monica["worker"]["codex_profile"] == "monica"
    assert monica["worker"]["codex_sandbox"] == "workspace-write"
    assert monica["worker"]["codex_approval_policy"] == "never"
    assert monica["worker"]["timeout_minutes"] == 99
    assert saved[0]["plugins"]["enabled"] == ["mobile-bug-agent"]
    assert "GITHUB_TOKEN" not in str(saved[0])
    assert "SLACK_BOT_TOKEN" not in str(saved[0])


def test_configure_local_fix_only_persists_non_secret_local_code_config(capsys):
    existing_config = {
        "plugins": {
            "enabled": ["mobile-bug-agent"],
        },
        "mobile_bug_agent": {
            "enabled": True,
            "rollout_mode": "linear_only",
            "slack": {
                "bot_user_ids": ["U123MONICA"],
                "allowed_channels": ["D123MONICA"],
            },
            "linear": {
                "team_id": "team-id",
                "label_ids": ["bug-label"],
            },
            "worker": {
                "backend": "codex_cli",
                "codex_command": "codex",
                "codex_sandbox": "danger-full-access",
                "codex_approval_policy": "on-request",
            },
        },
    }
    saved: list[dict] = []

    exit_code = run_configure_local_fix_only_command(
        approver_user_ids=("U123APPROVER",),
        repo_url="git@github.com:acme/mobile-app.git",
        verification_commands=("npm test", "npm run lint"),
        repo_local_name="mobile-app",
        default_branch="main",
        branch_prefix="monica",
        load_config_fn=lambda: copy.deepcopy(existing_config),
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Configured Monica for local_fix_only rollout" in out
    assert len(saved) == 1
    monica = saved[0]["mobile_bug_agent"]
    assert monica["enabled"] is True
    assert monica["rollout_mode"] == "local_fix_only"
    assert monica["dry_run"] is False
    assert monica["loop"]["create_linear"] is True
    assert monica["loop"]["require_fix_approval"] is True
    assert monica["slack"]["approver_user_ids"] == ["U123APPROVER"]
    assert monica["slack"]["bot_user_ids"] == ["U123MONICA"]
    assert monica["slack"]["allowed_channels"] == ["D123MONICA"]
    assert monica["linear"]["team_id"] == "team-id"
    assert monica["linear"]["label_ids"] == ["bug-label"]
    assert monica["repo"]["url"] == "git@github.com:acme/mobile-app.git"
    assert monica["repo"]["local_name"] == "mobile-app"
    assert monica["repo"]["default_branch"] == "main"
    assert monica["repo"]["branch_prefix"] == "monica"
    assert monica["verification"]["commands"] == ["npm test", "npm run lint"]
    assert monica["worker"]["codex_sandbox"] == "workspace-write"
    assert monica["worker"]["codex_approval_policy"] == "never"
    assert "GITHUB_TOKEN" not in str(saved[0])


def test_configure_linear_only_rejects_handles_and_channel_names_without_saving(capsys):
    saved: list[dict] = []

    exit_code = run_configure_linear_only_command(
        bot_user_ids=("@monica",),
        channel_ids=("#mobile-bugs",),
        linear_team_id="team-id",
        load_config_fn=lambda: {},
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert saved == []
    assert "Slack mention user IDs like U012ABCDEF, not handles like @monica" in out
    assert "Slack channel IDs like C123 or G123, not names like #mobile-bugs" in out


def test_configure_approved_pr_rejects_unsafe_values_without_saving(capsys):
    saved: list[dict] = []

    exit_code = run_configure_approved_pr_command(
        approver_user_ids=("@ritik",),
        repo_url="",
        verification_commands=(),
        repo_local_name="../mobile-app",
        default_branch="main branch",
        branch_prefix="chandler",
        load_config_fn=lambda: {},
        save_config_fn=lambda config: saved.append(config),
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert saved == []
    assert "Slack user IDs like U123, not handles like @ritik" in out
    assert "--repo-url is required" in out
    assert "at least one --verification-command is required" in out
    assert "repo.local_name must be a simple directory name" in out
    assert "repo.branch_prefix must not point at Chandler" in out
    assert "repo.default_branch must be a safe git branch name" in out


def test_doctor_allows_dry_run_with_slack_token(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("UMONICA",), allowed_channels=("C123",)),
        ),
        environ={"MONICA_SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Monica doctor: ready" in out
    assert "Monica runtime root:" in out


def test_doctor_requires_monica_token_even_when_chandler_slack_token_exists(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("UMONICA",), allowed_channels=("C123",)),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-chandler"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: MONICA_SLACK_BOT_TOKEN is missing" in out
    assert "xoxb-chandler" not in out
    assert "Monica doctor: not ready" in out


def test_doctor_preflights_target_rollout_without_mutating_config(capsys):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="dry_run",
        dry_run=True,
        slack=SlackConfig(
            bot_user_ids=("U123MONICA",),
            allowed_channels=("C123MOBILE",),
        ),
    )

    exit_code = run_doctor_command(
        config=config,
        target_rollout_mode="linear_only",
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin_api_token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda _name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert config.rollout_mode == "dry_run"
    assert config.dry_run is True
    assert "Monica rollout mode: linear_only" in out
    assert "FAIL: linear.team_id is missing" in out
    assert "Monica doctor: not ready" in out


def test_doctor_json_outputs_machine_readable_readiness(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("U123MONICA",), allowed_channels=("C123MOBILE",)),
        ),
        json_output=True,
        environ={"MONICA_SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: name != "slack_sdk",
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ready"] is False
    assert payload["rollout_mode"] == "dry_run"
    assert payload["runtime_root"]
    assert payload["warnings"] == [
        {
            "code": "slack_app_token",
            "message": "MONICA_SLACK_APP_TOKEN is missing; Monica Slack Socket Mode may not start",
        }
    ]
    assert payload["failures"] == [
        {
            "code": "slack_sdk",
            "message": "slack-sdk Python package is not installed; install `hermes-agent[slack]` or `hermes-agent[messaging]`",
        }
    ]
    assert "xoxb-token" not in json.dumps(payload)


def test_doctor_fails_when_slack_sdk_is_missing(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("U_MONICA",), allowed_channels=("C123",)),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: name != "slack_sdk",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack-sdk Python package is not installed; install "
        "`hermes-agent[slack]` or `hermes-agent[messaging]`"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_reports_invalid_runtime_root_without_traceback(capsys):
    exit_code = run_doctor_command(
        config=config_from_mapping(
            {
                "enabled": True,
                "rollout_mode": "dry_run",
                "runtime": {"home_subdir": "../outside-runtime"},
                "slack": {"bot_user_ids": ["U_MONICA"], "allowed_channels": ["C123"]},
            }
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: mobile_bug_agent.runtime.home_subdir must stay inside HERMES_HOME." in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_dry_run_with_bot_id_mentions(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("B012ABCDEF",), allowed_channels=("C123",)),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
        "not bot_id values like B012ABCDEF"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_dry_run_with_handle_style_bot_user_id(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("@monica",), allowed_channels=("C123",)),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
        "not handles like @monica"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_dry_run_with_malformed_bot_user_id(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("monica",), allowed_channels=("C123",)),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
        "not invalid values like monica"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_side_effect_rollouts_with_bot_id_mentions(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(bot_user_ids=("B012ABCDEF",), allowed_channels=("C123",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
        "not bot_id values like B012ABCDEF"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_side_effect_rollouts_with_handle_style_bot_user_id(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(bot_user_ids=("@monica",), allowed_channels=("C123",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
        "not handles like @monica"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_side_effect_rollouts_with_malformed_bot_user_id(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(bot_user_ids=("monica",), allowed_channels=("C123",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.bot_user_ids must contain Slack mention user IDs like U012ABCDEF, "
        "not invalid values like monica"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_side_effect_rollouts_without_bot_user_ids(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(allowed_channels=("C123",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: slack.bot_user_ids is empty; configure Monica's Slack mention user ID" in out
    assert "Monica doctor: not ready" in out


def test_doctor_fails_when_provided_slack_scopes_are_missing_required_post_scope(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("U_MONICA",), allowed_channels=("C123",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_BOT_SCOPES": "app_mentions:read,channels:history",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: SLACK_BOT_SCOPES is missing required scope: chat:write" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_without_repo_verification_and_tools(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(bot_user_ids=("BMONICA",), allowed_channels=("C123",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token", "LINEAR_API_KEY": "lin-key"},
        which=lambda name: None,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: repo.url is missing" in out
    assert "FAIL: verification.commands is empty" in out
    assert "FAIL: git executable was not found" in out
    assert "FAIL: gh executable was not found" in out
    assert "FAIL: codex executable was not found" in out


def test_doctor_blocks_approved_pr_without_configured_approvers(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(bot_user_ids=("U_MONICA",), allowed_channels=("C123",)),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: slack.approver_user_ids is empty; configure at least one Monica code approver" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_unsafe_repo_local_name(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git", local_name="../outside-runtime"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: repo.local_name must be a simple directory name." in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_chandler_repo_local_name(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git", local_name="chandler"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: repo.local_name must not point at a Chandler directory." in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_unsafe_repo_branch_prefix(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(
                url="git@github.com:acme/mobile.git",
                branch_prefix="../chandler",
            ),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: repo.branch_prefix must be a safe git branch prefix." in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_chandler_repo_branch_prefix(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(
                url="git@github.com:acme/mobile.git",
                branch_prefix="chandler",
            ),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: repo.branch_prefix must not point at Chandler." in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_unsafe_repo_default_branch(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(
                url="git@github.com:acme/mobile.git",
                default_branch="../main",
            ),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: repo.default_branch must be a safe git branch name." in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_handle_style_approver(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("@ritik",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: slack.approver_user_ids must contain Slack user IDs like U123, not handles like @ritik" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_malformed_approver_user_id(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("ritik",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "FAIL: slack.approver_user_ids must contain Slack user IDs like U123, "
        "not invalid values like ritik"
    ) in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_interactive_codex_approval_policy(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
            worker=WorkerConfig(codex_approval_policy="on-request"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: worker.codex_approval_policy must be `never` for approved_pr codex_cli runs" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_with_non_workspace_codex_sandbox(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
            worker=WorkerConfig(codex_sandbox="danger-full-access"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: worker.codex_sandbox must be `workspace-write` for approved_pr codex_cli runs" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_internal_worker_with_chandler_session_prefix(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            runtime=RuntimeConfig(worker_session_prefix="chandler"),
            verification=VerificationConfig(commands=("npm test",)),
            worker=WorkerConfig(backend="internal_agent"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: runtime.worker_session_prefix must include `monica` to keep worker sessions segregated" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_chandler_runtime_home_subdir(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            runtime=RuntimeConfig(home_subdir="agents/chandler"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: mobile_bug_agent.runtime.home_subdir must not point at a Chandler runtime path" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_when_linear_creation_is_disabled(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            loop=LoopConfig(create_linear=False),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: loop.create_linear must be true in approved_pr mode" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_approved_pr_when_fix_approval_gate_is_disabled(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            loop=LoopConfig(require_fix_approval=False),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: loop.require_fix_approval must be true in approved_pr mode" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_side_effect_rollouts_without_allowed_channels(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(bot_user_ids=("U_MONICA",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: slack.allowed_channels is empty; configure the Slack channels Monica may act in" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_dry_run_with_slack_channel_names(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="dry_run",
            slack=SlackConfig(bot_user_ids=("U_MONICA",), allowed_channels=("#mobile-bugs",)),
        ),
        environ={"SLACK_BOT_TOKEN": "xoxb-token"},
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: slack.allowed_channels must contain Slack channel IDs like C123 or G123, not names like #mobile-bugs" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_side_effect_rollouts_with_slack_channel_names(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(bot_user_ids=("U_MONICA",), allowed_channels=("#mobile-bugs",)),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: slack.allowed_channels must contain Slack channel IDs like C123 or G123, not names like #mobile-bugs" in out
    assert "Monica doctor: not ready" in out


def test_doctor_blocks_linear_only_when_linear_creation_is_disabled(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="linear_only",
            slack=SlackConfig(bot_user_ids=("U_MONICA",), allowed_channels=("C123",)),
            loop=LoopConfig(create_linear=False),
            linear=LinearConfig(team_id="team-id"),
        ),
        environ={
            "SLACK_BOT_TOKEN": "xoxb-token",
            "SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: loop.create_linear must be true in linear_only mode" in out
    assert "Monica doctor: not ready" in out


def test_doctor_accepts_approved_pr_when_required_setup_exists(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("C123",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
            "GITHUB_TOKEN": "gh-token",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Monica doctor: ready" in out


def test_doctor_accepts_local_fix_only_without_github_push_or_pr_tools(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="local_fix_only",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("D123MONICA",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
        ),
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: None if name == "gh" else f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "gh executable was not found" not in out
    assert "GITHUB_TOKEN is missing" not in out
    assert "Monica doctor: ready" in out


def test_doctor_warns_when_required_proof_has_no_commands(capsys):
    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="local_fix_only",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("D123MONICA",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
            proof=ProofConfig(enabled=True, required_for_done=True),
        ),
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "WARN: proof.commands is empty; Monica will block at proof after verification" in out
    assert "Monica doctor: ready" in out


def test_doctor_warns_when_required_proof_tooling_is_missing(capsys):
    missing = {"xcrun", "xcodebuild", "adb", "emulator"}

    exit_code = run_doctor_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="local_fix_only",
            slack=SlackConfig(
                bot_user_ids=("U_MONICA",),
                allowed_channels=("D123MONICA",),
                approver_user_ids=("U_APPROVER",),
            ),
            linear=LinearConfig(team_id="team-id"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
            proof=ProofConfig(
                enabled=True,
                required_for_done=True,
                commands=("npm run monica:proof",),
                platform_order=("ios", "android"),
            ),
        ),
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin-key",
        },
        which=lambda name: None if name in missing else f"/usr/bin/{name}",
        module_available=lambda name: True,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "WARN: iOS proof is configured but xcrun/simctl or xcodebuild was not found; "
        "Monica will block at proof until a simulator is available"
    ) in out
    assert (
        "WARN: Android proof is configured but adb or emulator was not found; "
        "Monica will block at proof until an emulator is available"
    ) in out
    assert "Monica doctor: ready" in out
