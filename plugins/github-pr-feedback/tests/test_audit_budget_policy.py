from __future__ import annotations

import argparse
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from github_pr_feedback.ci_runner import CIAuditReceipt
from github_pr_feedback.github_client import CheckState, PullRequestMergeState


@pytest.mark.parametrize("actions_enabled", [False, True])
def test_audit_pr_passes_fresh_canonical_actions_state_to_exact_head_runner(
    monkeypatch, tmp_path, actions_enabled: bool
) -> None:
    from github_pr_feedback.ci_runner import CIAuditIdentity
    from github_pr_feedback.cli import _audit_pr

    head_sha = "a" * 40
    base_sha = "b" * 40
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    policy = SimpleNamespace(
        local_ci_audit=SimpleNamespace(audit_only=True, required_for_open_prs=False),
        targets={"acme/widgets": object()},
        uses_budget_exhausted_local_ci=lambda repository: repository == "acme/widgets",
    )
    state = PullRequestMergeState(
        repository="acme/widgets",
        number=17,
        state="OPEN",
        is_draft=False,
        mergeable=True,
        merge_state_status="CLEAN",
        base_branch="stable",
        base_sha=base_sha,
        head_repository="acme/widgets",
        author_login="owner",
        head_ref_name="codex/fix",
        head_sha=head_sha,
        merged=False,
        merge_commit_oid=None,
    )

    class GitHub:
        def get_merge_state(self, _repository: str, _number: int):
            return state

        def get_pull_request(self, _repository: str, _number: int):
            return state

        def actions_enabled(self, repository: str, *, refresh: bool = False) -> bool:
            assert repository == "acme/widgets"
            assert refresh is True
            return actions_enabled

    class Ledger:
        def has_pending_mutation(self, _repository: str, _number: int) -> bool:
            return False

        def close(self) -> None:
            pass

    captured: list[bool | None] = []

    def run_audit(
        _github,
        _ledger,
        identity: CIAuditIdentity,
        _worktree,
        *,
        force_fresh: bool = False,
        actions_enabled_hint: bool | None = None,
        required_local_ci: bool = False,
    ) -> CIAuditReceipt:
        assert force_fresh is True
        captured.append(actions_enabled_hint)
        return CIAuditReceipt(
            receipt_id="r" * 64,
            identity=identity,
            manifest_digest="m" * 64,
            status="passed",
            started_at=datetime(2026, 9, 3, tzinfo=UTC),
            completed_at=datetime(2026, 9, 3, tzinfo=UTC),
            actions_state=CheckState(
                actions_enabled=actions_enabled,
                all_green=not actions_enabled,
                check_count=1 if actions_enabled else 0,
                billing_blocked=actions_enabled,
            ),
            commands=(),
            ci_mode=(
                "budget-exhausted-local-equivalent" if actions_enabled else "standard"
            ),
        )

    monkeypatch.setattr("github_pr_feedback.cli._load_policy_from_context", lambda _ctx: policy)
    monkeypatch.setattr("github_pr_feedback.cli._github_client", lambda _policy: GitHub())
    monkeypatch.setattr(
        "github_pr_feedback.cli.FeedbackLedger.for_current_profile", lambda: Ledger()
    )
    monkeypatch.setattr("github_pr_feedback.cli._run_grouped_exact_head_audit", run_audit)
    monkeypatch.setattr("github_pr_feedback.cli._complete_current_ci_task", lambda _receipt: None)

    result = _audit_pr(
        object(),
        argparse.Namespace(
            repository="acme/widgets",
            pr_number=17,
            head_sha=head_sha,
            worktree=str(worktree),
            fresh=True,
        ),
    )

    assert result == 0
    assert captured == [actions_enabled]
