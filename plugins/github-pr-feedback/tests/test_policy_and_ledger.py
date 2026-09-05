from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
from types import SimpleNamespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from github_pr_feedback.ledger import ClaimLease, FeedbackLedger
from github_pr_feedback.policy import (
    FeedbackReceipt,
    MergeMaintainerPolicy,
    PostMergePolicy,
    PullRequest,
    Reviewer,
    _is_git_worktree,
    load_policy,
)


def configured_policy(tmp_path: Path):
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    return load_policy(
        {
            "enabled": True,
            "repositories": [
                {
                    "base_repository": "acme/widgets",
                    "head_repository": "acme/widgets",
                    "local_path": str(repository_path),
                    "owner_login": "owner",
                    "branch_prefixes": ["codex/", "fix/"],
                }
            ],
            "reviewer_logins": ["trusted-reviewer"],
            "reviewer_associations": ["MEMBER"],
            "not_before": "2026-08-24T00:00:00Z",
            "assignee": "repair-agent",
            "board": "repairs",
        }
    )


def initialize_git_worktree(path: Path) -> None:
    subprocess.run(
        ["git", "init", "--quiet", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def enabled_raw_config(local_path: Path) -> dict[str, object]:
    return {
        "enabled": True,
        "repositories": [
            {
                "base_repository": "acme/widgets",
                "head_repository": "acme/widgets",
                "local_path": str(local_path),
                "owner_login": "owner",
                "branch_prefixes": ["codex/"],
            }
        ],
        "reviewer_logins": ["trusted-reviewer"],
        "reviewer_associations": [],
        "not_before": "2026-08-24T00:00:00Z",
        "assignee": "repair-agent",
        "board": "repairs",
    }


def enabled_merge_config(
    repository_path: Path, deployment_path: Path
) -> dict[str, object]:
    raw = enabled_raw_config(repository_path)
    raw["merge_maintainer"] = {
        "enabled": True,
        "assignee": "pr-merge-maintainer",
        "repository": "acme/widgets",
        "author_login": "owner",
        "base_branch": "stable",
        "merge_methods": ["squash", "rebase", "merge"],
        "receipt_max_age_seconds": 21600,
        "report_only": False,
        "post_merge": {
            "enabled": True,
            "deployment_path": str(deployment_path),
            "protected_runtime_entry": "main.py",
            "package_argv": [
                "python3",
                "tools/tb.py",
                "gui-package-macos",
                "--replace",
                "--json",
            ],
            "bundle_path": "desktop/macos/Example/build/Example.app",
            "bundle_identifier": "com.example.local.operator",
            "relaunch_argv": ["/usr/bin/open", "-n"],
        },
    }
    return raw


def enabled_release_maintenance_config(repository_path: Path) -> dict[str, object]:
    raw = enabled_raw_config(repository_path)
    raw["release_maintenance"] = {
        "enabled": True,
        "assignee": "release-maintenance-steward",
        "repository": "acme/widgets",
        "base_branch": "stable",
        "quiet_period_seconds": 900,
        "max_runtime_seconds": 7200,
        "lanes": [
            {
                "name": "unit-tests",
                "assignee": "test-contract-steward",
                "command": ["python3", "-m", "pytest", "-q"],
            },
            {
                "name": "static-analysis",
                "assignee": "code-quality-steward",
                "command": ["python3", "tools/check_static.py"],
            },
        ],
    }
    return raw


def test_enabled_policy_parses_release_maintenance_lane_matrix(tmp_path: Path) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)

    policy = load_policy(enabled_release_maintenance_config(repository_path))

    assert policy.release_maintenance.assignee == "release-maintenance-steward"
    assert policy.release_maintenance.repository == "acme/widgets"
    assert policy.release_maintenance.base_branch == "stable"
    assert policy.release_maintenance.quiet_period_seconds == 900
    assert policy.release_maintenance.max_runtime_seconds == 7200
    assert [lane.name for lane in policy.release_maintenance.lanes] == [
        "unit-tests",
        "static-analysis",
    ]
    assert policy.release_maintenance.lanes[0].command == (
        "python3",
        "-m",
        "pytest",
        "-q",
    )


def test_release_maintenance_rejects_a_protected_runtime_command(tmp_path: Path) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_release_maintenance_config(repository_path)
    raw["release_maintenance"]["lanes"][0]["command"] = ["python3", "main.py"]

    with pytest.raises(ValueError, match="protected runtime"):
        load_policy(raw)


def test_maintenance_head_quiet_clock_resets_only_when_base_head_changes(
    tmp_path: Path,
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    first = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    assert (
        ledger.observe_maintenance_head(
            "acme/widgets", "stable", "a" * 40, observed_at=first
        )
        == first
    )
    assert (
        ledger.observe_maintenance_head(
            "acme/widgets", "stable", "a" * 40, observed_at=first + timedelta(minutes=5)
        )
        == first
    )
    changed = ledger.observe_maintenance_head(
        "acme/widgets", "stable", "b" * 40, observed_at=first + timedelta(minutes=6)
    )

    assert changed == first + timedelta(minutes=6)


def test_maintenance_receipts_are_exact_head_and_lane_scoped(tmp_path: Path) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    completed_at = datetime(2026, 8, 25, 12, 30, tzinfo=UTC)

    ledger.record_maintenance_receipt(
        repository="acme/widgets",
        head_sha="a" * 40,
        lane="unit-tests",
        status="failed",
        summary="3 tests failed",
        completed_at=completed_at,
    )

    receipts = ledger.maintenance_receipts("acme/widgets", "a" * 40)
    assert receipts["unit-tests"].status == "failed"
    assert receipts["unit-tests"].summary == "3 tests failed"
    assert ledger.maintenance_receipts("acme/widgets", "b" * 40) == {}


def test_merge_maintainer_is_disabled_by_default(tmp_path: Path) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)

    policy = load_policy(enabled_raw_config(repository_path))

    assert policy.merge_maintainer is None


def test_enabled_policy_parses_strict_merge_and_post_merge_settings(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    deployment_path = tmp_path / "deployment"
    initialize_git_worktree(repository_path)
    initialize_git_worktree(deployment_path)

    policy = load_policy(enabled_merge_config(repository_path, deployment_path))

    assert policy.merge_maintainer == MergeMaintainerPolicy(
        assignee="pr-merge-maintainer",
        repository="acme/widgets",
        author_login="owner",
        base_branch="stable",
        merge_methods=("squash", "rebase", "merge"),
        receipt_max_age_seconds=21600,
        report_only=False,
        post_merge=PostMergePolicy(
            deployment_path=deployment_path.resolve(),
            protected_runtime_entry="main.py",
            package_argv=(
                "python3",
                "tools/tb.py",
                "gui-package-macos",
                "--replace",
                "--json",
            ),
            bundle_path="desktop/macos/Example/build/Example.app",
            bundle_identifier="com.example.local.operator",
            relaunch_argv=("/usr/bin/open", "-n"),
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda merge: merge.update({"unknown": True}),
        lambda merge: merge.update({"repository": "other/widgets"}),
        lambda merge: merge.update({"author_login": "someone-else"}),
        lambda merge: merge.update({"receipt_max_age_seconds": 0}),
        lambda merge: merge.update({"merge_method": "merge"}),
        lambda merge: merge.update({"merge_methods": ["squash", "octopus"]}),
        lambda merge: merge.update({"merge_methods": ["squash", "squash"]}),
        lambda merge: merge.update({"post_merge": {"deployment_path": "/tmp"}}),
        lambda merge: merge["post_merge"].update({"unknown": True}),
        lambda merge: merge["post_merge"].update(
            {"protected_runtime_entry": "/main.py"}
        ),
        lambda merge: merge["post_merge"].update({"bundle_path": "../Example.app"}),
        lambda merge: merge["post_merge"].update(
            {"package_argv": "python3 tools/tb.py"}
        ),
        lambda merge: merge["post_merge"].update({"relaunch_argv": []}),
    ],
)
def test_enabled_policy_rejects_unsafe_merge_maintainer_settings(
    tmp_path: Path, mutation
) -> None:
    repository_path = tmp_path / "widgets"
    deployment_path = tmp_path / "deployment"
    initialize_git_worktree(repository_path)
    initialize_git_worktree(deployment_path)
    raw = enabled_merge_config(repository_path, deployment_path)
    mutation(raw["merge_maintainer"])

    with pytest.raises(ValueError):
        load_policy(raw)


def test_disabled_post_merge_hook_requires_only_explicit_enabled_flag(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    deployment_path = tmp_path / "deployment"
    initialize_git_worktree(repository_path)
    initialize_git_worktree(deployment_path)
    raw = enabled_merge_config(repository_path, deployment_path)
    raw["merge_maintainer"]["post_merge"] = {"enabled": False}

    policy = load_policy(raw)

    assert policy.merge_maintainer is not None
    assert policy.merge_maintainer.post_merge is None


def test_policy_can_explicitly_admit_owner_and_bot_feedback_without_widening_human_reviewers(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["include_self_feedback"] = True
    raw["include_bot_feedback"] = True
    policy = load_policy(raw)

    assert policy.admit(admitted_pr(), Reviewer("owner", None), receipt()).admitted
    assert policy.admit(
        admitted_pr(),
        Reviewer("review-app[bot]", "NONE"),
        receipt(),
        is_bot=True,
    ).admitted
    denied = policy.admit(admitted_pr(), Reviewer("stranger", "NONE"), receipt())
    assert not denied.admitted
    assert denied.reason == "reviewer_not_allowed"


def admitted_pr(**overrides: object) -> PullRequest:
    values: dict[str, object] = {
        "number": 17,
        "state": "OPEN",
        "base_repository": "acme/widgets",
        "head_repository": "acme/widgets",
        "author_login": "owner",
        "head_ref_name": "codex/fix-ledger",
        "head_sha": "a" * 40,
    }
    values.update(overrides)
    return PullRequest(**values)


def receipt(**overrides: object) -> FeedbackReceipt:
    values: dict[str, object] = {
        "repository": "acme/widgets",
        "pr_number": 17,
        "feedback_kind": "issue_comment",
        "feedback_id": "comment-42",
        "head_sha": "a" * 40,
    }
    values.update(overrides)
    return FeedbackReceipt(**values)


def claim_lease(
    ledger: FeedbackLedger,
    item: FeedbackReceipt,
    *,
    owner: str = "test-scanner",
    claimed_at: datetime | None = None,
) -> ClaimLease | None:
    now = claimed_at or datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    return ledger.claim(
        item,
        owner=owner,
        claimed_at=now,
        stale_before=now - timedelta(minutes=5),
    )


def test_disabled_config_is_not_admitted(tmp_path: Path) -> None:
    policy = load_policy({"enabled": False})

    assert (
        policy.admit(
            admitted_pr(), Reviewer("trusted-reviewer", "MEMBER"), receipt()
        ).reason
        == "disabled"
    )


def test_dispatched_feedback_is_not_actioned_until_explicit_exact_head_acknowledgement(
    tmp_path: Path,
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    item = receipt()
    lease = claim_lease(ledger, item)
    assert lease is not None
    ledger.finalize(item, "task-17", lease)

    assert ledger.was_completed_on_any_head(item) is True
    assert ledger.was_actioned_on_any_head(item) is False

    ledger.mark_feedback_actioned(
        item,
        resolved_head_sha="b" * 40,
        actioned_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )

    assert ledger.was_actioned_on_any_head(item) is True
    ledger.close()


@pytest.mark.parametrize(
    "raw",
    [
        {"enabled": True},
        {"enabled": True, "repositories": []},
        {
            "enabled": True,
            "repositories": [
                {
                    "base_repository": "acme/widgets",
                    "head_repository": "acme/widgets",
                    "local_path": "relative/path",
                    "owner_login": "owner",
                    "branch_prefixes": ["codex/"],
                }
            ],
            "reviewer_logins": [],
            "reviewer_associations": [],
            "not_before": "2026-08-24T00:00:00Z",
            "assignee": "repair-agent",
            "board": "repairs",
        },
        {
            "enabled": True,
            "repositories": [
                {
                    "base_repository": "acme/widgets",
                    "head_repository": "acme/widgets",
                    "local_path": "/not/a/repository",
                    "owner_login": "owner",
                    "branch_prefixes": ["codex/"],
                }
            ],
            "reviewer_logins": "trusted-reviewer",
            "reviewer_associations": ["MEMBER"],
            "assignee": "repair-agent",
            "board": "repairs",
        },
        {
            "enabled": True,
            "repositories": [
                {
                    "base_repository": "acme/widgets",
                    "head_repository": "acme/widgets",
                    "local_path": "/not/a/repository",
                    "owner_login": "owner",
                    "branch_prefixes": ["codex/"],
                }
            ],
            "reviewer_logins": ["trusted-reviewer"],
            "reviewer_associations": [],
            "not_before": "2026-08-24T00:00:00Z",
            "assignee": "repair-agent",
            "board": "repairs",
        },
    ],
)
def test_enabled_incomplete_config_fails_closed(raw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        load_policy(raw)


def test_enabled_config_rejects_empty_string_reviewer_list(tmp_path: Path) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["reviewer_logins"] = ""
    raw["reviewer_associations"] = ["MEMBER"]

    with pytest.raises(ValueError):
        load_policy(raw)


@pytest.mark.parametrize("not_before", [None, "", "2026-08-24T00:00:00", "not-a-time"])
def test_enabled_policy_requires_a_timezone_aware_iso8601_intake_boundary(
    tmp_path: Path, not_before: object
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    if not_before is None:
        del raw["not_before"]
    else:
        raw["not_before"] = not_before

    with pytest.raises(ValueError):
        load_policy(raw)


def test_enabled_policy_normalizes_not_before_to_utc(tmp_path: Path) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["not_before"] = "2026-08-23T17:00:00-07:00"

    policy = load_policy(raw)

    assert policy.not_before == datetime(2026, 8, 24, tzinfo=UTC)


def test_enabled_policy_parses_a_bounded_local_ci_audit_lane(tmp_path: Path) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
    }

    policy = load_policy(raw)

    assert policy.local_ci_audit is not None
    assert policy.local_ci_audit.assignee == "pr-local-ci-auditor"
    assert policy.local_ci_audit.post_results is True
    assert policy.local_ci_audit.required_for_open_prs is False
    assert policy.local_ci_audit.max_dispatches_per_scan == 1
    assert policy.local_ci_audit.max_open_prs_per_scan == 300


def test_enabled_policy_parses_bounded_required_local_ci_settings(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["local_ci_audit"] = {
        "enabled": True,
        "assignee": "pr-local-ci-auditor",
        "post_results": True,
        "required_for_open_prs": True,
        "max_dispatches_per_scan": 2,
        "max_open_prs_per_scan": 17,
    }

    policy = load_policy(raw)

    assert policy.local_ci_audit is not None
    assert policy.local_ci_audit.required_for_open_prs is True
    assert policy.local_ci_audit.max_dispatches_per_scan == 2
    assert policy.local_ci_audit.max_open_prs_per_scan == 17


@pytest.mark.parametrize(
    "local_ci_audit",
    [
        True,
        {},
        {"enabled": True, "assignee": "pr-local-ci-auditor"},
        {
            "enabled": True,
            "assignee": "pr-local-ci-auditor",
            "post_results": True,
            "unknown": True,
        },
    ],
)
def test_enabled_policy_rejects_incomplete_or_unknown_local_ci_audit_settings(
    tmp_path: Path, local_ci_audit: object
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["local_ci_audit"] = local_ci_audit

    with pytest.raises(ValueError):
        load_policy(raw)


def test_enabled_policy_parses_bounded_assignee_rules_and_uses_fallback_on_a_tie(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["assignee_rules"] = [
        {"assignee": "performance-specialist", "match_any": ["latency", "performance"]},
        {"assignee": "runtime-specialist", "match_any": ["runtime", "crash"]},
    ]

    policy = load_policy(raw)

    assert (
        policy.assignee_for("Reduce runtime latency and performance overhead")
        == "performance-specialist"
    )
    assert policy.assignee_for("Investigate a runtime crash") == "runtime-specialist"
    assert policy.assignee_for("Documentation typo") == "repair-agent"
    assert policy.assignee_for("Performance and runtime regression") == "repair-agent"


def test_typed_routing_prefers_explicit_risk_precedence_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["routing_rules"] = [
        {
            "assignee": "performance-specialist",
            "precedence": 20,
            "match_any": ["latency"],
            "match_labels_any": ["type/perf"],
            "tags": ["type/perf", "comp/agent"],
            "priority": "P2",
            "blast_radius": "moderate",
            "risks": [],
            "requires_review": False,
        },
        {
            "assignee": "session-state-steward",
            "precedence": 100,
            "match_any": ["resume failed"],
            "match_labels_any": ["sweeper:risk-session-state"],
            "tags": ["type/bug", "area/sessions"],
            "priority": "P1",
            "blast_radius": "broad",
            "risks": ["session-state"],
            "requires_review": True,
        },
    ]

    policy = load_policy(raw)
    decision = policy.route(
        "Latency regressed while resume failed",
        labels=("type/perf", "sweeper:risk-session-state"),
    )

    assert decision.assignee == "session-state-steward"
    assert decision.tags == ("type/bug", "area/sessions")
    assert decision.priority == "P1"
    assert decision.blast_radius == "broad"
    assert decision.risks == ("session-state",)
    assert decision.requires_review is True
    assert decision.ambiguous is False


def test_typed_routing_fails_an_equal_precedence_tie_to_the_fallback_reviewer(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["assignee"] = "task-orchestrator"
    raw["routing_rules"] = [
        {
            "assignee": "runtime-specialist",
            "precedence": 50,
            "match_any": ["crash"],
            "match_labels_any": [],
            "tags": ["type/bug"],
            "priority": "P1",
            "blast_radius": "moderate",
            "risks": [],
            "requires_review": False,
        },
        {
            "assignee": "gateway-specialist",
            "precedence": 50,
            "match_any": ["crash"],
            "match_labels_any": [],
            "tags": ["comp/gateway"],
            "priority": "P1",
            "blast_radius": "moderate",
            "risks": [],
            "requires_review": False,
        },
    ]

    decision = load_policy(raw).route("gateway crash", labels=())

    assert decision.assignee == "task-orchestrator"
    assert decision.ambiguous is True
    assert decision.requires_review is True
    assert decision.tags == ("routing/ambiguous",)


def test_enabled_policy_requires_explicit_opt_in_for_automatic_worker_dispatch(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    default_raw = enabled_raw_config(repository_path)
    opted_in_raw = dict(default_raw)
    opted_in_raw["auto_dispatch"] = True

    default_policy = load_policy(default_raw)
    opted_in_policy = load_policy(opted_in_raw)

    assert getattr(default_policy, "auto_dispatch", None) is False
    assert getattr(opted_in_policy, "auto_dispatch", None) is True


@pytest.mark.parametrize(
    "assignee_rules",
    [
        "performance-specialist",
        [],
        [{"assignee": "performance-specialist"}],
        [{"assignee": "performance-specialist", "match_any": []}],
        [
            {
                "assignee": "performance-specialist",
                "match_any": ["latency"],
                "extra": True,
            }
        ],
    ],
)
def test_enabled_policy_rejects_malformed_assignee_rules(
    tmp_path: Path, assignee_rules: object
) -> None:
    repository_path = tmp_path / "widgets"
    initialize_git_worktree(repository_path)
    raw = enabled_raw_config(repository_path)
    raw["assignee_rules"] = assignee_rules

    with pytest.raises(ValueError):
        load_policy(raw)


def test_completed_feedback_identity_is_detected_across_head_changes(
    tmp_path: Path,
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    prior = receipt(head_sha="a" * 40)
    lease = claim_lease(ledger, prior)
    ledger.finalize(prior, "kanban-1", lease)

    assert ledger.was_completed_on_any_head(receipt(head_sha="b" * 40)) is True
    assert (
        ledger.was_completed_on_any_head(
            receipt(feedback_id="different", head_sha="b" * 40)
        )
        is False
    )
    ledger.close()


@pytest.mark.parametrize(
    ("pr_overrides", "reviewer", "receipt_overrides", "reason"),
    [
        (
            {"base_repository": "acme/other"},
            Reviewer("trusted-reviewer", "MEMBER"),
            {},
            "base_repository_not_allowed",
        ),
        (
            {"head_repository": "fork/widgets"},
            Reviewer("trusted-reviewer", "MEMBER"),
            {},
            "head_repository_not_allowed",
        ),
        (
            {"author_login": "someone-else"},
            Reviewer("trusted-reviewer", "MEMBER"),
            {},
            "author_not_allowed",
        ),
        (
            {"head_ref_name": "feature/no-prefix"},
            Reviewer("trusted-reviewer", "MEMBER"),
            {},
            "branch_not_allowed",
        ),
        ({}, Reviewer("stranger", "CONTRIBUTOR"), {}, "reviewer_not_allowed"),
        (
            {},
            Reviewer("trusted-reviewer", "MEMBER"),
            {"head_sha": "b" * 40},
            "head_changed",
        ),
    ],
)
def test_policy_rejects_untrusted_or_changed_pull_request_state(
    tmp_path: Path,
    pr_overrides: dict[str, object],
    reviewer: Reviewer,
    receipt_overrides: dict[str, object],
    reason: str,
) -> None:
    policy = configured_policy(tmp_path)

    admission = policy.admit(
        admitted_pr(**pr_overrides), reviewer, receipt(**receipt_overrides)
    )

    assert admission.admitted is False
    assert admission.reason == reason


def test_policy_admits_exact_allowed_state(tmp_path: Path) -> None:
    policy = configured_policy(tmp_path)

    admission = policy.admit(
        admitted_pr(), Reviewer("trusted-reviewer", "CONTRIBUTOR"), receipt()
    )

    assert admission.admitted is True
    assert admission.reason is None


def test_policy_allows_only_an_explicitly_configured_head_fork(tmp_path: Path) -> None:
    initialize_git_worktree(tmp_path)
    policy = load_policy(
        {
            "enabled": True,
            "repositories": [
                {
                    "base_repository": "upstream/widgets",
                    "head_repository": "owner/widgets",
                    "local_path": str(tmp_path),
                    "owner_login": "owner",
                    "branch_prefixes": ["codex/"],
                }
            ],
            "reviewer_logins": ["trusted-reviewer"],
            "reviewer_associations": [],
            "not_before": "2026-08-24T00:00:00Z",
            "assignee": "repair-agent",
            "board": "repairs",
        }
    )
    pr = admitted_pr(
        base_repository="upstream/widgets", head_repository="owner/widgets"
    )
    admitted_receipt = receipt(repository="upstream/widgets")

    admission = policy.admit(
        pr, Reviewer("trusted-reviewer", "CONTRIBUTOR"), admitted_receipt
    )

    assert admission.admitted is True


def test_feedback_receipt_is_immutable_and_head_scoped() -> None:
    original = receipt()
    replacement_head = receipt(head_sha="b" * 40)

    assert original.key != replacement_head.key
    with pytest.raises(AttributeError):
        original.head_sha = "b" * 40  # type: ignore[misc]


def test_ledger_deduplicates_completed_receipt_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "ledger.sqlite3"
    first = FeedbackLedger(db_path)

    lease = claim_lease(first, receipt())
    assert lease is not None
    first.finalize(receipt(), "kanban-123", lease)
    first.close()

    restarted = FeedbackLedger(db_path)
    assert claim_lease(restarted, receipt()) is None
    restarted.close()


def test_ledger_deduplicates_an_in_progress_receipt_after_restart(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state" / "ledger.sqlite3"
    first = FeedbackLedger(db_path)

    assert claim_lease(first, receipt()) is not None
    first.close()

    restarted = FeedbackLedger(db_path)
    assert claim_lease(restarted, receipt()) is None
    restarted.close()


def test_local_ci_claim_waits_for_active_exact_head_repair(tmp_path: Path) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    repair = receipt(feedback_kind="pr_repair", feedback_id="repair:merge_conflict")
    audit = receipt(feedback_kind="pr_local_ci", feedback_id="local-ci-audit-v1")
    repair_lease = claim_lease(ledger, repair)

    assert repair_lease is not None
    assert claim_lease(ledger, audit, owner="ci-scanner") is None

    ledger.finalize(repair, "repair-task", repair_lease)
    assert claim_lease(ledger, audit, owner="ci-scanner") is None

    ledger.mark_feedback_actioned(
        repair,
        resolved_head_sha=repair.head_sha,
        actioned_at=datetime(2026, 8, 24, 13, 0, tzinfo=UTC),
    )
    assert claim_lease(ledger, audit, owner="ci-scanner") is not None
    ledger.close()


def test_feedback_action_uses_a_retryable_resolving_transition(tmp_path: Path) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    item = receipt(feedback_kind="review_comment", feedback_id="42")
    lease = claim_lease(ledger, item)
    assert lease is not None
    ledger.finalize(item, "review-task", lease)
    resolved_head = "b" * 40
    started_at = datetime(2026, 8, 24, 13, 0, tzinfo=UTC)

    ledger.begin_feedback_action(
        item, resolved_head_sha=resolved_head, actioned_at=started_at
    )
    ledger.begin_feedback_action(
        item, resolved_head_sha=resolved_head, actioned_at=started_at
    )
    row = ledger._connection.execute(
        "SELECT action_status, actioned_head_sha FROM feedback_receipts WHERE "
        "repository = ? AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? "
        "AND head_sha = ?",
        item.key,
    ).fetchone()
    assert row == ("resolving", resolved_head)

    ledger.mark_feedback_actioned(
        item, resolved_head_sha=resolved_head, actioned_at=started_at
    )
    ledger.begin_feedback_action(
        item, resolved_head_sha=resolved_head, actioned_at=started_at
    )
    assert ledger.was_actioned_on_any_head(item)
    ledger.close()


def test_ledger_retries_a_receipt_after_task_creation_failure(tmp_path: Path) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    lease = claim_lease(ledger, receipt())
    assert lease is not None
    ledger.fail(receipt(), "kanban unavailable", lease)
    assert claim_lease(ledger, receipt()) is None
    assert (
        ledger.retry(
            receipt(),
            owner="retry-scanner",
            claimed_at=datetime(2026, 8, 24, 12, 1, tzinfo=UTC),
        )
        is not None
    )
    ledger.close()


def test_ledger_reclaims_only_a_stale_claim_with_a_new_owner_time_and_version(
    tmp_path: Path,
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)

    first = ledger.claim(
        receipt(),
        owner="scanner-a",
        claimed_at=claimed_at,
        stale_before=claimed_at - timedelta(minutes=5),
    )
    active_duplicate = ledger.claim(
        receipt(),
        owner="scanner-b",
        claimed_at=claimed_at + timedelta(minutes=1),
        stale_before=claimed_at - timedelta(minutes=4),
    )
    reclaimed = ledger.claim(
        receipt(),
        owner="scanner-b",
        claimed_at=claimed_at + timedelta(minutes=6),
        stale_before=claimed_at + timedelta(minutes=1),
    )

    assert first is not None
    assert (first.owner, first.claimed_at, first.version) == (
        "scanner-a",
        claimed_at,
        1,
    )
    assert active_duplicate is None
    assert reclaimed is not None
    assert (reclaimed.owner, reclaimed.claimed_at, reclaimed.version) == (
        "scanner-b",
        claimed_at + timedelta(minutes=6),
        2,
    )
    with pytest.raises(RuntimeError, match="receipt lease is not held"):
        ledger.finalize(receipt(), "stale-task", first)
    ledger.finalize(receipt(), "task-2", reclaimed)
    ledger.close()


def test_ledger_persists_the_materialized_workspace_and_expected_sha_under_the_lease(
    tmp_path: Path,
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    lease = ledger.claim(
        receipt(),
        owner="scanner-a",
        claimed_at=claimed_at,
        stale_before=claimed_at - timedelta(minutes=5),
    )
    assert lease is not None
    workspace = tmp_path / "worktrees" / "receipt"

    ledger.record_workspace(receipt(), lease, workspace, receipt().head_sha)

    row = ledger._connection.execute(
        "SELECT workspace_path, expected_sha FROM feedback_receipts WHERE repository = ? "
        "AND pr_number = ? AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
        receipt().key,
    ).fetchone()
    assert row == (str(workspace), receipt().head_sha)
    with pytest.raises(ValueError, match="expected SHA must equal receipt head SHA"):
        ledger.record_workspace(receipt(), lease, workspace, "b" * 40)
    ledger.close()


def test_ledger_migrates_and_reclaims_a_legacy_claim_without_lease_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE feedback_receipts ("
        "repository TEXT NOT NULL, pr_number INTEGER NOT NULL, feedback_kind TEXT NOT NULL, "
        "feedback_id TEXT NOT NULL, head_sha TEXT NOT NULL, status TEXT NOT NULL, task_id TEXT, "
        "last_error TEXT, attempts INTEGER NOT NULL, "
        "PRIMARY KEY (repository, pr_number, feedback_kind, feedback_id, head_sha))"
    )
    connection.execute(
        "INSERT INTO feedback_receipts VALUES (?, ?, ?, ?, ?, 'claimed', NULL, NULL, 1)",
        receipt().key,
    )
    connection.commit()
    connection.close()
    ledger = FeedbackLedger(path)
    claimed_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)

    reclaimed = ledger.claim(
        receipt(),
        owner="migration-scanner",
        claimed_at=claimed_at,
        stale_before=claimed_at - timedelta(minutes=5),
    )

    assert reclaimed is not None
    assert (reclaimed.owner, reclaimed.version) == ("migration-scanner", 1)
    columns = {
        row[1]
        for row in ledger._connection.execute("PRAGMA table_info(feedback_receipts)")
    }
    assert {
        "claim_owner",
        "claimed_at",
        "lease_version",
        "workspace_path",
        "expected_sha",
    } <= columns
    ledger.close()


def test_policy_rejects_an_empty_dot_git_directory(tmp_path: Path) -> None:
    repository_path = tmp_path / "not-a-repository"
    (repository_path / ".git").mkdir(parents=True)

    with pytest.raises(ValueError):
        load_policy(enabled_raw_config(repository_path))


def test_policy_accepts_a_linked_git_worktree(tmp_path: Path) -> None:
    main = tmp_path / "main"
    linked = tmp_path / "linked"
    initialize_git_worktree(main)
    (main / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(main), "add", "README.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(main),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(main),
            "worktree",
            "add",
            "--quiet",
            "-b",
            "linked",
            str(linked),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    policy = load_policy(enabled_raw_config(linked))

    assert policy.enabled is True


def test_ledger_uses_profile_scoped_hermes_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "github_pr_feedback.ledger.get_hermes_home", lambda: tmp_path / "profile"
    )

    ledger = FeedbackLedger.for_current_profile()

    assert ledger.path == tmp_path / "profile" / "github-pr-feedback" / "ledger.sqlite3"
    ledger.close()


def test_ledger_enables_bounded_busy_waits_and_wal_autocheckpoint(
    tmp_path: Path,
) -> None:
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")

    assert ledger._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
    assert ledger._connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1_000
    ledger.close()


def test_ledger_retries_transient_wal_open_failure(monkeypatch) -> None:
    import github_pr_feedback.ledger as ledger_module

    class Cursor:
        def __init__(self, value):
            self.value = value

        def fetchone(self):
            return (self.value,)

    class Connection:
        def __init__(self):
            self.wal_attempts = 0
            self.current_mode = "delete"

        def execute(self, sql):
            if sql == "PRAGMA journal_mode":
                return Cursor(self.current_mode)
            if sql == "PRAGMA journal_mode=WAL":
                self.wal_attempts += 1
                if self.wal_attempts == 1:
                    raise sqlite3.OperationalError("unable to open database file")
                self.current_mode = "wal"
                return Cursor("wal")
            return Cursor(None)

    connection = Connection()
    monkeypatch.setattr(ledger_module.time, "sleep", lambda _delay: None)

    ledger_module._enable_wal_with_bounded_retry(connection)

    assert connection.wal_attempts == 2
    connection.wal_attempts = 0
    ledger_module._enable_wal_with_bounded_retry(connection)
    assert connection.wal_attempts == 0


def test_worktree_policy_allows_ten_seconds_for_local_git_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout=f"true\n{tmp_path.resolve()}\n")

    monkeypatch.setattr("github_pr_feedback.policy.subprocess.run", fake_run)

    assert _is_git_worktree(tmp_path) is True
    assert observed["timeout"] == 10


def test_archived_dispatch_replacement_is_all_or_nothing(tmp_path: Path) -> None:
    from github_pr_feedback.ledger import PendingTaskBinding

    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    receipt = FeedbackReceipt(
        "acme/widgets", 17, "review_comment", "review-1", "a" * 40
    )
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    lease = ledger.claim(receipt, owner="scanner", claimed_at=now, stale_before=now)
    assert lease is not None
    ledger.finalize(receipt, "real-task", lease)
    replacement = FeedbackReceipt(
        "acme/widgets", 17, "pr_repair", "repair:base_refresh_required", "a" * 40
    )

    acquired = ledger.replace_archived_dispatches(
        replacement,
        archived=(PendingTaskBinding(receipt, "wrong-task"),),
        owner="repair-scanner",
        claimed_at=now,
    )

    assert acquired is None
    row = ledger._connection.execute(
        "SELECT action_status FROM feedback_receipts WHERE repository = ? AND pr_number = ? "
        "AND feedback_kind = ? AND feedback_id = ? AND head_sha = ?",
        receipt.key,
    ).fetchone()
    assert row == ("pending",)
    assert ledger.exact_receipt_status(replacement) is None
    ledger.close()


def test_plugin_directory_exposes_hermes_register_entry_point() -> None:
    plugin_root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "plugin_under_test",
        plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.register)
