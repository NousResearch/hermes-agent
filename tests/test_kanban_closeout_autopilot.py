"""Autopilot closeout integration regression tests."""

from __future__ import annotations


def _review_ready_closeout_evidence() -> dict:
    return {
        "summary": "worker completed implementation",
        "proof": "focused tests and git diff check passed",
        "cleanup": {"proof": "git status --short clean", "worktree_clean": True, "artifacts_removed": []},
        "residue": {"summary": "no residue", "items": [{"kind": "none", "disposition": "none"}]},
        "pr": {"number": 58, "url": "https://github.com/chriskim12/hermes-agent/pull/58"},
        "git": {"head_sha": "abc123", "worktree_clean": True, "status_short": ""},
    }


def _live_pr_provider(_evidence, _repo_path):
    return {
        "live": True,
        "state": "open",
        "is_draft": False,
        "head_sha": "abc123",
        "checks": [{"name": "test", "status": "completed", "conclusion": "success"}],
    }


def test_review_ready_closeout_requires_autopilot_verifier_pass():
    from hermes_cli.kanban_closeout import verify_closeout_transition

    missing = verify_closeout_transition(
        "review_ready",
        _review_ready_closeout_evidence(),
        current_phase="worker_done",
        live_pr_provider=_live_pr_provider,
    )
    failed = verify_closeout_transition(
        "review_ready",
        {**_review_ready_closeout_evidence(), "verifier_verdict": {"verdict": "FAIL", "reason_codes": ["criterion_failed"]}},
        current_phase="worker_done",
        live_pr_provider=_live_pr_provider,
    )
    passed = verify_closeout_transition(
        "review_ready",
        {**_review_ready_closeout_evidence(), "verifier_verdict": {"verdict": "PASS"}},
        current_phase="worker_done",
        live_pr_provider=_live_pr_provider,
    )

    assert missing.allowed is False
    assert "missing_verifier_pass" in missing.blockers
    assert failed.allowed is False
    assert "missing_verifier_pass" in failed.blockers
    assert passed.allowed is True
    assert "missing_verifier_pass" not in passed.blockers
