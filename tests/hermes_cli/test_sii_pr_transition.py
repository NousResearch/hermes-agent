from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.sii_pr_transition import ReviewVerdict, consume_pass, parse_review_verdict


HEAD = "a" * 40
PR_URL = "https://github.com/acme/repo/pull/7"


def _verdict(**metadata) -> ReviewVerdict:
    base = {
        "review_outcome": "pass",
        "pr": 7,
        "url": PR_URL,
        "head": HEAD,
        "check_name": "Lint, typecheck, migrations, tests",
        "check_conclusion": "success",
        "check_job": 42,
        "unresolved_blocking_findings": [],
    }
    base.update(metadata)
    return parse_review_verdict(base)


def test_parse_review_verdict_requires_explicit_pass_and_no_blockers():
    assert _verdict().head == HEAD
    with pytest.raises(ValueError, match="PASS"):
        _verdict(review_outcome="changes_requested")
    with pytest.raises(ValueError, match="blocking"):
        _verdict(unresolved_blocking_findings=["P1 stale state"])


def test_consume_pass_merges_exact_head_verifies_merge_and_creates_closeout(tmp_path: Path):
    calls: list[list[str]] = []
    states = iter([
        {"state": "OPEN", "headRefOid": HEAD, "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "mergedAt": None, "mergeCommit": None},
        {"state": "MERGED", "headRefOid": HEAD, "mergedAt": "2026-10-03T00:00:00Z", "mergeCommit": {"oid": "b" * 40}},
    ])

    def run(command: list[str]) -> str:
        calls.append(command)
        if command[:3] == ["gh", "pr", "view"]:
            return json.dumps(next(states))
        if command[:3] == ["gh", "pr", "merge"]:
            return ""
        if command[:4] == ["hermes", "kanban", "--board", "strategic-industrial-intelligence"]:
            return json.dumps({"id": "t_closeout"})
        raise AssertionError(command)

    receipt = {"ok": True, "classification": "success", "head_sha": HEAD, "checks": [{
        "id": 42, "name": "Lint, typecheck, migrations, tests", "head_sha": HEAD,
        "classification": "success", "conclusion": "success",
    }]}
    result = consume_pass(
        _verdict(), receipt=receipt, board="strategic-industrial-intelligence",
        developer="strategic-industrial-intelligence", workspace=str(tmp_path), run=run,
    )

    assert result["merge_sha"] == "b" * 40
    assert result["closeout_task"] == "t_closeout"
    merge = next(c for c in calls if c[:3] == ["gh", "pr", "merge"])
    assert merge[-2:] == ["--match-head-commit", HEAD]
    create = next(c for c in calls if "create" in c)
    body = create[create.index("--body") + 1]
    assert "PR_MERGED" in body and "ACTIVITY_CLOSED" in body
    assert "110" not in body
    assert "Do not start" in body
    assert create[create.index("--idempotency-key") + 1] == f"sii-closeout-7-{HEAD}"


def test_consume_pass_rejects_stale_or_failed_exact_head(tmp_path: Path):
    bad_receipts = [
        {"ok": True, "classification": "success", "head_sha": "c" * 40, "checks": []},
        {"ok": False, "classification": "failure", "head_sha": HEAD, "checks": []},
        {"ok": True, "classification": "success", "head_sha": HEAD, "checks": [{"id": 42, "head_sha": "c" * 40, "classification": "success"}]},
    ]
    for receipt in bad_receipts:
        called = False
        def run(command: list[str]) -> str:
            nonlocal called
            called = True
            raise AssertionError(command)
        with pytest.raises(ValueError, match="exact-head"):
            consume_pass(_verdict(), receipt=receipt, board="b", developer="d", workspace=str(tmp_path), run=run)
        assert called is False
