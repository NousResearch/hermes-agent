from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli.update_doctor import (
    ConflictBucket,
    LOCKFILE_METADATA_VALIDATION_COMMANDS,
    classify_conflict,
    main as doctor_main,
    render_report,
    _batch_upstream_run,
    _classify_pr_risk,
    _classify_upstream_batch_risk,
    _fallback_batch_to_individual_commits,
    _lockfile_metadata_only_details,
    _lockfile_metadata_only_payload,
    _plan_upstream_batches,
    _prepare_medium_web_sandbox,
    _publish_run_artifacts,
    _run_medium_commit_sandbox_tests,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _sample_report(*, repair: dict | None = None, replay_result: str = "conflict", conflict: dict | None = None) -> dict:
    report = {
        "schema_version": 1,
        "tool": {"name": "hermes-update-doctor", "mode": "analyze"},
        "timestamp": "2026-04-30T00:00:00Z",
        "environment": {
            "project_root": "/opt/hermes-agent",
            "branch": "main",
            "upstream": "fork/main",
            "remotes": {
                "fork": {"fetch": "git@github.com:Vottam/hermes-agent.git", "push": "git@github.com:Vottam/hermes-agent.git"},
                "origin": {"fetch": "git@github.com:NousResearch/hermes-agent.git", "push": "git@github.com:NousResearch/hermes-agent.git"},
            },
            "status_before": ["## main...fork/main"],
            "ahead": 111,
            "behind": 99,
            "origin_ahead_count": 99,
            "origin_behind_count": 111,
        },
        "replay": {
            "base": "origin/main",
            "candidate_count": 104,
            "applied_count": 103,
            "skipped_duplicate_commits": [
                {
                    "commit": "deadbeef",
                    "subject": "fix(update): skip duplicate",
                    "patch_id": "patch-1",
                    "bucket": "patch-id-duplicate",
                }
            ],
            "skipped_safe_commits": [
                {
                    "commit": "cafebabe",
                    "subject": "fix(update): already covered",
                    "patch_id": "patch-2",
                    "bucket": "already-covered-in-fork-main",
                }
            ],
            "replay_continued_after_skip": False,
            "rescue_ref": "refs/rescue/hermes-update-doctor-20260430-000000",
            "result": replay_result,
        },
        "conflict": conflict
        if conflict is not None
        else {
            "commit": "7c7fedae2a3a2bf92cdffc9c67026286af1baf46",
            "subject": "fix(update): print final update safety report",
            "patch_id": "c270a715455b204a1ca445a906bf1e1e7f1c9c70",
            "bucket": "already-covered-in-fork-main",
            "touched_files": ["hermes_cli/main.py"],
            "conflicted_files": ["hermes_cli/main.py"],
            "coverage_refs": ["refs/heads/main", "refs/remotes/fork/main"],
            "main_contains": ["main"],
            "remote_contains": ["fork/main"],
            "worktree_status": ["UU hermes_cli/main.py"],
        },
        "verification": {"origin_untouched": True, "main_untouched": True},
        "covered_upstream_commits": [
            {
                "upstream_commit": "1745cfc6d73b69506118526760eb67456e1ef422",
                "covered_by_commits": [
                    "a83188672ea860bc5e8b61861b89b97bc41b49b9",
                    "8e251577d1f44d35eba333bbaa25a2eae5e35cd0",
                ],
                "covered_by_pr": 18,
                "reason": "equivalent fork fix with baseline-aware test",
                "evidence": [
                    "web build passed",
                    "browser-safe-imports test passed",
                    "update_doctor tests passed",
                    "profile test passed",
                ],
            }
        ],
        "covered_upstream_count": 1,
        "coverage_used": False,
        "batch_covered_commits": [],
    }
    if repair is not None:
        report["repair"] = repair
    return report


def test_formal_buckets_are_stable_and_unique() -> None:
    values = [bucket.value for bucket in ConflictBucket]
    assert len(values) == len(set(values))
    assert set(values) == {
        "patch-id-duplicate",
        "already-covered-in-fork-main",
        "obsolete",
        "test-desync",
        "single-file-hunk-conflict",
        "runtime-sensitive",
        "critical",
        "multi-file-or-runtime-conflict",
    }


def test_classify_conflict_prefers_coverage_over_runtime_risk() -> None:
    bucket = classify_conflict(
        patch_id="abc123",
        coverage_refs=["refs/remotes/fork/main"],
        touched_files=["hermes_cli/main.py", "tests/hermes_cli/test_update_final_report.py"],
        conflicted_files=["hermes_cli/main.py"],
        subject="fix(update): print final update safety report",
    )
    assert bucket is ConflictBucket.ALREADY_COVERED_IN_FORK_MAIN


def test_classify_conflict_covers_basic_buckets() -> None:
    assert (
        classify_conflict(
            patch_id="dup-1",
            coverage_refs=[],
            touched_files=["hermes_cli/main.py"],
            conflicted_files=["hermes_cli/main.py"],
            seen_patch_ids={"dup-1"},
        )
        is ConflictBucket.PATCH_ID_DUPLICATE
    )

    assert (
        classify_conflict(
            patch_id=None,
            coverage_refs=[],
            touched_files=["tests/hermes_cli/test_update_doctor.py"],
            conflicted_files=["tests/hermes_cli/test_update_doctor.py"],
        )
        is ConflictBucket.TEST_DESYNC
    )

    assert (
        classify_conflict(
            patch_id=None,
            coverage_refs=[],
            touched_files=["gateway/run.py"],
            conflicted_files=["gateway/run.py"],
        )
        is ConflictBucket.CRITICAL
    )

    assert (
        classify_conflict(
            patch_id=None,
            coverage_refs=[],
            touched_files=["hermes_cli/main.py"],
            conflicted_files=["hermes_cli/main.py"],
        )
        is ConflictBucket.SINGLE_FILE_HUNK_CONFLICT
    )


def test_render_report_json_and_yaml_round_trip() -> None:
    report = _sample_report()
    json_text = render_report(report, "json")
    yaml_text = render_report(report, "yaml")

    assert json.loads(json_text) == report
    assert yaml.safe_load(yaml_text) == report
    assert json.loads(json_text)["replay"]["skipped_duplicate_commits"][0]["bucket"] == "patch-id-duplicate"
    assert json.loads(json_text)["replay"]["skipped_safe_commits"][0]["bucket"] == "already-covered-in-fork-main"
    assert json.loads(json_text)["environment"]["origin_ahead_count"] == 99
    assert json.loads(json_text)["covered_upstream_commits"][0]["upstream_commit"] == "1745cfc6d73b69506118526760eb67456e1ef422"
    assert json.loads(json_text)["covered_upstream_count"] == 1
    assert json.loads(json_text)["coverage_used"] is False


def test_render_report_repair_round_trip() -> None:
    report = _sample_report(
        repair={
            "mode": "repair",
            "result": "skip-safe",
            "bucket": "already-covered-in-fork-main",
            "repair_status": "skip-safe",
            "repair_action": "no-op",
            "safety_level": "safe",
            "next_step": "Commit is already covered in fork/main or main; no repair was applied.",
        }
    )

    json_text = render_report(report, "json")
    yaml_text = render_report(report, "yaml")

    assert json.loads(json_text) == report
    assert yaml.safe_load(yaml_text) == report
    assert json.loads(json_text)["repair"]["repair_status"] == "skip-safe"


def test_render_report_run_round_trip() -> None:
    report = _sample_report(
        replay_result="passed",
        conflict=None,
    )
    report.update(
        {
            "mode": "run",
            "result": "clean",
            "bucket": None,
            "run_status": "clean",
            "repair_status": "not-needed",
            "action_taken": "no-op",
            "safety_level": "safe",
            "next_step": "Replay completed cleanly; no repair was needed.",
            "tests_run": [],
            "pr_url": None,
        }
    )

    json_text = render_report(report, "json")
    yaml_text = render_report(report, "yaml")

    assert json.loads(json_text) == report
    assert yaml.safe_load(yaml_text) == report
    assert json.loads(json_text)["run_status"] == "clean"


def test_batch_upstream_run_skips_explicitly_covered_high_risk_commit(monkeypatch) -> None:
    covered_commit = "1745cfc6d73b69506118526760eb67456e1ef422"
    monkeypatch.setattr(
        "hermes_cli.update_doctor._plan_upstream_batches",
        lambda root, batch_size: [{"index": 1, "size": 1, "commits": [covered_commit]}],
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._batch_changed_files",
        lambda root, commits: ["hermes_cli/main.py"] if commits else [],
    )

    result = _batch_upstream_run(
        _sample_report(replay_result="passed", conflict=None),
        root=Path("/tmp/hermes-update-doctor-test"),
        request_pr=False,
        request_auto_merge_low_risk=False,
        batch_size=5,
    )

    assert result["coverage_used"] is True
    assert result["batch_covered_commits"][0]["upstream_commit"] == covered_commit
    assert result["batches_blocked"] == 0
    assert result["final_status"] == "completed"
    assert result["next_step"] == "All upstream batches were covered by fork fixes."


def test_batch_upstream_run_marks_ae11_as_covered_and_keeps_it_unblocked(monkeypatch) -> None:
    covered_commit = "ae11a310582ac936cbbffc516891cc2bd9fdd458"
    monkeypatch.setattr(
        "hermes_cli.update_doctor._plan_upstream_batches",
        lambda root, batch_size: [{"index": 1, "size": 1, "commits": [covered_commit]}],
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._batch_changed_files",
        lambda root, commits: ["hermes_cli/web_server.py"] if commits else [],
    )

    result = _batch_upstream_run(
        {},
        root=Path("/tmp/hermes-update-doctor-test"),
        request_pr=False,
        request_auto_merge_low_risk=False,
        batch_size=5,
    )

    assert result["coverage_used"] is True
    assert result["batch_covered_commits"][0]["upstream_commit"] == covered_commit
    assert any(item["upstream_commit"] == covered_commit for item in result["covered_upstream_commits"])
    assert result["batches_blocked"] == 0
    assert result["final_status"] == "completed"
    assert result["next_step"] == "All upstream batches were covered by fork fixes."


def test_batch_upstream_run_marks_469_as_covered_and_keeps_it_unblocked(monkeypatch) -> None:
    covered_commit = "469e4df3c2579dcf24fbf2acc7d802a54970b460"
    monkeypatch.setattr(
        "hermes_cli.update_doctor._plan_upstream_batches",
        lambda root, batch_size: [{"index": 1, "size": 1, "commits": [covered_commit]}],
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._batch_changed_files",
        lambda root, commits: ["hermes_cli/main.py"] if commits else [],
    )

    result = _batch_upstream_run(
        {},
        root=Path("/tmp/hermes-update-doctor-test"),
        request_pr=False,
        request_auto_merge_low_risk=False,
        batch_size=5,
    )

    assert result["coverage_used"] is True
    assert result["batch_covered_commits"][0]["upstream_commit"] == covered_commit
    assert any(item["upstream_commit"] == covered_commit for item in result["covered_upstream_commits"])
    assert result["batches_blocked"] == 0
    assert result["final_status"] == "completed"
    assert result["next_step"] == "All upstream batches were covered by fork fixes."


def test_batch_upstream_run_blocks_uncovered_high_risk_commit(monkeypatch) -> None:
    uncovered_commit = "feedfacefeedfacefeedfacefeedfacefeedface"
    monkeypatch.setattr(
        "hermes_cli.update_doctor._plan_upstream_batches",
        lambda root, batch_size: [{"index": 1, "size": 1, "commits": [uncovered_commit]}],
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._batch_changed_files",
        lambda root, commits: ["hermes_cli/main.py"] if commits else [],
    )

    result = _batch_upstream_run(
        _sample_report(replay_result="passed", conflict=None),
        root=Path("/tmp/hermes-update-doctor-test"),
        request_pr=False,
        request_auto_merge_low_risk=False,
        batch_size=5,
    )

    assert result["coverage_used"] is False
    assert result["batches_blocked"] == 1
    assert result["blocked_batch_commits"] == [uncovered_commit]
    assert result["blocked_batch_reason"] == "batch-risk:high"
    assert result["integration_risk_level"] == "high"


def test_batch_upstream_run_keeps_uncovered_high_risk_blocking_even_with_covered_neighbor(monkeypatch) -> None:
    covered_commit = "1745cfc6d73b69506118526760eb67456e1ef422"
    uncovered_commit = "feedfacefeedfacefeedfacefeedfacefeedface"
    monkeypatch.setattr(
        "hermes_cli.update_doctor._plan_upstream_batches",
        lambda root, batch_size: [{"index": 1, "size": 2, "commits": [covered_commit, uncovered_commit]}],
    )

    def fake_batch_changed_files(root, commits):
        if commits == [uncovered_commit]:
            return ["hermes_cli/main.py"]
        if commits == [covered_commit]:
            return ["hermes_cli/main.py"]
        return ["docs/readme.md"]

    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", fake_batch_changed_files)

    result = _batch_upstream_run(
        _sample_report(replay_result="passed", conflict=None),
        root=Path("/tmp/hermes-update-doctor-test"),
        request_pr=False,
        request_auto_merge_low_risk=False,
        batch_size=5,
    )

    assert result["coverage_used"] is True
    assert [item["upstream_commit"] for item in result["batch_covered_commits"]] == [covered_commit]
    assert result["batches_blocked"] == 1
    assert result["blocked_batch_commits"] == [uncovered_commit]
    assert result["blocked_batch_reason"] == "batch-risk:high"
    assert result["integration_risk_level"] == "high"


def test_repair_mode_skip_safe_and_preserves_working_tree(monkeypatch, capsys) -> None:
    report_before = _git_status()
    monkeypatch.setattr("hermes_cli.update_doctor.build_report", lambda **kwargs: _sample_report())

    exit_code = doctor_main(["--repair", "--format", "json"])
    out = capsys.readouterr().out
    report_after = _git_status()

    assert exit_code == 0
    assert report_before == report_after

    rendered = json.loads(out)
    assert rendered["tool"]["mode"] == "repair"
    assert rendered["repair"]["result"] == "skip-safe"
    assert rendered["repair"]["repair_status"] == "skip-safe"
    assert rendered["repair"]["repair_action"] == "no-op"
    assert rendered["repair"]["safety_level"] == "safe"
    assert rendered["repair"]["next_step"] == "Commit is already covered in fork/main or main; no repair was applied."


def test_analyze_mode_still_works(monkeypatch, capsys) -> None:
    monkeypatch.setattr("hermes_cli.update_doctor.build_report", lambda **kwargs: _sample_report())

    exit_code = doctor_main(["--analyze", "--format", "json"])
    out = capsys.readouterr().out

    assert exit_code == 1
    rendered = json.loads(out)
    assert rendered["tool"]["mode"] == "analyze"
    assert "repair" not in rendered
    assert "run_status" not in rendered



def test_run_mode_needs_integration_after_skip_safe_and_preserves_working_tree(monkeypatch, capsys) -> None:
    report_before = _git_status()
    report = _sample_report(replay_result="passed", conflict=None)
    report["replay"]["replay_continued_after_skip"] = True
    report["environment"]["behind"] = 19
    report["environment"]["origin_ahead_count"] = 19
    monkeypatch.setattr("hermes_cli.update_doctor.build_report", lambda **kwargs: report)

    exit_code = doctor_main(["--run", "--format", "json"])
    out = capsys.readouterr().out
    report_after = _git_status()

    assert exit_code == 1
    assert report_before == report_after

    rendered = json.loads(out)
    assert rendered["mode"] == "run"
    assert rendered["run_status"] == "needs-integration"
    assert rendered["result"] == "needs-integration"
    assert rendered["repair_status"] == "skip-safe"
    assert rendered["action_taken"] == "no-op"
    assert rendered["safety_level"] == "safe"
    assert rendered["risk_level"] == "low"
    assert rendered["pr_status"] == "no-pr-needed"
    assert rendered["merge_status"] == "not-needed"
    assert rendered["branch_name"] == "main"
    assert rendered["next_step"] == "Sandbox replay continued after safe skips, but origin/main is still 19 commits ahead; no integration was applied."
    assert rendered["tests_run"] == []
    assert rendered["pr_url"] is None
    assert rendered["merge_commit"] is None
    assert rendered["final_validation"]["status"] == "not-needed"
    assert rendered["origin_ahead_count"] == 19
    assert rendered["skipped_safe_commits"]
    assert rendered["replay_continued_after_skip"] is True
    assert rendered["integration_status"] == "needs-integration"
    assert rendered["material_changes_detected"] is False



def test_run_mode_clean_when_no_conflict(monkeypatch, capsys) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    report["conflict"] = None
    report["environment"]["behind"] = 0
    report["environment"]["origin_ahead_count"] = 0
    monkeypatch.setattr("hermes_cli.update_doctor.build_report", lambda **kwargs: report)

    exit_code = doctor_main(["--run", "--format", "json"])
    out = capsys.readouterr().out

    assert exit_code == 0
    rendered = json.loads(out)
    assert rendered["mode"] == "run"
    assert rendered["run_status"] == "clean"
    assert rendered["result"] == "clean"
    assert rendered["bucket"] is None
    assert rendered["repair_status"] == "not-needed"
    assert rendered["pr_status"] == "no-pr-needed"
    assert rendered["merge_status"] == "not-needed"
    assert rendered["risk_level"] == "low"
    assert rendered["final_validation"]["status"] == "not-needed"
    assert rendered["origin_ahead_count"] == 0


def test_auto_merge_requires_pr() -> None:
    with pytest.raises(SystemExit):
        doctor_main(["--run", "--auto-merge-low-risk", "--format", "json"])


def test_pr_and_auto_merge_noop_path_does_not_create_pr(monkeypatch, capsys) -> None:
    report = _sample_report()
    monkeypatch.setattr("hermes_cli.update_doctor.build_report", lambda **kwargs: report)
    monkeypatch.setattr("hermes_cli.update_doctor._create_pr", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PR should not be created")))
    monkeypatch.setattr("hermes_cli.update_doctor._merge_pr", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("merge should not run")))

    exit_code = doctor_main(["--run", "--pr", "--auto-merge-low-risk", "--format", "json"])
    out = capsys.readouterr().out

    assert exit_code == 1
    rendered = json.loads(out)
    assert rendered["run_status"] == "blocked"
    assert rendered["result"] == "blocked"
    assert rendered["integration_status"] == "blocked-high-risk"
    assert rendered["integration_risk_level"] == "high"
    assert rendered["integration_blockers"] == ["broad-upstream-sync"]
    assert rendered["pr_status"] == "not-created-risk"
    assert rendered["merge_status"] == "not-needed"
    assert rendered["pr_url"] is None
    assert rendered["origin_ahead_count"] == 99


def test_classify_pr_risk_handles_low_and_high_paths() -> None:
    docs_report = _sample_report()
    docs_report["repair"] = {
        "mode": "repair",
        "result": "changed",
        "bucket": "test-desync",
        "repair_status": "applied",
        "repair_action": "patch-applied",
        "safety_level": "safe",
        "next_step": "done",
        "changed_files": ["docs/notes.md", "docs/plans/2026-04-30-hermes-update-doctor.md"],
    }
    tests_report = _sample_report()
    tests_report["repair"] = {
        "mode": "repair",
        "result": "changed",
        "bucket": "test-desync",
        "repair_status": "applied",
        "repair_action": "patch-applied",
        "safety_level": "safe",
        "next_step": "done",
        "changed_files": ["tests/hermes_cli/test_update_doctor.py"],
    }
    runtime_report = _sample_report()
    runtime_report["repair"] = {
        "mode": "repair",
        "result": "changed",
        "bucket": "runtime-sensitive",
        "repair_status": "applied",
        "repair_action": "patch-applied",
        "safety_level": "guarded",
        "next_step": "done",
        "changed_files": ["hermes_cli/main.py"],
    }

    assert _classify_pr_risk(docs_report) == "low"
    assert _classify_pr_risk(tests_report) == "low"
    assert _classify_pr_risk(runtime_report) == "high"


def test_publish_artifacts_merges_low_risk_pr(monkeypatch) -> None:
    report = _sample_report()
    report["environment"]["behind"] = 19
    report["environment"]["origin_ahead_count"] = 19
    report["repair"] = {
        "mode": "repair",
        "result": "changed",
        "bucket": "test-desync",
        "repair_status": "applied",
        "repair_action": "patch-applied",
        "safety_level": "safe",
        "next_step": "done",
        "changed_files": ["docs/notes.md"],
    }
    monkeypatch.setattr("hermes_cli.update_doctor._ensure_fork_safe_publication", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.update_doctor._create_pr", lambda *args, **kwargs: ("https://github.com/Vottam/hermes-agent/pull/99", 99))
    monkeypatch.setattr("hermes_cli.update_doctor._review_pr_for_auto_merge", lambda *args, **kwargs: {"mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN", "autoMergeRequest": None, "baseRefName": "main", "headRefName": "update-doctor-pr-1", "url": "https://github.com/Vottam/hermes-agent/pull/99"})
    monkeypatch.setattr("hermes_cli.update_doctor._merge_pr", lambda *args, **kwargs: "merge-commit-sha")
    monkeypatch.setattr("hermes_cli.update_doctor._refresh_main_and_validate", lambda *args, **kwargs: ["./venv/bin/python -m pytest tests/hermes_cli/test_update_doctor.py -q", "hermes doctor", "hermes --version"])

    published = _publish_run_artifacts(report, root=REPO_ROOT, request_pr=True, request_auto_merge_low_risk=True)

    assert published["pr_status"] == "created"
    assert published["merge_status"] == "merged"
    assert published["merge_commit"] == "merge-commit-sha"
    assert published["pr_url"] == "https://github.com/Vottam/hermes-agent/pull/99"
    assert published["final_validation"]["status"] == "passed"
    assert published["final_validation"]["checks"]
    assert published["action_taken"] == "pr-created-and-merged"


def test_lockfile_metadata_only_payload_allows_peer_only_changes() -> None:
    before = {
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "web", "version": "1.0.0"},
            "node_modules/example": {
                "version": "2.0.0",
                "resolved": "https://registry.npmjs.org/example/-/example-2.0.0.tgz",
                "integrity": "sha512-before",
            },
        },
    }
    after = json.loads(json.dumps(before))
    after["packages"][""]["peer"] = True
    after["packages"]["node_modules/example"]["peer"] = True

    assert _lockfile_metadata_only_payload(before, after)


def test_lockfile_metadata_only_payload_rejects_resolved_change() -> None:
    before = {
        "packages": {
            "": {"name": "web", "version": "1.0.0"},
            "node_modules/example": {
                "version": "2.0.0",
                "resolved": "https://registry.npmjs.org/example/-/example-2.0.0.tgz",
                "integrity": "sha512-before",
            },
        },
    }
    after = json.loads(json.dumps(before))
    after["packages"]["node_modules/example"]["resolved"] = "https://registry.npmjs.org/example/-/example-2.0.1.tgz"

    assert not _lockfile_metadata_only_payload(before, after)


def test_lockfile_metadata_only_payload_rejects_integrity_change() -> None:
    before = {
        "packages": {
            "": {"name": "web", "version": "1.0.0"},
            "node_modules/example": {
                "version": "2.0.0",
                "resolved": "https://registry.npmjs.org/example/-/example-2.0.0.tgz",
                "integrity": "sha512-before",
            },
        },
    }
    after = json.loads(json.dumps(before))
    after["packages"]["node_modules/example"]["integrity"] = "sha512-after"

    assert not _lockfile_metadata_only_payload(before, after)


def test_lockfile_metadata_only_details_rejects_package_json_change(monkeypatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.update_doctor._commit_files",
        lambda *args, **kwargs: ["web/package-lock.json", "web/package.json"],
    )

    assert _lockfile_metadata_only_details(REPO_ROOT, "deadbeef") is None


def test_classify_upstream_batch_risk_treats_9b62_lockfile_metadata_only_as_low() -> None:
    assert _classify_upstream_batch_risk(REPO_ROOT, ["9b62c98170c481c8a45fe828ef01c65964c0cf01"]) == "low"


def test_publish_artifacts_merges_lockfile_metadata_only_pr(monkeypatch) -> None:
    report = _sample_report()
    report["environment"]["behind"] = 19
    report["environment"]["origin_ahead_count"] = 19
    report["repair"] = {
        "mode": "repair",
        "result": "changed",
        "bucket": "test-desync",
        "repair_status": "applied",
        "repair_action": "patch-applied",
        "safety_level": "safe",
        "next_step": "done",
        "changed_files": ["web/package-lock.json"],
    }
    report["lockfile_metadata_only"] = True
    report["lockfile_metadata_commit"] = "9b62c98170c481c8a45fe828ef01c65964c0cf01"
    report["lockfile_metadata_files"] = ["web/package-lock.json"]
    report["lockfile_metadata_validation"] = {"status": "passed", "checks": ["web/package-lock.json only"]}

    monkeypatch.setattr("hermes_cli.update_doctor._ensure_fork_safe_publication", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.update_doctor._create_pr", lambda *args, **kwargs: ("https://github.com/Vottam/hermes-agent/pull/199", 199))
    monkeypatch.setattr(
        "hermes_cli.update_doctor._review_pr_for_auto_merge",
        lambda *args, **kwargs: {
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
            "autoMergeRequest": None,
            "baseRefName": "main",
            "headRefName": "update-doctor-pr-1",
            "url": "https://github.com/Vottam/hermes-agent/pull/199",
        },
    )
    monkeypatch.setattr("hermes_cli.update_doctor._merge_pr", lambda *args, **kwargs: "merge-commit-sha")
    monkeypatch.setattr("hermes_cli.update_doctor._run_validation_commands", lambda *args, **kwargs: {"status": "passed", "checks": list(LOCKFILE_METADATA_VALIDATION_COMMANDS), "failed_command": None})
    monkeypatch.setattr("hermes_cli.update_doctor._refresh_main_and_validate", lambda *args, **kwargs: ["./venv/bin/python -m pytest tests/hermes_cli/test_update_doctor.py -q", "hermes doctor", "hermes --version"])

    published = _publish_run_artifacts(report, root=REPO_ROOT, request_pr=True, request_auto_merge_low_risk=True)

    assert published["risk_level"] == "low"
    assert published["lockfile_metadata_only"] is True
    assert published["lockfile_metadata_commit"] == "9b62c98170c481c8a45fe828ef01c65964c0cf01"
    assert published["lockfile_metadata_files"] == ["web/package-lock.json"]
    assert published["lockfile_metadata_validation"]["status"] == "passed"
    assert published["lockfile_metadata_validation"]["checks"] == list(LOCKFILE_METADATA_VALIDATION_COMMANDS)
    assert published["merge_status"] == "merged"
    assert published["merge_commit"] == "merge-commit-sha"
def test_batch_upstream_flag_requires_run() -> None:
    with pytest.raises(SystemExit):
        doctor_main(["--batch-upstream", "--format", "json"])


def test_plan_upstream_batches_chunks_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.update_doctor._upstream_candidate_commits",
        lambda *args, **kwargs: [f"commit-{index}" for index in range(1, 13)],
    )

    batches = _plan_upstream_batches(REPO_ROOT, 5)

    assert [batch["size"] for batch in batches] == [5, 5, 2]
    assert batches[0]["first_commit"] == "commit-1"
    assert batches[0]["last_commit"] == "commit-5"
    assert batches[-1]["first_commit"] == "commit-11"
    assert batches[-1]["last_commit"] == "commit-12"


def test_classify_upstream_batch_risk_low_medium_high(monkeypatch) -> None:
    file_map = {
        ("docs",): ["docs/notes.md", "docs/plans/2026-04-30-hermes-update-doctor.md"],
        ("tests",): ["tests/hermes_cli/test_update_doctor.py"],
        ("runtime",): ["hermes_cli/main.py"],
        ("mixed",): ["docs/notes.md", "hermes_cli/main.py"],
    }
    monkeypatch.setattr(
        "hermes_cli.update_doctor._batch_changed_files",
        lambda *args, **kwargs: file_map[tuple(args[1])],
    )

    assert _classify_upstream_batch_risk(REPO_ROOT, ["docs"]) == "low"
    assert _classify_upstream_batch_risk(REPO_ROOT, ["tests"]) == "low"
    assert _classify_upstream_batch_risk(REPO_ROOT, ["runtime"]) == "high"
    assert _classify_upstream_batch_risk(REPO_ROOT, ["mixed"]) == "high"


def test_batch_upstream_run_falls_back_to_individual_commits(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    report["environment"]["behind"] = 147
    report["environment"]["origin_ahead_count"] = 147

    batches = [
        {"index": 1, "size": 5, "commits": ["docs-1", "docs-2", "runtime", "docs-4", "docs-5"], "first_commit": "docs-1", "last_commit": "docs-5"},
        {"index": 2, "size": 5, "commits": ["later-1", "later-2", "later-3", "later-4", "later-5"], "first_commit": "later-1", "last_commit": "later-5"},
    ]
    risk_map = {
        ("docs-1", "docs-2", "runtime", "docs-4", "docs-5"): "high",
        ("docs-1",): "low",
        ("docs-2",): "low",
        ("runtime",): "high",
        ("docs-4",): "low",
        ("docs-5",): "low",
    }
    file_map = {
        ("docs-1",): ["docs/notes.md"],
        ("docs-2",): ["docs/plans/2026-04-30-hermes-update-doctor.md"],
        ("runtime",): ["hermes_cli/main.py"],
        ("docs-4",): ["docs/notes.md"],
        ("docs-5",): ["docs/notes.md"],
        ("docs-1", "docs-2", "runtime", "docs-4", "docs-5"): ["docs/notes.md", "hermes_cli/main.py"],
        ("later-1", "later-2", "later-3", "later-4", "later-5"): ["docs/notes.md"],
    }
    process_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr("hermes_cli.update_doctor._plan_upstream_batches", lambda *args, **kwargs: batches)
    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._refresh_main_and_validate", lambda *args, **kwargs: ["pytest", "doctor", "version"])

    def fake_process(report, *, root, batch, request_pr, request_auto_merge_low_risk, refresh_after_merge=False):
        commits = tuple(batch["commits"])
        process_calls.append(commits)
        if commits == ("runtime",):
            raise AssertionError("high-risk runtime commit must not be auto-merged")
        return {
            "status": "merged",
            "reason": None,
            "risk": "low",
            "commits": list(commits),
            "files": file_map[commits],
            "branch_name": f"update-doctor-batch-upstream-{batch['index']}",
            "pr_url": f"https://github.com/Vottam/hermes-agent/pull/{100 + len(process_calls)}",
            "merge_commit": f"merge-{commits[0]}",
            "published": {
                "tests_run": ["pytest"],
                "final_validation": {"status": "passed", "checks": ["pytest"]},
                "merge_status": "merged",
            },
        }

    monkeypatch.setattr("hermes_cli.update_doctor._process_upstream_batch", fake_process)

    summary = _batch_upstream_run(report, root=REPO_ROOT, request_pr=True, request_auto_merge_low_risk=True, batch_size=5)

    assert summary["batch_fallback_used"] is True
    assert summary["fallback_from_batch_size"] == 5
    assert summary["fallback_commits_processed"] == 3
    assert summary["fallback_commits_merged"] == 2
    assert summary["fallback_blocked_commit"] == "runtime"
    assert summary["fallback_blocked_files"] == ["hermes_cli/main.py"]
    assert summary["fallback_blocked_reason"] == "batch-risk:high"
    assert summary["batches_total"] == 2
    assert summary["batches_processed"] == 1
    assert summary["batches_blocked"] == 1
    assert summary["blocked_batch_reason"] == "batch-risk:high"
    assert summary["blocked_batch_commits"] == ["runtime"]
    assert summary["run_status"] == "blocked"
    assert summary["integration_status"] == "blocked-batch-risk"
    assert process_calls == [("docs-1",), ("docs-2",)]


def test_fallback_batch_to_individual_commits_blocks_on_runtime(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    batch = {"index": 1, "size": 2, "commits": ["docs", "runtime"], "first_commit": "docs", "last_commit": "runtime"}
    risk_map = {("docs", "runtime"): "high", ("docs",): "low", ("runtime",): "high"}
    file_map = {("docs",): ["docs/notes.md"], ("runtime",): ["hermes_cli/main.py"], ("docs", "runtime"): ["docs/notes.md", "hermes_cli/main.py"]}
    process_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr(
        "hermes_cli.update_doctor._process_upstream_batch",
        lambda *args, **kwargs: process_calls.append(tuple(kwargs["batch"]["commits"])) or {
            "status": "merged",
            "reason": None,
            "risk": "low",
            "commits": kwargs["batch"]["commits"],
            "files": file_map[tuple(kwargs["batch"]["commits"])],
            "branch_name": "update-doctor-batch-upstream-01",
            "pr_url": "https://github.com/Vottam/hermes-agent/pull/101",
            "merge_commit": "merge-docs",
            "published": {"tests_run": ["pytest"], "final_validation": {"status": "passed", "checks": ["pytest"]}, "merge_status": "merged"},
        },
    )

    summary = _fallback_batch_to_individual_commits(report, root=REPO_ROOT, batch=batch, request_pr=True, request_auto_merge_low_risk=True)

    assert summary["batch_fallback_used"] is True
    assert summary["fallback_from_batch_size"] == 2
    assert summary["fallback_commits_processed"] == 2
    assert summary["fallback_commits_merged"] == 1
    assert summary["fallback_blocked_commit"] == "runtime"
    assert summary["fallback_blocked_files"] == ["hermes_cli/main.py"]
    assert summary["fallback_blocked_reason"] == "batch-risk:high"
    assert process_calls == [("docs",)]
def test_fallback_batch_to_individual_commits_marks_baseline_failure_and_keeps_pr_eligible(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    batch = {
        "index": 1,
        "size": 2,
        "commits": ["docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"],
        "first_commit": "docs",
        "last_commit": "4523965de9eb9a55ba7a67315adc3188c31eaec4",
    }
    risk_map = {
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): "high",
        ("docs",): "low",
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): "medium",
    }
    medium_files = [
        "hermes_cli/web_server.py",
        "tests/hermes_cli/test_web_server.py",
        "web/src/App.tsx",
        "web/src/i18n/en.ts",
        "web/src/i18n/types.ts",
        "web/src/i18n/zh.ts",
        "web/src/lib/api.ts",
        "web/src/pages/ProfilesPage.tsx",
    ]
    file_map = {
        ("docs",): ["docs/notes.md"],
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): medium_files,
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): ["docs/notes.md", *medium_files],
    }
    process_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr(
        "hermes_cli.update_doctor._run_medium_commit_sandbox_tests",
        lambda *args, **kwargs: {
            "status": "baseline-failure",
            "tests_run": ["pytest tests/hermes_cli/test_web_server.py -k profile -q", "pytest tests/hermes_cli/test_web_server.py -q"],
            "tests_passed": ["pytest tests/hermes_cli/test_web_server.py -k profile -q"],
            "test_failures": ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"],
            "medium_baseline_tests_run": ["pytest tests/hermes_cli/test_web_server.py -q"],
            "medium_baseline_failures": ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"],
            "medium_baseline_failure_matched": True,
            "medium_sandbox_prepared": False,
            "medium_sandbox_preparation": [],
            "branch_name": "update-doctor-medium-tested-4523965d",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._process_upstream_batch",
        lambda *args, **kwargs: process_calls.append(tuple(kwargs["batch"]["commits"])) or (_ for _ in ()).throw(AssertionError("medium-tested PR should not be created when sandbox tests fail")),
    )

    summary = _fallback_batch_to_individual_commits(report, root=REPO_ROOT, batch=batch, request_pr=False, request_auto_merge_low_risk=True)

    assert summary["status"] == "medium-tested"
    assert summary["fallback_blocked_commit"] is None
    assert summary["medium_triage_used"] is True
    assert summary["medium_tested_status"] == "baseline-failure"
    assert summary["medium_baseline_failure_matched"] is True
    assert summary["medium_pr_eligible"] is True
    assert summary["medium_tests_run"] == ["pytest tests/hermes_cli/test_web_server.py -k profile -q", "pytest tests/hermes_cli/test_web_server.py -q"]
    assert summary["medium_tests_passed"] == ["pytest tests/hermes_cli/test_web_server.py -k profile -q"]
    assert summary["medium_test_failures"] == ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"]
    assert summary["medium_baseline_tests_run"] == ["pytest tests/hermes_cli/test_web_server.py -q"]
    assert summary["medium_baseline_failures"] == ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"]
    assert summary["medium_sandbox_prepared"] is False
    assert summary["medium_sandbox_preparation"] == []
    assert process_calls == []


def test_fallback_batch_to_individual_commits_creates_pr_for_baseline_failure_when_requested(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    batch = {
        "index": 1,
        "size": 2,
        "commits": ["docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"],
        "first_commit": "docs",
        "last_commit": "4523965de9eb9a55ba7a67315adc3188c31eaec4",
    }
    risk_map = {
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): "high",
        ("docs",): "low",
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): "medium",
    }
    medium_files = [
        "hermes_cli/web_server.py",
        "tests/hermes_cli/test_web_server.py",
        "web/src/App.tsx",
        "web/src/i18n/en.ts",
        "web/src/i18n/types.ts",
        "web/src/i18n/zh.ts",
        "web/src/lib/api.ts",
        "web/src/pages/ProfilesPage.tsx",
    ]
    file_map = {
        ("docs",): ["docs/notes.md"],
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): medium_files,
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): ["docs/notes.md", *medium_files],
    }
    process_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr(
        "hermes_cli.update_doctor._run_medium_commit_sandbox_tests",
        lambda *args, **kwargs: {
            "status": "baseline-failure",
            "tests_run": ["pytest tests/hermes_cli/test_web_server.py -k profile -q", "pytest tests/hermes_cli/test_web_server.py -q"],
            "tests_passed": ["pytest tests/hermes_cli/test_web_server.py -k profile -q"],
            "test_failures": ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"],
            "medium_baseline_tests_run": ["pytest tests/hermes_cli/test_web_server.py -q"],
            "medium_baseline_failures": ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"],
            "medium_baseline_failure_matched": True,
            "medium_sandbox_prepared": False,
            "medium_sandbox_preparation": [],
            "branch_name": "update-doctor-medium-tested-4523965d",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._process_upstream_batch",
        lambda *args, **kwargs: process_calls.append(tuple(kwargs["batch"]["commits"])) or {
            "status": "pr-created",
            "reason": None,
            "risk": "medium",
            "commits": list(kwargs["batch"]["commits"]),
            "files": medium_files,
            "branch_name": "update-doctor-medium-tested-4523965d",
            "pr_url": "https://github.com/Vottam/hermes-agent/pull/999",
            "merge_commit": None,
            "published": {
                "pr_status": "available-not-created",
                "merge_status": "not-eligible",
                "next_step": "PR created, but auto-merge is blocked because risk is not low.",
                "pr_url": "https://github.com/Vottam/hermes-agent/pull/999",
                "merge_commit": None,
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "baseRefName": "main",
                "autoMergeRequest": None,
                "tests_run": [],
                "final_validation": {"status": "not-needed", "checks": []},
            },
        },
    )

    summary = _fallback_batch_to_individual_commits(report, root=REPO_ROOT, batch=batch, request_pr=True, request_auto_merge_low_risk=True)

    assert summary["status"] == "pr-created"
    assert summary["medium_tested_status"] == "baseline-failure"
    assert summary["medium_baseline_failure_matched"] is True
    assert summary["medium_pr_eligible"] is True
    assert summary["pr_urls"][0] == "https://github.com/Vottam/hermes-agent/pull/999"
    assert summary["pr_urls"].count("https://github.com/Vottam/hermes-agent/pull/999") >= 1
    assert summary["merge_status"] == "not-eligible"
    assert process_calls


def test_fallback_batch_to_individual_commits_blocks_when_baseline_differs(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    batch = {
        "index": 1,
        "size": 2,
        "commits": ["docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"],
        "first_commit": "docs",
        "last_commit": "4523965de9eb9a55ba7a67315adc3188c31eaec4",
    }
    risk_map = {
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): "high",
        ("docs",): "low",
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): "medium",
    }
    medium_files = [
        "hermes_cli/web_server.py",
        "tests/hermes_cli/test_web_server.py",
        "web/src/App.tsx",
        "web/src/i18n/en.ts",
        "web/src/i18n/types.ts",
        "web/src/i18n/zh.ts",
        "web/src/lib/api.ts",
        "web/src/pages/ProfilesPage.tsx",
    ]
    file_map = {
        ("docs",): ["docs/notes.md"],
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): medium_files,
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): ["docs/notes.md", *medium_files],
    }

    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr(
        "hermes_cli.update_doctor._run_medium_commit_sandbox_tests",
        lambda *args, **kwargs: {
            "status": "failed",
            "tests_run": ["pytest tests/hermes_cli/test_web_server.py -q"],
            "tests_passed": [],
            "test_failures": ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"],
            "medium_baseline_tests_run": ["pytest tests/hermes_cli/test_web_server.py -q"],
            "medium_baseline_failures": ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"],
            "medium_baseline_failure_matched": False,
            "medium_sandbox_prepared": False,
            "medium_sandbox_preparation": [],
            "branch_name": "update-doctor-medium-tested-4523965d",
        },
    )

    summary = _fallback_batch_to_individual_commits(report, root=REPO_ROOT, batch=batch, request_pr=False, request_auto_merge_low_risk=True)

    assert summary["status"] == "blocked"
    assert summary["fallback_blocked_commit"] == "4523965de9eb9a55ba7a67315adc3188c31eaec4"
    assert summary["fallback_blocked_reason"] == "medium-tests-failed"
    assert summary["medium_triage_used"] is True
    assert summary["medium_tested_status"] == "failed"
    assert summary["medium_baseline_failure_matched"] is False
    assert summary["medium_pr_eligible"] is False
    assert summary["medium_tests_run"] == ["pytest tests/hermes_cli/test_web_server.py -q"]
    assert summary["medium_tests_passed"] == []
    assert summary["medium_test_failures"] == ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"]
    assert summary["medium_baseline_tests_run"] == ["pytest tests/hermes_cli/test_web_server.py -q"]
    assert summary["medium_baseline_failures"] == ["pytest tests/hermes_cli/test_web_server.py -q (exit 1)"]


def test_prepare_medium_web_sandbox_links_checkout_node_modules(tmp_path: Path) -> None:
    root = tmp_path / "root"
    worktree = tmp_path / "worktree"
    (root / "web" / "node_modules").mkdir(parents=True)

    result = _prepare_medium_web_sandbox(root, worktree)

    linked = worktree / "web" / "node_modules"
    assert result["prepared"] is True
    assert result["preparation"] == ["linked web/node_modules"]
    assert result["failure"] is None
    assert linked.is_symlink()
    assert linked.resolve() == (root / "web" / "node_modules").resolve()


def test_prepare_medium_web_sandbox_does_not_overwrite_existing_node_modules(tmp_path: Path) -> None:
    root = tmp_path / "root"
    worktree = tmp_path / "worktree"
    (root / "web" / "node_modules").mkdir(parents=True)
    existing = worktree / "web" / "node_modules"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep")

    result = _prepare_medium_web_sandbox(root, worktree)

    assert result["prepared"] is False
    assert result["preparation"] == []
    assert result["failure"] is None
    assert sentinel.read_text() == "keep"
    assert existing.is_dir()
    assert not existing.is_symlink()


def test_medium_sandbox_does_not_prepare_without_web_build(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd and cmd[0].endswith("python") and cmd[1:3] == ["-m", "pytest"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.update_doctor._prepare_medium_web_sandbox", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sandbox prep should not run")))
    monkeypatch.setattr("hermes_cli.update_doctor.subprocess.run", fake_run)

    result = _run_medium_commit_sandbox_tests(root, "4523965de9eb9a55ba7a67315adc3188c31eaec4", ["pytest tests/hermes_cli/test_web_server.py -q"])

    assert result["status"] == "passed"
    assert result["medium_sandbox_prepared"] is False
    assert result["medium_sandbox_preparation"] == []


def test_medium_sandbox_prepares_web_build_with_linked_node_modules(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "web" / "node_modules").mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd and cmd[0] == "bash" and cmd[1:3] == ["-lc", "cd web && npm run build"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.update_doctor.subprocess.run", fake_run)

    result = _run_medium_commit_sandbox_tests(root, "4523965de9eb9a55ba7a67315adc3188c31eaec4", ["cd web && npm run build"])

    linked = result["medium_sandbox_preparation"]
    assert result["status"] == "passed"
    assert result["medium_sandbox_prepared"] is True
    assert linked == ["linked web/node_modules"]


def test_fallback_batch_to_individual_commits_marks_medium_tested_without_pr(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    batch = {
        "index": 1,
        "size": 2,
        "commits": ["docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"],
        "first_commit": "docs",
        "last_commit": "4523965de9eb9a55ba7a67315adc3188c31eaec4",
    }
    risk_map = {
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): "high",
        ("docs",): "low",
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): "medium",
    }
    medium_files = [
        "hermes_cli/web_server.py",
        "tests/hermes_cli/test_web_server.py",
        "web/src/App.tsx",
        "web/src/i18n/en.ts",
        "web/src/i18n/types.ts",
        "web/src/i18n/zh.ts",
        "web/src/lib/api.ts",
        "web/src/pages/ProfilesPage.tsx",
    ]
    file_map = {
        ("docs",): ["docs/notes.md"],
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): medium_files,
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): ["docs/notes.md", *medium_files],
    }

    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr(
        "hermes_cli.update_doctor._run_medium_commit_sandbox_tests",
        lambda *args, **kwargs: {
            "status": "passed",
            "tests_run": [
                "pytest tests/hermes_cli/test_web_server.py -k profile -q",
                "cd web && npm run build",
                "pytest tests/hermes_cli/test_web_server.py -q",
            ],
            "tests_passed": [
                "pytest tests/hermes_cli/test_web_server.py -k profile -q",
                "cd web && npm run build",
                "pytest tests/hermes_cli/test_web_server.py -q",
            ],
            "test_failures": [],
            "medium_sandbox_prepared": True,
            "medium_sandbox_preparation": ["linked web/node_modules"],
            "branch_name": "update-doctor-medium-tested-4523965d",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._process_upstream_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("medium-tested PR should not be created when --pr is disabled")),
    )

    summary = _fallback_batch_to_individual_commits(report, root=REPO_ROOT, batch=batch, request_pr=False, request_auto_merge_low_risk=True)

    assert summary["status"] == "medium-tested"
    assert summary["pr_status"] == "available-not-created"
    assert summary["merge_status"] == "not-needed"
    assert summary["medium_triage_used"] is True
    assert summary["medium_tested_status"] == "passed"
    assert summary["medium_pr_eligible"] is True
    assert summary["medium_sandbox_prepared"] is True
    assert summary["medium_sandbox_preparation"] == ["linked web/node_modules"]
    assert summary["medium_tests_run"] == [
        "pytest tests/hermes_cli/test_web_server.py -k profile -q",
        "cd web && npm run build",
        "pytest tests/hermes_cli/test_web_server.py -q",
    ]
    assert summary["medium_tests_passed"] == summary["medium_tests_run"]
    assert summary["medium_test_failures"] == []


def test_fallback_batch_to_individual_commits_creates_pr_for_medium_tested_commit_without_auto_merge(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    batch = {
        "index": 1,
        "size": 2,
        "commits": ["docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"],
        "first_commit": "docs",
        "last_commit": "4523965de9eb9a55ba7a67315adc3188c31eaec4",
    }
    risk_map = {
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): "high",
        ("docs",): "low",
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): "medium",
    }
    medium_files = [
        "hermes_cli/web_server.py",
        "tests/hermes_cli/test_web_server.py",
        "web/src/App.tsx",
        "web/src/i18n/en.ts",
        "web/src/i18n/types.ts",
        "web/src/i18n/zh.ts",
        "web/src/lib/api.ts",
        "web/src/pages/ProfilesPage.tsx",
    ]
    file_map = {
        ("docs",): ["docs/notes.md"],
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",): medium_files,
        ("docs", "4523965de9eb9a55ba7a67315adc3188c31eaec4"): ["docs/notes.md", *medium_files],
    }
    process_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr("hermes_cli.update_doctor._classify_upstream_batch_risk", lambda *args, **kwargs: risk_map[tuple(args[1])])
    monkeypatch.setattr("hermes_cli.update_doctor._batch_changed_files", lambda *args, **kwargs: file_map[tuple(args[1])])
    monkeypatch.setattr(
        "hermes_cli.update_doctor._run_medium_commit_sandbox_tests",
        lambda *args, **kwargs: {
            "status": "passed",
            "tests_run": [
                "pytest tests/hermes_cli/test_web_server.py -k profile -q",
                "cd web && npm run build",
                "pytest tests/hermes_cli/test_web_server.py -q",
            ],
            "tests_passed": [
                "pytest tests/hermes_cli/test_web_server.py -k profile -q",
                "cd web && npm run build",
                "pytest tests/hermes_cli/test_web_server.py -q",
            ],
            "test_failures": [],
            "medium_sandbox_prepared": True,
            "medium_sandbox_preparation": ["linked web/node_modules"],
            "branch_name": "update-doctor-medium-tested-4523965d",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.update_doctor._process_upstream_batch",
        lambda *args, **kwargs: process_calls.append(tuple(kwargs["batch"]["commits"])) or {
            "status": "created",
            "reason": None,
            "risk": "medium",
            "commits": kwargs["batch"]["commits"],
            "files": file_map[tuple(kwargs["batch"]["commits"])],
            "branch_name": "update-doctor-batch-upstream-01",
            "pr_url": "https://github.com/Vottam/hermes-agent/pull/202" if tuple(kwargs["batch"]["commits"]) == ("4523965de9eb9a55ba7a67315adc3188c31eaec4",) else "https://github.com/Vottam/hermes-agent/pull/101",
            "merge_commit": None,
            "published": {
                "pr_status": "created",
                "merge_status": "not-eligible",
                "next_step": "PR created, but auto-merge is blocked because risk is not low.",
                "final_validation": {"status": "not-run", "checks": []},
                "tests_run": [],
            },
        },
    )

    summary = _fallback_batch_to_individual_commits(report, root=REPO_ROOT, batch=batch, request_pr=True, request_auto_merge_low_risk=True)

    assert summary["status"] == "pr-created"
    assert summary["pr_status"] == "created"
    assert summary["merge_status"] == "not-eligible"
    assert summary["pr_urls"] == [
        "https://github.com/Vottam/hermes-agent/pull/101",
        "https://github.com/Vottam/hermes-agent/pull/202",
    ]
    assert summary["medium_triage_used"] is True
    assert summary["medium_tested_status"] == "passed"
    assert summary["medium_pr_eligible"] is True
    assert summary["medium_sandbox_prepared"] is True
    assert summary["medium_sandbox_preparation"] == ["linked web/node_modules"]
    assert summary["medium_tests_run"] == [
        "pytest tests/hermes_cli/test_web_server.py -k profile -q",
        "cd web && npm run build",
        "pytest tests/hermes_cli/test_web_server.py -q",
    ]
    assert summary["medium_tests_passed"] == summary["medium_tests_run"]
    assert summary["medium_test_failures"] == []
    assert process_calls == [
        ("docs",),
        ("4523965de9eb9a55ba7a67315adc3188c31eaec4",),
    ]


def test_batch_upstream_auto_merge_requires_mergeable_clean(monkeypatch) -> None:
    report = _sample_report(replay_result="passed", conflict=None)
    report["environment"]["behind"] = 19
    report["environment"]["origin_ahead_count"] = 19
    report["repair"] = {
        "mode": "repair",
        "result": "changed",
        "bucket": "test-desync",
        "repair_status": "applied",
        "repair_action": "patch-applied",
        "safety_level": "safe",
        "next_step": "done",
        "changed_files": ["docs/notes.md"],
    }
    monkeypatch.setattr("hermes_cli.update_doctor._ensure_fork_safe_publication", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.update_doctor._create_pr", lambda *args, **kwargs: ("https://github.com/Vottam/hermes-agent/pull/102", 102))
    monkeypatch.setattr(
        "hermes_cli.update_doctor._review_pr_for_auto_merge",
        lambda *args, **kwargs: {
            "mergeable": "CONFLICTING",
            "mergeStateStatus": "DIRTY",
            "autoMergeRequest": None,
            "baseRefName": "main",
            "headRefName": "update-doctor-pr-1",
            "url": "https://github.com/Vottam/hermes-agent/pull/102",
        },
    )
    monkeypatch.setattr("hermes_cli.update_doctor._merge_pr", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("merge should not run")))

    published = _publish_run_artifacts(report, root=REPO_ROOT, request_pr=True, request_auto_merge_low_risk=True)

    assert published["pr_status"] == "created"
    assert published["merge_status"] == "blocked"
    assert published["pr_url"] == "https://github.com/Vottam/hermes-agent/pull/102"
