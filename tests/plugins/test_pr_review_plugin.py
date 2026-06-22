from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.pr_review import cli as pr_review_cli
from plugins.pr_review import core, evals, register


def test_register_adds_cli_command():
    mgr = PluginManager()
    manifest = PluginManifest(name="pr-review")
    ctx = PluginContext(manifest, mgr)

    register(ctx)

    assert "pr-review" in mgr._cli_commands
    entry = mgr._cli_commands["pr-review"]
    assert entry["plugin"] == "pr-review"
    assert callable(entry["setup_fn"])
    assert callable(entry["handler_fn"])


def test_pr_review_is_enabled_as_first_party_bundled_cli_by_default():
    assert "pr-review" in DEFAULT_CONFIG["plugins"]["enabled"]


def test_default_docs_stay_on_broad_conventions_not_larry_specific_docs():
    assert "AGENTS.md" in core.DEFAULT_DOC_PATHS
    assert ".github/copilot-instructions.md" in core.DEFAULT_DOC_PATHS
    assert "docs/ARCHITECTURE.md" not in core.DEFAULT_DOC_PATHS
    assert "docs/WORKFLOW.md" not in core.DEFAULT_DOC_PATHS


def test_parse_pr_ref_accepts_url_and_short_form():
    url = core.parse_pr_ref("https://github.com/NousResearch/hermes-agent/pull/123")
    short = core.parse_pr_ref("NousResearch/hermes-agent#123")

    assert url == short
    assert url.full_name == "NousResearch/hermes-agent"
    assert url.number == 123
    assert url.storage_name == "NousResearch_hermes-agent"


def test_parse_pr_ref_rejects_unknown_shape():
    try:
        core.parse_pr_ref("not-a-pr")
    except ValueError as exc:
        assert "GitHub URL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_filter_files_skips_generated_defaults():
    files = [
        {"filename": "src/app.ts"},
        {"filename": "package-lock.json"},
        {"filename": "web/dist/bundle.js"},
        {"filename": "src/generated/client.ts"},
    ]

    included, skipped = core.filter_files(files)

    assert [f["filename"] for f in included] == ["src/app.ts"]
    assert [f["filename"] for f in skipped] == [
        "package-lock.json",
        "web/dist/bundle.js",
        "src/generated/client.ts",
    ]
    assert all(f["skip_reason"] == "ignored_path" for f in skipped)


def test_fetch_pr_files_slurps_and_flattens_paginated_api(monkeypatch):
    calls = []

    def fake_run_gh_json(args, timeout=120):
        calls.append(args)
        return [[{"filename": "a.py"}], [{"filename": "b.py"}]]

    monkeypatch.setattr(core, "run_gh_json", fake_run_gh_json)

    files = core.fetch_pr_files(core.PullRequestRef("owner", "repo", 1))

    assert [f["filename"] for f in files] == ["a.py", "b.py"]
    assert calls == [["api", "repos/owner/repo/pulls/1/files", "--paginate", "--slurp"]]


def test_build_review_diff_excludes_skipped_files_when_filtering():
    full_diff = "diff --git a/src/app.py b/src/app.py\n+ok\n\ndiff --git a/dist/app.js b/dist/app.js\n+generated"
    included = [{"filename": "src/app.py", "patch": "@@\n+ok"}]
    skipped = [{"filename": "dist/app.js", "skip_reason": "ignored_path"}]

    review_diff = core.build_review_diff(full_diff, included, skipped)

    assert "src/app.py" in review_diff
    assert "+ok" in review_diff
    assert "dist/app.js" not in review_diff
    assert "generated" not in review_diff


def test_build_review_input_records_truncation_and_docs():
    metadata = {
        "number": 7,
        "title": "Add thing",
        "baseRefName": "main",
        "headRefName": "feat/thing",
        "headRefOid": "abcdef1234567890",
        "changedFiles": 1,
        "additions": 10,
        "deletions": 2,
        "url": "https://github.com/o/r/pull/7",
        "statusCheckRollup": [
            {
                "name": "validate",
                "workflowName": "CI",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "detailsUrl": "https://github.com/o/r/actions/runs/1",
            },
            {
                "name": "e2e",
                "workflowName": "CI",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
            },
        ],
    }

    context, manifest = core.build_review_input(
        metadata=metadata,
        diff="x" * 1200,
        docs={"AGENTS.md": "Follow the repo rules."},
        included_files=[{"filename": "src/app.ts"}],
        skipped_files=[{"filename": "dist/app.js", "skip_reason": "ignored_path"}],
        max_diff_chars=1000,
    )

    assert "Follow the repo rules" in context
    assert "[TRUNCATED" in context
    assert manifest["diff_truncated"] is True
    assert manifest["docs_loaded"] == ["AGENTS.md"]
    assert manifest["included_files"] == ["src/app.ts"]
    assert manifest["skipped_files"] == [{"filename": "dist/app.js", "reason": "ignored_path"}]
    assert manifest["check_context"]["counts"] == {"success": 1, "failure": 1}
    assert "Observed GitHub checks" in context


def test_write_artifacts_renders_markdown(tmp_path: Path):
    manifest = {
        "head_sha": "abcdef1234567890",
        "docs_loaded": ["AGENTS.md"],
        "skipped_files": [],
        "diff_truncated": False,
    }
    review = {
        "verdict": "comment",
        "risk": "medium",
        "summary": "One issue found.",
        "findings": [
            {
                "severity": "warning",
                "path": "src/app.ts",
                "line": 42,
                "title": "Guard missing",
                "evidence": "foo can be None",
                "why_it_matters": "runtime crash",
                "suggested_fix": "add guard",
                "confidence": "high",
            }
        ],
        "verification_notes": ["Fetched PR diff through gh."],
    }

    paths = core.write_artifacts(tmp_path, context="ctx", manifest=manifest, review=review)

    rendered = Path(paths["review"]).read_text()
    findings = json.loads(Path(paths["findings"]).read_text())
    assert "<!-- hermes-pr-review:summary:v1 -->" in rendered
    assert "Guard missing" in rendered
    assert "`src/app.ts:42`" in rendered
    assert findings["risk"] == "medium"
    assert findings["findings"][0]["fingerprint"]
    assert findings["findings"][0]["category"] == "correctness"
    assert findings["review_fingerprint"]
    assert Path(paths["trace"]).exists()
    assert "GitHub checks observed" in rendered


def test_config_from_base_branch_extends_docs_and_ignore_patterns(monkeypatch):
    ref = core.PullRequestRef("owner", "repo", 1)

    def fake_fetch(_ref, path, base_ref):
        assert base_ref == "main"
        return {
            ".github/hermes-pr-reviewer.json": json.dumps(
                {
                    "extraDocPaths": ["docs/ARCHITECTURE.md", "../unsafe.md"],
                    "ignorePatterns": ["**/snapshots/**", "/absolute/**"],
                }
            ),
            "AGENTS.md": "Agent rules",
            "docs/ARCHITECTURE.md": "Architecture rules",
        }.get(path)

    monkeypatch.setattr(core, "fetch_file_from_base", fake_fetch)
    monkeypatch.setattr(core, "fetch_instruction_glob_from_base", lambda _ref, _base: {})

    cfg = core.load_reviewer_config(ref, "main")
    docs = core.collect_trusted_docs(ref, "main", extra_doc_paths=cfg["extra_doc_paths"])
    included, skipped = core.filter_files(
        [{"filename": "src/app.py"}, {"filename": "tests/snapshots/out.txt"}],
        patterns=(*core.DEFAULT_IGNORE_PATTERNS, *cfg["ignore_patterns"]),
    )

    assert cfg["config_path"] == ".github/hermes-pr-reviewer.json"
    assert cfg["extra_doc_paths"] == ["docs/ARCHITECTURE.md"]
    assert cfg["ignore_patterns"] == ["**/snapshots/**"]
    assert "docs/ARCHITECTURE.md" in docs
    assert [f["filename"] for f in included] == ["src/app.py"]
    assert [f["filename"] for f in skipped] == ["tests/snapshots/out.txt"]


def test_normalize_review_caps_findings_and_adds_fingerprints():
    raw = {
        "verdict": "bad",
        "risk": "urgent",
        "summary": "Summary",
        "findings": [
            {
                "severity": "critical",
                "path": f"src/{idx}.py",
                "line": "not-int",
                "title": "Bug",
                "evidence": "Evidence",
                "why_it_matters": "Breaks",
                "suggested_fix": "Fix",
                "confidence": "certain",
            }
            for idx in range(core.MAX_FINDINGS + 2)
        ],
        "verification_notes": ["note"],
    }

    review = core.normalize_review(raw)

    assert review["verdict"] == "comment"
    assert review["risk"] == "medium"
    assert len(review["findings"]) == core.MAX_FINDINGS
    assert review["findings"][0]["confidence"] == "medium"
    assert review["findings"][0]["line"] is None
    assert review["findings"][0]["category"] == "correctness"
    assert review["findings"][0]["blocking"] is True
    assert review["findings"][0]["fingerprint"]
    assert review["review_fingerprint"]


def test_normalize_review_clamps_approve_to_advisory_comment():
    review = core.normalize_review(
        {
            "verdict": "approve",
            "risk": "low",
            "summary": "No issues.",
            "findings": [],
            "verification_notes": [],
        }
    )

    assert review["verdict"] == "comment"
    assert review["model_verdict"] == "approve"


def test_post_or_update_summary_comment_updates_existing_without_duplicate(monkeypatch):
    calls = []
    ref = core.PullRequestRef("owner", "repo", 7)
    body = f"{core.SUMMARY_COMMENT_MARKER}\n## Hermes PR Review\nnew"

    def fake_run_gh_json(args, timeout=120):
        calls.append(args)
        if args == ["api", "user", "--jq", ".login"]:
            return "hermes-bot"
        if args[:2] == ["api", "repos/owner/repo/issues/comments/123"]:
            return {"id": 123, "body": body, "html_url": "new-url"}
        raise AssertionError(f"unexpected gh api call: {args}")

    def fake_run_gh_paginated_json(args, timeout=120):
        calls.append(args)
        if args[:2] == ["api", "repos/owner/repo/issues/7/comments"]:
            return [
                {
                    "id": 123,
                    "body": f"{core.SUMMARY_COMMENT_MARKER}\nold",
                    "html_url": "old-url",
                    "user": {"login": "hermes-bot"},
                }
            ]
        raise AssertionError(f"unexpected paginated gh api call: {args}")

    monkeypatch.setattr(core, "run_gh_json", fake_run_gh_json)
    monkeypatch.setattr(core, "run_gh_paginated_json", fake_run_gh_paginated_json)

    result = core.post_or_update_summary_comment(ref, body)

    assert result == {"action": "updated", "comment_id": "123", "url": "new-url"}
    assert any("PATCH" in call for call in calls)
    assert not any("POST" in call for call in calls)


def test_post_or_update_summary_comment_ignores_marker_from_other_author(monkeypatch):
    calls = []
    ref = core.PullRequestRef("owner", "repo", 7)
    body = f"{core.SUMMARY_COMMENT_MARKER}\n## Hermes PR Review\nnew"

    def fake_run_gh_json(args, timeout=120):
        calls.append(args)
        if args == ["api", "user", "--jq", ".login"]:
            return "hermes-bot"
        if args[:2] == ["api", "repos/owner/repo/issues/7/comments"] and "POST" in args:
            return {"id": 456, "body": body, "html_url": "created-url"}
        raise AssertionError(f"unexpected gh api call: {args}")

    def fake_run_gh_paginated_json(args, timeout=120):
        calls.append(args)
        if args[:2] == ["api", "repos/owner/repo/issues/7/comments"]:
            return [
                {
                    "id": 123,
                    "body": f"{core.SUMMARY_COMMENT_MARKER}\nspoof",
                    "html_url": "old-url",
                    "user": {"login": "someone-else"},
                }
            ]
        raise AssertionError(f"unexpected paginated gh api call: {args}")

    monkeypatch.setattr(core, "run_gh_json", fake_run_gh_json)
    monkeypatch.setattr(core, "run_gh_paginated_json", fake_run_gh_paginated_json)

    result = core.post_or_update_summary_comment(ref, body)

    assert result == {"action": "created", "comment_id": "456", "url": "created-url"}
    assert not any("PATCH" in call for call in calls)
    assert any("POST" in call for call in calls)


def test_cmd_review_posts_comment_with_mocked_gh_and_llm(monkeypatch, tmp_path):
    ref = core.PullRequestRef("owner", "repo", 9)
    monkeypatch.setattr(core, "artifacts_root", lambda: tmp_path)
    monkeypatch.setattr(core, "fetch_pr_metadata", lambda _ref: {
        "number": 9,
        "title": "Fix bug",
        "baseRefName": "main",
        "headRefName": "feat/bug",
        "headRefOid": "abc123def456",
        "changedFiles": 1,
        "additions": 3,
        "deletions": 1,
        "url": "https://github.com/owner/repo/pull/9",
    })
    monkeypatch.setattr(core, "fetch_pr_diff", lambda _ref: "diff --git a/app.py b/app.py")
    monkeypatch.setattr(core, "fetch_pr_files", lambda _ref: [{"filename": "app.py"}])
    monkeypatch.setattr(core, "load_reviewer_config", lambda _ref, _base: {"config_path": None, "extra_doc_paths": [], "ignore_patterns": [], "config_error": None})
    monkeypatch.setattr(core, "collect_trusted_docs", lambda _ref, _base, extra_doc_paths=(): {"AGENTS.md": "Rules"})
    posted = {}
    monkeypatch.setattr(core, "post_or_update_summary_comment", lambda _ref, body: posted.setdefault("result", {"action": "created", "comment_id": "1", "url": "url"}) if core.SUMMARY_COMMENT_MARKER in body else None)

    class FakeLlm:
        def complete_structured(self, **kwargs):
            assert kwargs["purpose"] == "pr-review.review"
            return SimpleNamespace(parsed={"verdict": "comment", "risk": "low", "summary": "Looks fine", "findings": [], "verification_notes": ["mocked"]})

    args = argparse.Namespace(
        pr="owner/repo#9",
        no_llm=False,
        dry_run=False,
        max_diff_chars=5000,
        post_comment=True,
        json=True,
    )

    rc = pr_review_cli._cmd_review(args, ctx=SimpleNamespace(llm=FakeLlm()))

    assert rc == 0
    assert posted["result"]["action"] == "created"
    review_path = tmp_path / ref.storage_name / "9" / "abc123def456" / "review.md"
    assert review_path.exists()


def test_public_eval_manifest_parses_and_summarizes_seed_corpus():
    manifest = evals.load_eval_manifest()
    summary = evals.summarize_eval_manifest(manifest)

    assert manifest.schema_version == 1
    assert summary["case_count"] >= 8
    assert set(summary["categories"]) == evals.CASE_CATEGORIES
    assert summary["categories"]["small-docs"] == 1
    assert summary["observed_check_status"]["failure"] >= 1
    assert summary["totals"]["changed_files"] > 0
    assert all("#" in ref and not ref.startswith(("larry/", "NousResearch/")) for ref in summary["prs"])


def test_eval_manifest_rejects_duplicate_case_ids():
    raw = {
        "schema_version": 1,
        "name": "bad",
        "description": "bad",
        "cases": [
            {
                "id": "dup",
                "pr": "owner/repo#1",
                "category": "small-docs",
                "title": "One",
                "observed_head_sha": "abc",
                "observed_check_status": {"success": 1},
            },
            {
                "id": "dup",
                "pr": "owner/repo#2",
                "category": "backend",
                "title": "Two",
                "observed_head_sha": "def",
                "observed_check_status": {"success": 1},
            },
        ],
    }

    try:
        evals.parse_eval_manifest(raw)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected duplicate case id to fail")


def test_cmd_eval_manifest_prints_summary(capsys):
    args = argparse.Namespace(pr_review_command="eval-manifest", manifest=None, json=False)

    rc = pr_review_cli.pr_review_command(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "public-oss-pr-review-beta" in captured.out
    assert "Cases:" in captured.out
    assert "large-stress" in captured.out


def test_cmd_review_dry_run_fixture_path_uses_public_manifest_without_posting(monkeypatch, tmp_path: Path):
    manifest = evals.load_eval_manifest()
    case = next(item for item in manifest.cases if item.category == "small-docs")
    ref = core.parse_pr_ref(case.pr)
    monkeypatch.setattr(core, "artifacts_root", lambda: tmp_path)
    monkeypatch.setattr(core, "fetch_pr_metadata", lambda _ref: {
        "number": ref.number,
        "title": case.title,
        "baseRefName": "main",
        "headRefName": "docs/fix-link",
        "headRefOid": case.observed_head_sha,
        "changedFiles": case.changed_files,
        "additions": case.additions,
        "deletions": case.deletions,
        "url": f"https://github.com/{ref.full_name}/pull/{ref.number}",
        "statusCheckRollup": [
            {"name": status, "status": "COMPLETED", "conclusion": status.upper()}
            for status, count in case.observed_check_status.items()
            for _ in range(count)
        ],
    })
    monkeypatch.setattr(core, "fetch_pr_diff", lambda _ref: "diff --git a/docs.md b/docs.md\n+fixed link")
    monkeypatch.setattr(core, "fetch_pr_files", lambda _ref: [{"filename": "docs.md"}])
    monkeypatch.setattr(core, "load_reviewer_config", lambda _ref, _base: {"config_path": None, "extra_doc_paths": [], "ignore_patterns": [], "config_error": None})
    monkeypatch.setattr(core, "collect_trusted_docs", lambda _ref, _base, extra_doc_paths=(): {"README.md": "Project docs"})

    def unexpected_post(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("dry-run/no-post eval path must not write GitHub comments")

    monkeypatch.setattr(core, "post_or_update_summary_comment", unexpected_post)
    args = argparse.Namespace(
        pr=case.pr,
        no_llm=True,
        dry_run=True,
        max_diff_chars=5000,
        post_comment=False,
        json=True,
        mode="balanced",
    )

    rc = pr_review_cli._cmd_review(args, ctx=None)

    assert rc == 0
    review_path = tmp_path / ref.storage_name / str(ref.number) / case.observed_head_sha[:12] / "review.md"
    manifest_path = review_path.with_name("context-manifest.json")
    assert review_path.exists()
    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["head_sha"] == case.observed_head_sha
    assert saved_manifest["check_context"]["observed"] is True
