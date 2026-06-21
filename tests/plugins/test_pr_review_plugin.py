from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.pr_review import cli as pr_review_cli
from plugins.pr_review import core, register


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
    assert findings["review_fingerprint"]


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
    assert review["findings"][0]["fingerprint"]
    assert review["review_fingerprint"]


def test_post_or_update_summary_comment_updates_existing_without_duplicate(monkeypatch):
    calls = []
    ref = core.PullRequestRef("owner", "repo", 7)
    body = f"{core.SUMMARY_COMMENT_MARKER}\n## Hermes PR Review\nnew"

    def fake_run_gh_json(args, timeout=120):
        calls.append(args)
        if args[:2] == ["api", "repos/owner/repo/issues/7/comments"] and "--method" not in args:
            return [{"id": 123, "body": f"{core.SUMMARY_COMMENT_MARKER}\nold", "html_url": "old-url"}]
        if args[:2] == ["api", "repos/owner/repo/issues/comments/123"]:
            return {"id": 123, "body": body, "html_url": "new-url"}
        raise AssertionError(f"unexpected gh api call: {args}")

    monkeypatch.setattr(core, "run_gh_json", fake_run_gh_json)

    result = core.post_or_update_summary_comment(ref, body)

    assert result == {"action": "updated", "comment_id": "123", "url": "new-url"}
    assert any("PATCH" in call for call in calls)
    assert not any("POST" in call for call in calls)


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
