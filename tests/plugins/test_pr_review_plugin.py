from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
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
