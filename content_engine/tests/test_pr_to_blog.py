"""Tests for the PR-to-Blog pipeline (blog/pr_to_blog.py).

Tests the plan builder, title converter, state management, and the
dry-run path with mocked GitHub + blog pipeline calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure content_engine is on the path
ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))


# ── _pr_title_to_blog_title ─────────────────────────────────────────

def test_pr_title_strips_fix_prefix():
    from blog.pr_to_blog import _pr_title_to_blog_title
    result = _pr_title_to_blog_title("fix(kanban): hold reclaim while the worker is still alive")
    assert "Hold reclaim while the worker is still alive" in result
    assert "fix(kanban)" not in result
    assert result.endswith("a kanban fix")


def test_pr_title_strips_feat_prefix():
    from blog.pr_to_blog import _pr_title_to_blog_title
    result = _pr_title_to_blog_title("feat(profile): add interactive profile creation wizard")
    assert "Add interactive profile creation wizard" in result
    assert "feat(profile)" not in result
    # feat is framed as a feature, not a "fix".
    assert result.endswith("a profile feature")


def test_pr_title_strips_chore_prefix():
    from blog.pr_to_blog import _pr_title_to_blog_title
    result = _pr_title_to_blog_title("chore(release): add Sahil-SS9 to AUTHOR_MAP")
    assert "Add Sahil-SS9 to AUTHOR_MAP" in result
    assert "chore(release)" not in result


def test_pr_title_no_prefix():
    from blog.pr_to_blog import _pr_title_to_blog_title
    result = _pr_title_to_blog_title("update documentation for new API")
    assert "update documentation for new API" in result


def test_pr_title_drops_salvage_ref_and_is_distinct():
    """Two different PRs must not collide on the old blanket headline."""
    from blog.pr_to_blog import _pr_title_to_blog_title
    a = _pr_title_to_blog_title("fix(kanban): machine-global singleton lock (#41448)")
    b = _pr_title_to_blog_title("fix(kanban): hold reclaim while the worker is still alive")
    assert "(#41448)" not in a
    assert a != b


# ── _build_pr_plan ──────────────────────────────────────────────────

def test_build_pr_plan_has_signals():
    """The generator requires plan['signals'] to be non-empty."""
    from blog.pr_to_blog import _build_pr_plan
    pr = {
        "number": 49064,
        "title": "fix(kanban): hold reclaim while the worker is still alive",
        "body": "Salvage of #44909 by @Sahil-SS9 onto current main.",
        "mergedAt": "2026-06-19T14:38:10Z",
        "additions": 257,
        "deletions": 1,
        "files": [{"path": "hermes_cli/kanban_db.py"}, {"path": "tests/hermes_cli/test_kanban_db.py"}],
        "commits": [{"oid": "b9e521da23"}],
        "url": "https://github.com/NousResearch/hermes-agent/pull/49064",
    }
    plan = _build_pr_plan(pr, "NousResearch/hermes-agent")
    assert len(plan["signals"]) == 1
    sig = plan["signals"][0]
    assert sig["signal_type"] == "hermes_pr"
    assert sig["sha"] == "b9e521da23"
    assert sig["pr_number"] == 49064
    assert "hold reclaim" in sig["summary"]


def test_build_pr_plan_topic_id():
    from blog.pr_to_blog import _build_pr_plan
    pr = {"number": 42, "title": "fix: something", "body": "", "mergedAt": "",
          "additions": 0, "deletions": 0, "files": [], "commits": [], "url": ""}
    plan = _build_pr_plan(pr, "NousResearch/hermes-agent")
    assert plan["topic_id"] == "pr-hermes-agent-42"


def test_build_pr_plan_stream_is_builder():
    from blog.pr_to_blog import _build_pr_plan
    pr = {"number": 1, "title": "fix: test", "body": "", "mergedAt": "",
          "additions": 0, "deletions": 0, "files": [], "commits": [], "url": ""}
    plan = _build_pr_plan(pr, "NousResearch/hermes-agent")
    assert plan["stream"] == "builder"


def test_build_pr_plan_source_is_gitradar():
    from blog.pr_to_blog import _build_pr_plan
    pr = {"number": 1, "title": "fix: test", "body": "", "mergedAt": "",
          "additions": 0, "deletions": 0, "files": [], "commits": [], "url": ""}
    plan = _build_pr_plan(pr, "NousResearch/hermes-agent")
    assert plan["source"] == "gitradar"


def test_build_pr_plan_handles_empty_commits():
    """Should not crash when commits list is empty."""
    from blog.pr_to_blog import _build_pr_plan
    pr = {"number": 1, "title": "fix: test", "body": "x", "mergedAt": "2026-06-19",
          "additions": 10, "deletions": 2, "files": [{"path": "a.py"}],
          "commits": [], "url": "http://example.com"}
    plan = _build_pr_plan(pr, "NousResearch/hermes-agent")
    assert plan["signals"][0]["sha"] == ""


def test_build_pr_plan_handles_missing_body():
    from blog.pr_to_blog import _build_pr_plan
    pr = {"number": 1, "title": "fix: test", "body": None, "mergedAt": "",
          "additions": 0, "deletions": 0, "files": [], "commits": [], "url": ""}
    plan = _build_pr_plan(pr, "NousResearch/hermes-agent")
    assert plan["signals"][0]["body"] == ""


# ── State management ────────────────────────────────────────────────

def test_state_load_returns_empty_when_no_file(tmp_path):
    from blog.pr_to_blog import _load_state, STATE_FILE
    with patch("blog.pr_to_blog.STATE_FILE", tmp_path / "state.json"):
        state = _load_state()
        assert state["processed_prs"] == {}
        assert state["last_run"] is None


def test_state_save_and_load_roundtrip(tmp_path):
    from blog.pr_to_blog import _save_state, _load_state
    state_file = tmp_path / "state.json"
    with patch("blog.pr_to_blog.STATE_FILE", state_file):
        state = {"processed_prs": {}, "last_run": None}
        _mark = state["processed_prs"]
        _mark["NousResearch/hermes-agent#49064"] = {"slug": "test", "processed_at": "2026-06-28"}
        _save_state(state)
        loaded = _load_state()
        assert "NousResearch/hermes-agent#49064" in loaded["processed_prs"]
        assert loaded["last_run"] is not None


def test_is_processed_returns_false_for_new_pr(tmp_path):
    from blog.pr_to_blog import _is_processed, _load_state
    with patch("blog.pr_to_blog.STATE_FILE", tmp_path / "state.json"):
        state = _load_state()
        assert not _is_processed(state, "NousResearch/hermes-agent", 99999)


def test_is_processed_returns_true_for_known_pr(tmp_path):
    from blog.pr_to_blog import _is_processed, _load_state
    state = {"processed_prs": {"NousResearch/hermes-agent#49064": {}}, "last_run": None}
    with patch("blog.pr_to_blog.STATE_FILE", tmp_path / "state.json"):
        assert _is_processed(state, "NousResearch/hermes-agent", 49064)


def test_mark_processed_adds_entry(tmp_path):
    from blog.pr_to_blog import _mark_processed, _load_state
    with patch("blog.pr_to_blog.STATE_FILE", tmp_path / "state.json"):
        state = _load_state()
        _mark_processed(state, "NousResearch/hermes-agent", 49064, "test-slug")
        key = "NousResearch/hermes-agent#49064"
        assert key in state["processed_prs"]
        assert state["processed_prs"][key]["slug"] == "test-slug"


# ── X post builder ──────────────────────────────────────────────────

def test_post_to_x_dry_run_does_not_call_xurl():
    from blog.pr_to_blog import post_to_x
    result = post_to_x("Test title", "test-slug",
                       "https://github.com/NousResearch/hermes-agent/pull/49064",
                       dry_run=True)
    assert result is None


def test_post_to_x_builds_tweet_under_280():
    from blog.pr_to_blog import post_to_x
    # Use a short title so we can check the tweet builds correctly
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0,
                                          stdout='{"data":{"id":"123"}}',
                                          stderr="")
        result = post_to_x("Short title", "short-slug",
                           "https://github.com/NousResearch/hermes-agent/pull/1",
                           dry_run=False)
        # Check xurl was called
        assert mock_run.called
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "xurl"
        assert call_args[0][0][1] == "post"


def test_post_to_x_truncates_long_title():
    from blog.pr_to_blog import post_to_x
    long_title = "A" * 300  # way over 280 chars total
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0,
                                          stdout='{"data":{"id":"456"}}',
                                          stderr="")
        post_to_x(long_title, "long-slug",
                  "https://github.com/NousResearch/hermes-agent/pull/2",
                  dry_run=False)
        call_args = mock_run.call_args
        tweet_text = call_args[0][0][2]
        assert len(tweet_text) <= 280


# ── LinkedIn draft ──────────────────────────────────────────────────

def test_draft_linkedin_post_creates_file(tmp_path):
    from blog.pr_to_blog import draft_linkedin_post, LINKEDIN_DRAFTS_DIR
    with patch("blog.pr_to_blog.LINKEDIN_DRAFTS_DIR", tmp_path):
        pr = {"number": 49064, "url": "https://github.com/NousResearch/hermes-agent/pull/49064",
               "additions": 257, "deletions": 1, "files": [{"path": "a.py"}]}
        path = draft_linkedin_post("Test title", "test-slug", pr,
                                   "NousResearch/hermes-agent", dry_run=True)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "LinkedIn Draft" in content
        assert "49064" in content
        assert "algorithmiccompass.com" in content


# ── fetch_merged_prs (mocked gh CLI) ────────────────────────────────

def test_fetch_merged_prs_parses_gh_output():
    from blog.pr_to_blog import fetch_merged_prs
    fake_json = json.dumps([
        {"number": 49064, "title": "fix(kanban): test", "mergedAt": "2026-06-19"},
    ])
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=fake_json, stderr="")
        prs = fetch_merged_prs("NousResearch/hermes-agent")
        assert len(prs) == 1
        assert prs[0]["number"] == 49064


def test_fetch_merged_prs_returns_empty_on_error():
    from blog.pr_to_blog import fetch_merged_prs
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        prs = fetch_merged_prs("NousResearch/hermes-agent")
        assert prs == []


def test_fetch_merged_prs_returns_empty_on_empty_output():
    from blog.pr_to_blog import fetch_merged_prs
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        prs = fetch_merged_prs("NousResearch/hermes-agent")
        assert prs == []

# ── Source-aware PR imagery / approval governance ───────────────────────────

def test_extract_pr_infographic_url_from_markdown_image():
    from blog.pr_to_blog import _extract_pr_infographic_url
    pr = {"body": "Here is the diagram:\n![async fix](https://github.com/user-attachments/assets/abc123)"}
    assert _extract_pr_infographic_url(pr) == "https://github.com/user-attachments/assets/abc123"


def test_generate_blog_post_requests_approval_not_publish(monkeypatch, tmp_path):
    import blog.pr_to_blog as p2b

    pr = {
        "number": 49064,
        "title": "fix: approval safe",
        "body": "",
        "mergedAt": "2026-06-19T14:38:10Z",
        "additions": 1,
        "deletions": 0,
        "files": [{"path": "a.py"}],
        "commits": [{"oid": "abc"}],
        "url": "https://github.com/NousResearch/hermes-agent/pull/49064",
    }
    repo = tmp_path / "blog"
    (repo / "src/content/blog").mkdir(parents=True)
    (repo / "public/blog").mkdir(parents=True)

    draft = {
        "title": "Approval Safe", "description": "d", "body_md": "body",
        "slug": "approval-safe", "tier": "builder", "tags": [],
        "format": "essay", "source": "gitradar",
    }
    calls = []

    monkeypatch.setattr("blog.blog_generator.write_with_gate", lambda plan, stream, strict_review=False: draft)
    monkeypatch.setattr(p2b, "_download_pr_infographic", lambda pr, slug_hint: "/tmp/pr-image.png")

    def fake_assemble(d, images, repo=None):
        assert images["hero_path"] == "/tmp/pr-image.png"
        mdx = Path(repo) / "src/content/blog/approval-safe.mdx"
        mdx.write_text("---\ntitle: \"Approval Safe\"\napproved: false\n---\nbody")
        return mdx

    monkeypatch.setattr("blog.blog_assembler.assemble", fake_assemble)
    monkeypatch.setattr("blog.blog_publisher.stage_draft", lambda mdx, repo=None: "approval-safe")
    monkeypatch.setattr("blog.blog_approval.request", lambda slug, title, stream, tier, mdx_path: calls.append((slug, mdx_path)) or "aid")
    monkeypatch.setattr("config.SAHILBLOG_REPO", repo)

    result = p2b.generate_blog_post(pr, "NousResearch/hermes-agent", dry_run=False)
    assert result["approved"] is False
    assert result["approval_id"] == "aid"
    assert result["image_source"] == "upstream_pr_infographic"
    assert calls == [("approval-safe", str(repo / "src/content/blog/approval-safe.mdx"))]
