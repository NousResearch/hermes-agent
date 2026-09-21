"""Regression tests for blog publisher failure semantics."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def approved_mdx(tmp_path):
    """An MDX file whose frontmatter already reads approved:true."""
    mdx = tmp_path / "src/content/blog/retry-post.mdx"
    mdx.parent.mkdir(parents=True)
    mdx.write_text('---\ntitle: "Retry Post"\napproved: true\n---\nbody')
    return mdx


def test_flip_approved_states(tmp_path):
    from blog import blog_publisher as bp

    draft = tmp_path / "draft.mdx"
    draft.write_text('---\ntitle: "D"\napproved: false\n---\nbody')
    assert bp._flip_approved(draft) == "flipped"
    assert bp._flip_approved(draft) == "already_flipped"

    nofield = tmp_path / "nofield.mdx"
    nofield.write_text('---\ntitle: "N"\n---\nbody')
    assert bp._flip_approved(nofield) == "missing_field"


def test_approve_retry_when_already_flipped(monkeypatch, tmp_path, approved_mdx):
    """The 2026-09-20 flip_failed loop: flip already written, build+push fine,
    publish() must succeed and clear the tracker instead of erroring every tick."""
    from blog import blog_publisher as bp
    from blog import blog_approval

    repo = tmp_path
    monkeypatch.setattr(bp, "SAHILBLOG_REPO", str(repo))
    monkeypatch.setattr(bp, "_run_build", lambda repo_path: 0)
    monkeypatch.setattr(bp, "_git", lambda repo_path, *args: MagicMock(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(bp, "_git_push", lambda repo_path: MagicMock(returncode=0, stdout="", stderr=""))

    result = bp.approve("retry-post", repo=str(repo))
    assert result["status"] == "ok"

    # End-to-end: the tracker entry is cleared on the retried publish.
    monkeypatch.setattr(blog_approval, "TRACKER_PATH", tmp_path / "pending_approvals.jsonl")
    blog_approval._write_tracker([{
        "slug": "retry-post", "title": "Retry Post", "stream": "builder",
        "tier": "builder", "mdx_path": str(approved_mdx), "status": "approved",
    }])
    published = blog_approval.publish("retry-post")
    assert published["status"] == "ok"
    assert blog_approval._read_tracker() == []


def test_approve_reports_push_failure(monkeypatch, tmp_path):
    from blog import blog_publisher as bp

    repo = tmp_path
    mdx = repo / "src/content/blog/test-post.mdx"
    mdx.parent.mkdir(parents=True)
    mdx.write_text('---\ntitle: "Test Post"\napproved: false\n---\nbody')

    monkeypatch.setattr(bp, "_flip_approved", lambda path: True)
    monkeypatch.setattr(bp, "_run_build", lambda repo_path: 0)
    monkeypatch.setattr(bp, "_git", lambda repo_path, *args: MagicMock(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(bp, "_git_push", lambda repo_path: MagicMock(returncode=1, stdout="", stderr="push denied"))

    result = bp.approve("test-post", repo=str(repo))
    assert result["status"] == "push_failed"
    assert result["push_rc"] == 1
    assert "push denied" in result["error"]


def test_stage_draft_raises_on_git_add_failure(monkeypatch, tmp_path):
    from blog import blog_publisher as bp

    mdx = tmp_path / "src/content/blog/test-post.mdx"
    mdx.parent.mkdir(parents=True)
    mdx.write_text('---\ntitle: "Test Post"\n---\nbody')
    monkeypatch.setattr(bp, "_git", lambda repo_path, *args: MagicMock(returncode=1, stdout="", stderr="add failed"))

    try:
        bp.stage_draft(str(mdx), repo=str(tmp_path))
    except RuntimeError as exc:
        assert "add failed" in str(exc)
    else:
        raise AssertionError("stage_draft should raise on git add failure")


def test_stage_draft_raises_when_excluded(monkeypatch, tmp_path, blog_exclusions):
    from blog import blog_publisher as bp

    mdx = tmp_path / "src/content/blog/cheap-first-model-routing.mdx"
    mdx.parent.mkdir(parents=True)
    mdx.write_text('---\ntitle: "Cheap-first model routing"\n---\nbody')
    monkeypatch.setattr(bp, "_git", lambda repo_path, *args: MagicMock(returncode=0, stdout="", stderr=""))

    try:
        bp.stage_draft(str(mdx), repo=str(tmp_path))
    except bp.ExcludedContentError as exc:
        assert "Excluded by policy" in str(exc)
    else:
        raise AssertionError("stage_draft should raise on excluded content")
