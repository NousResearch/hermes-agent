"""Regression tests for blog publisher failure semantics."""
from pathlib import Path
from unittest.mock import MagicMock


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
