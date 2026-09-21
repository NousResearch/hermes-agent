"""Regression tests for blog pipeline fixes (issues A/B/C).

A: digest builder emits a compact status index (no article bodies or file:// links)
B: audit escalates at attempts>=3, reports retry status
C: audit skips approved:false slugs in published_exempt.jsonl
"""
import json
import sys
from pathlib import Path

CONTENT_ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTENT_ENGINE))

import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location('ce_blog_audit', Path(__file__).resolve().parents[1] / 'tools/blog_pipeline_audit.py')
assert _spec is not None and _spec.loader is not None
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
audit, _read_exempt, render_report = _audit.audit, _audit._read_exempt, _audit.render_report
from scripts.build_pending_digest import build  # noqa: E402


def test_exempt_slug_skips_false_flag(tmp_path, monkeypatch):
    """FIX C: a manually-published slug must not be flagged."""
    exempt = tmp_path / "published_exempt.jsonl"
    exempt.write_text(json.dumps({"slug": "already-live-post"}) + "\n")
    monkeypatch.setattr(
        _audit, "EXEMPT", exempt
    )
    posts = tmp_path / "SahilBlog" / "src" / "content" / "blog"
    posts.mkdir(parents=True)
    (posts / "already-live-post.md").write_text(
        "---\ntitle: Live\napproved: false\n---\nbody\n"
    )
    monkeypatch.setattr(
        _audit, "POSTS", posts
    )
    monkeypatch.setattr(
        _audit, "BLOG", tmp_path / "SahilBlog"
    )
    # no tracker entry for this slug -> would normally flag, but exempt skips it
    issues = audit()
    assert not any("already-live-post" in i for i in issues), issues


def test_escalate_at_attempts_threshold(tmp_path, monkeypatch):
    """FIX B: attempts>=3 -> escalated (single report), not repeated 'failed'."""
    failed = tmp_path / "failed_images.jsonl"
    failed.write_text(json.dumps({
        "slug": "hard-fail-post", "stream": "ai", "date": "2026-07-06",
        "attempts": 3, "last_error": "boom", "first_failure": "2026-07-04"
    }) + "\n")
    monkeypatch.setattr(_audit, "FAILED_IMAGES", failed)
    monkeypatch.setattr(_audit, "POSTS", tmp_path / "noposts")
    issues = audit()
    esc = [i for i in issues if "escalated" in i]
    tracked = [i for i in issues if "failed image tracked" in i and "hard-fail-post" in i]
    assert len(esc) == 1, issues
    assert not tracked, "should not double-report escalated post as 'tracked'"


def test_retry_status_reported(tmp_path, monkeypatch):
    """FIX B: audit surfaces last retry outcome from status JSON."""
    status = tmp_path / "blog-failed-retry-status.json"
    status.write_text(json.dumps({
        "rc": 0, "finished_at": "2026-07-09T08:15:00",
        "recovered": ["a"], "still_failed": ["b"], "idle": False,
    }))
    monkeypatch.setattr(_audit, "RETRY_STATUS", status)
    monkeypatch.setattr(_audit, "POSTS", tmp_path / "noposts")
    issues = audit()
    assert any("image-retry" in i and "recovered 1" in i and "still_failed 1" in i for i in issues), issues


def test_digest_is_status_index(tmp_path, monkeypatch):
    """FIX A (corrected): digest is a lightweight STATUS/SUMMARY index.

    It must NOT inline article bodies (that blew to 13.6MB and Discord
    rejected it). Each article's openable preview (with images) is attached
    SEPARATELY by the cron. The index stays tiny and has no embedded bodies.
    """
    tracker = tmp_path / "pending_approvals.jsonl"
    tracker.write_text(json.dumps({
        "slug": "my-post", "title": "My Post", "stream": "ai", "tier": "ai",
        "status": "pending", "preview_path": str(tmp_path / "my-post.html"),
    }) + "\n")
    preview = tmp_path / "my-post.html"
    preview.write_text(
        "<html><body><h1>My Post</h1><p>body content here</p>"
        "<img src='data:image/jpeg;base64,AAAA'></body></html>"
    )
    monkeypatch.setattr(_audit, "TRACKER", tracker)
    monkeypatch.setattr("scripts.build_pending_digest.TRACKER", tracker)
    monkeypatch.setattr(
        "scripts.build_pending_digest.PREVIEW_DIR", tmp_path / "previews"
    )
    out = build()
    html = Path(out).read_text()
    assert "file://" not in html, "index must not contain broken file:// links"
    # index lists the article but does NOT inline its body
    assert "My Post" in html, "index should list the article title"
    assert "body content here" not in html, "index must NOT inline article bodies"
    assert "data:image" not in html, "index must not embed images (per-article previews do)"
    assert Path(out).stat().st_size < 8 * 1024 * 1024, "index must stay under Discord's 8MB cap"


def test_audit_report_is_silent_when_healthy_and_capped_when_noisy():
    assert render_report([]) == ""

    report = render_report([f"problem {index}: " + "x" * 200 for index in range(20)])
    assert report.startswith("[Blog Pipeline Audit]")
    assert len(report) <= 1600
    assert "more issue(s)" in report
    assert "problem 19" not in report
