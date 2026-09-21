"""Tests for Discord-based approval workflow and manual article intake.

Process 1 (Draft Approvals): blog_approval.py
Process 2 (Manual Articles): blog_discord_intake.py
"""

import json
import tempfile
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolated_approval_tracker(tmp_path, monkeypatch):
    monkeypatch.setattr("blog.blog_approval.TRACKER_PATH", tmp_path / "pending_approvals.jsonl")
    monkeypatch.setattr("blog.blog_approval.PREVIEWS_DIR", tmp_path / "previews")


def _make_mdx(tmp_path: Path, slug: str, title: str = "Test Article") -> str:
    mdx = tmp_path / f"{slug}.mdx"
    mdx.write_text(
        f"---\ntitle: \"{title}\"\ndescription: \"Preview test\"\npubDate: 2026-06-30\ntags: [\"test\"]\ntier: pm\nformat: essay\napproved: false\nsource: manual\n---\n\n# {title}\n\nBody.",
        encoding="utf-8",
    )
    return str(mdx)


# ── Process 1: Approval Tracker ─────────────────────────────────

class TestApprovalTracker:
    """blog_approval — state management + Discord commands."""

    def test_request_creates_pending_entry(self, tmp_path):
        from blog.blog_approval import request, pending, _read_tracker, TRACKER_PATH
        # Use a unique slug per test.
        slug = f"test-approval-{id(self)}"
        request(slug, "Test Article", "ai", "pm", _make_mdx(tmp_path, slug))
        items = pending()
        slugs = [i["slug"] for i in items]
        assert slug in slugs, f"Expected {slug} in pending: {slugs}"

    def test_approve_changes_status(self, tmp_path):
        from blog.blog_approval import request, approve, pending
        slug = f"test-approve-{id(self)}"
        request(slug, "Approve Test", "ai", mdx_path=_make_mdx(tmp_path, slug, "Approve Test"))
        assert approve(slug) is True
        items = pending()
        assert slug not in [i["slug"] for i in items]

    def test_reject_changes_status(self, tmp_path):
        from blog.blog_approval import request, reject, pending
        slug = f"test-reject-{id(self)}"
        request(slug, "Reject Test", "ai", mdx_path=_make_mdx(tmp_path, slug, "Reject Test"))
        reject(slug, "Not relevant")
        items = pending()
        assert slug not in [i["slug"] for i in items]

    def test_approve_nonexistent_returns_false(self):
        from blog.blog_approval import approve
        assert approve("nonexistent-slug") is False

    def test_amend_changes_status(self, tmp_path):
        from blog.blog_approval import request, amend, _read_tracker
        slug = f"test-amend-{id(self)}"
        request(slug, "Amend Test", "ai", mdx_path=_make_mdx(tmp_path, slug, "Amend Test"))
        amend(slug, "Add sources")
        entries = _read_tracker()
        e = next((x for x in entries if x["slug"] == slug), None)
        assert e is not None
        assert e["status"] == "amend"
        assert "sources" in e.get("notes", "")

    def test_publish_removes_from_tracker(self, monkeypatch, tmp_path):
        from blog.blog_approval import request, publish, _read_tracker
        slug = f"test-publish-{id(self)}"
        request(slug, "Publish Test", "ai", mdx_path=_make_mdx(tmp_path, slug, "Publish Test"))

        def mock_approve(slug):
            return {"status": "ok"}
        monkeypatch.setattr("blog.blog_publisher.approve", mock_approve)

        result = publish(slug)
        assert result["status"] == "ok"
        entries = _read_tracker()
        assert slug not in [e["slug"] for e in entries]

    def test_summary_includes_key_fields(self, tmp_path):
        from blog.blog_approval import summary
        preview = tmp_path / "my-post.html"
        preview.write_text("<html></html>")
        entry = {
            "title": "My Post", "stream": "ai", "tier": "pm",
            "slug": "my-post", "approval_id": "2026-06-29-my-post",
            "mdx_path": str(tmp_path / "my-post.mdx"),
            "preview_path": str(preview),
        }
        s = summary(entry)
        assert "My Post" in s
        assert "my-post" in s
        assert "!approve" in s
        assert "!reject" in s
        assert f"MEDIA:{preview}" in s

    def test_request_preview_embeds_article_image(self, tmp_path):
        from blog.blog_approval import request, _read_tracker
        from PIL import Image
        import io
        slug = "image-preview-post"
        hero = tmp_path / "hero.png"
        # Create a 100x100 red PNG via Pillow
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        hero.write_bytes(buf.getvalue())
        mdx = tmp_path / f"{slug}.mdx"
        mdx.write_text(
            f"---\ntitle: \"Image Preview Post\"\ndescription: \"Check image\"\npubDate: 2026-06-30\nheroImage: \"{hero}\"\ntags: [\"test\"]\ntier: pm\nformat: essay\napproved: false\nsource: manual\n---\n\n# Image Preview Post\n\nBody.",
            encoding="utf-8",
        )
        request(slug, "Image Preview Post", "pm", "pm", str(mdx))
        entry = _read_tracker()[0]
        preview = Path(entry["preview_path"])
        assert preview.exists()
        html = preview.read_text(encoding="utf-8")
        assert "data:image/jpeg;base64," in html  # compressed images embedded
        assert "compressed" in html  # compression meta label

    def test_request_blocks_duplicate_topic_cluster_without_series(self, tmp_path):
        from blog.blog_approval import request
        first = _make_mdx(
            tmp_path,
            "token-one",
            "The Token-Efficiency Frontier: A Framework for PMs",
        )
        second = _make_mdx(
            tmp_path,
            "token-two",
            "The Token-Efficiency Frontier: A Builder's Map",
        )
        request("token-one", "The Token-Efficiency Frontier: A Framework for PMs", "pm", "pm", first)
        with pytest.raises(ValueError, match="duplicate pending topic cluster"):
            request("token-two", "The Token-Efficiency Frontier: A Builder's Map", "builder", "builder", second)

    def test_batch_validation_allows_explicit_series(self):
        from blog.blog_approval import batch_validation_issues
        entries = [
            {"slug": "one", "title": "The Token-Efficiency Frontier: Part One", "status": "pending", "series": "token-efficiency"},
            {"slug": "two", "title": "The Token-Efficiency Frontier: Part Two", "status": "pending", "series": "token-efficiency"},
        ]
        assert batch_validation_issues(entries) == []

    def test_batch_validation_flags_unplanned_duplicate_clusters(self):
        from blog.blog_approval import batch_validation_issues
        entries = [
            {"slug": "one", "title": "The Token-Efficiency Frontier: Part One", "status": "pending"},
            {"slug": "two", "title": "The Token-Efficiency Frontier: Part Two", "status": "pending"},
        ]
        issues = batch_validation_issues(entries)
        assert len(issues) == 1
        assert "token-efficiency-frontier" in issues[0]

    def test_parse_discord_command_approve(self):
        from blog.blog_approval import parse_discord_command
        r = parse_discord_command("!approve my-post")
        assert r == {"command": "approve", "slug": "my-post", "args": ""}

    def test_parse_discord_command_reject_with_reason(self):
        from blog.blog_approval import parse_discord_command
        r = parse_discord_command("!reject my-post Needs more data")
        assert r["command"] == "reject"
        assert r["slug"] == "my-post"
        assert r["args"] == "Needs more data"

    def test_parse_discord_command_amend(self):
        from blog.blog_approval import parse_discord_command
        r = parse_discord_command("!amend my-post Add examples")
        assert r["command"] == "amend"


    def test_handle_discord_command_outcomes(self, monkeypatch, tmp_path):
        from blog.blog_approval import request, handle_discord_command
        slug = "test-cmd-outcome"
        request(slug, "Cmd Test", "ai", mdx_path=_make_mdx(tmp_path, slug, "Cmd Test"))

        def mock_approve(slug):
            return {"status": "ok"}
        monkeypatch.setattr("blog.blog_publisher.approve", mock_approve)

        r = handle_discord_command(f"!approve {slug}")
        assert r["handled"] is True
        assert r["action"] == "approved"

    def test_handle_discord_command_reject(self, tmp_path):
        from blog.blog_approval import request, handle_discord_command
        slug = "test-cmd-rej"
        request(slug, "Cmd Reject", "ai", mdx_path=_make_mdx(tmp_path, slug, "Cmd Reject"))

        r = handle_discord_command(f"!reject {slug} Bad topic")
        assert r["handled"] is True
        assert r["action"] == "rejected"


# ── Process 2: Manual Article Intake ────────────────────────────

class TestManualIntake:
    """blog_discord_intake — form parsing + validation + processing."""

    def test_parse_full_form(self):
        from blog.blog_discord_intake import parse
        text = """!article
link: https://example.com/post
topic: AI Agent Memory
images: /tmp/img1.png, /tmp/img2.png
pipeline: ai"""
        r = parse(text)
        assert r["link"] == "https://example.com/post"
        assert r["topic"] == "AI Agent Memory"
        assert r["images"] == ["/tmp/img1.png", "/tmp/img2.png"]
        assert r["pipeline"] == "ai"

    def test_parse_minimal(self):
        from blog.blog_discord_intake import parse
        text = "!article\ntext: Quick thoughts on agent design"
        r = parse(text)
        assert r["article_text"] == "Quick thoughts on agent design"
        assert r["pipeline"] == "ai"  # default

    def test_parse_pipeline_mapping(self):
        from blog.blog_discord_intake import parse
        assert parse("!article\ntopic: X\npipeline: builders")["pipeline"] == "builder"
        assert parse("!article\ntopic: X\npipeline: pm")["pipeline"] == "pm"

    def test_validate_valid(self):
        from blog.blog_discord_intake import validate
        assert validate({"topic": "Test", "pipeline": "ai"}) == []

    def test_validate_missing_all(self):
        from blog.blog_discord_intake import validate
        issues = validate({"link": "", "topic": "", "article_text": "", "pipeline": "ai"})
        assert len(issues) >= 1
        assert any("link" in i.lower() or "topic" in i.lower() or "text" in i.lower() for i in issues)

    def test_process_text_stages_article(self, monkeypatch):
        from blog.blog_discord_intake import process
        intake = {"link": "", "topic": "My Article", "article_text": "Hello world",
                  "images": [], "pipeline": "pm"}

        def mock_stage_adhoc(mdx_path, stream, repo):
            return {"status": "staged", "slug": "my-article"}
        monkeypatch.setattr("blog.blog_publisher.stage_adhoc", mock_stage_adhoc)

        result = process(intake, repo="/tmp")
        assert result["status"] == "staged"
        assert result["slug"] == "my-article"

    def test_process_topic_queues_and_runs(self, monkeypatch):
        from blog.blog_discord_intake import process
        intake = {"link": "", "topic": "New Concept", "article_text": "",
                  "images": [], "pipeline": "ai"}

        def mock_run_stream(stream, repo):
            return {"status": "ok", "slug": "new-concept"}
        monkeypatch.setattr("blog.blog_pipeline.run_stream", mock_run_stream)

        result = process(intake, repo="/tmp")
        assert result["status"] == "staged"

    def test_result_discord_staged(self):
        from blog.blog_discord_intake import result_discord
        r = result_discord({"status": "staged", "slug": "my-post", "title": "My Post"})
        assert "My Post" in r
        assert "!approve" in r

    def test_result_discord_invalid(self):
        from blog.blog_discord_intake import result_discord
        r = result_discord({"status": "invalid", "issues": ["Missing topic"]})
        assert "Missing topic" in r
        assert "❌" in r

    def test_result_discord_generation_failed(self):
        from blog.blog_discord_intake import result_discord
        r = result_discord({"status": "generation_failed", "message": "No content"})
        assert "No content" in r
