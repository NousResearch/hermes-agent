"""Tests for blog.blog_pipeline — orchestrator + CLI.

run_stream(stream): router.choose -> generator -> gate -> illustrator ->
  assembler -> publisher.stage_draft -> router.record (on success).
run_all(streams): run_stream for each stream.
"""
import os
from pathlib import Path

import blog.blog_pipeline as bpl
import pytest


@pytest.fixture(autouse=True)
def no_real_approval_tracker(monkeypatch):
    monkeypatch.setattr(bpl, "_maybe_request_approval", lambda *args, **kwargs: None)


_DRAFT = {
    "title": "Token-Maxing at the Edge",
    "description": "A counterintuitive claim.",
    "body_md": "# Token-Maxing at the Edge\n\nLede\n\n## The mechanism\n\nText.\n\n## Worked example\n\nCode.\n\n## What I'd try next\n\nTakeaway.",
    "slug": "token-maxing-at-the-edge",
    "tier": "pm",
    "tags": ["ai"],
    "format": "essay",
    "source": "research-paper",
    "stream": "ai",
    "signals": [{"signal_id": "t1", "summary": "token-maxing"}],
    "context": "",
    "kb_snippets": [],
}


def _setup_tmp_repo(tmp_path):
    """Create a tmp SahilBlog repo skeleton with git init."""
    import subprocess
    (tmp_path / "src/content/blog").mkdir(parents=True)
    (tmp_path / "public/blog").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
    (tmp_path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


def test_run_stream_happy_path(monkeypatch, tmp_path):
    """Full happy path: one staged draft per stream, topic recorded on success."""
    repo = _setup_tmp_repo(tmp_path)
    calls = []

    # Mock router.choose.
    plan = {"topic_id": "t1", "title_hint": "token-maxing", "tags": ["ai-adoption"],
            "source": "research-paper",
            "signals": [{"signal_id": "t1", "summary": "token-maxing"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)

    # Mock generator.
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: _DRAFT)

    # Mock illustrator.
    monkeypatch.setattr(bpl, "illustrate", lambda d, out_dir=None, max_sections=None: {"hero_path": "/tmp/fake_hero.png", "section_paths": {}})

    # Mock assembler.
    def fake_assemble(d, imgs, repo=None, pub_date=None):
        p = Path(repo) / "src/content/blog" / f"{d['slug']}.mdx"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntitle: \"t\"\n---\nbody")
        calls.append(("assemble", d["slug"]))
        return p
    monkeypatch.setattr(bpl, "assemble", fake_assemble)

    # Mock publisher.
    monkeypatch.setattr(bpl, "stage_draft", lambda mdx, repo=None: calls.append(("stage", mdx)) or "ai")

    # Mock router.record.
    monkeypatch.setattr(bpl, "record", lambda stream, tid, title: calls.append(("record", stream, tid)))

    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "ok"
    # Now record is called pre-emptively (before assemble) and also on success.
    op_names = [c[0] for c in calls]
    assert "record" in op_names  # called pre-emptively after choose()
    assert "assemble" in op_names
    assert "stage" in op_names
    assert op_names.index("assemble") < op_names.index("stage")


def test_run_stream_skips_when_router_returns_none(monkeypatch, tmp_path):
    """When the router returns None, run_stream skips."""
    repo = _setup_tmp_repo(tmp_path)
    monkeypatch.setattr(bpl, "choose", lambda stream: None)
    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "skipped_router"


def test_run_stream_skips_when_excluded_by_policy(monkeypatch, tmp_path, blog_exclusions):
    """Excluded topics are rejected before generation begins."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {
        "topic_id": "manual-1234",
        "title_hint": "Cheap-first model routing",
        "tags": [],
        "source": "manual",
        "signals": [{"signal_id": "t1", "summary": "s"}],
    }
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    generator_called = {"n": 0}

    def fake_write(p, stream):
        generator_called["n"] += 1
        return _DRAFT

    monkeypatch.setattr(bpl, "write_with_gate", fake_write)
    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "skipped_excluded"
    assert generator_called["n"] == 0


def test_run_stream_skips_when_generator_returns_none(monkeypatch, tmp_path):
    """When the generator returns None (LLM dead), run_stream skips."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: None)
    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "skipped_generator"


def test_run_stream_does_not_record_on_generator_failure(monkeypatch, tmp_path):
    """router.record is NOT called when the generator fails."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: None)
    record_called = []
    monkeypatch.setattr(bpl, "record", lambda s, tid, t: record_called.append(tid))
    reserve_called = []
    release_called = []
    monkeypatch.setattr(bpl, "reserve", lambda s, tid, title: reserve_called.append(tid) or "tok")
    monkeypatch.setattr(bpl, "release", lambda token: release_called.append(token))

    bpl.run_stream("ai", repo=str(repo))
    # Failed generations release the temporary reservation and do not burn the
    # topic into permanent usage.
    assert reserve_called == ["t1"]
    assert release_called == ["tok"]
    assert record_called == []


def test_run_all_runs_all_streams(monkeypatch, tmp_path):
    """run_all() runs each configured stream and returns per-stream results."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: _DRAFT)
    monkeypatch.setattr(bpl, "illustrate", lambda d, out_dir=None, max_sections=None: {"hero_path": "/tmp/fake_hero.png", "section_paths": {}})
    def fake_assemble(d, imgs, repo=None, pub_date=None):
        p = Path(repo) / "src/content/blog" / f"{d['slug']}.mdx"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntitle: \"t\"\n---\nbody")
        return p
    monkeypatch.setattr(bpl, "assemble", fake_assemble)
    monkeypatch.setattr(bpl, "stage_draft", lambda mdx, repo=None: "slug")
    monkeypatch.setattr(bpl, "record", lambda s, tid, t: None)
    result = bpl.run_all(streams=("ai", "pm", "builder"), repo=str(repo))
    assert result["status"] == "ok"
    assert "ai" in result["results"]
    assert "pm" in result["results"]
    assert "builder" in result["results"]


def test_run_stream_skipped_when_disabled(monkeypatch, tmp_path):
    """When BLOG_ENABLED is False, run_stream skips."""
    repo = _setup_tmp_repo(tmp_path)
    monkeypatch.setattr(bpl, "BLOG_ENABLED", False)
    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "skipped_disabled"


def test_run_stream_graceful_when_reviewer_degraded(monkeypatch, tmp_path):
    """Daily pipeline (strict_review=False default) still produces a draft
    when the editorial reviewer degrades. The reviewer LLM returning None
    must not block the daily path."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)

    # write_with_gate in non-strict mode returns a draft even when the
    # reviewer degrades. Mock it to simulate that path.
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: _DRAFT)
    monkeypatch.setattr(bpl, "illustrate",
                        lambda d, out_dir=None, max_sections=None:
                        {"hero_path": "/tmp/fake_hero.png", "section_paths": {}})
    def fake_assemble(d, imgs, repo=None, pub_date=None):
        p = Path(repo) / "src/content/blog" / f"{d['slug']}.mdx"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntitle: \"t\"\n---\nbody")
        return p
    monkeypatch.setattr(bpl, "assemble", fake_assemble)
    monkeypatch.setattr(bpl, "stage_draft", lambda mdx, repo=None: "slug")
    monkeypatch.setattr(bpl, "record", lambda s, tid, t: None)

    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "ok", \
        "daily pipeline must stage draft even when reviewer degrades"

# -- Failed-image handling tests -----------------------------------------------

def test_run_stream_failed_images_status(monkeypatch, tmp_path):
    """When all images fail (hero AND sections None), status is failed_images."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: _DRAFT)
    # All images fail.
    monkeypatch.setattr(bpl, "illustrate",
                        lambda d, out_dir=None, max_sections=None:
                        {"hero_path": None, "section_paths": {}})
    # Redirect tracking file to tmp.
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")

    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "failed_images"
    assert result["slug"] == _DRAFT["slug"]


def test_run_stream_partial_images_proceeds(monkeypatch, tmp_path):
    """When hero succeeds but sections fail, pipeline proceeds (partial imagery)."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: _DRAFT)
    # Hero succeeds, sections all fail.
    monkeypatch.setattr(bpl, "illustrate",
                        lambda d, out_dir=None, max_sections=None:
                        {"hero_path": "/tmp/fake_hero.png", "section_paths": {}})
    def fake_assemble(d, imgs, repo=None, pub_date=None):
        p = Path(repo) / "src/content/blog" / f"{d['slug']}.mdx"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntitle: \"t\"\n---\nbody")
        return p
    monkeypatch.setattr(bpl, "assemble", fake_assemble)
    monkeypatch.setattr(bpl, "stage_draft", lambda mdx, repo=None: "slug")
    monkeypatch.setattr(bpl, "record", lambda s, tid, t: None)
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")

    result = bpl.run_stream("ai", repo=str(repo))
    assert result["status"] == "ok"


def test_failed_image_tracking_creates_jsonl(monkeypatch, tmp_path):
    """_track_failed_image creates the JSONL file and writes a valid entry."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    bpl._track_failed_image("test-slug", "ai", "timeout")
    assert (tmp_path / "failed_images.jsonl").exists()
    import json
    entries = [json.loads(l) for l in (tmp_path / "failed_images.jsonl").read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    assert entries[0]["slug"] == "test-slug"
    assert entries[0]["stream"] == "ai"
    assert entries[0]["last_error"] == "timeout"
    assert entries[0]["attempts"] == 1


def test_failed_image_tracking_increments_attempts(monkeypatch, tmp_path):
    """_track_failed_image increments attempts for existing slugs."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    bpl._track_failed_image("test-slug", "ai", "error1")
    bpl._track_failed_image("test-slug", "ai", "error2")
    import json
    entries = [json.loads(l) for l in (tmp_path / "failed_images.jsonl").read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    assert entries[0]["attempts"] == 2
    assert entries[0]["last_error"] == "error2"


def test_remove_from_failed(monkeypatch, tmp_path):
    """_remove_from_failed removes a slug from the tracking file."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    bpl._track_failed_image("slug-a", "ai", "err")
    bpl._track_failed_image("slug-b", "pm", "err")
    bpl._remove_from_failed("slug-a")
    import json
    entries = [json.loads(l) for l in (tmp_path / "failed_images.jsonl").read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    assert entries[0]["slug"] == "slug-b"


def test_stale_failed_images_flags_old_entries(monkeypatch, tmp_path):
    """Entries older than the stale threshold are flagged."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    import json
    from datetime import date, timedelta
    # Write an old entry (10 days ago).
    old_date = (date.today() - timedelta(days=10)).isoformat()
    entry = {"slug": "old-slug", "stream": "ai", "date": old_date,
             "attempts": 3, "last_error": "err", "first_failure": old_date}
    (tmp_path / "failed_images.jsonl").write_text(json.dumps(entry) + "\n")
    # Write a recent entry (1 day ago).
    recent_date = (date.today() - timedelta(days=1)).isoformat()
    entry2 = {"slug": "recent-slug", "stream": "pm", "date": recent_date,
              "attempts": 1, "last_error": "err", "first_failure": recent_date}
    with open(tmp_path / "failed_images.jsonl", "a") as f:
        f.write(json.dumps(entry2) + "\n")

    stale = bpl.get_stale_failed_images()
    assert len(stale) == 1
    assert stale[0]["slug"] == "old-slug"



# ── Holistic retry (failed-image recovery + Codex cap defer) ────────────

def _seed_failed(tmp_path, monkeypatch, slug="token-maxing-at-the-edge", stream="ai"):
    """Point trackers at tmp and seed one held post with a persisted draft."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    monkeypatch.setattr(bpl, "PENDING_DRAFTS_DIR", tmp_path / "pending_images")
    bpl._track_failed_image(slug, stream, "seed")
    bpl._save_pending_draft({**_DRAFT, "slug": slug}, stream)
    return slug


def test_retry_all_defers_when_codex_capped(monkeypatch, tmp_path):
    """A usage cap must defer the whole run and burn NO attempts."""
    from blog.blog_illustrator import CodexCapExceeded
    slug = _seed_failed(tmp_path, monkeypatch)

    def capped(draft, **kw):
        raise CodexCapExceeded("capped")
    monkeypatch.setattr(bpl, "illustrate", capped)

    res = bpl.retry_all_pending_images(repo=str(tmp_path))
    assert res["status"] == "deferred"
    assert res["reason"] == "codex_capped"
    assert slug in res["pending"]
    # Attempt count unchanged (still the single seed attempt), draft preserved.
    entries = bpl._get_failed_entries()
    assert entries and entries[0]["attempts"] == 1
    assert (tmp_path / "pending_images" / f"{slug}.json").exists()


def test_retry_recovers_held_post_with_persisted_draft(monkeypatch, tmp_path):
    """On success the post is assembled, staged, and cleared from both trackers."""
    slug = _seed_failed(tmp_path, monkeypatch)
    seen = {}

    def ok_illustrate(draft, **kw):
        seen["body"] = draft.get("body_md", "")
        return {"hero_path": "/tmp/h.png", "section_paths": {"The mechanism": "/tmp/s.png"}}
    monkeypatch.setattr(bpl, "illustrate", ok_illustrate)
    monkeypatch.setattr(bpl, "assemble", lambda d, imgs, repo=None, pub_date=None: tmp_path / f"{d['slug']}.mdx")
    monkeypatch.setattr(bpl, "stage_draft", lambda mdx, repo=None: slug)

    res = bpl.retry_all_pending_images(repo=str(tmp_path))
    assert res["status"] == "ok"
    assert slug in res["recovered"]
    # Used the FULL persisted body, not a slug-only draft.
    assert "The mechanism" in seen["body"]
    # Cleared from both trackers.
    assert bpl._get_failed_entries() == []
    assert not (tmp_path / "pending_images" / f"{slug}.json").exists()


def test_retry_reports_legacy_entry_without_draft(monkeypatch, tmp_path):
    """Held entries with no persisted draft are surfaced, not silently retried."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    monkeypatch.setattr(bpl, "PENDING_DRAFTS_DIR", tmp_path / "pending_images")
    bpl._track_failed_image("legacy-slug", "ai", "seed")  # no _save_pending_draft

    monkeypatch.setattr(bpl, "illustrate", lambda *a, **k: pytest.fail("should not illustrate"))
    res = bpl.retry_all_pending_images(repo=str(tmp_path))
    assert res["status"] == "ok"
    assert "legacy-slug" in res["no_draft"]


def test_retry_prunes_stale_legacy_no_draft(monkeypatch, tmp_path):
    """A no-draft entry older than the stale threshold is pruned, not reported."""
    monkeypatch.setattr(bpl, "FAILED_IMAGES_PATH", tmp_path / "failed_images.jsonl")
    monkeypatch.setattr(bpl, "PENDING_DRAFTS_DIR", tmp_path / "pending_images")
    bpl._track_failed_image("ancient-slug", "ai", "seed")
    # Backdate first_failure well past the stale threshold.
    entries = bpl._get_failed_entries()
    entries[0]["first_failure"] = "2020-01-01"
    (tmp_path / "failed_images.jsonl").write_text(
        "\n".join(__import__("json").dumps(e) for e in entries) + "\n")

    monkeypatch.setattr(bpl, "illustrate", lambda *a, **k: pytest.fail("no illustrate for no_draft"))
    res = bpl.retry_all_pending_images(repo=str(tmp_path))
    assert "ancient-slug" in res["pruned"]
    assert "ancient-slug" not in res["no_draft"]
    assert bpl._get_failed_entries() == []


# ── Safe bounded scheduler seam ─────────────────────────────────────────────

def test_stage_draft_only_assembles_with_three_image_bound_and_no_delivery(monkeypatch, tmp_path):
    """The scheduler seam writes one worktree draft but never commits or delivers it."""
    repo = _setup_tmp_repo(tmp_path)
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    seen = {}
    monkeypatch.setattr(bpl, "choose", lambda stream: plan)
    monkeypatch.setattr(bpl, "write_with_gate", lambda p, stream, **kw: _DRAFT)
    monkeypatch.setattr(
        bpl,
        "illustrate",
        lambda draft, **kwargs: seen.update(kwargs) or {"hero_path": "/tmp/hero.png", "section_paths": {}},
    )
    monkeypatch.setattr(bpl, "assemble", lambda draft, images, repo=None, pub_date=None: Path(repo) / "draft.mdx")
    monkeypatch.setattr(bpl, "stage_draft", lambda *args, **kwargs: pytest.fail("must not commit"))
    monkeypatch.setattr(bpl, "_maybe_request_approval", lambda *args: pytest.fail("must not request approval"))

    result = bpl.run_stage_draft_only("ai", repo=str(repo), max_new_drafts=1, max_images=3)

    assert result["status"] == "staged"
    assert result["stage_only"] is True
    assert result["drafts_created"] == 1
    assert result["images_requested"] == 3
    assert seen["max_sections"] == 2


def test_stage_draft_only_dry_run_refuses_pipeline_provider_execution(monkeypatch, tmp_path):
    repo = _setup_tmp_repo(tmp_path)
    monkeypatch.setattr(bpl, "choose", lambda stream: pytest.fail("dry-run must not call providers"))

    result = bpl.run_stage_draft_only("ai", repo=str(repo), max_new_drafts=1, max_images=3, dry_run=True)

    assert result == {
        "status": "dry_run",
        "stage_only": True,
        "provider_execution": "refused",
        "stream": "ai",
        "repo": str(repo),
        "max_new_drafts": 1,
        "max_images": 3,
    }
