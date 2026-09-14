"""Tests for discord_digest.post_article — article preview delivery."""
import discord_digest as dd
import pytest


@pytest.fixture(autouse=True)
def isolated_previews(monkeypatch, tmp_path):
    import content_preview
    monkeypatch.setattr(dd, '_PREVIEWS_DIR', tmp_path / 'previews')
    monkeypatch.setattr(content_preview, '_PREVIEWS_DIR', tmp_path / 'previews')


def _bundle(tmp_path, body=None, title="How I tuned routing"):
    body = body or (
        f"# {title}\n\nLede paragraph that is more than two sentences long so the preview has some real text to work with. "
        "We need enough material for the lede preview truncation to make sense.\n\n"
        "## First\n\nBody one.\n\n"
        "## Second\n\nBody two.\n\n"
        "## Third\n\nBody three.\n\n"
        "## What I'd try next\n\nTakeaway.\n"
    )
    img = tmp_path / "imgs" / "01-hero-hero.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    return dd.ArticleBundle(
        dir=tmp_path, article_md=body, article_md_path=tmp_path / "article.md",
        image_paths=[img], title=title, lede=body[:200],
        mode="deep_dive", pillar="harness_tuning",
    )


def test_post_article_payload_shape(monkeypatch, tmp_path):
    """Payload contains title, lede, section headers, bundle path, image count."""
    captured = []

    def fake_post(channel_id, content, file_path=None):
        captured.append((channel_id, content, file_path))
        return True

    class FakeResp:
        status_code = 0
        text = ""

    # Channel type lookup must say "text" (0), not forum (15).
    monkeypatch.setattr(dd, "_channel_type", lambda cid: 0)
    monkeypatch.setattr(dd, "_post", fake_post)
    monkeypatch.setattr(dd, "_token", lambda: "X")
    bundle = _bundle(tmp_path)
    out = dd.post_article(bundle, channel_id="12345")
    assert out is not None  # message id (when no forum thread, return None)
    # Inspect the messages.
    flat = "\n---\n".join(c[1] for c in captured)
    assert bundle.title in flat
    assert "##" in flat  # at least one H2 in the section header list
    # The bundle dir path appears in the delivery.
    assert str(tmp_path) in flat or bundle.dir.name in flat


def test_post_article_attaches_images(monkeypatch, tmp_path):
    """The hero image is attached to the preview message."""
    captured = []
    monkeypatch.setattr(dd, "_channel_type", lambda cid: 0)
    monkeypatch.setattr(dd, "_post",
                        lambda cid, c, file_path=None: captured.append((c, file_path)) or True)
    monkeypatch.setattr(dd, "_token", lambda: "X")
    bundle = _bundle(tmp_path)
    dd.post_article(bundle, channel_id="12345")
    # At least one captured call carries the hero image path.
    assert any(fp and fp.endswith(".png") for _, fp in captured)


def test_post_article_no_token_returns_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr(dd, "_token", lambda: "")
    bundle = _bundle(tmp_path)
    assert dd.post_article(bundle) is None


def test_post_article_forum_thread_creation(monkeypatch, tmp_path):
    """Channel type 15 (forum) opens a thread; the rest of the cards go into it."""
    created = {"thread_id": "99999", "calls": []}

    def fake_thread(forum_id, name, content):
        created["thread_id"] = "99999"
        return "99999"

    monkeypatch.setattr(dd, "_channel_type", lambda cid: 15)
    monkeypatch.setattr(dd, "_create_forum_thread", fake_thread)
    monkeypatch.setattr(dd, "_post",
                        lambda cid, c, file_path=None: created["calls"].append((cid, c[:40])) or True)
    monkeypatch.setattr(dd, "_token", lambda: "X")
    bundle = _bundle(tmp_path)
    dd.post_article(bundle, channel_id="11111")
    # All subsequent posts routed into the thread.
    assert all(cid == "99999" for cid, _ in created["calls"])
