"""Tests for article_delivery — paste-ready body + image attachments on approve."""
from pathlib import Path
import article_delivery as adl


def _bundle(tmp_path):
    body = (
        "# How I tuned routing\n\nLede.\n\n"
        "## First\n\nBody one.\n\n"
        "## What I'd try next\n\nTakeaway.\n"
    )
    img = tmp_path / "imgs" / "01-hero-hero.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    article_path = tmp_path / "article.md"
    article_path.write_text(body, encoding="utf-8")
    return adl.ArticleBundle(
        dir=tmp_path, article_md=body, article_md_path=article_path,
        image_paths=[img], title="How I tuned routing", lede="Lede.",
        mode="deep_dive", pillar="harness_tuning",
    )


def test_ready_posts_paste_ready_body_and_attachments(monkeypatch, tmp_path):
    """On approve, post the body in chunks (Discord 1900 limit) and attach all images."""
    sent: list = []
    monkeypatch.setattr(adl, "_post",
                        lambda cid, c, file_path=None: sent.append((c, file_path)) or True)
    monkeypatch.setattr(adl, "_channel_type", lambda cid: 0)
    monkeypatch.setattr(adl, "_token", lambda: "X")
    bundle = _bundle(tmp_path)
    out = adl.ready(bundle, channel_id="12345")
    assert out is not None
    # Body text shipped.
    full_text = "\n".join(c for c, _ in sent)
    assert "How I tuned routing" in full_text
    # Hero image attached.
    assert any(fp and fp.endswith(".png") for _, fp in sent)


def test_ready_called_with_bundle_returns_message_id(monkeypatch, tmp_path):
    monkeypatch.setattr(adl, "_post",
                        lambda cid, c, file_path=None: True)
    monkeypatch.setattr(adl, "_channel_type", lambda cid: 0)
    monkeypatch.setattr(adl, "_token", lambda: "X")
    bundle = _bundle(tmp_path)
    assert adl.ready(bundle, channel_id="12345") is not None


def test_reject_discards_bundle(tmp_path):
    bundle = _bundle(tmp_path)
    # Create some marker files in the bundle dir to verify they get removed.
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    assert (tmp_path / "marker.txt").exists()
    adl.discard(bundle)
    assert not tmp_path.exists()


def test_ready_no_token_returns_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr(adl, "_token", lambda: "")
    bundle = _bundle(tmp_path)
    assert adl.ready(bundle) is None
