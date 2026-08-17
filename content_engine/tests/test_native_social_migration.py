"""Tests the native Codex migration of the personal-brand image path.

Verifies that X/LinkedIn drafts route through the explicit native Codex seam
(image_jobs + image_job_service) rather than a legacy provider path,
and that the path is fail-closed (no FAL fallback on seam error).
"""

import draft_media


def _personal_draft():
    return {
        "id": "t1", "brand": "sahil_twitter", "platform": "twitter",
        "title": "Debug in 5 min", "body_text": "Paste the error. Read the fix.",
    }


def test_personal_brand_routes_to_native_codex(monkeypatch, tmp_path):
    """X/LinkedIn drafts call the native seam, never the FAL chain."""
    calls = {"native": 0, "fal": 0}
    img = tmp_path / "x.png"
    img.write_bytes(b"PNG")

    def fake_native(draft, output_dir=None):
        calls["native"] += 1
        return str(img)

    def fake_fal(*a, **k):
        calls["fal"] += 1
        return str(img)

    monkeypatch.setattr(draft_media, "_generate_native_codex", fake_native)
    monkeypatch.setattr(draft_media, "generate_draft_image", fake_fal)

    out = draft_media.generate_post_image(_personal_draft())
    assert out == str(img)
    assert calls["native"] == 1
    assert calls["fal"] == 0


def test_personal_brand_fails_closed_without_fal_fallback(monkeypatch, tmp_path):
    """A native seam error returns None and never degrades to FAL."""
    calls = {"fal": 0}

    def boom(draft, output_dir=None):
        raise RuntimeError("native codex unavailable")

    def fake_fal(*a, **k):
        calls["fal"] += 1
        return str(tmp_path / "x.png")

    monkeypatch.setattr(draft_media, "_generate_native_codex", boom)
    monkeypatch.setattr(draft_media, "generate_draft_image", fake_fal)

    out = draft_media.generate_post_image(_personal_draft())
    assert out is None
    assert calls["fal"] == 0


def test_non_personal_brand_keeps_legacy_chain(monkeypatch, tmp_path):
    """Product brands outside the native Codex set keep the legacy path."""
    calls = {"native": 0, "fal": 0}
    img = tmp_path / "x.png"
    img.write_bytes(b"PNG")

    def fake_native(draft, output_dir=None):
        calls["native"] += 1
        return str(img)

    def fake_fal(prompt, brand="", platform="", draft_id="", model=None,
                 negative_prompt="", aspect=None):
        calls["fal"] += 1
        return str(img)

    monkeypatch.setattr(draft_media, "_generate_native_codex", fake_native)
    monkeypatch.setattr(draft_media, "generate_draft_image", fake_fal)
    monkeypatch.setattr(draft_media, "_verify", lambda path, exp: (True, []))
    monkeypatch.setattr(draft_media, "_can_spend", lambda c: True)

    out = draft_media.generate_post_image({
        "id": "t2", "brand": "coachos", "platform": "twitter",
        "title": "Debug in 5 min", "body_text": "Paste the error. Read the fix.",
    })
    assert out == str(img)
    assert calls["native"] == 0
    assert calls["fal"] == 1




def test_personal_brand_cannot_bypass_native_codex_with_legacy_arguments(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"PNG")
    calls = {"native": 0, "fal": 0}

    def fake_native(draft, output_dir=None):
        calls["native"] += 1
        return str(img)

    def fake_fal(*args, **kwargs):
        calls["fal"] += 1
        return str(img)

    monkeypatch.setattr(draft_media, "_generate_native_codex", fake_native)
    monkeypatch.setattr(draft_media, "generate_draft_image", fake_fal)

    result = draft_media.generate_post_image(
        _personal_draft(), model="seedream45", scene_prompt="legacy bypass"
    )

    assert result == str(img)
    assert calls == {"native": 1, "fal": 0}
    assert draft_media._native_aspect("landscape") == "landscape"
    assert draft_media._native_aspect("portrait_4_5") == "portrait"
    assert draft_media._native_aspect("portrait_9_16") == "portrait"
    assert draft_media._native_aspect("square") == "square"
    assert draft_media._native_aspect("") == "square"
