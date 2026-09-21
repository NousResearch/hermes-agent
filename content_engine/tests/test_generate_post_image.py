"""Tests the OCR regen loop in draft_media.generate_post_image. Uses
monkeypatched FAL + OCR so no network or sample images are needed."""

import types
import draft_media


def test_regen_on_bad_text_then_succeeds(monkeypatch, tmp_path):
    calls = {"gen": 0}
    img = tmp_path / "x.png"
    img.write_bytes(b"PNG")

    def fake_gen(prompt, brand="", platform="", draft_id="", model=None,
                 negative_prompt="", aspect=None):
        calls["gen"] += 1
        return str(img)

    # first OCR fails, second passes -> proves the regen loop
    seq = iter([(False, ["DEBUG"]), (True, [])])
    monkeypatch.setattr(draft_media, "generate_draft_image", fake_gen)
    monkeypatch.setattr(draft_media, "_verify", lambda path, exp: next(seq))
    monkeypatch.setattr(draft_media, "_can_spend", lambda c: True)

    out = draft_media.generate_post_image({
        "id": "t1", "brand": "coachos", "platform": "twitter",
        "title": "Debug in 5 min", "body_text": "Paste the error. Read the fix.",
    })
    assert out == str(img)
    assert calls["gen"] == 2
