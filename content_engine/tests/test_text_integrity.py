"""Tests for text_integrity fail-closed OCR behaviour."""

import text_integrity as ti


def test_correct_text_passes(monkeypatch):
    monkeypatch.setattr(
        ti,
        "ocr_text",
        lambda _path: "debug a schema mismatch in 5 min buildinpublic",
    )

    ok, missing = ti.verify_text(
        "unused.png", ["DEBUG A SCHEMA MISMATCH", "buildinpublic"]
    )

    assert ok
    assert missing == []


def test_misspelled_text_fails(monkeypatch):
    monkeypatch.setattr(ti, "ocr_text", lambda _path: "xybug x schema wispatch")

    ok, missing = ti.verify_text("unused.png", ["DEBUG A SCHEMA MISMATCH"])

    assert not ok
    assert missing == ["DEBUG A SCHEMA MISMATCH"]


def test_normalisation():
    assert ti._norm("Don't  WATCH.\n") == "dont watch"
    assert ti._norm("Hello   World!!!") == "hello world"
    assert ti._norm("") == ""


def test_textless_image_passes(monkeypatch):
    monkeypatch.setattr(ti, "ocr_text", lambda _path: "")

    assert not ti.has_significant_text("unused.png")


def test_textless_image_with_text_is_rejected(monkeypatch):
    monkeypatch.setattr(ti, "ocr_text", lambda _path: "debug build production deploy")

    assert ti.has_significant_text("unused.png")


def test_unavailable_ocr_fails_closed(monkeypatch):
    monkeypatch.setattr(ti, "ocr_text", lambda _path: None)

    ok, missing = ti.verify_text("unreadable.png", ["required label"])

    assert not ok
    assert missing == ["required label"]
    assert ti.has_significant_text("unreadable.png")
