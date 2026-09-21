"""Tests for the free Gemini vision client (features 1 & 2) and the topics
screenshot-ingestion adapter. Network is always mocked; image OUTPUT is paid,
but vision INPUT (image->text) is what these features use and is exercised here
through a stubbed _call_vision."""
import json
import os

import config as cfg
import gemini_vision as gv
import topics as tp


# ── key / availability ─────────────────────────────────────────────────

def test_key_prefers_gemini_over_google_ai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-1")
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "goog-2")
    assert gv._key() == "gem-1"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert gv._key() == "goog-2"


def test_available_requires_key_and_flag(monkeypatch):
    monkeypatch.setattr(cfg, "GEMINI_VISION_ENABLED", True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    assert gv.available() is False
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert gv.available() is True
    monkeypatch.setattr(cfg, "GEMINI_VISION_ENABLED", False)
    assert gv.available() is False


# ── JSON extraction / pillar mapping ───────────────────────────────────

def test_strip_json_unfences():
    assert gv._strip_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert gv._strip_json('here you go: {"a": 1} thanks') == '{"a": 1}'


def test_map_pillar_constrains_to_brand_vocab():
    assert gv._map_pillar("build in public", "sahil_twitter") == "build_in_public"
    assert gv._map_pillar("leadership", "sahil_linkedin") == "leadership"
    # unknown -> first allowed pillar (never an invented one)
    assert gv._map_pillar("nonsense", "sahil_twitter") == "build_in_public"


# ── feature 1: describe_screenshot ─────────────────────────────────────

def _stub_call(monkeypatch, reply):
    monkeypatch.setattr(gv, "_call_vision", lambda *a, **k: reply)
    monkeypatch.setattr(gv, "available", lambda: True)
    monkeypatch.setattr(gv, "_b64", lambda p: ("image/png", "Zm9v"))


def test_describe_screenshot_builds_topic(monkeypatch):
    _stub_call(monkeypatch, json.dumps({
        "title": "Colour-coded drill planner shipped",
        "summary": "A React Native schedule view with colour-coded session blocks.",
        "pillar": "build in public", "usable": True,
    }))
    t = gv.describe_screenshot("/tmp/x.png", "sahil_twitter")
    assert t["source"] == "screenshot"
    assert t["educational"] is True
    assert t["pillar"] == "build_in_public"
    assert "schedule view" in t["context"]
    assert t["title"].startswith("Colour-coded")


def test_describe_screenshot_drops_unusable(monkeypatch):
    _stub_call(monkeypatch, json.dumps({"title": "", "summary": "", "usable": False}))
    assert gv.describe_screenshot("/tmp/x.png", "sahil_twitter") is None


def test_describe_screenshot_handles_garbage(monkeypatch):
    _stub_call(monkeypatch, "not json at all")
    assert gv.describe_screenshot("/tmp/x.png", "sahil_twitter") is None


def test_describe_screenshot_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(gv, "available", lambda: False)
    assert gv.describe_screenshot("/tmp/x.png") is None


# ── feature 2: qa_image ────────────────────────────────────────────────

def test_qa_image_neutral_pass_when_unavailable(monkeypatch):
    monkeypatch.setattr(gv, "available", lambda: False)
    v = gv.qa_image("/tmp/x.png", {"title": "t"})
    assert v["passed"] is True and v["available"] is False


def test_qa_image_pass_and_fail(monkeypatch):
    monkeypatch.setattr(cfg, "IMAGERY_QA_MIN_SCORE", 6)
    _stub_call(monkeypatch, json.dumps({"score": 8, "issues": []}))
    assert gv.qa_image("/tmp/x.png", {"title": "t"})["passed"] is True

    _stub_call(monkeypatch, json.dumps(
        {"score": 3, "issues": ["garbled text in header", "palette off"]}))
    bad = gv.qa_image("/tmp/x.png", {"title": "t"})
    assert bad["passed"] is False
    assert bad["score"] == 3
    assert "garbled text in header" in bad["issues"]


def test_qa_image_clamps_and_survives_bad_json(monkeypatch):
    _stub_call(monkeypatch, "garbage")
    # bad JSON -> neutral pass, never a crash
    assert gv.qa_image("/tmp/x.png", {"title": "t"})["passed"] is True


# ── topics adapter: ingestion + file move ──────────────────────────────

def test_screenshot_topics_ingests_and_moves(monkeypatch, tmp_path):
    inbox = tmp_path / "shots"
    inbox.mkdir()
    (inbox / "a.png").write_bytes(b"\x89PNG fake")
    (inbox / "b.jpg").write_bytes(b"\xff\xd8 fake")
    (inbox / "notes.txt").write_text("ignore me")

    monkeypatch.setattr(cfg, "GEMINI_VISION_ENABLED", True)
    monkeypatch.setattr(cfg, "CONTENT_SCREENSHOT_INBOX", str(inbox))
    import gemini_vision
    monkeypatch.setattr(gemini_vision, "available", lambda: True)
    monkeypatch.setattr(
        gemini_vision, "describe_screenshot",
        lambda path, brand: {"pillar": "build_in_public", "topic": "t",
                             "title": "t", "educational": True,
                             "context": "c", "kb_snippets": [], "source": "screenshot"})

    out = tp._screenshot_topics("sahil_twitter", "twitter", max_n=3)
    assert len(out) == 2                       # only the 2 images
    assert all(t["source"] == "screenshot" and t["id"] for t in out)
    # consumed files moved into processed/, inbox cleared of images
    assert not [f for f in os.listdir(inbox) if f.endswith((".png", ".jpg"))]
    assert set(os.listdir(inbox / "processed")) == {"a.png", "b.jpg"}


def test_screenshot_topics_empty_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "GEMINI_VISION_ENABLED", True)
    monkeypatch.setattr(cfg, "CONTENT_SCREENSHOT_INBOX", str(tmp_path))
    import gemini_vision
    monkeypatch.setattr(gemini_vision, "available", lambda: False)
    assert tp._screenshot_topics("sahil_twitter", "twitter", 3) == []


def test_screenshot_topics_zero_budget_noop(monkeypatch):
    # max_n<=0 must never touch the filesystem or the vision client
    called = {"hit": False}
    import gemini_vision
    monkeypatch.setattr(gemini_vision, "available",
                        lambda: called.__setitem__("hit", True) or True)
    assert tp._screenshot_topics("sahil_twitter", "twitter", 0) == []
    assert called["hit"] is False
