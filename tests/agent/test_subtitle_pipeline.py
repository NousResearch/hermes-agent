"""The live-subtitle pipeline's decisions, with OCR and the model faked.

One frame in, one decision out: unchanged / empty / translate-and-box. The
OCR engine is faked at the engine boundary (RapidOCR is an optional dep that
CI does not install) and translation at the auxiliary-client boundary, so
these exercise every real line between the endpoint and those two calls.
"""

import types
from unittest.mock import MagicMock

import pytest

import agent.subtitle_pipeline as sp


class FakeOcrResult:
    """Shape-compatible with RapidOCR's output (numpy-backed in real life)."""

    def __init__(self, rows):
        # rows: list of (text, score, quad) — quad is 4 [x, y] points.
        self.txts = tuple(row[0] for row in rows)
        self.scores = tuple(row[1] for row in rows)
        self.boxes = [row[2] for row in rows]


def quad(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Module-level caches must not leak between tests."""
    monkeypatch.setattr(sp, "_translation_cache", type(sp._translation_cache)())
    monkeypatch.setattr(sp, "_stream_contexts", {})
    monkeypatch.setattr(sp, "_settings", lambda: {"max_chars_per_line": 42, "min_ocr_confidence": 0.5})


@pytest.fixture
def ocr(monkeypatch):
    """Install a fake OCR engine; tests assign .result per scenario."""
    holder = types.SimpleNamespace(result=FakeOcrResult([]))
    monkeypatch.setattr(sp, "_get_ocr_engine", lambda: (lambda _img: holder.result))
    monkeypatch.setattr(sp, "_decode_image", lambda _data: "fake-image")
    return holder


@pytest.fixture
def translator(monkeypatch):
    """Fake auxiliary call_llm; records prompts, answers PT[<line>]."""
    calls = []

    def fake_call_llm(task=None, *, messages, **kwargs):
        assert task == "subtitles"
        calls.append(messages)
        source = messages[-1]["content"].rsplit("Translate: ", 1)[-1]
        response = MagicMock()
        response.choices[0].message.content = f"PT[{source}]"
        return response

    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "call_llm", fake_call_llm)
    return calls


class TestReadingTheBand:
    def test_new_line_translates_and_returns_the_union_box(self, ocr, translator):
        ocr.result = FakeOcrResult(
            [
                ("We are not so", 0.93, quad(200, 20, 400, 40)),
                ("different, you and I.", 0.91, quad(180, 70, 460, 40)),
            ]
        )

        out = sp.process_frame(b"png", "pt", prev_text="", stream_id="s1")

        assert out["ok"] is True
        assert out["source_text"] == "We are not so\ndifferent, you and I."
        assert out["text"] == "PT[We are not so different, you and I.]"
        # Union of both quads.
        assert out["box"] == {"x": 180, "y": 20, "width": 460, "height": 90}

    def test_same_text_reports_unchanged_without_touching_the_model(self, ocr, translator):
        ocr.result = FakeOcrResult([("Hold the line!", 0.95, quad(10, 10, 300, 40))])

        out = sp.process_frame(b"png", "pt", prev_text="hold  THE line!", stream_id="s1")

        assert out == {"ok": True, "unchanged": True}
        assert translator == []

    def test_empty_band_clears(self, ocr, translator):
        ocr.result = FakeOcrResult([])

        out = sp.process_frame(b"png", "pt", prev_text="Hold the line!", stream_id="s1")

        assert out == {"ok": True, "text": "", "source_text": "", "box": None}
        assert translator == []

    def test_low_confidence_and_player_chrome_are_noise(self, ocr, translator):
        ocr.result = FakeOcrResult(
            [
                ("1:23:45", 0.99, quad(10, 80, 80, 20)),      # timestamp
                ("02 / 10", 0.99, quad(500, 80, 60, 20)),     # counter
                ("maybe words", 0.30, quad(100, 10, 200, 30)),  # below confidence
            ]
        )

        # A real line was up; the noise-only frame must CLEAR it, not translate
        # the player chrome that remains.
        out = sp.process_frame(b"png", "pt", prev_text="Hold the line!", stream_id="s1")

        assert out == {"ok": True, "text": "", "source_text": "", "box": None}
        assert translator == []

    def test_lines_are_ordered_top_to_bottom_not_ocr_order(self, ocr, translator):
        ocr.result = FakeOcrResult(
            [
                ("second line", 0.9, quad(100, 60, 300, 30)),
                ("first line", 0.9, quad(120, 15, 280, 30)),
            ]
        )

        out = sp.process_frame(b"png", "pt", prev_text="", stream_id="s1")

        assert out["source_text"] == "first line\nsecond line"


class TestTranslationMemory:
    def test_repeated_line_hits_the_cache(self, ocr, translator):
        ocr.result = FakeOcrResult([("See you, old friend.", 0.9, quad(10, 10, 300, 40))])

        first = sp.process_frame(b"png", "pt", prev_text="", stream_id="s1")
        # Same line reappears later in the movie (prev_text has moved on).
        second = sp.process_frame(b"png", "pt", prev_text="something else", stream_id="s1")

        assert first["text"] == second["text"]
        assert len(translator) == 1

    def test_rolling_context_rides_along_for_gender_agreement(self, ocr, translator):
        ocr.result = FakeOcrResult([("I never meant this.", 0.9, quad(10, 10, 300, 40))])
        sp.process_frame(b"png", "pt", prev_text="", stream_id="s1")

        ocr.result = FakeOcrResult([("Then where were you?", 0.9, quad(10, 10, 300, 40))])
        sp.process_frame(b"png", "pt", prev_text="I never meant this.", stream_id="s1")

        assert len(translator) == 2
        assert "I never meant this." in translator[1][-1]["content"]

    def test_streams_do_not_share_context(self, ocr, translator):
        ocr.result = FakeOcrResult([("line one", 0.9, quad(10, 10, 300, 40))])
        sp.process_frame(b"png", "pt", prev_text="", stream_id="movie-a")

        ocr.result = FakeOcrResult([("line two", 0.9, quad(10, 10, 300, 40))])
        sp.process_frame(b"png", "pt", prev_text="line one", stream_id="movie-b")

        assert "line one" not in translator[1][-1]["content"]


class TestHelpers:
    def test_wrap_balances_words_and_never_drops_them(self):
        assert sp.wrap_subtitle("a b c d e f", 5) == "a b c\nd e f"
        assert sp.wrap_subtitle("supercalifragilistic", 5) == "supercalifragilistic"
        assert sp.wrap_subtitle("", 42) == ""

    def test_normalize_absorbs_ocr_jitter(self):
        assert sp.normalize_subtitle_text(" We  ARE\nnot ") == "we are not"

    def test_decode_rejects_non_png_oversize_and_bad_base64(self):
        import base64

        good = "data:image/png;base64," + base64.b64encode(b"x" * 10).decode()
        assert sp.decode_image_data_url(good, 100) == b"x" * 10

        with pytest.raises(ValueError):
            sp.decode_image_data_url("data:image/jpeg;base64,abcd", 100)
        with pytest.raises(ValueError):
            sp.decode_image_data_url("data:image/png;base64,!!!!", 100)
        with pytest.raises(ValueError):
            sp.decode_image_data_url(
                "data:image/png;base64," + base64.b64encode(b"y" * 200).decode(), 100
            )

    def test_missing_ocr_dep_fails_with_the_install_hint(self, monkeypatch, translator):
        monkeypatch.setattr(sp, "_ocr_engine", None)
        monkeypatch.setattr(sp, "_ocr_engine_error", None)
        monkeypatch.setattr(sp, "_decode_image", lambda _data: "img")

        import builtins

        real_import = builtins.__import__

        def no_rapidocr(name, *args, **kwargs):
            if name == "rapidocr":
                raise ImportError("No module named 'rapidocr'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_rapidocr)

        with pytest.raises(RuntimeError, match="hermes-agent\\[subtitles\\]"):
            sp.process_frame(b"png", "pt")

    def test_translation_unwraps_a_quoted_answer(self, ocr, monkeypatch):
        import agent.auxiliary_client as aux

        response = MagicMock()
        response.choices[0].message.content = '"Segurem a linha!"'
        monkeypatch.setattr(aux, "call_llm", lambda *a, **k: response)

        ocr.result = FakeOcrResult([("Hold the line!", 0.9, quad(10, 10, 300, 40))])
        out = sp.process_frame(b"png", "pt", prev_text="", stream_id="s1")

        assert out["text"] == "Segurem a linha!"
