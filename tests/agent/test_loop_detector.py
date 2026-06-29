"""Unit tests for the streaming loop detector (agent/loop_detector.py).

Pure/fast: feed text in small fragments (simulating streamed deltas) and assert
the detector trips on degenerate repetition but NOT on legitimate (even very
long) varied output, code, or tables.
"""

from __future__ import annotations

import random

from agent.loop_detector import (
    LoopDetectionConfig,
    StreamLoopDetector,
    build_stream_loop_detector,
    load_loop_detection_config,
)


def _feed_all(det: StreamLoopDetector, text: str, chunk: int = 7) -> bool:
    for i in range(0, len(text), chunk):
        if det.feed(text[i : i + chunk]):
            return True
    return False


# ----- SHOULD trip -----

def test_trips_on_repeated_line():
    det = StreamLoopDetector()
    assert _feed_all(det, "The quick brown fox sat here.\n" * 50) is True
    assert "consecutive" in det.reason()


def test_trips_on_no_newline_short_period():
    det = StreamLoopDetector()
    assert _feed_all(det, "abcd" * 300) is True  # period 4, no newlines


def test_trips_on_no_newline_phrase():
    det = StreamLoopDetector()
    # The exact pathology seen in the wild: "I'll go. I'll go. ..." on one line.
    assert _feed_all(det, "I'll go. " * 80) is True


def test_trips_on_low_entropy_block_via_heavy_path():
    # Long single "line" of low-entropy filler with no exploitable short period
    # at the tail (spaces vary) still trips the heavy redundancy check.
    det = StreamLoopDetector()
    assert _feed_all(det, ("data data data data data data " * 400)) is True


def test_trips_on_block_repetition():
    # Paragraph/plan cycling: 2 distinct multi-line PROSE blocks repeated. High
    # per-char entropy + moderate n-gram diversity, so only the distinct-line-ratio
    # (block-repetition) check catches it. Regression for the real-world loop where
    # the model cycled "You're right... let me write a proper script" paragraphs.
    a = (
        "You're right, the whole approach is fundamentally broken because it uses "
        "text label positions as proxies for the figure extent.\n"
        "Let me write a proper figure extraction script now.\n"
    )
    b = (
        "Yes, this is doable in Python with no inference needed at all here.\n"
        "The better approach uses embedded image extraction from the document.\n"
    )
    det = StreamLoopDetector()
    assert _feed_all(det, (a + b) * 8) is True
    assert "block repetition" in det.reason()


# ----- should NOT trip -----

def test_no_trip_on_distinct_imports():
    det = StreamLoopDetector()
    mods = [
        "os", "sys", "json", "math", "re", "time", "typing", "pathlib", "collections",
        "itertools", "functools", "dataclasses", "asyncio", "logging", "subprocess",
        "hashlib", "random", "shutil", "tempfile", "threading", "queue", "socket",
        "struct", "enum", "copy", "io", "abc", "datetime", "uuid", "zlib",
    ]
    text = "".join(f"import {m}\n" for m in mods)
    assert _feed_all(det, text) is False


def test_no_trip_on_markdown_table():
    det = StreamLoopDetector()
    rows = "".join(f"| row{i} | value {i*7} | note about item {i} |\n" for i in range(12))
    text = "| col a | col b | col c |\n| --- | --- | --- |\n" + rows
    assert _feed_all(det, text) is False


def test_no_trip_on_repeated_line_inside_code_fence():
    # 8 identical short lines inside a fence must NOT trip (doubled threshold +
    # short-line bar). 8 < 12.
    det = StreamLoopDetector()
    body = "    return None\n" * 8
    text = "Here is the code:\n```python\ndef f():\n" + body + "```\nDone.\n"
    assert _feed_all(det, text) is False


def test_no_trip_on_long_varied_prose():
    # 50 KB of high-entropy varied text -> proves length-independence.
    random.seed(42)
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
             "nu xi omicron pi rho sigma tau upsilon phi chi psi omega quick brown "
             "fox lazy dog jumps river mountain code agent stream token vector").split()
    lines = []
    while sum(len(x) for x in lines) < 50_000:
        lines.append(" ".join(random.choices(words, k=random.randint(8, 16))) + ".\n")
    det = StreamLoopDetector()
    assert _feed_all(det, "".join(lines)) is False


def test_no_trip_on_short_repeats():
    det = StreamLoopDetector()
    assert _feed_all(det, "Yes. Yes. Yes.\nNo. No.\n") is False


# ----- overhead -----

def test_heavy_check_overhead_bounded():
    cfg = LoopDetectionConfig(check_every_bytes=1024)
    det = StreamLoopDetector(cfg)
    calls = {"n": 0}
    orig = det._heavy_check

    def counting():
        calls["n"] += 1
        return orig()

    det._heavy_check = counting  # type: ignore[method-assign]
    random.seed(1)
    text = "".join(
        " ".join(random.choices("alpha beta gamma delta token vector code".split(), k=12)) + ".\n"
        for _ in range(800)
    )
    _feed_all(det, text)
    # heavy check runs at most ~ len(text)/check_every_bytes times.
    assert calls["n"] <= (len(text) // cfg.check_every_bytes) + 2


# ----- config / factory -----

def test_factory_returns_none_when_disabled():
    cfg = load_loop_detection_config({"loop_detection": {"enabled": False}})
    assert cfg.enabled is False

    class _A:
        _loop_detection_cfg = cfg

    assert build_stream_loop_detector(_A()) is None


def test_env_override_disables(monkeypatch):
    monkeypatch.setenv("HERMES_LOOP_DETECTION_ENABLED", "0")
    cfg = load_loop_detection_config({"loop_detection": {"enabled": True}})
    assert cfg.enabled is False


def test_config_parses_overrides():
    cfg = load_loop_detection_config(
        {"loop_detection": {"enabled": True, "consecutive_line_threshold": 3, "window_chars": 2048}}
    )
    assert cfg.enabled is True
    assert cfg.consecutive_line_threshold == 3
    assert cfg.window_chars == 2048
