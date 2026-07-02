"""Unit tests for the reasoning-trace loop detector (agent/loop_detector.py).

The content detector never sees reasoning tokens; this second detector, fed
reasoning deltas with looser thresholds, catches egregious <think> loops (e.g.
Qwen3 cycling a plan for minutes) while leaving normal varied reasoning alone.
"""

from __future__ import annotations

from agent.loop_detector import (
    StreamLoopDetector,
    build_reasoning_loop_detector,
    load_reasoning_loop_detection_config,
)


def _feed_all(det, text, chunk=13):
    for i in range(0, len(text), chunk):
        if det.feed(text[i : i + chunk]):
            return True
    return False


# The recurring cycle from the real 06-29 Qwen reasoning loop (condensed).
_CYCLE = [
    "Actually, I think the most productive thing is to just try to log in through my browser instance.",
    "OK, I'm going to take action now. No more deliberation.",
    "Navigate to sign-in page. Fill credentials. Click sign in. Check if authenticated.",
    "Let me do that now.",
    "Actually, I realize I've been going in circles. Let me just be direct with the user.",
    "Let me take a screenshot to see the visual state of the page.",
    "Actually, I just realized - I should check if the user signed in on the same browser instance.",
]


# ----- SHOULD trip -----

def test_trips_on_real_reasoning_loop():
    cfg = load_reasoning_loop_detection_config({"loop_detection": {"reasoning": {"enabled": True}}})
    det = StreamLoopDetector(cfg)
    assert _feed_all(det, "\n".join(_CYCLE * 25) + "\n") is True
    assert "block repetition" in det.reason()


# ----- should NOT trip -----

def test_no_trip_on_varied_reasoning():
    cfg = load_reasoning_loop_detection_config({"loop_detection": {"reasoning": {"enabled": True}}})
    det = StreamLoopDetector(cfg)
    text = "\n".join(
        f"Step {i}: {w} changes the {x} path, so I will {y} before continuing."
        for i, (w, x, y) in enumerate(
            [("auth", "cookie", "check headers"), ("retry", "backoff", "add jitter"),
             ("cache", "ttl", "invalidate it"), ("schema", "migration", "add a column"),
             ("timeout", "socket", "raise the limit"), ("encoding", "utf8", "sanitize input"),
             ("index", "query", "add a btree"), ("lock", "contention", "shard the keyspace")] * 6
        )
    ) + "\n"
    assert _feed_all(det, text) is False


# ----- config / factory -----

def test_factory_none_when_disabled():
    cfg = load_reasoning_loop_detection_config({"loop_detection": {"reasoning": {"enabled": False}}})
    assert cfg.enabled is False

    class _A:
        _reasoning_loop_detection_cfg = cfg

    assert build_reasoning_loop_detector(_A()) is None


def test_env_override_disables(monkeypatch):
    monkeypatch.setenv("HERMES_REASONING_LOOP_DETECTION_ENABLED", "0")
    cfg = load_reasoning_loop_detection_config({"loop_detection": {"reasoning": {"enabled": True}}})
    assert cfg.enabled is False


def test_reasoning_defaults_are_looser_than_content():
    # Sanity: the reasoning window is larger and its block threshold stricter
    # (lower) than the content defaults, so normal reasoning is safer.
    from agent.loop_detector import LoopDetectionConfig, REASONING_DEFAULTS

    content = LoopDetectionConfig()
    assert REASONING_DEFAULTS.window_chars > content.window_chars
    assert REASONING_DEFAULTS.consecutive_line_threshold > content.consecutive_line_threshold
    assert REASONING_DEFAULTS.block_repeat_ratio_threshold < content.block_repeat_ratio_threshold
