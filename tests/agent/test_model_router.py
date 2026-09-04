from agent.model_router import (
    Candidate,
    extract_features,
    route_turn,
)


def candidates():
    return [
        Candidate("fast", "openai", context_window=128_000, reasoning=False, vision=False, quality=0.4, cost=0.1),
        Candidate("think", "kimi-coding", context_window=1_000_000, reasoning=True, vision=True, quality=1.0, cost=0.8),
        Candidate("vision", "openrouter", context_window=200_000, reasoning=True, vision=True, quality=0.7, cost=0.5),
    ]


def test_features_detect_coding_reasoning_and_images():
    features = extract_features("请 debug this Python traceback and explain why it fails", has_images=True)
    assert features.coding is True
    assert features.reasoning is True
    assert features.vision is True


def test_features_detect_image_content_blocks():
    features = extract_features([{"type": "text", "text": "inspect"}, {"type": "image_url", "image_url": {"url": "x"}}])
    assert features.vision is True


def test_off_preserves_current_model():
    decision = route_turn("simple question", candidates(), current_model="current", mode="off")
    assert decision.selected_model == "current"
    assert decision.reason == "disabled"


def test_suggest_never_changes_model_but_explains_choice():
    decision = route_turn("implement a complex algorithm", candidates(), current_model="fast", mode="suggest")
    assert decision.selected_model == "fast"
    assert decision.suggestion == "think"
    assert decision.reason == "suggestion"


def test_auto_filters_vision_and_selects_once():
    decision = route_turn("analyze this screenshot", candidates(), current_model="fast", mode="auto", has_images=True)
    assert decision.selected_model == "think"
    assert decision.features.vision is True
    assert "vision" in decision.explanation


def test_auto_falls_back_when_no_candidate_meets_constraints():
    decision = route_turn("review this image", [candidates()[0]], current_model="fast", mode="auto", has_images=True)
    assert decision.selected_model == "fast"
    assert decision.reason == "fallback"
    assert decision.rejected[0].startswith("fast:")


def test_scoring_is_deterministic_on_ties():
    pool = [
        Candidate("b", "p", context_window=1000, quality=0.5, cost=0.5),
        Candidate("a", "p", context_window=1000, quality=0.5, cost=0.5),
    ]
    assert route_turn("hello", pool, current_model="x", mode="auto").selected_model == "a"
