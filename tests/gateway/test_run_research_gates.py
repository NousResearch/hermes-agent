from gateway.run import (
    _looks_like_manual_research_request,
    _normalize_research_rigor,
    _tool_progress_label,
)


def test_normalize_research_rigor_detects_inline_tier():
    assert _normalize_research_rigor("Rigor tier: Deep") == "deep"
    assert _normalize_research_rigor("standard") == "standard"


def test_manual_research_request_heuristic_matches_target_cases():
    assert _looks_like_manual_research_request(
        "Research whether Dripos is better than the competition"
    )
    assert _looks_like_manual_research_request(
        "Research what most researchers believe will happen in 10 years with AI"
    )
    assert _looks_like_manual_research_request(
        "Research bagel places near my home that's open right now"
    )


def test_tool_progress_label_is_abstract():
    assert _tool_progress_label("web_search") == "🌐 browsing"
    assert _tool_progress_label("terminal") == "🛠️ tinkering"
    assert _tool_progress_label("session_search") == "🧠 remembering"
