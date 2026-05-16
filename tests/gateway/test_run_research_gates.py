from gateway.run import (
    _extract_public_research_url,
    _extract_research_progress_lines,
    _format_direct_research_progress,
    _format_direct_research_result,
    _format_mock_research_progress,
    _format_mock_research_result,
    _looks_like_manual_research_request,
    _normalize_research_rigor,
    _research_subject,
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
    assert _looks_like_manual_research_request(
        "mock research the best browser for hermes"
    )


def test_tool_progress_label_is_abstract():
    assert _tool_progress_label("web_search") == "🌐 browsing"
    assert _tool_progress_label("terminal") == "🛠️ tinkering"
    assert _tool_progress_label("session_search") == "🧠 remembering"


def test_extract_research_progress_lines_returns_last_three_labels():
    output = "\n".join([
        "┊ 📚 skill evidence-report-publishing 0.0s",
        "┊ 💻 $ which hermes && pwd && whoami 0.8s",
        "┊ 🔎 web_search some query 1.2s",
        "┊ 💻 $ python verify.py 0.4s",
        "┊ 💻 $ python3 render.py slug 0.4s",
    ])
    assert _extract_research_progress_lines(output, limit=3) == [
        "🌐 browsing",
        "🛠️ tinkering",
        "🔗 publishing",
    ]


def test_extract_public_research_url_prefers_real_slug_over_placeholder():
    output = "\n".join([
        "Final response must contain ONLY the public https://research.briankeefe.dev/... URL.",
        "https://research.briankeefe.dev/20260516-mild-shellfish-reaction\x1b[0m",
    ])
    assert _extract_public_research_url(output) == (
        "https://research.briankeefe.dev/20260516-mild-shellfish-reaction"
    )


def test_research_subject_truncates_long_prompt():
    subject = _research_subject("Research " + "x" * 100)
    assert not subject.startswith("Research ")
    assert subject.endswith("...")


def test_research_subject_strips_mock_research_prefix():
    assert _research_subject("mock research the best browser") == "the best browser"


def test_format_direct_research_progress_prefixes_subject():
    text = _format_direct_research_progress("the best browser", ["📚 skimming", "🌐 browsing", "🛠️ tinkering"])
    assert text.startswith("Researching the best browser\nlive run · gathering sources\n\n")
    assert "✓ framing" in text
    assert "✓ browsing" in text
    assert "◉ shaping" in text
    assert text.endswith("○ publishing")


def test_format_direct_research_result_includes_report_link():
    text = _format_direct_research_result(
        "the best browser",
        "https://research.briankeefe.dev/browser-report",
    )
    assert text.startswith("Research complete · the best browser\nlive run · published\n\n")
    assert "✓ publishing" in text
    assert "\n\nReport\n" in text
    assert text.endswith("https://research.briankeefe.dev/browser-report")


def test_format_mock_research_progress_marks_current_step():
    text = _format_mock_research_progress("the best browser", 2)
    assert text.startswith("Researching the best browser\nmock preview · no tokens burned\n\n")
    assert "✓ framing" in text
    assert "✓ browsing" in text
    assert "◉ shaping" in text
    assert "○ publishing" in text


def test_format_mock_research_result_includes_report_link():
    text = _format_mock_research_result(
        "the best browser",
        "https://research.briankeefe.dev/mock-browser-report",
    )
    assert text.startswith("Research complete · the best browser\n")
    assert "✓ publishing" in text
    assert text.endswith("https://research.briankeefe.dev/mock-browser-report")
