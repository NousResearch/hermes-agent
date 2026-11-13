"""The stance section is measured, not assumed: one canonical heading, counts in telemetry."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_compressor_prompt_rules import UNVERIFIED_SECTION_HEADING, normalize_stance_section

_CANONICAL = UNVERIFIED_SECTION_HEADING


def _user_turns() -> list[dict]:
    return [
        {"role": "user", "content": "Investigate the flaky gateway timeout."},
        {"role": "assistant", "content": "I suspect DNS lag, though I have not verified it yet."},
    ]


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        instance = ContextCompressor(
            model="test/model", threshold_percent=0.50, protect_first_n=0, protect_last_n=2, quiet_mode=True,
        )
    instance.tail_token_budget = 80
    return instance


def _summary_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _run_summary(compressor: ContextCompressor, summary_text: str) -> None:
    with patch("agent.context_compressor.call_llm", return_value=_summary_response(summary_text)):
        compressor._generate_summary(_user_turns())


def test_counts_hypothesis_lines_in_the_section():
    text = (
        f"## Goal\nFix it.\n\n{_CANONICAL}\n"
        "UNVERIFIED: DNS lag is the cause — one slow lookup observed\n"
        "- UNVERIFIED: the retry budget is too small — inferred from timeouts\n"
        "UNVERIFIED: proxy idle timeout — no evidence yet\n"
    )

    _, report = normalize_stance_section(text)

    assert report == {"present": True, "hypothesis_lines": 3, "explicit_none": False}


def test_variant_headings_are_canonicalized():
    for variant in ("## Unverified Hypotheses", "### Working Hypotheses", "## Unverified", "## Unverified Claims"):
        text, report = normalize_stance_section(f"## Goal\nx\n\n{variant}\nUNVERIFIED: a — b\n")
        assert report["present"] is True, variant
        heading_lines = [line for line in text.splitlines() if line.lstrip("# ").strip() == "Unverified / Working Hypotheses"]
        assert heading_lines == [_CANONICAL], (variant, text)


def test_line_count_stops_at_the_next_heading():
    """A later section that happens to mention UNVERIFIED must not inflate the stance count."""
    text = (
        f"## Goal\nx\n\n{_CANONICAL}\nUNVERIFIED: one — basis\n\n"
        "## Next Steps\nUNVERIFIED: not part of the stance section\n"
    )

    _, report = normalize_stance_section(text)

    assert report["hypothesis_lines"] == 1


def test_explicit_none_only_counts_inside_the_section():
    inside = normalize_stance_section(f"## Goal\nx\n\n{_CANONICAL}\nNone.\n")[1]
    outside = normalize_stance_section(f"## Goal\nx\n\n{_CANONICAL}\nUNVERIFIED: a — b\n\n## Open Questions\nNone.")[1]

    assert inside["explicit_none"] is True
    assert outside["explicit_none"] is False


def test_missing_section_is_reported_and_warned(compressor, caplog):
    compressor._begin_compression_telemetry(current_tokens=1_000)

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        _run_summary(compressor, "## Goal\nFix the timeout.\n\n## Completed Actions\n1. Ran the suite — passed [tool: terminal]")

    telemetry = compressor._active_compression_telemetry
    assert telemetry["stance_section_present"] is False
    assert telemetry["stance_hypothesis_lines"] == 0
    assert any("omitted the" in record.message for record in caplog.records)


def test_present_section_is_recorded_without_a_warning(compressor, caplog):
    compressor._begin_compression_telemetry(current_tokens=1_000)
    summary = (
        f"## Goal\nFix the timeout.\n\n{_CANONICAL}\n"
        "UNVERIFIED: DNS lag is the cause — one slow lookup observed\n"
    )

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        _run_summary(compressor, summary)

    telemetry = compressor._active_compression_telemetry
    assert telemetry["stance_section_present"] is True
    assert telemetry["stance_hypothesis_lines"] == 1
    assert not any("omitted the" in record.message for record in caplog.records)
