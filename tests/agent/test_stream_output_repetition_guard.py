import pytest

from agent.stream_output_repetition_guard import (
    StreamOutputRepetitionError,
    StreamOutputRepetitionGuard,
    truncate_repeated_tail,
)


INCIDENT_LINE = (
    "- **Texture**: `TEXTURE_LOCAL.md` (creative trace, emergence). "
    "This line should not repeat indefinitely.\n"
)


def test_guard_detects_dominated_multiline_tail() -> None:
    guard = StreamOutputRepetitionGuard(
        min_total_chars=200,
        repeat_threshold=4,
    )

    with pytest.raises(StreamOutputRepetitionError) as exc:
        for _ in range(4):
            guard.feed(INCIDENT_LINE)

    assert exc.value.repeat_count == 4
    assert "TEXTURE_LOCAL.md" in exc.value.repeated_unit


def test_guard_spares_repeated_status_interleaved_with_unique_content() -> None:
    guard = StreamOutputRepetitionGuard(
        min_total_chars=200,
        repeat_threshold=4,
        max_distinct_lines=3,
    )

    for index in range(20):
        guard.feed(
            f"Unique report section {index} with enough detail to count as a line.\n"
        )
        guard.feed(
            "No changes detected for this source since the previous scan.\n"
        )


def test_guard_detects_periodic_tail_without_newlines() -> None:
    guard = StreamOutputRepetitionGuard(
        min_total_chars=120,
        periodic_min_repeats=5,
    )

    with pytest.raises(StreamOutputRepetitionError) as exc:
        guard.feed("I will check the same thing again. " * 12)

    assert "periodic tail" in str(exc.value)


def test_truncate_repeated_tail_keeps_one_copy_and_marks_cut() -> None:
    text = "Healthy preamble.\n" + INCIDENT_LINE * 12

    truncated = truncate_repeated_tail(text, INCIDENT_LINE.strip())

    assert truncated.count("TEXTURE_LOCAL.md") == 1
    assert truncated.startswith("Healthy preamble.")
    assert truncated.endswith("[Output stopped: repetitive generation detected.]")
