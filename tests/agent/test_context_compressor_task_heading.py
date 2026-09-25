"""Task-snapshot heading identity across the summarizer prompt, the template, and grounding.

The iterative-update instruction, the emitted template, and ``_ground_historical_task_snapshot`` must agree on
one heading. A leftover ``## Active Task`` section is not disclaimed by SUMMARY_PREFIX and reads as live work,
so grounding must replace it (and any duplicate task section) rather than prepend a second one.
"""


from agent.context_compressor import ContextCompressor, HISTORICAL_TASK_HEADING

_LEGACY_ACTIVE_TASK_HEADING = "## Active Task"


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]




def test_grounding_collapses_alias_and_duplicate_task_sections():
    """A summarizer that emits the legacy alias, or both headings, ends up with exactly one grounded section."""
    body = (
        f"{HISTORICAL_TASK_HEADING}\nUser asked: 'stale canonical'\n\n"
        "## Goal\nthing\n\n"
        f"{_LEGACY_ACTIVE_TASK_HEADING}\nUser asked: 'stale alias'\n\n"
        "## Constraints & Preferences\n- none\n"
    )
    grounded = ContextCompressor._ground_historical_task_snapshot.__func__(
        ContextCompressor, body, [{"role": "user", "content": "fresh ask"}]
    )

    headings = _headings(grounded)
    assert headings.count(HISTORICAL_TASK_HEADING) == 1
    assert _LEGACY_ACTIVE_TASK_HEADING not in headings
    assert headings[1:] == ["## Goal", "## Constraints & Preferences"]
    assert "fresh ask" in grounded
    assert "stale" not in grounded
