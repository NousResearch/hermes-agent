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


def _summary_prompt(*, previous_summary: str = "") -> str:
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.tail_mode = "lean"
    compressor._previous_summary = previous_summary
    source = "USER: " + ("Keep this deployment constraint and its rationale. " * 20)
    return compressor._build_summary_prompt(source, 800, "deployment safety", "", True)


def test_source_copy_limit_is_the_final_prompt_rule():
    for previous_summary in ("", "## Historical Task\nolder checkpoint"):
        prompt = _summary_prompt(previous_summary=previous_summary)
        rule_pos = prompt.rfind("SOURCE COPY LIMIT:")

        assert rule_pos > prompt.rfind("MUST be quoted VERBATIM")
        assert rule_pos > prompt.rfind("FOCUS TOPIC:")
        assert "longer than 200 characters verbatim" in prompt[rule_pos:]
        assert "quote only a short key fragment and paraphrase the rest" in prompt[rule_pos:]


def test_grounding_restores_latest_user_words_after_summary_generation():
    latest_request = (
        "Please keep this exact deployment constraint: never modify production data "
        "until the dry-run output has been reviewed. " * 7
    ).strip()
    generated = (
        f"{HISTORICAL_TASK_HEADING}\nUser asked for a deployment check\n\n"
        "## Goal\nValidate deployment safety"
    )

    grounded = ContextCompressor._ground_historical_task_snapshot.__func__(
        ContextCompressor,
        generated,
        [{"role": "user", "content": latest_request}],
    )

    assert latest_request in grounded
    assert "User asked for a deployment check" not in grounded
    assert "## Goal\nValidate deployment safety" in grounded
