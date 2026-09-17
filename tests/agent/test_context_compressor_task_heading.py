"""The iterative-update instruction must name the heading the template emits."""

from types import SimpleNamespace

from agent.context_compressor import (
    ContextCompressor,
    HISTORICAL_TASK_HEADING,
    _HISTORICAL_TASK_SECTION_RE,
)


def _iterative_prompt() -> str:
    """The prompt built when a previous summary exists (the update path)."""
    stub = SimpleNamespace(
        tail_mode="lean",
        _previous_summary="PREVIOUS SUMMARY BODY",
        _bound_summary_input=lambda text: text,
    )
    stub._summary_template_sections = ContextCompressor._summary_template_sections
    stub._build_summary_prompt = ContextCompressor._build_summary_prompt.__get__(stub)
    return stub._build_summary_prompt("NEW TURNS", 2000, None, "", True)


def test_update_instruction_names_the_emitted_heading():
    prompt = _iterative_prompt()
    assert f'Update "{HISTORICAL_TASK_HEADING}"' in prompt
    assert '"## Active Task"' not in prompt


def test_grounding_prepends_when_the_task_heading_differs():
    """Why the names must agree: a foreign heading yields two task sections."""
    body = "## Active Task\nUser asked: 'do X'\n\n## Goal\nthing\n"
    assert _HISTORICAL_TASK_SECTION_RE.search(body) is None

    grounded = ContextCompressor._ground_historical_task_snapshot.__func__(
        ContextCompressor, body, [{"role": "user", "content": "do X"}]
    )
    headings = [line for line in grounded.splitlines() if line.startswith("## ")]
    assert headings.count(HISTORICAL_TASK_HEADING) == 1
    assert "## Active Task" in headings
