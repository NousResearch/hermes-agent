"""Behavioural contracts for the mode workflow prompts."""

from __future__ import annotations

import re

import pytest

from hermes_cli.mode_prompts import PLAN_PROMPT, ULTRAPLAN_PROMPT


def _workflow_steps(prompt: str) -> list[int]:
    workflow = prompt.split("Workflow:", 1)[1]
    return [int(match.group(1)) for match in re.finditer(r"(?m)^(\d+)\. ", workflow)]


@pytest.mark.parametrize("prompt", [PLAN_PROMPT, ULTRAPLAN_PROMPT])
def test_mode_workflow_steps_are_unique_and_sequential(prompt: str) -> None:
    steps = _workflow_steps(prompt)

    assert steps == list(range(1, len(steps) + 1))


@pytest.mark.parametrize("prompt", [PLAN_PROMPT, ULTRAPLAN_PROMPT])
def test_execute_exit_names_the_registered_config_set_tool(prompt: str) -> None:
    assert "Call `config_set`" in prompt
    assert "Call `config.set`" not in prompt


@pytest.mark.parametrize("prompt", [PLAN_PROMPT, ULTRAPLAN_PROMPT])
def test_saved_plan_has_an_editable_review_gate_before_exit(prompt: str) -> None:
    review_at = prompt.index("editable review")
    exit_at = prompt.index("A/B/C/D exit")

    assert review_at < exit_at
    assert "Accept the saved file" in prompt
    assert "Edit the saved file" in prompt
    assert "Request agent changes" in prompt


@pytest.mark.parametrize("prompt", [PLAN_PROMPT, ULTRAPLAN_PROMPT])
def test_diagram_question_fits_the_four_option_tool_limit(prompt: str) -> None:
    assert "multiSelect: true" in prompt
    assert "Excalidraw, Mermaid, HTML, or No diagrams" in prompt
    assert "All diagrams" not in prompt


@pytest.mark.parametrize("prompt", [PLAN_PROMPT, ULTRAPLAN_PROMPT])
def test_exit_does_not_promise_an_unimplemented_compression_step(prompt: str) -> None:
    assert "Switch to Auto and Execute" in prompt
    assert "Compress and Execute" not in prompt
