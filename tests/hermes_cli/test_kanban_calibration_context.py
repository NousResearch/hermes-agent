"""Regression coverage for optional estimate-calibration prompt context."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_decomposer_appends_plugin_calibration_context_without_history(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Document the API", body="Write docs.", triage=True)

    answer = json.dumps({
        "fanout": False,
        "rationale": "fits",
        "title": "Document the API",
        "body": "Write the docs.",
        "estimated_context_tokens": 20_000,
    })
    calibration = (
        "Estimate calibration for task type 'documentation' (aggregated; no raw history):\n"
        "- tokens: median actual 3,000; median signed error -20%; median absolute error 25%\n"
        "- minutes: median actual 12; median signed error +10%; median absolute error 20%\n"
        "Use this only to calibrate a range, not as a point guarantee."
    )

    with patch("agent.auxiliary_client.call_llm", return_value=_response(answer)) as call_llm, patch(
        "hermes_cli.plugins.invoke_hook", return_value=[{"context": calibration}]
    ) as invoke_hook:
        outcome = decomp.decompose_task(task_id, author="test")

    assert outcome.ok
    invoke_hook.assert_called_once_with(
        "kanban_decomposer_context",
        task_id=task_id,
        title="Document the API",
        body="Write docs.",
        context_budget_tokens=150_000,
    )
    messages = call_llm.call_args.kwargs["messages"]
    assert calibration in messages[1]["content"]
    assert "never treat it as\n    a point guarantee" in messages[0]["content"]


def test_decomposer_keeps_prompt_valid_when_no_plugin_context(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Small task", triage=True)

    answer = json.dumps({
        "fanout": False,
        "rationale": "fits",
        "title": "Small task",
        "body": "Do it.",
        "estimated_context_tokens": 20_000,
    })
    with patch("agent.auxiliary_client.call_llm", return_value=_response(answer)) as call_llm, patch(
        "hermes_cli.plugins.invoke_hook", return_value=[]
    ):
        outcome = decomp.decompose_task(task_id, author="test")

    assert outcome.ok
    assert "Calibration context:" not in call_llm.call_args.kwargs["messages"][1]["content"]
