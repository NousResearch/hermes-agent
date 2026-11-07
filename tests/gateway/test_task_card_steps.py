"""Native task cards name each step the way text progress does, and keep its argument.

Text progress already renders "Searching the web for alpha" (``agent.display`` friendly labels,
``tool_preview_length``). The card used "web_search - alpha" with the argument cut at 64
characters inside the title. Each step now carries the friendly verb as its title and the
argument as a separate ``detail``, which Slack renders under the title.
"""

from __future__ import annotations

import pytest

from agent import display
from gateway.run_turn_progress import GatewayTurnProgressMixin
from plugins.platforms.slack.adapter import SlackAdapter

TaskCard = GatewayTurnProgressMixin._TaskCardState


@pytest.fixture
def friendly_labels():
    previous = display._friendly_tool_labels
    display.set_friendly_tool_labels(True)
    yield
    display.set_friendly_tool_labels(previous)


def _started(card, call_id, tool, preview):
    card.apply_event({"type": "tool.started", "tool_call_id": call_id, "tool_name": tool, "preview": preview})
    return card.tasks[call_id]


def test_step_title_is_the_friendly_verb_and_the_argument_is_its_detail(friendly_labels):
    card = TaskCard(adapter=None)
    task = _started(card, "c1", "web_search", "alpha")
    assert (task["title"], task["detail"]) == ("Searching the web", "alpha")
    assert card.fallback_text().splitlines()[1] == "- Searching the web for alpha - running"


def test_tools_without_a_verb_keep_their_name_and_text_format(friendly_labels):
    card = TaskCard(adapter=None)
    task = _started(card, "c1", "mcp__jira__search", "project = X")
    assert (task["title"], task["detail"]) == ("mcp__jira__search", "project = X")
    assert card.fallback_text().splitlines()[1] == "- mcp__jira__search - project = X - running"


def test_disabled_friendly_labels_fall_back_to_the_tool_name():
    previous = display._friendly_tool_labels
    display.set_friendly_tool_labels(False)
    try:
        task = _started(TaskCard(adapter=None), "c1", "web_search", "alpha")
    finally:
        display.set_friendly_tool_labels(previous)
    assert (task["title"], task["detail"]) == ("web_search", "alpha")


def test_detail_is_not_cut_at_the_old_64_char_title_budget(friendly_labels):
    query = "SELECT org_id, count(*) FROM trackings WHERE created_at > '2026-01-01' GROUP BY 1 ORDER BY 2 DESC"
    task = _started(TaskCard(adapter=None), "c1", "terminal", f"bq query '{query}'")
    assert task["detail"] == f"bq query '{query}'"


def test_completion_only_event_has_a_titled_step_without_detail(friendly_labels):
    card = TaskCard(adapter=None)
    card.apply_event({"type": "tool.completed", "tool_call_id": "c9", "tool_name": "terminal"})
    assert card.tasks["c9"]["title"] == "Running" and card.tasks["c9"]["detail"] == ""
    assert card.tasks["c9"]["status"] == "complete"


def test_slack_chunk_carries_the_detail_and_omits_an_empty_one():
    with_detail = SlackAdapter._task_update_chunk(
        {"id": "c1", "title": "Searching the web", "detail": "alpha", "status": "in_progress"})
    assert with_detail["title"] == "Searching the web" and with_detail["details"] == "alpha"
    bare = SlackAdapter._task_update_chunk({"id": "c2", "title": "todo", "status": "complete"})
    assert "details" not in bare
