"""Kanban dispatcher route announcement formatting."""

import re

from gateway.kanban_watchers_dispatcher import _format_spawn_routes


def _kind_for(summary: str, task_id: str) -> str:
    """Extract the exact announced ``kind=`` token for *task_id*.

    Matching the whole token (not a substring) keeps a longer value such as
    ``firepower-override`` from satisfying an assertion that means
    ``firepower``.
    """
    match = re.search(rf"\b{re.escape(task_id)} route=\S+ kind=([\w-]+)", summary)
    assert match is not None, f"no route announced for {task_id}: {summary!r}"
    return match.group(1)


def test_spawn_route_summary_names_each_task_route():
    summary = _format_spawn_routes({
        "t_a": "openai-codex/gpt-5.6-sol-900k",
        "t_b": "claude-apr/claude-opus-5",
        "t_c": "openai-codex/gpt-6-astra-900k",
    })
    assert "t_a route=openai-codex/gpt-5.6-sol-900k kind=standard" in summary
    assert "t_b route=claude-apr/claude-opus-5 kind=standard" in summary
    assert "t_c route=openai-codex/gpt-6-astra-900k kind=firepower" in summary

    # The ANNOUNCE contract is the exact token `firepower`, not a prefix of it.
    assert _kind_for(summary, "t_a") == "standard"
    assert _kind_for(summary, "t_b") == "standard"
    assert _kind_for(summary, "t_c") == "firepower"
