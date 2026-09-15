"""Kanban's headless auxiliary calls retain OpenCode session affinity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_decompose as decompose
from hermes_cli import kanban_specify as specify


def _response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


@pytest.mark.parametrize("caller", [specify._call_aux, decompose._call_aux])
def test_kanban_aux_call_supplies_stable_opencode_session(caller):
    """Specify and decompose headless calls need a task-stable relay key."""
    seen = []

    def call_llm(**kwargs):
        from agent.auxiliary_client import _runtime_main_value
        from agent.opencode_affinity import merge_opencode_session_headers

        seen.append(merge_opencode_session_headers(
            {"extra_headers": {"x-existing": "keep"}}, "opencode-go", None,
            _runtime_main_value("session_id"),
        ).get("extra_headers", {}))
        return _response()

    with patch("agent.auxiliary_client.call_llm", call_llm):
        reply, reason = caller(
            "specify", "t_123", aux_task="triage_specifier", system="system", user="user",
            max_tokens=100, timeout=10,
        )

    assert (reply, reason) == ("ok", "")
    assert seen == [{"x-existing": "keep", "x-opencode-session": "kanban:t_123"}]


def test_kanban_aux_call_does_not_add_session_for_other_providers():
    """The task binding must remain inert for non-OpenCode auxiliary routes."""
    seen = []

    def call_llm(**kwargs):
        from agent.auxiliary_client import _runtime_main_value
        from agent.opencode_affinity import merge_opencode_session_headers

        seen.append(merge_opencode_session_headers(
            {"extra_headers": {"x-existing": "keep"}}, "openai", None,
            _runtime_main_value("session_id"),
        ).get("extra_headers", {}))
        return _response()

    with patch("agent.auxiliary_client.call_llm", call_llm):
        reply, reason = specify._call_aux(
            "specify", "t_123", aux_task="triage_specifier", system="system", user="user",
            max_tokens=100, timeout=10,
        )

    assert (reply, reason) == ("ok", "")
    assert seen == [{"x-existing": "keep"}]
