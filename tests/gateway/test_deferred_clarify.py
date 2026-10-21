"""Deferred clarify marker and tool-envelope semantics."""

import concurrent.futures
import json
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_deferred_marker_round_trips():
    from gateway.extensions.deferred_clarify import (
        DEFERRED_CLARIFY_KIND,
        is_deferred_clarify_result,
        make_deferred_marker,
        parse_deferred_marker,
    )

    marker = make_deferred_marker("cld_test")
    payload = json.loads(marker)
    assert payload["status"] == "deferred"
    assert payload["kind"] == DEFERRED_CLARIFY_KIND
    assert is_deferred_clarify_result(marker)
    assert parse_deferred_marker(marker) == "cld_test"


def test_marker_detection_uses_current_clarify_tool_envelope():
    from gateway.extensions.deferred_clarify import (
        find_deferred_clarify_interaction_id,
        make_deferred_marker,
    )
    from tools.clarify_tool import clarify_tool

    marker = make_deferred_marker("cld_envelope")
    content = clarify_tool(
        [{"question": "Deploy where?", "choices": ["staging", "production"]}],
        callback=lambda _questions: {"answers": {"q0": marker}, "outcome": "deferred"},
    )
    messages = [{
        "role": "tool",
        "name": "clarify",
        "tool_call_id": "call_clarify",
        "content": content,
    }]
    assert find_deferred_clarify_interaction_id(messages, {"call_clarify"}) == "cld_envelope"


def test_marker_detection_rejects_other_tool_and_previous_turn():
    from gateway.extensions.deferred_clarify import (
        find_deferred_clarify_interaction_id,
        make_deferred_marker,
    )

    marker = make_deferred_marker("cld_hidden")
    other_tool = {
        "role": "tool", "name": "terminal", "tool_call_id": "call", "content": marker,
    }
    previous_turn = {
        "role": "assistant", "tool_calls": [{"id": "call"}], "content": None,
    }
    assert find_deferred_clarify_interaction_id([other_tool], {"call"}) is None
    assert find_deferred_clarify_interaction_id([other_tool, previous_turn], {"call"}) is None


def test_late_durable_send_failure_cancels_interaction(tmp_path):
    from gateway.run_turn_runner import _durable_clarify_send_result
    from tools.clarify_interaction import create_clarify_interaction, get_interaction

    class _TimedOutFuture:
        def result(self, timeout=None):
            raise concurrent.futures.TimeoutError()

        def add_done_callback(self, callback):
            self.callback = callback

    token = set_hermes_home_override(tmp_path)
    try:
        interaction = create_clarify_interaction(
            session_key="telegram:chat",
            platform="telegram",
            question="Deploy?",
            choices=["yes"],
            ttl_seconds=3600,
        )
        future = _TimedOutFuture()
        marker = _durable_clarify_send_result(
            future, interaction_id=interaction.interaction_id, timeout=0.01
        )
        assert '"status":"deferred"' in marker

        future.callback(SimpleNamespace(success=False, error="late send failure"))
        loaded = get_interaction(interaction.interaction_id)
        assert loaded is not None
        assert loaded.status == "cancelled"
    finally:
        reset_hermes_home_override(token)


def test_recovery_prompt_carries_question_and_answer():
    from gateway.extensions.deferred_clarify import build_recovery_prompt

    prompt = build_recovery_prompt(question="Deploy where?", answer="staging")
    assert "Deploy where?" in prompt
    assert "staging" in prompt
    assert "Do not ask the same clarification again" in prompt
