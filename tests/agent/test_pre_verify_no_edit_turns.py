"""#109815: the turn-end ``pre_verify`` gate must be reachable on answer-only turns.

By default the gate stays edit-only — a turn that edited no files never even consults a
``pre_verify`` hook, so nothing about default behavior changes. Setting
``agent.pre_verify_without_edits: true`` opts such turns into the same bounded hook
consultation (the hook sees ``changed_paths=[]`` and decides).
"""

from __future__ import annotations

from typing import Any, Tuple
from unittest.mock import patch

import pytest

from agent.turn_stop_gates import apply_stop_gates


class _StubAgent:
    """Minimum agent surface ``apply_stop_gates`` touches."""

    session_id = "pre-verify-no-edit"
    platform = "cli"
    model = "test/model"

    def __init__(self) -> None:
        self._pre_verify_nudges = 0
        self._kanban_stop_nudges = 0
        self._session_messages: list[dict[str, Any]] = []

    def _emit_interim_assistant_message(self, final_msg: dict[str, Any]) -> None:
        pass

    def _flush_messages_to_session_db(self, messages, conversation_history) -> None:
        pass

    def _interim_content_was_streamed(self, text: str) -> bool:
        return False


@pytest.fixture(autouse=True)
def _only_the_pre_verify_gate_decides(monkeypatch):
    """Silence the sibling stop gates (verify-on-stop / kanban) for a clean verdict."""
    for var in ("HERMES_VERIFY_ON_STOP", "HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)


def _run_answer_only_turn(agent: _StubAgent) -> Tuple[list, Any]:
    """One real gate pass for a turn that edited no files (no mutation paths set)."""
    messages = [{"role": "user", "content": "how many tests does the suite have?"}]
    final_msg = {"role": "assistant", "content": "About 39,000."}
    verdict = apply_stop_gates(
        agent, final_msg, final_response=final_msg["content"], messages=messages,
        conversation_history=[], pending_verification_response=None,
        pending_verification_response_previewed=False,
    )
    return messages, verdict


def test_answer_only_turn_never_reaches_the_hook_by_default():
    """Default off: a registered hook is not consulted and the turn finishes unchanged."""
    agent = _StubAgent()
    with (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message", return_value="verify it"
        ) as hook,
    ):
        messages, verdict = _run_answer_only_turn(agent)

    assert verdict.continue_turn is False
    assert verdict.final_response == "About 39,000."
    assert hook.call_count == 0
    assert agent._pre_verify_nudges == 0
    assert not any(m.get("_pre_verify_synthetic") for m in messages)


def test_opt_in_nudges_an_answer_only_turn(monkeypatch, tmp_path):
    """``agent.pre_verify_without_edits: true`` (real config.yaml) → the hook gates the turn."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "agent:\n  pre_verify_without_edits: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = _StubAgent()
    with (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="count the tests, then answer",
        ) as hook,
    ):
        messages, verdict = _run_answer_only_turn(agent)

    assert verdict.continue_turn is True
    assert verdict.final_response is None
    assert verdict.pending_verification_response == "About 39,000."
    # The hook sees the no-edit turn as an empty change set and decides from that.
    assert hook.call_args.kwargs["changed_paths"] == []
    nudge_row = messages[-1]
    assert nudge_row["role"] == "user"
    assert nudge_row["content"] == "count the tests, then answer"
    assert nudge_row["_pre_verify_synthetic"] is True
    assert agent._pre_verify_nudges == 1
