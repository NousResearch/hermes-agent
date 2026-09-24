"""Behavior contract for task-aware pre-compaction block transforms."""

from unittest.mock import patch

import pytest

from agent.compaction_hooks import (
    COMPACTION_HOOK_PROVENANCE_KEY,
    transform_compaction_input,
)
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
)
from hermes_state import SessionDB


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=8_000):
        return ContextCompressor(model="test-model", quiet_mode=True, config_context_length=8_000)


def _history() -> list[dict]:
    messages = [{"role": "system", "content": "system"}]
    for index in range(30):
        messages.append({"role": "user", "content": f"question {index} " + "u" * 400})
        if index == 2:
            messages.extend([
                {
                    "role": "assistant",
                    "content": "running tool",
                    "tool_calls": [{
                        "id": "call-big",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "tool_call_id": "call-big", "content": "RAW_TOOL_RESULT_" + "x" * 20_000},
            ])
        messages.append({
            "role": "assistant",
            "content": ("DROP_ME " if index == 4 else f"answer {index} ") + "a" * 400,
        })
    messages.append({"role": "user", "content": "CURRENT TASK: prepare the release checklist"})
    return messages


def _summary_input(monkeypatch, *, hook_enabled: bool, invoke_hook) -> list[dict]:
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: hook_enabled)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    compressor = _compressor()
    captured: dict = {}

    def summarize(_messages, turns, _scan, _focus, _memory, _bypass):
        captured["turns"] = turns
        return "## Active Task\nContinue the release work."

    monkeypatch.setattr(compressor, "_summarize_window", summarize)
    compressor.compress(_history(), current_tokens=100_000, force=True)
    return captured["turns"]


def test_hook_transforms_selected_blocks_and_persists_host_task_provenance(tmp_path, monkeypatch):
    captured: dict = {}

    def hook(_name, **payload):
        captured["payload"] = payload
        decisions = []
        for block in payload["blocks"]:
            content = block["content"] if isinstance(block["content"], str) else ""
            if "RAW_TOOL_RESULT_" in content:
                decisions.append({
                    "block_index": block["block_index"],
                    "action": "shorten",
                    "content": "[task-irrelevant terminal output omitted]",
                })
            elif "DROP_ME" in content:
                decisions.append({"block_index": block["block_index"], "action": "drop"})
            elif "question 3" in content:
                decisions.append({"block_index": block["block_index"], "action": "keep"})
        return [{"decisions": decisions, "task_source": {"content": "forged"}}]

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "transform_compaction_input")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)

    compressor = _compressor()
    summarized: dict = {}

    def summarize(_messages, turns, _scan, _focus, _memory, _bypass):
        summarized["turns"] = turns
        return "## Active Task\nContinue the release work."

    monkeypatch.setattr(compressor, "_summarize_window", summarize)
    output = compressor.compress(_history(), current_tokens=100_000, force=True, task_id="task-17")

    payload = captured["payload"]
    assert payload["task_text"] == "CURRENT TASK: prepare the release checklist"
    assert payload["task_source"]["task_id"] == "task-17"
    assert "terminal" in payload["tool_names"]
    assert any("RAW_TOOL_RESULT_" in str(block["content"]) for block in payload["blocks"])

    summary_input = "\n".join(str(message.get("content", "")) for message in summarized["turns"])
    assert "[task-irrelevant terminal output omitted]" in summary_input
    assert "RAW_TOOL_RESULT_" not in summary_input
    assert "DROP_ME" not in summary_input

    carrier = next(message for message in output if message.get(COMPRESSED_SUMMARY_METADATA_KEY))
    provenance = carrier["display_metadata"][COMPACTION_HOOK_PROVENANCE_KEY]
    assert provenance["task_source"] == {
        "message_index": len(_history()) - 1,
        "content": "CURRENT TASK: prepare the release checklist",
        "task_id": "task-17",
    }
    assert {record["action"] for record in provenance["decisions"]} == {"keep", "drop", "shorten"}

    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-1", source="cli")
    db.archive_and_compact("session-1", output)
    reloaded = next(
        message for message in db.get_messages_as_conversation("session-1")
        if COMPACTION_HOOK_PROVENANCE_KEY in message.get("display_metadata", {})
    )
    assert reloaded["display_metadata"][COMPACTION_HOOK_PROVENANCE_KEY] == provenance


def test_invalid_hook_result_fails_open(monkeypatch):
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": "original"}]
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda _name, **_payload: [{"decisions": [{"block_index": 0, "action": "shorten"}]}],
    )

    result = transform_compaction_input(
        messages,
        task_text="task",
        task_message_index=3,
        task_id="task-1",
        session_id="session-1",
    )

    assert result.messages is messages
    assert result.provenance is None
    assert result.applied is False


@pytest.mark.parametrize("outcome", ["malformed", "raising", "timeout_skipped"])
def test_unsuccessful_hook_outcomes_resume_baseline_pruning(monkeypatch, outcome):
    baseline = _summary_input(
        monkeypatch,
        hook_enabled=False,
        invoke_hook=lambda _name, **_payload: [],
    )

    if outcome == "malformed":
        def invoke_hook(_name, **_payload):
            return [{"decisions": [{"block_index": 0, "action": "shorten"}]}]
    elif outcome == "raising":
        def invoke_hook(_name, **_payload):
            raise RuntimeError("plugin failed")
    else:
        # Bounded hook callbacks that time out are omitted from the result list.
        def invoke_hook(_name, **_payload):
            return []

    actual = _summary_input(
        monkeypatch,
        hook_enabled=True,
        invoke_hook=invoke_hook,
    )

    assert actual == baseline
    assert "RAW_TOOL_RESULT_" not in "\n".join(str(message.get("content", "")) for message in actual)


def test_valid_empty_decisions_intentionally_keep_original_blocks(monkeypatch):
    turns = _summary_input(
        monkeypatch,
        hook_enabled=True,
        invoke_hook=lambda _name, **_payload: [{"decisions": []}],
    )

    assert "RAW_TOOL_RESULT_" in "\n".join(str(message.get("content", "")) for message in turns)


def _state_after_aborted_summary(monkeypatch, *, hook_enabled: bool) -> tuple:
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: hook_enabled)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda _name, **_payload: [{"decisions": [{"block_index": 0, "action": "shorten"}]}],
    )
    compressor = _compressor()
    compressor.abort_on_summary_failure = True
    messages = _history()
    handoff = next(message for message in messages if message.get("content", "").startswith("answer 6"))
    handoff["content"] = f"{SUMMARY_PREFIX}\nold summary"
    monkeypatch.setattr(compressor, "_summarize_window", lambda *_args: None)

    compressor.compress(messages, current_tokens=100_000, force=True)

    assert compressor._last_compress_aborted is True
    return compressor._previous_summary, compressor._summary_has_user_turn


def test_unsuccessful_hook_abort_restores_baseline_handoff_state(monkeypatch):
    baseline = _state_after_aborted_summary(monkeypatch, hook_enabled=False)
    actual = _state_after_aborted_summary(monkeypatch, hook_enabled=True)

    assert actual == baseline
    assert actual[0] is None
