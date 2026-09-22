"""Behavior contract for task-aware pre-compaction block transforms."""

from unittest.mock import patch

from agent.compaction_hooks import (
    COMPACTION_HOOK_PROVENANCE_KEY,
    transform_compaction_input,
)
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
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
