"""Automatic exports preserve represented history; dataset exports retain their prompt override."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.session_persistence import SessionPersistenceMixin
from agent.tool_dispatch_helpers import _trajectory_normalize_msg


def _image_question():
    return [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://image.invalid/example.png"}},
    ]


def _history(kind):
    if kind == "none":
        return None
    question = _image_question() if kind == "image" else "What is two plus two?"
    rows = [{"role": "user", "content": question}, {"role": "assistant", "content": "Four."}]
    if kind == "system":
        rows.insert(0, {"role": "system", "content": "Previously supplied system instructions."})
    if kind == "summary":
        rows = [{"role": "assistant", "content": "Summary of the previous discussion.", "_compressed_summary": True}]
    return rows


def _assert_history_rows(rows, messages):
    represented = [m for m in messages if m["role"] in {"user", "assistant"}]
    assert [r["from"] for r in rows] == ["system"] + [
        "human" if m["role"] == "user" else "gpt" for m in represented
    ]
    first_user = True
    for row, message in zip(rows[1:], represented):
        if message["role"] == "user":
            expected = (
                _summarize_user_message_for_log(message["content"]) if first_user
                else _trajectory_normalize_msg(message)["content"]
            )
            assert row["value"] == expected
            first_user = False
        else:
            assert row["value"].endswith(message["content"])


@pytest.mark.parametrize("history_kind, turns, save", [
    pytest.param("none", ["First question."], True, id="single"),
    pytest.param("none", ["First question.", "Second question."], True, id="two-turns"),
    pytest.param("none", ["First question.", "Second question.", "Third question."], True, id="three-turns"),
    pytest.param("none", ["Same question.", "Same question."], True, id="repeated-question"),
    pytest.param("text", ["Second question."], True, id="supplied-history"),
    pytest.param("system", ["Second question."], True, id="leading-system"),
    pytest.param("summary", ["Question after compaction."], True, id="assistant-summary"),
    pytest.param("image", ["Question about earlier image."], True, id="historical-image"),
    pytest.param("none", [_image_question()], True, id="single-image"),
    pytest.param("text", [_image_question()], True, id="later-image"),
    pytest.param("none", ["First question.", "Second question."], False, id="disabled"),
])
def test_automatic_jsonl_keeps_the_public_conversation(tmp_path, monkeypatch, history_kind, turns, save):
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    model = "test/model"
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=131072),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://provider.invalid/v1", model=model,
            provider="custom", api_mode="chat_completions", quiet_mode=True,
            skip_memory=True, skip_context_files=True, enabled_toolsets=[], save_trajectories=save,
            max_iterations=2,
        )
    agent.client = MagicMock()
    monkeypatch.setattr(agent, "_cached_system_prompt", "Test assistant.")
    monkeypatch.setattr(agent, "_use_prompt_caching", False)
    monkeypatch.setattr(agent, "compression_enabled", False)
    monkeypatch.setattr(agent, "_session_db", None)
    monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
    history = _history(history_kind)
    snapshots = []
    requests = []
    for turn, question in enumerate(turns):
        answer = f"Answer {turn}."

        def complete(**kwargs):
            requests.append(deepcopy(kwargs["messages"]))
            return SimpleNamespace(
                id="test-response", model=model, usage=None,
                choices=[SimpleNamespace(index=0, finish_reason="stop", message=SimpleNamespace(
                    role="assistant", content=answer, tool_calls=None, reasoning=None,
                    reasoning_content=None, reasoning_details=None,
                ))],
            )

        agent.client.chat.completions.create.side_effect = complete
        old_history = deepcopy(history)
        with patch.object(agent, "_cleanup_task_resources"):
            result = agent.run_conversation(question, conversation_history=history)
        assert result["completed"] is True and result["final_response"] == answer
        assert history == old_history
        expected_users = [m["content"] for m in history or [] if m["role"] == "user"] + [question]
        assert [m["content"] for m in requests[-1] if m["role"] == "user"] == expected_users
        snapshots.append(deepcopy(result["messages"]))
        history = result["messages"]
    assert len(requests) == len(turns)
    output = tmp_path / "trajectory_samples.jsonl"
    if not save:
        assert not output.exists()
        return
    entries = [json.loads(line) for line in output.read_text(encoding="utf-8-sig").splitlines()]
    assert len(entries) == len(snapshots)
    for entry, messages in zip(entries, snapshots):
        assert entry["completed"] is True and entry["model"] == model
        _assert_history_rows(entry["conversations"], messages)
    if history_kind == "none" and len(turns) == 1:
        from run_agent import _save_sample_trajectory

        canonical = "Original dataset prompt, separate from model-facing content."
        original = deepcopy(result["messages"])
        _save_sample_trajectory(agent, result, canonical, model)
        sample = json.loads(next(tmp_path.glob("sample_*.json")).read_text(encoding="utf-8-sig"))
        assert sample["query"] == canonical
        assert sample["conversations"][1] == {"from": "human", "value": canonical}
        assert sample["conversations"][-1]["value"].endswith(result["final_response"])
        assert result["messages"] == original


class _Exporter(SessionPersistenceMixin):
    model = "test/model"
    save_trajectories = True

    def _format_tools_for_system_message(self):
        return '{"name": "lookup", "parameters": {"type": "object"}}'


@pytest.mark.parametrize("query, completed, layout", [
    pytest.param("Original dataset prompt.", True, "normal", id="canonical-prompt"),
    pytest.param("", True, "normal", id="empty-canonical-prompt"),
    pytest.param("Original dataset prompt.", False, "normal", id="failed-canonical"),
    pytest.param("Original dataset prompt.", True, "empty", id="empty-canonical-history"),
    pytest.param(None, True, "normal", id="history-mode"),
    pytest.param(None, False, "normal", id="failed-history"),
    pytest.param(None, True, "system", id="system-history"),
    pytest.param(None, True, "summary", id="user-summary"),
    pytest.param(None, True, "empty", id="empty-history"),
])
def test_history_and_canonical_exports_keep_their_contracts(tmp_path, monkeypatch, query, completed, layout):
    agent = _Exporter()
    tool_content = {
        "_multimodal": True, "text_summary": "Lookup found a match.",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,not-for-export"}}],
    }
    messages = [
        {"role": "user", "content": "Model-facing first question."},
        {"role": "assistant", "content": "", "reasoning": "Check the lookup.", "tool_calls": [{
            "id": "call_lookup", "type": "function",
            "function": {"name": "lookup", "arguments": '{"query": "first"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_lookup", "content": tool_content},
        {"role": "assistant", "content": "<REASONING_SCRATCHPAD>Found it.</REASONING_SCRATCHPAD> First answer."},
        {"role": "user", "content": "Follow-up question."},
        {"role": "assistant", "content": "Follow-up answer."},
    ]
    if layout == "empty":
        messages = []
    elif layout == "system":
        messages.insert(0, {"role": "system", "content": "Do not export this system prompt."})
    elif layout == "summary":
        messages[0] = {"role": "user", "content": "Summary of the previous discussion.", "_compressed_summary": True}
    original = deepcopy(messages)
    rows = agent._convert_to_trajectory_format(messages, query, completed)
    assert messages == original
    assert rows[0]["from"] == "system"
    assert agent._format_tools_for_system_message() in rows[0]["value"]
    assert sum(row["from"] == "system" for row in rows) == 1
    monkeypatch.chdir(tmp_path)
    agent._save_trajectory(messages, query, completed)
    target = tmp_path / ("trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl")
    saved = json.loads(target.read_text(encoding="utf-8-sig"))
    assert saved["conversations"] == rows and saved["completed"] is completed
    assert messages == original
    if layout == "empty":
        assert rows[1:] == ([] if query is None else [{"from": "human", "value": query}])
        return
    first = messages[1] if layout == "system" else messages[0]
    assert [r["value"] for r in rows if r["from"] == "human"] == [
        first["content"] if query is None else query, "Follow-up question.",
    ]
    assert [r["from"] for r in rows] == ["system", "human", "gpt", "tool", "gpt", "human", "gpt"]
    assert "Check the lookup." in rows[2]["value"] and '<tool_call>' in rows[2]["value"]
    payload = rows[3]["value"].removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
    assert json.loads(payload) == {"tool_call_id": "call_lookup", "name": "lookup", "content": tool_content["text_summary"]}
    assert "<think>Found it.</think>" in rows[4]["value"]
    assert rows[-1]["value"].endswith(messages[-1]["content"])
    assert "not-for-export" not in json.dumps(rows)
