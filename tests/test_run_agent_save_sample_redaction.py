"""``run_agent.py --save_sample`` writes a trajectory file; secrets must be redacted (#108495)."""

import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import convert_to_trajectory_format
from run_agent import _save_sample_trajectory

SECRET = "sk-" + "proj-savesample-secret-1234567890"


def test_save_sample_redacts_query_and_conversations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    query = f"export OPENAI_API_KEY={SECRET} and deploy"
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "terminal", "arguments": json.dumps({"command": f"export OPENAI_API_KEY={SECRET}"})},
        }]},
        {"role": "tool", "tool_call_id": "c1", "content": f"OPENAI_API_KEY={SECRET}"},
    ]
    agent = SimpleNamespace(_format_tools_for_system_message=lambda: "[]")
    agent._convert_to_trajectory_format = lambda msgs, q, done: convert_to_trajectory_format(agent, msgs, q, done)

    _save_sample_trajectory(agent, {"messages": messages, "completed": True}, query, "test-model")

    (sample,) = tmp_path.glob("sample_*.json")
    raw = sample.read_text(encoding="utf-8")
    assert SECRET not in raw
    entry = json.loads(raw)
    assert entry["model"] == "test-model" and entry["completed"] is True
    assert entry["query"].endswith("and deploy")
    assert [t["from"] for t in entry["conversations"]] == ["system", "human", "gpt", "tool"]
