"""A batch trajectory that used the Tool Search bridge survives the merge.

The agent defers tools (process_manage from the terminal toolset, image_generate, MCP and
plugin tools) behind tool_search / tool_describe / tool_call. The trajectories.jsonl merge
drops any row whose tool_stats name a tool outside ALL_POSSIBLE_TOOLS as a hallucination,
so every finished run that went through the bridge used to vanish from the dataset while
its prompt stayed marked completed.
"""

import json

import batch_runner
from agent.agent_runtime_helpers import convert_to_trajectory_format
from batch_runner import BatchRunner
from tools.tool_search_catalog import BRIDGE_TOOL_NAMES


def test_trajectory_through_the_bridge_is_kept_in_the_merge(tmp_path, monkeypatch):
    bridge_turns = [
        {"role": "assistant", "content": f"<REASONING_SCRATCHPAD>need {name}</REASONING_SCRATCHPAD>",
         "tool_calls": [{"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": "{}"}}]}
        for name in sorted(BRIDGE_TOOL_NAMES)
    ]
    messages = [{"role": "user", "content": "q"}]
    for turn in bridge_turns:
        messages += [turn, {"role": "tool", "tool_call_id": turn["tool_calls"][0]["id"], "content": '{"ok": true}'}]
    messages.append({"role": "assistant", "content": "<REASONING_SCRATCHPAD>done</REASONING_SCRATCHPAD>answer"})

    class BridgeAgent:
        _convert_to_trajectory_format = convert_to_trajectory_format

        def __init__(self, **kwargs):
            pass

        def _format_tools_for_system_message(self):
            return "[]"

        def run_conversation(self, prompt, task_id=None):
            return {"messages": messages, "completed": True, "api_calls": 4}

        def close(self):
            pass

    monkeypatch.setattr(batch_runner, "AIAgent", BridgeAgent)
    monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda name: ["terminal"])
    config = {"distribution": "default", "model": "m", "max_iterations": 5, "verbose": False}
    batch_runner._process_batch_worker((0, [(0, {"prompt": "q"})], str(tmp_path), set(), config))

    runner = BatchRunner.__new__(BatchRunner)
    runner.output_dir = tmp_path
    kept, _files = runner._combine_batch_files()

    assert kept == 1
    row = json.loads((tmp_path / "trajectories.jsonl").read_text(encoding="utf-8"))
    assert all(row["tool_stats"][name]["count"] == 1 for name in BRIDGE_TOOL_NAMES)
