"""Repeat-read dedup across transcript rewrites that are not a compaction boundary.

skill_view and read_file answer a repeat of unchanged content with an "unchanged" stub that points at the
earlier result. Post-turn micro-compaction summarizes older exchanges away, and a native Responses compaction
checkpoint takes every earlier item off the wire. After either, the result a stub would point at may be gone,
so the next repeat must serve content again, as after compaction (#32106).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.file_tools_read_tracking import _read_tracker, _read_tracker_lock, _task_data
from tools.skills_tool import _skill_view_with_bump
from tools.skills_tool_dedup import _check_skill_view_dedup, reset_skill_view_dedup

TASK = "rewrite-boundary-task"
FILE_KEY = ("/x/big.txt", 1, 2000)


@pytest.fixture
def served(tmp_path, monkeypatch):
    """One real skill served to TASK, and one file read already served in the current generation."""
    home = tmp_path / ".hermes"
    skill_dir = home / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo.\n---\n# demo-skill\n\nDo the demo steps.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    call = {"id": "call_skill", "type": "function",
            "function": {"name": "skill_view", "arguments": json.dumps({"name": "demo-skill"})}}
    body = _skill_view_with_bump({"name": "demo-skill"}, task_id=TASK)
    with _read_tracker_lock:
        _read_tracker.pop(TASK, None)
        td = _task_data(TASK)
        td["dedup"][FILE_KEY] = 1.0
        td["dedup_generation_reads"].add(FILE_KEY)
    assert _dedup_state() == (True, True)
    yield [{"role": "user", "content": "load the skill"},
           {"role": "assistant", "content": "", "tool_calls": [call]},
           {"role": "tool", "tool_call_id": "call_skill", "content": body},
           {"role": "assistant", "content": "loaded"}]
    reset_skill_view_dedup()
    with _read_tracker_lock:
        _read_tracker.pop(TASK, None)


def _dedup_state():
    """(a skill_view repeat would get the stub, a read_file repeat would get the stub)."""
    with _read_tracker_lock:
        file_stubbed = FILE_KEY in _read_tracker.get(TASK, {}).get("dedup_generation_reads", ())
    return _check_skill_view_dedup(TASK, "demo-skill", None) is not None, file_stubbed


def _run(agent, history, monkeypatch):
    tool_defs = [{"type": "function", "function": {"name": "skill_view", "description": "skill_view",
                                                   "parameters": {"type": "object", "properties": {}}}}]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: tool_defs)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    agent.tool_delay = 0
    agent.save_trajectories = False
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call",
              lambda name, args, task_id=None, **_kw: _skill_view_with_bump(args, task_id=task_id)),
    ):
        result = agent.run_conversation("go on", conversation_history=history, task_id=TASK)
    assert result["completed"] is True
    return result


@pytest.mark.parametrize("splice", [True, False], ids=["spliced", "no_op"])
def test_micro_compaction_splice_rearms_read_dedup(served, splice, monkeypatch):
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=4)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    compressor = MagicMock()  # never asks for full compaction; only the post-turn micro pass runs
    compressor.protect_first_n, compressor.protect_last_n = 3, 20
    compressor.threshold_tokens, compressor.context_length = 500_000, 1_000_000
    compressor.last_prompt_tokens = 1_000
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_defer_preflight_to_real_usage.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor._micro_compact_enabled = True
    compressor._flush_scan_cursor_invalidated = False
    # A splice returns a new list with the oldest exchange folded into the rolling summary;
    # a no-op (nothing to absorb, or a defrag) returns the input list.
    compressor._micro_compact = (
        (lambda messages: [{"role": "user", "content": "[MICRO] loaded demo-skill"}, *messages[4:]])
        if splice else (lambda messages: messages))
    agent.context_compressor = compressor
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", reasoning_content=None, reasoning=None, tool_calls=None), finish_reason="stop")],
        model="test/model", usage=None)

    result = _run(agent, served, monkeypatch)

    assert any(m.get("content") == "[MICRO] loaded demo-skill" for m in result["messages"]) is splice
    # A splice took the skill body and may have taken the file body: both repeats serve content again.
    # A no-op pass removed nothing, so both stubs stay.
    assert _dedup_state() == ((False, False) if splice else (True, True))


def test_native_checkpoint_rearms_read_dedup(served, monkeypatch):
    """The response carrying a server checkpoint re-views the skill: the next request starts at the
    checkpoint, so that re-view must deliver the body, not a stub for a result the wire dropped."""
    from run_agent import AIAgent
    monkeypatch.setattr("agent.model_metadata._fetch_codex_oauth_context_lengths_with_source", lambda _t, **_kw: ({}, False))
    agent = AIAgent(model="gpt-5.6", base_url="https://chatgpt.com/backend-api/codex", api_key="codex-token",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=4)
    agent.codex_responses_native_compaction = True
    agent.runtime_capabilities = {"native_compaction": True}
    usage = SimpleNamespace(input_tokens=9, output_tokens=1, total_tokens=10)
    responses = [
        SimpleNamespace(output=[
            SimpleNamespace(type="compaction", encrypted_content="opaque-checkpoint"),
            SimpleNamespace(type="function_call", id="fc_view", call_id="call_view", name="skill_view",
                            arguments=json.dumps({"name": "demo-skill"}))],
            usage=usage, status="completed", model="gpt-5.6"),
        SimpleNamespace(output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="done")])],
                        usage=usage, status="completed", model="gpt-5.6"),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = _run(agent, served, monkeypatch)

    re_view = [m for m in result["messages"] if m.get("role") == "tool" and m.get("tool_call_id") == "call_view"]
    assert re_view and "Do the demo steps." in re_view[-1]["content"]
    assert _dedup_state()[1] is False  # the file read served before the checkpoint is off the wire too
