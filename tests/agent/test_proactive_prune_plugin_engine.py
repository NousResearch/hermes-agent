"""Pin: a plugin context engine that does not override ``prune_tool_results_only`` gets no host prune.

Such an engine keeps its own store, so the host must not shorten its transcript or write archives for
it. Both real host routes are driven (after a tool round, and the cold-cache turn start), so a later
refactor of the dispatch gate cannot turn the prune on under a plugin engine without this test changing.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_engine import ContextEngine
from agent.turn_context_compaction import CompactionOutcome
from agent.turn_preflight import compress_after_tool_results
from agent.turn_proactive_prune import turn_start_cold_prune


class _PluginEngine(ContextEngine):
    @property
    def name(self):
        return "plugin-store"

    def update_from_response(self, usage):
        return None

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        return messages


def _drive_route(engine, route):
    engine.last_prompt_tokens = 400_000
    agent = SimpleNamespace(
        context_compressor=engine, compression_enabled=True, compression_checkpoint_required=False,
        session_id="s1", tools=[], _compression_feasibility_checked=True,
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read them"}]
    for i in range(20):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"row {i}\n" + "x" * 24_000},
        ])
    turn_start = len(messages)
    messages.append({"role": "user", "content": "next question"})
    original = deepcopy(messages)
    with patch("agent.turn_proactive_prune.prompt_cache_is_cold", return_value=True):
        if route == "after_tool":
            out = compress_after_tool_results(
                agent, messages=messages, system_message="sys", user_message="next question",
                active_system_prompt="sys", conversation_history=[], compression_attempts=0,
                max_compression_attempts=3, effective_task_id="t", final_response="",
                turn_exit_reason=None, current_turn_user_idx=turn_start,
            )
            assert not out.end_turn
        else:
            out = CompactionOutcome(
                messages=messages, active_system_prompt="sys", conversation_history=None,
                current_turn_user_idx=turn_start,
            )
            with patch("agent.turn_context._preflight_request_tokens", return_value=400_000):
                turn_start_cold_prune(agent, out, "t")
    assert out.messages is messages
    assert out.messages == original


@pytest.mark.parametrize("route", ["after_tool", "turn_start"])
def test_plugin_inheriting_default_hook_keeps_messages_and_creates_no_archives(tmp_path, monkeypatch, route):
    """Pin both host routes to leave a plugin's own store and transcript untouched without a host DB."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _drive_route(_PluginEngine(), route)
    assert not list((tmp_path / "cache" / "pruned").glob("**/*"))
