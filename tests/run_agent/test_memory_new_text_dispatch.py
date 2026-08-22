"""The agent loop must forward ``new_text`` to memory_tool.

``new_text`` is an alias for ``content`` on the memory tool's single-op
shape (see MEMORY_SCHEMA in tools/memory_tool.py). The alias is resolved
inside ``memory_tool`` itself, so it only works if the caller actually
passes the argument through.

``memory`` never reaches the registry dispatcher — both agent loops
special-case it and call ``memory_tool`` directly. That makes these call
sites the only thing standing between the schema's promise and the tool,
and unit tests that call ``memory_tool`` directly cannot see a break in
them.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.memory_tool as memory_tool_module
from agent.agent_runtime_helpers import invoke_tool
from agent.tool_executor import execute_tool_calls_sequential
from run_agent import AIAgent
from tools.memory_tool import MemoryStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_tool_module, "get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=5000, user_char_limit=5000)
    s.load_from_disk()
    return s


def _memory_call(call_id: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="memory", arguments=json.dumps(arguments)),
    )


def _make_agent(tmp_path: Path, store: MemoryStore) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._memory_store = store
    agent._memory_manager = None
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(
        side_effect=lambda _name, _args, result, **_kwargs: result
    )
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(
        side_effect=lambda _name, result: result
    )
    return agent


class TestInvokeToolPath:
    """agent/agent_runtime_helpers.py::invoke_tool — the concurrent path."""

    def _invoke(self, store, args):
        agent = SimpleNamespace(
            _memory_store=store,
            _memory_manager=None,
            session_id="test-session",
            _current_turn_id="turn-1",
            _current_api_request_id="req-1",
            _turns_since_memory=0,
            _iters_since_skill=0,
        )
        return invoke_tool(
            agent,
            "memory",
            args,
            "task-1",
            tool_call_id="call-1",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    def test_new_text_writes_the_entry(self, store):
        result = json.loads(
            self._invoke(
                store,
                {"action": "add", "target": "user", "new_text": "prefers metric units"},
            )
        )

        assert result["success"] is True
        assert "prefers metric units" in store.user_entries

    def test_content_still_works(self, store):
        result = json.loads(
            self._invoke(
                store,
                {"action": "add", "target": "user", "content": "prefers dark mode"},
            )
        )

        assert result["success"] is True
        assert "prefers dark mode" in store.user_entries

    def test_content_wins_when_both_are_set(self, store):
        """MEMORY_SCHEMA documents this precedence; keep it pinned."""
        self._invoke(
            store,
            {
                "action": "add",
                "target": "user",
                "content": "from content",
                "new_text": "from new_text",
            },
        )

        assert "from content" in store.user_entries
        assert "from new_text" not in store.user_entries


class TestSequentialPath:
    """agent/tool_executor.py::execute_tool_calls_sequential."""

    def test_new_text_writes_the_entry(self, tmp_path, store):
        agent = _make_agent(tmp_path, store)
        assistant = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                _memory_call(
                    "call-1",
                    {"action": "add", "target": "user", "new_text": "likes celsius"},
                )
            ],
        )
        messages = []

        execute_tool_calls_sequential(agent, assistant, messages, "task-1", 0)

        assert "likes celsius" in store.user_entries
