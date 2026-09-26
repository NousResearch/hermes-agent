"""``pre_verify`` fires for turns that changed the world through effect-capable tools
(terminal, execute_code, …) with no ``write_file``/``patch`` edit, and stays silent
for read-only turns. Before this, a deploy script or an email send never reached the
gate because only the file-mutation ledger was consulted.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_stop_gates import _pre_verify_nudge
from run_agent import AIAgent


def _response(content="composed report"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="verify-effect-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _gated(hook):
    return (
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch("hermes_cli.plugins.get_pre_verify_continue_message", side_effect=hook),
        patch("agent.verify_hooks.max_verify_nudges", return_value=2),
    )


def test_effect_tool_turn_reaches_pre_verify_without_file_edits(agent):
    agent._turn_file_mutation_paths = set()
    agent._turn_effect_tools = {"terminal"}
    hook = MagicMock(return_value="show the deploy output")
    p1, p2, p3 = _gated(hook)
    with p1, p2, p3:
        assert _pre_verify_nudge(agent, "Deployed.", 0) == "show the deploy output"
    kwargs = hook.call_args.kwargs
    assert kwargs["changed_paths"] == [] and kwargs["effect_tools"] == ["terminal"]


def test_read_only_turn_never_reaches_pre_verify(agent):
    agent._turn_file_mutation_paths = set()
    agent._turn_effect_tools = set()
    hook = MagicMock(return_value="should not be asked")
    p1, p2, p3 = _gated(hook)
    with p1, p2, p3:
        assert _pre_verify_nudge(agent, "Here is what I found.", 0) is None
    hook.assert_not_called()


def test_turn_loop_records_effect_tools_and_resets_per_turn(agent, monkeypatch):
    """The ledger is populated by the tool-commit recorder and cleared at turn start."""
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    agent._interruptible_api_call = lambda _kwargs: _response()
    with patch("hermes_cli.lifecycle.has_hook", return_value=False), patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        agent.run_conversation("look around")
    assert agent._turn_effect_tools == set(), "reset at turn start; a read-only turn adds nothing"

    agent._record_file_mutation_result("terminal", {"command": "true"}, '{"output": "ok"}', False)
    agent._record_file_mutation_result("read_file", {"path": "x"}, '{"content": ""}', False)
    agent._record_file_mutation_result("web_search", {"query": "x"}, "{}", False)
    assert agent._turn_effect_tools == {"terminal"}
    assert agent._turn_file_mutation_paths == set(), "terminal is not a file mutation"
