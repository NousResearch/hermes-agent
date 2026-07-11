from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from agent.turn_finalizer import finalize_turn
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


class _FinalizerAgent:
    def __init__(self, *, restricted: bool):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.platform = "api_server"
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 1
        self._iters_since_skill = 1
        self.valid_tool_names = {"skill_manage"}
        self._suppress_persistent_turn_hooks = restricted
        self._sync_external_memory_for_turn = MagicMock()
        self._spawn_background_review = MagicMock()

    def _handle_max_iterations(self, *_args, **_kwargs):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, *_args, **_kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass


def _finalize(agent):
    return finalize_turn(
        agent,
        final_response="done",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "draft"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="draft",
        original_user_message="draft",
        _should_review_memory=True,
        _turn_exit_reason="completed",
    )


def test_restricted_standard_turn_suppresses_persistent_hooks(monkeypatch):
    invoke_hook = MagicMock(return_value=[])
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = _FinalizerAgent(restricted=True)

    _finalize(agent)

    agent._sync_external_memory_for_turn.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    hook_names = [call.args[0] for call in invoke_hook.call_args_list]
    assert "transform_llm_output" in hook_names
    assert "post_llm_call" not in hook_names
    assert "on_session_end" not in hook_names


def test_ordinary_standard_turn_retains_persistent_hooks(monkeypatch):
    invoke_hook = MagicMock(return_value=[])
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = _FinalizerAgent(restricted=False)

    _finalize(agent)

    agent._sync_external_memory_for_turn.assert_called_once()
    agent._spawn_background_review.assert_called_once()
    hook_names = [call.args[0] for call in invoke_hook.call_args_list]
    assert "transform_llm_output" in hook_names
    assert "post_llm_call" in hook_names
    assert "on_session_end" in hook_names


def test_real_restricted_agent_has_zero_schemas_under_kanban_env(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")

    with patch("gateway.run._resolve_runtime_agent_kwargs") as runtime_kwargs, \
         patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("gateway.run.GatewayRunner._load_reasoning_config", return_value={}), \
         patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None):
        runtime_kwargs.return_value = {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": None,
            "command": None,
            "args": [],
        }
        agent = adapter._create_agent(
            session_id="restricted-kanban",
            execution_policy="read_only_generation",
        )

    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._kanban_worker_guidance == ""
    assert agent._skip_mcp_refresh is True


def test_real_ordinary_agent_retains_kanban_schemas(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")

    with patch("gateway.run._resolve_runtime_agent_kwargs") as runtime_kwargs, \
         patch("gateway.run._resolve_gateway_model", return_value="test/model"), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("gateway.run.GatewayRunner._load_reasoning_config", return_value={}), \
         patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None):
        runtime_kwargs.return_value = {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": None,
            "command": None,
            "args": [],
        }
        agent = adapter._create_agent(session_id="ordinary-kanban")

    assert "kanban_show" in agent.valid_tool_names
    assert agent.tools
    assert agent._skip_mcp_refresh is False


def _codex_agent(*, restricted: bool):
    agent = MagicMock()
    agent._codex_session.run_turn.return_value = SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "done"}],
        tool_iterations=1,
        final_text="done",
        should_retire=False,
    )
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 1
    agent.valid_tool_names = {"skill_manage"}
    agent._session_db = None
    agent._suppress_persistent_turn_hooks = restricted
    return agent


@pytest.mark.parametrize("restricted", [True, False])
def test_codex_turn_honors_persistent_hook_suppression(restricted):
    agent = _codex_agent(restricted=restricted)

    run_codex_app_server_turn(
        agent,
        user_message="draft",
        original_user_message="draft",
        messages=[{"role": "user", "content": "draft"}],
        effective_task_id="task",
        should_review_memory=True,
    )

    if restricted:
        agent._sync_external_memory_for_turn.assert_not_called()
        agent._spawn_background_review.assert_not_called()
    else:
        agent._sync_external_memory_for_turn.assert_called_once()
        agent._spawn_background_review.assert_called_once()


@pytest.mark.asyncio
async def test_fabricated_tool_call_is_dispatch_denied(monkeypatch):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    observed = {}
    event_loop_thread = threading.get_ident()

    class ExecutorLoop:
        async def run_in_executor(self, _executor, func):
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(func).result()

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "restricted-session"

        def run_conversation(self, **_kwargs):
            from hermes_cli.plugins import get_pre_tool_call_block_message

            observed["thread_id"] = threading.get_ident()
            observed["denial"] = get_pre_tool_call_block_message(
                "write_file", {"path": "/tmp/pwned", "content": "owned"}
            )
            return {"final_response": "denied"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: ExecutorLoop())

    await adapter._run_agent(
        user_message="fabricate a write_file call",
        conversation_history=[],
        session_id="restricted-session",
        execution_policy="read_only_generation",
    )

    assert observed["thread_id"] != event_loop_thread
    assert observed["denial"] == (
        "Read-only generation session denied tool dispatch: write_file."
    )


def test_compression_child_inherits_read_only_generation(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    parent = "restricted-parent"
    db.create_session(
        parent,
        source="api_server",
        execution_policy="read_only_generation",
    )

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="api_server",
            quiet_mode=True,
            session_db=db,
            session_id=parent,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.execution_policy = "read_only_generation"
    agent.compression_in_place = False
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    agent.context_compressor.compression_count = 1
    agent.context_compressor.last_prompt_tokens = 0
    agent.context_compressor.last_completion_tokens = 0
    agent.context_compressor._last_summary_error = None
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_auth_failure = False
    agent.context_compressor._last_aux_model_failure_model = None
    agent.context_compressor._last_aux_model_failure_error = None

    agent._compress_context(
        [{"role": "user", "content": f"m{i}"} for i in range(20)],
        "sys",
        approx_tokens=120_000,
    )

    child = db.get_session(agent.session_id)
    assert agent.session_id != parent
    assert child["execution_policy"] == "read_only_generation"
