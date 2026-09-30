from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_init import _apply_agent_section
from agent.codex_runtime import run_codex_app_server_turn


def test_configured_codex_turn_timeout_reaches_runtime():
    agent = SimpleNamespace(run_budget_seconds=1, _empty_guard_enabled=True)
    _apply_agent_section(agent, {"agent": {"codex_app_server_turn_timeout": 123}})
    session = SimpleNamespace(
        run_turn=lambda **kwargs: (setattr(session, "kwargs", kwargs) or SimpleNamespace(
            interrupted=False, error=None, final_text="done", tool_iterations=0,
            projected_messages=[], thread_id=None, turn_id=None, should_retire=False,
        )),
    )
    agent._codex_session = session
    agent.compression_checkpoint_required = False
    agent._user_interrupt_requested = False
    agent._interrupt_requested = False
    agent._session_db = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 10
    agent.valid_tool_names = set()
    agent._sync_external_memory_for_turn = None
    agent._record_codex_app_server_compaction = None
    agent._record_codex_app_server_usage = None
    with patch("agent.codex_runtime._ensure_codex_session"), patch("agent.codex_runtime._start_codex_thread"), patch(
        "agent.codex_runtime._consume_user_interrupt", return_value=(False, None)
    ), patch("agent.codex_runtime._finish_codex_turn", return_value={}):
        run_codex_app_server_turn(agent, user_message="hello", original_user_message="hello", messages=[], effective_task_id="t")
    assert session.kwargs["turn_timeout"] == 123.0


def test_codex_turn_timeout_default_preserves_six_hundred_seconds():
    agent = SimpleNamespace(run_budget_seconds=1, _empty_guard_enabled=True)
    _apply_agent_section(agent, {"agent": {}})
    assert agent._codex_app_server_turn_timeout == 600.0
