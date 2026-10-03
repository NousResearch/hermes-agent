"""Cache-safe tool-iteration pacing governor contracts."""

from http.server import HTTPServer
import threading
from types import SimpleNamespace

from tests.agent.test_empty_tool_name_loop_dampening import _MockHandler, _tc_resp, _text_resp


def _agent(used=30):
    return SimpleNamespace(
        iteration_budget=SimpleNamespace(used=used, max_total=60),
        _iteration_governor_soft_injected=False,
        _iteration_governor_checkpoint_injected=False,
    )


def test_governor_emits_each_threshold_once_on_tool_tail():
    from agent.turn_iteration_prep import _maybe_inject_iteration_governor

    agent = _agent()
    messages = [{"role": "tool", "content": "result"}]
    assert _maybe_inject_iteration_governor(agent, messages)
    assert "50%" in messages[-1]["content"]
    assert not _maybe_inject_iteration_governor(agent, messages)
    agent.iteration_budget.used = 45
    assert _maybe_inject_iteration_governor(agent, messages)
    assert "75%" in messages[-1]["content"]


def test_prepare_iteration_wires_governor_into_provider_bound_messages(monkeypatch):
    from agent.turn_iteration_prep import prepare_iteration

    agent = _agent()
    agent.step_callback = None
    agent._nous_wire_pending = None
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._drain_pending_steer = lambda: None
    agent.run_budget_seconds = None
    agent.budget_warning_ratio = None
    agent.logger = None
    agent._sanitize_args_cursor = None
    agent.session_id = "test"
    agent._sanitize_tool_call_arguments = lambda *args, **kwargs: 0
    messages = [{"role": "tool", "content": "result"}]
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.repair_message_sequence_with_cursor", lambda *_args: 0
    )
    prepared = prepare_iteration(agent, messages=messages, api_call_count=1)
    assert "tool-iteration budget at 50%" in prepared.messages[-1]["content"]


def test_both_governor_notices_reach_real_provider_payload(monkeypatch, tmp_path):
    """Exercise both thresholds through the production loop and a local HTTP provider."""
    from run_agent import AIAgent

    class Handler(_MockHandler):
        captured_requests = []
        response_queue = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / ".hermes"
    (home / "logs").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    try:
        agent = AIAgent(
            api_key="test", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            provider="openai-compat", model="test", max_iterations=4,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, platform="cli",
        )
        Handler.captured_requests = []
        Handler.response_queue = [
            _tc_resp("unknown_test_tool"),
            _tc_resp("unknown_test_tool"),
            _tc_resp("unknown_test_tool"),
            _text_resp("done"),
        ]
        agent.run_conversation("work", conversation_history=[])
        tool_contents = [m.get("content", "") for request in Handler.captured_requests[1:]
                         for m in request.get("messages", []) if m.get("role") == "tool"]
        assert any("tool-iteration budget at 50%" in content for content in tool_contents)
        assert any("tool-iteration budget at 75%" in content for content in tool_contents)
    finally:
        server.shutdown()
        thread.join(timeout=2)
