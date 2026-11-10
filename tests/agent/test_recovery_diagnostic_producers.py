"""Real recovery producers retain work/evidence while hiding diagnostic copies."""
import json
from unittest.mock import patch

import pytest
from agent.status_output import StatusOutputMixin
from agent.turn_overflow import _Recovery, _recover_payload_too_large
from agent.turn_retry_state import TurnRetryState


class Agent(StatusOutputMixin):
    log_prefix = ""
    platform = "cli"
    suppress_status_output = False
    _mute_post_response = False
    _executing_tools = False
    model = "fixture"
    tools = []
    max_iterations = 1

    def __init__(self):
        self.printed, self.observed, self.persisted = [], [], []
        self._print_fn = lambda *a, **k: self.printed.append(" ".join(map(str, a)))
        self.status_callback = lambda k, t: self.observed.append((k, t))

    def _has_stream_consumers(self):
        return False

    def _compress_context(self, messages, *a, **k):
        return self.compressed, "system"

    def _persist_session(self, *args):
        self.persisted.append(args)


@pytest.fixture(params=[None, False, True])
def policy(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    config = {} if request.param is None else {"display": {"suppress_warning_notifications": request.param}}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    return request.param is True


@pytest.mark.parametrize("local", [False, True])
def test_outer_loop_diagnostic_preserves_verdict_and_trace(policy, local, capsys, caplog):
    from agent.turn_loop_errors import handle_outer_loop_error
    import agent.conversation_loop as loop
    agent = Agent()
    # Traceback classification executes normally using a real frame filename.
    with patch.object(loop, "_LOCAL_PROCESSING_MODULES", {"test_recovery_diagnostic_producers"} if local else set()):
        try:
            raise RuntimeError("raw engine diagnostic detail")
        except RuntimeError as exc:
            verdict = handle_outer_loop_error(agent, e=exc, _outer_error_count=0,
                api_call_count=1, messages=[], conversation_history=[],
                _turn_exit_reason=None, failed=False, final_response="")
    output = capsys.readouterr().out + "\n".join(agent.printed)
    assert ("raw engine diagnostic detail" in output) is not policy
    assert verdict.action == "break"
    assert bool(verdict.final_response)
    assert verdict.failed is (not local)
    assert "Outer loop error" in caplog.text
    assert "raw engine diagnostic detail" in caplog.text


def test_missing_requirements_keeps_tool_inventory(policy, monkeypatch, capsys):
    from agent.agent_init import _load_tools
    import model_tools
    agent = Agent()
    agent.quiet_mode = agent.save_trajectories = agent._use_prompt_caching = False
    agent.ephemeral_system_prompt = None
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [{"function": {"name": "offline_tool"}}])
    monkeypatch.setattr(model_tools, "check_toolset_requirements", lambda: {"fixture_dependency": False})
    _load_tools(agent, None, None)
    output = capsys.readouterr().out + "\n".join(agent.printed)
    assert ("Some tools may not work due to missing requirements" in output) is not policy
    assert "Loaded 1 tools" in output
    assert agent.valid_tool_names == {"offline_tool"}


@pytest.mark.parametrize("shape", ["message_count", "token_count", "payload_bytes"])
def test_retry_buffer_classification_survives_terminal_flush(policy, shape, monkeypatch):
    from gateway.warning_notifications import DiagnosticText
    agent = Agent()
    messages = [{"role": "user", "content": "a" * 4000}, {"role": "assistant", "content": "b" * 4000}]
    agent.compressed = messages[:1] if shape == "message_count" else [
        {"role": m["role"], "content": m["content"][:10]} for m in messages]
    st = _Recovery(agent=agent, messages=messages, api_messages=messages,
        system_message="system", active_system_prompt="system", conversation_history=[],
        approx_tokens=2000, compression_attempts=0, effective_task_id="offline",
        api_call_count=1, max_compression_attempts=3)
    retry = TurnRetryState()
    monkeypatch.setattr("agent.conversation_compression.conversation_history_after_compression", lambda a, m, h: h)
    monkeypatch.setattr("agent.turn_overflow.time.sleep", lambda _: None)
    if shape == "payload_bytes":
        _recover_payload_too_large(st, retry)
        assert retry.restart_with_compressed_messages
    else:
        _, shrank, _ = st.compress_scored_by_tokens(2000)
        assert shrank
    assert agent._retry_status_buffer
    assert all(isinstance(text, DiagnosticText) for _, text in agent._retry_status_buffer)
    outcome = st.fail_turn("Requested task failed.", notices=("diagnostic terminal detail",), log=("offline terminal failure",))
    assert outcome.result["failed"] is True
    assert outcome.result["final_response"] == "Requested task failed."
    assert len(agent.persisted) == 1
    assert not agent._retry_status_buffer
    assert bool(agent.printed) is not policy
    assert agent.observed  # operator callback retained even when presentation hides
    agent.printed.clear()
    agent._emit_status("ordinary compaction progress")
    assert agent.printed == ["ordinary compaction progress"]
