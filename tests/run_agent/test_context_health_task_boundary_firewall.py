from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent
from agent.working_context_packet import enforce_working_context_packet


TASK_A_ID = "phase5-task-a"
TASK_B_ID = "phase5-task-b"
RAW_A_NEEDLE = "RAW_CLOSED_TASK_A_PHASE5_PROVIDER_NEEDLE"
A_ASSISTANT_NEEDLE = "ASSISTANT_CLOSED_TASK_A_PHASE5_NEEDLE"
A_TOOL_NEEDLE = "TOOL_RESULT_CLOSED_TASK_A_PHASE5_NEEDLE"
A_COMPACT_NEEDLE = "COMPACT_SUMMARY_CLOSED_TASK_A_PHASE5_NEEDLE"
A_PLUGIN_NEEDLE = "PLUGIN_CONTEXT_CLOSED_TASK_A_PHASE5_NEEDLE"
A_MEMORY_NEEDLE = "MEMORY_PREFETCH_CLOSED_TASK_A_PHASE5_NEEDLE"
A_SESSION_SEARCH_NEEDLE = "SESSION_SEARCH_RESULT_CLOSED_TASK_A_PHASE5_NEEDLE"
TASK_A_CURRENT_PIN = "/tmp/phase5/task-a/current-pin.md"
TASK_A_WORKSPEC = "/tmp/phase5/task-a/WorkSpec.md"
TASK_A_TASK_STATE = "/tmp/phase5/task-a/task-state.json"
TASK_B_WORKSPEC = "/tmp/phase5/task-b/WorkSpec.md"


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, reasoning=None, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _make_agent(tmp_path, *, registry_enabled: bool = True, wcp_enabled: bool = True, intake_enabled: bool = False) -> AIAgent:
    cfg = {
        "context_health": {
            "enabled": bool(registry_enabled or wcp_enabled or intake_enabled),
            "runtime_behavior_enabled": bool(registry_enabled or wcp_enabled or intake_enabled),
            "pre_model_intake": {
                "enabled": bool(intake_enabled),
                "long_prompt_char_threshold": 40,
                "long_prompt_line_threshold": 3,
            },
            "task_boundary": {
                "enabled": bool(registry_enabled),
                "default_without_clear_continuation": "new_task",
                "ambiguous_action": "hold",
            },
            "task_registry": {"enabled": bool(registry_enabled)},
            "working_context_packet": {"enabled": bool(wcp_enabled)},
            "task_boundary_firewall": {"enabled": True},
        },
        "agent": {"api_max_retries": 1},
    }
    with (
        patch("run_agent.OpenAI"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=cfg),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    setattr(agent, "context_health", cfg["context_health"])
    setattr(agent, "_context_health_intake_dir", tmp_path / "intake")
    agent.client.chat.completions.create.return_value = _response()
    agent._persist_session = lambda *a, **k: None
    agent._save_trajectory = lambda *a, **k: None
    agent._cleanup_task_resources = lambda *a, **k: None
    return agent


def _capture_provider_kwargs(agent: AIAgent):
    captured: dict[str, object] = {"provider_called": False}
    real_build = agent._build_api_kwargs

    def capture_build(api_messages):
        captured["api_messages_input_to_build_kwargs"] = [dict(m) for m in api_messages]
        api_kwargs = real_build(api_messages)
        captured["api_kwargs_after_build"] = dict(api_kwargs)
        return api_kwargs

    def fake_create(**kwargs):
        captured["provider_called"] = True
        captured["provider_kwargs_at_call"] = dict(kwargs)
        return _response()

    agent._build_api_kwargs = capture_build
    agent.client.chat.completions.create.side_effect = fake_create
    return captured


def _payload_text(captured: dict[str, object]) -> str:
    return repr(
        {
            "api_messages": captured.get("api_messages_input_to_build_kwargs"),
            "api_kwargs_after_build": captured.get("api_kwargs_after_build"),
            "provider_kwargs_at_call": captured.get("provider_kwargs_at_call"),
        }
    )


def _registry_snapshot(*, active_task_id: str = TASK_B_ID):
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": active_task_id,
        "tasks": {
            TASK_A_ID: {
                "task_id": TASK_A_ID,
                "status": "closed" if active_task_id != TASK_A_ID else "active",
                "workspec_path": TASK_A_WORKSPEC,
                "task_state_path": TASK_A_TASK_STATE,
                "current_pin_path": TASK_A_CURRENT_PIN,
                "latest_turn_id": "session:phase5-task-a:aaaa1111",
                "raw_transcript_excerpt": RAW_A_NEEDLE,
                "compact_summary": A_COMPACT_NEEDLE,
            },
            TASK_B_ID: {
                "task_id": TASK_B_ID,
                "status": "active" if active_task_id == TASK_B_ID else "linked",
                "workspec_path": TASK_B_WORKSPEC,
                "latest_turn_id": "session:phase5-task-b:bbbb2222",
                "linked_task_ids": [TASK_A_ID] if active_task_id == TASK_A_ID else [],
            },
        },
    }


def _install_registry(monkeypatch, *, action: str = "new_task", active_task_id: str = TASK_B_ID, hold: bool = False):
    calls: list[dict[str, object]] = []
    module = types.ModuleType("agent.task_registry")

    def resolve_task_for_turn(**kwargs):
        calls.append(kwargs)
        if hold:
            return SimpleNamespace(
                action="hold",
                effective_task_id=None,
                reason="ambiguous_task_relation",
                registry_snapshot=_registry_snapshot(active_task_id=active_task_id),
                linked_task_id=None,
                hold_response="Context Health HOLD: ambiguous task relation; no provider call was made.",
            )
        return SimpleNamespace(
            action=action,
            effective_task_id=active_task_id,
            reason="explicit_continuation_reference" if active_task_id == TASK_A_ID else "no_clear_continuation_evidence",
            registry_snapshot=_registry_snapshot(active_task_id=active_task_id),
            linked_task_id=TASK_A_ID if active_task_id == TASK_A_ID else None,
            hold_response=None,
        )

    setattr(module, "resolve_task_for_turn", resolve_task_for_turn)
    setattr(module, "record_completed_workspec_state", lambda **kwargs: {"task_id": kwargs.get("task_id")})
    monkeypatch.setitem(sys.modules, "agent.task_registry", module)
    return calls


def _install_firewall_exception(monkeypatch):
    module = types.ModuleType("agent.task_boundary_firewall")

    def enforce_task_boundary_firewall(**_kwargs):
        raise RuntimeError("synthetic_firewall_failure")

    setattr(module, "enforce_task_boundary_firewall", enforce_task_boundary_firewall)
    monkeypatch.setitem(sys.modules, "agent.task_boundary_firewall", module)


def _history_with_closed_a(*, include_tool: bool = True, include_compact: bool = True):
    history = [
        {"role": "user", "content": f"task A raw context {RAW_A_NEEDLE}"},
        {"role": "assistant", "content": f"task A assistant context {A_ASSISTANT_NEEDLE}"},
    ]
    if include_tool:
        history.extend(
            [
                {"role": "assistant", "content": "tool call holder", "tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "session_search", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call-a", "name": "session_search", "content": f"session search found A {A_TOOL_NEEDLE} {A_SESSION_SEARCH_NEEDLE}"},
            ]
        )
    if include_compact:
        history.append({"role": "assistant", "content": f"compacted A summary {A_COMPACT_NEEDLE}"})
    return history


def test_firewall_disabled_preserves_existing_provider_payload_pass_through(tmp_path):
    agent = _make_agent(tmp_path, registry_enabled=False, wcp_enabled=False, intake_enabled=False)
    captured = _capture_provider_kwargs(agent)

    result = agent.run_conversation(
        "short B while firewall disabled",
        conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
        task_id="explicit-disabled-task",
    )

    payload = _payload_text(captured)
    assert result["completed"] is True
    assert captured["provider_called"] is True
    assert RAW_A_NEEDLE in payload
    assert getattr(agent, "_current_task_id") == "explicit-disabled-task"


def test_closed_a_raw_text_not_in_b_provider_payload(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation(
        "start unrelated B task",
        conversation_history=_history_with_closed_a(),
    )

    payload = _payload_text(captured)
    assert TASK_B_ID in payload
    assert RAW_A_NEEDLE not in payload
    assert A_ASSISTANT_NEEDLE not in payload
    assert A_TOOL_NEEDLE not in payload


def test_closed_a_safe_paths_not_in_independent_b_wcp(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation(
        "start unrelated B task without using A",
        conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
    )

    payload = _payload_text(captured)
    assert TASK_B_ID in payload
    assert TASK_A_CURRENT_PIN not in payload
    assert TASK_A_WORKSPEC not in payload
    assert TASK_A_TASK_STATE not in payload


def test_explicit_continue_a_includes_only_safe_a_pointer(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="continue_task", active_task_id=TASK_A_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation(
        f"continue {TASK_A_ID} using its WorkSpec",
        conversation_history=_history_with_closed_a(),
    )

    payload = _payload_text(captured)
    assert TASK_A_ID in payload
    assert TASK_A_WORKSPEC in payload or TASK_A_TASK_STATE in payload or TASK_A_CURRENT_PIN in payload
    assert RAW_A_NEEDLE not in payload
    assert A_TOOL_NEEDLE not in payload
    assert A_COMPACT_NEEDLE not in payload


def test_ambiguous_relation_returns_hold_without_provider_call(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)
    persisted: list[object] = []
    agent._persist_session = lambda *a, **k: persisted.append((a, k))
    raw_prompt = f"continue the previous one {RAW_A_NEEDLE}"

    result = agent.run_conversation(
        raw_prompt,
        conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
    )

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert captured["provider_called"] is False
    assert "hold" in result.get("error", "").lower()
    assert raw_prompt not in repr(persisted)
    assert RAW_A_NEEDLE not in repr(result)


def test_enabled_wcp_firewall_exception_returns_safe_hold_without_unfiltered_registry(tmp_path, monkeypatch):
    _install_firewall_exception(monkeypatch)
    agent = _make_agent(tmp_path)
    setattr(agent, "_context_health_task_registry_snapshot", _registry_snapshot(active_task_id=TASK_B_ID))
    setattr(agent, "_context_health_task_registry_decision", SimpleNamespace(action="new_task", effective_task_id=TASK_B_ID))

    decision = enforce_working_context_packet(
        policy=getattr(agent, "context_health"),
        agent=agent,
        api_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "start unrelated B"}],
        messages=[{"role": "user", "content": "start unrelated B"}],
        original_user_message="start unrelated B",
        current_turn_user_idx=0,
        effective_task_id=TASK_B_ID,
        turn_id="turn-b",
        session_id="session-b",
    )

    assert decision.action == "hold"
    assert decision.api_messages is None
    rendered = repr(decision)
    assert TASK_A_CURRENT_PIN not in rendered
    assert TASK_A_WORKSPEC not in rendered
    assert TASK_A_TASK_STATE not in rendered
    assert RAW_A_NEEDLE not in rendered
    assert "synthetic_firewall_failure" not in rendered


def test_enabled_pre_append_firewall_exception_holds_without_provider_or_raw_persist(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    _install_firewall_exception(monkeypatch)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)
    persisted: list[object] = []
    agent._persist_session = lambda *a, **k: persisted.append((a, k))
    raw_prompt = f"start unrelated B with raw private body {RAW_A_NEEDLE}"

    result = agent.run_conversation(
        raw_prompt,
        conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
    )

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert captured["provider_called"] is False
    assert "hold" in result.get("error", "").lower()
    assert "task_boundary_firewall" in result.get("error", "")
    assert getattr(agent, "_context_health_task_boundary_firewall_decision") is not None
    assert raw_prompt not in repr(persisted)
    assert RAW_A_NEEDLE not in repr(result)
    assert "synthetic_firewall_failure" not in repr(result)


def test_firewall_disabled_with_exception_preserves_pass_through(tmp_path, monkeypatch):
    _install_firewall_exception(monkeypatch)
    agent = _make_agent(tmp_path, registry_enabled=False, wcp_enabled=False, intake_enabled=False)
    captured = _capture_provider_kwargs(agent)

    result = agent.run_conversation(
        "short B while firewall disabled",
        conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
        task_id="explicit-disabled-task",
    )

    assert result["completed"] is True
    assert captured["provider_called"] is True
    assert RAW_A_NEEDLE in _payload_text(captured)
    assert getattr(agent, "_current_task_id") == "explicit-disabled-task"


def test_fake_pre_llm_call_plugin_a_context_not_in_b_provider_payload(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)

    with patch("hermes_cli.plugins.invoke_hook", return_value=[{"context": f"plugin returns closed A {A_PLUGIN_NEEDLE}"}]):
        agent.run_conversation(
            "start unrelated B task",
            conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
        )

    assert A_PLUGIN_NEEDLE not in _payload_text(captured)


class _FakeMemoryManager:
    def on_turn_start(self, *_args, **_kwargs):
        return None

    def prefetch_all(self, _query):
        return f"memory returns closed A {A_MEMORY_NEEDLE}"


def test_fake_memory_prefetch_a_context_not_in_b_provider_payload(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    agent._memory_manager = _FakeMemoryManager()
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation(
        "start unrelated B task",
        conversation_history=_history_with_closed_a(include_tool=False, include_compact=False),
    )

    assert A_MEMORY_NEEDLE not in _payload_text(captured)


def test_prior_session_search_tool_result_a_not_in_b_provider_payload(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation(
        "start unrelated B task",
        conversation_history=_history_with_closed_a(include_tool=True, include_compact=False),
    )

    payload = _payload_text(captured)
    assert A_SESSION_SEARCH_NEEDLE not in payload
    assert A_TOOL_NEEDLE not in payload


def test_firewall_filters_provider_payload_without_db_schema_or_transcript_deletion(tmp_path, monkeypatch):
    _install_registry(monkeypatch, action="new_task", active_task_id=TASK_B_ID)
    agent = _make_agent(tmp_path)
    captured = _capture_provider_kwargs(agent)
    original_history = _history_with_closed_a(include_tool=False, include_compact=False)
    history_copy = [dict(m) for m in original_history]

    result = agent.run_conversation(
        "start unrelated B task",
        conversation_history=original_history,
    )

    assert result["completed"] is True
    assert original_history == history_copy
    assert RAW_A_NEEDLE in repr(original_history)
    assert RAW_A_NEEDLE not in _payload_text(captured)
