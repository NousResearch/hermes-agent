from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


OLD_TASK_A_ID = "phase4-task-a"
NEW_TASK_B_ID = "phase4-task-b"
OLD_TASK_A_NEEDLE = "OLD_TASK_A_PHASE4_REGISTRY_NEEDLE"
CURRENT_TASK_B_MARKER = "CURRENT_TASK_B_PHASE4_REGISTRY_MARKER"
COMPLETED_TASK_STATE_ID = "phase4-completed-task"


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, reasoning=None, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _make_agent(
    tmp_path,
    *,
    registry_enabled: bool,
    wcp_enabled: bool = True,
    intake_enabled: bool = False,
) -> AIAgent:
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
    assert agent.client is not None
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

    assert agent.client is not None
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


def _install_fake_registry(monkeypatch, *, task_id: str = NEW_TASK_B_ID, closed: bool = False):
    calls: list[dict[str, object]] = []
    close_events: list[dict[str, object]] = []
    module = types.ModuleType("agent.task_registry")

    def resolve_task_for_turn(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            action="new_task",
            effective_task_id=task_id,
            reason="no_clear_continuation_evidence",
            registry_snapshot={
                "active_task_id": task_id,
                "tasks": {
                    task_id: {"task_id": task_id, "status": "active"},
                    OLD_TASK_A_ID: {"task_id": OLD_TASK_A_ID, "status": "closed" if closed else "active"},
                },
            },
            linked_task_id=None,
            hold_response=None,
        )

    def record_completed_workspec_state(**kwargs):
        close_events.append(kwargs)
        return {"task_id": kwargs.get("task_id"), "status": "closed"}

    setattr(module, "resolve_task_for_turn", resolve_task_for_turn)
    setattr(module, "record_completed_workspec_state", record_completed_workspec_state)
    monkeypatch.setitem(sys.modules, "agent.task_registry", module)
    return calls, close_events


def _install_failing_registry(monkeypatch, reason: str = "resolver failed token=raw-secret"):
    calls: list[dict[str, object]] = []
    module = types.ModuleType("agent.task_registry")

    def resolve_task_for_turn(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(reason)

    setattr(module, "resolve_task_for_turn", resolve_task_for_turn)
    monkeypatch.setitem(sys.modules, "agent.task_registry", module)
    return calls


def test_registry_disabled_preserves_existing_task_id_pass_through(tmp_path):
    agent = _make_agent(tmp_path, registry_enabled=False, wcp_enabled=False, intake_enabled=False)
    captured = _capture_provider_kwargs(agent)

    result = agent.run_conversation(
        "short task while registry disabled",
        conversation_history=[{"role": "user", "content": OLD_TASK_A_NEEDLE}],
        task_id="explicit-existing-task",
    )

    payload = _payload_text(captured)
    assert result["completed"] is True
    assert captured["provider_called"] is True
    assert getattr(agent, "_current_task_id") == "explicit-existing-task"
    assert OLD_TASK_A_NEEDLE in payload
    assert not (tmp_path / "intake").exists()


def test_registry_enabled_resolves_b_task_id_before_phase2_intake(tmp_path, monkeypatch):
    calls, _close_events = _install_fake_registry(monkeypatch, task_id=NEW_TASK_B_ID)
    agent = _make_agent(tmp_path, registry_enabled=True, wcp_enabled=True, intake_enabled=True)
    _capture_provider_kwargs(agent)
    long_task_b_prompt = (
        f"{CURRENT_TASK_B_MARKER} start independent B work\n"
        "line two with requirements\n"
        "line three with checklist\n"
        "line four with enough text to trigger pre-model intake"
    )

    agent.run_conversation(
        long_task_b_prompt,
        conversation_history=[{"role": "assistant", "content": f"old task A {OLD_TASK_A_NEEDLE}"}],
    )

    assert calls, "Phase 4 registry hook was not called before Phase 2 intake"
    intake_paths = [str(path) for path in (tmp_path / "intake").rglob("summary.md")]
    assert intake_paths, "Phase 2 intake did not create a summary path"
    assert any(NEW_TASK_B_ID in path for path in intake_paths), intake_paths
    assert getattr(agent, "_current_task_id") == NEW_TASK_B_ID


def test_closed_a_then_b_provider_payload_uses_registry_task_b(tmp_path, monkeypatch):
    calls, _close_events = _install_fake_registry(monkeypatch, task_id=NEW_TASK_B_ID, closed=True)
    agent = _make_agent(tmp_path, registry_enabled=True, wcp_enabled=True, intake_enabled=False)
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation(
        f"new independent B request {CURRENT_TASK_B_MARKER}",
        conversation_history=[
            {"role": "user", "content": f"old task A raw {OLD_TASK_A_NEEDLE}"},
            {"role": "assistant", "content": "old task A done"},
        ],
    )

    payload = _payload_text(captured)
    assert calls, "Phase 4 registry hook was not called, so WCP cannot receive registry task B"
    assert NEW_TASK_B_ID in payload
    assert OLD_TASK_A_NEEDLE not in payload


def test_completed_workspec_task_state_records_registry_close_event(tmp_path, monkeypatch):
    _calls, close_events = _install_fake_registry(monkeypatch, task_id=COMPLETED_TASK_STATE_ID)
    task_state = tmp_path / "task-state.json"
    task_state.write_text(
        '{"schema":"workspec_task_state_v1","status":"completed","tasks":[{"id":"t1","status":"completed"}]}',
        encoding="utf-8",
    )
    agent = _make_agent(tmp_path, registry_enabled=True, wcp_enabled=False, intake_enabled=False)
    setattr(agent, "_context_health_active_task_state_path", str(task_state))
    _capture_provider_kwargs(agent)

    agent.run_conversation(
        "finalize completed WorkSpec task",
        conversation_history=[],
        task_id=COMPLETED_TASK_STATE_ID,
    )

    assert close_events, "completed task-state was not reflected into the durable task registry"
    assert close_events[-1]["task_id"] == COMPLETED_TASK_STATE_ID
    assert close_events[-1]["task_state_path"] == str(task_state)


def test_registry_enabled_resolver_exception_returns_safe_hold_without_provider_or_persist(tmp_path, monkeypatch):
    calls = _install_failing_registry(monkeypatch)
    agent = _make_agent(tmp_path, registry_enabled=True, wcp_enabled=False, intake_enabled=False)
    captured = _capture_provider_kwargs(agent)
    persisted: list[object] = []
    agent._persist_session = lambda *a, **k: persisted.append((a, k))
    raw_prompt = "new B request with token=RAW_PHASE4_SECRET and password=RAW_PHASE4_PASSWORD"

    result = agent.run_conversation(
        raw_prompt,
        conversation_history=[{"role": "user", "content": OLD_TASK_A_NEEDLE}],
    )

    result_text = repr(result)
    assert calls, "registry resolver should be reached when registry is enabled"
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert captured["provider_called"] is False
    assert "context_health_hold" in result.get("error", "")
    assert "RAW_PHASE4_SECRET" not in result_text
    assert "RAW_PHASE4_PASSWORD" not in result_text
    assert "token=" not in result_text.lower()
    assert "password=" not in result_text.lower()
    assert raw_prompt not in repr(persisted)
    assert getattr(agent, "_current_task_id", None) in {None, ""}


def test_registry_enabled_corrupt_registry_json_returns_safe_hold_without_provider(tmp_path):
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    (registry_root / "registry.json").write_text("{not valid json", encoding="utf-8")
    agent = _make_agent(tmp_path, registry_enabled=True, wcp_enabled=False, intake_enabled=False)
    setattr(agent, "_context_health_task_registry_dir", registry_root)
    captured = _capture_provider_kwargs(agent)

    result = agent.run_conversation(
        "new B request after corrupt registry",
        conversation_history=[{"role": "assistant", "content": OLD_TASK_A_NEEDLE}],
    )

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert captured["provider_called"] is False
    assert "context_health_hold" in result.get("error", "")
    assert getattr(agent, "_context_health_task_registry_decision", None) is None
