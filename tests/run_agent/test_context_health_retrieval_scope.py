from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent
from tools.session_search_tool import session_search


TASK_A_ID = "phase6-task-a"
TASK_B_ID = "phase6-task-b"
TASK_A_SESSION_ID = "session-phase6-a"
TASK_B_SESSION_ID = "session-phase6-b"
TASK_A_ANCHOR_ID = 61001
TASK_A_NEEDLE = "PHASE6_CLOSED_TASK_A_RETRIEVAL_NEEDLE"
MEMORY_A_NEEDLE = "PHASE6_MEMORY_PREFETCH_CLOSED_A_NEEDLE"
PLUGIN_A_NEEDLE = "PHASE6_PLUGIN_CONTEXT_CLOSED_A_NEEDLE"


class FakeSessionDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.schema_fingerprint = "messages(id,session_id,role,content,timestamp)"
        self.sessions = {
            TASK_A_SESSION_ID: {
                "id": TASK_A_SESSION_ID,
                "title": "closed task A",
                "source": "cli",
                "started_at": 1,
                "model": "test-model",
            },
            TASK_B_SESSION_ID: {
                "id": TASK_B_SESSION_ID,
                "title": "active task B",
                "source": "cli",
                "started_at": 2,
                "model": "test-model",
            },
        }
        self.messages = {
            TASK_A_SESSION_ID: [
                {"id": TASK_A_ANCHOR_ID, "session_id": TASK_A_SESSION_ID, "role": "user", "content": TASK_A_NEEDLE, "timestamp": 1},
                {"id": TASK_A_ANCHOR_ID + 1, "session_id": TASK_A_SESSION_ID, "role": "assistant", "content": "closed A answer", "timestamp": 2},
            ],
            TASK_B_SESSION_ID: [
                {"id": 62001, "session_id": TASK_B_SESSION_ID, "role": "user", "content": "active B", "timestamp": 3},
            ],
        }

    def get_session(self, session_id: str):
        self.calls.append(("get_session", session_id))
        return self.sessions.get(session_id)

    def search_messages(self, **kwargs):
        self.calls.append(("search_messages", dict(kwargs)))
        return [
            {
                "id": TASK_A_ANCHOR_ID,
                "session_id": TASK_A_SESSION_ID,
                "role": "user",
                "snippet": TASK_A_NEEDLE,
                "source": "cli",
                "model": "test-model",
                "session_started": 1,
            }
        ]

    def get_anchored_view(self, session_id: str, msg_id: int, window: int = 5, bookend: int = 3):
        self.calls.append(("get_anchored_view", {"session_id": session_id, "msg_id": msg_id}))
        rows = self.messages.get(session_id, [])
        return {
            "window": rows,
            "bookend_start": rows[:bookend],
            "bookend_end": rows[-bookend:],
            "messages_before": 0,
            "messages_after": 0,
        }

    def get_messages(self, session_id: str):
        self.calls.append(("get_messages", session_id))
        return list(self.messages.get(session_id, []))

    def get_messages_around(self, session_id: str, around_message_id: int, window: int = 5):
        self.calls.append(("get_messages_around", {"session_id": session_id, "around_message_id": around_message_id}))
        return {
            "window": list(self.messages.get(session_id, [])),
            "messages_before": 0,
            "messages_after": 0,
        }

    def list_sessions_rich(self, **kwargs):
        self.calls.append(("list_sessions_rich", dict(kwargs)))
        return list(self.sessions.values())


def _json(text: str):
    return json.loads(text)


def _closed_a_registry_snapshot():
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": TASK_B_ID,
        "tasks": {
            TASK_A_ID: {"task_id": TASK_A_ID, "status": "closed", "session_id": TASK_A_SESSION_ID, "latest_turn_id": str(TASK_A_ANCHOR_ID)},
            TASK_B_ID: {"task_id": TASK_B_ID, "status": "active", "session_id": TASK_B_SESSION_ID},
        },
    }


def _enabled_retrieval_scope(*, linked: bool = False, ambiguous: bool = False):
    allowed_task_ids = [TASK_B_ID, TASK_A_ID] if linked else [TASK_B_ID]
    allowed_session_ids = [TASK_B_SESSION_ID, TASK_A_SESSION_ID] if linked else [TASK_B_SESSION_ID]
    excluded_task_ids = [] if linked else [TASK_A_ID]
    excluded_session_ids = [] if linked else [TASK_A_SESSION_ID]
    return {
        "enabled": True,
        "mode": "explicit_link" if linked else "current_task_only",
        "relation": "ambiguous" if ambiguous else ("explicit_link" if linked else "new_independent_task"),
        "current_task_id": TASK_B_ID,
        "allowed_task_ids": allowed_task_ids,
        "excluded_task_ids": excluded_task_ids,
        "allowed_session_ids": allowed_session_ids,
        "excluded_session_ids": excluded_session_ids,
        "linked_task_ids": [TASK_A_ID] if linked else [],
        "registry_snapshot": _closed_a_registry_snapshot(),
    }


def test_session_search_discovery_with_enabled_scope_excludes_closed_a_for_independent_b():
    db = FakeSessionDB()

    result = _json(
        session_search(
            query=TASK_A_NEEDLE,
            db=db,
            current_session_id=TASK_B_SESSION_ID,
            retrieval_scope=_enabled_retrieval_scope(),
        )
    )

    assert result.get("success") is True or result.get("mode") in {"hold", "blocked"}
    assert TASK_A_NEEDLE not in repr(result)
    assert result.get("results") in ([], None) or result.get("mode") in {"hold", "blocked"}
    assert not any(call[0] == "search_messages" for call in db.calls)
    assert not any(call[0] == "get_anchored_view" for call in db.calls)


def test_session_search_read_with_enabled_scope_blocks_closed_a_when_b_is_independent():
    db = FakeSessionDB()

    result = _json(
        session_search(
            session_id=TASK_A_SESSION_ID,
            db=db,
            current_session_id=TASK_B_SESSION_ID,
            retrieval_scope=_enabled_retrieval_scope(),
        )
    )

    assert result.get("success") is False or result.get("mode") in {"hold", "blocked"}
    assert TASK_A_NEEDLE not in repr(result)


def test_session_search_scroll_with_enabled_scope_blocks_closed_a_when_b_is_independent():
    db = FakeSessionDB()

    result = _json(
        session_search(
            session_id=TASK_A_SESSION_ID,
            around_message_id=TASK_A_ANCHOR_ID,
            db=db,
            current_session_id=TASK_B_SESSION_ID,
            retrieval_scope=_enabled_retrieval_scope(),
        )
    )

    assert result.get("success") is False or result.get("mode") in {"hold", "blocked"}
    assert TASK_A_NEEDLE not in repr(result)


def test_explicit_continuation_reads_linked_a_only_when_scope_allows_it():
    db = FakeSessionDB()
    scope = _enabled_retrieval_scope(linked=True)

    result = _json(
        session_search(
            session_id=TASK_A_SESSION_ID,
            db=db,
            current_session_id=TASK_B_SESSION_ID,
            retrieval_scope=scope,
        )
    )

    assert TASK_A_ID in scope["linked_task_ids"]
    assert TASK_A_SESSION_ID in scope["allowed_session_ids"]
    assert result.get("success") is True
    assert result.get("mode") == "read"
    assert result.get("session_id") == TASK_A_SESSION_ID
    assert len(result.get("messages", [])) <= 30


def test_ambiguous_previous_reference_with_enabled_scope_blocks_before_search():
    db = FakeSessionDB()

    result = _json(
        session_search(
            query="use the previous one",
            db=db,
            current_session_id=TASK_B_SESSION_ID,
            retrieval_scope=_enabled_retrieval_scope(ambiguous=True),
        )
    )

    assert result.get("success") is False or result.get("mode") in {"hold", "blocked"}
    assert not any(call[0] == "search_messages" for call in db.calls)
    assert TASK_A_NEEDLE not in repr(result)


def test_disabled_retrieval_scope_preserves_existing_session_search_pass_through():
    db = FakeSessionDB()

    result = _json(session_search(query=TASK_A_NEEDLE, db=db, current_session_id=TASK_B_SESSION_ID))

    assert result.get("success") is True
    assert TASK_A_NEEDLE in repr(result)
    assert any(call[0] == "search_messages" for call in db.calls)


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, reasoning=None, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _make_agent(tmp_path, *, retrieval_scope_enabled: bool = True) -> AIAgent:
    cfg = {
        "context_health": {
            "enabled": True,
            "runtime_behavior_enabled": True,
            "pre_model_intake": {"enabled": False},
            "task_boundary": {"enabled": True, "default_without_clear_continuation": "new_task", "ambiguous_action": "hold"},
            "task_registry": {"enabled": True},
            "working_context_packet": {"enabled": True},
            "task_boundary_firewall": {"enabled": True},
            "retrieval_scope": {"enabled": retrieval_scope_enabled},
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


def _install_registry(monkeypatch, *, linked: bool = False):
    module = types.ModuleType("agent.task_registry")

    def resolve_task_for_turn(**_kwargs):
        return SimpleNamespace(
            action="link_task" if linked else "new_task",
            effective_task_id=TASK_B_ID,
            reason="explicit_link" if linked else "no_clear_continuation_evidence",
            registry_snapshot=_closed_a_registry_snapshot(),
            linked_task_id=TASK_A_ID if linked else None,
            hold_response=None,
        )

    setattr(module, "resolve_task_for_turn", resolve_task_for_turn)
    setattr(module, "record_completed_workspec_state", lambda **kwargs: {"task_id": kwargs.get("task_id")})
    monkeypatch.setitem(sys.modules, "agent.task_registry", module)


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
    return repr(captured)


class ScopeCapableMemoryManager:
    supports_retrieval_scope = True

    def __init__(self) -> None:
        self.prefetch_calls: list[dict[str, object]] = []
        self.turn_start_calls: list[str] = []

    def on_turn_start(self, _turn_number, message, **_kwargs):
        self.turn_start_calls.append(str(message))

    def prefetch_all(self, query: str, **kwargs):
        self.prefetch_calls.append({"query": query, **kwargs})
        return f"memory returned closed A despite scope {MEMORY_A_NEEDLE}"


class ScopeUnsupportedMemoryManager:
    supports_retrieval_scope = False

    def __init__(self) -> None:
        self.prefetch_calls: list[str] = []
        self.turn_start_calls: list[str] = []

    def on_turn_start(self, _turn_number, message, **_kwargs):
        self.turn_start_calls.append(str(message))

    def prefetch_all(self, query: str):
        self.prefetch_calls.append(query)
        return f"unsupported memory returned closed A {MEMORY_A_NEEDLE}"


def test_scope_capable_memory_prefetch_receives_scope_and_excludes_closed_a_from_provider_payload(tmp_path, monkeypatch):
    _install_registry(monkeypatch)
    agent = _make_agent(tmp_path, retrieval_scope_enabled=True)
    fake_memory = ScopeCapableMemoryManager()
    agent._memory_manager = fake_memory
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation("start unrelated B task", conversation_history=[])

    assert len(fake_memory.prefetch_calls) == 1
    call = fake_memory.prefetch_calls[0]
    assert call["query"] == "start unrelated B task"
    assert call["current_task_id"] == TASK_B_ID
    assert TASK_B_ID in call["allowed_task_ids"]
    assert TASK_A_ID in call["excluded_task_ids"]
    assert TASK_A_SESSION_ID in call["excluded_session_ids"]
    assert MEMORY_A_NEEDLE not in _payload_text(captured)


def test_scope_unsupported_memory_prefetch_is_skipped_or_holds_in_enabled_mode(tmp_path, monkeypatch):
    _install_registry(monkeypatch)
    agent = _make_agent(tmp_path, retrieval_scope_enabled=True)
    fake_memory = ScopeUnsupportedMemoryManager()
    agent._memory_manager = fake_memory
    captured = _capture_provider_kwargs(agent)

    agent.run_conversation("start unrelated B task", conversation_history=[])

    assert fake_memory.prefetch_calls == [] or captured.get("provider_called") is False
    assert MEMORY_A_NEEDLE not in _payload_text(captured)


def test_plugin_pre_llm_call_receives_scope_and_closed_a_result_is_quarantined(tmp_path, monkeypatch):
    _install_registry(monkeypatch)
    agent = _make_agent(tmp_path, retrieval_scope_enabled=True)
    captured = _capture_provider_kwargs(agent)
    hook_calls: list[dict[str, object]] = []

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            hook_calls.append({"name": name, **kwargs})
            return [{"context": f"plugin returned closed A {PLUGIN_A_NEEDLE}"}]
        return []

    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
        agent.run_conversation("start unrelated B task", conversation_history=[])

    assert len(hook_calls) == 1
    scope = hook_calls[0]["retrieval_scope"]
    assert scope["enabled"] is True
    assert scope["current_task_id"] == TASK_B_ID
    assert TASK_B_ID in scope["allowed_task_ids"]
    assert TASK_A_ID in scope["excluded_task_ids"]
    assert PLUGIN_A_NEEDLE not in _payload_text(captured)


def test_enabled_retrieval_scope_build_failure_fails_closed_before_unscoped_memory_plugin_or_provider(tmp_path, monkeypatch):
    _install_registry(monkeypatch)
    agent = _make_agent(tmp_path, retrieval_scope_enabled=True)
    fake_memory = ScopeUnsupportedMemoryManager()
    agent._memory_manager = fake_memory
    captured = _capture_provider_kwargs(agent)
    hook_calls: list[dict[str, object]] = []

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            hook_calls.append({"name": name, **kwargs})
            return [{"context": f"plugin returned closed A {PLUGIN_A_NEEDLE}"}]
        return []

    with (
        patch(
            "agent.retrieval_scope.build_retrieval_scope",
            side_effect=RuntimeError("synthetic retrieval scope failure with token password secret"),
        ),
        patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook),
    ):
        result = agent.run_conversation("start unrelated B task", conversation_history=[])

    assert captured.get("provider_called") is False
    assert fake_memory.prefetch_calls == []
    assert hook_calls == []
    assert getattr(agent, "_context_health_retrieval_scope", {}) != {"enabled": False}
    rendered = repr(result) + _payload_text(captured)
    assert MEMORY_A_NEEDLE not in rendered
    assert PLUGIN_A_NEEDLE not in rendered
    assert TASK_A_NEEDLE not in rendered
    assert "synthetic retrieval scope failure" not in rendered
    assert "token" not in rendered.lower()
    assert "password" not in rendered.lower()
    assert "secret" not in rendered.lower()


def test_disabled_retrieval_scope_build_failure_preserves_pass_through(tmp_path, monkeypatch):
    _install_registry(monkeypatch)
    agent = _make_agent(tmp_path, retrieval_scope_enabled=False)
    fake_memory = ScopeUnsupportedMemoryManager()
    agent._memory_manager = fake_memory
    captured = _capture_provider_kwargs(agent)
    hook_calls: list[dict[str, object]] = []

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            hook_calls.append({"name": name, **kwargs})
        return []

    with (
        patch(
            "agent.retrieval_scope.build_retrieval_scope",
            side_effect=RuntimeError("disabled synthetic retrieval scope failure"),
        ),
        patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook),
    ):
        agent.run_conversation("disabled scope legacy pass-through", conversation_history=[])

    assert captured.get("provider_called") is True
    assert fake_memory.prefetch_calls == ["disabled scope legacy pass-through"]
    assert len(hook_calls) == 1
    assert hook_calls[0]["retrieval_scope"] == {"enabled": False}
    assert getattr(agent, "_context_health_retrieval_scope", {}) == {"enabled": False}


def test_no_db_schema_or_transcript_mutation_during_phase6_scope_checks():
    db = FakeSessionDB()
    schema_before = db.schema_fingerprint
    messages_before = {sid: [dict(row) for row in rows] for sid, rows in db.messages.items()}

    session_search(query=TASK_A_NEEDLE, db=db, current_session_id=TASK_B_SESSION_ID)

    assert db.schema_fingerprint == schema_before
    assert db.messages == messages_before
    assert not any(call[0].startswith("write") or call[0].startswith("delete") for call in db.calls)
