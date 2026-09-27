"""Real config/runtime/child-construction path; no external model calls."""

import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, Mock

import pytest

from agent.subagent_lifecycle import SubagentHandle, SubagentLaunchRequest, SubagentLifecycleError, SubagentLifecycleService, SubagentState
from agent import secret_scope
from hermes_constants import set_hermes_home_override, reset_hermes_home_override, get_hermes_home


def _profile(home: Path, provider: str, model: str, key: str):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        f"model:\n  provider: {provider}\n  default: {model}\n"
        + "delegation:\n  inherit_mcp_toolsets: true\n  orchestrator_enabled: true\n"
        + "  max_spawn_depth: 5\n  provider: openrouter\n  model: wrong/delegation\n"
        + "  fallback_providers:\n    - provider: openrouter\n      model: wrong/fallback\n", encoding="utf-8",
    )
    (home / ".env").write_text(key, encoding="utf-8")


def _parent():
    return SimpleNamespace(session_id="parent", model="active/alternative", provider="openrouter",
                           api_key="parent-secret", base_url="https://openrouter.ai/api/v1",
                           enabled_toolsets=["terminal", "delegation", "code_execution", "mcp-example"],
                           disabled_toolsets=[], _fallback_chain=[{"provider": "openrouter", "model": "parent/fallback"}],
                           _delegate_depth=0)


def test_private_default_route_profile_scope_and_no_observers(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from hermes_cli import lifecycle
    from hermes_state import SessionDB

    a, b = tmp_path / "a", tmp_path / "b"
    _profile(a, "deepseek", "a/default", "DEEPSEEK_API_KEY=key-a\n")
    _profile(b, "openrouter", "b/default", "OPENROUTER_API_KEY=key-b\n")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setattr("run_agent._hermes_home", a)
    db = SessionDB(db_path=a / "state.db")
    parent = _parent()
    parent._session_db = db
    parent._memory_manager = Mock()

    def rows():
        with sqlite3.connect(a / "state.db") as conn:
            return conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    before = rows()
    service = SubagentLifecycleService(lambda: parent)
    seen = []
    hooks = []
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *args, **kwargs: hooks.append(args))

    def run(child, *, user_message, **kwargs):
        seen.append((str(get_hermes_home()), child.provider, child.model, child.base_url,
                     child.api_key, child._fallback_chain, child.tools, child.valid_tool_names,
                     child._session_db, child._delegate_role, child._persist_disabled,
                     child._skip_mcp_refresh, child._credential_pool, user_message))
        return {"final_response": "done"}

    monkeypatch.setattr(AIAgent, "run_conversation", run)
    secret_scope.set_multiplex_active(True)
    try:
        first_handle = None
        for index, home in enumerate((a, b, a)):
            ht = set_hermes_home_override(str(home))
            st = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
            try:
                handle = service.launch(SubagentLaunchRequest(goal="sensitive excerpt", context="private context",
                                                                 role="leaf", private_default_route=True,
                                                                 correlation_id="same-id" if index < 2 else "next-id"))
                assert service.wait(handle, timeout_seconds=3).state == SubagentState.SUCCEEDED
                assert service.result(handle).summary == "done"
                assert handle.model == ("a/default" if home == a else "b/default")
                if index == 0:
                    first_handle = handle
                elif index == 1:
                    assert first_handle is not None
                    assert service.status(first_handle).state == SubagentState.UNKNOWN
                else:
                    assert first_handle is not None
                    assert service.status(first_handle).state == SubagentState.SUCCEEDED
            finally:
                secret_scope.reset_secret_scope(st)
                reset_hermes_home_override(ht)
    finally:
        secret_scope.set_multiplex_active(False)
        db.close()
    assert [entry[:5] for entry in seen] == [
        (str(a), "deepseek", "a/default", "https://api.deepseek.com/v1", "key-a"),
        (str(b), "openrouter", "b/default", "https://openrouter.ai/api/v1", "key-b"),
        (str(a), "deepseek", "a/default", "https://api.deepseek.com/v1", "key-a"),
    ]
    assert all(entry[5] == [] and entry[6] == [] and not entry[7] and entry[8] is None
               and entry[9] == "leaf" and entry[10] and entry[11] and entry[12] is None
               and entry[13] == "sensitive excerpt" for entry in seen)
    assert not hooks
    assert rows() == before
    assert parent._fallback_chain == [{"provider": "openrouter", "model": "parent/fallback"}]
    parent._memory_manager.on_delegation.assert_not_called()


@pytest.mark.parametrize("provider,key", [
    ("deepseek", ""),
    ("unregistered-provider", "OPENROUTER_API_KEY=not-the-requested-provider\n"),
])
def test_private_route_fails_before_child_or_submit(tmp_path, monkeypatch, provider, key):
    from agent import subagent_lifecycle as module
    from tools import delegate_tool

    home = tmp_path / "unavailable"
    _profile(home, provider, "a/default", key)
    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-profile-key-must-not-win")
    parent = _parent()
    service = SubagentLifecycleService(lambda: parent)
    built = []
    monkeypatch.setattr(delegate_tool, "_build_child_agent", lambda **kw: built.append(kw))
    monkeypatch.setattr(module._EXECUTOR, "submit", lambda *a, **kw: pytest.fail("submitted before preflight"))
    secret_scope.set_multiplex_active(True)
    ht = set_hermes_home_override(str(home))
    st = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        with pytest.raises(SubagentLifecycleError):
            service.launch(SubagentLaunchRequest(goal="secret", private_default_route=True))
    finally:
        secret_scope.reset_secret_scope(st)
        reset_hermes_home_override(ht)
        secret_scope.set_multiplex_active(False)
    assert not built


def test_private_route_rejects_overrides_before_construction():
    service = SubagentLifecycleService(_parent)
    for request in (
        SubagentLaunchRequest(goal="x", private_default_route=True, model="other"),
        SubagentLaunchRequest(goal="x", private_default_route=True, allowed_toolsets=("file",)),
        SubagentLaunchRequest(goal="x", private_default_route=True, role="orchestrator"),
    ):
        with pytest.raises(SubagentLifecycleError):
            service.launch(request)


def test_private_turn_has_no_hook_middleware_dump_or_session_row(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from hermes_cli import lifecycle, middleware
    from tools import delegate_tool

    home = tmp_path / "profile"
    _profile(home, "deepseek", "test/default", "DEEPSEEK_API_KEY=key-private\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setattr("run_agent._hermes_home", home)
    db = SessionDB(db_path=home / "state.db")
    parent = _parent()
    parent._session_db = db
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("private text reached an observer")
    monkeypatch.setattr(lifecycle, "invoke_hook", forbidden)
    monkeypatch.setattr(middleware, "apply_llm_request_middleware", forbidden)
    original = delegate_tool._build_child_agent
    outbound = []
    child_ids = []

    def build(**kwargs):
        child = original(**kwargs)
        child_ids.append(child.session_id)
        client = MagicMock()

        def reply(**payload):
            outbound.append(payload)
            message = SimpleNamespace(content="private answer", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                                   model=child.model, usage=None)
        client.chat.completions.create.side_effect = reply
        child.client = client
        return child
    monkeypatch.setattr(delegate_tool, "_build_child_agent", build)
    secret_scope.set_multiplex_active(True)
    ht = set_hermes_home_override(str(home))
    st = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        service = SubagentLifecycleService(lambda: parent)
        handle = service.launch(SubagentLaunchRequest(goal="sensitive excerpt", context="private context",
                                                       private_default_route=True))
        terminal = service.wait(handle, timeout_seconds=5)
        assert terminal.state == SubagentState.SUCCEEDED, service.result(handle)
        assert "private answer" in (service.result(handle).summary or "")
    finally:
        secret_scope.reset_secret_scope(st)
        reset_hermes_home_override(ht)
        secret_scope.set_multiplex_active(False)
        db.close()
    assert outbound and all(not payload.get("tools") for payload in outbound)
    assert "private context" in str(outbound[0])
    assert not calls
    with sqlite3.connect(home / "state.db") as conn:
        assert not conn.execute("SELECT id FROM sessions").fetchall()
    assert not list(home.glob("**/*request*.json"))
    for log in (home / "logs").glob("*.log"):
        text = log.read_text(errors="replace")
        assert "sensitive excerpt" not in text
        assert "private context" not in text
        assert all(sid not in text for sid in child_ids)


def test_private_wait_is_bounded_and_cancel_is_cooperative(tmp_path, monkeypatch):
    from run_agent import AIAgent

    home = tmp_path / "profile"
    _profile(home, "deepseek", "test/default", "DEEPSEEK_API_KEY=key-private\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("run_agent._hermes_home", home)
    entered = threading.Event()

    def run(child, **kwargs):
        entered.set()
        while not child._interrupt_requested:
            time.sleep(0.005)
        return {"final_response": "", "completed": False}
    monkeypatch.setattr(AIAgent, "run_conversation", run)
    secret_scope.set_multiplex_active(True)
    ht = set_hermes_home_override(str(home))
    st = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        service = SubagentLifecycleService(_parent)
        handle = service.launch(SubagentLaunchRequest(goal="x", private_default_route=True))
        assert entered.wait(2)
        assert service.wait(handle, timeout_seconds=0.01).timed_out
        assert service.cancel(handle, reason="bounded wait").accepted
        assert service.wait(handle, timeout_seconds=2).state == SubagentState.CANCELLED
    finally:
        secret_scope.reset_secret_scope(st)
        reset_hermes_home_override(ht)
        secret_scope.set_multiplex_active(False)


def test_private_cancellation_does_not_log_caller_reason(monkeypatch):
    from agent import subagent_lifecycle as module

    service = SubagentLifecycleService(_parent)
    record = SimpleNamespace(result=None, agent=object(), state=SubagentState.PENDING,
                             updated_at=0, profile_key="private-profile")
    monkeypatch.setattr(service, "_record", lambda _handle: record)
    reasons = []
    monkeypatch.setattr(module, "request_hard_interrupt",
                        lambda _agent, reason, **_kwargs: reasons.append(reason) or True)

    assert service.cancel(cast(SubagentHandle, object()), reason="sensitive-user-text").accepted
    assert reasons == ["Lifecycle cancellation requested: private child"]
