from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _remote_entry(body: bytes):
    from tools.mcp_skills_protocol import SkillEntry
    uri = "skill://fixture/remote-demo/SKILL.md"
    return SkillEntry.model_validate({
        "uri": uri,
        "frontmatter": {"name": "remote-demo", "description": "Remote startup description"},
        "resources": [{"uri": uri, "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "size": len(body)}],
    })


@pytest.fixture()
def remote_context(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    local = home / "skills" / "general" / "remote-demo"
    local.mkdir(parents=True)
    workspace.mkdir()
    (local / "SKILL.md").write_text(
        "---\nname: remote-demo\ndescription: Local description wins bare identity\n---\n\n# Local\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(workspace)
    from tools import mcp_skills_cache as cache
    from tools import mcp_skills_registry as registry
    monkeypatch.setattr(cache, "_ensure_get_verified", lambda *_args, **_kwargs: None)
    registry.clear_runtime_state()
    body = b"---\nname: remote-demo\ndescription: Remote startup description\n---\n\n# Remote\n"
    entry = _remote_entry(body)
    registry.publish_live_catalog(home, "fixture", "config-a", [entry])
    yield home, body, entry
    registry.clear_runtime_state()


def test_native_list_keeps_local_and_remote_qualified(remote_context):
    _home, _body, entry = remote_context
    from tools.skills_tool import skills_list
    result = json.loads(skills_list(session_id="session-a"))
    rows = [row for row in result["skills"] if row["name"] == "remote-demo"]
    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"local", "mcp"}
    remote = next(row for row in rows if row["source"] == "mcp")
    assert remote["qualified_name"] == f"mcp:fixture:{entry.uri}"


def test_startup_prompt_has_remote_description_without_body_fetch_and_stays_pinned(remote_context, monkeypatch):
    home, _body, entry = remote_context
    from agent import system_prompt
    from tools.mcp_skills_registry import publish_live_catalog
    from tools.mcp_skills_protocol import SkillEntry

    agent = SimpleNamespace(
        valid_tool_names=["skill_view", "skills_list"], platform="cli", session_id="session-a",
        _hermes_home=home,
    )
    prompt = system_prompt._skills_prompt(agent)
    assert "Local description wins bare identity" in prompt
    assert f"mcp:fixture:{entry.uri}" in prompt
    assert "Remote startup description" in prompt

    changed_body = b"---\nname: changed\ndescription: Changed later\n---\n"
    changed = SkillEntry.model_validate({
        "uri": "skill://fixture/changed/SKILL.md",
        "frontmatter": {"name": "changed", "description": "Changed later"},
        "resources": [{"uri": "skill://fixture/changed/SKILL.md",
                       "digest": "sha256:" + hashlib.sha256(changed_body).hexdigest(),
                       "size": len(changed_body)}],
    })
    publish_live_catalog(home, "fixture", "config-a", [changed])
    assert system_prompt._skills_prompt(agent) == prompt
    assert "Changed later" not in prompt


def test_qualified_preload_routes_through_native_skill_view(remote_context, monkeypatch):
    _home, body, entry = remote_context
    from agent.skill_commands import build_preloaded_skills_prompt
    from tools import mcp_skills_cache as cache

    monkeypatch.setattr(cache, "_fetch", lambda _record, _resource, home=None: (body, "text/markdown", True))
    prompt, loaded, missing = build_preloaded_skills_prompt(
        [f"mcp:fixture:{entry.uri}"], task_id="session-a")
    assert missing == []
    assert loaded == ["remote-demo"]
    assert "REMOTE MCP SKILL" in prompt
    assert "origin" in prompt.lower()
    assert "/cache/mcp-skills/" not in prompt


def test_materialize_is_not_suppressed_by_repeat_view_dedup(remote_context, monkeypatch):
    _home, body, entry = remote_context
    from tools import mcp_skills_cache as cache
    from tools.skills_tool import _skill_view_with_bump
    from tools.skills_tool_dedup import reset_skill_view_dedup

    monkeypatch.setattr(cache, "_fetch", lambda _record, _resource, home=None: (body, "text/markdown", True))
    name = f"mcp:fixture:{entry.uri}"
    reset_skill_view_dedup()
    first = json.loads(_skill_view_with_bump({"name": name}, task_id="session-a", session_id="session-a"))
    second = json.loads(_skill_view_with_bump(
        {"name": name, "materialize": True, "destination": "copies/remote-SKILL.md"},
        task_id="session-a", session_id="session-a"))
    assert first["success"] is True
    assert second["materialization"]["status"] == "created"
    assert Path(second["materialization"]["path"]).read_bytes() == body


def test_skill_manage_refuses_remote_namespace_before_mutation():
    from tools.skill_manager_tool import skill_manage
    result = json.loads(skill_manage(action="delete", name="mcp:fixture:skill://x/demo/SKILL.md"))
    assert "immutable" in result["error"]


def test_real_compression_continuation_carries_remote_skill_security_state(remote_context, monkeypatch):
    home, _body, _entry = remote_context
    from agent.conversation_compression import _adopt_live_compression_child
    from tools.mcp_skills_consent import enforce_remote_skill_gate
    from tools.mcp_skills_registry import (
        active_skills, has_execution_consent, mark_active, pin_session,
        record_execution_consent,
    )
    import tools.approval_prompt as approval_prompt

    parent = "session-a"
    child = "session-compressed"
    record = pin_session(home, parent)[0]
    mark_active(record, home, parent)
    record_execution_consent(record, home, parent)

    class DB:
        def get_compression_tip(self, session_id):
            return child if session_id == parent else session_id

        def get_session(self, session_id):
            return {"ended_at": None, "system_prompt": "remote-demo remains in compressed context"}

        def get_messages_as_conversation(self, session_id):
            return [{"role": "system", "content": "remote-demo remains in compressed context"}]

    callbacks = []
    compressor = SimpleNamespace(on_session_start=lambda *args, **kwargs: callbacks.append((args, kwargs)))
    agent = SimpleNamespace(
        session_id=parent, _hermes_home=home, context_compressor=compressor,
        _memory_manager=None, platform="cli", _gateway_session_key="conversation",
    )
    snapshot = pin_session(home, parent)
    agent._mcp_skill_snapshot = snapshot

    recovered = _adopt_live_compression_child(agent, DB(), parent)
    assert recovered and agent.session_id == child
    assert "remote-demo" in agent._cached_system_prompt
    assert agent._mcp_skill_snapshot == snapshot
    assert [row["uri"] for row in active_skills(home, child)] == [record["uri"]]
    assert has_execution_consent(record, home, child)
    assert callbacks and callbacks[0][1]["old_session_id"] == parent

    # Consent retained only on the proven continuation; an unrelated session has no active marker.
    assert active_skills(home, "unrelated-new-session") == []
    assert not has_execution_consent(record, home, "unrelated-new-session")

    # Removing prior consent from this test's runtime makes every consequential route prove it
    # sees the continuation's active origin before dispatch.
    from tools import mcp_skills_registry as registry
    registry._consented.clear()
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *a, **k: "deny")
    calls = [
        ("terminal", {"command": "printf no"}),
        ("execute_code", {"code": "print('no')"}),
        ("browser_exec", {"code": "print('no')"}),
        ("process_manage", {"action": "write", "data": "no"}),
        ("delegate_task", {"action": "spawn", "goal": "no"}),
    ]
    assert all(enforce_remote_skill_gate(name, args, session_id=child) is not None for name, args in calls)


def test_compression_publish_carries_security_state_before_agent_switch(remote_context, monkeypatch):
    home, _body, _entry = remote_context
    from agent import conversation_compression as compression
    from tools.mcp_skills_registry import (
        active_skills, has_execution_consent, mark_active, pin_session, record_execution_consent,
    )

    parent = "publish-parent"
    record = pin_session(home, parent)[0]
    mark_active(record, home, parent)
    record_execution_consent(record, home, parent)

    class DB:
        def __init__(self):
            self.child = None
            self.prompt = None
        def get_session(self, _session_id):
            return {"ended_at": None}
        def get_active_message_watermark(self, _session_id):
            return None
        def get_session_title(self, _session_id):
            return None
        def publish_compression_child(self, **kwargs):
            # The DB publication makes the child discoverable to other
            # processes, so its security state must already exist now.
            assert agent.session_id == parent
            self.child = kwargs["child_session_id"]
            self.prompt = kwargs["system_prompt"]
            assert [row["uri"] for row in active_skills(home, self.child)] == [record["uri"]]
            assert has_execution_consent(record, home, self.child)

    db = DB()
    agent = SimpleNamespace(
        session_id=parent, _hermes_home=home, _session_db=db,
        _persist_user_message_idx=None, _flush_messages_to_session_db=lambda *_a, **_k: None,
        platform="cli", model="fixture-model", _session_init_model_config={},
        working_directory=str(home), _session_messages=[],
    )
    monkeypatch.setattr(compression, "_carry_session_state_to_child", lambda *_args: None)
    skill_prompt = "REMOTE MCP SKILL remote-demo remains in compressed context"
    compression._publish_rotated_compaction(
        agent, [], [{"role": "system", "content": skill_prompt}],
        new_system_prompt=skill_prompt,
        lease=cast(Any, SimpleNamespace(holder=None, ttl=60.0, watermark=None)),
        old_session_id=parent, compressed_user_turn_outcome="already_present")

    assert db.child and agent.session_id == db.child
    assert db.prompt == skill_prompt
    assert [row["uri"] for row in active_skills(home, db.child)] == [record["uri"]]
    assert has_execution_consent(record, home, db.child)


def test_runtime_dispatchers_deny_effects_after_real_compression_publish_and_adoption(remote_context, monkeypatch):
    home, _body, _entry = remote_context
    from agent import agent_runtime_helpers
    from agent import conversation_compression as compression
    from tools import mcp_skills_registry as registry
    from tools import mcp_tool
    import model_tools
    import tools.approval_prompt as approval_prompt
    import agent.inline_tool_executors as inline_tool_executors

    parent = "runtime-parent"
    record = registry.pin_session(home, parent)[0]
    registry.mark_active(record, home, parent)

    class PublishDB:
        def __init__(self):
            self.child = None
            self.prompt = None
        def get_session(self, _session_id):
            return {"ended_at": None}
        def get_active_message_watermark(self, _session_id):
            return None
        def get_session_title(self, _session_id):
            return None
        def publish_compression_child(self, **kwargs):
            self.child = kwargs["child_session_id"]
            self.prompt = kwargs["system_prompt"]

    published = PublishDB()
    publisher = SimpleNamespace(
        session_id=parent, _hermes_home=home, _session_db=published,
        _persist_user_message_idx=None, _flush_messages_to_session_db=lambda *_a, **_k: None,
        platform="cli", model="fixture-model", _session_init_model_config={},
        working_directory=str(home), _session_messages=[],
    )
    monkeypatch.setattr(compression, "_carry_session_state_to_child", lambda *_args: None)
    skill_prompt = "REMOTE MCP SKILL remote-demo remains in compressed context"
    compression._publish_rotated_compaction(
        publisher, [], [{"role": "system", "content": skill_prompt}],
        new_system_prompt=skill_prompt,
        lease=cast(Any, SimpleNamespace(holder=None, ttl=60.0, watermark=None)),
        old_session_id=parent, compressed_user_turn_outcome="already_present")
    child = published.child
    assert child

    class AdoptDB:
        def get_compression_tip(self, session_id):
            return child if session_id == parent else session_id
        def get_session(self, session_id):
            return {"ended_at": None, "system_prompt": skill_prompt} if session_id == child else None
        def get_messages_as_conversation(self, session_id):
            return [{"role": "system", "content": skill_prompt}]

    resumed = SimpleNamespace(
        session_id=parent, _hermes_home=home,
        context_compressor=SimpleNamespace(on_session_start=lambda *_a, **_k: None),
        _memory_manager=None, platform="cli", _gateway_session_key="conversation",
        _mcp_skill_snapshot=registry.pin_session(home, parent),
    )
    assert compression._adopt_live_compression_child(resumed, AdoptDB(), parent)
    assert resumed.session_id == child
    assert [row["uri"] for row in registry.active_skills(home, child)] == [record["uri"]]

    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *_a, **_k: "deny")
    effects = []
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **k: effects.append((a, k)) or "{}")
    # This generated MCP name is test-only metadata; the real model_tools dispatcher
    # and origin gate still run before the effect spy.
    cross_origin_name = "mcp__other__read_resource"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, cross_origin_name, "other")
    calls = [
        ("terminal", {"command": "printf no"}),
        ("execute_code", {"code": "print('no')"}),
        ("process_manage", {"action": "write", "session_id": "deferred", "data": "no"}),
        (cross_origin_name, {"uri": "resource://other/private"}),
    ]
    denied = [json.loads(model_tools.handle_function_call(
        name, args, task_id=child, session_id=child)) for name, args in calls]
    assert all("BLOCKED" in result["error"] for result in denied)
    assert effects == []

    delegated = []
    delegate_agent = SimpleNamespace(
        session_id=child, _memory_manager=None,
        _dispatch_delegate_task=lambda args: delegated.append(args) or "{}",
    )
    monkeypatch.setattr(inline_tool_executors, "emit_terminal_post_tool_call", lambda *_a, **_k: None)
    result = json.loads(agent_runtime_helpers.invoke_tool(
        delegate_agent, "delegate_task", {"action": "spawn", "goal": "no"}, child))
    assert "BLOCKED" in result["error"]
    assert delegated == []
