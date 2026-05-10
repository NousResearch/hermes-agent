import asyncio
import hashlib
import hmac
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.linear import LinearAdapter
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import build_session_key
from toolsets import get_toolset


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _make_adapter(**extra):
    config = PlatformConfig(enabled=True, extra={
        "client_id": "linear-client",
        "client_secret": "linear-secret",
        "webhook_secret": "whsec",
        "public_base_url": "https://jaxmind.xyz",
        **extra,
    })
    return LinearAdapter(config)


def test_apply_env_overrides_configures_linear(monkeypatch):
    config = GatewayConfig()
    monkeypatch.setenv("LINEAR_ENABLED", "true")
    monkeypatch.setenv("LINEAR_CLIENT_ID", "cid")
    monkeypatch.setenv("LINEAR_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "whsec")
    monkeypatch.setenv("LINEAR_PUBLIC_BASE_URL", "https://linear.example.com/")
    monkeypatch.setenv("LINEAR_HOST", "0.0.0.0")
    monkeypatch.setenv("LINEAR_PORT", "9001")
    monkeypatch.setenv("LINEAR_SCOPES", "read,comments:create,app:mentionable")
    monkeypatch.setenv("LINEAR_MAX_CONCURRENT_SESSIONS", "5")
    monkeypatch.setenv("LINEAR_DEFAULT_EXECUTION_MODE", "human_gate")
    monkeypatch.setenv("LINEAR_PROJECT_EXECUTION_MODES", '{"Jax Control Plane":"autonomous_with_testing"}')
    monkeypatch.setenv("LINEAR_SUPPORTED_TASK_TYPES", "engineering,ops,research")

    _apply_env_overrides(config)

    assert Platform.LINEAR in config.platforms
    linear = config.platforms[Platform.LINEAR]
    assert linear.enabled is True
    assert linear.extra["client_id"] == "cid"
    assert linear.extra["client_secret"] == "csecret"
    assert linear.extra["webhook_secret"] == "whsec"
    assert linear.extra["public_base_url"] == "https://linear.example.com"
    assert linear.extra["host"] == "0.0.0.0"
    assert linear.extra["port"] == 9001
    assert linear.extra["scopes"] == ["read", "comments:create", "app:mentionable"]
    assert linear.extra["max_concurrent_sessions"] == 5
    assert linear.extra["default_execution_mode"] == "human_gate"
    assert linear.extra["project_execution_modes"] == {"Jax Control Plane": "autonomous_with_testing"}
    assert linear.extra["supported_task_types"] == ["engineering", "ops", "research"]


def test_get_connected_platforms_includes_linear_with_required_credentials():
    config = GatewayConfig(platforms={
        Platform.LINEAR: PlatformConfig(
            enabled=True,
            extra={
                "client_id": "cid",
                "client_secret": "secret",
                "webhook_secret": "whsec",
            },
        )
    })

    assert Platform.LINEAR in config.get_connected_platforms()


def test_linear_authorize_url_uses_actor_app_and_scope_list():
    adapter = _make_adapter(scopes=["read", "write", "app:mentionable"])

    url = adapter._build_authorize_url("state-123")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "linear.app"
    assert parsed.path == "/oauth/authorize"
    assert params["client_id"] == ["linear-client"]
    assert params["redirect_uri"] == ["https://jaxmind.xyz/linear/oauth/callback"]
    assert params["actor"] == ["app"]
    assert params["state"] == ["state-123"]
    assert params["scope"] == ["read,write,app:mentionable"]


def test_linear_default_scopes_include_write_for_agent_activity_mutations():
    adapter = _make_adapter()

    assert adapter._scopes == ["read", "write", "app:mentionable", "app:assignable"]


def test_validate_signature_accepts_linear_hmac():
    adapter = _make_adapter()
    body = b'{"type":"AgentSessionEvent"}'
    sig = hmac.new(b"whsec", body, hashlib.sha256).hexdigest()

    assert adapter._validate_signature(body, sig) is True
    assert adapter._validate_signature(body, "bad") is False


@pytest.mark.asyncio
async def test_callback_rejects_expired_oauth_state():
    adapter = _make_adapter()
    adapter._save_json(adapter._states_path, {"expired-state": {"created_at": 0}})

    request = SimpleNamespace(query={"code": "code-123", "state": "expired-state", "error": ""})

    response = await adapter._handle_callback(request)

    assert response.status == 400
    assert response.text == "Invalid or expired OAuth state.\n"
    assert adapter._load_json(adapter._states_path) == {}


def test_linear_uses_canonical_oauth_token_store_filename():
    adapter = _make_adapter()

    assert adapter._tokens_path.name == "linear_oauth_tokens.json"


@pytest.mark.asyncio
async def test_ensure_access_token_clears_revoked_refresh_token(monkeypatch):
    adapter = _make_adapter()
    adapter._save_json(adapter._tokens_path, {
        "app-user-1": {
            "access_token": "expired-token",
            "refresh_token": "revoked-refresh",
            "expires_at": time.time() - 10,
            "viewer_name": "jax",
        }
    })

    async def _fake_refresh_token(app_user_id, refresh_token):
        assert app_user_id == "app-user-1"
        assert refresh_token == "revoked-refresh"
        raise RuntimeError('HTTP 400: {"error":"invalid_grant","error_description":"Refresh token revoked"}')

    monkeypatch.setattr(adapter, "_refresh_token", _fake_refresh_token)

    with pytest.raises(RuntimeError, match="re-authorize"):
        await adapter._ensure_access_token("app-user-1")

    assert adapter._load_json(adapter._tokens_path) == {}


def test_build_prompt_for_created_event_uses_prompt_context_and_flow_metadata():
    adapter = _make_adapter(project_execution_modes={"Jax Control Plane": "autonomous_with_testing"})
    payload = {
        "action": "created",
        "promptContext": "<issue>Investigate regression</issue>",
        "guidance": [{"rule": "stay concise"}],
        "agentSession": {
            "id": "session-123",
            "url": "https://linear.app/session/123",
            "issue": {
                "id": "issue-1",
                "identifier": "PAB-80",
                "title": "Linear agent",
                "project": {"id": "proj-1", "name": "Jax Control Plane"},
                "labels": {"nodes": [{"name": "type:ops"}]},
            },
        },
        "appUserId": "app-user-1",
    }

    adapter._store_session_metadata(payload)
    prompt = adapter._build_prompt(payload)

    assert "promptContext" in prompt
    assert "PAB-80" in prompt
    assert "Investigate regression" in prompt
    assert "Task type: ops" in prompt
    assert "Project execution mode: autonomous_with_testing" in prompt
    assert "```hermes_workflow" in prompt
    assert 'Allowed decisions: done, ready_for_testing' in prompt
    assert 'choose `ready_for_testing` only when actual testing is still pending' in prompt



def test_extract_workflow_decision_from_response_strips_machine_block():
    adapter = _make_adapter()
    response = (
        "Implemented the first slice.\n\n"
        "```hermes_workflow\n"
        '{"decision": "stay_in_progress", "reason": "partial implementation shipped"}\n'
        "```"
    )

    cleaned, decision = adapter._extract_workflow_decision_from_response(response)

    assert cleaned == "Implemented the first slice."
    assert decision == {"decision": "stay_in_progress", "reason": "partial implementation shipped"}


def test_extract_workflow_decision_from_response_ignores_non_terminal_block():
    adapter = _make_adapter()
    response = (
        "Implemented the first slice.\n\n"
        "```hermes_workflow\n"
        '{"decision": "stay_in_progress", "reason": "partial implementation shipped"}\n'
        "```\n\n"
        "Extra trailing text that should prevent parsing."
    )

    cleaned, decision = adapter._extract_workflow_decision_from_response(response)

    assert cleaned == response
    assert decision is None


def test_resolve_success_workflow_decision_normalizes_ready_for_testing_to_done_for_autonomous_dev():
    adapter = _make_adapter()

    decision = adapter._resolve_success_workflow_decision({
        "execution_mode": "autonomous_dev",
        "workflow_decision": "ready_for_testing",
    })

    assert decision == "done"


def test_resolve_success_workflow_decision_normalizes_done_to_review_for_human_gate():
    adapter = _make_adapter()

    decision = adapter._resolve_success_workflow_decision({
        "execution_mode": "human_gate",
        "workflow_decision": "done",
    })

    assert decision == "needs_human_review"


def test_resolve_success_workflow_decision_keeps_done_for_autonomous_with_testing():
    adapter = _make_adapter()

    decision = adapter._resolve_success_workflow_decision({
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "done",
    })

    assert decision == "done"


def test_resolve_success_workflow_decision_defaults_to_done_for_autonomous_with_testing_when_missing():
    adapter = _make_adapter()

    decision = adapter._resolve_success_workflow_decision({
        "execution_mode": "autonomous_with_testing",
    })

    assert decision == "done"


def test_apply_agent_result_metadata_updates_session_and_cleans_response():
    adapter = _make_adapter()
    adapter._session_info["linear:session-1"] = {"issue_identifier": "PAB-80"}
    agent_result = {
        "final_response": (
            "Narrowed the task to the first executable step.\n\n"
            "```hermes_workflow\n"
            '{"decision": "change_scope", "reason": "first slice only"}\n'
            "```"
        ),
        "messages": [
            {"role": "assistant", "content": (
                "Narrowed the task to the first executable step.\n\n"
                "```hermes_workflow\n"
                '{"decision": "change_scope", "reason": "first slice only"}\n'
                "```"
            )}
        ],
    }

    adapter.apply_agent_result_metadata("linear:session-1", agent_result)

    assert agent_result["final_response"] == "Narrowed the task to the first executable step."
    assert agent_result["messages"][-1]["content"] == "Narrowed the task to the first executable step."
    assert adapter._session_info["linear:session-1"]["workflow_decision"] == "change_scope"
    assert adapter._session_info["linear:session-1"]["workflow_decision_reason"] == "first slice only"


def test_apply_agent_result_metadata_clears_stale_workflow_decision_when_block_missing():
    adapter = _make_adapter()
    adapter._session_info["linear:session-1"] = {
        "issue_identifier": "PAB-80",
        "workflow_decision": "change_scope",
        "workflow_decision_reason": "stale",
    }
    agent_result = {"final_response": "Implemented more of the task without an explicit workflow block."}

    adapter.apply_agent_result_metadata("linear:session-1", agent_result)

    assert agent_result["final_response"] == "Implemented more of the task without an explicit workflow block."
    assert "workflow_decision" not in adapter._session_info["linear:session-1"]
    assert "workflow_decision_reason" not in adapter._session_info["linear:session-1"]


def test_store_session_metadata_captures_project_registry_pointers_from_prompt_context():
    adapter = _make_adapter(project_execution_modes={"Jax Control Plane": "autonomous_with_testing"})
    payload = {
        "action": "created",
        "promptContext": (
            '<issue identifier="PAB-143">'
            '<project name="Jax Control Plane">'
            'Internal Jax ops/control plane. '
            'Obsidian: /lab/obsidian_vault/Projects/Jax Control Plane/ | '
            'Repo: github.com/pablots99/jax-control-plane | '
            'Discord: #jax-control-plane (1493569165100056596)'
            '</project>'
            '</issue>'
        ),
        "agentSession": {
            "id": "session-123",
            "url": "https://linear.app/session/123",
            "issue": {
                "id": "issue-1",
                "identifier": "PAB-143",
                "title": "Capture metadata",
                "project": {"id": "proj-1", "name": "Jax Control Plane"},
                "labels": {"nodes": [{"name": "type:engineering"}]},
            },
        },
        "appUserId": "app-user-1",
    }

    session = adapter._store_session_metadata(payload)

    assert session["project_key"] == "Jax_Control_Plane"
    assert session["obsidian_path"] == "/lab/obsidian_vault/Projects/Jax Control Plane/"
    assert session["repo_url"] == "https://github.com/pablots99/jax-control-plane"
    assert session["discord_channel_name"] == "jax-control-plane"
    assert session["discord_channel_id"] == "1493569165100056596"


@pytest.mark.asyncio
async def test_send_posts_response_activity_via_session_mapping(monkeypatch):
    adapter = _make_adapter()
    adapter._session_info["linear:session-1"] = {
        "agent_session_id": "session-1",
        "app_user_id": "app-user-1",
        "chat_name": "PAB-80",
    }

    captured = {}

    async def _fake_create_activity(**kwargs):
        captured.update(kwargs)
        return {"agentActivityCreate": {"success": True, "agentActivity": {"id": "activity-1"}}}

    monkeypatch.setattr(adapter, "_create_activity", _fake_create_activity)

    result = await adapter.send("linear:session-1", "Done.")

    assert result.success is True
    assert result.message_id == "activity-1"
    assert captured["app_user_id"] == "app-user-1"
    assert captured["agent_session_id"] == "session-1"
    assert captured["activity_type"] == "response"
    assert captured["body"] == "Done."


@pytest.mark.asyncio
async def test_send_posts_comment_when_session_has_no_agent_session(monkeypatch):
    adapter = _make_adapter()
    adapter._session_info["linear:issue:issue-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "app_user_id": "app-user-1",
        "chat_name": "PAB-80",
    }
    captured = {}

    async def _fake_create_comment(*, app_user_id, issue_id, body):
        captured.update({"app_user_id": app_user_id, "issue_id": issue_id, "body": body})
        return {"commentCreate": {"success": True, "comment": {"id": "comment-1"}}}

    monkeypatch.setattr(adapter, "_create_issue_comment", _fake_create_comment)

    result = await adapter.send("linear:issue:issue-1", "Done.")

    assert result.success is True
    assert result.message_id == "comment-1"
    assert captured == {
        "app_user_id": "app-user-1",
        "issue_id": "issue-1",
        "body": "Done.",
    }


@pytest.mark.asyncio
async def test_list_delegated_started_issues_paginates(monkeypatch):
    adapter = _make_adapter()
    payloads = []

    async def _fake_ensure_access_token(app_user_id):
        assert app_user_id == "app-user-1"
        return "token-1"

    responses = iter([
        {"data": {"issues": {"nodes": [{"id": "issue-1"}], "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"}}}},
        {"data": {"issues": {"nodes": [{"id": "issue-2"}], "pageInfo": {"hasNextPage": False, "endCursor": None}}}},
    ])

    def _fake_http_json(url, payload, headers):
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(adapter, "_ensure_access_token", _fake_ensure_access_token)
    monkeypatch.setattr(adapter, "_http_json", _fake_http_json)

    issues = await adapter._list_delegated_started_issues("app-user-1")

    assert [issue["id"] for issue in issues] == ["issue-1", "issue-2"]
    assert payloads[1]["variables"]["after"] == "cursor-1"


@pytest.mark.asyncio
async def test_transition_issue_for_session_posts_comment_when_agent_session_missing(monkeypatch):
    adapter = _make_adapter()
    session = {
        "issue_id": "issue-1",
        "team_id": "team-1",
        "app_user_id": "app-user-1",
    }
    transitions = []
    comments = []

    async def _fake_update_issue_state(issue_id, team_id, app_user_id, state_name, *, assignee_id=None):
        transitions.append((issue_id, team_id, app_user_id, state_name, assignee_id))

    async def _fake_issue_comment(*, app_user_id, issue_id, body):
        comments.append((app_user_id, issue_id, body))
        return {"commentCreate": {"success": True, "comment": {"id": "comment-1"}}}

    monkeypatch.setattr(adapter, "_update_issue_state", _fake_update_issue_state)
    monkeypatch.setattr(adapter, "_create_issue_comment", _fake_issue_comment)

    await adapter._transition_issue_for_session(session, "In Progress", assignee_id="user-1", comment="hello")

    assert transitions == [("issue-1", "team-1", "app-user-1", "In Progress", "user-1")]
    assert comments == [("app-user-1", "issue-1", "hello")]


@pytest.mark.asyncio
async def test_reconcile_delegated_started_issues_enqueues_runnable_issue(monkeypatch):
    adapter = _make_adapter()
    adapter._save_json(adapter._tokens_path, {
        "app-user-1": {"access_token": "token-1", "refresh_token": "refresh-1"}
    })
    dispatched = []

    async def _fake_list(app_user_id):
        assert app_user_id == "app-user-1"
        return [{
            "id": "issue-1",
            "identifier": "PAB-80",
            "title": "Finish direct Linear integration",
            "description": "Close the execution gap for delegated started issues.",
            "updatedAt": "2026-04-17T18:00:00.000Z",
            "state": {"name": "In Progress", "type": "started"},
            "team": {"id": "team-1", "name": "Pablo"},
            "project": {"id": "project-1", "name": "Jax Control Plane"},
            "assignee": {"id": "user-1", "name": "pablo Torres"},
            "labels": {"nodes": [{"name": "type:engineering"}]},
        }]

    async def _fake_handle_message(event):
        dispatched.append(event)

    monkeypatch.setattr(adapter, "_list_delegated_started_issues", _fake_list)
    monkeypatch.setattr(adapter, "handle_message", _fake_handle_message)

    await adapter._reconcile_delegated_started_issues_once()

    assert len(dispatched) == 1
    event = dispatched[0]
    assert event.source.chat_id == "linear:issue:issue-1"
    assert event.message_type == MessageType.TEXT
    assert "Finish direct Linear integration" in event.text
    session = adapter._session_info["linear:issue:issue-1"]
    assert session["context_type"] == "linear_issue_reconcile"
    assert session["issue_identifier"] == "PAB-80"
    assert session["can_execute"] is True


@pytest.mark.asyncio
async def test_reconcile_delegated_started_issues_skips_active_reconciled_session(monkeypatch):
    adapter = _make_adapter()
    adapter._save_json(adapter._tokens_path, {
        "app-user-1": {"access_token": "token-1", "refresh_token": "refresh-1"}
    })
    source = adapter.build_source(
        chat_id="linear:issue:issue-1",
        chat_name="PAB-80",
        chat_type="thread",
        user_id="linear:reconcile:app-user-1",
        user_name="Linear reconciler",
    )
    adapter._active_sessions[build_session_key(source)] = asyncio.Event()
    dispatched = []

    async def _fake_list(_app_user_id):
        return [{
            "id": "issue-1",
            "identifier": "PAB-80",
            "title": "Finish direct Linear integration",
            "description": "",
            "updatedAt": "2026-04-17T18:00:00.000Z",
            "state": {"name": "In Progress", "type": "started"},
            "team": {"id": "team-1", "name": "Pablo"},
            "project": {"id": "project-1", "name": "Jax Control Plane"},
            "assignee": {"id": "user-1", "name": "pablo Torres"},
            "labels": {"nodes": [{"name": "type:engineering"}]},
        }]

    async def _fake_handle_message(event):
        dispatched.append(event)

    monkeypatch.setattr(adapter, "_list_delegated_started_issues", _fake_list)
    monkeypatch.setattr(adapter, "handle_message", _fake_handle_message)

    await adapter._reconcile_delegated_started_issues_once()

    assert dispatched == []


@pytest.mark.asyncio
async def test_reconcile_delegated_started_issues_skips_testing_state(monkeypatch):
    adapter = _make_adapter(testing_state_name="Testing")
    adapter._save_json(adapter._tokens_path, {
        "app-user-1": {"access_token": "token-1", "refresh_token": "refresh-1"}
    })
    dispatched = []

    async def _fake_list(_app_user_id):
        return [{
            "id": "issue-1",
            "identifier": "PAB-80",
            "title": "Ready for testing",
            "description": "",
            "updatedAt": "2026-04-17T18:00:00.000Z",
            "state": {"name": "Testing", "type": "started"},
            "team": {"id": "team-1", "name": "Pablo"},
            "project": {"id": "project-1", "name": "Jax Control Plane"},
            "assignee": {"id": "user-1", "name": "pablo Torres"},
            "labels": {"nodes": [{"name": "type:engineering"}]},
        }]

    async def _fake_handle_message(event):
        dispatched.append(event)

    monkeypatch.setattr(adapter, "_list_delegated_started_issues", _fake_list)
    monkeypatch.setattr(adapter, "handle_message", _fake_handle_message)

    await adapter._reconcile_delegated_started_issues_once()

    assert dispatched == []


@pytest.mark.asyncio
async def test_reconcile_delegated_started_issues_skips_recently_leased_unchanged_issue(monkeypatch):
    adapter = _make_adapter()
    adapter._save_json(adapter._tokens_path, {
        "app-user-1": {"access_token": "token-1", "refresh_token": "refresh-1"}
    })
    adapter._reconciled_issue_leases["issue-1"] = {
        "updated_at": "2026-04-17T18:00:00.000Z",
        "leased_at": time.time(),
    }
    dispatched = []

    async def _fake_list(_app_user_id):
        return [{
            "id": "issue-1",
            "identifier": "PAB-80",
            "title": "Finish direct Linear integration",
            "description": "",
            "updatedAt": "2026-04-17T18:00:00.000Z",
            "state": {"name": "In Progress", "type": "started"},
            "team": {"id": "team-1", "name": "Pablo"},
            "project": {"id": "project-1", "name": "Jax Control Plane"},
            "assignee": {"id": "user-1", "name": "pablo Torres"},
            "labels": {"nodes": [{"name": "type:engineering"}]},
        }]

    async def _fake_handle_message(event):
        dispatched.append(event)

    monkeypatch.setattr(adapter, "_list_delegated_started_issues", _fake_list)
    monkeypatch.setattr(adapter, "handle_message", _fake_handle_message)

    await adapter._reconcile_delegated_started_issues_once()

    assert dispatched == []


@pytest.mark.asyncio
async def test_linear_processing_respects_max_concurrent_sessions(monkeypatch):
    adapter = _make_adapter(max_concurrent_sessions=1)
    adapter._session_info.update({
        "linear:session-1": {"agent_session_id": "session-1", "app_user_id": "app-user-1", "can_execute": True},
        "linear:session-2": {"agent_session_id": "session-2", "app_user_id": "app-user-2", "can_execute": True},
    })

    started_first = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    queue_updates = []
    running = 0
    max_running = 0

    async def _fake_super(self, event, session_key):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        if session_key == "session-1":
            started_first.set()
            await release_first.wait()
        else:
            second_started.set()
        running -= 1

    async def _fake_queue_activity(chat_id, body):
        queue_updates.append((chat_id, body))

    monkeypatch.setattr("gateway.platforms.base.BasePlatformAdapter._process_message_background", _fake_super)
    monkeypatch.setattr(adapter, "_maybe_send_queue_activity", _fake_queue_activity)

    event1 = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    event2 = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-2"))

    task1 = asyncio.create_task(adapter._process_message_background(event1, "session-1"))
    await started_first.wait()
    task2 = asyncio.create_task(adapter._process_message_background(event2, "session-2"))
    await asyncio.sleep(0.05)

    assert second_started.is_set() is False

    release_first.set()
    await asyncio.gather(task1, task2)

    assert second_started.is_set() is True
    assert max_running == 1
    assert queue_updates == [
        (
            "linear:session-2",
            "Jax queued this session and will pick it up once one of the 1 active slots frees up.",
        ),
        ("linear:session-2", "Jax is starting work on this session now."),
    ]


@pytest.mark.asyncio
async def test_dependency_block_prevents_processing_and_moves_issue_to_blocked(monkeypatch):
    adapter = _make_adapter()
    adapter._session_info["linear:session-1"] = {
        "agent_session_id": "session-1",
        "app_user_id": "app-user-1",
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "creator_id": "user-1",
        "creator_name": "Pablo",
        "current_assignee_id": "user-1",
        "can_execute": True,
    }
    transitions = []
    activities = []
    super_mock = AsyncMock()

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_activity(chat_id, body):
        activities.append((chat_id, body))

    async def _fake_unresolved_dependencies(session):
        return [{"identifier": "PAB-79", "state": "In Progress"}]

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_maybe_send_queue_activity", _fake_activity)
    monkeypatch.setattr(adapter, "_list_unresolved_dependencies", _fake_unresolved_dependencies)
    monkeypatch.setattr("gateway.platforms.base.BasePlatformAdapter._process_message_background", super_mock)

    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    await adapter._process_message_background(event, "session-1")

    assert transitions == [(
        "Blocked",
        "user-1",
        "Jax did not start because this issue depends on unresolved work: PAB-79 (In Progress).",
    )]
    assert activities == [(
        "linear:session-1",
        "Jax did not start because this issue depends on unresolved work: PAB-79 (In Progress).",
    )]
    super_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_executable_task_is_blocked_and_reassigned(monkeypatch):
    adapter = _make_adapter(supported_task_types=["engineering", "ops"])
    adapter._session_info["linear:session-1"] = {
        "agent_session_id": "session-1",
        "app_user_id": "app-user-1",
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "marketing",
        "execution_mode": "autonomous_with_testing",
        "creator_id": "user-1",
        "creator_name": "Pablo",
        "current_assignee_id": None,
        "current_assignee_name": None,
        "can_execute": False,
        "block_reason": "Task type 'marketing' is not executable by the current Jax executor.",
    }
    blocked = []
    activities = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        blocked.append((target_state, assignee_id, comment))

    async def _fake_activity(chat_id, body):
        activities.append((chat_id, body))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_maybe_send_queue_activity", _fake_activity)
    monkeypatch.setattr("gateway.platforms.base.BasePlatformAdapter._process_message_background", AsyncMock())

    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    await adapter._process_message_background(event, "session-1")

    assert blocked == [(
        "Blocked",
        "user-1",
        "Jax cannot execute this issue automatically: Task type 'marketing' is not executable by the current Jax executor.",
    )]
    assert activities == [(
        "linear:session-1",
        "Jax cannot execute this task automatically and moved it to Blocked for Pablo. Reason: Task type 'marketing' is not executable by the current Jax executor.",
    )]


@pytest.mark.asyncio
async def test_processing_start_moves_issue_to_in_progress(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_start(event)

    assert transitions == [(
        "In Progress",
        "user-1",
        "Jax started working on this issue automatically (task type: engineering, mode: autonomous_with_testing).",
    )]


@pytest.mark.asyncio
async def test_processing_complete_moves_autonomous_with_testing_issue_to_done_when_testing_not_needed(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "done",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_state_name_exists", AsyncMock(return_value=True))

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "Done",
        "user-1",
        "Jax finished implementation work and marked this issue Done automatically.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_backlog_decision_moves_issue_to_backlog(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "backlog",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "Backlog",
        "user-1",
        "Jax decided this issue should return to Backlog before more autonomous execution.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_stay_in_progress_decision_keeps_issue_active(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"), message_id="orig-1")
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "stay_in_progress",
        "workflow_decision_reason": "first slice landed but remaining implementation work is still concrete",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []
    reruns = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_schedule(event_obj, session, *, rerun_kind, reason):
        reruns.append((rerun_kind, reason, session["issue_identifier"]))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_schedule_autonomous_rerun", _fake_schedule)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert reruns == [(
        "continue",
        "workflow_decision:stay_in_progress:first slice landed but remaining implementation work is still concrete",
        "PAB-80",
    )]
    assert transitions == [(
        "In Progress",
        "user-1",
        "Jax made partial progress and is keeping this issue in In Progress for continued autonomous execution.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_stay_in_progress_blocks_when_success_rerun_budget_exhausted(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"), message_id="orig-1")
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "stay_in_progress",
        "workflow_decision_reason": "still not done",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "success_rerun_count": 3,
        "max_success_reruns": 3,
    }
    transitions = []
    reruns = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_schedule(event_obj, session, *, rerun_kind, reason):
        reruns.append((rerun_kind, reason, session["issue_identifier"]))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_schedule_autonomous_rerun", _fake_schedule)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert reruns == []
    assert transitions == [(
        "Blocked",
        "user-1",
        "Jax exhausted autonomous continuation attempts for this issue and moved it to Blocked for follow-up.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_change_scope_decision_retriggers_success_path(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"), message_id="orig-1")
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "issue_title": "Ship direct Linear integration",
        "team_id": "team-1",
        "project_id": "project-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "change_scope",
        "workflow_decision_reason": "split webhook hardening from OAuth install verification",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "app_user_id": "app-user-1",
    }
    transitions = []
    reruns = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_schedule(event_obj, session, *, rerun_kind, reason):
        reruns.append((rerun_kind, reason, session["issue_identifier"]))

    async def _fake_ensure_followup(session):
        session["scope_followup_issue_identifier"] = "PAB-81"
        return {"id": "issue-81", "identifier": "PAB-81", "title": "Ship direct Linear integration — follow-up slice"}

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_schedule_autonomous_rerun", _fake_schedule)
    monkeypatch.setattr(adapter, "_ensure_scope_followup_issue", _fake_ensure_followup)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert reruns == [("refit", "workflow_decision:change_scope", "PAB-80")]
    assert transitions == [(
        "In Progress",
        "user-1",
        "Jax narrowed the active work on this issue into a smaller executable slice and will continue autonomously. Created follow-up issue PAB-81 to track the deferred remaining scope.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_change_scope_continues_when_followup_creation_fails(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"), message_id="orig-1")
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "issue_title": "Ship direct Linear integration",
        "team_id": "team-1",
        "project_id": "project-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "change_scope",
        "workflow_decision_reason": "split webhook hardening from OAuth install verification",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "app_user_id": "app-user-1",
    }
    transitions = []
    reruns = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_schedule(event_obj, session, *, rerun_kind, reason):
        reruns.append((rerun_kind, reason, session["issue_identifier"]))

    async def _fake_ensure_followup(_session):
        raise RuntimeError("linear issue create failed")

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_schedule_autonomous_rerun", _fake_schedule)
    monkeypatch.setattr(adapter, "_ensure_scope_followup_issue", _fake_ensure_followup)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert reruns == [("refit", "workflow_decision:change_scope", "PAB-80")]
    assert transitions == [(
        "In Progress",
        "user-1",
        "Jax narrowed the active work on this issue into a smaller executable slice and will continue autonomously. Jax could not create the deferred-scope follow-up issue automatically.",
    )]


@pytest.mark.asyncio
async def test_ensure_scope_followup_issue_deduplicates_existing_followup(monkeypatch):
    adapter = _make_adapter()
    session = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "issue_title": "Ship direct Linear integration",
        "team_id": "team-1",
        "project_id": "project-1",
        "workflow_decision_reason": "split webhook hardening from OAuth install verification",
        "current_assignee_id": "user-1",
        "app_user_id": "app-user-1",
        "scope_followup_issue_reason": "split webhook hardening from OAuth install verification",
        "scope_followup_issue_id": "issue-81",
        "scope_followup_issue_identifier": "PAB-81",
        "scope_followup_issue_title": "Ship direct Linear integration — follow-up slice",
    }
    created = []

    async def _fake_create_followup_issue(*args, **kwargs):
        created.append((args, kwargs))
        return {"id": "issue-82", "identifier": "PAB-82", "title": "unexpected"}

    monkeypatch.setattr(adapter, "_create_followup_issue", _fake_create_followup_issue)

    followup = await adapter._ensure_scope_followup_issue(session)

    assert created == []
    assert followup == {
        "id": "issue-81",
        "identifier": "PAB-81",
        "title": "Ship direct Linear integration — follow-up slice",
    }


@pytest.mark.asyncio
async def test_processing_complete_needs_human_review_decision_moves_issue_to_in_review(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "workflow_decision": "needs_human_review",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "In Review",
        "user-1",
        "Jax needs human review before continuing this issue.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_defaults_autonomous_with_testing_issue_to_done_when_no_testing_decision(monkeypatch):
    adapter = _make_adapter(testing_state_name="Testing", testing_fallback_state_name="In Review")
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    monkeypatch.setattr(adapter, "_state_name_exists", AsyncMock(side_effect=lambda _team_id, name, _app_user_id=None: name != "Testing"))
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "Done",
        "user-1",
        "Jax finished implementation work and marked this issue Done automatically.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_moves_human_gate_issue_to_in_review(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "human_gate",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "In Review",
        "user-1",
        "Jax needs human review before continuing this issue.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_moves_autonomous_dev_issue_to_done(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_dev",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "Done",
        "user-1",
        "Jax finished implementation work and marked this issue Done automatically.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_retryable_failure_leaves_issue_in_progress_for_retry(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "retry_policy": "standard",
        "retry_attempt_count": 0,
        "last_error_class": "transient_api",
    }
    transitions = []
    reruns = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_schedule(event_obj, session, *, rerun_kind, reason):
        reruns.append((rerun_kind, reason, session["issue_identifier"]))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_schedule_autonomous_rerun", _fake_schedule)

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert adapter._session_info["linear:session-1"]["retry_attempt_count"] == 1
    assert adapter._session_info["linear:session-1"]["retry_requested"] is True
    assert reruns == [("retry", "transient_api", "PAB-80")]
    assert transitions == [(
        "In Progress",
        "user-1",
        "Jax hit a transient_api failure and left this issue in In Progress for retry (attempt 1/2).",
    )]


@pytest.mark.asyncio
async def test_processing_complete_exhausted_retryable_failure_moves_issue_to_blocked(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "retry_policy": "standard",
        "retry_attempt_count": 2,
        "last_error_class": "transient_api",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert adapter._session_info["linear:session-1"]["retry_requested"] is False
    assert transitions == [(
        "Blocked",
        "user-1",
        "Jax exhausted retry attempts after a transient_api failure and moved this issue to Blocked for follow-up.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_autonomy_first_error_refits_scope_and_retriggers(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "autonomy_first": True,
        "refit_attempt_count": 0,
        "last_error_class": "agent_execution_error",
    }
    transitions = []
    reruns = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    async def _fake_schedule(event_obj, session, *, rerun_kind, reason):
        reruns.append((rerun_kind, reason, session["issue_identifier"]))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)
    monkeypatch.setattr(adapter, "_schedule_autonomous_rerun", _fake_schedule)

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert adapter._session_info["linear:session-1"]["refit_attempt_count"] == 1
    assert adapter._session_info["linear:session-1"]["retrigger_requested"] is True
    assert adapter._session_info["linear:session-1"]["workflow_decision"] == "change_scope"
    assert reruns == [("refit", "agent_execution_error", "PAB-80")]
    assert transitions == [(
        "In Progress",
        "user-1",
        "Jax hit an agent_execution_error failure, narrowed the scope into a smaller executable slice, and left this issue in In Progress for an autonomous rerun (refit 1/1).",
    )]


@pytest.mark.asyncio
async def test_processing_complete_exhausted_refit_moves_issue_to_blocked(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
        "autonomy_first": True,
        "refit_attempt_count": 1,
        "last_error_class": "agent_execution_error",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert adapter._session_info["linear:session-1"]["retrigger_requested"] is False
    assert transitions == [(
        "Blocked",
        "user-1",
        "Jax could not retrofit this issue into a smaller executable slice after an agent_execution_error failure and moved it to Blocked for follow-up.",
    )]


@pytest.mark.asyncio
async def test_processing_complete_moves_failed_issue_to_blocked(monkeypatch):
    adapter = _make_adapter()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="linear:session-1"))
    adapter._session_info["linear:session-1"] = {
        "issue_id": "issue-1",
        "issue_identifier": "PAB-80",
        "team_id": "team-1",
        "task_type": "engineering",
        "execution_mode": "autonomous_with_testing",
        "can_execute": True,
        "current_assignee_id": "user-1",
    }
    transitions = []

    async def _fake_transition(session, target_state, *, assignee_id=None, comment=None):
        transitions.append((target_state, assignee_id, comment))

    monkeypatch.setattr(adapter, "_transition_issue_for_session", _fake_transition)

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert transitions == [(
        "Blocked",
        "user-1",
        "Jax could not finish this issue and moved it to Blocked for follow-up.",
    )]


@pytest.mark.asyncio
async def test_schedule_autonomous_rerun_dispatches_internal_followup_event(monkeypatch):
    adapter = _make_adapter()
    source = adapter.build_source(
        chat_id="linear:session-1",
        chat_name="PAB-80",
        chat_type="thread",
        user_id="user-1",
        user_name="Pablo",
    )
    event = SimpleNamespace(source=source, message_id="orig-1")
    session = {"issue_identifier": "PAB-80"}
    handled = []

    async def _fake_handle_message(rerun_event):
        handled.append(rerun_event)

    monkeypatch.setattr(adapter, "handle_message", _fake_handle_message)

    await adapter._schedule_autonomous_rerun(event, session, rerun_kind="retry", reason="transient_api")
    await asyncio.sleep(0)

    assert len(handled) == 1
    assert handled[0].internal is True
    assert handled[0].source is source
    assert handled[0].message_id.startswith("orig-1:rerun:retry:")
    assert "Retry the current issue autonomously" in handled[0].text
    assert "Previous workflow reason: transient_api" in handled[0].text


@pytest.mark.asyncio
async def test_schedule_autonomous_rerun_does_not_override_existing_pending_message(monkeypatch):
    adapter = _make_adapter()
    source = adapter.build_source(
        chat_id="linear:session-1",
        chat_name="PAB-80",
        chat_type="thread",
        user_id="user-1",
        user_name="Pablo",
    )
    event = SimpleNamespace(source=source, message_id="orig-1")
    session = {"issue_identifier": "PAB-80"}
    session_key = build_session_key(source)
    adapter._pending_messages[session_key] = MessageEvent(
        text="human follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="human-1",
    )
    handled = []

    async def _fake_handle_message(rerun_event):
        handled.append(rerun_event)

    monkeypatch.setattr(adapter, "handle_message", _fake_handle_message)

    await adapter._schedule_autonomous_rerun(event, session, rerun_kind="retry", reason="transient_api")
    await asyncio.sleep(0)

    assert handled == []
    assert adapter._pending_messages[session_key].message_id == "human-1"


@pytest.mark.asyncio
async def test_connect_acquires_platform_lock(monkeypatch):
    adapter = _make_adapter(port=8647)
    calls = []

    monkeypatch.setattr(
        adapter,
        "_acquire_platform_lock",
        lambda scope, identity, resource: calls.append((scope, identity, resource)) or True,
    )
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: calls.append(("release", None, None)))
    monkeypatch.setattr("gateway.platforms.linear._socket.socket", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unused")))

    runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())
    site = SimpleNamespace(start=AsyncMock())
    monkeypatch.setattr("gateway.platforms.linear.web.AppRunner", lambda app: runner)
    monkeypatch.setattr("gateway.platforms.linear.web.TCPSite", lambda _runner, _host, _port: site)

    connected = await adapter.connect()

    assert connected is True
    assert calls == [(
        "linear_app",
        "linear-client",
        "Linear app credentials",
    )]
    runner.setup.assert_awaited_once()
    site.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_releases_platform_lock(monkeypatch):
    adapter = _make_adapter()
    cleanup = AsyncMock()
    adapter._runner = SimpleNamespace(cleanup=cleanup)
    adapter._site = object()
    released = []

    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: released.append(True))

    await adapter.disconnect()

    cleanup.assert_awaited_once()
    assert released == [True]
    assert adapter._runner is None
    assert adapter._site is None


@pytest.mark.asyncio
async def test_connect_releases_platform_lock_when_port_is_busy(monkeypatch):
    adapter = _make_adapter()
    events = []

    monkeypatch.setattr(
        adapter,
        "_acquire_platform_lock",
        lambda scope, identity, resource: events.append((scope, identity, resource)) or True,
    )
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: events.append(("release", None, None)))

    class _BusySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, _timeout):
            return None

        def connect(self, _addr):
            return None

    monkeypatch.setattr("gateway.platforms.linear._socket.socket", lambda *args, **kwargs: _BusySocket())

    connected = await adapter.connect()

    assert connected is False
    assert events == [
        ("linear_app", "linear-client", "Linear app credentials"),
        ("release", None, None),
    ]


def test_platform_registries_include_linear():
    from hermes_cli.platforms import PLATFORMS as SHARED_PLATFORMS
    from hermes_cli.gateway import _PLATFORMS as GATEWAY_PLATFORMS
    from hermes_cli.gateway import _all_platforms as setup_platforms

    assert "linear" in SHARED_PLATFORMS
    assert SHARED_PLATFORMS["linear"].default_toolset == "hermes-linear"
    assert get_toolset("hermes-linear") is not None
    assert any(p["key"] == "linear" for p in GATEWAY_PLATFORMS)
    assert any(p["key"] == "linear" for p in setup_platforms())
