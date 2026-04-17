import asyncio
import hashlib
import hmac
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.linear import LinearAdapter
from gateway.platforms.base import ProcessingOutcome
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
async def test_processing_complete_moves_autonomous_with_testing_issue_to_testing(monkeypatch):
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
    monkeypatch.setattr(adapter, "_state_name_exists", AsyncMock(return_value=True))

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert transitions == [(
        "Testing",
        "user-1",
        "Jax finished implementation work and moved this issue to Testing (task type: engineering, mode: autonomous_with_testing).",
    )]


@pytest.mark.asyncio
async def test_processing_complete_falls_back_to_in_review_when_testing_missing(monkeypatch):
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
        "In Review",
        "user-1",
        "Jax finished implementation work and moved this issue to In Review because the team has no Testing state configured.",
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
        "Jax finished implementation work and left this issue in In Review for human approval.",
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
    from hermes_cli.setup import _GATEWAY_PLATFORMS as SETUP_GATEWAY_PLATFORMS

    assert "linear" in SHARED_PLATFORMS
    assert SHARED_PLATFORMS["linear"].default_toolset == "hermes-linear"
    assert get_toolset("hermes-linear") is not None
    assert any(p["key"] == "linear" for p in GATEWAY_PLATFORMS)
    assert any(name == "Linear Agent Sessions" for name, _env, _func in SETUP_GATEWAY_PLATFORMS)
