import asyncio
import hashlib
import hmac
import json
import base64
import threading
import time

import pytest

from tools.registry import registry as tool_registry
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.linear_agent.adapter import (
    LinearAgentAdapter,
    register,
    validate_config,
)
from plugins.platforms.linear_agent.client import LinearGraphQLClient
from plugins.platforms.linear_agent import oauth as linear_oauth
from plugins.platforms.linear_agent.oauth import (
    LinearOAuthConfig,
    LinearOAuthTokenManager,
    build_auth_token_update_callback,
    build_authorization_url,
    persist_auth_token,
    read_auth_token,
    read_env_file,
    remove_env_keys,
    update_env_file,
)
from plugins.platforms.linear_agent.registry import set_active_adapter
from plugins.platforms.linear_agent import tools as linear_agent_tools  # noqa: F401 - registers tools
from plugins.platforms.linear_agent.webhook import verify_linear_signature


class _FakeLinearClient:
    def __init__(self):
        self.calls = []
        self.thought_ephemeral_flags = []

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        self.calls.append(("thought", agent_session_id, body))
        self.thought_ephemeral_flags.append(ephemeral)
        return {"id": "thought-1"}

    async def create_response(self, agent_session_id, body, *, response_type="response", ephemeral=False, signal=None):
        # Keyed by activity type so elicitation is distinguishable from a
        # normal response; the default keeps existing tuple assertions valid.
        self.calls.append((response_type, agent_session_id, body))
        return {"id": "response-1"}

    async def create_error(self, agent_session_id, body):
        self.calls.append(("error", agent_session_id, body))
        return {"id": "error-1"}

    async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
        self.calls.append(("comment", issue_id, body, mutation_policy))
        return {"id": "comment-1"}


class _CaptureGraphQLClient(LinearGraphQLClient):
    def __init__(self):
        super().__init__("token")
        self.operations = []

    async def execute(self, query, variables=None):
        self.operations.append((query, variables or {}))
        return {
            "agentActivityCreate": {
                "success": True,
                "agentActivity": {"id": "activity-1"},
            }
        }

    async def create_response(self, agent_session_id, content, *, response_type="response", ephemeral=False, signal=None):
        # Record both the logical call and the exact variables shape the real
        # implementation sends, so existing assertions continue to pass.
        variables = {
            "input": {
                "agentSessionId": agent_session_id,
                "content": {
                    "type": response_type,
                    "body": content,
                },
            }
        }
        if ephemeral:
            variables["input"]["ephemeral"] = True
        if signal:
            variables["input"]["signal"] = signal
        self.operations.append((None, variables))  # shape expected by the test
        return {"id": "activity-1"}


class _FakeGraphQLResponse:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return json.dumps(self.payload)

    async def json(self):
        return self.payload


class _FakeGraphQLSession:
    def __init__(self, factory):
        self.factory = factory
        self.closed = False

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, data=None, headers=None):
        self.factory.calls.append({
            "url": url,
            "json": json,
            "data": dict(data or {}) if isinstance(data, dict) else data,
            "headers": dict(headers or {}),
        })
        return self.factory.responses.pop(0)


class _FakeGraphQLSessionFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.created = 0

    def __call__(self, **kwargs):
        self.created += 1
        return _FakeGraphQLSession(self)


class _FakeMutationClient:

    def __init__(self):
        self.calls = []

    async def update_issue(self, issue_id, input_payload, *, mutation_policy=None):
        self.calls.append(("update_issue", issue_id, input_payload, mutation_policy))
        return {"id": issue_id}

    async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
        self.calls.append(("create_comment", issue_id, body, mutation_policy))
        return {"id": "comment-1"}

    async def create_issue(self, input_payload, *, mutation_policy=None):
        self.calls.append(("create_issue", input_payload.get("teamId"), input_payload, mutation_policy))
        return {"id": "issue-1", "identifier": "PLAT-1"}

    async def create_project(self, input_payload, *, mutation_policy=None):
        self.calls.append(("create_project", input_payload, mutation_policy))
        return {"id": "proj-1", "name": input_payload.get("name", "New project")}

    async def update_project(self, project_id, input_payload, *, mutation_policy=None):
        self.calls.append(("update_project", project_id, input_payload, mutation_policy))
        return {"id": project_id, "name": "Project 1"}

    async def create_project_update(self, input_payload, *, mutation_policy=None):
        self.calls.append(("create_project_update", input_payload, mutation_policy))
        return {"id": "project-update-1", "project": {"id": input_payload.get("projectId"), "name": "Project 1"}}


class _ActiveMutationAdapter:
    def __init__(self):
        self._client = _FakeMutationClient()
        self._mutation_policy = {
            "update_issues": True,
            "create_comments": True,
            "create_issues": True,
            "update_projects": True,
        }


def _body(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signature(raw_body, secret):
    return "sha256=" + hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()


def _headers(raw_body, secret, delivery_id="evt-1"):
    return {
        "Linear-Signature": _signature(raw_body, secret),
        "Linear-Delivery-Id": delivery_id,
    }


def _created_payload(event_id="evt-1"):
    return {
        "action": "created",
        "id": event_id,
        "agentSession": {
            "id": "session-1",
            "workspaceId": "workspace-1",
            "issue": {
                "id": "issue-1",
                "identifier": "LIN-123",
                "title": "Fix the flaky job",
                "team": {"id": "team-1"},
            },
            "comment": {
                "id": "comment-1",
                "body": "@Hermes please investigate this failure.",
            },
        },
        "actor": {"id": "user-1", "name": "Ada"},
        "promptContext": "The issue has a failing CI check.",
        "guidance": "Prefer a concise answer.",
    }


def _prompted_payload(event_id="evt-2"):
    return {
        "action": "prompted",
        "id": event_id,
        "agentSession": {
            "id": "session-1",
            "workspaceId": "workspace-1",
            "issue": {
                "id": "issue-1",
                "identifier": "LIN-123",
                "title": "Fix the flaky job",
                "team": {"id": "team-1"},
            },
        },
        "agentActivity": {
            "id": "activity-user-1",
            "body": "Can you check the test log too?",
            "actor": {"id": "user-1", "name": "Ada"},
        },
    }


def _make_adapter(*, secret="secret", extra=None, client=None):
    merged_extra = {
        "access_token": "linear-token",
        "webhook_secret": secret,
        # Authorization is fail-closed (no allowlist → deny); tests simulate
        # authorized traffic unless they explicitly override this.
        "allow_all_users": True,
    }
    if extra:
        merged_extra.update(extra)
    return LinearAgentAdapter(
        PlatformConfig(enabled=True, extra=merged_extra),
        client=client or _FakeLinearClient(),
    )


def test_plugin_registration_works():
    class Ctx:
        def __init__(self):
            self.registered = None

        def register_platform(self, **kwargs):
            self.registered = kwargs

    ctx = Ctx()
    register(ctx)

    assert ctx.registered is not None
    assert ctx.registered["name"] == "linear_agent"
    assert ctx.registered["label"] == "Linear Agent"
    assert callable(ctx.registered["adapter_factory"])
    assert callable(ctx.registered["check_fn"])
    assert callable(ctx.registered["validate_config"])
    # Zero-core guard: every registration kwarg must be a PlatformEntry
    # field that exists at HEAD — register_platform forwards extras to the
    # dataclass constructor, so an unknown kwarg would TypeError at plugin
    # load. This catches any future kwarg that depends on unmerged core.
    from dataclasses import fields as dataclass_fields
    from gateway.platform_registry import PlatformEntry

    valid_fields = {f.name for f in dataclass_fields(PlatformEntry)}
    unknown = set(ctx.registered) - valid_fields
    assert not unknown, f"register() uses non-HEAD PlatformEntry fields: {sorted(unknown)}"


def test_linear_agent_mutation_tools_dispatch_positional_dicts():
    adapter = _ActiveMutationAdapter()
    set_active_adapter(adapter)  # type: ignore[arg-type]
    try:
        for name in (
            "linear_agent_update_issue",
            "linear_agent_create_comment",
            "linear_agent_create_issue",
            "linear_agent_create_project",
            "linear_agent_update_project",
            "linear_agent_create_project_update",
        ):
            entry = tool_registry.get_entry(name)
            assert entry is not None
            assert entry.is_async is True

        update_result = tool_registry.dispatch(
            "linear_agent_update_issue",
            {"issue_id": "PLAT-18557", "input": {"stateId": "state-1"}},
        )
        comment_result = tool_registry.dispatch(
            "linear_agent_create_comment",
            {"issue_id": "PLAT-18557", "body": "Moving to In Review for QA"},
        )
        create_result = tool_registry.dispatch(
            "linear_agent_create_issue",
            {"team_id": "12345678-1234-1234-1234-1234567890ab", "input": {"title": "New issue"}},
        )
        project_create = tool_registry.dispatch(
            "linear_agent_create_project",
            {"input": {"name": "Test Project", "teamIds": ["team-1"]}},
        )
        project_update = tool_registry.dispatch(
            "linear_agent_update_project",
            {"project_id": "proj-1", "input": {"state": "started"}},
        )
        project_status = tool_registry.dispatch(
            "linear_agent_create_project_update",
            {"input": {"projectId": "proj-1", "body": "On track", "health": "onTrack"}},
        )
    finally:
        set_active_adapter(None)

    assert "✅ Updated PLAT-18557" in update_result
    assert "✅ Comment added to PLAT-18557" in comment_result
    assert "✅ Created PLAT-1" in create_result
    assert "✅ Created project Test Project" in project_create
    assert "✅ Updated project proj-1" in project_update
    assert "✅ Created project update" in project_status
    assert adapter._client.calls == [
        ("update_issue", "PLAT-18557", {"stateId": "state-1"}, adapter._mutation_policy),
        ("create_comment", "PLAT-18557", "Moving to In Review for QA", adapter._mutation_policy),
        ("create_issue", "12345678-1234-1234-1234-1234567890ab", {"title": "New issue", "teamId": "12345678-1234-1234-1234-1234567890ab"}, adapter._mutation_policy),
        ("create_project", {"name": "Test Project", "teamIds": ["team-1"]}, adapter._mutation_policy),
        ("update_project", "proj-1", {"state": "started"}, adapter._mutation_policy),
        ("create_project_update", {"projectId": "proj-1", "body": "On track", "health": "onTrack"}, adapter._mutation_policy),
    ]


@pytest.mark.asyncio
async def test_missing_token_disables_adapter_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LINEAR_AGENT_ACCESS_TOKEN", raising=False)
    config = PlatformConfig(enabled=True, extra={})

    assert validate_config(config) is False

    adapter = LinearAgentAdapter(config, client=_FakeLinearClient())
    assert await adapter.connect() is False


def test_client_credentials_config_validates_without_access_token(monkeypatch):
    monkeypatch.delenv("LINEAR_AGENT_ACCESS_TOKEN", raising=False)
    config = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "client-1",
            "client_secret": "secret-1",
        },
    )

    assert validate_config(config) is True


def test_default_webhook_path_matches_linear_app_url():
    adapter = LinearAgentAdapter(
        PlatformConfig(enabled=True, extra={"access_token": "linear-token"}),
        client=_CaptureGraphQLClient(),
    )

    assert adapter._webhook_path == "/hermes/linear-agent"


def test_oauth_authorization_url_uses_app_actor_and_state():
    url = build_authorization_url(
        client_id="client-1",
        redirect_uri="http://localhost:8765/oauth/linear/callback",
        scope="read,write",
        state="state-1",
    )

    assert "https://linear.app/oauth/authorize?" in url
    assert "client_id=client-1" in url
    assert "scope=read%2Cwrite" in url
    assert "state=state-1" in url
    assert "actor=app" in url


def test_update_env_file_preserves_existing_values_and_quotes(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=value\nLINEAR_AGENT_ACCESS_TOKEN=old\n", encoding="utf-8")

    update_env_file(
        env_path,
        {
            "LINEAR_AGENT_ACCESS_TOKEN": "new-token",
            "LINEAR_AGENT_REFRESH_TOKEN": "refresh token with space",
        },
    )

    text = env_path.read_text(encoding="utf-8")
    assert "OTHER=value" in text
    assert "LINEAR_AGENT_ACCESS_TOKEN=new-token" in text
    assert 'LINEAR_AGENT_REFRESH_TOKEN="refresh token with space"' in text

    remove_env_keys(env_path, {"LINEAR_AGENT_ACCESS_TOKEN", "LINEAR_AGENT_REFRESH_TOKEN"})
    values = read_env_file(env_path)
    assert values == {"OTHER": "value"}


def test_read_env_file_handles_quoted_values_without_shelling_out(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LINEAR_AGENT_CLIENT_ID=client-1\n"
        "LINEAR_AGENT_CLIENT_SECRET=secret-1\n"
        'OTHER_TOOL_PATH="/Users/example/Library/Application Support/Tool/data.db"\n',
        encoding="utf-8",
    )

    values = read_env_file(env_path)

    assert values["LINEAR_AGENT_CLIENT_ID"] == "client-1"
    assert values["LINEAR_AGENT_CLIENT_SECRET"] == "secret-1"
    assert values["OTHER_TOOL_PATH"].endswith("Application Support/Tool/data.db")


def test_auth_json_token_round_trip(tmp_path):
    auth_path = tmp_path / "auth.json"

    persist_auth_token(
        auth_path,
        {
            "access_token": "access-1",
            "expires_at": 1234567890,
            "scope": "read,write",
        },
        client_id="client-1",
        token_url="https://api.linear.app/oauth/token",
    )

    state = read_auth_token(auth_path)
    assert state["access_token"] == "access-1"
    assert state["expires_at"] == 1234567890
    assert state["grant_type"] == "client_credentials"
    assert state["client_id"] == "client-1"


def test_auth_json_callback_persists_rotated_token(tmp_path):
    auth_path = tmp_path / "auth.json"
    callback = build_auth_token_update_callback(auth_path, client_id="client-1")

    callback({"access_token": "access-2", "expires_at": 1234567891})

    assert read_auth_token(auth_path)["access_token"] == "access-2"


def test_adapter_uses_cached_auth_json_token(tmp_path, monkeypatch):
    monkeypatch.setenv("LINEAR_AGENT_ACCESS_TOKEN", "legacy-env-token")
    auth_path = tmp_path / "auth.json"
    persist_auth_token(auth_path, {"access_token": "cached-token", "expires_at": 1234567890})
    config = PlatformConfig(enabled=True, extra={"auth_path": str(auth_path)})

    assert validate_config(config) is True
    adapter = LinearAgentAdapter(config, client=_FakeLinearClient())
    assert adapter._access_token == "cached-token"


def test_oauth_manager_persists_runtime_refresh_to_auth_json(tmp_path, monkeypatch):
    monkeypatch.delenv("LINEAR_AGENT_ACCESS_TOKEN", raising=False)
    auth_path = tmp_path / "auth.json"
    extra = {
        "client_id": "client-1",
        "client_secret": "secret-1",
        "auth_path": str(auth_path),
        "persist_tokens": True,
    }
    adapter = LinearAgentAdapter(PlatformConfig(enabled=True, extra=extra), client=_FakeLinearClient())

    assert adapter._oauth_manager is not None
    callback = adapter._oauth_manager.config.persist_callback
    assert callback is not None
    callback({
        "access_token": "runtime-token",
        "expires_at": 1234567892,
    })
    assert read_auth_token(auth_path)["access_token"] == "runtime-token"


def test_oauth_cli_reads_client_credentials_from_profile_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    update_env_file(
        env_path,
        {
            "LINEAR_AGENT_CLIENT_ID": "client-1",
            "LINEAR_AGENT_CLIENT_SECRET": "secret-1",
            "LINEAR_AGENT_OAUTH_SCOPES": "read,write,admin",
        },
    )
    captured = {}

    def fake_issue_client_credentials_token(**kwargs):
        captured.update(kwargs)
        return {
            "env_path": str(kwargs["env_path"]),
            "auth_path": str(kwargs["auth_path"]),
            "expires_at": 123,
            "scope": "read,write",
        }

    monkeypatch.delenv("LINEAR_AGENT_CLIENT_ID", raising=False)
    monkeypatch.delenv("LINEAR_AGENT_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(linear_oauth, "issue_client_credentials_token", fake_issue_client_credentials_token)

    assert linear_oauth.main(["--env-path", str(env_path), "--client-credentials"]) == 0
    assert captured["client_id"] == "client-1"
    assert captured["client_secret"] == "secret-1"
    assert captured["env_path"] == env_path
    assert captured["auth_path"] == env_path.with_name("auth.json")
    assert captured["scope"] == "read,write,admin"


@pytest.mark.asyncio
async def test_oauth_manager_refreshes_and_persists_rotated_tokens(monkeypatch):
    persisted = []

    async def fake_refresh_token(**kwargs):
        assert kwargs["client_id"] == "client-1"
        assert kwargs["client_secret"] == "secret-1"
        assert kwargs["refresh_token"] == "refresh-old"
        return {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
            "expires_in": 3600,
        }

    async def fake_client_credentials_failure(**kwargs):
        raise linear_oauth.LinearOAuthError("client credentials unavailable")

    monkeypatch.setattr(linear_oauth, "_async_client_credentials_token", fake_client_credentials_failure)
    monkeypatch.setattr(linear_oauth, "_async_refresh_token", fake_refresh_token)
    manager = LinearOAuthTokenManager(
        LinearOAuthConfig(
            client_id="client-1",
            client_secret="secret-1",
            refresh_token="refresh-old",
            expires_at=1,
            persist_callback=persisted.append,
        )
    )

    token = await manager.get_access_token()

    assert token == "access-new"
    assert manager.access_token == "access-new"
    assert manager.config.refresh_token == "refresh-new"
    assert persisted[0]["access_token"] == "access-new"
    assert persisted[0]["refresh_token"] == "refresh-new"
    assert persisted[0]["expires_at"] > time.time()


@pytest.mark.asyncio
async def test_oauth_manager_prefers_client_credentials_and_persists_access_token_only(monkeypatch):
    persisted = []

    async def fake_client_credentials_token(**kwargs):
        assert kwargs["client_id"] == "client-1"
        assert kwargs["client_secret"] == "secret-1"
        assert kwargs["scope"] == "read,write"
        return {
            "access_token": "app-access-new",
            "expires_in": 3600,
        }

    async def fail_refresh_token(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("refresh_token grant should not be used when client_credentials succeeds")

    monkeypatch.setattr(linear_oauth, "_async_client_credentials_token", fake_client_credentials_token)
    monkeypatch.setattr(linear_oauth, "_async_refresh_token", fail_refresh_token)
    manager = LinearOAuthTokenManager(
        LinearOAuthConfig(
            client_id="client-1",
            client_secret="secret-1",
            refresh_token="refresh-old",
            expires_at=1,
            oauth_scopes="read,write",
            persist_callback=persisted.append,
        )
    )

    token = await manager.get_access_token()

    assert token == "app-access-new"
    assert manager.access_token == "app-access-new"
    assert manager.config.refresh_token == "refresh-old"
    assert persisted[0]["access_token"] == "app-access-new"
    assert "refresh_token" not in persisted[0]
    assert persisted[0]["expires_at"] > time.time()


@pytest.mark.asyncio
async def test_oauth_manager_uses_client_credentials_with_real_session_factory():
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(
                200,
                {"access_token": "app-access-new", "expires_in": 3600},
            ),
        ]
    )
    manager = LinearOAuthTokenManager(
        LinearOAuthConfig(
            client_id="client-1",
            client_secret="secret-1",
            expires_at=1,
            oauth_scopes="read,write",
            token_url="https://api.linear.app/oauth/token",
            session_factory=session_factory,
        )
    )

    token = await manager.get_access_token()

    assert token == "app-access-new"
    assert session_factory.calls == [
        {
            "url": "https://api.linear.app/oauth/token",
            "json": None,
            "data": {
                "grant_type": "client_credentials",
                "client_id": "client-1",
                "client_secret": "secret-1",
                "scope": "read,write",
            },
            "headers": {},
        }
    ]


@pytest.mark.asyncio
async def test_graphql_client_reissues_client_credentials_token_after_401(monkeypatch):
    issued = []

    async def fake_client_credentials_token(**kwargs):
        token = f"app-access-{len(issued) + 1}"
        issued.append(token)
        return {"access_token": token, "expires_in": 3600}

    monkeypatch.setattr(linear_oauth, "_async_client_credentials_token", fake_client_credentials_token)
    manager = LinearOAuthTokenManager(
        LinearOAuthConfig(
            client_id="client-1",
            client_secret="secret-1",
            access_token="app-access-old",
            expires_at=time.time() + 3600,
        )
    )
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(401, {"error": "Authentication required, not authenticated"}),
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "app-user-1"}}}),
        ]
    )
    client = LinearGraphQLClient(token_manager=manager, session_factory=session_factory)

    data = await client.execute("query { viewer { id } }")

    assert data == {"viewer": {"id": "app-user-1"}}
    assert issued == ["app-access-1"]
    assert session_factory.calls[0]["headers"]["Authorization"] == "Bearer app-access-old"
    assert session_factory.calls[1]["headers"]["Authorization"] == "Bearer app-access-1"


@pytest.mark.asyncio
async def test_graphql_client_reuses_only_the_bound_event_loop_session():
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "main-loop"}}}),
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "main-loop-again"}}}),
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "worker-loop"}}}),
        ]
    )
    client = LinearGraphQLClient(
        access_token="access-1",
        session_factory=session_factory,
    )
    client.bind_pooled_loop()

    assert await client.execute("query { viewer { id } }") == {"viewer": {"id": "main-loop"}}
    assert await client.execute("query { viewer { id } }") == {"viewer": {"id": "main-loop-again"}}
    assert session_factory.created == 1

    result_holder = []

    def run_in_worker_loop():
        result_holder.append(
            asyncio.run(client.execute("query { viewer { id } }"))
        )

    worker = threading.Thread(target=run_in_worker_loop)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result_holder == [{"viewer": {"id": "worker-loop"}}]
    assert session_factory.created == 2


@pytest.mark.asyncio
async def test_graphql_client_refreshes_after_401_and_retries(monkeypatch):
    async def fake_refresh_token(**kwargs):
        return {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
            "expires_in": 3600,
        }

    async def fake_client_credentials_failure(**kwargs):
        raise linear_oauth.LinearOAuthError("client credentials unavailable")

    monkeypatch.setattr(linear_oauth, "_async_client_credentials_token", fake_client_credentials_failure)
    monkeypatch.setattr(linear_oauth, "_async_refresh_token", fake_refresh_token)
    manager = LinearOAuthTokenManager(
        LinearOAuthConfig(
            client_id="client-1",
            client_secret="secret-1",
            refresh_token="refresh-old",
            access_token="access-old",
            expires_at=time.time() + 3600,
        )
    )
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(401, {"error": "Authentication required, not authenticated"}),
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "user-1"}}}),
        ]
    )
    client = LinearGraphQLClient(token_manager=manager, session_factory=session_factory)

    data = await client.execute("query { viewer { id } }")

    assert data == {"viewer": {"id": "user-1"}}
    assert len(session_factory.calls) == 2
    assert session_factory.calls[0]["headers"]["Authorization"] == "Bearer access-old"
    assert session_factory.calls[1]["headers"]["Authorization"] == "Bearer access-new"


def test_webhook_signature_verification_passes_for_valid_signature():
    raw = _body({"hello": "linear"})
    assert verify_linear_signature(
        {"Linear-Signature": _signature(raw, "secret")},
        raw,
        "secret",
    )


def test_webhook_signature_verification_rejects_invalid_signature():
    raw = _body({"hello": "linear"})
    assert not verify_linear_signature(
        {"Linear-Signature": "sha256=bad"},
        raw,
        "secret",
    )


def test_webhook_signature_verification_accepts_sha256_colon_signature():
    raw = _body({"hello": "linear"})
    digest = hmac.new(b"secret", raw, hashlib.sha256).hexdigest()

    assert verify_linear_signature(
        {"Linear-Signature": f"sha256:{digest}"},
        raw,
        "secret",
    )


def test_webhook_signature_verification_accepts_linear_millisecond_timestamp():
    raw = _body({"hello": "linear"})
    digest = hmac.new(b"secret", raw, hashlib.sha256).hexdigest()
    timestamp_ms = str(int(time.time() * 1000))

    assert verify_linear_signature(
        {"Linear-Signature": digest, "Linear-Timestamp": timestamp_ms},
        raw,
        "secret",
    )


def test_webhook_signature_verification_rejects_stale_linear_timestamp():
    raw = _body({"hello": "linear"})
    digest = hmac.new(b"secret", raw, hashlib.sha256).hexdigest()
    stale_timestamp_ms = str(int((time.time() - 600) * 1000))

    assert not verify_linear_signature(
        {"Linear-Signature": digest, "Linear-Timestamp": stale_timestamp_ms},
        raw,
        "secret",
    )


def test_webhook_signature_verification_accepts_standard_webhook_signature():
    raw = _body({"hello": "linear"})
    timestamp = str(int(time.time()))
    webhook_id = "evt_standard_1"
    signed = f"{webhook_id}.{timestamp}.".encode("utf-8") + raw
    digest = hmac.new(b"secret", signed, hashlib.sha256).digest()
    signature = "v1," + base64.b64encode(digest).decode("ascii")

    assert verify_linear_signature(
        {
            "Webhook-Id": webhook_id,
            "Webhook-Timestamp": timestamp,
            "Webhook-Signature": signature,
        },
        raw,
        "secret",
    )


def test_webhook_signature_verification_accepts_v1_equals_signature():
    raw = _body({"hello": "linear"})
    digest = hmac.new(b"secret", raw, hashlib.sha256).digest()
    signature = "v1=" + base64.b64encode(digest).decode("ascii")

    assert verify_linear_signature(
        {"X-Linear-Webhook-Signature": signature},
        raw,
        "secret",
    )


@pytest.mark.asyncio
async def test_created_webhook_dispatches_session_and_emits_thought():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    payload = _created_payload()
    raw = _body(payload)

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response["status"] == "accepted"
    assert fake_client.calls[0] == (
        "thought",
        "session-1",
        "I’m starting work on this.",
    )
    assert len(captured) == 1
    event = captured[0]
    assert event.source.chat_id == "session-1"
    assert event.source.thread_id == "issue-1"
    assert event.source.user_id == "user-1"
    assert "You were invoked from Linear." in event.text
    assert "The issue has a failing CI check." in event.text
    assert "@Hermes please investigate this failure." in event.text


@pytest.mark.asyncio
async def test_prompted_webhook_routes_followup_to_same_session():
    adapter = _make_adapter()
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    payload = _prompted_payload()
    raw = _body(payload)

    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-2"),
        raw,
    )

    assert status == 200
    assert response["agent_session_id"] == "session-1"
    assert len(captured) == 1
    assert captured[0].source.chat_id == "session-1"
    assert captured[0].source.thread_id == "issue-1"
    assert captured[0].text == "Can you check the test log too?"


def _stop_prompted_payload(event_id="evt-stop"):
    payload = _prompted_payload(event_id=event_id)
    payload["agentActivity"]["signal"] = "stop"
    payload["agentActivity"]["body"] = "stop"
    return payload


@pytest.mark.asyncio
async def test_stop_signal_interrupts_and_confirms_without_dispatch():
    """A human `stop` signal halts the turn, posts one confirmation activity,
    and does NOT dispatch the activity body as a prompt (agent-signals contract)."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)

    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    interrupts = []

    async def fake_interrupt(session_key, chat_id):
        interrupts.append((session_key, chat_id))

    adapter.interrupt_session_activity = fake_interrupt
    # Seed a typing mark so we can assert it is cleared on stop.
    adapter._typing_marks["session-1"] = 123.0

    payload = _stop_prompted_payload()
    raw = _body(payload)

    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-stop"), raw
    )

    assert status == 200
    assert response["status"] == "stopped"
    assert response["agent_session_id"] == "session-1"
    # No prompt dispatched.
    assert dispatched == []
    # Interrupt was invoked for this session (chat_id == agent session id).
    assert len(interrupts) == 1
    assert interrupts[0][1] == "session-1"
    assert interrupts[0][0].endswith("session-1") or "session-1" in interrupts[0][0]
    # Exactly one confirming `response` activity was posted.
    responses = [c for c in fake_client.calls if c[0] == "response"]
    assert len(responses) == 1
    assert responses[0][1] == "session-1"
    assert "Stopped" in responses[0][2]
    # Typing mark cleared.
    assert "session-1" not in adapter._typing_marks


@pytest.mark.asyncio
async def test_stop_interrupt_key_matches_real_dispatch_session_key():
    """Contract: _session_key_for (the stop-signal interrupt path) must equal
    the session key that base handle_message files the running turn under —
    if they ever drift (grouping flags, source shape, profile handling),
    /stop would miss the _active_sessions entry and interrupt nothing.

    Drives a real prompted webhook through the REAL handle_message, capturing
    only _process_message_background (which receives the authoritative key).
    """
    from plugins.platforms.linear_agent.webhook import extract_context

    adapter = _make_adapter(client=_FakeLinearClient())
    captured = []

    async def capture_background(event, session_key):
        captured.append(session_key)

    async def message_handler(event):
        return None

    # handle_message returns immediately without a registered handler; the
    # handler itself is never reached because _process_message_background is
    # captured before it would run.
    adapter._message_handler = message_handler
    adapter._process_message_background = capture_background

    payload = _prompted_payload(event_id="evt-key-1")
    raw = _body(payload)
    headers = _headers(raw, "secret", delivery_id="evt-key-1")

    _, status = await adapter.handle_webhook(headers, raw)
    await asyncio.sleep(0)  # let the dispatch task reach the capture

    assert status == 200
    assert captured, "real handle_message never reached background dispatch"

    context = extract_context(payload, headers)
    assert adapter._session_key_for(context) == captured[0]


@pytest.mark.asyncio
async def test_prompted_without_signal_still_dispatches():
    """Regression: a normal prompted follow-up (no signal) dispatches as usual."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    payload = _prompted_payload()
    raw = _body(payload)

    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-2"), raw
    )

    assert status == 200
    assert response["status"] == "accepted"
    assert len(dispatched) == 1
    # No stop confirmation posted for a normal follow-up.
    assert not any("Stopped" in c[-1] for c in fake_client.calls if c[0] == "response")


@pytest.mark.asyncio
async def test_scoped_lock_acquired_on_connect_and_released_on_disconnect(monkeypatch):
    """AGENTS.md scoped-lock: connect() takes a per-credential lock and
    disconnect() releases it, so two profiles can't share one credential."""
    import gateway.status as status_mod

    acquired = []
    released = []

    def fake_acquire(scope, identity, metadata=None):
        acquired.append((scope, identity))
        return True, None

    def fake_release(scope, identity):
        released.append((scope, identity))

    monkeypatch.setattr(status_mod, "acquire_scoped_lock", fake_acquire)
    monkeypatch.setattr(status_mod, "release_scoped_lock", fake_release)

    # Ephemeral port so the test never collides with a real gateway.
    adapter = _make_adapter(extra={"port": 0})

    assert await adapter.connect() is True
    assert len(acquired) == 1
    scope, identity = acquired[0]
    assert scope == "linear_agent"
    # Identity is derived from the access token (hashed), never the raw token.
    assert identity.startswith("token:")
    assert "linear-token" not in identity

    await adapter.disconnect()
    assert released == [("linear_agent", identity)]


@pytest.mark.asyncio
async def test_scoped_lock_conflict_fails_connect(monkeypatch):
    import gateway.status as status_mod

    def fake_acquire(scope, identity, metadata=None):
        return False, {"pid": 4242}

    monkeypatch.setattr(status_mod, "acquire_scoped_lock", fake_acquire)
    adapter = _make_adapter(extra={"port": 0})

    assert await adapter.connect() is False


class _ViewerClient(_FakeLinearClient):
    """Fake client whose viewer query succeeds, fails, or records calls."""

    def __init__(self, viewer_id="viewer-1", fail=False):
        super().__init__()
        self._viewer_id = viewer_id
        self._fail = fail

    async def execute(self, query, variables=None):
        self.calls.append(("execute", query))
        if self._fail:
            raise RuntimeError("Linear is down")
        return {"viewer": {"id": self._viewer_id}}


def _lock_free(monkeypatch):
    import gateway.status as status_mod

    monkeypatch.setattr(
        status_mod, "acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None)
    )
    monkeypatch.setattr(status_mod, "release_scoped_lock", lambda scope, identity: None)


@pytest.mark.asyncio
async def test_connect_discovers_app_user_id_from_viewer(monkeypatch):
    """Env-only deployments without LINEAR_AGENT_APP_USER_ID self-heal at
    connect: the viewer query fills the id BEFORE the webhook server starts,
    enabling the self-echo filter and delegation verification."""
    _lock_free(monkeypatch)
    client = _ViewerClient(viewer_id="app-discovered")
    adapter = _make_adapter(extra={"port": 0}, client=client)
    assert adapter._app_user_id == ""

    assert await adapter.connect() is True
    try:
        assert adapter._app_user_id == "app-discovered"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_survives_viewer_discovery_failure(monkeypatch):
    """A Linear outage must not stop the adapter from starting — discovery
    failure warns and proceeds; the webhook secret still gates inbound."""
    _lock_free(monkeypatch)
    adapter = _make_adapter(extra={"port": 0}, client=_ViewerClient(fail=True))

    assert await adapter.connect() is True
    try:
        assert adapter._app_user_id == ""
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_never_overrides_configured_app_user_id(monkeypatch):
    """Explicit config wins: with app_user_id set, the viewer query is not
    even attempted."""
    _lock_free(monkeypatch)
    client = _ViewerClient(viewer_id="other-id")
    adapter = _make_adapter(extra={"port": 0, "app_user_id": "app-1"}, client=client)

    assert await adapter.connect() is True
    try:
        assert adapter._app_user_id == "app-1"
        assert not any(c[0] == "execute" for c in client.calls)
    finally:
        await adapter.disconnect()


class _AutoStartClient(LinearGraphQLClient):
    """Fake client for auto-start tests: records get_issue/list/update calls.

    Subclasses LinearGraphQLClient so the real ``first_started_state`` (the
    lowest-position 'started' picker) is exercised against list_issue_statuses.
    """

    def __init__(self, issue, states):
        self.calls = []
        self._issue = issue
        self._states = states

    async def create_thought(self, agent_session_id, body, *, ephemeral=False):
        self.calls.append(("thought", agent_session_id))
        return {"id": "t"}

    async def create_error(self, agent_session_id, body):
        self.calls.append(("error", agent_session_id, body))
        return {"id": "e"}

    async def get_issue(self, id):
        self.calls.append(("get_issue", id))
        return self._issue

    async def list_issue_statuses(self, team_id=None, team_name=None):
        self.calls.append(("list_issue_statuses", team_id))
        return self._states

    async def update_issue(self, issue_id, input_payload, *, mutation_policy=None):
        self.calls.append(("update_issue", issue_id, input_payload))
        return {"id": issue_id}


_STARTED_STATES = [
    {"id": "state-started-2", "name": "In Dev", "type": "started", "position": 2.0},
    {"id": "state-started-1", "name": "In Progress", "type": "started", "position": 1.0},
    {"id": "state-backlog", "name": "Backlog", "type": "backlog", "position": 0.0},
]


def _make_issue(state_type="backlog", delegate_id=None, issue_id="issue-1"):
    issue = {
        "id": issue_id,
        "identifier": "PLAT-1",
        "state": {"id": "s", "name": state_type.title(), "type": state_type},
        "team": {"id": "team-1"},
        "delegate": {"id": delegate_id} if delegate_id else None,
    }
    return issue


async def _noop_handler(event):
    return None


@pytest.mark.asyncio
async def test_auto_start_transitions_delegation_update_to_first_started_state():
    client = _AutoStartClient(_make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_start_on_delegation": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    _, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-as-1"), raw
    )

    assert status == 200
    update_calls = [c for c in client.calls if c[0] == "update_issue"]
    assert len(update_calls) == 1
    # Lowest-position started state wins (position 1.0). The adapter must
    # NEVER self-delegate — even for an unclaimed issue, only the state moves.
    assert update_calls[0][2] == {"stateId": "state-started-1"}


@pytest.mark.asyncio
async def test_auto_start_is_on_by_default_and_never_self_delegates():
    """auto_start_on_delegation defaults ON (opt-out): a genuine delegation
    with the update_issues policy enabled auto-starts without any flag. But
    self-delegation stays OFF by default — no delegateId, ever."""
    client = _AutoStartClient(_make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "mutation_policy": {"update_issues": True}},
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-8"), raw)

    update_calls = [c for c in client.calls if c[0] == "update_issue"]
    assert len(update_calls) == 1
    assert update_calls[0][2] == {"stateId": "state-started-1"}
    assert "delegateId" not in update_calls[0][2]


@pytest.mark.asyncio
async def test_auto_self_delegate_opt_in_claims_unclaimed_issue_on_created_session():
    """With auto_self_delegate: true, a REAL created agent session on an
    unclaimed issue claims it (delegateId) and starts it. Claiming only
    triggers on created sessions — never on generic update webhooks."""
    client = _AutoStartClient(_make_issue(state_type="unstarted"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_self_delegate": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_created_payload())

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-9"), raw)

    update_calls = [c for c in client.calls if c[0] == "update_issue"]
    assert update_calls[0][2] == {"stateId": "state-started-1", "delegateId": "app-1"}


@pytest.mark.asyncio
async def test_generic_issue_update_never_starts_or_claims():
    """HIGH-severity regression: an `update` webhook fires for ANY issue edit.
    Without a verified delegate == app user on the fetched issue, it must not
    auto-start — and must NEVER claim, even with auto_self_delegate: true."""
    client = _AutoStartClient(_make_issue(state_type="unstarted"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_self_delegate": True,  # even opted in
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-11"), raw)

    assert not any(c[0] == "update_issue" for c in client.calls)

    # Delegated to someone ELSE: equally untouchable.
    client2 = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="someone-else"), _STARTED_STATES
    )
    adapter2 = _make_adapter(
        client=client2,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_self_delegate": True,
        },
    )
    adapter2.handle_message = _noop_handler
    raw2 = _body(_issue_update_payload(actor_id="user-9"))

    await adapter2.handle_webhook(_headers(raw2, "secret", delivery_id="evt-as-12"), raw2)

    assert not any(c[0] == "update_issue" for c in client2.calls)


@pytest.mark.asyncio
async def test_update_payload_delegate_prefilter_skips_verification_fetch():
    """When the update payload itself serializes a delegate that isn't us,
    auto-start must bail WITHOUT the get_issue round-trip — every edit in the
    workspace fires an update webhook, so this is the hot path. The fake's
    fetched issue says delegated-to-us to prove we never looked."""
    client = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES
    )
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "mutation_policy": {"update_issues": True}},
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9", delegate_id="someone-else"))

    _, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-as-13"), raw
    )

    assert status == 200
    assert not any(c[0] == "get_issue" for c in client.calls)
    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_update_payload_delegate_to_us_is_never_trusted_unverified():
    """A payload claiming delegation to us is only a hint: the fetched issue
    stays authoritative. Here the fetch shows no delegate, so no auto-start —
    but the verification fetch must have happened."""
    client = _AutoStartClient(_make_issue(state_type="unstarted"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "mutation_policy": {"update_issues": True}},
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9", delegate_id="app-1"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-14"), raw)

    assert any(c[0] == "get_issue" for c in client.calls)
    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_auto_self_delegate_leaves_existing_delegate_alone():
    """Even opted in, an issue already delegated (to us or anyone) keeps its
    delegate — self-delegation only fills a vacuum."""
    client = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES
    )
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_self_delegate": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-10"), raw)

    update_calls = [c for c in client.calls if c[0] == "update_issue"]
    assert update_calls[0][2] == {"stateId": "state-started-1"}


@pytest.mark.asyncio
async def test_auto_start_skips_triage_state():
    client = _AutoStartClient(_make_issue(state_type="triage", delegate_id="app-1"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_start_on_delegation": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-2"), raw)

    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_auto_start_skips_already_started_issue():
    client = _AutoStartClient(_make_issue(state_type="started", delegate_id="app-1"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_start_on_delegation": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-3"), raw)

    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_auto_start_skipped_when_update_issues_policy_disabled():
    client = _AutoStartClient(_make_issue(delegate_id="app-1"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": False},
            "auto_start_on_delegation": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-4"), raw)

    # Policy gate fires before any issue fetch.
    assert client.calls == []


@pytest.mark.asyncio
async def test_auto_start_disabled_by_flag():
    client = _AutoStartClient(_make_issue(delegate_id="app-1"), _STARTED_STATES)
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_start_on_delegation": False,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_issue_update_payload(actor_id="user-9"))

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-5"), raw)

    assert client.calls == []


@pytest.mark.asyncio
async def test_auto_start_skips_mention_session_delegated_to_someone_else():
    """A created (mention) session where the agent is not the delegate is not
    delegation — do not auto-start."""
    client = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="someone-else"), _STARTED_STATES
    )
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_start_on_delegation": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_created_payload())

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-6"), raw)

    assert not any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_auto_start_fires_for_created_session_delegated_to_app_user():
    client = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES
    )
    adapter = _make_adapter(
        client=client,
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
            "auto_start_on_delegation": True,
        },
    )
    adapter.handle_message = _noop_handler
    raw = _body(_created_payload())

    await adapter.handle_webhook(_headers(raw, "secret", delivery_id="evt-as-7"), raw)

    update_calls = [c for c in client.calls if c[0] == "update_issue"]
    assert len(update_calls) == 1
    # Already delegated to us → no delegateId re-set, just the state move.
    assert update_calls[0][2] == {"stateId": "state-started-1"}


@pytest.mark.asyncio
async def test_permission_revoked_event_logs_warning_and_is_ignored(caplog):
    adapter = _make_adapter()
    raw = _body({"type": "OAuthApp", "action": "revoked"})

    with caplog.at_level("WARNING"):
        response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response == {"status": "ignored", "reason": "revoked"}
    assert any("revoked" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_permission_team_access_changed_event_logs_warning_and_is_ignored(caplog):
    adapter = _make_adapter()
    raw = _body({"action": "teamAccessChanged"})

    with caplog.at_level("WARNING"):
        response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response == {"status": "ignored", "reason": "teamAccessChanged"}
    assert any("team access changed" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_final_response_maps_to_agent_activity_create_response():
    client = _CaptureGraphQLClient()

    activity = await client.create_response("session-1", "Done.")

    assert activity == {"id": "activity-1"}
    _, variables = client.operations[0]
    assert variables["input"]["agentSessionId"] == "session-1"
    assert variables["input"]["content"] == {
        "type": "response",
        "body": "Done.",
    }


@pytest.mark.asyncio
async def test_adapter_send_uses_response_activity():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)

    result = await adapter.send("session-1", "Final answer")

    assert result.success is True
    assert result.message_id == "response-1"
    assert fake_client.calls == [("response", "session-1", "Final answer")]


@pytest.mark.asyncio
async def test_dispatch_error_maps_to_error_activity():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)

    async def explode(event):
        raise RuntimeError("boom")

    adapter.handle_message = explode
    payload = _created_payload()
    raw = _body(payload)

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response["status"] == "error"
    # The ack thought runs concurrently with dispatch (Linear's 5s webhook /
    # 10s ack deadlines), so its ordering relative to the error activity is
    # not guaranteed — assert both were sent rather than their order.
    kinds = [c[0] for c in fake_client.calls]
    assert sorted(kinds) == ["error", "thought"]
    error_call = next(c for c in fake_client.calls if c[0] == "error")
    assert error_call[1] == "session-1"
    assert "boom" in error_call[2]


@pytest.mark.asyncio
async def test_duplicate_webhook_event_is_ignored():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    payload = _created_payload(event_id="evt-dup")
    raw = _body(payload)
    headers = _headers(raw, "secret", delivery_id="evt-dup")

    first_response, first_status = await adapter.handle_webhook(headers, raw)
    second_response, second_status = await adapter.handle_webhook(headers, raw)

    assert first_status == 200
    assert first_response["status"] == "accepted"
    assert second_status == 200
    assert second_response["status"] == "duplicate"
    assert len(captured) == 1
    assert fake_client.calls == [
        (
            "thought",
            "session-1",
            "I’m starting work on this.",
        )
    ]


@pytest.mark.asyncio
async def test_no_allowlist_fails_closed_with_zero_side_effects():
    """With NO allowlist configured (no allow_all, no users), the webhook is
    rejected BEFORE any side effect — no ack thought, no auto-start fetch or
    mutation, no dispatch. The gateway env layer enforces the same policy at
    dispatch, but ack/auto-start/stop run pre-dispatch, so the adapter check
    must fail closed too."""
    client = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES
    )
    adapter = _make_adapter(
        client=client,
        extra={
            "allow_all_users": False,  # override the fixture's test default
            "app_user_id": "app-1",
            "mutation_policy": {"update_issues": True},
        },
    )
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture

    for payload, delivery in (
        (_created_payload(), "evt-fc-1"),
        (_issue_update_payload(actor_id="user-9"), "evt-fc-2"),
    ):
        raw = _body(payload)
        response, status = await adapter.handle_webhook(
            _headers(raw, "secret", delivery_id=delivery), raw
        )
        assert status == 403
        assert response["error"] == "Unauthorized Linear user or team"

    assert captured == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_gateway_authorization_grant_is_honored():
    """A grant from the gateway-registered auth callback (env wildcard,
    GATEWAY_ALLOWED_USERS, pairing store) must authorize the webhook even
    with no adapter-local allowlist — the adapter delegates to the same
    chain dispatch uses instead of duplicating (and drifting from) it."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client, extra={"allow_all_users": False})
    checked = []

    def gateway_check(user_id, chat_type=None, chat_id=None):
        checked.append((user_id, chat_type, chat_id))
        return user_id == "user-1"

    adapter.set_authorization_check(gateway_check)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    raw = _body(_created_payload())

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response["status"] == "accepted"
    assert len(dispatched) == 1
    assert checked and checked[0][0] == "user-1"


@pytest.mark.asyncio
async def test_team_narrowing_applies_even_over_gateway_grant():
    """allowed_teams narrows and never grants — a gateway-authorized user is
    still rejected when the webhook's team is outside the adapter allowlist."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(
        client=fake_client,
        extra={"allow_all_users": False, "allowed_teams": ["team-other"]},
    )
    adapter.set_authorization_check(lambda user_id, chat_type=None, chat_id=None: True)
    raw = _body(_created_payload())  # team-1

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 403
    assert fake_client.calls == []


def test_explicit_port_zero_is_preserved():
    """`port: 0` means "bind an ephemeral port" — it must not collapse into
    the fixed default via an `or` chain (that made connect tests race a live
    gateway on the real port)."""
    assert _make_adapter(extra={"port": 0})._port == 0
    assert _make_adapter(extra={"webhook_port": 0})._port == 0
    assert _make_adapter()._port == 8651


def test_persist_tokens_false_in_yaml_is_honored(monkeypatch):
    """Explicit `persist_tokens: false` must survive (the `or` chain used to
    collapse it into the env/default and keep writing tokens to auth.json)."""
    monkeypatch.delenv("LINEAR_AGENT_PERSIST_TOKENS", raising=False)
    adapter = _make_adapter(
        extra={"client_id": "cid", "client_secret": "cs", "persist_tokens": False}
    )
    assert adapter._oauth_manager is not None
    assert adapter._oauth_manager.config.persist_callback is None
    # And the default stays ON when unspecified.
    adapter2 = _make_adapter(extra={"client_id": "cid", "client_secret": "cs"})
    assert adapter2._oauth_manager.config.persist_callback is not None


@pytest.mark.asyncio
async def test_stale_body_timestamp_is_rejected():
    """Replay guard: with no timestamp header, the signed webhookTimestamp
    body field must stop a captured delivery from replaying after the dedup
    TTL. Fresh timestamps pass; stale ones get 400."""
    adapter = _make_adapter(client=_FakeLinearClient())
    adapter.handle_message = _noop_handler

    stale = _created_payload()
    stale["webhookTimestamp"] = int((time.time() - 3600) * 1000)
    raw = _body(stale)
    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-stale-1"), raw
    )
    assert status == 400
    assert "Stale" in response["error"]

    fresh = _created_payload()
    fresh["webhookTimestamp"] = int(time.time() * 1000)
    raw2 = _body(fresh)
    _, status2 = await adapter.handle_webhook(
        _headers(raw2, "secret", delivery_id="evt-stale-2"), raw2
    )
    assert status2 == 200


@pytest.mark.asyncio
async def test_send_chunks_oversized_responses():
    """Responses over MAX_MESSAGE_LENGTH split into multiple activities
    instead of failing the whole send."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)
    paragraphs = "\n\n".join(f"paragraph {i} " + "x" * 90 for i in range(600))
    assert len(paragraphs) > adapter.MAX_MESSAGE_LENGTH

    result = await adapter.send("session-1", paragraphs, metadata={})

    assert result.success is True
    responses = [c for c in fake_client.calls if c[0] == "response"]
    assert len(responses) >= 2
    rejoined = "\n".join(c[2] for c in responses)
    assert "paragraph 599" in rejoined


def test_guidance_list_renders_rule_bodies():
    """Linear sends guidance as [{body, origin}] — the prompt must carry the
    rule text, never a stringified dict."""
    from plugins.platforms.linear_agent.webhook import build_created_prompt, extract_context

    payload = _created_payload()
    payload["guidance"] = [
        {"body": "Prefer small PRs.", "origin": "team"},
        {"body": "Never force-push.", "origin": "workspace"},
    ]
    context = extract_context(payload, {})
    prompt = build_created_prompt(context, payload)

    assert "Prefer small PRs." in prompt
    assert "Never force-push." in prompt
    assert "{'body'" not in prompt


@pytest.mark.asyncio
async def test_configured_workspace_ignores_other_workspaces():
    """One Linear app (and webhook secret) can be installed in multiple
    workspaces — a valid signature does not imply the configured workspace.
    Foreign-workspace events are ignored (200, so Linear doesn't retry) with
    zero side effects; the configured workspace still processes."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client, extra={"workspace_id": "workspace-1"})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    foreign = _created_payload()
    foreign["agentSession"]["workspaceId"] = "workspace-other"
    raw = _body(foreign)
    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-ws-1"), raw
    )
    assert status == 200
    assert response["reason"] == "other workspace"
    assert dispatched == []
    assert fake_client.calls == []

    raw2 = _body(_created_payload())  # workspace-1
    _, status2 = await adapter.handle_webhook(
        _headers(raw2, "secret", delivery_id="evt-ws-2"), raw2
    )
    assert status2 == 200
    assert len(dispatched) == 1


@pytest.mark.asyncio
async def test_prompted_activity_content_object_body_is_extracted():
    """Agent activities serialize text as `content: {type, body}` — the
    prompt must be the body text, never a stringified dict."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    payload = _prompted_payload(event_id="evt-content-1")
    del payload["agentActivity"]["body"]
    payload["agentActivity"]["content"] = {
        "type": "prompt",
        "body": "Please also check the deploy logs.",
    }
    raw = _body(payload)

    _, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-content-1"), raw
    )

    assert status == 200
    assert len(dispatched) == 1
    assert "Please also check the deploy logs." in dispatched[0].text
    assert "{'type'" not in dispatched[0].text


@pytest.mark.asyncio
async def test_reply_in_source_thread_directive_is_opt_in():
    """When mentioned inside an existing thread, Linear supplies a source
    comment. With reply_in_source_thread on, the dispatched prompt instructs
    a reply on that comment; off (default), it never does."""
    async def _dispatch(extra):
        fake_client = _FakeLinearClient()
        adapter = _make_adapter(client=fake_client, extra=extra)
        dispatched = []

        async def capture(event):
            dispatched.append(event)

        adapter.handle_message = capture
        payload = _created_payload(event_id="evt-src-1")
        payload["sourceCommentId"] = "comment-src-9"
        raw = _body(payload)
        _, status = await adapter.handle_webhook(
            _headers(raw, "secret", delivery_id="evt-src-1"), raw
        )
        assert status == 200
        assert len(dispatched) == 1
        return dispatched[0].text

    # Opt-in: directive present, naming the exact source comment id.
    on = await _dispatch({"reply_in_source_thread": True})
    assert "Reply in the source thread" in on
    assert "comment-src-9" in on
    assert "linear_agent_create_comment" in on

    # Default: no directive, even though a source comment is present.
    off = await _dispatch({})
    assert "Reply in the source thread" not in off
    assert "comment-src-9" not in off


def test_register_routes_tools_through_plugin_context():
    """Tools re-register through ctx.register_tool so the plugin manager
    tracks them and the `linear_agent` toolset is discoverable in
    `hermes tools` / saved platform_toolsets selections."""
    from plugins.platforms.linear_agent.tools import TOOL_NAMES

    class Ctx:
        def __init__(self):
            self.registered = None
            self.tools = []

        def register_platform(self, **kwargs):
            self.registered = kwargs

        def register_tool(self, *, name, toolset, schema, handler, **kwargs):
            self.tools.append((name, toolset))

    ctx = Ctx()
    register(ctx)

    assert ctx.registered is not None
    assert {name for name, _ in ctx.tools} == set(TOOL_NAMES)
    assert all(toolset == "linear_agent" for _, toolset in ctx.tools)


@pytest.mark.asyncio
async def test_config_allowlist_denies_unauthorized_team():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(
        client=fake_client,
        extra={"allowed_teams": ["team-allowed"]},
    )
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    payload = _created_payload()
    raw = _body(payload)

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 403
    assert response["error"] == "Unauthorized Linear user or team"
    assert captured == []
    assert fake_client.calls == []


# ---------------------------------------------------------------------------
# Webhook signature fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsigned_webhook_rejected_when_no_secret_configured(monkeypatch):
    monkeypatch.delenv("LINEAR_AGENT_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_AGENT_ALLOW_UNSIGNED_WEBHOOKS", raising=False)
    adapter = _make_adapter(secret="")
    raw = _body(_created_payload())

    response, status = await adapter.handle_webhook({"Linear-Delivery-Id": "evt-1"}, raw)

    assert status == 401
    assert "secret" in response["error"].lower()


@pytest.mark.asyncio
async def test_unsigned_webhook_accepted_with_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("LINEAR_AGENT_WEBHOOK_SECRET", raising=False)
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(
        secret="",
        extra={"allow_unsigned_webhooks": True},
        client=fake_client,
    )

    async def _noop(event):
        return None

    adapter.handle_message = _noop
    raw = _body(_created_payload())

    response, status = await adapter.handle_webhook({"Linear-Delivery-Id": "evt-1"}, raw)

    assert status == 200
    assert response["status"] == "accepted"


# ---------------------------------------------------------------------------
# Standalone (out-of-process) cron delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_send_posts_comment_on_home_issue(monkeypatch):
    from plugins.platforms.linear_agent import adapter as adapter_module

    class _FakeStandaloneClient:
        last = None

        def __init__(self, token, *, api_url=None, token_manager=None, proxy_url=None):
            self.comments = []
            _FakeStandaloneClient.last = self

        @property
        def configured(self):
            return True

        async def create_comment(self, issue_id, body, *, parent_id=None, mutation_policy=None):
            self.comments.append((issue_id, body, mutation_policy))
            return {"id": "comment-99"}

    monkeypatch.setattr(adapter_module, "LinearGraphQLClient", _FakeStandaloneClient)

    result = await adapter_module._standalone_send(
        PlatformConfig(enabled=True, extra={"access_token": "tok"}),
        "ENG-123",
        "Nightly cron result",
    )

    assert result == {"success": True, "message_id": "comment-99", "issue": "ENG-123"}
    assert _FakeStandaloneClient.last.comments == [
        ("ENG-123", "Nightly cron result", {"create_comments": True}),
    ]


@pytest.mark.asyncio
async def test_standalone_send_rejects_missing_or_malformed_target():
    from plugins.platforms.linear_agent.adapter import _standalone_send

    config = PlatformConfig(enabled=True, extra={"access_token": "tok"})

    result = await _standalone_send(config, "", "message")
    assert "LINEAR_AGENT_HOME_TARGET" in result["error"]

    result = await _standalone_send(config, 'ENG-123"; mutation {', "message")
    assert "invalid issue target" in result["error"]


@pytest.mark.asyncio
async def test_graphql_client_discards_stale_session_after_loop_rebind():
    """Rebinding the pooled loop (reconnect without a clean disconnect, e.g.
    a gateway revive) must not reuse the session created on the old loop."""
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "old-loop"}}}),
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "new-loop"}}}),
        ]
    )
    client = LinearGraphQLClient(access_token="access-1", session_factory=session_factory)

    def run_in_old_loop():
        async def _go():
            client.bind_pooled_loop()
            await client.execute("query { viewer { id } }")

        asyncio.run(_go())

    worker = threading.Thread(target=run_in_old_loop)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert session_factory.created == 1

    # Simulate connect() on a fresh loop with no aclose() in between.
    client.bind_pooled_loop()
    result = await client.execute("query { viewer { id } }")

    assert result == {"viewer": {"id": "new-loop"}}
    assert session_factory.created == 2


# ---------------------------------------------------------------------------
# Issue-update (delegation) webhooks: self-actor filtering + comment replies
# ---------------------------------------------------------------------------


_OMIT = object()


def _issue_update_payload(actor_id="user-9", issue_id="issue-1", delegate_id=_OMIT):
    """Issue-data update webhook. By default the payload does NOT serialize a
    delegate field (delegate unknown → the adapter must fetch to verify);
    pass delegate_id (including None) to serialize one explicitly."""
    data = {
        "id": issue_id,
        "identifier": "PLAT-1",
        "title": "Fix the flaky job",
        "team": {"id": "team-1"},
    }
    if delegate_id is not _OMIT:
        data["delegate"] = {"id": delegate_id} if delegate_id else None
    return {
        "action": "update",
        "type": "Issue",
        "data": data,
        "actor": {"id": actor_id, "name": "Ada"},
    }


@pytest.mark.asyncio
async def test_issue_update_from_agent_itself_is_ignored():
    """The agent's own issue mutations echo back as update webhooks; without
    this filter each of its writes would spawn a session reviewing itself."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(extra={"app_user_id": "app-1"}, client=fake_client)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    raw = _body(_issue_update_payload(actor_id="app-1"))

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response == {"status": "ignored", "reason": "self-update"}
    assert dispatched == []
    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_issue_update_session_replies_as_comment():
    """Issue-update webhooks have no real agent session, so the reply path
    must post a comment on the issue instead of an agent activity."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(
        extra={
            "app_user_id": "app-1",
            "mutation_policy": {"create_comments": True},
            "dispatch_issue_updates": True,  # opt in to issue-edit turns
        },
        client=fake_client,
    )
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    raw = _body(_issue_update_payload(actor_id="user-9"))

    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-update-1"), raw
    )

    assert status == 200
    assert response["status"] == "accepted"
    assert len(dispatched) == 1
    assert dispatched[0].source.chat_id == "update:issue-1"

    result = await adapter.send("update:issue-1", "Reviewed — looks fine.")

    assert result.success is True
    assert fake_client.calls[-1] == (
        "comment",
        "issue-1",
        "Reviewed — looks fine.",
        adapter._mutation_policy,
    )

    # Error path routes to a comment as well.
    err = await adapter.send_error_activity("update:issue-1", "Something broke.")
    assert err.success is True
    assert fake_client.calls[-1][0] == "comment"


@pytest.mark.asyncio
async def test_issue_update_default_feeds_auto_start_but_never_dispatches():
    """Default posture: delegation already arrives as a real `created` agent
    session, so an issue-update webhook must feed auto-start only — a full
    turn per issue edit would double-process every delegation."""
    client = _AutoStartClient(
        _make_issue(state_type="unstarted", delegate_id="app-1"), _STARTED_STATES
    )
    adapter = _make_adapter(
        client=client,
        extra={"app_user_id": "app-1", "mutation_policy": {"update_issues": True}},
    )
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    raw = _body(_issue_update_payload(actor_id="user-9"))

    response, status = await adapter.handle_webhook(
        _headers(raw, "secret", delivery_id="evt-nodispatch-1"), raw
    )

    assert status == 200
    assert response["reason"] == "auto-start only"
    assert dispatched == []
    # Auto-start still ran and moved the delegated issue.
    assert any(c[0] == "update_issue" for c in client.calls)


@pytest.mark.asyncio
async def test_issue_update_without_issue_id_is_ignored():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(extra={"app_user_id": "app-1"}, client=fake_client)
    payload = _issue_update_payload(actor_id="user-9")
    payload["data"]["id"] = ""
    raw = _body(payload)

    response, status = await adapter.handle_webhook(_headers(raw, "secret"), raw)

    assert status == 200
    assert response["status"] == "ignored"


# ---------------------------------------------------------------------------
# Linear Agent Interaction Guidelines: elicitation, working-state, timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_clarify_posts_elicitation_activity():
    """Clarifying questions post as `elicitation` so Linear shows the session
    as awaitingInput; the reply arrives via a `prompted` webhook."""
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)

    result = await adapter.send_clarify(
        "session-1",
        "Which environment should I target?",
        ["production", "staging"],
        clarify_id="clarify-1",
        session_key="agent:main:linear_agent:thread:session-1",
    )

    assert result.success is True
    kind, session_id, body = fake_client.calls[-1]
    assert kind == "elicitation"
    assert session_id == "session-1"
    assert "1. production" in body and "2. staging" in body
    assert "Reply with the number" in body


@pytest.mark.asyncio
async def test_send_clarify_open_ended_posts_bare_question():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)

    result = await adapter.send_clarify(
        "session-1",
        "What should the project be called?",
        None,
        clarify_id="clarify-2",
        session_key="agent:main:linear_agent:thread:session-1",
    )

    assert result.success is True
    kind, _, body = fake_client.calls[-1]
    assert kind == "elicitation"
    assert body == "What should the project be called?"


@pytest.mark.asyncio
async def test_send_typing_posts_rate_limited_ephemeral_thought():
    fake_client = _FakeLinearClient()
    adapter = _make_adapter(client=fake_client)

    await adapter.send_typing("session-1")
    await adapter.send_typing("session-1")  # within the interval — suppressed

    thoughts = [c for c in fake_client.calls if c[0] == "thought"]
    assert len(thoughts) == 1
    assert fake_client.thought_ephemeral_flags == [True]

    # A different session gets its own indicator.
    await adapter.send_typing("session-2")
    thoughts = [c for c in fake_client.calls if c[0] == "thought"]
    assert len(thoughts) == 2

    # Synthetic issue-update sessions have no Linear session to indicate in.
    await adapter.send_typing("update:issue-1")
    thoughts = [c for c in fake_client.calls if c[0] == "thought"]
    assert len(thoughts) == 2


@pytest.mark.asyncio
async def test_client_retries_once_on_429_with_retry_after():
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(429, {"error": "rate limited"}, headers={"Retry-After": "0"}),
            _FakeGraphQLResponse(200, {"data": {"viewer": {"id": "ok"}}}),
        ]
    )
    client = LinearGraphQLClient(access_token="access-1", session_factory=session_factory)

    data = await client.execute("query { viewer { id } }")

    assert data == {"viewer": {"id": "ok"}}
    assert len(session_factory.calls) == 2


@pytest.mark.asyncio
async def test_client_gives_up_after_second_429():
    session_factory = _FakeGraphQLSessionFactory(
        [
            _FakeGraphQLResponse(429, {"error": "rate limited"}, headers={"Retry-After": "0"}),
            _FakeGraphQLResponse(429, {"error": "rate limited"}, headers={"Retry-After": "0"}),
        ]
    )
    client = LinearGraphQLClient(access_token="access-1", session_factory=session_factory)

    with pytest.raises(Exception) as excinfo:
        await client.execute("query { viewer { id } }")

    assert "429" in str(excinfo.value)
    assert len(session_factory.calls) == 2


def test_env_enablement_seeds_all_documented_env_vars(monkeypatch):
    """Every LINEAR_AGENT_* env var the setup docs promise must land in the
    platform config seed (the guide's env-wiring test pattern)."""
    from plugins.platforms.linear_agent.adapter import _env_enablement

    expected = {
        "LINEAR_AGENT_ACCESS_TOKEN": ("access_token", "tok-1"),
        "LINEAR_AGENT_WEBHOOK_SECRET": ("webhook_secret", "whsec-1"),
        "LINEAR_AGENT_APP_USER_ID": ("app_user_id", "app-1"),
        "LINEAR_AGENT_WORKSPACE_ID": ("workspace_id", "ws-1"),
        "LINEAR_AGENT_CLIENT_ID": ("client_id", "cid-1"),
        "LINEAR_AGENT_CLIENT_SECRET": ("client_secret", "csec-1"),
        "LINEAR_AGENT_REFRESH_TOKEN": ("refresh_token", "ref-1"),
        "LINEAR_AGENT_TOKEN_EXPIRES_AT": ("token_expires_at", "123"),
        "LINEAR_AGENT_REDIRECT_URI": ("redirect_uri", "http://localhost/cb"),
        "LINEAR_AGENT_OAUTH_SCOPES": ("oauth_scopes", "read,write"),
        "LINEAR_AGENT_HOME_TARGET": ("home_target", "ENG-123"),
    }
    for env_name, (_, value) in expected.items():
        monkeypatch.setenv(env_name, value)
    # Ensure the cached-token path doesn't shadow the env token.
    monkeypatch.setattr(
        "plugins.platforms.linear_agent.adapter._cached_access_token", lambda extra: ""
    )

    seed = _env_enablement()

    assert seed is not None
    for env_name, (key, value) in expected.items():
        assert seed.get(key) == value, f"{env_name} did not seed extra[{key!r}]"


def test_session_source_round_trips_through_dict():
    """Guide checklist: the SessionSource built for Linear sessions must
    survive to_dict -> from_dict unchanged (session persistence)."""
    source = SessionSource(
        platform=Platform("linear_agent"),
        chat_id="session-1",
        chat_name="LIN-123 Fix the flaky job",
        chat_type="thread",
        user_id="user-1",
        user_name="Ada",
        thread_id="issue-1",
        guild_id="workspace-1",
        message_id="evt-1",
    )

    restored = SessionSource.from_dict(source.to_dict())

    assert restored.platform == source.platform
    assert restored.chat_id == source.chat_id
    assert restored.chat_type == source.chat_type
    assert restored.user_id == source.user_id
    assert restored.thread_id == source.thread_id
    assert restored.guild_id == source.guild_id
    assert restored.message_id == source.message_id


# ---------------------------------------------------------------------------
# Interactive setup wizard (hermes gateway setup → Linear Agent)
# ---------------------------------------------------------------------------


def test_interactive_setup_static_token_flow(monkeypatch):
    """Drive the static-token wizard path end-to-end: credentials saved,
    app user ID auto-detected from the viewer query, allowlist built."""
    import importlib

    from plugins.platforms.linear_agent import adapter as adapter_module

    # Import (or RE-import) the wizard's lazy-import targets and patch the
    # module objects directly. String-target monkeypatch resolves via the
    # hermes_cli package attribute, which goes stale if another test popped
    # a submodule from sys.modules (test_langfuse_plugin does exactly that
    # to hermes_cli.config) — the patch would land on the orphaned module
    # while the wizard's fresh `from hermes_cli.config import ...` gets the
    # real, unpatched functions.
    hermes_config = importlib.import_module("hermes_cli.config")
    hermes_setup = importlib.import_module("hermes_cli.setup")
    hermes_cli_output = importlib.import_module("hermes_cli.cli_output")

    saved = {}
    prompts = iter(["tok-123", "whsec-abc", "user-ada", ""])  # token, secret, allowlist, home
    choices = iter([1, 0])  # auth method: static token; authorization: allowlist

    monkeypatch.setattr(hermes_config, "get_env_value", lambda k: "")
    monkeypatch.setattr(hermes_config, "save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(hermes_setup, "prompt_choice", lambda *a, **kw: next(choices))
    monkeypatch.setattr(hermes_cli_output, "prompt", lambda *a, **kw: next(prompts))
    monkeypatch.setattr(hermes_cli_output, "prompt_yes_no", lambda *a, **kw: True)
    monkeypatch.setattr(hermes_cli_output, "print_header", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_cli_output, "print_success", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_cli_output, "print_warning", lambda *a, **kw: None)

    def fake_graphql(token, query, timeout=15.0):
        assert token == "tok-123"
        if "viewer" in query:
            return {"viewer": {"id": "app-9", "name": "Agent"}}
        return {"users": {"nodes": [{"id": "user-ada", "name": "Ada"}]}}

    monkeypatch.setattr(adapter_module, "_setup_graphql", fake_graphql)

    adapter_module.interactive_setup()

    assert saved["LINEAR_AGENT_ACCESS_TOKEN"] == "tok-123"
    assert saved["LINEAR_AGENT_WEBHOOK_SECRET"] == "whsec-abc"
    # The wizard auto-detected and saved the app user ID (echo-filter var).
    assert saved["LINEAR_AGENT_APP_USER_ID"] == "app-9"
    assert saved["LINEAR_AGENT_ALLOWED_USERS"] == "user-ada"
    assert "LINEAR_AGENT_HOME_TARGET" not in saved


def test_register_wires_interactive_setup():
    class Ctx:
        def __init__(self):
            self.registered = None

        def register_platform(self, **kwargs):
            self.registered = kwargs

    ctx = Ctx()
    register(ctx)
    assert callable(ctx.registered.get("setup_fn"))
