from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Reuse the repository's dependency bootstrap for environments without Slack SDKs.
import tests.gateway.test_slack  # noqa: F401
from gateway.config import PlatformConfig, Platform
from gateway.session import SessionSource
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "external_resource_authz": {
                "resource": "agent-draper",
                "endpoint": "https://access.example/api/admin/members",
                "token_secret": "ACCESS_CENTER_READ_TOKEN",
            }
        },
    )
    result = SlackAdapter(config)
    result._app = MagicMock()
    result._app.client = AsyncMock()
    result._get_client = MagicMock(return_value=result._app.client)
    result._app.client.users_info = AsyncMock(return_value={
        "user": {"profile": {"email": "User@Example.com", "email_verified": True}}
    })
    return result


@pytest.mark.asyncio
async def test_external_membership_allows_verified_member_and_caches(adapter):
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"active": True})
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get.return_value = request
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    source = SessionSource(Platform.SLACK, "D1", user_id="U1")
    with patch("plugins.platforms.slack.adapter.get_secret", return_value="read-token"), patch(
        "plugins.platforms.slack.adapter.aiohttp.ClientSession", return_value=session
    ):
        assert await adapter._authorize_external_resource("U1", chat_id="D1", team_id="T1", source=source)
        assert await adapter._authorize_external_resource("U1", chat_id="D1", team_id="T1", source=source)
    assert source.external_resource_authorized is True
    assert session.get.call_count == 1
    assert "email=user%40example.com" in session.get.call_args.args[0]


@pytest.mark.asyncio
async def test_external_membership_fails_closed_for_unverified_or_null(adapter):
    source = SessionSource(Platform.SLACK, "D1", user_id="U1")
    adapter._app.client.users_info.return_value = {
        "user": {"profile": {"email": "user@example.com", "email_verified": False}}
    }
    with patch("plugins.platforms.slack.adapter.get_secret", return_value="read-token"):
        assert not await adapter._authorize_external_resource("U1", chat_id="D1", team_id="T1", source=source)
    adapter._app.client.users_info.return_value = {
        "user": {"profile": {"email": "user@example.com", "email_verified": True}}
    }
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=None)
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get.return_value = request
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    with patch("plugins.platforms.slack.adapter.get_secret", return_value="read-token"), patch(
        "plugins.platforms.slack.adapter.aiohttp.ClientSession", return_value=session
    ):
        assert not await adapter._authorize_external_resource("U1", chat_id="D1", team_id="T1", source=source)


@pytest.mark.asyncio
async def test_expired_allow_is_not_used(adapter):
    source = SessionSource(Platform.SLACK, "D1", user_id="U1")
    adapter._external_resource_cache[("agent-draper", "user@example.com")] = 1.0
    adapter._app.client.users_info.return_value = {
        "user": {"profile": {"email": "user@example.com", "email_verified": True}}
    }
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"member": True})
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get.return_value = request
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    with patch("plugins.platforms.slack.adapter.get_secret", return_value="read-token"), patch(
        "plugins.platforms.slack.adapter.aiohttp.ClientSession", return_value=session
    ), patch("plugins.platforms.slack.adapter.time.monotonic", return_value=2.0):
        assert not await adapter._authorize_external_resource("U1", chat_id="D1", team_id="T1", source=source)
    assert source.external_resource_authorized is False


@pytest.mark.asyncio
async def test_interactive_path_requires_external_membership_before_gateway_auth(adapter):
    seen = {}

    class Runner:
        def _is_user_authorized(self, source):
            seen["marker"] = source.external_resource_authorized
            return True

    adapter._message_handler = Runner()._is_user_authorized
    async def external(_user_id, *, source, **_kwargs):
        source.external_resource_authorized = True
        return True

    external = AsyncMock(side_effect=external)
    with patch.object(adapter, "_authorize_external_resource", external):
        assert await adapter._authorize_interactive_user(
            "U1", channel_id="D1", team_id="T1"
        )
    external.assert_awaited_once()
    assert seen["marker"] is True


def test_disabled_behavior_and_wire_marker_fail_closed():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    source = SessionSource(Platform.SLACK, "D1", user_id="U1", external_resource_authorized=True)
    assert not SessionSource.from_dict(source.to_dict()).external_resource_authorized
    assert not adapter.external_resource_authorization_required()
    assert adapter.external_resource_authorized(source)


def test_required_adapter_needs_local_marker_for_final_gate(adapter):
    source = SessionSource(Platform.SLACK, "D1", user_id="U1")
    assert not adapter.external_resource_authorized(source)
    source.external_resource_authorized = True
    assert adapter.external_resource_authorized(source)
