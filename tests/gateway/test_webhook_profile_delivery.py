"""Profile isolation contracts for webhook reply delivery."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


class _RecordingTarget:
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.calls: list[tuple[str, str, object]] = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append((chat_id, content, metadata))
        return SendResult(success=True)


class _Runner(GatewayAuthorizationMixin):
    def __init__(self, default_target=None, coder_target=None):
        default_config = PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="default-home",
                name="Default home",
            ),
        )
        self.config = GatewayConfig(
            multiplex_profiles=True,
            platforms={Platform.TELEGRAM: default_config},
        )
        self.adapters = {Platform.TELEGRAM: default_target} if default_target else {}
        self._profile_adapters = {
            "coder": ({Platform.TELEGRAM: coder_target} if coder_target else {})
        }


def _target(*, home_chat_id: str | None) -> _RecordingTarget:
    home = None
    if home_chat_id is not None:
        home = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id=home_chat_id,
            name="Profile home",
        )
    return _RecordingTarget(PlatformConfig(enabled=True, home_channel=home))


def _adapter(route: dict, runner: _Runner) -> WebhookAdapter:
    adapter = WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {"r": route}},
        )
    )
    adapter.gateway_runner = runner
    return adapter


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


@pytest.fixture
def served_profiles(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", None), ("coder", None)],
    )


@pytest.mark.asyncio
async def test_profiled_deliver_only_uses_only_url_profile_adapter(
    served_profiles,
):
    default_target = _target(home_chat_id="default-home")
    coder_target = _target(home_chat_id="coder-home")
    adapter = _adapter(
        {
            "secret": _INSECURE_NO_AUTH,
            "deliver": "telegram",
            "deliver_only": True,
            "deliver_extra": {
                "chat_id": "destination",
                "profile": "{requested_profile}",
            },
            "prompt": "{message}",
        },
        _Runner(default_target, coder_target),
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/coder/webhooks/r",
            json={"message": "hello", "requested_profile": "default"},
            headers={"X-GitHub-Delivery": "profiled-direct"},
        )

    assert response.status == 200
    assert coder_target.calls == [("destination", "hello", None)]
    assert default_target.calls == []


@pytest.mark.asyncio
async def test_default_profile_url_uses_only_default_adapter(served_profiles):
    default_target = _target(home_chat_id="default-home")
    coder_target = _target(home_chat_id="coder-home")
    adapter = _adapter(
        {
            "secret": _INSECURE_NO_AUTH,
            "deliver": "telegram",
            "deliver_only": True,
            "deliver_extra": {
                "chat_id": "destination",
                "profile": "{requested_profile}",
            },
            "prompt": "{message}",
        },
        _Runner(default_target, coder_target),
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/default/webhooks/r",
            json={"message": "hello", "requested_profile": "coder"},
            headers={"X-GitHub-Delivery": "profiled-default"},
        )

    assert response.status == 200
    assert default_target.calls == [("destination", "hello", None)]
    assert coder_target.calls == []


@pytest.mark.asyncio
async def test_profiled_agent_reply_uses_named_profiles_home_channel(
    served_profiles,
):
    default_target = _target(home_chat_id="default-home")
    coder_target = _target(home_chat_id="coder-home")
    adapter = _adapter(
        {
            "secret": _INSECURE_NO_AUTH,
            "deliver": "telegram",
            "prompt": "{message}",
        },
        _Runner(default_target, coder_target),
    )
    received = []

    async def _capture(event):
        received.append(event)

    adapter.handle_message = _capture

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/p/coder/webhooks/r",
            json={"message": "question"},
            headers={"X-GitHub-Delivery": "profiled-agent"},
        )
        await asyncio.sleep(0)

    assert response.status == 202
    assert len(received) == 1
    result = await adapter.send(received[0].source.chat_id, "answer")

    assert result.success is True
    assert coder_target.calls == [("coder-home", "answer", None)]
    assert default_target.calls == []


@pytest.mark.asyncio
async def test_named_profile_with_no_target_adapter_fails_closed():
    default_target = _target(home_chat_id="default-home")
    adapter = _adapter(
        {},
        _Runner(default_target=default_target, coder_target=None),
    )

    result = await adapter._deliver_cross_platform(
        "telegram",
        "secret",
        {"profile": "coder", "deliver_extra": {"chat_id": "destination"}},
    )

    assert result.success is False
    assert default_target.calls == []


@pytest.mark.asyncio
async def test_named_profile_with_no_home_does_not_use_default_home():
    default_target = _target(home_chat_id="default-home")
    coder_target = _target(home_chat_id=None)
    adapter = _adapter({}, _Runner(default_target, coder_target))

    result = await adapter._deliver_cross_platform(
        "telegram", "secret", {"profile": "coder", "deliver_extra": {}}
    )

    assert result.success is False
    assert coder_target.calls == []
    assert default_target.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("with_default", [True, False])
async def test_unprefixed_delivery_keeps_default_then_secondary_fallback(with_default):
    default_target = _target(home_chat_id="default-home") if with_default else None
    coder_target = _target(home_chat_id="coder-home")
    adapter = _adapter({}, _Runner(default_target, coder_target))

    result = await adapter._deliver_cross_platform(
        "telegram", "legacy", {"deliver_extra": {"chat_id": "destination"}}
    )

    assert result.success is True
    expected = default_target if with_default else coder_target
    assert expected.calls == [("destination", "legacy", None)]
    if with_default:
        assert coder_target.calls == []
