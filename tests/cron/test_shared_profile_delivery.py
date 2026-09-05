"""Routed cron delivery uses the owner transport, not satellite credentials."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron import scheduler
from cron.scheduler_preflight import (
    SharedRouteAdapters,
    _primary_profile_routes_for_current_home,
)
from gateway.config import Platform, PlatformConfig, load_gateway_config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def satellite(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "research"
    home.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text(
        yaml.safe_dump({
            "gateway": {
                "multiplex_profiles": True,
                "profile_routes": [
                    {
                        "platform": "telegram",
                        "chat_id": "-123",
                        "thread_id": "3",
                        "profile": "research",
                    },
                    {
                        "platform": "telegram",
                        "chat_id": "-456",
                        "profile": "research",
                        "enabled": False,
                    },
                    {"platform": "telegram", "chat_id": "-789", "profile": "other"},
                ],
            }
        })
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "platforms": {
                "telegram": {
                    "enabled": False,
                    "extra": {"cron_continuable_surface": "thread"},
                },
            }
        })
    )
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def deliver(
    home,
    target="telegram:-123:3",
    *,
    shared=True,
    live=True,
    adapter_enabled=True,
    send_fails=False,
):
    sent = []

    async def send(chat_id, content, metadata=None):
        if send_fails:
            raise RuntimeError("controlled transport failure")
        sent.append((chat_id, content, metadata))
        return {"success": True, "message_id": "fixture-message"}

    adapter = SimpleNamespace(config=PlatformConfig(enabled=adapter_enabled), send=send)
    adapters = {Platform.TELEGRAM: adapter}
    if shared:
        adapters = SharedRouteAdapters(
            adapters, _primary_profile_routes_for_current_home()
        )
    loop = MagicMock()
    loop.is_running.return_value = live

    def submit(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    # Real profile config loader, route lookup and DeliveryRouter. Only the event
    # loop bridge and actual network transport are replaced; no bot credentials.
    with (
        patch.object(
            scheduler, "load_config", return_value={"cron": {"wrap_response": False}}
        ),
        patch("asyncio.run_coroutine_threadsafe", side_effect=submit),
        patch("tools.send_message_tool._send_to_platform") as standalone,
    ):
        error = scheduler._deliver_result(
            {"id": "fixture-job", "deliver": target},
            "fixture report",
            adapters=adapters,
            loop=loop,
        )
    standalone.assert_not_called()
    return error, sent


def test_disabled_satellite_uses_route_scoped_live_transport(satellite):
    before = (satellite / "config.yaml").read_bytes()
    error, sent = deliver(satellite)
    assert error is None
    assert len(sent) == 1
    assert sent[0][0:2] == ("-123", "fixture report")
    assert sent[0][2]["thread_id"] == "3"
    assert (satellite / "config.yaml").read_bytes() == before
    assert not load_gateway_config().platforms[Platform.TELEGRAM].enabled


def test_credentialless_satellite_without_platform_block(satellite):
    (satellite / "config.yaml").write_text("{}\n")
    error, sent = deliver(satellite)
    assert error is None
    assert len(sent) == 1


def test_target_config_preserves_satellite_settings_not_owner_credentials(satellite):
    from cron.scheduler_delivery import _resolve_target_transport

    config = load_gateway_config()
    primary = SimpleNamespace(
        config=PlatformConfig(enabled=True, token="owner-fixture")
    )
    shared = SharedRouteAdapters(
        {Platform.TELEGRAM: primary},
        _primary_profile_routes_for_current_home(),
    )
    resolved, error = _resolve_target_transport(
        {"id": "fixture"},
        Platform.TELEGRAM,
        "telegram",
        {"chat_id": "-123", "thread_id": "3"},
        shared,
        config,
    )
    assert error is None
    assert resolved is not None
    transport, pconfig, _, _, target_config = resolved
    assert transport is not None
    assert transport.adapter is primary
    assert pconfig.enabled
    assert pconfig.token != primary.config.token
    assert pconfig.extra == config.platforms[Platform.TELEGRAM].extra
    assert target_config is not config
    assert not config.platforms[Platform.TELEGRAM].enabled


@pytest.mark.parametrize(
    "target", ["telegram:-999", "telegram:-456", "telegram:-789", "telegram:-123:4"]
)
def test_unrouted_disabled_other_profile_and_other_topic_stay_closed(satellite, target):
    error, sent = deliver(satellite, target)
    assert error and "not configured/enabled" in error
    assert sent == []


def test_target_override_does_not_leak_to_next_target(satellite):
    error, sent = deliver(satellite, "telegram:-123:3,telegram:-999")
    assert error and "not configured/enabled" in error
    assert [item[0] for item in sent] == ["-123"]


def test_continuable_mode_cannot_move_delivery_outside_authorized_topic(satellite):
    from cron.scheduler_delivery import _prepare_target_delivery

    config = load_gateway_config()
    config.platforms[Platform.TELEGRAM].extra["cron_continuable_surface"] = "in_channel"
    primary = SimpleNamespace(
        config=PlatformConfig(enabled=True),
        supports_inchannel_continuable=True,
    )
    shared = SharedRouteAdapters(
        {Platform.TELEGRAM: primary},
        _primary_profile_routes_for_current_home(),
    )
    origin = {"platform": "telegram", "chat_id": "-123", "thread_id": "3"}
    errors = []
    prepared = _prepare_target_delivery(
        {"id": "fixture", "deliver": "origin", "origin": origin},
        origin,
        adapters=shared,
        loop=SimpleNamespace(is_running=lambda: True),
        config=config,
        notify_delivery=False,
        mirror_enabled=False,
        mirror_text="fixture",
        delivery_errors=errors,
    )
    assert prepared is None
    assert errors and "route" in errors[0]


def test_ordinary_adapter_map_does_not_bypass_disabled_config(satellite):
    error, sent = deliver(satellite, shared=False)
    assert error and "not configured/enabled" in error
    assert sent == []


def test_disabled_owner_adapter_is_not_revived(satellite):
    error, sent = deliver(satellite, adapter_enabled=False)
    assert error and "not configured/enabled" in error
    assert sent == []


@pytest.mark.parametrize("kwargs", [{"live": False}, {"send_fails": True}])
def test_shared_transport_failure_never_uses_standalone_credentials(satellite, kwargs):
    error, sent = deliver(satellite, **kwargs)
    assert error
    if kwargs.get("send_fails"):
        assert "controlled transport failure" in error
    assert sent == []
