"""Provider-wall notice delivery: fan out to every served home channel, once per chat."""

import json
import time

import pytest

from agent import provider_wall_notice as wall
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner

TEXT = "⚠️ Provider wall — no usable route\n\n🔴 custom · deepseek-v4-flash — weekly quota\n"


class _FailingAdapter(RestartTestAdapter):
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=False, error="chat not found")


def _marker(tmp_path, monkeypatch):
    monkeypatch.setattr(wall, "marker_path", lambda home=None: tmp_path / wall.WALL_MARKER_NAME)
    return tmp_path / wall.WALL_MARKER_NAME


def _write_pending(**overrides):
    payload = {
        "v": 1, "signature": "sig1", "profile": "default", "text": TEXT,
        "created_at": time.time(), "updated_at": time.time(),
        "delivered_targets": [], "delivered_at": None, "expires_at": time.time() + 3600,
    }
    payload.update(overrides)
    wall._write(payload)
    return payload


def _runner(adapter=None, *, notifications_enabled=True, monkeypatch=None, profile_config=None):
    runner, adapter = make_restart_runner(adapter)
    for name in (
        "_replay_pending_provider_wall_notice",
        "_profile_runs_incident_route",
        "_served_home_channel_configs",
        "_served_home_channel_transports",
        "_home_channel_transports",
        "_send_home_channel_message",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    # Routing metadata is not this suite's subject; the real helper needs session plumbing.
    runner._thread_metadata_for_target = lambda *args, **kwargs: {}
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, token="***",
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="HOME", name="home"),
    )})
    if monkeypatch is not None:
        import gateway.warning_notifications as warnings
        import hermes_cli.config as hermes_config

        monkeypatch.setattr(warnings, "warning_notifications_enabled",
                            lambda platform, user_config=None: notifications_enabled)
        monkeypatch.setattr(hermes_config, "load_config_readonly",
                            lambda *args, **kwargs: profile_config or {})
    return runner, adapter


@pytest.mark.asyncio
async def test_fanout_delivers_to_the_home_channel_and_clears_the_marker(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending()
    runner, adapter = _runner(monkeypatch=monkeypatch)

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == [TEXT]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_fanout_skips_the_chat_that_already_has_the_notice_in_band(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending()
    runner, adapter = _runner(monkeypatch=monkeypatch)

    from gateway.run_shutdown import _delivery_target_key

    await runner._replay_pending_provider_wall_notice(
        skip_chats={_delivery_target_key("telegram", "HOME", None)})

    assert adapter.sent == []
    assert not marker.exists()  # skipped counts as served; the marker must not linger


@pytest.mark.asyncio
async def test_fanout_respects_the_diagnostics_opt_out(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending()
    runner, adapter = _runner(monkeypatch=monkeypatch, notifications_enabled=False)

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == []
    assert not marker.exists()


@pytest.mark.asyncio
async def test_failed_send_keeps_the_marker_for_a_later_pass(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending()
    runner, _adapter = _runner(adapter=_FailingAdapter(), monkeypatch=monkeypatch)

    await runner._replay_pending_provider_wall_notice()

    assert marker.exists()
    assert json.loads(marker.read_text())["delivered_targets"] == []


@pytest.mark.asyncio
async def test_second_pass_does_not_re_send_a_delivered_notice(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending()
    runner, adapter = _runner(monkeypatch=monkeypatch)
    # A channel that could not be reached on the first pass keeps the marker alive.
    runner._send_home_channel_message = _failing_first_send(runner)
    await runner._replay_pending_provider_wall_notice()
    runner._send_home_channel_message = GatewayRunner._send_home_channel_message.__get__(runner, GatewayRunner)

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == [TEXT]  # delivered exactly once
    assert not marker.exists()


def _failing_first_send(runner):
    sent = {"n": 0}

    async def _send(*args, **kwargs):
        sent["n"] += 1
        return False

    return _send


@pytest.mark.asyncio
async def test_replay_without_a_marker_is_a_noop(tmp_path, monkeypatch):
    _marker(tmp_path, monkeypatch)
    runner, adapter = _runner(monkeypatch=monkeypatch)

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == []


# ── only the profiles the wall can actually hurt ─────────────────────────

WALL_ROUTES = [{"provider": "custom", "model": "deepseek-v4-flash"}]


@pytest.mark.asyncio
async def test_fanout_reaches_a_profile_running_the_affected_route(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending(routes=WALL_ROUTES)
    runner, adapter = _runner(
        monkeypatch=monkeypatch,
        profile_config={"model": {"default": "deepseek-v4-flash", "provider": "custom"}},
    )

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == [TEXT]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_fanout_skips_a_profile_running_an_unrelated_route(tmp_path, monkeypatch):
    marker = _marker(tmp_path, monkeypatch)
    _write_pending(routes=WALL_ROUTES)
    runner, adapter = _runner(
        monkeypatch=monkeypatch,
        profile_config={"model": {"default": "grok-4", "provider": "xai"}},
    )

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == []
    assert not marker.exists()  # nothing owed here: the marker clears instead of lingering


@pytest.mark.asyncio
async def test_fanout_reaches_a_profile_whose_chain_uses_the_affected_route(tmp_path, monkeypatch):
    _marker(tmp_path, monkeypatch)
    _write_pending(routes=WALL_ROUTES)
    runner, adapter = _runner(
        monkeypatch=monkeypatch,
        profile_config={
            "model": {"default": "grok-4", "provider": "xai"},
            "fallback_providers": [{"provider": "custom", "model": "deepseek-v4-flash"}],
        },
    )

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == [TEXT]


@pytest.mark.asyncio
async def test_fanout_notifies_when_the_route_config_cannot_be_read(tmp_path, monkeypatch):
    _marker(tmp_path, monkeypatch)
    _write_pending(routes=WALL_ROUTES)
    runner, adapter = _runner(monkeypatch=monkeypatch)

    import hermes_cli.config as hermes_config

    def _boom(*args, **kwargs):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(hermes_config, "load_config_readonly", _boom)

    await runner._replay_pending_provider_wall_notice()

    assert adapter.sent == [TEXT]  # doubt resolves to notifying, never to silence

