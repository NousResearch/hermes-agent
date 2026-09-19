"""``gateway_notice_channel`` — lifecycle broadcasts leave the home channel.

The home channel is where everything the agent says unprompted lands: cron
deliveries, ``send_message "home"``, CLI handoffs. Gateway lifecycle notices
("Gateway shutting down" / "Gateway online") went there too, for lack of a
destination of their own. On a deployment that restarts regularly they bury the
traffic the operator actually reads, and the only existing knob
(``gateway_restart_notification``) silences them rather than redirecting them.

``platforms.<name>.gateway_notice_channel`` gives those two notices a channel of
their own without moving the home channel. Unset means unchanged behaviour — and
the first tests below are exactly that control: if they fail, the change moved
more than it promised.
"""

from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from tests.gateway.restart_test_helpers import make_restart_runner


def _runner_with_home(notice=None):
    runner, adapter = make_restart_runner()
    cfg = runner.config.platforms[Platform.TELEGRAM]
    cfg.home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    cfg.gateway_notice_channel = notice
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="x"))
    return runner, adapter


# --------------------------------------------------------------- controls
# These must pass with and without the change. They are what proves the
# change moved the two lifecycle broadcasts and nothing else.

@pytest.mark.asyncio
async def test_control_without_config_shutdown_still_goes_to_home_channel():
    runner, adapter = _runner_with_home(notice=None)

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "home-42"


@pytest.mark.asyncio
async def test_control_without_config_startup_still_goes_to_home_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = _runner_with_home(notice=None)

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "home-42", None)}


@pytest.mark.asyncio
async def test_control_disabled_flag_still_silences_even_with_channel_set():
    runner, adapter = _runner_with_home(notice={"chat_id": "gateway-99"})
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_db_warning_broadcast_still_goes_to_home_channel():
    """The state.db warning is NOT a lifecycle notice and must not be redirected.

    ``home = platform_cfg.home_channel`` appears twice in run.py — once in the
    startup notice loop and once here. This is the only test that fails if both
    were changed instead of just the first.
    """
    runner, adapter = _runner_with_home(notice={"chat_id": "gateway-99"})
    # the broadcast returns early unless an init error is recorded
    runner._session_db_init_error = OSError("disk I/O error")

    await runner._send_session_db_warning_notifications()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "home-42"


# --------------------------------------------------------------- behaviour

@pytest.mark.asyncio
async def test_shutdown_notice_goes_to_notice_channel_not_home():
    runner, adapter = _runner_with_home(notice={"chat_id": "gateway-99"})

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited_once()
    chat_id, msg = adapter.send.await_args.args[0], adapter.send.await_args.args[1]
    assert chat_id == "gateway-99"
    assert "Gateway shutting down" in msg
    # the home channel gets no copy: one notice, one destination
    assert all(call.args[0] != "home-42" for call in adapter.send.await_args_list)


@pytest.mark.asyncio
async def test_startup_notice_goes_to_notice_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = _runner_with_home(notice={"chat_id": "gateway-99"})

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "gateway-99", None)}
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "gateway-99"
    assert "Gateway online" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_thread_id_reaches_metadata():
    runner, adapter = _runner_with_home(
        notice={"chat_id": "gateway-99", "thread_id": "t-7"}
    )

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "gateway-99"
    metadata = adapter.send.await_args.kwargs.get("metadata") or {}
    assert metadata.get("thread_id") == "t-7"


@pytest.mark.asyncio
async def test_blank_chat_id_falls_back_to_home():
    runner, adapter = _runner_with_home(notice={"chat_id": "   "})

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "home-42"


# ------------------------------------------------------------ config parse

def test_config_accepts_a_bare_chat_id():
    cfg = PlatformConfig.from_dict(
        {"enabled": True, "gateway_notice_channel": "gateway-99"}
    )
    assert cfg.gateway_notice_channel == {"chat_id": "gateway-99"}


def test_config_accepts_a_mapping_with_name_and_thread():
    cfg = PlatformConfig.from_dict(
        {
            "enabled": True,
            "gateway_notice_channel": {
                "chat_id": 12345,
                "name": "Gateway",
                "thread_id": "t-7",
            },
        }
    )
    assert cfg.gateway_notice_channel == {
        "chat_id": "12345",
        "name": "Gateway",
        "thread_id": "t-7",
    }


def test_config_reads_the_key_bridged_into_extra():
    """load_gateway_config() bridges shared keys into extra; both routes work."""
    cfg = PlatformConfig.from_dict(
        {"enabled": True, "extra": {"gateway_notice_channel": "gateway-99"}}
    )
    assert cfg.gateway_notice_channel == {"chat_id": "gateway-99"}


def test_config_control_unset_key_is_none():
    cfg = PlatformConfig.from_dict({"enabled": True})
    assert cfg.gateway_notice_channel is None


def test_config_drops_a_malformed_value_instead_of_half_parsing_it():
    """A list is neither a chat id nor a mapping. Fail closed, do not guess."""
    cfg = PlatformConfig.from_dict(
        {"enabled": True, "gateway_notice_channel": ["gateway-99"]}
    )
    assert cfg.gateway_notice_channel is None


def test_config_drops_unknown_subkeys():
    cfg = PlatformConfig.from_dict(
        {
            "enabled": True,
            "gateway_notice_channel": {"chat_id": "g-1", "platform": "slack"},
        }
    )
    assert cfg.gateway_notice_channel == {"chat_id": "g-1"}


def test_config_round_trips_through_to_dict():
    cfg = PlatformConfig.from_dict(
        {"enabled": True, "gateway_notice_channel": "gateway-99"}
    )
    assert PlatformConfig.from_dict(cfg.to_dict()).gateway_notice_channel == {
        "chat_id": "gateway-99"
    }


# --------------------------------------------------- relay provenance
# A Relay-fronted platform re-attaches user_id/scope_id on egress, and the caches
# it falls back to are filled only by *inbound* events. A dedicated operations
# channel is one nobody speaks in, so that cache stays cold and the connector's
# fail-closed tenant guard declines the send. The override therefore inherits the
# authenticated discriminators of the platform home channel rather than asking an
# operator to author them.

def _home_with_provenance():
    return HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
        user_id="U123",
        scope_id="S456",
    )


def test_notice_channel_inherits_relay_provenance_from_home():
    runner, _ = make_restart_runner()
    cfg = runner.config.platforms[Platform.TELEGRAM]
    cfg.home_channel = _home_with_provenance()
    cfg.gateway_notice_channel = {"chat_id": "gateway-99"}

    target = runner._gateway_notice_channel(Platform.TELEGRAM)

    assert target.chat_id == "gateway-99"
    assert target.user_id == "U123"
    assert target.scope_id == "S456"


def test_control_provenance_is_inherited_not_invented():
    """No provenance on the home channel means none on the override either."""
    runner, _ = make_restart_runner()
    cfg = runner.config.platforms[Platform.TELEGRAM]
    cfg.home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="home-42", name="Ops Home"
    )
    cfg.gateway_notice_channel = {"chat_id": "gateway-99"}

    target = runner._gateway_notice_channel(Platform.TELEGRAM)

    assert target.chat_id == "gateway-99"
    assert target.user_id is None
    assert target.scope_id is None


def test_config_does_not_accept_hand_authored_provenance():
    """Provenance is security-relevant; config is not where it is authored."""
    cfg = PlatformConfig.from_dict(
        {
            "enabled": True,
            "gateway_notice_channel": {
                "chat_id": "g-1",
                "user_id": "spoofed",
                "scope_id": "spoofed",
            },
        }
    )
    assert cfg.gateway_notice_channel == {"chat_id": "g-1"}
