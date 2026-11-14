"""Cron delivery to the generic ``relay`` platform (#150378).

A relay-only gateway uses the generic ``relay`` identity (``GATEWAY_RELAY_PLATFORMS`` unset, so
``relay_platform_identities()`` returns ``[("relay", "")]``). Interactive replies work, but every
cron lane refused the relay because ``_is_known_delivery_platform`` only accepted the env-var-less
core rows ``webhook`` / ``wecom_callback`` and plugin platforms carrying a
``PlatformEntry.cron_deliver_env_var`` — the relay has neither, so:

* ``deliver: relay`` (bare) resolved to no target at all,
* ``deliver: relay:<chat_id>`` was refused by preflight as "not a known cron delivery target",
* ``deliver: origin`` on a job with no origin (API/portal/CLI-created routines) never fell back to
  the relay home, because ``_iter_home_target_platforms`` lists env-var platforms only.

The relay's home lives in ``platforms.relay.home_channel`` (it has no ``<PLATFORM>_HOME_CHANNEL``
mirror), so the row belongs with ``webhook``/``wecom_callback`` and the platform belongs in the
home-target iteration. These tests pin all three lanes plus the negative controls that must keep
refusing an unknown platform name.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cron import scheduler_delivery as sched_delivery
from cron.scheduler_delivery import (
    _resolve_single_delivery_target,
    _is_known_delivery_platform,
    cron_delivery_targets,
)
from cron.scheduler_preflight import _preflight_check_delivery

RELAY_CHAT = "relay-chat-1"


def _relay_home(monkeypatch, chat_id=RELAY_CHAT, thread_id=None):
    """The relay's config.yaml ``home_channel`` and nothing else: no env-var mirror, no plugins."""
    home = SimpleNamespace(chat_id=chat_id, thread_id=thread_id)
    monkeypatch.setattr(
        sched_delivery, "_get_config_home_channel",
        lambda name: home if str(name).lower() == "relay" else None)
    monkeypatch.setattr(sched_delivery, "_env_home_target_chat_id", lambda name: "")
    monkeypatch.setattr(sched_delivery, "_plugin_cron_env_var", lambda name: "")


def _gateway_config(connected_values):
    config = MagicMock()
    config.get_connected_platforms.return_value = [
        MagicMock(value=v) for v in connected_values
    ]
    return config


class TestRelayIsADeliveryPlatform:
    def test_relay_is_known_and_unknown_names_still_rejected(self):
        assert _is_known_delivery_platform("relay")
        assert _is_known_delivery_platform("RELAY")  # user-written values are case-insensitive
        assert not _is_known_delivery_platform("no-such-platform")

    def test_relay_is_a_home_target_platform(self):
        """``deliver: origin``/``all`` iterate this, so the env-var-less relay must appear here."""
        assert "relay" in list(sched_delivery._iter_home_target_platforms())


class TestBareRelayDeliver:
    def test_bare_relay_resolves_to_the_config_home_channel(self, monkeypatch):
        _relay_home(monkeypatch)
        target = _resolve_single_delivery_target({"name": "relay-job"}, "relay")
        assert target == {
            "platform": "relay",
            "chat_id": RELAY_CHAT,
            "thread_id": None,
            "_resolved_from": "home",
        }

    def test_bare_relay_without_a_home_channel_resolves_nothing(self, monkeypatch):
        """No configured home: the target stays unresolved instead of inventing a chat id."""
        _relay_home(monkeypatch, chat_id="")
        assert _resolve_single_delivery_target({"name": "relay-job"}, "relay") is None


class TestExplicitRelayDeliver:
    def test_explicit_relay_chat_passes_preflight(self, monkeypatch):
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"relay"})):
            assert _preflight_check_delivery({"deliver": f"relay:{RELAY_CHAT}"}) is None

    def test_explicit_relay_chat_resolves_to_that_chat(self, monkeypatch):
        monkeypatch.setattr("tools.send_message_tool.prepare_send_message_platforms",
                            lambda: None)
        monkeypatch.setattr("tools.send_message_tool.resolve_send_target",
                            lambda platform, rest, **kw: (rest, None, None))
        target = _resolve_single_delivery_target({"name": "relay-job"}, f"relay:{RELAY_CHAT}")
        assert target is not None
        assert target["platform"] == "relay"
        assert target["chat_id"] == RELAY_CHAT
        assert target["_resolved_from"] == "explicit"

    def test_unknown_platform_still_fails_preflight(self, monkeypatch):
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"relay"})):
            reason = _preflight_check_delivery({"deliver": "no-such-platform:x"})
        assert reason is not None
        assert "no-such-platform" in reason


class TestOriginFallbackReachesTheRelay:
    def test_job_without_origin_falls_back_to_the_relay_home(self, monkeypatch):
        _relay_home(monkeypatch)
        target = _resolve_single_delivery_target({"name": "created-from-api"}, "origin")
        assert target is not None
        assert target["platform"] == "relay"
        assert target["chat_id"] == RELAY_CHAT
        assert target["_resolved_from"] == "origin_fallback"

    def test_no_home_anywhere_still_resolves_nothing(self, monkeypatch):
        _relay_home(monkeypatch, chat_id="")
        assert _resolve_single_delivery_target({"name": "created-from-api"}, "origin") is None


class TestDeliveryTargetsList:
    def test_relay_is_offered_when_connected_with_a_home(self, monkeypatch):
        _relay_home(monkeypatch)
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"relay"})):
            targets = {t["id"]: t for t in cron_delivery_targets()}
        assert "relay" in targets
        assert targets["relay"]["home_target_set"] is True
