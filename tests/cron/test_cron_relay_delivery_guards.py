"""Fire-time guards: stale Slack creation-thread routing + relay-fronted preflight.

Two related defects on relay-fronted Slack deployments:

1. Jobs persisted before the synthetic-thread capture fix carry the creation
   message's own id as ``origin.thread_id``. At fire time ``deliver=origin``
   replayed it unconditionally, and the Slack origin-affinity re-attach put it
   back even on explicit ``slack:<chat_id>`` targets. Guard: when the resolved
   Slack chat IS the configured home chat, the origin thread is a stale
   per-message artifact — deliver top-level (home thread config still wins).

2. ``_preflight_check_delivery`` validated the ``slack:`` prefix against
   natively-configured platforms only; in relay-only topology that set is
   ``{relay}`` and the job was refused with "no gateway credentials configured"
   although fire-time routing (resolve_delivery_transport + fronts_platform)
   would have delivered it. Preflight must consult the relay's fronted set.
"""

from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler as sched
from cron.scheduler import (
    _preflight_check_delivery,
    _resolve_single_delivery_target,
    cron_delivery_targets,
)


def _slack_home(monkeypatch, chat_id="D0BJTDCSR7C", thread_id=None):
    monkeypatch.setattr(sched, "_get_home_target_chat_id",
                        lambda p: chat_id if p == "slack" else None)
    monkeypatch.setattr(sched, "_get_home_target_thread_id",
                        lambda p: thread_id if p == "slack" else None)


SYNTH = "1755043010.123456"


class TestOriginThreadStaleGuard:
    def test_origin_thread_dropped_when_chat_is_home(self, monkeypatch):
        """deliver=origin, slack origin chat == home chat: creation thread is stale."""
        _slack_home(monkeypatch)
        job = {"origin": {"platform": "slack", "chat_id": "D0BJTDCSR7C",
                          "thread_id": SYNTH}}
        target = _resolve_single_delivery_target(job, "origin")
        assert target == {"platform": "slack", "chat_id": "D0BJTDCSR7C",
                          "thread_id": None, "_resolved_from": "origin"}

    def test_origin_thread_kept_when_chat_not_home(self, monkeypatch):
        """A non-home Slack origin thread may be a genuine working thread: keep it."""
        _slack_home(monkeypatch, chat_id="D_OTHER_HOME")
        job = {"origin": {"platform": "slack", "chat_id": "C0AGENERAL",
                          "thread_id": "1755040000.000100"}}
        target = _resolve_single_delivery_target(job, "origin")
        assert target["thread_id"] == "1755040000.000100"

    def test_home_thread_config_still_wins(self, monkeypatch):
        """When the home target itself pins a thread, deliver there, not top-level."""
        _slack_home(monkeypatch, thread_id="1755000000.000001")
        job = {"origin": {"platform": "slack", "chat_id": "D0BJTDCSR7C",
                          "thread_id": SYNTH}}
        target = _resolve_single_delivery_target(job, "origin")
        assert target["thread_id"] == "1755000000.000001"

    def test_non_slack_origin_thread_untouched(self, monkeypatch):
        """Telegram forum-topic origins replay their thread verbatim."""
        _slack_home(monkeypatch)
        job = {"origin": {"platform": "telegram", "chat_id": "-1003941067111",
                          "thread_id": "2203"}}
        target = _resolve_single_delivery_target(job, "origin")
        assert target["thread_id"] == "2203"

    def test_explicit_target_no_reattach_when_chat_is_home(self, monkeypatch):
        """slack:<home_chat> must not inherit the stale creation thread."""
        _slack_home(monkeypatch)
        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None)
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda platform, rest, **kw: (rest, None, None))
        job = {"origin": {"platform": "slack", "chat_id": "D0BJTDCSR7C",
                          "thread_id": SYNTH}}
        target = _resolve_single_delivery_target(job, "slack:D0BJTDCSR7C")
        assert target["thread_id"] is None

    def test_explicit_target_reattach_kept_for_non_home_chat(self, monkeypatch):
        """Origin-affinity re-attach is preserved for genuine non-home threads."""
        _slack_home(monkeypatch, chat_id="D_OTHER_HOME")
        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None)
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda platform, rest, **kw: (rest, None, None))
        job = {"origin": {"platform": "slack", "chat_id": "C0AGENERAL",
                          "thread_id": "1755040000.000100"}}
        target = _resolve_single_delivery_target(job, "slack:C0AGENERAL")
        assert target["thread_id"] == "1755040000.000100"


def _gateway_config(connected_values):
    config = MagicMock()
    config.get_connected_platforms.return_value = [
        MagicMock(value=v) for v in connected_values
    ]
    return config


class TestPreflightRelayFronted:
    def test_relay_fronted_slack_accepted(self, monkeypatch):
        """Relay-only topology fronting slack: slack:CHAT passes preflight."""
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"relay"})):
            assert _preflight_check_delivery(
                {"deliver": "slack:D0BJTDCSR7C"}) is None

    def test_unfronted_platform_still_rejected(self, monkeypatch):
        """The relay fronting slack does not whitelist other platforms."""
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"relay"})):
            reason = _preflight_check_delivery({"deliver": "discord:12345"})
            assert reason is not None
            assert "discord" in reason

    def test_native_strictness_without_relay(self, monkeypatch):
        """No relay configured: the native credential check is unchanged."""
        monkeypatch.delenv("GATEWAY_RELAY_PLATFORMS", raising=False)
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"telegram"})):
            reason = _preflight_check_delivery(
                {"deliver": "slack:D0BJTDCSR7C"})
            assert reason is not None
            assert "slack" in reason

    def test_delivery_targets_include_relay_fronted(self, monkeypatch):
        """The UI dropdown source offers relay-fronted platforms."""
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
        _slack_home(monkeypatch)
        monkeypatch.setattr(sched, "_iter_home_target_platforms",
                            lambda: ["slack", "telegram"])
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config({"relay"})):
            ids = {t["id"] for t in cron_delivery_targets()}
        assert "slack" in ids
        assert "telegram" not in ids
