"""``deliver="<platform>:<chat>:none"`` opts one job out of origin-thread inheritance (#148033).

A job created inside a Slack thread (or a Telegram topic) delivers a chat-only target into that
thread. That carry is deliberate and stays the default; the ``:none`` suffix is the per-job
escape that previously existed only platform-wide (``cron_continuable_surface: in_channel``).
"""
import logging

from cron import scheduler_delivery as sd

SLACK_ORIGIN = {"platform": "slack", "chat_id": "C0TEAM", "thread_id": "1700000000.000100"}
TELEGRAM_ORIGIN = {"platform": "telegram", "chat_id": "167384518", "thread_id": "631677"}


def _resolved(deliver, origin, monkeypatch, home="DHOME"):
    monkeypatch.setattr(sd, "_get_home_target_chat_id", lambda platform: home)
    targets = sd._resolve_delivery_targets({"id": "j1", "name": "brief", "deliver": deliver, "origin": origin})
    assert len(targets) == 1, targets
    return targets[0]


def test_slack_none_suffix_posts_top_level_on_the_origin_chat(monkeypatch):
    target = _resolved("slack:C0TEAM:none", SLACK_ORIGIN, monkeypatch)
    assert (target["platform"], target["chat_id"], target["thread_id"]) == ("slack", "C0TEAM", None)


def test_default_still_inherits_the_origin_thread(monkeypatch):
    target = _resolved("slack:C0TEAM", SLACK_ORIGIN, monkeypatch)
    assert target["thread_id"] == "1700000000.000100"


def test_suffix_is_case_insensitive_and_works_on_telegram(monkeypatch):
    target = _resolved("telegram:167384518:NONE", TELEGRAM_ORIGIN, monkeypatch)
    assert (target["chat_id"], target["thread_id"]) == ("167384518", None)


def test_suffix_with_no_chat_is_not_a_target(monkeypatch):
    monkeypatch.setattr(sd, "_get_home_target_chat_id", lambda platform: "DHOME")
    targets = sd._resolve_delivery_targets({"id": "j1", "deliver": "slack:none", "origin": SLACK_ORIGIN})
    assert all(t.get("chat_id") != "" for t in targets)
    assert not any(t.get("_top_level") for t in targets)


class _Relay:
    def fronts_platform(self, _platform):
        return True


def _prepare(target):
    from gateway.config import GatewayConfig, Platform
    job = {"id": "j1", "deliver": "slack:C0TEAM:none", "origin": SLACK_ORIGIN}
    return sd._prepare_target_delivery(
        job, target, adapters={Platform.RELAY: _Relay()}, loop=None, config=GatewayConfig(),
        notify_delivery=True, mirror_enabled=False, mirror_text="", delivery_errors=[])


def test_top_level_target_delivers_flat_without_a_lost_thread_warning(monkeypatch, caplog):
    target = _resolved("slack:C0TEAM:none", SLACK_ORIGIN, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=sd.logger.name):
        prepared = _prepare(target)
    assert prepared is not None and prepared.thread_id is None
    assert "lost it" not in caplog.text


def test_an_accidentally_flat_target_still_warns(monkeypatch, caplog):
    """The warning keeps catching real losses: only the deliberate suffix silences it."""
    with caplog.at_level(logging.WARNING, logger=sd.logger.name):
        _prepare({"platform": "slack", "chat_id": "C0TEAM", "thread_id": None})
    assert "lost it" in caplog.text
