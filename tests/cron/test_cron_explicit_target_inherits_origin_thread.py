"""``deliver="<platform>:<chat>"`` must keep the origin's thread (topic) lane.

A job created from inside a topic carries ``origin.thread_id`` — the lane the user scheduled
from. Its delivery target is usually written as an explicit ``platform:chat_id`` pair (the
``cronjob`` tool, the dashboard, and hand-edited ``jobs.json`` all produce that form), which
names no thread. Only Slack inherited the origin thread for a chat-only target, so on every
other topic-capable platform the report fell into the chat's top-level/General lane:

    WARNING cron.scheduler: Job 'ba51c5f0dd27': origin has thread_id=631677 but delivery
    target lost it (deliver=telegram:167384518, target={... 'thread_id': None ...})

Production evidence (Telegram DM topics, job fired daily for weeks): 38 occurrences in
``agent.log`` in one log rotation, one per fire, for every job stored with an explicit
chat-only target.

The inherit is scoped to the origin conversation (same platform, same chat) and never
overrides an explicit thread, so a fan-out target cannot be redirected into a topic it was
not addressed to, and the platform's General sentinel (``telegram:<chat>:1``) still selects
the top-level lane deliberately.
"""

import pytest

from cron import scheduler_delivery as sd


def _job(deliver, origin):
    return {"id": "j1", "name": "nightly brief", "deliver": deliver, "origin": origin}


TELEGRAM_ORIGIN = {
    "platform": "telegram",
    "chat_id": "167384518",
    "chat_name": "Artem Ponomarenko",
    "thread_id": "631677",
    "user_id": "167384518",
}


def _resolved(job):
    targets = sd._resolve_delivery_targets(job)
    assert len(targets) == 1, targets
    return targets[0]


def test_telegram_chat_only_target_inherits_origin_topic_thread():
    """The reported bug: chat-only target on the origin chat keeps the DM-topic thread."""
    target = _resolved(_job("telegram:167384518", TELEGRAM_ORIGIN))
    assert target["platform"] == "telegram"
    assert target["chat_id"] == "167384518"
    assert target["thread_id"] == "631677"


def test_explicit_thread_in_target_wins_over_origin():
    """An explicit lane is never rewritten — the report goes where it was addressed."""
    target = _resolved(_job("telegram:167384518:999", TELEGRAM_ORIGIN))
    assert target["thread_id"] == "999"


def test_general_sentinel_still_selects_the_top_level_lane():
    """Escape hatch for a deliberate top-level post in the origin chat: Telegram's General
    sentinel parses to a thread id, so it survives the inherit instead of being replaced."""
    target = _resolved(_job("telegram:167384518:1", TELEGRAM_ORIGIN))
    assert target["thread_id"] == "1"


def test_different_chat_does_not_inherit_the_origin_thread():
    """Fan-out to another chat is not the origin conversation — no thread adoption."""
    target = _resolved(_job("telegram:-100555", TELEGRAM_ORIGIN))
    assert target["chat_id"] == "-100555"
    assert target["thread_id"] is None


@pytest.mark.parametrize("platform, chat, thread", [
    ("matrix", "!abc:srv", "$evt1"),
    ("mattermost", "chan1", "post9"),
    ("discord", "42", "77"),
])
def test_platforms_without_durable_topic_lanes_keep_posting_top_level(platform, chat, thread):
    """Matrix (auto_thread) and Mattermost (reply_mode: thread) stamp the creation message's own
    id as the origin thread; inheriting it would move existing room jobs into that message's
    reply thread with no top-level opt-out. Only Telegram topics and Slack threads inherit."""
    origin = {"platform": platform, "chat_id": chat, "thread_id": thread}
    target = _resolved(_job(f"{platform}:{chat}", origin))
    assert target["thread_id"] is None


def test_stale_slack_per_message_thread_is_not_inherited(monkeypatch):
    """Regression guard for the Slack rule this generalisation came from: a thread-per-message
    session key is not a location, so it must not be adopted even on the origin chat."""
    monkeypatch.setattr(sd, "_get_home_target_chat_id", lambda platform: "D0ORIGIN")
    origin = {"platform": "slack", "chat_id": "D0ORIGIN", "thread_id": "1700000000.000100"}
    target = _resolved(_job("slack:D0ORIGIN", origin))
    assert target["thread_id"] is None


def test_genuine_slack_thread_on_a_non_home_chat_is_still_inherited(monkeypatch):
    """Removing the Slack-only gate must not weaken Slack: a real thread on a non-home chat is
    a location and keeps being inherited."""
    monkeypatch.setattr(sd, "_get_home_target_chat_id", lambda platform: "DHOME")
    origin = {"platform": "slack", "chat_id": "C0TEAM", "thread_id": "1700000000.000100"}
    target = _resolved(_job("slack:C0TEAM", origin))
    assert target["thread_id"] == "1700000000.000100"


def test_origin_thread_is_stale_keeps_its_slack_scope(monkeypatch):
    """The stale-key heuristic must stay Slack-only — on Telegram a DM-topic id IS a lane."""
    monkeypatch.setattr(sd, "_get_home_target_chat_id", lambda platform: "167384518")
    assert sd._origin_thread_is_stale(TELEGRAM_ORIGIN) is False
    assert sd._origin_thread_is_stale(
        {"platform": "slack", "chat_id": "167384518", "thread_id": "631677"}) is True


def test_chat_only_target_and_origin_collapse_to_one_delivery():
    """Before the inherit, ``telegram:<chat>,origin`` resolved to TWO targets — the same chat in
    the General lane and in the origin topic — so a job addressed twice posted its report twice,
    once outside the topic. Both spellings now resolve to the same lane and dedupe to one send."""
    targets = sd._resolve_delivery_targets(_job("telegram:167384518,origin", TELEGRAM_ORIGIN))
    assert [(t["platform"], t["chat_id"], t["thread_id"]) for t in targets] == [
        ("telegram", "167384518", "631677")]
