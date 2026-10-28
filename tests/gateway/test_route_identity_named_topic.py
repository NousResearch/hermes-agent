"""Regression: async-delegation wakes for NAMED Telegram forum topics were refused with
``admission_conflict`` on every retry until the 48h replay cap terminally dropped them.

Root cause: ``_route_identity`` compared the FULL source dict (minus ``message_id``), but
the persisted routing origin of a delegation does not carry ``chat_topic`` — so the wake's
source could never byte-match the committed human source of a named topic. Unnamed topics
(``chat_topic`` absent on BOTH sides) delivered normally, which is why the bug looked
selective: same code, same chat, opposite outcomes decided solely by whether the forum
topic had a name.
"""
from dataclasses import replace

from gateway.session import Platform, SessionSource
from gateway.session_automation import _route_identity


def _topic_source(chat_topic):
    """A live source in the incident shape: Telegram group forum topic 9952."""
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id='-1004308125450', chat_type='group',
        thread_id='9952', chat_name='Hermes VPS Team', chat_topic=chat_topic,
        user_id='58509012', user_name='Gold', is_bot=False,
    )


def test_named_topic_wake_matches_committed_source():
    """The live incident: the wake's origin source (chat_topic lost in routing persistence)
    vs the committed human source (named topic). Routing identity must be equal so the
    wake is admitted."""
    origin = _topic_source(None)                         # routing origin: chat_topic lost
    committed = _topic_source('virgo-investor-portal')   # last human message: named topic
    assert _route_identity(origin) == _route_identity(committed)


def test_route_identity_still_discriminates_routes():
    """Weakening the identity must not collapse routing distinctions: different threads,
    chats and platforms are still different routes."""
    base = _topic_source('virgo-investor-portal')
    assert _route_identity(replace(base, thread_id='342')) != _route_identity(base)
    assert _route_identity(replace(base, chat_id='-999')) != _route_identity(base)
    assert _route_identity(replace(base, chat_type='dm')) != _route_identity(base)
    assert _route_identity(replace(base, platform=Platform.DISCORD)) != _route_identity(base)


def test_message_id_stays_out_of_route_identity():
    """The wake clears ``message_id`` (it is not a reply to the stale trigger); identity
    must ignore it (#52694)."""
    source = _topic_source('virgo-investor-portal')
    assert _route_identity(source) == _route_identity(replace(source, message_id='18768'))
