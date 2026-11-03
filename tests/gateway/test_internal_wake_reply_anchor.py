"""An internal (synthetic) wake never reply-anchors to its ``message_id``.

An admitted automation wake restores ``message_id`` as its producer identity (a JSON string), not a
platform message id. On a Telegram DM topic that identity became the reply anchor, ``int()`` failed
on it in the adapter, and the final send fell back to ledger redelivery behind a "Recovered reply"
banner. Synthetic events never reply-anchor (#52694); an explicit redirect override still wins.
"""

from gateway.config import Platform
from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

PRODUCER_IDENTITY = '["completion","proc_0123456789ab",1760000000.0]'


def _telegram_dm_topic_event(*, internal):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="u1", chat_type="dm",
                           thread_id="7")
    return MessageEvent(text="[process completed]", source=source, message_id=PRODUCER_IDENTITY,
                        internal=internal)


def test_internal_wake_on_telegram_dm_topic_has_no_reply_anchor():
    assert _reply_anchor_for_event(_telegram_dm_topic_event(internal=True)) is None


def test_user_message_on_telegram_dm_topic_still_anchors_to_its_message_id():
    assert _reply_anchor_for_event(_telegram_dm_topic_event(internal=False)) == PRODUCER_IDENTITY


def test_redirect_override_still_wins_on_an_internal_wake():
    event = _telegram_dm_topic_event(internal=True)
    event.reply_anchor_override = "1234"
    assert _reply_anchor_for_event(event) == "1234"
