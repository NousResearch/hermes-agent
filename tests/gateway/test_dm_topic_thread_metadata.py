"""Lock down the DM-topic metadata key for `_thread_metadata_for_target`.

Regression test for the bot-API-routing fix: the topic id must live
under `message_thread_id` (the standard Bot API field), NOT under
`direct_messages_topic_id` (a legacy key that `sendMessage` silently
ignores — which is why startup/shutdown notifications used to land
in the DM root instead of the configured topic).
"""
from unittest.mock import patch

from gateway.config import Platform
from gateway.run import GatewayRunner


def test_dm_topic_target_uses_message_thread_id():
    runner = GatewayRunner.__new__(GatewayRunner)

    with patch.object(GatewayRunner, "_is_telegram_dm_topic_target", return_value=True):
        meta = runner._thread_metadata_for_target(
            Platform.TELEGRAM, "777", "239",
        )

    assert meta["message_thread_id"] == "239"
    assert "direct_messages_topic_id" not in meta
