"""Per-platform streaming defaults + dashboard exposure.

Streaming is smooth on Telegram (native sendMessageDraft) but flickers on
edit-only platforms like Discord and Slack (repeated editMessage). The shipped
defaults encode that: display.platforms.telegram.streaming=true,
.discord.streaming=false, .slack.streaming=false. These are gap-fillers (user
values win via deep-merge) and, because the dashboard schema is generated from
DEFAULT_CONFIG, they automatically appear as editable toggles in the web UI.
"""

from __future__ import annotations


def test_default_per_platform_streaming_flags():
    from hermes_cli.config import DEFAULT_CONFIG
    plats = DEFAULT_CONFIG["display"]["platforms"]
    assert plats["telegram"]["streaming"] is True
    assert plats["discord"]["streaming"] is False
    assert plats["slack"]["streaming"] is False


