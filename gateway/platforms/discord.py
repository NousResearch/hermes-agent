"""Compatibility wrapper for the bundled Discord platform plugin.

Discord moved from ``gateway.platforms.discord`` into
``plugins.platforms.discord.adapter``. Keep the old import path alive for
existing tests and external integrations without carrying a second copy of the
adapter implementation.
"""

from plugins.platforms.discord import adapter as _adapter
from plugins.platforms.discord.adapter import *  # noqa: F401,F403

# Re-export selected private helpers used by compatibility tests/integrations.
_has_raw_user_mention = _adapter._has_raw_user_mention
_parse_discord_bot_msg_v1 = _adapter._parse_discord_bot_msg_v1
_discord_bot_reply_false_reaction = _adapter._discord_bot_reply_false_reaction
_build_allowed_mentions = _adapter._build_allowed_mentions
_define_discord_view_classes = _adapter._define_discord_view_classes
