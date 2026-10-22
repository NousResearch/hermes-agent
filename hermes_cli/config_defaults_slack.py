"""Slack CLI defaults; runtime adapter config is loaded independently by the gateway."""

SLACK_DEFAULT_CONFIG = {
    "require_mention": True,  # require @mention to respond in channels
    "free_response_channels": "",  # comma-separated channel IDs answered without mention
    "allowed_channels": "",  # if set, ONLY respond in these channel IDs (whitelist)
    "require_mention_channels": "",  # channel IDs where @mention is ALWAYS required
    # Channel IDs where ONLY a native <@bot> mention counts — mention_patterns
    # wake words are ignored there. Env: SLACK_NATIVE_MENTION_ONLY_CHANNELS.
    "native_mention_only_channels": "",
    # Ignore messages whose first token @mentions another user unless the bot is also mentioned.
    # Env: SLACK_IGNORE_OTHER_USER_MENTIONS.
    "ignore_other_user_mentions": False,
    "thread_require_mention": False,  # require @mention in thread replies too
    "channel_prompts": {},  # per-channel ephemeral system prompts
}
