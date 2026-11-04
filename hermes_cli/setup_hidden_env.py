"""Which platform env vars the setup surfaces hide.

Hiding is a *presentation* decision only: the vars keep working through ``hermes config set``,
``.env`` and ``config.yaml``, and the gateway reads them exactly as before.
"""

# Suffix match, so plugin adapters nobody enumerated (IRC, SimpleX, LINE, ntfy) get the same
# treatment without a code change here.
#
#   *_HOME_CHANNEL*        the bot offers /sethome on the first chat
#   *_ALLOW_ALL_USERS      defaults off; enabling it is a security decision
#   *_REPLY_TO_MODE / *_REPLY_MODE   cosmetic threading preference (Mattermost spelling too)
#   *_REQUIRE_MENTION / *_AUTO_THREAD   behavior toggles with sane defaults
#   *_FREE_RESPONSE_* / *_ALLOWED_CHANNELS   per-channel tuning, done once the bot is in a server
#   *_PROXY                only for networks that block the platform
#
# Allowlists (*_ALLOWED_USERS) deliberately stay visible: that IS the decision a new user has to
# make, and the gateway denies everyone until it's set.
SETUP_HIDDEN_ENV_SUFFIXES = (
    "_HOME_CHANNEL", "_HOME_CHANNEL_NAME", "_HOME_CHANNEL_THREAD_ID", "_HOME_ADDRESS", "_ALLOW_ALL_USERS",
    "_REPLY_TO_MODE", "_REPLY_MODE", "_REQUIRE_MENTION", "_AUTO_THREAD", "_FREE_RESPONSE_CHANNELS",
    "_FREE_RESPONSE_ROOMS", "_ALLOWED_CHANNELS", "_PROXY",
)


def is_setup_hidden_env(name: str) -> bool:
    """True when a var is self-configuring and shouldn't appear in setup forms. Callers must still
    keep any var a platform lists as *required* — hiding a required credential would make that
    platform unconfigurable from the UI."""
    return name.endswith(SETUP_HIDDEN_ENV_SUFFIXES)
