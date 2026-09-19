"""Copy and owner-side signalling for unauthorized inbound senders.

Two audiences, two rules:

* The **stranger** gets either a pairing code (behaviour ``pair``) or nothing at all (behaviour
  ``ignore``: an allowlist is configured, so any reply would leak that the bot exists).
* The **owner** is the one who can fix a mis-typed allowlist, so an ignored DM is logged at
  WARNING with the sender's ID and the allowlist / pairing-mode fix, and surfaced once per
  (platform, user) per gateway process in the platform's home channel when one is configured.
  No pairing request is minted for an ignored sender: strangers must not create server state
  when the owner has restricted access.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from gateway.pairing import CODE_TTL_SECONDS, _allowlist_env_for_platform

# Display names come from the stranger. Bound them and keep the mention/markdown surface small in
# the owner's channel: a name is never a reason to render a link, mention or new section.
_MAX_SENDER_NAME_CHARS = 64
# Distinct (platform, user_id) senders remembered per process; a bot rotating IDs evicts the oldest
# instead of growing the set without limit.
_MAX_SEEN_SENDERS = 2048

logger = logging.getLogger("gateway.run")


def pairing_profile_arg(pairing_store) -> str:
    """``-p <profile> `` when the store belongs to a non-default profile, else ``""``."""
    store_profile = getattr(pairing_store, "profile", None)
    if isinstance(store_profile, str) and store_profile and store_profile != "default":
        return f"-p {store_profile} "
    return ""


def pairing_code_reply(platform_name: str, code: str, profile_arg: str = "") -> str:
    """The DM a first-time sender receives: what happened, how long the code lives, what to do
    whether they are the owner or a guest, and that they must message again after approval."""
    hours = max(1, CODE_TTL_SECONDS // 3600)
    validity = f"{hours} hour" if hours == 1 else f"{hours} hours"
    approve_cmd = f"hermes {profile_arg}pairing approve {platform_name} {code}"
    return (
        "Hi! I don't recognize you yet, so I can't reply until the person running this bot "
        "approves you.\n\n"
        f"Your pairing code: `{code}` (valid for {validity})\n\n"
        f"If you run this bot, open a terminal and run: `{approve_cmd}`. "
        "Otherwise send that command to the bot owner. After approval, send your message again."
    )


PAIRING_RATE_LIMITED_REPLY = (
    "Too many pairing requests right now. Wait a few minutes, then send your message again.")


def unauthorized_owner_hint(
    platform_name: str, user_id: str, user_name: str = "", *, hermes_home: str,
) -> str:
    """One-line hint for the owner (log + home channel): who was dropped and how to let them in.
    No pairing request is minted for an ignored sender (a configured allowlist means the owner chose
    to restrict access), so the ways in are the allowlist itself or switching the platform to
    pairing mode."""
    from gateway.session import neutralize_untrusted_inline_text

    safe_name = neutralize_untrusted_inline_text(user_name or "", max_chars=_MAX_SENDER_NAME_CHARS)
    # Mention and link sigils are stripped so a hostile display name cannot ping the channel or
    # smuggle a link; the ID next to it is the value the owner actually acts on.
    safe_name = "".join(ch for ch in safe_name if ch not in "@<>[]()`*_~#").strip()
    who = f"{safe_name} ({user_id})" if safe_name else str(user_id)
    env_var = _allowlist_env_for_platform(platform_name)
    allowlist = (
        f"add the ID to {env_var} in {hermes_home}/.env and restart the gateway"
        if env_var else "add the ID to this platform's allowed-users list and restart the gateway"
    )
    return (
        f"Dropped a message from unrecognized {platform_name} user {who}. If that is you or someone "
        f"you trust, {allowlist}; or set `unauthorized_dm_behavior: pair` for {platform_name} in "
        f"{hermes_home}/config.yaml so unknown senders receive a pairing code you can approve with "
        f"`hermes pairing approve {platform_name} <code>`."
    )


class UnauthorizedOwnerNotifier:
    """Tells the owner's home channel about the first drop of each unrecognized DM sender.

    One notice per (platform, user_id) per gateway process: the first drop is the useful signal (an
    owner who typo'd their own ID); repeats would only let a stranger spam the home channel.
    """

    def __init__(self, max_seen: int = _MAX_SEEN_SENDERS) -> None:
        self._seen: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._max_seen = max(1, int(max_seen))

    def first_time(self, platform_name: str, user_id: str) -> bool:
        key = (platform_name, str(user_id))
        if key in self._seen:
            self._seen.move_to_end(key)
            return False
        self._seen[key] = None
        while len(self._seen) > self._max_seen:
            self._seen.popitem(last=False)
        return True

    async def notify(self, runner, source, hint: str) -> None:
        """Best-effort post to the source platform's home channel; silent when none is configured."""
        for platform, _cfg, home, transport in runner._home_channel_transports():
            if platform != source.platform:
                continue
            if str(home.chat_id) == str(source.chat_id):
                # The stranger's DM *is* the home channel (misconfiguration); posting there would
                # answer the unauthorized user, which the ignore behaviour exists to prevent.
                return
            await runner._send_home_channel_message(
                platform, home, transport, f"⚠️ {hint}", "unauthorized-sender notice failed for %s:%s: %s",
            )
            return
