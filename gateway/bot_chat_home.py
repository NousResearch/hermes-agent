"""Bot Chat as a profile's primary home (``gateway.home: bot-chat``).

A Desktop or Webapp user has no messaging chat to ``/sethome``. With this setting the profile's
canonical Bot Chat takes over what a platform home channel receives: an origin-less cron result
(``cron/scheduler_delivery.py``, as a Bot Chat turn) and the gateway's own notices (here, as history
notes that cost no model call). An explicit ``deliver=<platform>`` keeps its platform home channel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from gateway.config import HOME_BOT_CHAT, HomeChannel, Platform, PlatformConfig, persist_primary_home

logger = logging.getLogger(__name__)

# Bot Chat is the TUI/Desktop transcript: its warning policy is display.platforms.tui, as for cron.
NOTICE_POLICY_PLATFORM = "tui"
_NOTE_PRODUCER = "gateway_notice"


def bot_chat_is_home(config) -> bool:
    return getattr(config, "home", None) == HOME_BOT_CHAT


def is_bot_chat_home(home) -> bool:
    """*home* is the stand-in :func:`home_entry` yields for a Bot Chat home."""
    return getattr(home, "platform", None) == Platform.LOCAL and getattr(home, "chat_id", None) == HOME_BOT_CHAT


def notice_platform(platform, home):
    """The warning-policy platform for a notice to *home*."""
    return NOTICE_POLICY_PLATFORM if is_bot_chat_home(home) else platform


@dataclass(frozen=True)
class BotChatHomeTransport:
    """The send side of a Bot Chat home: notes into *home*'s Bot Chat, through its live authority."""

    home: Path
    is_relay = False
    adapter = None

    async def send(self, text: str) -> bool:
        return await post_note(self.home, text)


def home_platform_config() -> PlatformConfig:
    return PlatformConfig(enabled=True, home_channel=HomeChannel(
        platform=Platform.LOCAL, chat_id=HOME_BOT_CHAT, name="Bot Chat"))


def home_entry(config, home: Path):
    """``(platform, platform_cfg, home_channel, transport)`` when *config*'s home is its Bot Chat, else None."""
    if not bot_chat_is_home(config):
        return None
    platform_cfg = home_platform_config()
    return Platform.LOCAL, platform_cfg, platform_cfg.home_channel, BotChatHomeTransport(Path(home))


async def post_note(home: Path, text: str) -> bool:
    """Admit *text* as a note into *home*'s Bot Chat (created when missing). False when it was not
    admitted: no authority serves *home* in this process, or the Bot Chat moved to a messaging lane."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_cron import owner_for_home
    owner = owner_for_home(home)
    if owner is None:
        logger.warning("Bot Chat home notice skipped: no session authority serves %s", home)
        return False
    # Minting the Bot Chat builds its launch policy, which reads the profile's own secrets.
    async with _async_profile_runtime_scope(home):
        return await _admit_note(owner[0], text)


async def _admit_note(authority, text: str) -> bool:
    from gateway.owner_identity import host_owner_subject
    from gateway.session_bot import _target
    from gateway.session_contract import SessionActor
    from gateway.session_cron import _admit_local_delivery
    from hermes_state_runtime import RuntimeStoreError
    actor = SessionActor(host_owner_subject(), authority.profile_id,
                         frozenset({"session:create", "session:submit"}), "gateway-home-notice")
    try:
        ref, live, _entry = await _target(authority, actor)
        if live.source.platform != Platform.LOCAL:
            # Handed off (/handoff): the note would be a turn on that messaging lane.
            logger.info("Bot Chat home notice skipped: the Bot Chat continues on %s", live.source.platform.value)
            return False
        return await _admit_local_delivery(authority, ref.session_id, text, {}, producer=_NOTE_PRODUCER)
    except RuntimeStoreError as exc:
        logger.warning("Bot Chat home notice not admitted: %s", exc)
        return False


def scoped_config(runner):
    """The active profile scope's live ``GatewayConfig`` on *runner* (the launch profile's by default)."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home().resolve()
    for profile, served in (getattr(runner, "_served_profile_homes", None) or {}).items():
        if Path(served).resolve() == home:
            return (getattr(runner, "_profile_configs", None) or {}).get(profile)
    return getattr(runner, "config", None)


def set_home(runner, value) -> None:
    """Persist ``gateway.home`` for the active profile and apply it to the running gateway."""
    persist_primary_home(value)
    config = scoped_config(runner)
    if config is not None:
        config.home = value
