"""Propagate a CLI session rename onto the bound Telegram DM topic.

``hermes sessions rename`` and the in-REPL ``sessions rename`` write only the session
database — unlike the gateway's ``/title`` lane (``_schedule_telegram_topic_title_rename``),
which also renames the Telegram topic via ``editForumTopic``. A user who renames a session
from the CLI (or asks the agent to) therefore sees the change land in Hermes but not in the
Telegram topic, and must rename the topic manually.

This module closes that gap: after a successful title write, mirror the title onto the
Telegram DM topic the session is bound to. Best-effort — a Telegram-side failure never fails
the rename — and it applies the same gating rules as the gateway lane so the two surfaces
cannot disagree:

- only sessions with a Telegram DM topic binding (``telegram_dm_topic_bindings``) are touched;
- the chat must still have Telegram topic mode enabled;
- the operator kill-switch ``platforms.telegram.extra.disable_topic_auto_rename`` is honoured;
- operator-declared topics (``platforms.telegram.extra.dm_topics``) are skipped: their fixed
  names are operator config, not session state.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("hermes_cli.sessions")

_TELEGRAM_TOPIC_NAME_MAX = 128


def _telegram_topic_name(title: str) -> str:
    """Bot API-safe forum topic name: 1-128 chars, whitespace collapsed (mirrors the
    gateway's ``_sanitize_telegram_topic_title`` truncation at 120 to keep room for
    multi-byte characters)."""
    cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
    return cleaned if len(cleaned) <= 120 else cleaned[:117].rstrip() + "..."


def _telegram_extra_config() -> dict:
    """``platforms.telegram.extra`` from the gateway config, or {} when unavailable."""
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        platform_cfg = (
            config.platforms.get(Platform.TELEGRAM)
            if getattr(config, "platforms", None)
            else None
        )
        return getattr(platform_cfg, "extra", None) or {}
    except Exception:
        logger.debug(
            "sessions rename: could not read gateway config for Telegram extras",
            exc_info=True,
        )
        return {}


def _operator_topic_names(extra: dict) -> set[str]:
    """Names of operator-declared DM topics (``extra.dm_topics``) — renames skip these."""
    names: set[str] = set()
    for chat_entry in extra.get("dm_topics") or ():
        if isinstance(chat_entry, dict):
            for topic in chat_entry.get("topics") or ():
                if isinstance(topic, dict) and topic.get("name"):
                    names.add(str(topic["name"]))
    return names


def _telegram_token() -> str | None:
    """The Telegram bot token from the environment / ``.env``, or None."""
    try:
        from hermes_cli.config import get_env_value

        token = (get_env_value("TELEGRAM_BOT_TOKEN") or "").strip()
        return token or None
    except Exception:
        logger.debug(
            "sessions rename: could not read the Telegram bot token", exc_info=True
        )
        return None


async def _rename_telegram_topic(chat_id: str, thread_id: str, name: str) -> None:
    """One-shot ``editForumTopic`` via the standalone, proxy-aware Bot factory."""
    from tools.send_message_senders import _telegram_bot

    bot = _telegram_bot(_telegram_token())
    try:
        try:
            await bot.initialize()
        except Exception:
            logger.debug(
                "sessions rename: Telegram bot initialize() failed", exc_info=True
            )
        await bot.edit_forum_topic(
            chat_id=int(chat_id), message_thread_id=int(thread_id), name=name
        )
    finally:
        try:
            await bot.shutdown()
        except Exception:
            logger.debug(
                "sessions rename: Telegram bot shutdown() failed", exc_info=True
            )


def propagate_session_title_to_telegram_topic(
    session_db, session_id: str, title: str
) -> bool:
    """Best-effort mirror of *title* onto the Telegram DM topic bound to *session_id*.

    True when a rename was pushed to Telegram; False when there is nothing to do or the
    gateway's rules say not to. Never raises: a Telegram-side failure is logged and the
    session rename it accompanies still stands.
    """
    if not session_id or not title:
        return False
    try:
        binding = session_db.get_telegram_topic_binding_by_session(
            session_id=str(session_id)
        )
    except Exception:
        logger.debug(
            "sessions rename: Telegram topic binding lookup failed", exc_info=True
        )
        return False
    if not binding:
        return False
    chat_id = str(binding.get("chat_id") or "")
    thread_id = str(binding.get("thread_id") or "")
    if not chat_id or not thread_id:
        return False
    try:
        topic_mode_on = session_db.is_telegram_topic_mode_enabled(
            chat_id=chat_id,
            user_id=str(binding.get("user_id") or ""),
            profile_name=str(binding.get("profile_name") or "default"),
        )
    except Exception:
        logger.debug("sessions rename: Telegram topic mode check failed", exc_info=True)
        return False
    if not topic_mode_on:
        return False
    extra = _telegram_extra_config()
    from utils import is_truthy_value

    if is_truthy_value(extra.get("disable_topic_auto_rename")):
        return False
    if _operator_topic_names(extra):
        # Operator-declared topics carry fixed names chosen in config.yaml; a session title
        # must not silently rewrite operator config (same rule as the gateway rename lane).
        return False
    if not _telegram_token():
        logger.debug("sessions rename: Telegram not configured; topic left as-is")
        return False

    from model_tools import _run_async

    try:
        _run_async(
            _rename_telegram_topic(chat_id, thread_id, _telegram_topic_name(title))
        )
    except Exception:
        logger.debug("sessions rename: Telegram topic rename failed", exc_info=True)
        return False
    logger.info(
        "sessions rename: Telegram topic %s in chat %s renamed to %r",
        thread_id,
        chat_id,
        title,
    )
    return True
