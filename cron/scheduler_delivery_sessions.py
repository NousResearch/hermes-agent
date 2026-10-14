"""Create reply-facing sessions for continuable scheduled deliveries."""

from __future__ import annotations

from typing import Optional


def _seed_cron_session(
    job: dict, adapter, platform_name: str, chat_id: str, text: str, *, thread_id: Optional[str],
    chat_type: str, user_id: Optional[str], user_name: Optional[str] = None,
    chat_name: Optional[str], scope_id: Optional[str], discord_keys_on_thread: bool = False, account_id: Optional[str] = None,
) -> bool:
    """Create the session row (so the mirror has a target) and mirror the brief as a USER turn.
    The seeded key must equal the reply's ``build_session_key``: chat_type, user_id, thread_id and
    scope_id (Slack team id) are all part of it, so callers pass exactly what the reply carries."""
    from cron.scheduler_delivery import _cron_mirror_message
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.mirror import mirror_to_session
    seeded_session_id: Optional[str] = None
    session_store = getattr(adapter, "_session_store", None)
    if session_store is not None:
        try:
            platform_enum = Platform(platform_name.lower())
        except (ValueError, KeyError):
            platform_enum = None
        if platform_enum is not None:
            # Discord keys in-thread messages with chat_id == thread_id; Slack/Telegram use the
            # parent channel.
            seed_chat_id = (
                str(thread_id)
                if discord_keys_on_thread and platform_enum == Platform.DISCORD
                else str(chat_id)
            )
            dest_source = SessionSource(
                platform=platform_enum, chat_id=seed_chat_id, chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id, user_name=user_name, thread_id=thread_id,
                scope_id=str(scope_id) if scope_id else None,
                account_id=(account_id or getattr(adapter, "_default_account", None)
                            or getattr(adapter, "_account_id", None)) if platform_enum == Platform.WEIXIN else None)
            # Create the row and pass its exact id to the mirror — origin-heuristic rediscovery
            # bails on populated chats.
            _entry = session_store.get_or_create_session(dest_source)
            seeded_session_id = getattr(_entry, "session_id", None)
    return mirror_to_session(
        platform_name, str(chat_id), _cron_mirror_message(job, text),
        source_label="cron", thread_id=thread_id, user_id=user_id, role="user",
        session_id=seeded_session_id,
    )
