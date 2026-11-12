"""``/branch`` destination on thread-capable platforms (#66023).

On Discord, Telegram, Slack and Matrix a plain ``/branch`` opens a NEW sibling thread for the
clone and leaves the current chat bound to the original session, so the user can keep both paths
alive. ``/branch --here`` keeps the legacy in-place switch; platforms without threads (and the CLI)
always branch in place. Pure helpers only — the handler lives in ``slash_commands_session.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from gateway.config import Platform
from gateway.session import SessionSource

# Adapters that override ``BasePlatformAdapter.create_handoff_thread`` (the base returns None).
BRANCH_THREAD_PLATFORMS = frozenset({Platform.DISCORD, Platform.TELEGRAM, Platform.SLACK, Platform.MATRIX})
BRANCH_HERE_FLAG = "--here"


def parse_branch_args(raw: str) -> tuple[bool, str]:
    """``(stay_here, title)`` from the text after ``/branch``.

    Only a leading ``--here`` is a flag; everything else is the optional title. The CLI strips the
    flag through the same parser so ``/branch --here` never becomes a session titled ``--here``.
    """
    text = (raw or "").strip()
    head, _, rest = text.partition(" ")
    if head.lower() == BRANCH_HERE_FLAG:
        return True, rest.strip()
    return False, text


def branch_thread_parent(source: SessionSource) -> Optional[str]:
    """Chat that can host a sibling thread for *source*, or None when nothing can (Discord DMs,
    a Discord thread whose parent channel is unknown)."""
    if source.platform not in BRANCH_THREAD_PLATFORMS:
        return None
    if source.platform == Platform.DISCORD:
        if source.chat_type == "dm":
            return None
        # Inbound Discord threads carry ``chat_id == thread_id``; the sibling goes under the real
        # text channel, which only ``parent_chat_id`` names.
        if source.thread_id or source.chat_type == "thread":
            return str(source.parent_chat_id) if source.parent_chat_id else None
    # Telegram forum topics, Slack threads and Matrix threads all key on the parent chat/room id.
    return str(source.chat_id) if source.chat_id else None


def branch_dest_source(source: SessionSource, *, parent_id: str, thread_id: str, title: str) -> SessionSource:
    """The ``SessionSource`` a follow-up typed in the new thread will arrive on — the shape must
    match each adapter's inbound source or the clone is bound to a key nobody ever reads (the
    CLI→platform handoff in ``run_startup.py::_handoff_resolve_destination`` mirrors the same
    rules). Per-message fields are dropped; identity/scope fields travel with the copy."""
    common = dict(thread_id=str(thread_id), message_id=None, prospective_thread_id=None,
                  auto_thread_created=False, auto_thread_initial_name=None)
    if source.platform == Platform.DISCORD:
        # Discord keys an in-thread message on the thread's OWN id as chat_id.
        return dataclasses.replace(source, chat_id=str(thread_id), chat_name=title or source.chat_name,
                                   chat_type="thread", parent_chat_id=str(parent_id), **common)
    # Telegram (``group:<chat>:<topic>``, private-chat topics stay ``dm``), Slack (parent channel's
    # dm/group + workspace scope) and Matrix (room type) key a thread reply on the PARENT chat's type.
    chat_type = "group" if source.chat_type == "thread" else source.chat_type
    return dataclasses.replace(source, chat_id=str(parent_id), chat_type=chat_type, **common)


def format_thread_ref(platform: Optional[Platform], thread_id: str) -> str:
    """Clickable pointer where the platform has one (Discord ``<#id>`` mentions); id otherwise."""
    if platform == Platform.DISCORD:
        return f"<#{thread_id}>"
    return f"`{thread_id}`"
