"""Typed DM/group scope for context construction.

``SessionSource.chat_type`` arrives as a free-form platform string; everything that assembles
or retrieves context needs ONE decision from it: does this conversation expose DM-scoped
surfaces (the owner's MEMORY.md/USER.md, the external memory provider, cross-chat session
search) or not. ``chat_scope()`` is that decision — the only place group-typed chats are
enumerated — so a new platform chat type is a one-line change, and an unknown type degrades
to DM (the local/CLI default) rather than guessing.

Scope consumers: ``agent_init._init_memory`` (memory surfaces), ``inline_tool_executors``
(memory/session_search tool gates), ``session_search`` (result visibility), and the system
prompt's explicit ``chat_scope=group`` marker.
"""

from typing import Literal, Optional

CHAT_SCOPE_DM: Literal["dm"] = "dm"
CHAT_SCOPE_GROUP: Literal["group"] = "group"

# Chat types that are never 1-on-1 with the owner. ``thread`` is absent on purpose: in this
# codebase it labels the owner's private DM topics, not multi-user chats. A future group-typed
# value lands here; an unknown value fails closed to DM, which only ever *under*-scopes memory
# (a group without group metadata) and never leaks DM content into a group.
GROUP_CHAT_TYPES = frozenset({"group", "supergroup", "forum", "channel"})


def chat_scope(chat_type: Optional[str]) -> Literal["dm", "group"]:
    """Map a platform ``chat_type`` to the typed scope used for context construction."""
    return CHAT_SCOPE_GROUP if str(chat_type or "").strip().lower() in GROUP_CHAT_TYPES else CHAT_SCOPE_DM
