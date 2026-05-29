"""Shared fixtures for Feishu adapter tests (admission, group policy, dispatch)."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Optional


def make_sender(sender_type: str = "user", open_id: str = "ou_human",
                user_id: Optional[str] = None, union_id: Optional[str] = None) -> Any:
    return SimpleNamespace(
        sender_type=sender_type,
        sender_id=SimpleNamespace(open_id=open_id, user_id=user_id, union_id=union_id),
    )


def make_message(message_id: str = "om_xxx", chat_type: str = "p2p",
                 chat_id: str = "oc_1", mentions: Optional[list] = None) -> Any:
    return SimpleNamespace(
        message_id=message_id,
        chat_type=chat_type,
        chat_id=chat_id,
        mentions=mentions,
        content="",
        message_type="text",
    )


def make_adapter_skeleton(
    *,
    bot_open_id: str = "ou_me",
    bot_user_id: str = "",
    allow_bots: str = "none",
    require_mention: bool = True,
    group_policy: str = "allowlist",
) -> Any:
    from gateway.platforms.feishu import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._bot_open_id = bot_open_id
    adapter._bot_user_id = bot_user_id
    adapter._bot_name = ""
    adapter._app_id = ""
    adapter._admins = set()
    adapter._group_rules = {}
    adapter._group_policy = group_policy
    adapter._default_group_policy = group_policy
    adapter._allowed_group_users = frozenset()
    adapter._allow_bots = allow_bots
    adapter._require_mention = require_mention
    return adapter


def install_dedup_state(adapter: Any, seen: Optional[dict] = None) -> None:
    adapter._seen_message_ids = dict(seen) if seen else {}
    adapter._seen_message_order = list((seen or {}).keys())
    adapter._dedup_cache_size = 100
    adapter._dedup_lock = threading.Lock()
    adapter._dedup_state_path = None
    adapter._persist_seen_message_ids = lambda: None


def stub_mention(adapter: Any, mentions_self: bool) -> None:
    adapter._mentions_self = lambda _message: mentions_self
