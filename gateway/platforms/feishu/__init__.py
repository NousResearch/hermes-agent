"""Feishu platform package.

Public surface (``from gateway.platforms.feishu import ...``) re-exports the
main adapter/settings/helper objects used by the gateway:

  - FeishuAdapter / FeishuAdapterSettings / FeishuGroupRule / FeishuMentionRef
    / JsonFileDedupStore / check_feishu_requirements
    / check_feishu_send_requirements / FEISHU_AVAILABLE
    / FEISHU_DOMAIN / LARK_DOMAIN / FEISHU_WEBHOOK_AVAILABLE
    / FEISHU_WEBSOCKET_AVAILABLE  ← gateway/platforms/feishu/adapter.py
  - qr_register / probe_bot                                 ← qr_register.py
  - normalize_feishu_message compatibility helper            ← compat.py
  - (sub-module direct imports)
        from gateway.platforms.feishu.dedup_store import JsonFileDedupStore
        from gateway.platforms.feishu.events_mapping import to_message_event
        from gateway.platforms.feishu.webhook_guard import start_webhook_server
        from gateway.platforms.feishu.comments import handle_drive_comment_event
        from gateway.platforms.feishu.approvals import _build_resolved_approval_card
"""
from gateway.platforms.feishu.adapter import (
    FeishuAdapter,
    FeishuAdapterSettings,
    FeishuGroupRule,
    FEISHU_AVAILABLE,
    FEISHU_DOMAIN,
    FEISHU_WEBHOOK_AVAILABLE,
    FEISHU_WEBSOCKET_AVAILABLE,
    LARK_DOMAIN,
    JsonFileDedupStore,
    _FEISHU_PROCESSING_REACTION_CACHE_SIZE,
    check_feishu_requirements,
    check_feishu_send_requirements,
)
from gateway.platforms.feishu.types import FeishuMentionRef
from gateway.platforms.feishu.events_mapping import (
    _build_mention_hint,
    _strip_edge_self_mentions,
)
from gateway.platforms.feishu.compat import (
    FeishuNormalizedMessage,
    FeishuPostMediaRef,
    normalize_feishu_message,
)
from gateway.platforms.feishu.qr_register import (
    qr_register,
    probe_bot,
)

__all__ = [
    "FeishuAdapter",
    "FeishuAdapterSettings",
    "FeishuGroupRule",
    "FeishuMentionRef",
    "FEISHU_AVAILABLE",
    "FEISHU_DOMAIN",
    "FEISHU_WEBHOOK_AVAILABLE",
    "FEISHU_WEBSOCKET_AVAILABLE",
    "LARK_DOMAIN",
    "JsonFileDedupStore",
    "FeishuNormalizedMessage",
    "FeishuPostMediaRef",
    "check_feishu_requirements",
    "check_feishu_send_requirements",
    "normalize_feishu_message",
    "qr_register",
    "probe_bot",
]
