"""Card-click operator authorization and chat-type mapping for ``FeishuAdapter``.

Who may answer an approval / update-prompt card (Feishu allowlist and admins first, else the
gateway authorization inbound messages get, tried for every operator id form), the shared
operator/chat re-checks both click lanes run, and the chat_mode -> dm/group/forum mapping the
chat-info cache and card clicks share. A chat the cache never saw stays ``group``.

The adapter-taking functions read its ``_admins``, ``_allowed_group_users``, ``_chat_info_cache``,
``_get_cached_sender_name`` and ``BasePlatformAdapter._is_sender_authorized``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def is_operator_authorized(
    adapter: Any, open_id: str, *, user_id: str = "", union_id: str = "", chat_id: str = "",
) -> bool:
    """Return whether this card-action operator may answer gated prompts.

    Feishu admins / FEISHU_ALLOWED_USERS decide when set. With neither (setup's pairing-mode
    default) the clicker must pass the gateway authorization inbound messages get (pairing
    store, GATEWAY_ALLOWED_USERS, ...), so a chat member who may not use the bot can't answer.
    """
    normalized = str(open_id or "").strip()
    if not normalized:
        return False
    allowed_ids = set(adapter._admins) | set(adapter._allowed_group_users)
    if allowed_ids:
        # Lists may name a user by any id intake accepts (_sender_identity).
        operator_ids = {normalized, str(user_id or "").strip(), str(union_id or "").strip()} - {""}
        return "*" in allowed_ids or bool(operator_ids & allowed_ids)
    # SessionSource.user_id is the tenant user_id, else open_id (_resolve_sender_profile).
    principal = str(user_id or "").strip() or normalized
    # Feishu chat ids don't say DM vs group; "group" is the check's own default. Resolve the
    # real type from the chat-info cache the message lane warms (a card only exists because a
    # message reached that chat), so a DM click follows DM admission instead of the group
    # allowlist; cache misses stay "group" (fail-closed).
    chat_type = cached_chat_type(adapter._chat_info_cache, chat_id)
    chat_ref = str(chat_id or "") or None
    # One human, three id forms (open_id/user_id/union_id): pairing records one form while the
    # event may present another (card operators carry the tenant user_id; pairing recorded the
    # open_id). Try each, primary first — the same any-form identity the allowlist branch
    # applies above.
    verdict = None
    for candidate in dict.fromkeys(
        x for x in (principal, str(union_id or "").strip(), str(open_id or "").strip()) if x
    ):
        verdict = adapter._is_sender_authorized(candidate, chat_type, chat_ref)
        if verdict:
            return True
    if verdict is None:
        logger.debug("[Feishu] No gateway authorization check for card operator %s; denying", normalized)
        return False
    return verdict


def validate_card_action(
    adapter: Any, *, event: Any, state: dict[str, str], label: str, ident: Any,
) -> Optional[tuple[str, str, str, str, str]]:
    """Shared operator/chat checks for approval + update-prompt clicks: ``(open_id, callback_chat_id,
    user_name, user_id, union_id)`` or None (already logged)."""
    operator = getattr(event, "operator", None)
    open_id = str(getattr(operator, "open_id", "") or "")
    user_id = str(getattr(operator, "user_id", "") or "")
    union_id = str(getattr(operator, "union_id", "") or "")
    callback_chat_id = str(getattr(getattr(event, "context", None), "open_chat_id", "") or "")
    expected_chat_id = str(state.get("chat_id", "") or "")
    if not is_operator_authorized(
            adapter, open_id, user_id=user_id, union_id=union_id, chat_id=expected_chat_id or callback_chat_id):
        logger.warning("[Feishu] Unauthorized %s click by %s", label, open_id or "<unknown>")
        return None
    if callback_chat_id and expected_chat_id and callback_chat_id != expected_chat_id:
        logger.warning("[Feishu] %s callback chat mismatch for %s (expected=%s, got=%s)",
                       label.capitalize(), ident, expected_chat_id, callback_chat_id)
        return None
    return open_id, callback_chat_id, adapter._get_cached_sender_name(open_id) or open_id, user_id, union_id


def pop_validated_prompt_state(
    adapter: Any, *, states: dict[int, dict[str, str]], ident: Any, label: str, open_id: str, chat_id: str,
    unauthorized_fmt: str, operator_repr: str, user_id: str = "", union_id: str = "",
) -> Optional[dict[str, str]]:
    """Re-validate on the loop thread (state may have changed since the callback) and pop."""
    state = states.get(ident)
    if not state:
        logger.debug("[Feishu] %s %s already resolved or unknown", label, ident)
        return None
    if not is_operator_authorized(
        adapter, open_id, user_id=user_id, union_id=union_id, chat_id=str(state.get("chat_id", "") or "") or chat_id,
    ):
        logger.warning(unauthorized_fmt, operator_repr, ident)
        return None
    expected_chat_id = str(state.get("chat_id", "") or "")
    if expected_chat_id and chat_id and expected_chat_id != chat_id:
        logger.warning("[Feishu] %s %s chat mismatch (expected=%s, got=%s)", label, ident, expected_chat_id, chat_id)
        return None
    state = states.pop(ident, None)
    if not state:
        logger.debug("[Feishu] %s %s already resolved while validating callback", label, ident)
    return state


def map_chat_type(raw_chat_type: str) -> str:
    normalized = (raw_chat_type or "").strip().lower()
    if normalized == "p2p":
        return "dm"
    if any(marker in normalized for marker in ("topic", "thread", "forum")):
        return "forum"
    return "group" if normalized == "group" else "dm"


def resolve_source_chat_type(*, chat_info: dict[str, Any], event_chat_type: str) -> str:
    resolved = str(chat_info.get("type") or "").strip().lower()
    if resolved in {"group", "forum"}:
        return resolved
    return "dm" if event_chat_type == "p2p" else "group"


def cached_chat_type(chat_info_cache: dict[str, dict[str, Any]], chat_id: str) -> str:
    """dm/group/forum for a card click, derived from the cache's raw_mode (conversation mode
    p2p/group/topic — NOT raw_type, which is visibility private/public); "group" when unknown."""
    info = chat_info_cache.get(str(chat_id or "").strip()) or {}
    raw_mode = str(info.get("raw_mode") or "").strip().lower()
    if raw_mode == "p2p":
        return "dm"
    if any(m in raw_mode for m in ("topic", "thread", "forum")):
        return "forum"
    return "group"
