"""Per-chat relay routing metadata shared by outbound lanes."""

from typing import Any, Optional


def with_cached_scope(
    adapter, chat_id: str, metadata: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Outbound metadata carrying the tenant discriminators (see _capture_scope).
    Both are attached when known and not already set; the connector tries scope_id
    first and only falls back to user_id on a route miss, so carrying both never
    overrides routing-table resolution."""
    meta: dict[str, Any] = dict(metadata or {})
    # ``getattr``: relay tests build bare adapters via ``__new__`` without ``__init__``.
    for key, cache in (
        ("scope_id", adapter._scope_by_chat),
        ("user_id", adapter._dm_user_by_chat),
        ("profile", getattr(adapter, "_profile_by_chat", {})),
    ):
        if not meta.get(key):
            value = cache.get(str(chat_id))
            if value:
                meta[key] = value
    return meta
