"""Profile-scoped watermarks; Codex owns when and how to compact its live thread."""
from __future__ import annotations


def native_auto_compact_limit(config: dict) -> int | None:
    compression = config.get("compression") or {}
    if compression.get("codex_app_server_auto", "native") != "native":
        return None
    limit = compression.get("codex_auto_compact_token_limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("compression.codex_auto_compact_token_limit must be a positive integer or null")
    return limit
