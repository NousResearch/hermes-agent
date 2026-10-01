"""Pure provider endpoint declaration precedence; never reads process environment."""
from __future__ import annotations

from collections.abc import Mapping


def declared_endpoint_override(
    *, base_url_env_var: str = "", explicit: str = "", configured: str = "",
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve only declared non-secret endpoints, never infer variable roles."""
    for value in (
        explicit, configured,
        (environment or {}).get(base_url_env_var, "") if base_url_env_var else "",
    ):
        candidate = str(value or "").strip().rstrip("/")
        if candidate:
            return candidate
    return ""
