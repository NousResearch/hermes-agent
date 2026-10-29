"""The public, non-secret record of one profile: ``hermes profile list --json`` and the dashboard's
``/api/profiles`` serve this one shape, so automation never parses the human table.

Credential values, SOUL.md, memory and other profile contents are never included.
"""

from __future__ import annotations

import functools
from typing import Any


def profile_info_to_dict(info) -> dict[str, Any]:
    attr = functools.partial(getattr, info)
    return {
        "name": attr("name", ""), "path": str(attr("path", "")),
        "is_default": bool(attr("is_default", False)),
        "model": attr("model", None), "provider": attr("provider", None),
        "has_env": bool(attr("has_env", False)),
        "skill_count": int(attr("skill_count", 0) or 0),
        "gateway_running": bool(attr("gateway_running", False)),
        "description": attr("description", "") or "",
        "description_auto": bool(attr("description_auto", False)),
        "display_name": attr("display_name", "") or "",
        "bot_title": attr("bot_title", "") or "",
        "distribution_name": attr("distribution_name", None),
        "distribution_version": attr("distribution_version", None),
        "distribution_source": attr("distribution_source", None),
        "has_alias": attr("alias_path", None) is not None}
