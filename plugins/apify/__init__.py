"""Apify Actor execution plugin — bundled, auto-loaded.

Registers three tools (``apify_discover``, ``apify_start``, ``apify_collect``)
into the ``apify`` toolset. Each tool is gated by ``_check_token()`` — when the
user has not set ``APIFY_API_TOKEN`` the tools stay registered (so they appear
in ``hermes tools``) but the runtime check prevents dispatch.

Why a plugin instead of top-level ``tools/`` files?

- ``plugins/`` is where third-party service integrations live (see
  ``plugins/spotify/`` for the same pattern — optional SaaS, token-gated,
  default-off toolset). ``tools/`` is reserved for foundational capabilities
  (terminal, read_file, web_search, etc.).
- Bundled + ``kind: backend`` auto-loads on startup just like the Spotify
  plugin — no user opt-in needed, no ``plugins.enabled`` config.
- Keeps the three Apify tools out of ``_HERMES_CORE_TOOLS`` in ``toolsets.py``;
  the plugin loader registers them via ``ctx.register_tool()``.

The ``apify`` toolset is default-off (``_DEFAULT_OFF_TOOLSETS`` in
``hermes_cli/tools_config.py``) and the ``APIFY_API_TOKEN`` setup UX is wired
through ``hermes tools`` (``TOOL_CATEGORIES``) and ``OPTIONAL_ENV_VARS``.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from plugins.apify.tools import (
    _COLLECT_SCHEMA,
    _DISCOVER_SCHEMA,
    _START_SCHEMA,
    _check_token,
    _collect_handler,
    _discover_handler,
    _start_handler,
)


async def _collect_handler_str(args: Dict[str, Any], **_kw: Any) -> str:
    return json.dumps(await _collect_handler(args), default=str)


def register(ctx) -> None:
    """Register the Apify Actor tools. Called once by the plugin loader."""
    ctx.register_tool(
        name="apify_discover",
        toolset="apify",
        schema=_DISCOVER_SCHEMA,
        handler=lambda args, **kw: json.dumps(_discover_handler(args), default=str),
        check_fn=_check_token,
        requires_env=["APIFY_API_TOKEN"],
        emoji="🔍",
    )
    ctx.register_tool(
        name="apify_start",
        toolset="apify",
        schema=_START_SCHEMA,
        handler=lambda args, **kw: json.dumps(_start_handler(args), default=str),
        check_fn=_check_token,
        requires_env=["APIFY_API_TOKEN"],
        emoji="▶️",
    )
    ctx.register_tool(
        name="apify_collect",
        toolset="apify",
        schema=_COLLECT_SCHEMA,
        handler=_collect_handler_str,
        check_fn=_check_token,
        requires_env=["APIFY_API_TOKEN"],
        is_async=True,
        emoji="📦",
    )
