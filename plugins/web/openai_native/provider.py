"""OpenAI native web search — declares the Responses API server-side ``web_search`` built-in.

Config: ``web.search_backend: openai-native`` (or ``web.backend``).
Auth: openai-codex OAuth (``hermes auth --provider openai-codex``); no API key of its own.

Unlike every other provider here, this one never executes a search itself. Selecting it
tells the Codex Responses transport to declare the provider-executed ``web_search`` tool
(``{"type": "web_search"}``) in place of the client-side ``web_search`` function, so the
model drives search server-side. The transport performs that swap; this class exists so
``web.search_backend`` has a real provider name to point at.

Auth gating lives here rather than in the transport because the transport must stay
reachable for a user who configured this backend but has not signed in yet — they get the
clear "sign in" error from :meth:`search` rather than a silently different backend.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, search_fail

_UNSUPPORTED_MSG = (
    "openai-native declares OpenAI's server-side web_search tool; it cannot run as a "
    "client-side search and requires the Codex Responses transport (provider "
    "openai-codex). For client-side search use firecrawl (default) or another backend."
)


def _dget(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else None


def has_codex_credentials() -> bool:
    """Cheap probe: True when openai-codex OAuth tokens are *likely* usable.

    Mirrors ``tools/xai_http.has_xai_credentials`` — deliberately avoids
    ``resolve_codex_runtime_credentials`` (disk locks, OAuth network refresh), because
    this runs on every ``hermes tools`` repaint. Checks, fast-to-slow:
    ``providers.openai-codex.tokens.access_token`` in ``auth.json``, then any
    ``credential_pool.openai-codex`` entry carrying an ``access_token`` (pool-only
    multi-account grants never write the providers singleton). Returns False on any
    exception so a corrupted auth store cannot block other availability scans.
    """
    try:
        from hermes_constants import get_hermes_home

        auth_path = get_hermes_home() / "auth.json"
        if not auth_path.exists():
            return False
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        tokens = _dget(_dget(_dget(store, "providers"), "openai-codex"), "tokens")
        if str(_dget(tokens, "access_token") or "").strip():
            return True
        entries = _dget(_dget(store, "credential_pool"), "openai-codex")
        return isinstance(entries, list) and any(
            isinstance(e, dict) and str(e.get("access_token", "") or "").strip() for e in entries
        )
    except Exception:  # noqa: BLE001 — availability must never raise
        return False


class OpenAINativeWebSearchProvider(BaseWebSearchProvider):
    """Marker provider: the Codex Responses transport swaps the client ``web_search``
    function for the server-executed built-in when this backend is active."""

    NAME = "openai-native"
    DISPLAY_NAME = "OpenAI Native Web Search (Codex Responses)"

    def is_available(self) -> bool:
        return has_codex_credentials()

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Never called on a successful native turn — the transport replaces the tool
        before the request goes out. Reached only when the active transport cannot host
        the built-in, so fail loudly instead of returning an empty result set."""
        return search_fail(_UNSUPPORTED_MSG)

    def get_setup_schema(self) -> Dict[str, Any]:
        from plugins.web._common import setup_schema

        return setup_schema(
            self.DISPLAY_NAME,
            "native",
            "由模型服务端执行搜索（需 Codex Responses transport + openai-codex 登录）；仅搜索，提取仍用其他后端",
            "",
        )
