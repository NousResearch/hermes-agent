"""OpenAI native web search — search and extract through an openai-codex login.

Config: ``web.search_backend: openai-native`` / ``web.extract_backend: openai-native``
(or ``web.backend``). Auth: openai-codex OAuth (``hermes auth add openai-codex``); no API
key of its own.

Runs client-side against the Codex backend's standalone search endpoint
(``POST {codex base}/alpha/search``), the route the official Codex client uses for its
``web.run`` tool (``codex-rs/codex-api/src/endpoint/search.rs``). ``search_query`` commands
serve ``web_search``; ``open`` commands serve ``web_extract``.

An earlier version swapped the client ``web_search`` function for the hosted Responses
``{"type": "web_search"}`` built-in on the Codex transport. On the Codex backend that
built-in completes the search and then fails the turn with ``server_error`` as soon as the
answer starts (#135684), and the official client no longer declares it. Running the search
here also means the backend works whatever model the session uses.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, Optional

from plugins.web._common import (
    BaseWebSearchProvider, SEARCH_LIMIT_CAP, document, page_error, run_extract, run_search, search_ok, title_hit,
)

logger = logging.getLogger(__name__)

_SEARCH_PATH = "alpha/search"
# The endpoint requires ``model`` but does not validate it (probed 2026-10: an unknown id is
# accepted). Send the session's Codex model when there is one so the request matches what the
# official client sends; this default only covers non-Codex sessions.
_DEFAULT_MODEL = "gpt-5.6-sol"
_SEARCH_MAX_OUTPUT_TOKENS = 2500
_OPEN_MAX_OUTPUT_TOKENS = 8000
_TIMEOUT = 60.0
_MAX_ERROR_BODY_CHARS = 300

_NO_AUTH = (
    "No Codex/ChatGPT OAuth credentials available. Run "
    "`hermes auth add openai-codex` (or `hermes setup` → Codex) to sign in.")

# Codex citation markup in ``output``: cite<ref>; link refs read "<n>†<text>[†<domain>]".
_LINK_CITE = re.compile("cite\\d+†([^†]*)(?:†[^]*)?")
_ANY_CITE = re.compile("[^]*")
_LINE_PREFIX = re.compile(r"(?:^|\s)L\d+: ")
_TITLE_LINE = re.compile(r"^(.*) \((https?://\S+)\)$")


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
    except Exception:
        return False


def _read_codex_credential() -> tuple[Optional[str], Optional[str]]:
    """``(token, base_url)`` from one resolution, so the token only goes to the host its
    credential routes to (#121486) — same authority as the openai-codex image plugin."""
    from agent.auxiliary_client import _resolve_codex_credential_and_base

    token, base_url = _resolve_codex_credential_and_base()
    if isinstance(token, str) and token.strip():
        return token.strip(), base_url
    return None, None


def _search_model() -> str:
    from hermes_cli.config import load_config

    model = (load_config() or {}).get("model") or {}
    if model.get("provider") == "openai-codex" and str(model.get("default") or "").strip():
        return str(model["default"]).strip()
    return _DEFAULT_MODEL


def _error_summary(body: str) -> str:
    try:
        message = _dget(_dget(json.loads(body or ""), "error"), "message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:_MAX_ERROR_BODY_CHARS]
    except (TypeError, ValueError):
        pass
    return (body or "")[:_MAX_ERROR_BODY_CHARS]


def _post_search(commands: Dict[str, Any], max_output_tokens: int) -> Dict[str, Any]:
    """POST one search request; raises ValueError with a user-facing message on failure."""
    import httpx

    from agent.codex_headers import codex_cloudflare_headers

    token, base_url = _read_codex_credential()
    if not token or not base_url:
        raise ValueError(_NO_AUTH)
    base_url = base_url.strip().rstrip("/")
    body = {
        "id": str(uuid.uuid4()),
        "model": _search_model(),
        "commands": commands,
        "settings": {"external_web_access": True, "allowed_callers": ["direct"]},
        "max_output_tokens": max_output_tokens,
    }
    headers = codex_cloudflare_headers(token, base_url=base_url)
    headers["Authorization"] = f"Bearer {token}"
    response = httpx.post(f"{base_url}/{_SEARCH_PATH}", json=body, headers=headers, timeout=_TIMEOUT)
    if response.status_code >= 400:
        raise ValueError(
            f"Codex search returned HTTP {response.status_code}: {_error_summary(response.text)}")
    return response.json()


def _page(output: str, url: str) -> Dict[str, Any]:
    """``open`` output → extract document: first line is ``Title (url)``, the second a crawl
    header, the rest line-numbered text carrying citation markup."""
    lines = output.split("\n", 2)
    match = _TITLE_LINE.match(lines[0]) if lines else None
    title = match.group(1) if match else ""
    text = lines[2] if len(lines) > 2 else ""
    text = _ANY_CITE.sub("", _LINK_CITE.sub(r"\1", text))
    text = re.sub(r"\n{3,}", "\n\n", _LINE_PREFIX.sub("\n", text)).strip()
    return document(url, title, text)


class OpenAINativeWebSearchProvider(BaseWebSearchProvider):
    """Search + extract via the Codex backend's standalone search endpoint."""

    NAME = "openai-native"
    DISPLAY_NAME = "OpenAI Native Web Search (Codex)"

    def is_available(self) -> bool:
        return has_codex_credentials()

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            data = _post_search({"search_query": [{"q": query}]}, _SEARCH_MAX_OUTPUT_TOKENS)
            rows = [r for r in (_dget(data, "results") or []) if str(_dget(r, "url") or "").strip()]
            n = max(1, min(int(limit or 5), SEARCH_LIMIT_CAP))
            return search_ok([
                title_hit(str(r.get("title") or r.get("domain") or r["url"]), str(r["url"]),
                          _ANY_CITE.sub("", str(r.get("snippet") or "")), i + 1)
                for i, r in enumerate(rows[:n])
            ])

        return run_search("OpenAI native", logger, body)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        import httpx

        def body() -> list[dict[str, Any]]:
            pages = []
            for url in urls:  # one ``open`` per URL so a dead link fails alone
                try:
                    data = _post_search({"open": [{"ref_id": url}]}, _OPEN_MAX_OUTPUT_TOKENS)
                    pages.append(_page(str(_dget(data, "output") or ""), url))
                except (ValueError, httpx.HTTPError) as exc:
                    pages.append(page_error(url, str(exc)))
            return pages

        return run_extract("OpenAI native", logger, urls, body)

    def get_setup_schema(self) -> dict[str, Any]:
        from plugins.web._common import setup_schema

        return setup_schema(
            self.DISPLAY_NAME,
            "native",
            "Search and extract through your openai-codex login (the endpoint the Codex client uses); works with any chat model",
            "",
        )
