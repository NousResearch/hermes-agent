"""Exa web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
official Exa SDK (``exa-py``) which is lazy-loaded via
:func:`tools.lazy_deps.ensure` so that cold-start CLI users don't pay the
SDK import cost when Exa isn't configured.

Config keys this provider responds to::

    web:
      search_backend: "exa"      # explicit per-capability
      extract_backend: "exa"     # explicit per-capability
      backend: "exa"             # shared fallback for both

Env var::

    EXA_API_KEY=...    # https://exa.ai (paid tier; free trial available)

The previous in-tree implementation lived at
``tools.web_tools._exa_search`` / ``_exa_extract``; this file is the
canonical replacement. Behavior is bit-for-bit identical aside from the
ABC method-name change.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "exa"
_ROTATE_STATUSES = frozenset({401, 403, 429})
# Anchored to HTTP/status so trace IDs containing 401/403/429 don't false-positive.
_STATUS_RE = re.compile(r"\b(?:HTTP|status[a-z_ ]*)\W{0,4}(401|403|429)\b", re.IGNORECASE)

T = TypeVar("T")

# Module-level note: the canonical ``_exa_client`` cache slot lives on
# :mod:`tools.web_tools` so tests that do ``tools.web_tools._exa_client =
# None`` between cases see fresh state. The plugin reads/writes through
# that public module (see :func:`_get_exa_client`).


def _load_exa_pool() -> Optional[CredentialPool]:
    """Return the Exa credential pool, or ``None`` if loading fails."""
    try:
        return load_pool(_POOL_PROVIDER)
    except Exception as exc:  # noqa: BLE001 — pool failures must not break web_search
        logger.warning("exa: failed to load credential pool: %s", exc)
        return None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _resolve_exa_key() -> Tuple[Optional[str], Optional[CredentialPool], Optional[PooledCredential]]:
    """Resolve an Exa API key (pool-first, env fallback). ``pool``/``entry`` are ``None`` for env."""
    pool = _load_exa_pool()
    if pool is not None and pool.has_credentials():
        entry = pool.select()
        if entry is not None:
            api_key = _pool_runtime_api_key(entry)
            if api_key:
                return api_key, pool, entry

    env_key = os.getenv("EXA_API_KEY", "").strip()
    return (env_key or None), None, None


def _pool_has_entries() -> bool:
    """Cheap probe for ``is_available()`` — no seeding/persisting."""
    try:
        from hermes_cli.auth import read_credential_pool

        entries = read_credential_pool(_POOL_PROVIDER)
    except Exception:
        return False
    return bool(entries)


def _build_exa_client(
    api_key: str,
    pool: Optional[CredentialPool] = None,
    entry: Optional[PooledCredential] = None,
) -> Any:
    """Build, cache, and return an Exa SDK client; stash pool state for rotation."""
    import tools.web_tools as _wt

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.exa", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — lazy_deps surfaces install hints
        raise ImportError(str(exc))

    from exa_py import Exa  # noqa: WPS433 — deliberately lazy

    client = Exa(api_key=api_key)
    client.headers["x-exa-integration"] = "hermes-agent"
    client._hermes_pool_state = (
        (pool, entry, api_key) if pool is not None and entry is not None else None
    )
    _wt._exa_client = client
    return client


def _get_exa_client() -> Any:
    """Return a cached Exa SDK client (pool-first, env fallback). Raises ``ValueError`` if neither."""
    import tools.web_tools as _wt

    cached = getattr(_wt, "_exa_client", None)
    if cached is not None:
        return cached

    api_key, pool, entry = _resolve_exa_key()
    if not api_key:
        raise ValueError(
            "EXA_API_KEY environment variable not set. "
            "Get your API key at https://exa.ai"
        )

    return _build_exa_client(api_key, pool, entry)


def _exa_error_status(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an Exa SDK exception."""
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    match = _STATUS_RE.search(str(exc))
    if match:
        return int(match.group(1))
    return None


def _call_with_rotation(fn: Callable[[Any], T]) -> T:
    """Invoke ``fn(client)``; on 401/403/429 mark the cached client's entry exhausted and retry once."""
    import tools.web_tools as _wt

    cached = getattr(_wt, "_exa_client", None)
    if cached is not None:
        client = cached
        state = getattr(client, "_hermes_pool_state", None)
        if state is not None:
            pool, entry, api_key = state
        else:
            pool, entry, api_key = None, None, str(getattr(client, "api_key", "") or "")
    else:
        api_key, pool, entry = _resolve_exa_key()
        if not api_key:
            raise ValueError(
                "EXA_API_KEY environment variable not set. "
                "Get your API key at https://exa.ai"
            )
        client = _build_exa_client(api_key, pool, entry)

    for attempt in (1, 2):
        try:
            return fn(client)
        except (ValueError, ImportError):
            raise
        except Exception as exc:  # noqa: BLE001 — Exa SDK error class varies
            status = _exa_error_status(exc)
            if (
                attempt == 1
                and pool is not None
                and entry is not None
                and status in _ROTATE_STATUSES
            ):
                rotated = pool.mark_exhausted_and_rotate(
                    status_code=status,
                    error_context={"message": str(exc), "status_code": status},
                )
                if rotated is not None:
                    new_key = _pool_runtime_api_key(rotated)
                    if new_key and new_key != api_key:
                        logger.info(
                            "exa: rotating credential after HTTP %s and retrying", status
                        )
                        entry = rotated
                        api_key = new_key
                        client = _build_exa_client(api_key, pool, entry)
                        continue
            raise

    raise RuntimeError("exa: _call_with_rotation exited retry loop unexpectedly")


class ExaWebSearchProvider(WebSearchProvider):
    """Exa search + extract provider.

    Both methods are sync — Exa's SDK is sync-only. The web_extract_tool
    dispatcher wraps sync extracts via ``asyncio.to_thread`` when it
    needs to keep the event loop responsive.
    """

    @property
    def name(self) -> str:
        return "exa"

    @property
    def display_name(self) -> str:
        return "Exa"

    def is_available(self) -> bool:
        """Return True when an Exa credential is available via env or pool. Must stay cheap."""
        if os.getenv("EXA_API_KEY", "").strip():
            return True
        return _pool_has_entries()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an Exa search.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure (incl.
        missing API key and SDK install errors).
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = _call_with_rotation(
                lambda client: client.search(
                    query,
                    num_results=limit,
                    contents={"highlights": True},
                )
            )

            web_results = []
            for i, result in enumerate(response.results or []):
                highlights = result.highlights or []
                web_results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "",
                        "description": " ".join(highlights) if highlights else "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            # Raised by _get_exa_client when EXA_API_KEY missing
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {"success": False, "error": f"Exa SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Exa search error: %s", exc)
            return {"success": False, "error": f"Exa search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Exa.

        Returns a list of result dicts shaped for the legacy LLM
        post-processing pipeline. On per-URL or whole-batch failure,
        results carry an ``error`` field rather than raising.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Exa extract: %d URL(s)", len(urls))
            response = _call_with_rotation(
                lambda client: client.get_contents(urls, text=True)
            )

            results: List[Dict[str, Any]] = []
            for result in response.results or []:
                content = result.text or ""
                url = result.url or ""
                title = result.title or ""
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url, "title": title},
                    }
                )
            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Exa SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Exa extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Exa",
            "badge": "paid",
            "tag": "Semantic + neural web search with content extraction.",
            "env_vars": [
                {
                    "key": "EXA_API_KEY",
                    "prompt": "Exa API key",
                    "url": "https://exa.ai",
                },
            ],
        }
