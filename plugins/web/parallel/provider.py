"""Parallel.ai web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses two
distinct Parallel SDK clients:

- ``Parallel`` (sync, cached)                — for :meth:`search`
- ``AsyncParallel`` (async, request-scoped)  — for :meth:`extract`

This is the first plugin to exercise the **async-extract** code path in
the ABC: :meth:`extract` is declared ``async def``, and the dispatcher
in :func:`tools.web_tools.web_extract_tool` detects coroutines via
:func:`inspect.iscoroutinefunction` and awaits.

Config keys this provider responds to::

    web:
      search_backend: "parallel"      # explicit per-capability
      extract_backend: "parallel"     # explicit per-capability
      backend: "parallel"             # shared fallback
      # Optional: search mode (default "agentic"; also "fast" or "one-shot")
      # via the PARALLEL_SEARCH_MODE env var.

Env vars::

    PARALLEL_API_KEY=...             # https://parallel.ai (required)
    PARALLEL_SEARCH_MODE=agentic     # optional: agentic|fast|one-shot
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Module-level note: the canonical sync cache slot ``_parallel_client`` lives
# on :mod:`tools.web_tools` so tests that reset it between cases see fresh
# state. Async clients are deliberately request-scoped: httpx transports are
# bound to the event loop that first uses them and cannot be shared safely by
# the per-thread loops used for concurrent tool execution.


def _ensure_parallel_sdk_installed() -> None:
    """Trigger lazy install of the parallel SDK if it isn't present.

    Mirrors the lazy-deps pattern used by the legacy implementation.
    Swallows benign ImportError from the lazy_deps helper itself; if the
    SDK is genuinely missing the subsequent ``from parallel import ...``
    raises ImportError that the caller can handle.
    """
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.parallel", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — surface install hint as ImportError
        raise ImportError(str(exc))


def _get_sync_client() -> Any:
    """Lazy-load + cache the sync Parallel client.

    Cache lives on :mod:`tools.web_tools` (as ``_parallel_client``) so unit
    tests that reset that name between cases keep working.
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_parallel_client", None)
    if cached is not None:
        return cached

    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError(
            "PARALLEL_API_KEY environment variable not set. "
            "Get your API key at https://parallel.ai"
        )

    _ensure_parallel_sdk_installed()
    from parallel import Parallel  # noqa: WPS433 — deliberately lazy

    client = Parallel(api_key=api_key)
    _wt._parallel_client = client
    return client


def _get_async_client() -> Any:
    """Create an async Parallel client owned by the current extraction.

    The caller must close the client on the same event loop that uses it.
    Caching this process-wide lets concurrent tool workers race to publish
    loop-affine clients; losing clients can then be finalized from
    prompt_toolkit's loop after their worker loops have died.
    """
    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError(
            "PARALLEL_API_KEY environment variable not set. "
            "Get your API key at https://parallel.ai"
        )

    _ensure_parallel_sdk_installed()
    from parallel import AsyncParallel  # noqa: WPS433 — deliberately lazy

    return AsyncParallel(api_key=api_key)


def _reset_clients_for_tests() -> None:
    """Drop the cached sync client so tests can re-instantiate cleanly.

    The async client is request-scoped and therefore has no cache to reset.
    """
    import tools.web_tools as _wt

    _wt._parallel_client = None


# Backward-compatible aliases for the names that lived in tools.web_tools
# before the migration (matches existing tests + external callers).
_get_parallel_client = _get_sync_client
_get_async_parallel_client = _get_async_client


def _resolve_search_mode() -> str:
    """Return the validated PARALLEL_SEARCH_MODE value (default "agentic")."""
    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    if mode not in {"fast", "one-shot", "agentic"}:
        mode = "agentic"
    return mode


class ParallelWebSearchProvider(WebSearchProvider):
    """Parallel.ai search + async extract provider."""

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def display_name(self) -> str:
        return "Parallel"

    def is_available(self) -> bool:
        """Return True when ``PARALLEL_API_KEY`` is set to a non-empty value.

        Deliberately does NOT consider the keyless free tier — that would
        let the legacy preference walk route keyed users of lower-priority
        backends onto Parallel's anonymous tier. Keyless availability is a
        separate, last-resort signal (:meth:`is_keyless_available`).
        """
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("PARALLEL_API_KEY"))

    def is_keyless_available(self) -> bool:
        """Parallel serves anonymous free-tier calls via its public MCP endpoint.

        False when the user forced ``web.provider_tier.parallel: paid`` —
        an explicit paid selection must never silently resolve keyless.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("parallel") != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Parallel search (sync).

        Uses the ``beta.search`` endpoint with the configured mode
        (``PARALLEL_SEARCH_MODE`` env var, default "agentic"). Limit is
        capped at 20 server-side.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import search_with_failover, use_keyless

            if use_keyless("parallel", get_provider_env("PARALLEL_API_KEY")):
                # Keyless free tier — public MCP endpoint, no SDK needed.
                logger.info(
                    "Parallel keyless search: '%s' (limit=%d)", query, limit
                )
                return search_with_failover("parallel", query, limit)

            mode = _resolve_search_mode()
            logger.info(
                "Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit
            )
            response = _get_sync_client().beta.search(
                search_queries=[query],
                objective=query,
                mode=mode,
                max_results=min(limit, 20),
            )

            web_results = []
            for i, result in enumerate(response.results or []):
                excerpts = result.excerpts or []
                web_results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "",
                        "description": " ".join(excerpts) if excerpts else "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {
                "success": False,
                "error": f"Parallel SDK not installed: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel search error: %s", exc)
            return {"success": False, "error": f"Parallel search failed: {exc}"}

    async def extract(
        self, urls: List[str], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via the async SDK.

        Returns the legacy list-of-results shape that
        :func:`tools.web_tools.web_extract_tool` expects: one entry per
        successful URL plus one entry per failed URL with an ``error``
        field. Errors are not raised — they're returned as per-URL items.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import extract_with_failover, use_keyless

            if use_keyless("parallel", get_provider_env("PARALLEL_API_KEY")):
                # Keyless free tier — blocking HTTP, so hop off the loop.
                import asyncio

                logger.info("Parallel keyless extract: %d URL(s)", len(urls))
                return await asyncio.to_thread(
                    extract_with_failover, "parallel", list(urls)
                )

            logger.info("Parallel extract: %d URL(s)", len(urls))
            client = _get_async_client()
            try:
                response = await client.beta.extract(
                    urls=urls,
                    full_content=True,
                )
            finally:
                # AsyncParallel owns an httpx connection pool whose transports
                # are tied to this running loop. Drain it here, before the
                # concurrent-tool worker and its loop go out of scope.
                #
                # Never let a teardown failure destroy an otherwise successful
                # extraction: ``close()`` funnels into ``httpx.aclose()`` ->
                # ``transport.aclose()``, which can raise (a mid-shutdown TLS
                # error, or the very RuntimeError this cleanup exists to
                # prevent). Without this guard that exception propagates out
                # of ``finally``, past the response handling below, and the
                # outer ``except Exception`` rewrites every URL into an error
                # result even though the content was already fetched.
                #
                # Masked from the caller, but NOT from operators: a failed
                # close can leave partial resources behind, and an "Event loop
                # is closed" here would mean this ownership fix regressed, so
                # it must stay visible. ``except Exception`` deliberately lets
                # ``CancelledError`` (BaseException since 3.8) propagate so
                # cancellation semantics are preserved.
                try:
                    await client.close()
                except Exception as close_exc:  # noqa: BLE001 — cleanup is best-effort
                    logger.warning(
                        "Parallel async client close failed; preserving "
                        "extraction result: %s: %s",
                        type(close_exc).__name__,
                        close_exc,
                    )

            results: List[Dict[str, Any]] = []
            for result in response.results or []:
                content = result.full_content or ""
                if not content:
                    content = "\n\n".join(result.excerpts or [])
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

            for error in response.errors or []:
                results.append(
                    {
                        "url": error.url or "",
                        "title": "",
                        "content": "",
                        "error": error.content or error.error_type or "extraction failed",
                        "metadata": {"sourceURL": error.url or ""},
                    }
                )

            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Parallel · Free (keyless)",
            "badge": "free · no key",
            "tag": (
                "Objective-tuned search + page extraction on Parallel's "
                "anonymous free tier. Rate-limited under burst load."
            ),
            "env_vars": [],
            "web_tier": "free",
            "variants": [
                {
                    "name": "Parallel · Paid (API key)",
                    "badge": "paid",
                    "tag": (
                        "Objective-tuned search + parallel page extraction "
                        "via the Parallel SDK. Unthrottled, guaranteed service."
                    ),
                    "env_vars": [
                        {
                            "key": "PARALLEL_API_KEY",
                            "prompt": "Parallel API key",
                            "url": "https://parallel.ai",
                        },
                    ],
                    "web_tier": "paid",
                },
            ],
        }
