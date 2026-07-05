"""hc-401 (was patch_native_agent_web_tools.py) — narrow the ``web_extract``
tool's availability gate to backends that can actually extract.

What this replaces
==================
``tools/web_tools.py`` registers two tools — ``web_search`` and ``web_extract``
— both with ``check_fn=check_web_api_key``. ``check_web_api_key()`` is a
*search-oriented* gate: it returns True whenever any web backend is configured,
including search-only backends like ``ddgs`` (DuckDuckGo) that have **no**
extract capability. On a China deployment whose configured backend is ``ddgs``
the model still sees ``web_extract`` in its toolset, calls it, and the request
spins until the Feishu bridge times out (there is no extract implementation for
ddgs to answer with).

The original cloud fix was a build-time in-place patch that inserted a
``check_web_extract_available()`` helper into ``tools/web_tools.py`` and rewrote
the ``web_extract`` registration's ``check_fn`` to it. This module re-expresses
that as a zero-in-place overlay seam: the upstream file stays byte-for-byte
upstream, and at plugin-discovery time we swap the ``check_fn`` on the already
registered ``web_extract`` entry in the tool registry.

How the seam works
==================
The registry (``tools.registry.registry``) is a module-level singleton and each
``ToolEntry`` carries a mutable ``.check_fn`` attribute. ``web_tools.py``
registers ``web_extract`` at import time; overlay ``apply()`` runs *after* that
import (plugin discovery imports ``tools.web_tools`` to force registration if it
isn't already loaded), so the entry exists by the time we patch it. We replace
``registry._tools["web_extract"].check_fn`` with a gate that mirrors the
original helper exactly::

    backend = tools.web_tools._get_extract_backend()
    return backend in {"exa","parallel","firecrawl","tavily"} and _is_backend_available(backend)

Only extract-capable backends that are also *available* keep ``web_extract``
exposed; anything else (ddgs, searxng, brave-free, xai, or an unavailable
extract backend) hides it. ``web_search`` is untouched — it keeps
``check_web_api_key`` and stays available on search-only backends.

Idempotent (a sentinel on the installed check_fn + a module ``_APPLIED`` flag)
and fail-safe: any error leaves upstream behavior in place and returns False so
the plugin can warn (and the seam-test turns a missing target into a loud CI
failure). ``apply()`` returns False only if a target symbol is missing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The upstream tool name and the module + helpers we bind to. Centralized so the
# seam-test can pin them; an upstream rename breaks the patch AND the test.
_TOOL_NAME = "web_extract"
_TARGET_WEB_TOOLS_MODULE = "tools.web_tools"
_TARGET_EXTRACT_BACKEND_FN = "_get_extract_backend"
_TARGET_BACKEND_AVAILABLE_FN = "_is_backend_available"

# Backends that actually implement content extraction. Mirrors the exact set the
# original in-place hc-401 helper used.
_EXTRACT_CAPABLE_BACKENDS = frozenset({"exa", "parallel", "firecrawl", "tavily"})

_APPLIED = False
_MARK = "_apex_overlay_web_extract_gate"


def check_web_extract_available() -> bool:
    """Return True only when the configured EXTRACT backend can extract.

    Reproduces the original overlay helper verbatim:
    ``backend in {exa, parallel, firecrawl, tavily} and _is_backend_available(backend)``.
    Reads the backend fresh per call (config can change between tool-definition
    passes) and is fully defensive — any error resolves to False (hide the tool)
    rather than raising into the registry's ``check_fn`` invocation.
    """
    try:
        import importlib

        web_tools = importlib.import_module(_TARGET_WEB_TOOLS_MODULE)
        backend = getattr(web_tools, _TARGET_EXTRACT_BACKEND_FN)()
        if backend not in _EXTRACT_CAPABLE_BACKENDS:
            return False
        return bool(getattr(web_tools, _TARGET_BACKEND_AVAILABLE_FN)(backend))
    except Exception as exc:  # never raise into check_fn — fail closed (hidden)
        logger.debug("apex_overlay: web_extract availability probe failed: %s", exc)
        return False


# Mark the module-level gate so re-apply is a cheap no-op (idempotent even if the
# registry entry object is replaced by an MCP refresh between applies).
setattr(check_web_extract_available, _MARK, True)


def apply() -> bool:
    """Swap ``web_extract``'s registry ``check_fn`` to the extract-capable gate.

    Idempotent and fail-safe. Returns True when the gate is installed (or already
    present), False if a required upstream target is missing (the seam-test turns
    that into a hard CI failure). Never raises into the host.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    # 1. Force registration by importing web_tools (no-op if already imported),
    #    and pin the helper functions our gate calls.
    try:
        web_tools = importlib.import_module(_TARGET_WEB_TOOLS_MODULE)
        getattr(web_tools, _TARGET_EXTRACT_BACKEND_FN)
        getattr(web_tools, _TARGET_BACKEND_AVAILABLE_FN)
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not bind web_extract gate — %s.%s/%s missing. "
            "web_extract stays gated by the broad check_web_api_key (may hang on "
            "search-only backends like ddgs). Upstream may have renamed it. (%s)",
            _TARGET_WEB_TOOLS_MODULE, _TARGET_EXTRACT_BACKEND_FN,
            _TARGET_BACKEND_AVAILABLE_FN, exc,
        )
        return False

    # 2. Swap the check_fn on the already-registered web_extract entry.
    try:
        from tools.registry import registry

        entry = registry._tools.get(_TOOL_NAME)
        if entry is None:
            logger.error(
                "apex_overlay: web_extract not in registry at apply() time — the "
                "extract-capability gate was NOT installed. web_extract may hang "
                "on search-only backends.",
            )
            return False
        if not getattr(entry.check_fn, _MARK, False):
            entry.check_fn = check_web_extract_available
    except Exception as exc:  # noqa: BLE001 — never break plugin discovery
        logger.error(
            "apex_overlay: failed to install web_extract capability gate (%s)", exc,
        )
        return False

    _APPLIED = True
    logger.debug("apex_overlay: web_extract capability gate seam applied")
    return True
