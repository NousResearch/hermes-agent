"""SDK/CLI readiness probes and subscription auth-failure classification.

Safe error/traceback text, the conservative auth-failure needle list and the
lazy-install availability check. Extracted from ``claude_agent_sdk_session.py``.
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any, Optional
from agent.redact import redact_sensitive_text

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# Substrings in SDK/CLI errors that signal broken subscription credentials.
# Conservative on purpose — mirrors codex's _OAUTH_REFRESH_FAILURE_HINTS
# contract: every needle is a phrase, never a bare token. Bare "401" matched
# tool ids and byte offsets; bare "credentials" matched an MCP server
# complaining about its OWN files — and a hit retires the session, so a
# false positive is a wrong shutdown.
_AUTH_FAILURE_HINTS = (
    "not logged in",
    "please run /login",
    "invalid api key",
    "authentication_error",
    "oauth token",
    "token has expired",
    "expired token",
    "invalid bearer token",
    "setup-token",
)


_AUTH_401_UNAUTHORIZED_RE = re.compile(r"\b401\W{0,3}unauthorized\b")


_AUTH_UNAUTHORIZED_HTTP_401_RE = re.compile(
    r"\bunauthorized\W{0,15}\(?http\s*401\b"
)


def _safe_sdk_error_text(value: Any) -> str:
    """Redact provider/SDK diagnostics before they cross a transport boundary."""
    return redact_sensitive_text(str(value or ""), force=True)


def _safe_sdk_traceback(exc: BaseException) -> str:
    """Format traceback frames without re-emitting the exception payload.

    ``logger(..., exc_info=True)`` is useful for startup diagnosis, but it also
    appends ``str(exc)`` verbatim after the frames. SDK/import errors can carry
    credentials, so retain the actionable call stack while routing the error
    text itself through the normal forced-redaction boundary.
    """
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        return "".join(traceback.format_list(frames)).rstrip()
    except Exception:
        return "<traceback unavailable>"


def classify_auth_failure(
    *parts: str, mcp_attributed: bool = False
) -> Optional[str]:
    """Return a user-friendly re-auth hint if the strings look like a Claude
    subscription auth failure; otherwise None. The hint keeps the underlying
    error text: a hit retires the session, so the evidence must survive the
    redirect (codex surfaces the original error the same way)."""
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    if mcp_attributed or "mcp__" in haystack or "mcp server" in haystack:
        return None
    matched_401 = (
        _AUTH_401_UNAUTHORIZED_RE.search(haystack)
        or _AUTH_UNAUTHORIZED_HTTP_401_RE.search(haystack)
    )
    for needle in (None, *_AUTH_FAILURE_HINTS):
        if (needle is None and matched_401) or (
            needle is not None and needle in haystack
        ):
            original = _safe_sdk_error_text(
                next((p.strip() for p in parts if p and p.strip()), "")
            )
            if len(original) > 400:
                original = original[:400] + "…"
            return (
                "Claude authentication failed — the subscription OAuth token "
                "looks expired or invalid. Refresh it with `claude setup-token` "
                "(or `claude login` on this machine) and update "
                "CLAUDE_CODE_OAUTH_TOKEN, then retry. "
                f"(underlying error: {original})"
            )
    return None


def check_claude_sdk_available() -> tuple[bool, str]:
    """Preflight: the optional SDK extra must be importable, and it bundles /
    locates the Claude Code CLI itself. Mirrors check_codex_binary()."""
    # Fast path FIRST: when the SDK already imports, never enter the lazy
    # installer. ensure() can shell out to `uv pip install` and calls
    # importlib.invalidate_caches(); doing either immediately before importing
    # claude_agent_sdk -> mcp -> anyio rewrites site-packages and drops import
    # caches under a live interpreter, which intermittently surfaces as
    #     KeyError: 'anyio'
    # from importlib._bootstrap._find_and_load — a hard, flaky session-start
    # failure on installs where the extra is ALREADY present.
    try:
        import claude_agent_sdk  # noqa: F401

        return True, "ok"
    except ImportError:
        pass

    # Lazy-install lane, mirroring agent/anthropic_adapter._get_anthropic_sdk:
    # the extra is opt-in (excluded from [all]), so first use on a lean
    # install goes through tools.lazy_deps.ensure. FeatureUnavailable falls
    # through to the ImportError message below — same fail shape either way.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("provider.claude_agent_sdk", prompt=False)
    except ImportError:
        pass
    except Exception:
        # FeatureUnavailable — fall through to ImportError handling below
        pass
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return (
            False,
            "claude-agent-sdk is not installed. "
            "Install with: pip install 'hermes-agent[claude-agent-sdk]'",
        )
    return True, "ok"
