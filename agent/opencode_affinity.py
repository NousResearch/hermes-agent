"""``x-opencode-session`` — OpenCode relay session-affinity header.

OpenCode (opencode.ai Zen/Go/free relay) pins requests that share an
``x-opencode-session`` value to the same upstream backend, which is what
keeps its prompt cache warm across the turns of one conversation. The value
only has to be opaque and consistent per conversation, so it is derived the
same way as the other conversation-affinity hints Hermes already sends
(OpenRouter's sticky ``session_id``, xAI's ``x-grok-conv-id``): the
host-declared routing scope first, then the ambient conversation root, then
the physical session id — normalized through ``_cache_scope_from_session_id``
so cron fires of one job share a scope.

Every OpenCode request — main turn on any transport, auxiliary calls
(compression, titles, vision, MoA) — goes through :func:`opencode_session_headers`
so the header cannot drift per code path.
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar
from typing import Any, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"

_STATELESS_OP_CONTEXT: ContextVar[Optional[str]] = ContextVar(
    "stateless_op_context", default=None
)
_STATELESS_OP_KEY: ContextVar[Optional[str]] = ContextVar(
    "stateless_op_key", default=None
)


def stateless_operation_scope(key: Optional[str] = None):
    """Bind out-of-turn calls to one operation identity (context manager).

    Dashboard/plugin HTTP-handler threads run aux LLM calls with no conversation
    scope; the relay hard-rejects a header-less request (400 MissingSessionID).
    The owner of such an operation wraps its execution in this scope and every
    call inside projects the same operation key instead of falling back to an
    install-wide identity. Without a key a fresh one is minted per scope entry,
    so unrelated operations never share a backend.
    """
    import contextlib
    import uuid

    @contextlib.contextmanager
    def _scope():
        token_op = _STATELESS_OP_CONTEXT.set(key or f"hermes-op-{uuid.uuid4().hex[:12]}")
        token_key = _STATELESS_OP_KEY.set(None)
        try:
            yield
        finally:
            _STATELESS_OP_KEY.reset(token_key)
            _STATELESS_OP_CONTEXT.reset(token_op)

    return _scope()


def _stateless_session_key() -> Optional[str]:
    """Key for the current stateless operation, minted lazily inside the scope."""
    op = _STATELESS_OP_CONTEXT.get()
    if op is None:
        return None
    key = _STATELESS_OP_KEY.get()
    if key is None:
        key = f"{op}-{hashlib.sha256(op.encode('utf-8')).hexdigest()[:8]}"
        _STATELESS_OP_KEY.set(key)
    return key


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches the built-in opencode-zen/go/free providers, custom
    ``opencode-<family>-*`` providers, and any base_url hosted on opencode.ai.
    """
    try:
        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(provider) is not None:
            return True
    except Exception:
        pass
    try:
        from agent.anthropic_endpoints import _is_opencode_endpoint

        return _is_opencode_endpoint(str(base_url or ""))
    except Exception:
        return False


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{"x-opencode-session": <key>}`` for OpenCode targets, else ``{}``."""
    if not is_opencode_target(provider, base_url):
        return {}
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        key = _cache_scope_from_session_id(
            # Top-level session_id → OpenRouter's sticky routing key. Per their prompt-caching docs it is
            # used directly as the routing key instead of hashing the opening messages, and it activates
            # stickiness on the first successful request rather than only after a cache hit. Resolve it from
            # the declared routing scope first (set only by a host that names its own conversation, #96811),
            # then the ambient conversation contextvar, with the explicit argument as fallback. The gap this
            # closes is the auxiliary call sites — compression, title generation, vision, web_extract,
            # session_search, MoA slots — which funnel through ``agent.auxiliary_client``. That module has
            # no session handle and passes no ``session_id``, so those calls sent NO sticky key at all and
            # each routed independently of the conversation it belonged to (#70820). Mirrors the Nous Portal
            # profile, which resolves the same way (f2f4df064d). The ambient value is the session-lineage
            # ROOT, so it also stays stable for installs that opt out of the default ``compression.in_place:
            # true`` and across delegate-subagent trees.
            get_affinity_scope() or get_conversation_context() or session_id
        )
    except Exception:
        key = str(session_id or "")
    # Out-of-turn aux callers (dashboard/plugin HTTP-handler threads) have no scope at all:
    # only when the stateless operation owner wrapped the work in a scope do we project its
    # operation key. With no scope the header is omitted — an install-global fallback would
    # collapse every unrelated background operation onto one backend.
    key = key or _stateless_session_key()
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the affinity header into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Non-OpenCode targets are left untouched.
    """
    headers = opencode_session_headers(provider, base_url, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
