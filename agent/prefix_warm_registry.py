"""Registry of recent local-backend prompt prefixes for the gateway prefix warmer.

Local llama.cpp-style servers reuse cached KV/recurrent state only when a new
request's rendered prompt shares a byte-identical prefix with a cached state
(and, on hybrid-SSM architectures, a state checkpoint exists at or before the
divergence point). Hermes's system prompt is built once per session and is
byte-stable across turns, so every fresh session on the same profile pays a
full "entry tax" prefill of the same ~10-20k token prefix — unless some cached
state still holds it.

This module records, per (base_url, model), the most recent chat-completions
request kwargs the agent actually sent to a *local* endpoint. The gateway's
prefix-warmer watcher (see ``gateway/prefix_warmer.py``) periodically replays
a tiny request built from that snapshot — the same ``tools`` and the same
leading system message, a fixed one-character user turn, ``max_tokens=1`` — so
the server keeps a warm state whose prefix exactly matches what the next fresh
session will send. A warm request that hits its own cached state costs only a
handful of prefill tokens.

Capture happens in ``build_api_kwargs`` (agent/chat_completion_helpers.py) and
must never affect the request path: ``record_local_prefix`` swallows all
errors, and both record hooks are dormant no-ops unless ``enable_capture()``
has been called (the gateway calls it when the warmer starts), so processes
that never enable the warmer — including plain CLI runs — pay nothing.

Note the captured system message includes Hermes's volatile tail (memory
block, user profile, session timestamp), so a fresh session's prompt matches
the warm state up to that boundary rather than through the entire system
block. The stable head — tool schemas plus the fixed system prose, which is
the bulk of the prefix — is what the warmer keeps reusable.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import time
from typing import Any, Dict, List, Optional

# Newest-first cap on distinct (base_url, model, system-prompt) prefixes
# retained. A gateway normally sends one or two distinct prefixes; the cap
# bounds the warmer's worst-case work when auxiliary tasks (background
# review, MoA) contribute additional tool-carrying prefixes.
_MAX_ENTRIES = 4

_lock = threading.Lock()
# key: (base_url, model, system_hash) -> snapshot dict
_entries: Dict[tuple, Dict[str, Any]] = {}
# base_url -> monotonic-ish wall time of the last real local request seen
# (warmable or not). The warmer uses this to stand down while an endpoint
# has live traffic — see ``last_traffic_at``.
_last_traffic: Dict[str, float] = {}

# Process-global gate, set by the gateway when the warmer starts. While
# False (the default, and always in warmer-less processes such as plain CLI
# runs) both record hooks return immediately, so the request path pays
# nothing for a feature that isn't on.
_capture_enabled = False


def enable_capture() -> None:
    """Turn on prefix capture (called once when the warmer watcher starts)."""
    global _capture_enabled
    _capture_enabled = True


def disable_capture() -> None:
    """Turn capture back off (test helper)."""
    global _capture_enabled
    _capture_enabled = False


def _record(base_url: str, api_key: Optional[str], kwargs: Dict[str, Any], *,
            require_tools: bool) -> None:
    """Core capture: store a snapshot if this is a warmable local prefix."""
    if not base_url:
        return
    from agent.model_metadata import is_local_endpoint

    if not is_local_endpoint(base_url):
        return
    with _lock:
        _last_traffic[base_url] = time.time()
    messages = kwargs.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return
    head = messages[0].get("content")
    if not isinstance(head, str) or not head.strip():
        return
    if require_tools and not kwargs.get("tools"):
        return
    model = kwargs.get("model") or ""
    key = (base_url, model, hashlib.sha256(head.encode("utf-8", "replace")).hexdigest()[:16])
    snapshot = {
        "base_url": base_url,
        "api_key": api_key or "local",
        "model": model,
        "system_content": head,
        # Deep copies: the caller's objects can be mutated in place after
        # build time (e.g. conversation_loop's _force_ascii_payload
        # sanitization), which would silently corrupt a by-reference
        # snapshot. Capture only runs when the warmer is enabled, so the
        # copy cost is paid solely by users of the feature.
        "tools": copy.deepcopy(kwargs.get("tools")),
        "extra_body": copy.deepcopy(kwargs.get("extra_body")),
        "recorded_at": time.time(),
    }
    with _lock:
        _entries[key] = snapshot
        while len(_entries) > _MAX_ENTRIES:
            oldest = min(_entries, key=lambda k: _entries[k]["recorded_at"])
            del _entries[oldest]


def record_local_prefix(agent: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Record ``kwargs`` as the latest prompt prefix for the agent's endpoint.

    Called on the hot request path (``build_api_kwargs``) — returns ``kwargs``
    unchanged and never raises. A no-op unless ``enable_capture()`` has run.
    Records only when the endpoint is local, the api_mode is plain
    chat-completions, and the request opens with a system message (the shared
    prefix the warmer exists to keep hot).
    """
    if not _capture_enabled:
        return kwargs
    try:
        if getattr(agent, "api_mode", None) not in (None, "chat_completions"):
            return kwargs
        _record(
            getattr(agent, "base_url", None) or "",
            getattr(agent, "api_key", None),
            kwargs,
            require_tools=False,
        )
    except Exception:
        pass
    return kwargs


def record_call_prefix(base_url: Optional[str], api_key: Optional[str],
                       kwargs: Dict[str, Any]) -> None:
    """Record a ``call_llm`` request's prefix (never raises).

    Covers model calls that bypass ``build_api_kwargs`` — most importantly
    the MoA acting/aggregator call, which carries the agent's full system
    prompt and tool schemas. ``tools`` is required here so tool-less
    auxiliary calls (compression summaries, advisors, embeddings glue) don't
    churn the registry with prefixes no fresh session will ever share.
    A no-op unless ``enable_capture()`` has run.
    """
    if not _capture_enabled:
        return
    try:
        _record(str(base_url or ""), api_key, kwargs, require_tools=True)
    except Exception:
        pass


def get_snapshots() -> List[Dict[str, Any]]:
    """Return the current snapshots, newest first."""
    with _lock:
        return sorted(_entries.values(), key=lambda s: -s["recorded_at"])


def last_traffic_at(base_url: str) -> float:
    """Wall time of the last real local request seen for ``base_url``.

    Returns 0.0 if none has been seen. Updated for every local-endpoint
    request that reaches the capture hooks (warmable or not), so the warmer
    can stand down while an endpoint is actively serving a session.
    """
    with _lock:
        return _last_traffic.get(base_url, 0.0)


def clear() -> None:
    """Drop all snapshots and traffic timestamps (test helper)."""
    with _lock:
        _entries.clear()
        _last_traffic.clear()
