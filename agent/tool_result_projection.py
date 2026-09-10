"""Provider-bound projection of stale tool results — wire-only compaction.

The canonical transcript is the source of truth: session resume, the UI timeline,
summarization and ``session_search`` all read it. This module never touches it. It
rewrites only the per-call API copy (``api_messages``), which
``agent.turn_request_assembly.assemble_api_request`` already builds as a structural
clone (``_clone_message_for_send``) — the same seam ``evict_stale_outbound_tool_images``
(#89296) uses for stale screenshots.

Why this exists: the deterministic tool-result prune (``proactive_prune_tokens``) is
opt-in because it COMMITS the rewrite into the canonical transcript — lossy for resume
and the UI, and irreversibly so. Meanwhile reclamation below the compression threshold
never fires on large windows, so old tool payloads ride every request again (the failure
the config comment on ``proactive_prune_tokens`` names). Projection takes the same win
without the loss:

    canonical history ──lossless──▶ SQLite / session state
              │
              └──▶ provider projection ──▶ full : protected tail + small results
                                        ├─ stub : old, large, recoverable results
                                        └─ disk : full bytes in ``cache/spillover``

Invariants (each one has a test in ``tests/agent/test_tool_result_projection.py``):

1. The canonical transcript is never modified. Only ``content`` of ``tool`` rows in the
   API copy changes; roles, order and ``tool_call_id`` are untouched.
2. The full result is persisted to ``$HERMES_HOME/cache/spillover`` BEFORE the row is
   projected, and a row whose write failed is left intact. A stub that points at nothing
   is worse than the bytes it replaced.
3. The newest messages keep their full results: a protected tail by token budget with a
   message-count floor, so the working set is always in context.
4. Errors, multimodal results, small results and results that declare
   ``projection_safe: false`` are never projected.
5. Projection is monotone and sticky: a row projected once stays projected for the rest
   of the session, and its stub is a pure function of the row's own bytes. The wire
   prefix is therefore byte-stable between passes — only committing a pass costs one
   prompt-cache break.
6. A pass commits only when it is worth the break, and the gates are measured on the rows
   that are NEWLY eligible (``fresh``), not on the whole stale pile. A pile already
   projected contributes nothing new, so charging it to the budget would let the frontier
   advancing by one message authorize a cache break every single request. Fresh reclaim
   must clear the trigger and ``min_reclaim_tokens`` and, on a caching route, cover the
   region the rewrite invalidates — so each break has to be paid for by stale bytes that
   arrived since the last one.

Everything here is fail-open: any unexpected error leaves the request exactly as the rest
of the assembly produced it.

Not in scope: the iteration-summary path (``agent.chat_completion_helpers``) hand-builds its
own already-compacted input and is intentionally untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# First line of a projected stub. Deliberately distinct from the existing markers
# (``[Duplicate tool output...]``, ``_is_summary_stub``'s ``[tool] (N chars result)``,
# ``<persisted-output>``) so every pass that already understands those keeps working.
PROJECTION_MARKER = "[tool-result archived]"

# Defaults for a policy the caller did not configure (fresh compressor, bare agent in a
# test, agent built before the config keys existed).
DEFAULT_MIN_RESULT_CHARS = 4000
DEFAULT_MIN_RECLAIM_TOKENS = 8192
# Protected tail: the same shape as the compressor's "lean" tail (a clamped fraction of the
# window), because protecting less than the working set is what makes a stub dangerous.
DEFAULT_TAIL_RATIO = 0.025
DEFAULT_TAIL_MIN_TOKENS = 12000
DEFAULT_TAIL_MAX_TOKENS = 32000
# Message-count floor (never protect fewer than this) and cap (never walk further, so a
# session of tiny messages cannot turn the whole history into the tail).
DEFAULT_TAIL_MESSAGES = 8
DEFAULT_TAIL_MAX_MESSAGES = 60
# Stale-payload trigger when ``min_tokens`` is left at 0 (auto). This is hysteresis against
# churn, not the profitability test — that is the cache-break gate in ``_project``, which
# compares the reclaim against the region the rewrite invalidates. So these are absolute
# floors: a route with prompt caching pays for every pass with a cache break and needs a
# bigger pile of stale bytes; a route without caching pays nothing extra and reclaims
# eagerly. The floor is capped at a quarter of the window so it can never be unreachable
# on a small-window model.
_AUTO_MIN_TOKENS_UNCACHED = 16_384
_AUTO_MIN_TOKENS_CACHED = 32_768
_AUTO_MIN_TOKENS_FLOOR = 2_048

_ARGS_IN_STUB_MAX = 160
_MAX_STATE_ROWS = 4096

_MODES_OFF = {"off", "false", "disabled", "no", "0", "none"}
_MODES_AUTO = {"auto", "on", "true", "enabled", "yes", "1"}

# Result-text shapes that must keep their bytes: a bounded error body is cheap to re-send
# and is exactly the kind of row a later turn reasons about.
_ERROR_PREFIXES = ("error:", "[error]", "[tool error]", "traceback (most recent call last)")


@dataclass
class ProjectionPolicy:
    """Resolved knobs for one agent. ``min_tokens == 0`` means "derive from the window"."""

    enabled: bool = True
    min_tokens: int = 0
    min_result_chars: int = DEFAULT_MIN_RESULT_CHARS
    min_reclaim_tokens: int = DEFAULT_MIN_RECLAIM_TOKENS
    tail_ratio: float = DEFAULT_TAIL_RATIO
    tail_min_tokens: int = DEFAULT_TAIL_MIN_TOKENS
    tail_max_tokens: int = DEFAULT_TAIL_MAX_TOKENS
    tail_messages: int = DEFAULT_TAIL_MESSAGES
    tail_max_messages: int = DEFAULT_TAIL_MAX_MESSAGES


@dataclass
class ProjectionState:
    """Per-agent sticky memory: the rows already projected (so a re-applied pass cannot
    rewrite a prefix the provider has already cached)."""

    projected: Set[str] = field(default_factory=set)
    order: List[str] = field(default_factory=list)

    def remember(self, tool_call_id: str) -> None:
        if not tool_call_id or tool_call_id in self.projected:
            return
        self.projected.add(tool_call_id)
        self.order.append(tool_call_id)
        while len(self.order) > _MAX_STATE_ROWS:
            self.projected.discard(self.order.pop(0))


def is_projected_tool_result(content: Any) -> bool:
    """True when *content* is already a projection stub (cheap, prefix-based)."""
    return isinstance(content, str) and content.startswith(PROJECTION_MARKER)


def projection_state_for(agent: Any) -> ProjectionState:
    """Return the agent's projection state, creating it on first use (never raises)."""
    state = getattr(agent, "_tool_result_projection_state", None)
    if isinstance(state, ProjectionState):
        return state
    state = ProjectionState()
    try:
        agent._tool_result_projection_state = state
    except Exception:  # frozen/odd agent object: the pass still works, just not sticky
        logger.debug("Could not attach tool-result projection state", exc_info=True)
    return state


def _cfg_int(cc: Any, name: str, fallback: int) -> int:
    raw = getattr(cc, name, None)
    if raw is None or isinstance(raw, bool):
        return fallback
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _cfg_float(cc: Any, name: str, fallback: float) -> float:
    raw = getattr(cc, name, None)
    if raw is None or isinstance(raw, bool):
        return fallback
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def resolve_policy(agent: Any) -> ProjectionPolicy:
    """Read the policy off the agent's compressor (the context-reclamation owner).

    Absent attributes fall back to the module defaults, so an agent built before these
    keys existed — or a bare test double — behaves like an unconfigured install instead
    of blowing up the request.
    """
    cc = getattr(agent, "context_compressor", None)
    mode = str(getattr(cc, "tool_result_projection", "auto") or "auto").strip().lower()
    enabled = mode not in _MODES_OFF and mode in _MODES_AUTO
    return ProjectionPolicy(
        enabled=enabled,
        min_tokens=max(0, _cfg_int(cc, "tool_result_projection_min_tokens", 0)),
        min_result_chars=max(
            0, _cfg_int(cc, "tool_result_projection_min_result_chars", DEFAULT_MIN_RESULT_CHARS)
        ),
        min_reclaim_tokens=max(
            0, _cfg_int(cc, "tool_result_projection_min_reclaim_tokens", DEFAULT_MIN_RECLAIM_TOKENS)
        ),
        tail_ratio=max(0.0, _cfg_float(cc, "tool_result_projection_tail_ratio", DEFAULT_TAIL_RATIO)),
        tail_min_tokens=max(
            0, _cfg_int(cc, "tool_result_projection_tail_min_tokens", DEFAULT_TAIL_MIN_TOKENS)
        ),
        tail_max_tokens=max(
            0, _cfg_int(cc, "tool_result_projection_tail_max_tokens", DEFAULT_TAIL_MAX_TOKENS)
        ),
        tail_messages=max(0, _cfg_int(cc, "tool_result_projection_tail_messages", DEFAULT_TAIL_MESSAGES)),
        tail_max_messages=max(
            0, _cfg_int(cc, "tool_result_projection_tail_max_messages", DEFAULT_TAIL_MAX_MESSAGES)
        ),
    )


def context_window_for(agent: Any, cc: Any = None) -> Optional[int]:
    """The active window in tokens, or None when it cannot be resolved confidently."""
    candidates = (
        getattr(cc, "context_length", None) if cc is not None else None,
        getattr(agent, "_config_context_length", None),
        getattr(agent, "context_length", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def trigger_tokens(policy: ProjectionPolicy, window: Optional[int], cache_capable: bool) -> int:
    """Stale-payload pile (tokens) required before a pass is even considered."""
    if policy.min_tokens > 0:
        return policy.min_tokens
    base = _AUTO_MIN_TOKENS_CACHED if cache_capable else _AUTO_MIN_TOKENS_UNCACHED
    if window and window > 0:
        return max(_AUTO_MIN_TOKENS_FLOOR, min(base, window // 4))
    return base


def _estimate_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough

    return estimate_messages_tokens_rough(list(messages))


def tool_call_index(messages: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """``tool_call_id -> (tool name, raw arguments)`` from the assistant rows."""
    index: Dict[str, Tuple[str, str]] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or ():
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            if not call_id:
                continue
            fn = call.get("function") or {}
            index[call_id] = (str(fn.get("name") or "unknown"), str(fn.get("arguments") or ""))
    return index


def protected_tail_start(
    messages: Sequence[Dict[str, Any]], policy: ProjectionPolicy, window: Optional[int],
) -> int:
    """Index of the first message inside the protected verbatim tail.

    Walked from the end, bounded on BOTH sides so the boundary is predictable: at least
    ``tail_messages`` newest messages and at most ``tail_max_messages`` are protected, and
    the walk stops as soon as the token budget (``tail_ratio`` of the window, clamped to
    ``tail_min_tokens``/``tail_max_tokens``) is spent. The cap is what keeps a session of
    tiny messages from swallowing the whole history into the tail.
    """
    if not messages:
        return 0
    budget = policy.tail_min_tokens
    if window:
        budget = max(budget, int(window * policy.tail_ratio))
    # Cap only when it does not contradict a larger explicit floor (min > max keeps the min:
    # protecting MORE than the cap is the safe side of this boundary).
    if policy.tail_max_tokens and policy.tail_max_tokens >= policy.tail_min_tokens:
        budget = min(budget, policy.tail_max_tokens)
    used = 0
    kept = 0
    start = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        used += _estimate_tokens([messages[idx]])
        kept += 1
        start = idx
        if kept >= policy.tail_messages and (
            used >= budget or kept >= policy.tail_max_messages
        ):
            break
    return start


def _looks_like_error(content: str) -> bool:
    """Conservative error detector: JSON error envelopes, ``success: false``, text prefixes."""
    stripped = content.lstrip()
    if stripped[:1] in ("{", "["):
        # Cheap key guard first, then a real parse: keeping a large error body verbatim
        # costs little next to stubbing one the next turn needs back.
        if '"error"' in stripped or '"success"' in stripped:
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if parsed.get("error") not in (None, "", False):
                    return True
                if parsed.get("success") is False:
                    return True
    return stripped.lower().startswith(_ERROR_PREFIXES)


def declares_unsafe(value: Any) -> bool:
    """True when a structured tool result opts out (``projection_safe: false``)."""
    if not isinstance(value, str) or '"projection_safe"' not in value:
        return False
    try:
        parsed = json.loads(value.lstrip())
    except Exception:
        return False
    return isinstance(parsed, dict) and parsed.get("projection_safe") is False


def _is_multimodal(content: Any) -> bool:
    if isinstance(content, list):
        return True
    return isinstance(content, dict) and bool(content.get("_multimodal"))


def build_stub(
    *, tool_name: str, tool_args: str, content_len: int, line_count: int,
    digest: str, recovery_path: str, already_persisted: bool,
) -> str:
    """The replacement row. A pure function of the row it replaces — byte-stable, so a
    re-projection across turns cannot move the wire prefix."""
    args = tool_args if len(tool_args) <= _ARGS_IN_STUB_MAX else tool_args[:_ARGS_IN_STUB_MAX] + "…"
    kind = "full output" if already_persisted else "full output kept on disk"
    return (
        f"{PROJECTION_MARKER} tool={tool_name} bytes={content_len} lines={line_count} sha256={digest}\n"
        f"args={args}\n"
        f"{kind}: {recovery_path}\n"
        "Read that file with read_file (offset/limit) — or re-run the tool — if this result is "
        "needed again; its content is no longer in the conversation."
    )


def _recovery_path(content: str, tool_call_id: str) -> Optional[str]:
    """Persist the row's bytes to the canonical spillover store; path, or None on failure."""
    from tools.tool_result_storage import extract_persisted_path, store_spillover_content

    existing = extract_persisted_path(content)
    if existing:
        # Already spilled once: the file is the canonical home, keep pointing at it.
        return existing
    return store_spillover_content(content, tool_call_id)


def _stub_for(msg: Dict[str, Any], tool_name: str, tool_args: str) -> Optional[str]:
    """Build the stub for one row, or None when it must keep its bytes (fail-closed)."""
    content = msg.get("content")
    if not isinstance(content, str) or not content or _is_multimodal(content):
        return None
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    # The store's filename is derived from the key and an existing file is reused, so the key
    # must identify the BYTES and not just the call: a row without a tool_call_id, or two rows
    # sharing one (imported/merged history), would otherwise point a stub at another row's
    # content. Keying on the content digest makes a collision mean "same bytes".
    store_key = f"{str(msg.get('tool_call_id') or '') or 'tool_result'}_{digest[:16]}"
    path = _recovery_path(content, store_key)
    if not path:
        # Invariant 2: no recoverable home, no projection.
        return None
    stub = build_stub(
        tool_name=tool_name,
        tool_args=tool_args,
        content_len=len(content),
        line_count=content.count("\n") + 1,
        digest=digest[:16],
        recovery_path=path,
        already_persisted="<persisted-output>" in content,
    )
    return stub if len(stub) < len(content) else None


def is_candidate(msg: Any, policy: ProjectionPolicy) -> bool:
    """Whether a row is stubbable at all (position-independent part of the decision)."""
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return False
    content = msg.get("content")
    if not isinstance(content, str) or _is_multimodal(content):
        return False
    if is_projected_tool_result(content) or len(content) <= policy.min_result_chars:
        return False
    if _looks_like_error(content) or declares_unsafe(content):
        return False
    return True


def project_stale_tool_results(
    agent: Any, api_messages: List[Dict[str, Any]],
) -> int:
    """Replace stale, large, recoverable tool results on the API copy with stubs.

    Mutates ``api_messages`` in place (never the canonical transcript) and returns the
    number of rows projected. Fail-open: any error leaves the request as it was.
    """
    try:
        return _project(agent, api_messages)
    except Exception:
        logger.warning(
            "Tool-result projection failed; sending the unprojected transcript", exc_info=True,
        )
        return 0


def _project(agent: Any, api_messages: List[Dict[str, Any]]) -> int:
    if not api_messages:
        return 0
    policy = resolve_policy(agent)
    if not policy.enabled:
        return 0
    cc = getattr(agent, "context_compressor", None)
    window = context_window_for(agent, cc)
    cache_capable = bool(getattr(agent, "_use_prompt_caching", False))
    state = projection_state_for(agent)

    before_tokens = _estimate_tokens(api_messages)
    tail_start = protected_tail_start(api_messages, policy, window)
    index = tool_call_index(api_messages)

    # Dry run: pick the rows and build the stubs (which persists their bytes) before anything
    # on the wire changes, so a gate that declines leaves the request untouched.
    fresh: List[Tuple[int, str, int, int]] = []    # (idx, stub, old_tokens, new_tokens)
    sticky: List[Tuple[int, str]] = []             # rows already projected: (idx, stub)
    fresh_reclaim = 0
    for idx, msg in enumerate(api_messages):
        already = isinstance(msg, dict) and str(msg.get("tool_call_id") or "") in state.projected
        if idx >= tail_start and not already:
            # Inside the protected tail: only a row already projected there (invariant 5)
            # may be touched.
            continue
        if not is_candidate(msg, policy):
            continue
        tool_name, tool_args = index.get(str(msg.get("tool_call_id") or ""), ("unknown", ""))
        stub = _stub_for(msg, tool_name, tool_args)
        if stub is None:
            continue
        if already:
            sticky.append((idx, stub))
            continue
        old_tokens = _estimate_tokens([msg])
        new_tokens = _estimate_tokens([{**msg, "content": stub}])
        fresh.append((idx, stub, old_tokens, new_tokens))
        fresh_reclaim += max(0, old_tokens - new_tokens)

    # Stability first, unconditionally: a row already projected stays projected (invariant 5).
    # Re-stubbing it restores the exact bytes the previous request sent, so it is never a new
    # cache break, and deferring it would let a sliding tail un-stub a cached prefix.
    for idx, stub in sticky:
        api_messages[idx] = {**api_messages[idx], "content": stub}
    replayed = len(sticky)

    trigger = trigger_tokens(policy, window, cache_capable)
    if fresh_reclaim < trigger:
        logger.debug(
            "Tool-result projection idle: %s newly stale tokens below the %s trigger",
            f"{fresh_reclaim:,}", f"{trigger:,}",
        )
        return replayed

    if fresh_reclaim < policy.min_reclaim_tokens:
        # Below the hysteresis floor: wait for a bigger pile rather than fragmenting the
        # cache prefix for a marginal win.
        logger.debug(
            "Tool-result projection declined: reclaim %s below the %s minimum",
            f"{fresh_reclaim:,}", f"{policy.min_reclaim_tokens:,}",
        )
        return replayed

    first_index = fresh[0][0]
    after_tokens = before_tokens - fresh_reclaim
    if cache_capable:
        # The rewrite invalidates the cached prefix from the first stub onward, and that
        # region is re-prefilled once. Require the reclaim to cover it: a pass that cannot
        # pay for its own cache break is not committed.
        break_cost = _estimate_stub_region_tokens(api_messages, fresh, first_index)
        if fresh_reclaim < break_cost:
            logger.debug(
                "Tool-result projection declined: reclaim %s below the cache-break cost %s",
                f"{fresh_reclaim:,}", f"{break_cost:,}",
            )
            return replayed

    for idx, stub, _, _ in fresh:
        api_messages[idx] = {**api_messages[idx], "content": stub}
        state.remember(str(api_messages[idx].get("tool_call_id") or ""))

    logger.info(
        "Tool-result projection: %d stale result(s) archived, ~%s tokens reclaimed per request "
        "(%s -> %s message tokens); %d row(s) kept projected",
        len(fresh), f"{fresh_reclaim:,}", f"{before_tokens:,}", f"{after_tokens:,}",
        replayed,
    )
    return len(fresh) + replayed


def _estimate_stub_region_tokens(
    api_messages: List[Dict[str, Any]],
    candidates: List[Tuple[int, str, int, int]],
    first_index: int,
) -> int:
    """Tokens the provider re-prefills because of the rewrite: everything from the first
    stub onward, at its post-projection size."""
    new_content = {idx: stub for idx, stub, _, _ in candidates}
    region = [
        {**msg, "content": new_content[idx]} if idx in new_content else msg
        for idx, msg in enumerate(api_messages[first_index:], start=first_index)
    ]
    return _estimate_tokens(region)
