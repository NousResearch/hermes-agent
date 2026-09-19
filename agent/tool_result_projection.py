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

1. **The canonical transcript is never modified.** Only ``content`` of ``tool`` rows in the
   API copy changes; roles, order and ``tool_call_id`` are untouched.
2. **Recovery is confirmed, or the row keeps its bytes.** The full result is persisted to the
   canonical spillover store first and the returned path is verified readable *by the agent* —
   a host path on a host-side backend, a sandbox-visible path on a remote one. A row that
   already carries a ``<persisted-output>`` block is projected only when its existing file
   still verifies (the cache is pruned on a 24h schedule); otherwise its preview stays.
3. **A protected verbatim tail** keeps the working set live: ``tail_ratio`` of the window
   (floored at 12K tokens), with internal message bounds (8–60) that are deliberately NOT derived
   from ``compression.protect_last_n`` — that knob governs what a compaction summary keeps, and its
   default would put most of a tool-heavy session inside the tail.
4. **Errors, multimodal results, small results, and structured results declaring
   ``projection_safe: false`` are never projected.**
5. **Monotone and sticky.** A row projected once stays projected and its stub is a pure
   function of the row's own bytes, so the wire prefix is byte-stable between passes: only
   committing a pass costs a prompt-cache break.
6. **Two phases, and the commit cannot fail halfway.** Phase 1 discovers candidates and decides on
   estimates without touching disk; phase 2 persists, verifies, and re-checks BOTH economic gates
   on what actually survived (a row that failed verification stays complete in the request, which
   makes the invalidated region bigger than phase 1 assumed); then a single in-memory loop rewrites
   the wire. A pass rejected by the pre-persistence gate writes nothing — verification can leave
   spillover files for rows it then declines, which is the store the tool path uses on the same 24h
   prune schedule — and an unexpected error leaves the request as the assembly produced it.

Row identity is ``(tool_call_id, content digest)`` — the same key for the dump file and for the
stickiness set, because a bare ``tool_call_id`` is not unique (imported/merged history carries
duplicates, and rows can have none at all).

The stub tells the model to read the archived file and deliberately does NOT suggest re-running
the tool: the projection cannot know whether a call is idempotent, and
``terminal("terraform apply")`` / ``git push`` / a payment POST must never be invited to replay
because a result was archived.

Not in scope: the iteration-summary path (``agent.chat_completion_helpers``) hand-builds its own
already-compacted input and is intentionally untouched.
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

DEFAULT_MIN_RESULT_CHARS = 4000
DEFAULT_TAIL_RATIO = 0.025
DEFAULT_TAIL_FLOOR_TOKENS = 12_000
# Message bounds for the tail. Deliberately NOT derived from ``compression.protect_last_n``:
# that knob governs what a compaction summary keeps, and reusing its default (20) would put
# most of a tool-heavy session inside the tail, leaving the pass nothing to reclaim. The floor
# keeps the newest exchange verbatim; the cap only bounds the tiny-message case.
DEFAULT_TAIL_FLOOR_MESSAGES = 8
DEFAULT_TAIL_MESSAGE_CAP = 60

# Stale-payload trigger when ``min_tokens`` is left at 0 (auto). It is both the hysteresis floor
# against churn and the pass's minimum reclaim: a route with prompt caching pays for every pass
# with a cache break, so it needs a bigger pile of stale bytes, while a route without caching
# pays nothing extra and can reclaim eagerly. Capped at a quarter of the window so it is
# reachable on a small-window model.
_AUTO_MIN_TOKENS_UNCACHED = 16_384
_AUTO_MIN_TOKENS_CACHED = 32_768
_AUTO_MIN_TOKENS_FLOOR = 2_048

_ARGS_IN_STUB_MAX = 160
_MAX_STATE_ROWS = 4096

_MODES_AUTO = {"auto", "on", "true", "enabled", "yes", "1"}

# Result shapes that must keep their bytes: a bounded error body is cheap to re-send and is
# exactly the kind of row a later turn reasons about.
_ERROR_PREFIXES = ("error:", "[error]", "[tool error]", "traceback (most recent call last):")


@dataclass(frozen=True)
class ProjectionKey:
    """Identity of a projectable row: the call it belongs to AND the bytes it holds."""

    tool_call_id: str
    content_digest: str

    def token(self) -> str:
        return f"{self.tool_call_id or '-'}:{self.content_digest[:16]}"

    def storage_key(self) -> str:
        return f"{self.tool_call_id or 'tool_result'}_{self.content_digest[:16]}"


@dataclass
class ProjectionPolicy:
    """Resolved knobs for one agent."""

    enabled: bool = False
    min_tokens: int = 0
    min_result_chars: int = DEFAULT_MIN_RESULT_CHARS
    tail_ratio: float = DEFAULT_TAIL_RATIO


@dataclass
class ProjectionState:
    """Per-agent sticky memory: rows already projected, so a re-applied pass cannot rewrite a
    prefix the provider has already cached."""

    projected: Set[str] = field(default_factory=set)
    order: List[str] = field(default_factory=list)

    def remember(self, token: str) -> None:
        if not token or token in self.projected:
            return
        self.projected.add(token)
        self.order.append(token)
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

    Absent attributes fall back to the module defaults, so an agent built before these keys
    existed — or a bare test double — behaves like an unconfigured install instead of blowing up
    the request. The default is OFF: archiving old tool output is a semantic change to what the
    model sees, so it ships opt-in until task-success parity is measured on real sessions.
    """
    cc = getattr(agent, "context_compressor", None)
    mode = str(getattr(cc, "tool_result_projection", "off") or "off").strip().lower()
    return ProjectionPolicy(
        enabled=mode in _MODES_AUTO,
        min_tokens=max(0, _cfg_int(cc, "tool_result_projection_min_tokens", 0)),
        min_result_chars=max(
            0, _cfg_int(cc, "tool_result_projection_min_result_chars", DEFAULT_MIN_RESULT_CHARS)
        ),
        tail_ratio=max(0.0, _cfg_float(cc, "tool_result_projection_tail_ratio", DEFAULT_TAIL_RATIO)),
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


def tail_bounds(
    policy: ProjectionPolicy, window: Optional[int],
) -> Tuple[int, int, int]:
    """``(token_budget, message_floor, message_cap)`` for the protected verbatim tail.

    Tokens: ``tail_ratio`` of the window, floored at ``DEFAULT_TAIL_FLOOR_TOKENS`` — the same
    shape as the "lean" compaction tail, because protecting less than the working set is what
    makes a stub dangerous. Messages: a small constant floor (the newest exchange stays
    verbatim) and a cap, so a session of tiny messages cannot spend the whole history on the
    walk.
    """
    window_tokens = int(policy.tail_ratio * window) if window else 0
    return (max(DEFAULT_TAIL_FLOOR_TOKENS, window_tokens), DEFAULT_TAIL_FLOOR_MESSAGES, DEFAULT_TAIL_MESSAGE_CAP)


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


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def key_for(msg: Dict[str, Any], content: str) -> ProjectionKey:
    return ProjectionKey(str(msg.get("tool_call_id") or ""), content_digest(content))


def tool_call_index(messages: Sequence[Dict[str, Any]]) -> Dict[str, List[Tuple[str, str]]]:
    """``tool_call_id -> [(tool name, raw arguments), ...]`` in message order.

    A list, not a single entry: two assistant calls can share an id, and a plain dict would let
    the later one overwrite the earlier one's name/args so a stub could describe the wrong call.
    Callers consume occurrences with :func:`take_call`.
    """
    index: Dict[str, List[Tuple[str, str]]] = {}
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
            index.setdefault(call_id, []).append(
                (str(fn.get("name") or "unknown"), str(fn.get("arguments") or ""))
            )
    return index


def take_call(index: Dict[str, List[Tuple[str, str]]], msg: Dict[str, Any]) -> Tuple[str, str]:
    """Consume the next recorded call for this row's id (occurrence order)."""
    queue = index.get(str(msg.get("tool_call_id") or ""))
    return queue.pop(0) if queue else ("unknown", "")


def tool_call_metadata(messages: Sequence[Dict[str, Any]]) -> Dict[int, Tuple[str, str]]:
    """``message index -> (tool name, raw arguments)`` for every tool result row.

    One recorded occurrence is consumed per tool result **regardless of whether that result ends
    up projected**. Consuming lazily (only after a row passes the eligibility checks) breaks on
    duplicate ids with unequal candidacy: with ``dup -> terminal (small)`` followed by
    ``dup -> web_extract (large)``, the small row would leave the queue untouched and the large
    row would inherit the FIRST call's name and arguments, so its stub would describe a terminal
    call it never came from. Built before any filtering, the pairing is positional.
    """
    index = tool_call_index(messages)
    out: Dict[int, Tuple[str, str]] = {}
    for idx, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            out[idx] = take_call(index, msg)
    return out


def protected_tail_start(
    messages: Sequence[Dict[str, Any]], policy: ProjectionPolicy, window: Optional[int],
) -> int:
    """Index of the first message inside the protected verbatim tail.

    Walked from the end, bounded on both sides: at least the message floor and at most the
    message cap are protected, and the walk stops as soon as the token budget is spent.
    """
    if not messages:
        return 0
    budget, floor_messages, cap_messages = tail_bounds(policy, window)
    used = 0
    kept = 0
    start = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        used += _estimate_tokens([messages[idx]])
        kept += 1
        start = idx
        if kept >= floor_messages and (used >= budget or kept >= cap_messages):
            break
    return start


def _looks_like_error(content: str) -> bool:
    """Conservative error detector: JSON error envelopes, ``success: false``, text prefixes."""
    stripped = content.lstrip()
    if stripped[:1] in ("{", "["):
        if '"error"' in stripped or '"success"' in stripped or '"exit_code"' in stripped:
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if parsed.get("error") not in (None, "", False):
                    return True
                if parsed.get("success") is False:
                    return True
                # The real terminal envelope is
                # ``{"output": ..., "exit_code": rc, "error": null}`` — a non-zero exit is a
                # failure even though ``error`` is null, and stale failures are exactly where the
                # causal evidence lives (a failed build, a red test run, an OOM kill at 137).
                exit_code = parsed.get("exit_code")
                if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
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
    *, tool_name: str, tool_args: str, content_len: int, line_count: int, digest: str,
    recovery_path: str,
) -> str:
    """The replacement row. A pure function of the row it replaces — byte-stable, so
    re-projecting across turns cannot move the wire prefix.

    The recovery instruction covers reading the archived file only. It must not invite a
    replay: the projection cannot know whether a call is idempotent, and ``terraform apply``,
    ``git push`` or a payment POST must never be suggested for re-execution because a result
    was archived.
    """
    args = tool_args if len(tool_args) <= _ARGS_IN_STUB_MAX else tool_args[:_ARGS_IN_STUB_MAX] + "…"
    return (
        f"{PROJECTION_MARKER} tool={tool_name} bytes={content_len} lines={line_count} sha256={digest}\n"
        f"args={args}\n"
        f"full output archived at: {recovery_path}\n"
        "This result is no longer in the conversation. Read the archived file with read_file "
        "(offset/limit) if its content is needed again."
    )


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


# ── backend / env resolution ─────────────────────────────────────────────────


def destination_caches_prefixes(agent: Any) -> bool:
    """Whether this request's destination has a cached prefix that a rewrite would invalidate.

    ``agent._use_prompt_caching`` is only "Hermes emits explicit cache-control markers" (the
    Anthropic-style policy). It is ``False`` on routes that cache prefixes perfectly well on the
    provider side: OpenRouter reported 86,832 cached input tokens out of 86,835 for
    ``gpt-5.6-luna`` under the policy-explicit arm of this PR's live measurement, where the marker
    policy was off. Treating such a route as uncached would use the smaller stale-pile trigger and
    skip the cache-break gate altogether — the economics argument inverted on the very route used
    as evidence.

    The strongest available signal for the destination's own behaviour is its own usage report,
    which the session already accumulates: a provider that returns cached input tokens has a
    prefix cache. On a session's first request nothing is cached yet, so "no evidence" and "no
    cache to break" coincide. A session that mixes providers can only be over-conservative here
    (a bigger trigger and a break-cost gate), which is the safe direction for an optimization.
    """
    if getattr(agent, "_use_prompt_caching", False):
        return True
    try:
        return int(getattr(agent, "session_cache_read_tokens", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _backend_is_remote() -> Optional[bool]:
    """``True`` = remote backend, ``False`` = host-side, ``None`` = could not be determined.

    Tri-state on purpose: assuming "host-side" when resolution fails would be fail-open for exactly
    the case this check exists for (a plugin- or config-registered remote backend), and the failure
    mode is a stub pointing at a path the sandbox cannot read. Unknown is treated by the caller as
    "not confirmed local", so an env is required before anything is projected.
    """
    try:
        from tools.env_probe import (
            _REMOTE_BACKENDS, _plugin_backend_is_remote, _resolve_terminal_backend,
        )

        backend = _resolve_terminal_backend()
        return bool(backend in _REMOTE_BACKENDS or _plugin_backend_is_remote(backend))
    except Exception:
        logger.debug("Terminal backend resolution failed; treating readability as unconfirmed", exc_info=True)
        return None


def _resolve_active_env(agent: Any):
    """Best-effort live terminal env for this agent, or None.

    The sandbox is registered in ``_active_environments`` under the turn's ``effective_task_id``,
    which the turn prologue stores as ``agent._current_task_id`` (it is a fresh UUID when the caller
    passes no task id). ``session_id`` is deliberately NOT a fallback: a stale session-scoped id can
    resolve to a DIFFERENT sandbox, and verifying a path against the wrong sandbox is worse than
    declining to project.
    """
    task_id = (
        getattr(agent, "_current_task_id", None) or getattr(agent, "task_id", None)
        or getattr(agent, "_task_id", None)
    )
    if not task_id:
        return None
    try:
        from tools.terminal_tool_lifecycle import get_active_env

        return get_active_env(str(task_id))
    except Exception:
        logger.debug("Could not resolve an active terminal env", exc_info=True)
        return None


# ── the pass ─────────────────────────────────────────────────────────────────


def project_stale_tool_results(
    agent: Any, api_messages: List[Dict[str, Any]], *, env: Any = "auto",
) -> int:
    """Replace stale, large, recoverable tool results on the API copy with stubs.

    Mutates ``api_messages`` in place (never the canonical transcript) and returns the number of
    rows projected. ``env="auto"`` resolves the session's live terminal env; pass an explicit env
    (or ``None``) to drive the remote-backend ladder directly. Fail-open: any unexpected error
    leaves the request as the rest of the assembly produced it.
    """
    try:
        return _project(agent, api_messages, env)
    except Exception:
        logger.warning(
            "Tool-result projection failed; sending the unprojected transcript", exc_info=True,
        )
        return 0


def _project(agent: Any, api_messages: List[Dict[str, Any]], env: Any) -> int:
    if not api_messages:
        return 0
    policy = resolve_policy(agent)
    if not policy.enabled:
        return 0
    cc = getattr(agent, "context_compressor", None)
    window = context_window_for(agent, cc)
    # "Cache capable" is a property of the DESTINATION, not of whether Hermes emits cache-control
    # markers: a route can cache prefixes provider-side with the marker policy off, and then the
    # smaller trigger plus a skipped break-cost gate would be exactly the wrong economics.
    cache_capable = destination_caches_prefixes(agent)
    if env == "auto":
        env = _resolve_active_env(agent)
    backend_remote = _backend_is_remote()
    if backend_remote is not False and env is None:
        # A host path is no proof of readability inside Docker/SSH/Modal, and with no sandbox there
        # is nothing to probe or copy into. `None` (backend could not be resolved) lands here too:
        # when in doubt this optimization must decline rather than risk a dead pointer.
        logger.debug(
            "Tool-result projection skipped: %s backend with no live env to verify against",
            "remote" if backend_remote else "unresolved",
        )
        return 0

    state = projection_state_for(agent)
    tail_start = protected_tail_start(api_messages, policy, window)
    # Positional pairing for every tool result, built before any eligibility filtering: see
    # tool_call_metadata() for the duplicate-id case that lazy consumption gets wrong.
    call_meta = tool_call_metadata(api_messages)

    # ── phase 1: discovery + economics on estimates, no disk writes ───────────
    planned: List[Dict[str, Any]] = []
    sticky_plan: List[Dict[str, Any]] = []
    fresh_reclaim = 0
    for idx, msg in enumerate(api_messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str) or not content:
            continue
        key = key_for(msg, content)
        already = key.token() in state.projected
        if idx >= tail_start and not already:
            # Inside the protected tail: only a row already projected there (invariant 5) may
            # be touched.
            continue
        if not is_candidate(msg, policy):
            continue
        tool_name, tool_args = call_meta.get(idx, ("unknown", ""))
        estimate = build_stub(
            tool_name=tool_name, tool_args=tool_args, content_len=len(content),
            line_count=content.count("\n") + 1, digest=key.content_digest[:16],
            recovery_path="/" + "x" * 48,
        )
        if len(estimate) >= len(content):
            continue
        old_tokens = _estimate_tokens([msg])
        entry = {
            "idx": idx, "msg": msg, "content": content, "key": key,
            "tool_name": tool_name, "tool_args": tool_args,
            "line_count": content.count("\n") + 1, "old_tokens": old_tokens, "estimate": estimate,
        }
        if already:
            sticky_plan.append(entry)
        else:
            planned.append(entry)
            fresh_reclaim += max(0, old_tokens - _estimate_tokens([{**msg, "content": estimate}]))

    trigger = trigger_tokens(policy, window, cache_capable)
    approved = fresh_reclaim >= trigger
    if approved and cache_capable:
        # The rewrite invalidates the cached prefix from the first stub onward, and that region
        # is re-prefilled once. Require the reclaim to cover it.
        break_cost = _estimate_region_tokens(api_messages, {e["idx"]: e["estimate"] for e in planned})
        approved = fresh_reclaim >= break_cost
        if not approved:
            logger.debug(
                "Tool-result projection declined: reclaim %s below the cache-break cost %s",
                f"{fresh_reclaim:,}", f"{break_cost:,}",
            )
    if not approved:
        logger.debug(
            "Tool-result projection idle: %s newly stale tokens below the %s trigger",
            f"{fresh_reclaim:,}", f"{trigger:,}",
        )
        # Sticky rows are re-applied regardless: un-stubbing a cached prefix is the one thing
        # this layer must never do.
        return _commit(api_messages, _verified_stubs(sticky_plan, env, fresh=False), state)

    # ── phase 2: persist + verify, only for an approved pass ─────────────────
    fresh_stubs = _verified_stubs(planned, env, fresh=True)
    real_reclaim = sum(
        max(0, entry["old_tokens"] - _estimate_tokens([{**entry["msg"], "content": entry["stub"]}]))
        for entry in fresh_stubs
    )
    # Verification can drop rows (a dead persisted path, a sandbox copy that failed). Those rows stay
    # COMPLETE in the request, so the invalidated region grows and BOTH gates have to be re-checked on
    # what actually survived — a phase-1 approval assumed every planned row became a stub.
    if fresh_stubs:
        declined = real_reclaim < trigger
        if declined:
            logger.debug(
                "Tool-result projection declined after verification: %s reclaimed of %s estimated",
                f"{real_reclaim:,}", f"{fresh_reclaim:,}",
            )
        elif cache_capable:
            actual_break_cost = _estimate_region_tokens(
                api_messages, {e["idx"]: e["stub"] for e in fresh_stubs}
            )
            if real_reclaim < actual_break_cost:
                declined = True
                logger.debug(
                    "Tool-result projection declined after verification: reclaim %s below the "
                    "recomputed cache-break cost %s",
                    f"{real_reclaim:,}", f"{actual_break_cost:,}",
                )
        if declined:
            fresh_stubs = []
    stubs = fresh_stubs + _verified_stubs(sticky_plan, env, fresh=False)
    return _commit(api_messages, stubs, state, reclaimed=real_reclaim)


def _verified_stubs(entries: List[Dict[str, Any]], env: Any, *, fresh: bool) -> List[Dict[str, Any]]:
    """Persist/verify each row's recovery target; drop the ones that cannot be confirmed."""
    out: List[Dict[str, Any]] = []
    for entry in entries:
        stub = _confirmed_stub(entry, env)
        if stub is not None:
            out.append({**entry, "stub": stub, "fresh": fresh})
    return out


def _confirmed_stub(entry: Dict[str, Any], env: Any) -> Optional[str]:
    """Return this row's stub, or None to leave the row exactly as it is.

    A row already carrying a ``<persisted-output>`` block keeps its preview unless the file that
    block points at still verifies readable — replacing a live preview with a dead pointer would
    make the model strictly worse off.
    """
    from tools.tool_result_storage import (
        extract_persisted_path, spillover_path_is_readable, store_spillover_content,
    )

    key: ProjectionKey = entry["key"]
    content: str = entry["content"]
    existing = extract_persisted_path(content)
    if existing:
        if not spillover_path_is_readable(existing, env):
            logger.debug("Keeping a tool result: its archived file is unreadable (%s)", existing)
            return None
        path = existing
    else:
        path = store_spillover_content(content, key.storage_key(), env=env)
        if not path:
            # Invariant 2: no confirmed recovery target, no projection.
            return None
    return build_stub(
        tool_name=entry["tool_name"], tool_args=entry["tool_args"], content_len=len(content),
        line_count=entry["line_count"], digest=key.content_digest[:16], recovery_path=path,
    )


def _estimate_region_tokens(api_messages: List[Dict[str, Any]], replacement: Dict[int, str]) -> int:
    """Tokens the provider re-prefills after the rewrite: everything from the first replacement
    onward, measured at its POST-rewrite size — that region is what gets re-processed.

    ``replacement`` maps a row index to the content that will be there. Phase 1 passes the planned
    stub estimates, phase 2 the verified stubs; measuring pre-rewrite bytes instead would overstate
    the break by roughly a whole request and silently disable the pass on every cached route.
    """
    if not replacement:
        return 0
    first_index = min(replacement)
    region = [
        {**msg, "content": replacement[idx]} if idx in replacement else msg
        for idx, msg in enumerate(api_messages[first_index:], start=first_index)
    ]
    return _estimate_tokens(region)


def _commit(
    api_messages: List[Dict[str, Any]], stubs: List[Dict[str, Any]], state: ProjectionState,
    reclaimed: int = 0,
) -> int:
    """Apply the stubs and record stickiness — in-memory only, so it cannot fail halfway."""
    if not stubs:
        return 0
    for entry in stubs:
        api_messages[entry["idx"]] = {**api_messages[entry["idx"]], "content": entry["stub"]}
        state.remember(entry["key"].token())
    fresh = [entry for entry in stubs if entry["fresh"]]
    if fresh:
        logger.info(
            "Tool-result projection: %d stale result(s) archived, ~%s tokens reclaimed per "
            "request (%s message tokens now)%s",
            len(fresh), f"{reclaimed:,}", f"{_estimate_tokens(api_messages):,}",
            f"; {len(stubs) - len(fresh)} row(s) kept projected" if len(stubs) > len(fresh) else "",
        )
    return len(stubs)
