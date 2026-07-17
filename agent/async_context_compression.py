"""Background-prepared context compression candidates.

This module owns the *preparation* side of asynchronous context compression:
freezing a prefix of the conversation, generating a compressed candidate on a
single background worker, and validating that candidate before it is ever
applied. It deliberately knows nothing about persistence — applying a
candidate (SQLite lock, ``archive_and_compact``, session rotation, system
prompt rebuild) stays in ``agent/conversation_compression.py`` so both the
synchronous path and the background path share one commit implementation.

Design invariants (see tests/run_agent/test_async_context_compression.py):

* Preparation never mutates live state. The worker receives a deep copy of a
  frozen prefix; the live ``messages`` list, the live compressor, the session
  id and the database are never touched from the worker thread.
* A candidate is only usable when the live conversation still starts with the
  exact frozen prefix (same session, same generation, same canonical digest)
  and no tool call is currently open.
* Failures stay in the background: a worker exception marks the controller
  ``FAILED`` and the synchronous compression path remains the fallback.

This is a fresh implementation of the idea from PR #23892, rebuilt against
the current compression locks and persistence; no code is carried over.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── canonical digest & message-shape primitives ────────────────────────────


def _semantic_view(message: Any) -> Any:
    """Project a message onto the fields that are actually sent to the
    provider.

    Internal bookkeeping keys (``_db_persisted`` and any other
    underscore-prefixed marker) must not affect the digest: persistence
    flags flip between preparation and apply without changing what the
    model sees.
    """
    if not isinstance(message, dict):
        return repr(message)
    return {k: v for k, v in message.items() if not k.startswith("_")}


def canonical_prefix_digest(messages: List[Dict[str, Any]], count: int) -> str:
    """SHA-256 over the canonical JSON of the first ``count`` messages."""
    count = max(0, min(int(count), len(messages)))
    payload = json.dumps(
        [_semantic_view(m) for m in messages[:count]],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_tool_calls(message: Any) -> bool:
    return isinstance(message, dict) and bool(message.get("tool_calls"))


def align_prefix_boundary(messages: List[Dict[str, Any]], count: int) -> int:
    """Largest safe boundary ``<= count`` that never splits a tool group.

    A boundary is unsafe when it would separate an assistant message carrying
    ``tool_calls`` from any of its ``role="tool"`` results — either by cutting
    immediately after the assistant message or between two of its results.
    The boundary retreats until the cut lands before the whole group.
    """
    boundary = max(0, min(int(count), len(messages)))
    while 0 < boundary < len(messages):
        cut_follows_tool_call = _has_tool_calls(messages[boundary - 1])
        cut_inside_results = (
            isinstance(messages[boundary], dict)
            and messages[boundary].get("role") == "tool"
        )
        if not cut_follows_tool_call and not cut_inside_results:
            break
        boundary -= 1
    return boundary


def has_open_tool_call(messages: List[Dict[str, Any]]) -> bool:
    """True when any assistant ``tool_calls`` id lacks a ``tool`` result."""
    pending: set = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    pending.add(tc_id)
        elif msg.get("role") == "tool":
            pending.discard(msg.get("tool_call_id"))
    return bool(pending)


# ── data structures ────────────────────────────────────────────────────────


class CandidateState(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PreparedCompressionCandidate:
    """An immutable, pre-computed compression of a frozen conversation prefix."""

    session_id: str
    generation: int
    prefix_message_count: int
    prefix_digest: str
    prepared_messages: Tuple[Dict[str, Any], ...]
    source_prompt_tokens: int
    created_at_monotonic: float
    created_at_turn: int
    used_fallback: bool
    summary_error: Optional[str]


@dataclass
class PrepareResult:
    """Worker return value carrying summariser metadata alongside messages."""

    messages: List[Dict[str, Any]]
    used_fallback: bool = False
    summary_error: Optional[str] = None


def _coerce_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _coerce_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _coerce_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BackgroundCompressionConfig:
    """Parsed ``compression.background`` config block.

    The feature ships disabled and, when enabled, defaults to shadow mode:
    candidates are generated and validated but never applied.
    """

    enabled: bool = False
    shadow_only: bool = True
    prepare_threshold: float = 0.65
    apply_threshold: float = 0.82
    min_delta_tokens: int = 20_000
    min_frozen_messages: int = 12
    max_candidate_age_turns: int = 12
    max_workers: int = 1
    foreground_priority: bool = True
    fallback_sync: bool = True
    apply_only_between_turns: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "BackgroundCompressionConfig":
        raw = raw if isinstance(raw, dict) else {}
        defaults = cls()
        return cls(
            enabled=_coerce_bool(raw.get("enabled"), defaults.enabled),
            shadow_only=_coerce_bool(raw.get("shadow_only"), defaults.shadow_only),
            prepare_threshold=_coerce_float(
                raw.get("prepare_threshold"), defaults.prepare_threshold
            ),
            apply_threshold=_coerce_float(
                raw.get("apply_threshold"), defaults.apply_threshold
            ),
            min_delta_tokens=_coerce_int(
                raw.get("min_delta_tokens"), defaults.min_delta_tokens
            ),
            min_frozen_messages=_coerce_int(
                raw.get("min_frozen_messages"), defaults.min_frozen_messages
            ),
            max_candidate_age_turns=_coerce_int(
                raw.get("max_candidate_age_turns"), defaults.max_candidate_age_turns
            ),
            max_workers=max(1, _coerce_int(raw.get("max_workers"), defaults.max_workers)),
            foreground_priority=_coerce_bool(
                raw.get("foreground_priority"), defaults.foreground_priority
            ),
            fallback_sync=_coerce_bool(raw.get("fallback_sync"), defaults.fallback_sync),
            apply_only_between_turns=_coerce_bool(
                raw.get("apply_only_between_turns"), defaults.apply_only_between_turns
            ),
        )


# ── validation ─────────────────────────────────────────────────────────────


def validate_candidate(
    candidate: PreparedCompressionCandidate,
    *,
    session_id: str,
    messages: List[Dict[str, Any]],
    current_generation: Optional[int] = None,
    current_turn: Optional[int] = None,
    max_age_turns: Optional[int] = None,
) -> Tuple[bool, str]:
    """Check every safety invariant a candidate must satisfy before apply.

    Returns ``(ok, reason)`` where ``reason`` is a stable machine-readable
    slug used by telemetry and tests.
    """
    if candidate is None:
        return False, "no_candidate"
    if candidate.session_id != session_id:
        return False, "session_mismatch"
    if current_generation is not None and candidate.generation != current_generation:
        return False, "stale_generation"
    if (
        current_turn is not None
        and max_age_turns is not None
        and current_turn - candidate.created_at_turn > max_age_turns
    ):
        return False, "expired"
    if candidate.prefix_message_count > len(messages):
        return False, "prefix_out_of_range"
    live_digest = canonical_prefix_digest(messages, candidate.prefix_message_count)
    if live_digest != candidate.prefix_digest:
        return False, "digest_mismatch"
    return True, ""


def merge_candidate_with_live_messages(
    candidate: PreparedCompressionCandidate,
    live_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepared prefix + the live suffix, suffix objects preserved verbatim.

    The prepared messages are deep-copied so later in-place mutation of the
    live transcript can never reach back into the immutable candidate. The
    suffix keeps the exact live objects — byte-for-byte survival is the
    whole point.
    """
    merged: List[Dict[str, Any]] = [copy.deepcopy(m) for m in candidate.prepared_messages]
    merged.extend(live_messages[candidate.prefix_message_count:])
    return merged


# ── controller ─────────────────────────────────────────────────────────────


class BackgroundCompressionController:
    """Single-worker lifecycle manager for compression candidates.

    All public methods are safe to call from the foreground thread and never
    raise on worker failures. State transitions are serialized under one
    lock; a monotonically increasing *generation* invalidates in-flight work
    on supersede/cancel/reset without blocking on the worker.
    """

    def __init__(self, config: BackgroundCompressionConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._generation = 0
        self._state = CandidateState.IDLE
        self._candidate: Optional[PreparedCompressionCandidate] = None
        self._last_error: Optional[str] = None
        self._settled = threading.Event()
        self._settled.set()
        # Operational counters (formalized as telemetry in the config task).
        self.stats: Dict[str, int] = {}
        self.prepare_durations_ms: List[float] = []
        self.apply_durations_ms: List[float] = []

    # -- introspection ------------------------------------------------------

    @property
    def config(self) -> BackgroundCompressionConfig:
        return self._config

    @property
    def state(self) -> CandidateState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def peek_candidate(self) -> Optional[PreparedCompressionCandidate]:
        with self._lock:
            return self._candidate

    def _bump(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    # -- lifecycle ----------------------------------------------------------

    def try_start_preparation(
        self,
        *,
        session_id: str,
        messages: List[Dict[str, Any]],
        prepare_fn: Callable[[List[Dict[str, Any]]], Any],
        prefix_count: Optional[int] = None,
        current_turn: int = 0,
        source_prompt_tokens: int = 0,
    ) -> bool:
        """Freeze a prefix and submit background preparation.

        Returns False (without side effects) when the feature is disabled,
        the session id is empty, or the aligned prefix is too short. A new
        start supersedes any in-flight or ready candidate: only the newest
        generation can ever install its result.
        """
        if not self._config.enabled:
            return False
        if not session_id:
            return False

        requested = len(messages) if prefix_count is None else prefix_count
        boundary = align_prefix_boundary(messages, requested)
        if boundary < max(1, self._config.min_frozen_messages):
            logger.debug(
                "background compression: aligned prefix too short (%d < %d) — not starting",
                boundary, self._config.min_frozen_messages,
            )
            return False

        # Snapshot BEFORE taking the controller lock: deepcopy can be slow and
        # the foreground caller is the only writer of ``messages`` right now.
        frozen_prefix = copy.deepcopy(messages[:boundary])
        prefix_digest = canonical_prefix_digest(messages, boundary)

        with self._lock:
            self._generation += 1
            generation = self._generation
            self._candidate = None
            self._state = CandidateState.PREPARING
            self._last_error = None
            self._settled = threading.Event()
            settled = self._settled
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, self._config.max_workers),
                    thread_name_prefix="bg-compression",
                )
            executor = self._executor
        self._bump("candidate_started")

        executor.submit(
            self._run_preparation,
            generation,
            settled,
            session_id,
            frozen_prefix,
            prefix_digest,
            boundary,
            source_prompt_tokens,
            current_turn,
            prepare_fn,
        )
        return True

    def _run_preparation(
        self,
        generation: int,
        settled: threading.Event,
        session_id: str,
        frozen_prefix: List[Dict[str, Any]],
        prefix_digest: str,
        prefix_count: int,
        source_prompt_tokens: int,
        created_at_turn: int,
        prepare_fn: Callable[[List[Dict[str, Any]]], Any],
    ) -> None:
        """Worker body. Never raises; installs only if still the newest gen."""
        started = time.monotonic()
        try:
            raw = prepare_fn(frozen_prefix)
            if isinstance(raw, PrepareResult):
                prepared = raw.messages
                used_fallback = raw.used_fallback
                summary_error = raw.summary_error
            else:
                prepared = raw
                used_fallback = False
                summary_error = None
            if not prepared:
                raise RuntimeError("preparation produced no messages")
            candidate = PreparedCompressionCandidate(
                session_id=session_id,
                generation=generation,
                prefix_message_count=prefix_count,
                prefix_digest=prefix_digest,
                prepared_messages=tuple(copy.deepcopy(m) for m in prepared),
                source_prompt_tokens=source_prompt_tokens,
                created_at_monotonic=time.monotonic(),
                created_at_turn=created_at_turn,
                used_fallback=used_fallback,
                summary_error=summary_error,
            )
        except BaseException as exc:  # noqa: BLE001 — must never leak to foreground
            with self._lock:
                if generation == self._generation:
                    self._state = CandidateState.FAILED
                    self._candidate = None
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    settled.set()
                    current = True
                else:
                    current = False
            self._bump("candidate_failed" if current else "candidate_discarded_stale")
            logger.debug(
                "background compression preparation failed (gen=%d current=%s): %s",
                generation, current, exc,
            )
            return

        duration_ms = (time.monotonic() - started) * 1000.0
        with self._lock:
            if generation == self._generation and self._state is CandidateState.PREPARING:
                self._candidate = candidate
                self._state = CandidateState.READY
                settled.set()
                installed = True
            else:
                installed = False
        if installed:
            self._bump("candidate_ready")
            self.prepare_durations_ms.append(duration_ms)
            logger.debug(
                "background compression candidate ready: gen=%d prefix=%d prepared=%d in %.0fms",
                generation, prefix_count, len(candidate.prepared_messages), duration_ms,
            )
        else:
            self._bump("candidate_discarded_stale")

    def wait_until_settled(self, timeout: Optional[float] = None) -> bool:
        """Block until the newest preparation finished (test/replay helper)."""
        with self._lock:
            settled = self._settled
        return settled.wait(timeout)

    # -- consumption --------------------------------------------------------

    def take_valid_candidate(
        self,
        *,
        session_id: str,
        messages: List[Dict[str, Any]],
        current_turn: Optional[int] = None,
    ) -> Optional[PreparedCompressionCandidate]:
        """Return the candidate iff every safety invariant holds right now.

        An open tool call is a *temporal* refusal: the candidate survives and
        may apply once the result lands. Every other mismatch is permanent
        for this candidate, so it is discarded (``STALE``).

        The candidate is NOT consumed on success — callers clear it with
        :meth:`mark_applied` only after their commit is confirmed.
        """
        with self._lock:
            candidate = self._candidate
            generation = self._generation
        if candidate is None:
            return None

        if has_open_tool_call(messages):
            logger.debug(
                "background compression: open tool call — apply deferred, candidate kept"
            )
            return None

        ok, reason = validate_candidate(
            candidate,
            session_id=session_id,
            messages=messages,
            current_generation=generation,
            current_turn=current_turn,
            max_age_turns=self._config.max_candidate_age_turns,
        )
        if not ok:
            self.discard(reason)
            return None
        return candidate

    def discard(self, reason: str) -> None:
        with self._lock:
            self._candidate = None
            self._state = CandidateState.STALE
            self._last_error = reason
        self._bump("candidate_discarded_stale")
        logger.debug("background compression candidate discarded: %s", reason)

    def mark_applied(self, candidate: PreparedCompressionCandidate) -> None:
        """Clear the candidate after its commit was confirmed."""
        with self._lock:
            if self._candidate is candidate:
                self._candidate = None
                self._state = CandidateState.IDLE
        self._bump("candidate_applied")

    def record_shadow_validation(
        self, candidate: PreparedCompressionCandidate
    ) -> None:
        """Shadow mode: count a would-have-applied candidate, then drop it."""
        with self._lock:
            if self._candidate is candidate:
                self._candidate = None
                self._state = CandidateState.IDLE
        self._bump("candidate_shadow_validated")

    # -- invalidation -------------------------------------------------------

    def cancel(self) -> None:
        """Best-effort cancel: invalidate the in-flight generation."""
        with self._lock:
            self._generation += 1
            self._candidate = None
            self._state = CandidateState.CANCELLED
            self._settled.set()
        self._bump("candidate_cancelled")

    def reset(self) -> None:
        """Session boundary (/new, /reset, model switch): drop everything."""
        with self._lock:
            self._generation += 1
            self._candidate = None
            self._state = CandidateState.IDLE
            self._last_error = None
            self._settled.set()

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            self._generation += 1
            self._candidate = None
            self._state = CandidateState.IDLE
            self._settled.set()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)


# ── loop-facing helpers ────────────────────────────────────────────────────


def _numeric(value: Any) -> float:
    """Best-effort float for engine attributes that may be mocks/absent."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _agent_config(agent: Any) -> Optional[BackgroundCompressionConfig]:
    cfg = getattr(agent, "background_compression_config", None)
    if isinstance(cfg, BackgroundCompressionConfig):
        return cfg
    return None


def maybe_prepare_background_compression(
    agent: Any,
    messages: List[Dict[str, Any]],
    *,
    current_tokens: Optional[int] = None,
    current_turn: int = 0,
) -> bool:
    """Start background preparation when every trigger condition holds.

    Called between turns / after a completed tool round. Returns True only
    when a new preparation was actually submitted. Never raises and never
    touches the agent when the feature is disabled.
    """
    cfg = _agent_config(agent)
    if cfg is None or not cfg.enabled:
        return False
    if not getattr(agent, "compression_enabled", False):
        return False
    session_id = getattr(agent, "session_id", "") or ""
    if not session_id:
        return False
    engine = getattr(agent, "context_compressor", None)
    if engine is None:
        return False

    # Engines opt in via the ContextEngine hook; legacy plugins default to
    # False and keep the feature inert regardless of config.
    can_prepare = getattr(engine, "can_prepare_compression", None)
    try:
        if not callable(can_prepare) or not can_prepare(
            messages, current_tokens=current_tokens
        ):
            return False
    except Exception as exc:
        logger.debug("can_prepare_compression raised — treating as opt-out: %s", exc)
        return False

    tokens = _numeric(current_tokens)
    if tokens <= 0:
        tokens = _numeric(getattr(engine, "last_prompt_tokens", 0))
    context_length = _numeric(getattr(engine, "context_length", 0))
    if context_length > 0 and tokens < cfg.prepare_threshold * context_length:
        return False

    controller = getattr(agent, "background_compression", None)
    if controller is None:
        controller = BackgroundCompressionController(cfg)
        agent.background_compression = controller
    if controller.state is CandidateState.PREPARING:
        return False

    existing = controller.peek_candidate()
    if (
        existing is not None
        and tokens - existing.source_prompt_tokens < cfg.min_delta_tokens
    ):
        return False

    if has_open_tool_call(messages):
        return False

    def _prepare(frozen_prefix: List[Dict[str, Any]]) -> Any:
        return engine.prepare_compression(frozen_prefix, current_tokens=int(tokens) or None)

    return controller.try_start_preparation(
        session_id=session_id,
        messages=messages,
        prepare_fn=_prepare,
        current_turn=current_turn,
        source_prompt_tokens=int(tokens),
    )


def maybe_apply_prepared_candidate(
    agent: Any,
    messages: List[Dict[str, Any]],
    system_message: str,
    *,
    current_tokens: Optional[int] = None,
    current_turn: int = 0,
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Between-turn apply gate.

    Returns ``(compressed_messages, new_system_prompt)`` when a valid
    candidate was applied, or None in every other case (feature disabled,
    no/invalid candidate, shadow mode, below apply threshold) — callers then
    continue with the existing synchronous path.
    """
    cfg = _agent_config(agent)
    if cfg is None or not cfg.enabled:
        return None
    controller = getattr(agent, "background_compression", None)
    if not isinstance(controller, BackgroundCompressionController):
        return None

    # Apply gate: below ``apply_threshold`` the candidate stays warm — the
    # apply limit sits deliberately under the synchronous threshold so the
    # swap happens before the sync path would have paused the user. When the
    # context length is unknown (plugin engines) the gate is skipped and the
    # in-lock validation remains the only arbiter.
    engine = getattr(agent, "context_compressor", None)
    tokens = _numeric(current_tokens)
    if tokens <= 0 and engine is not None:
        tokens = _numeric(getattr(engine, "last_prompt_tokens", 0))
    context_length = (
        _numeric(getattr(engine, "context_length", 0)) if engine is not None else 0.0
    )
    if context_length > 0 and tokens < cfg.apply_threshold * context_length:
        return None

    session_id = getattr(agent, "session_id", "") or ""
    candidate = controller.take_valid_candidate(
        session_id=session_id,
        messages=messages,
        current_turn=current_turn,
    )
    if candidate is None:
        return None

    if cfg.shadow_only:
        controller.record_shadow_validation(candidate)
        logger.info(
            "background compression shadow: candidate for session=%s would apply "
            "(prefix=%d prepared=%d)",
            session_id, candidate.prefix_message_count,
            len(candidate.prepared_messages),
        )
        return None

    # Real apply: shared atomic commit lives in conversation_compression so
    # the synchronous path and the background path can never diverge. Any
    # failure here degrades to the synchronous fallback — a background
    # feature must never block the user's turn.
    from agent.conversation_compression import apply_prepared_candidate

    started = time.monotonic()
    try:
        result = apply_prepared_candidate(
            agent,
            candidate,
            messages,
            system_message,
            controller=controller,
        )
    except Exception as exc:
        logger.warning(
            "background compression apply failed — falling back to the "
            "synchronous path: %s", exc,
        )
        controller._bump("sync_fallback_count")
        return None
    if result is not None:
        controller.apply_durations_ms.append((time.monotonic() - started) * 1000.0)
    return result


__all__ = [
    "BackgroundCompressionConfig",
    "BackgroundCompressionController",
    "CandidateState",
    "PreparedCompressionCandidate",
    "PrepareResult",
    "align_prefix_boundary",
    "canonical_prefix_digest",
    "has_open_tool_call",
    "maybe_apply_prepared_candidate",
    "maybe_prepare_background_compression",
    "merge_candidate_with_live_messages",
    "validate_candidate",
]
