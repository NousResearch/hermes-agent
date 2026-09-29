"""Per-run structured trace lifecycle (schema v1.0.0 — ``specs/trace-schema.md``).

Every dispatched run — Kanban-backed or not, successful or failed — must emit
exactly one trace record. This module owns the in-process capture:

- :func:`start_run_trace` opens a run-scoped :class:`RunTraceContext` buffering
  identity (run_id, session_id, profile, board, task, model, provider, harness
  fingerprint) and the ``started_at`` clock.
- :func:`record_provider_usage` (``agent/turn_usage.record_response_usage``)
  folds per-call token buckets + cost into the aggregate counters.
- :func:`record_tool_call` (``agent/tool_executor._commit_tool_result``)
  appends one entry per executed tool call, arguments/result routed through
  the redaction hook (``agent/trace_redactor.redact``) before buffering.
- The four terminal kanban handlers (``tools/kanban_tools``) close the trace
  with the matching outcome the moment the board write lands.
- ``hermes_cli/cli_single_query`` wraps the whole ``-q`` worker path in
  try/finally so crashed and timed-out runs still emit ``outcome=crashed`` /
  ``timeout`` with the sanitized error class captured.

Invariants: emit-once (a second close is a no-op), capture never raises into
the caller, and the redactor's fail-closed contract (``redactor_error`` marker)
is preserved. ``request_dump_*`` writes are untouched by design — this is an
additive side channel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agent.run_trace")

SCHEMA_VERSION = "1.0.1"

_MESSAGE_ROLES = ("system", "user", "assistant", "tool")
_TERMINAL_OUTCOMES = {
    "kanban_complete": "completed",
    "kanban_block": "blocked",
    "kanban_request_review": "review_requested",
    "kanban_request_changes": "review_requested",
}
_TERMINAL_CALLS = frozenset(_TERMINAL_OUTCOMES)
_SAFETY_TIER_RANK = {"green": 1, "yellow": 2, "red": 3}

# Current run's open trace context (one per process; the worker main thread
# opens it, the provider/tool hooks may fire on executor threads — the lock
# covers every mutation).
_ACTIVE: Optional["RunTraceContext"] = None


def run_trace_enabled() -> bool:
    """Operator killswitch: ``HERMES_RUN_TRACE=off`` disables capture entirely."""
    return (os.environ.get("HERMES_RUN_TRACE") or "").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def active_trace() -> Optional["RunTraceContext"]:
    return _ACTIVE


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_builder_sha() -> str:
    """``git rev-parse HEAD:agent/prompt_builder.py`` against the running repo.

    Falls back to a content digest of the file when the checkout is partial or
    git is unavailable — the schema needs 40 lowercase hex, not a VCS object.
    """
    from pathlib import Path

    try:
        from hermes_constants import get_hermes_home  # noqa: F401  (import-cost parity only)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        pb_path = Path(__file__).resolve().parent / "prompt_builder.py"
        repo_root = pb_path.parent.parent
        out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            ["git", "-C", str(repo_root), "rev-parse", "HEAD:agent/prompt_builder.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = (out.stdout or "").strip().lower()
        if out.returncode == 0 and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
        if pb_path.is_file():
            return hashlib.sha256(pb_path.read_bytes()).hexdigest()[:40]
    except Exception:  # pragma: no cover - defensive
        logger.debug("prompt_builder_sha resolution failed", exc_info=True)
    return "0" * 40


def config_sha256() -> str:
    """SHA-256 of the effective profile config, secrets masked before hashing.

    Canonical JSON (sorted keys) of ``config.yaml`` after overlay resolution;
    key names survive, every leaf value is replaced by ``"<len:N>"`` so the
    digest changes with the config's *shape* without hashing credential bytes.
    """
    masked: Any = {}
    try:
        from hermes_cli.config_effective import load_user_config_effective

        raw = load_user_config_effective()
        if isinstance(raw, dict):
            masked = _mask_leaves(raw)
    except Exception:
        try:  # minimal fallback: profile config.yaml verbatim-but-masked
            import yaml

            from hermes_constants import get_hermes_home

            cfg_path = get_hermes_home() / "config.yaml"
            if cfg_path.is_file():
                raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                masked = _mask_leaves(raw if isinstance(raw, dict) else {"value": raw})
        except Exception:  # pragma: no cover - defensive
            logger.debug("config_sha256 resolution failed", exc_info=True)
            masked = {}
    with suppress(Exception):
        return hashlib.sha256(
            json.dumps(masked, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
    return "0" * 64


def _mask_leaves(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _mask_leaves(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_leaves(v) for v in obj]
    if isinstance(obj, str):
        return f"<len:{len(obj)}>"
    return obj


class RunTraceContext:
    """Buffers one run's trace; :meth:`close` assembles + appends the record."""

    def __init__(
        self,
        *,
        run_id: int,
        profile_slug: str,
        model: str,
        provider: str,
        board_slug: Optional[str] = None,
        task_id: Optional[str] = None,
        harness_version: Optional[Dict[str, str]] = None,
        trace_root: Optional[str] = None,
        clock=time.time,
    ) -> None:
        import threading

        self._lock = threading.Lock()
        self.run_id = int(run_id)
        self.session_id = ""
        self.profile_slug = profile_slug
        self.board_slug = board_slug or None
        self.task_id = task_id or None
        self.model = model
        self.provider = provider
        self.harness_version = harness_version or {
            "prompt_builder_sha": prompt_builder_sha(),
            "config_sha256": config_sha256(),
        }
        self.trace_root = trace_root
        self._clock = clock
        self.started_at = int(self._clock())
        self.tool_calls: List[Dict[str, Any]] = []
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.token_counts_complete = True
        self._saw_usage = False
        self.cost_usd: Optional[float] = None
        self.safety_tier_hit: Optional[str] = None
        self.killswitch: Optional[Any] = None
        self.redaction_log: List[Dict[str, Any]] = []
        self.terminal_call: Optional[str] = None
        self._closed = False

    # ------------------------------------------------------------------
    # Capture hooks (all fail-silent: tracing must never break a run)
    # ------------------------------------------------------------------

    def set_identity(self, *, session_id: str = "", model: str = "", provider: str = "") -> None:
        with self._lock:
            if session_id and not self.session_id:
                self.session_id = session_id
            if model and (not self.model or self.model == "unknown"):
                self.model = model
            if provider and (not self.provider or self.provider == "unknown"):
                self.provider = provider

    def record_usage(self, usage: Optional[Dict[str, Any]], cost_usd: Optional[float]) -> None:
        with self._lock:
            if not usage:
                self.token_counts_complete = False
                return
            self._saw_usage = True
            self.tokens["input"] += int(usage.get("input_tokens") or 0)
            self.tokens["output"] += int(usage.get("output_tokens") or 0)
            self.tokens["cache_read"] += int(usage.get("cache_read_tokens") or 0)
            self.tokens["cache_write"] += int(usage.get("cache_write_tokens") or 0)
            if cost_usd is not None:
                self.cost_usd = (self.cost_usd or 0.0) + float(cost_usd)

    def note_safety_tier(self, tier: Optional[str]) -> None:
        if tier not in _SAFETY_TIER_RANK:
            return
        with self._lock:
            current = self.safety_tier_hit
            if current is None or _SAFETY_TIER_RANK[tier] > _SAFETY_TIER_RANK[current]:
                self.safety_tier_hit = tier

    def note_killswitch(self, value: Any) -> None:
        with self._lock:
            self.killswitch = value

    def record_tool_call(
        self,
        name: str,
        arguments: Any,
        result: Any,
        *,
        call_id: str = "",
        duration_ms: int = 0,
        exit_code: Optional[int] = None,
    ) -> None:
        try:
            import agent.trace_redactor as _red_mod

            redact = getattr(_red_mod, "redact", None)
            RedactionContext = getattr(_red_mod, "RedactionContext", None)
        except Exception:  # pragma: no cover - redactor import path broken
            redact, RedactionContext = None, None
        index = 0
        with self._lock:
            index = len(self.tool_calls)
        args_out, result_out = arguments, result
        if redact is not None and RedactionContext is not None:
            try:
                ctx_args = RedactionContext(path=f"tool_calls[{index}].arguments")
                args_out = redact(arguments, context=ctx_args)
                ctx_res = RedactionContext(path=f"tool_calls[{index}].result")
                result_out = redact(result, context=ctx_res)
                events = [
                    {**e, "sha256": e.get("sha256")}
                    for ctx in (ctx_args, ctx_res)
                    for e in getattr(ctx, "log", [])
                    if isinstance(e, dict) and e.get("path") and e.get("reason")
                ]
            except Exception:  # fail-closed: mirrors trace_redactor's contract
                logger.debug("trace redaction raised; emitting redactor_error markers", exc_info=True)
                args_out = {"redacted": True, "reason": "redactor_error", "sha256": None}
                result_out = {"redacted": True, "reason": "redactor_error", "sha256": None}
                events = []
            if events:
                with self._lock:
                    self.redaction_log.extend(events)
        if not isinstance(args_out, dict):
            # Schema requires arguments to be an object (or the marker).
            args_out = {"value": args_out}
        if not call_id:
            call_id = str(uuid.uuid4())
        entry = {
            "id": str(call_id),
            "name": str(name),
            "arguments": args_out,
            "result": result_out,
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "duration_ms": max(0, int(duration_ms)),
            "is_terminal": False,
        }
        with self._lock:
            self.tool_calls.append(entry)

    # ------------------------------------------------------------------
    # Terminal path
    # ------------------------------------------------------------------

    def close(
        self,
        *,
        outcome: str,
        terminal_call: Optional[str] = None,
        error_class: Optional[str] = None,
        exit_code: Optional[int] = None,
        agent_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Assemble + append the record exactly once. Returns True when written."""
        # Under the lock: state mutations only. Redaction events produced by
        # _snapshot_messages (which runs the redactor on message content) come back
        # and merge here — mirroring record_tool_call's pattern: never take _lock
        # twice on the close path.
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            if terminal_call in _TERMINAL_CALLS:
                self.terminal_call = terminal_call
                for tc in reversed(self.tool_calls):
                    if tc["name"] == terminal_call:
                        tc["is_terminal"] = True
                        break
            if not self._saw_usage and outcome in ("crashed", "timeout"):
                # A run that died mid-first-call has unknowable usage, not zero usage.
                self.token_counts_complete = False
        ended_at = int(self._clock())
        with self._lock:
            if ended_at < self.started_at:
                ended_at = self.started_at
        record, message_events = self._assemble(
            outcome=outcome,
            error_class=error_class,
            exit_code=exit_code,
            agent_messages=agent_messages,
            ended_at=ended_at,
        )
        if message_events:
            # Merge the buffered events under the lock, then re-snapshot so the
            # record's log carries the union (tool-call events + message events).
            with self._lock:
                self.redaction_log.extend(message_events)
                record["redaction_log"] = list(self.redaction_log)
        try:
            from trace_writer import append_trace

            path = append_trace(record, self.trace_root, now=float(ended_at))
            logger.info("run trace written: run_id=%s outcome=%s path=%s", self.run_id, outcome, path)
            return True
        except Exception:
            logger.warning("run trace append failed (run_id=%s outcome=%s)", self.run_id, outcome, exc_info=True)
            return False

    def _assemble(
        self,
        *,
        outcome: str,
        error_class: Optional[str],
        exit_code: Optional[int],
        agent_messages: Optional[List[Dict[str, Any]]],
        ended_at: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        session_id = self.session_id or f"run-{self.run_id}"
        # Unlocked: the redactor runs on raw message content, events buffer locally.
        messages, message_events = self._snapshot_messages(agent_messages)
        total = self.tokens["input"] + self.tokens["output"] + self.tokens["cache_read"] + self.tokens["cache_write"]
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": session_id,
            "profile_slug": self.profile_slug,
            "board_slug": self.board_slug,
            "task_id": self.task_id,
            "trace_id": str(uuid.uuid4()),
            "model": self.model or "unknown",
            "provider": self.provider or "unknown",
            "harness_version": {
                "prompt_builder_sha": str(self.harness_version.get("prompt_builder_sha") or ("0" * 40)).lower(),
                "config_sha256": str(self.harness_version.get("config_sha256") or ("0" * 64)).lower(),
            },
            "messages": messages,
            "tool_calls": list(self.tool_calls),
            "outcome": outcome,
            "terminal_call": self.terminal_call,
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "error_class": error_class,
            "killswitch": self.killswitch,
            "safety_tier_hit": self.safety_tier_hit,
            "token_counts": {
                "input": max(0, self.tokens["input"]),
                "output": max(0, self.tokens["output"]),
                "cache_read": max(0, self.tokens["cache_read"]),
                "cache_write": max(0, self.tokens["cache_write"]),
                "total": max(0, total),
            },
            "cost_usd": self.cost_usd,
            "token_counts_complete": bool(self.token_counts_complete),
            "started_at": self.started_at,
            "ended_at": ended_at,
            "wall_clock_seconds": ended_at - self.started_at,
        }
        if self.redaction_log:
            record["redaction_log"] = list(self.redaction_log)
        return record, message_events

    def _snapshot_messages(
        self, agent_messages: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Project the live conversation into schema message shape, content redacted.

        Runs unlocked during close(): redaction events buffer into a local list
        returned alongside the messages; the caller merges them under _lock.
        """
        out: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        try:
            import agent.trace_redactor as _red_mod

            redact = getattr(_red_mod, "redact", None)
            RedactionContext = getattr(_red_mod, "RedactionContext", None)
        except Exception:
            redact, RedactionContext = None, None
        for index, msg in enumerate(agent_messages or []):
            role = msg.get("role")
            if role not in _MESSAGE_ROLES:
                continue
            content = msg.get("content")
            if content is not None and not isinstance(content, str):
                # Multimodal envelopes: keep text parts, mark binary parts.
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                            parts.append(part["text"])
                        else:
                            parts.append("[non-text content omitted]")
                    content = "\n".join(parts)
                else:
                    content = str(content)
            if content is not None and redact is not None and RedactionContext is not None:
                try:
                    ctx = RedactionContext(path=f"messages[{index}].content")
                    redacted = redact(content, context=ctx)
                    # Buffer locally — close() merges under the lock after assembly.
                    events.extend(
                        e for e in getattr(ctx, "log", [])
                        if isinstance(e, dict) and e.get("path") and e.get("reason")
                    )
                    if isinstance(redacted, str) or redacted is None:
                        content = redacted
                    else:
                        content = redacted  # marker object (whole content redacted)
                except Exception:
                    content = {"redacted": True, "reason": "redactor_error", "sha256": None}
            out.append(
                {
                    "role": role,
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id") if role == "tool" else None,
                    "name": msg.get("name") if role == "tool" else None,
                    "index": index,
                }
            )
        if not out:
            out.append({"role": "system", "content": None, "tool_call_id": None, "name": None, "index": 0})
        if not any(m["role"] == "system" for m in out):
            out.insert(0, {"role": "system", "content": None, "tool_call_id": None, "name": None, "index": 0})
            for i, m in enumerate(out):
                m["index"] = i
        return out, events


def _next_run_id() -> int:
    """Non-Kanban runs still need an integer run_id; monotonic per process,
    namespaced to the second so cross-process collisions are improbable."""
    return int(time.time())


def start_run_trace(
    *,
    run_id: Optional[int] = None,
    profile_slug: str = "",
    model: str = "",
    provider: str = "",
    board_slug: Optional[str] = None,
    task_id: Optional[str] = None,
    harness_version: Optional[Dict[str, str]] = None,
    trace_root: Optional[str] = None,
) -> Optional[RunTraceContext]:
    """Open the run's trace context. Returns None when disabled or already open."""
    global _ACTIVE
    if not run_trace_enabled():
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    try:
        _ACTIVE = RunTraceContext(
            run_id=run_id if isinstance(run_id, int) else _next_run_id(),
            profile_slug=profile_slug or (os.environ.get("HERMES_PROFILE") or "unknown"),
            model=model or "unknown",
            provider=provider or "unknown",
            board_slug=board_slug,
            task_id=task_id,
            harness_version=harness_version,
            trace_root=trace_root,
        )
    except Exception:
        logger.debug("run trace open failed", exc_info=True)
        _ACTIVE = None
    return _ACTIVE


def record_provider_usage(agent: Any, usage: Optional[Dict[str, Any]], cost_usd: Optional[float]) -> None:
    """Hook for ``agent/turn_usage.record_response_usage``."""
    ctx = _ACTIVE
    if ctx is None:
        return
    try:
        ctx.set_identity(
            session_id=str(getattr(agent, "session_id", "") or ""),
            model=str(getattr(agent, "model", "") or ""),
            provider=str(getattr(agent, "provider", "") or ""),
        )
        ctx.record_usage(usage, cost_usd)
    except Exception:
        logger.debug("trace usage fold failed", exc_info=True)


def record_usage_gap() -> None:
    """A provider call completed with no usage object (crashed mid-call, etc.)."""
    ctx = _ACTIVE
    if ctx is None:
        return
    with suppress(Exception):
        ctx.record_usage(None, None)


def record_tool_call(
    name: str,
    arguments: Any,
    result: Any,
    *,
    call_id: str = "",
    duration_ms: int = 0,
    exit_code: Optional[int] = None,
) -> None:
    """Hook for ``agent/tool_executor._commit_tool_result``."""
    ctx = _ACTIVE
    if ctx is None:
        return
    try:
        ctx.record_tool_call(
            name, arguments, result, call_id=call_id, duration_ms=duration_ms, exit_code=exit_code
        )
    except Exception:
        logger.debug("trace tool-call buffer failed", exc_info=True)


def note_terminal_call(tool_name: str) -> Tuple[Optional[str], Optional[RunTraceContext]]:
    """Called by the terminal kanban handlers *after* the board write lands.

    Returns ``(outcome, ctx)`` — the caller closes with the outcome so the
    trace still lands when the worker process exits before the agent's
    try/finally (the ``os._exit(0)`` signal path)."""
    ctx = _ACTIVE
    if ctx is None or tool_name not in _TERMINAL_CALLS:
        return None, None
    return _TERMINAL_OUTCOMES[tool_name], ctx


def close_run_trace(
    outcome: str,
    *,
    agent: Any = None,
    error: Optional[BaseException] = None,
    exit_code: Optional[int] = None,
    terminal_call: Optional[str] = None,
) -> bool:
    """Idempotent close; safe to call from try/finally on every exit path."""
    global _ACTIVE
    ctx = _ACTIVE
    if ctx is None:
        return False
    error_class = None
    if error is not None:
        error_class = type(error).__name__
    messages = None
    if agent is not None:
        with suppress(Exception):
            ctx.set_identity(
                session_id=str(getattr(agent, "session_id", "") or ""),
                model=str(getattr(agent, "model", "") or ""),
                provider=str(getattr(agent, "provider", "") or ""),
            )
        messages = getattr(agent, "messages", None)
        if not isinstance(messages, list):
            messages = None
    try:
        return ctx.close(
            outcome=outcome,
            terminal_call=terminal_call,
            error_class=error_class,
            exit_code=exit_code,
            agent_messages=messages,
        )
    finally:
        # Emit-once regardless of write success: a retry would double-append.
        _ACTIVE = None


def close_from_terminal_handler(tool_name: str) -> None:
    """kanban terminal handlers: the board write landed, close the trace now.

    The worker's signal path (``os._exit(0)`` on SIGTERM) skips the agent's
    try/finally, so the trace must be written at the terminal transition, not
    left to process teardown.
    """
    outcome, ctx = note_terminal_call(tool_name)
    if ctx is None or outcome is None:
        return
    try:
        close_run_trace(outcome, terminal_call=tool_name, exit_code=0)
    except Exception:  # close_run_trace is fail-silent internally; belt for the belt
        logger.debug("trace close for %s failed", tool_name, exc_info=True)
