"""Action dependency graph recorder for the loop-diagnostics subsystem.

Implements the recorder half of ``docs/loop-diagnostics-design.md`` against
the machine-readable contract in
``hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json``.

The recorder turns observer-hook events (pre_tool_call / post_tool_call /
agent turns / subagent boundaries / session finalize) into a dependency
graph of worker actions:

  * every action is a node with a stable id ``"<run_id>:<seq>"``,
  * causal/data/loop/retry dependencies become directed ``edge`` records,
  * nested and repeated loop iterations keep distinct ``loop_id`` /
    ``iteration`` identity so they are never conflated.

Design rules enforced here (from the contract):

  * Disabled means *zero cost*: nothing in this module runs unless the
    plugin registered hooks AND the ``kanban.loop_diagnostics.enabled``
    config is true.  There is no import-time side effect and no per-tool
    overhead on the uninstrumented path.
  * Redaction is by construction: only schema keys are ever written, all
    free-text fields go through the redaction helpers, raw tool args /
    results / prompts are never persisted, and every result body is stored
    as a deterministic SHA-256 of its redacted text.
  * The recorder is fail-open: a redaction or write error must never
    disturb the worker, so every public entry point is wrapped and logged
    at debug level.
  * Memory / disk growth is bounded: the in-memory window is capped at
    ``max_events_per_run`` and on-disk traces rotate per run via
    ``retain_runs`` pruning.
  * Incomplete actions (crash / kill / timeout / cancellation) are
    tolerated: ``action_end`` may never arrive; the trace simply keeps the
    ``action_start`` node, and the engine treats missing ends as
    ``status=unknown`` per the contract.

The module is deliberately backend-neutral: it knows nothing about SQLite,
kanban DB rows, or the dispatcher.  Identity comes from environment
variables the dispatcher already pins on worker spawn
(``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_RUN_ID`` / ``HERMES_KANBAN_BOARD``)
plus observer correlation ids, so it can be driven by hooks in any process
that carries those variables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "hermes.loop_diagnostics.v1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_EVENTS_PER_RUN = 10000
DEFAULT_RETAIN_RUNS = 20
DEFAULT_ENABLED = False

# Static allowlist used to derive ``data`` edges cheaply. A producer tool
# creates a persistent artifact (file, note, result set); a consumer tool
# reads artifacts. Sequential actions that form a producer/consumer pair get
# a ``data`` edge; everything else gets ``causal``.  This mirrors the
# contract's "recorder rule (simplification)" in §5.
_PRODUCER_TOOLS = frozenset({
    "write_file", "patch", "terminal", "web_search", "web_extract",
    "image_generate", "text_to_speech", "kanban_attach", "kanban_attach_url",
    "execute_code", "skill_manage",
})
_CONSUMER_TOOLS = frozenset({
    "read_file", "search_files", "terminal", "web_extract", "execute_code",
    "skill_view", "browser_navigate", "browser_snapshot",
})

# Tools whose arguments can contain raw command strings or file contents.
# Their summaries are always replaced with a bare redacted placeholder.
_RAW_ARG_TOOLS = frozenset({
    "terminal", "execute_code", "write_file", "patch",
})

# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------

_MAX_SUMMARY_CHARS = 512
_MAX_ERROR_CHARS = 1024
_MAX_ERROR_TYPE_CHARS = 128

_REDACTED = "<redacted>"

# Match long base64/hex blobs and long tokens so summaries stay bounded
# even before the char cap is applied.
_BLOB_RE = re.compile(r"[A-Za-z0-9+/=_-]{64,}")
_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[~./]?[A-Za-z0-9_./-]*\.(?:py|js|ts|json|yaml|yml|md|txt|log|db|sqlite|sh)(?![A-Za-z0-9])")


def _redact_secrets(text: str) -> str:
    """Strip known secret patterns from a string.

    Uses the same regex engine the rest of Hermes uses for log redaction
    (``agent.redact.redact_sensitive_text``) when available, and falls back
    to a conservative local scrubber otherwise.  Never raises.
    """
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        pass
    return _BLOB_RE.sub(_REDACTED, text)


def _redact_summary(text: Optional[str]) -> Optional[str]:
    """Redact + truncate a one-line summary (action start or end)."""
    if text is None:
        return None
    text = str(text)
    text = _redact_secrets(text)
    text = _PATH_RE.sub("<path>", text)
    if len(text) > _MAX_SUMMARY_CHARS:
        text = text[:_MAX_SUMMARY_CHARS] + "..."
    return text


def _redact_error_message(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = _redact_secrets(str(text))
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "..."
    return text


def _redact_error_type(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text)[:_MAX_ERROR_TYPE_CHARS]
    return text


def _result_hash(redacted_text: Optional[str]) -> Optional[str]:
    """Deterministic SHA-256 of the *redacted* result text.

    Equivalent sensitive content hashes identically across runs so the
    diagnosis engine can detect repeated identical failures without ever
    storing the bodies.
    """
    if redacted_text is None:
        return None
    return hashlib.sha256(redacted_text.encode("utf-8", "replace")).hexdigest()


def _summarize_tool_args(tool_name: str, args: Any) -> str:
    """Build a safe, short one-line summary of what a tool call is doing.

    Never includes raw argument values; only the tool name and (for safe
    tools) a tiny amount of structural context like a path basename.
    """
    name = str(tool_name or "tool")
    if name in _RAW_ARG_TOOLS:
        return f"ran {name}"
    if name == "read_file":
        path = _safe_arg(args, "path")
        return f"read {_basename(path)}" if path else f"ran {name}"
    if name == "web_search":
        query = _safe_arg(args, "query")
        if query:
            return f"search: {_redact_summary(str(query))[:80]}"
        return "web search"
    if name == "web_extract":
        urls = _safe_arg(args, "urls")
        if isinstance(urls, list) and urls:
            return f"extract {len(urls)} url(s)"
        return "web extract"
    if name == "write_file":
        path = _safe_arg(args, "path")
        return f"write {_basename(path)}" if path else f"ran {name}"
    if name == "kanban_comment":
        return "kanban comment"
    if name == "kanban_complete":
        return "kanban complete"
    if name == "delegate_task":
        return "delegate task"
    return f"ran {name}"


def _safe_arg(args: Any, key: str) -> Any:
    if not isinstance(args, dict):
        return None
    return args.get(key)


def _basename(path: Any) -> str:
    if not path:
        return ""
    try:
        return Path(str(path)).name or str(path)
    except Exception:
        return str(path)[:80]


# ---------------------------------------------------------------------------
# Trace storage
# ---------------------------------------------------------------------------


def loop_traces_dir(board: Optional[str] = None) -> Path:
    """Return the per-board loop-traces root directory.

    Mirrors ``worker_logs_dir`` in ``hermes_cli/kanban_db.py``: the
    ``default`` board keeps the legacy root under ``<kanban-home>/kanban/``,
    all other boards live under ``<kanban-home>/kanban/boards/<slug>/``.
    """
    from hermes_cli.kanban_db import board_dir, DEFAULT_BOARD, get_current_board

    slug = board or os.environ.get("HERMES_KANBAN_BOARD", "") or get_current_board()
    if slug == DEFAULT_BOARD:
        from hermes_cli.kanban_db import kanban_home
        return kanban_home() / "kanban" / "loop-traces"
    return board_dir(slug) / "loop-traces"


class TraceWriter:
    """Append-only JSONL writer with per-run cap and retention pruning."""

    def __init__(
        self,
        task_id: Optional[str],
        run_id: Optional[int],
        *,
        board: Optional[str] = None,
        base_dir: Optional[Path] = None,
        max_events_per_run: int = DEFAULT_MAX_EVENTS_PER_RUN,
        retain_runs: int = DEFAULT_RETAIN_RUNS,
    ):
        if not task_id or run_id is None:
            raise ValueError("TraceWriter requires task_id and run_id")
        self.task_id = task_id
        self.run_id = run_id
        self.max_events_per_run = max(1, int(max_events_per_run))
        self.retain_runs = max(1, int(retain_runs))
        self._base = Path(base_dir) if base_dir else loop_traces_dir(board)
        self.dir = self._base / task_id
        self.path = self.dir / f"{run_id}.jsonl"
        self._fh: Any = None
        self._count = 0
        self._edge_count = 0
        self._truncated = False
        self._closed = False
        self._lock = threading.Lock()

    def _open(self) -> None:
        if self._fh is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> bool:
        """Append one record; returns True when written, False when capped."""
        with self._lock:
            if self._closed:
                return False
            kind = record.get("kind")
            if kind not in ("run_header", "run_footer"):
                if self._count >= self.max_events_per_run:
                    if not self._truncated:
                        self._truncated = True
                        logger.debug(
                            "loop-diagnostics: run %s:%s hit event cap %d; truncating",
                            self.task_id, self.run_id, self.max_events_per_run,
                        )
                    return False
            try:
                self._open()
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                self._fh.write(line + "\n")
                self._fh.flush()
                if kind not in ("run_header", "run_footer"):
                    self._count += 1
                    if kind == "edge":
                        self._edge_count += 1
                return True
            except Exception as exc:
                logger.debug("loop-diagnostics: write failed (%s)", exc)
                return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None

    def prune(self) -> None:
        """Keep only the newest ``retain_runs`` run dirs for this task."""
        try:
            if not self.dir.exists():
                return
            runs: List[Tuple[int, Path]] = []
            for p in self.dir.iterdir():
                if p.suffix != ".jsonl":
                    continue
                try:
                    runs.append((int(p.stem), p))
                except ValueError:
                    continue
            runs.sort(key=lambda t: t[0])
            for _, p in runs[:-self.retain_runs]:
                try:
                    p.unlink()
                except OSError:
                    pass
                diag = p.with_suffix(".diagnosis.json")
                try:
                    if diag.exists():
                        diag.unlink()
                except OSError:
                    pass
        except Exception as exc:
            logger.debug("loop-diagnostics: prune failed (%s)", exc)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class LoopDiagnosticsRecorder:
    """Collects hook events into an action dependency graph + trace file.

    One instance lives for the lifetime of a worker process.  Identity is
    resolved once at construction from the environment the dispatcher pins,
    so a subagent or nested goal loop that inherits those variables records
    into the same run without conflating iterations.
    """

    def __init__(
        self,
        *,
        task_id: Optional[str] = None,
        run_id: Optional[int] = None,
        board: Optional[str] = None,
        base_dir: Optional[Path] = None,
        max_events_per_run: int = DEFAULT_MAX_EVENTS_PER_RUN,
        retain_runs: int = DEFAULT_RETAIN_RUNS,
        enabled: Optional[bool] = None,
    ):
        task_id = task_id or os.environ.get("HERMES_KANBAN_TASK", "").strip() or None
        raw_run = run_id
        if raw_run is None:
            try:
                raw_run = int(os.environ.get("HERMES_KANBAN_RUN_ID", "") or "")
            except (TypeError, ValueError):
                raw_run = None
        self.task_id = task_id
        self.run_id = raw_run
        self.board = board or os.environ.get("HERMES_KANBAN_BOARD", "") or None
        # ``enabled`` default resolves from ``kanban.loop_diagnostics.enabled``
        # so a recorder constructed with no explicit flag (plugin / worker)
        # honours the operator's config instead of always defaulting True.
        if enabled is None:
            try:
                enabled = bool(load_recorder_config().get("enabled", DEFAULT_ENABLED))
            except Exception:
                enabled = DEFAULT_ENABLED
        self.enabled = bool(enabled) and self.task_id is not None and self.run_id is not None
        self._max_events = max(1, int(max_events_per_run))
        self._retain = max(1, int(retain_runs))
        self._base = Path(base_dir) if base_dir else None

        self._seq = 0
        self._lock = threading.Lock()
        self._writer: Optional[TraceWriter] = None
        self._started = False
        self._closed = False

        # In-memory node table: action_id -> {kind, tool_name, loop_id,
        # iteration, ts, status, result_hash, parent_action_id, turn_id}
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._loop_iterations: Dict[str, int] = {}
        self._last_by_turn: Dict[str, Optional[str]] = {}
        self._last_in_loop: Dict[str, str] = {}
        self._last_global: Optional[str] = None
        self._tool_action_ids: Dict[str, str] = {}
        self._pending_by_tool: Dict[Tuple[str, str], str] = {}
        self._subagent_action_ids: Dict[str, str] = {}

        self._attempt = 1
        self._goal_mode = os.environ.get("HERMES_KANBAN_GOAL_MODE") == "1"

    # -- lifecycle ------------------------------------------------------

    def _resolve_writer(self) -> TraceWriter:
        if self._writer is None:
            # Only reachable when enabled, which guarantees task/run ids.
            if not self.task_id or self.run_id is None:
                logger.debug("loop-diagnostics: missing identity; disabling")
                self.enabled = False
                raise RuntimeError("recorder disabled: missing task/run identity")
            self._writer = TraceWriter(
                self.task_id,
                self.run_id,
                board=self.board,
                base_dir=self._base,
                max_events_per_run=self._max_events,
                retain_runs=self._retain,
            )
        return self._writer

    def start(self) -> None:
        """Write the run_header once.  Safe to call multiple times."""
        if self._closed or not self.enabled or self._started:
            return
        self._started = True
        header = {
            "schema_version": SCHEMA_VERSION,
            "kind": "run_header",
            "task_id": self.task_id,
            "run_id": self.run_id,
            "attempt": self._attempt,
            "profile": os.environ.get("HERMES_PROFILE") or None,
            "goal_mode": self._goal_mode,
            "ts": int(time.time()),
        }
        self._resolve_writer().write(header)

    def finish(self, *, outcome: str = "completed", error: Optional[str] = None) -> None:
        """Write the run_footer and close the trace.  Idempotent.

        If ``start()`` was never called (e.g. the process was killed before
        the first event), the footer is still written to a freshly-resolved
        writer so a partial trace always ends with a valid footer — the
        schema requires header+footer pairing to be tolerant of incomplete
        runs.
        """
        if self._closed or not self.enabled:
            return
        self._closed = True
        footer = {
            "schema_version": SCHEMA_VERSION,
            "kind": "run_footer",
            "task_id": self.task_id,
            "run_id": self.run_id,
            "ts": int(time.time()),
            "outcome": outcome,
            "error": _redact_error_message(error),
            "event_count": self._event_count(),
        }
        writer = self._resolve_writer()
        writer.write(footer)
        writer.close()
        writer.prune()

    def _event_count(self) -> int:
        """Number of trace events written, excluding header/footer.

        Each action contributes one action_start and, when ended, one
        action_end; edges add one record per dependency.  Cancelled or
        crashed actions contribute only their start (no end, no edges).
        """
        starts = len(self._nodes)
        ends = sum(1 for n in self._nodes.values() if n.get("ended"))
        return starts + ends + self._edge_count

    @property
    def _edge_count(self) -> int:
        if self._writer is None:
            return 0
        return self._writer._edge_count

    # -- identity -------------------------------------------------------

    def _next_action_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"{self.run_id}:{self._seq}"

    def _loop_iteration(self, loop_id: Optional[str]) -> Optional[int]:
        """Return the next 0-based iteration for a loop_id (monotonic)."""
        if loop_id is None:
            return None
        with self._lock:
            idx = self._loop_iterations.get(loop_id, 0)
            self._loop_iterations[loop_id] = idx + 1
            return idx

    # -- record construction --------------------------------------------

    def _base_record(self, kind: str, **extra: Any) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "task_id": self.task_id,
            "run_id": self.run_id,
        }
        rec.update(extra)
        return rec

    def record_action_start(
        self,
        *,
        action_kind: str = "tool_call",
        tool_name: Optional[str] = None,
        summary: Optional[str] = None,
        parent_action_id: Optional[str] = None,
        loop_id: Optional[str] = None,
        iteration: Optional[int] = None,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        api_request_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """Create a new action node; returns the action_id or None when
        disabled / capped.  Never raises."""
        try:
            if self._closed or not self.enabled:
                return None
            if self._count_at_cap():
                return None
            # First event implies the run header so a trace that starts
            # with a tool call (hooks fired before start() was called) still
            # gets a valid header.
            self.start()
            if self._closed or not self.enabled:
                return None
            action_id = self._next_action_id()
            node = {
                "kind": action_kind or "tool_call",
                "tool_name": tool_name,
                "parent_action_id": parent_action_id,
                "loop_id": loop_id,
                "iteration": iteration,
                "ts": int(time.time()),
                "turn_id": turn_id,
                "status": None,
                "result_hash": None,
                "error_type": None,
                "error_message": None,
                "ended": False,
            }
            with self._lock:
                self._nodes[action_id] = node
            rec = self._base_record(
                "action_start",
                action_id=action_id,
                parent_action_id=parent_action_id,
                loop_id=loop_id,
                iteration=iteration,
                ts=node["ts"],
                action_kind=node["kind"],
                tool_name=tool_name,
                summary=_redact_summary(summary),
                model=model,
                turn_id=turn_id,
                api_request_id=api_request_id,
                tool_call_id=tool_call_id,
            )
            self._write_or_drop(rec)
            return action_id
        except Exception as exc:
            logger.debug("loop-diagnostics: action_start failed (%s)", exc)
            return None

    def record_action_end(
        self,
        *,
        action_id: Optional[str],
        status: str = "ok",
        duration_ms: int = 0,
        summary: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        result_text: Optional[str] = None,
    ) -> bool:
        """Close an action node and derive edges from its predecessor.

        Returns True when the node was found and an ``action_end`` record
        was attempted.  Never raises.
        """
        try:
            if self._closed or not self.enabled or not action_id:
                return False
            with self._lock:
                node = self._nodes.get(action_id)
                if node is None or node.get("ended"):
                    return False
                node["ended"] = True
                node["status"] = status
                node["result_hash"] = _result_hash(_redact_secrets(str(result_text))) if result_text is not None else None
                node["error_type"] = _redact_error_type(error_type)
                node["error_message"] = _redact_error_message(error_message)
                node["duration_ms"] = int(duration_ms or 0)
            rec = self._base_record(
                "action_end",
                action_id=action_id,
                ts=int(time.time()),
                status=status,
                duration_ms=int(duration_ms or 0),
                summary=_redact_summary(summary),
                error_type=node["error_type"],
                error_message=node["error_message"],
                result_hash=node["result_hash"],
            )
            self._write_or_drop(rec)
            self._emit_edges(action_id)
            return True
        except Exception as exc:
            logger.debug("loop-diagnostics: action_end failed (%s)", exc)
            return False

    def _write_or_drop(self, rec: Dict[str, Any]) -> None:
        """Write a record via the writer, resolving it lazily.

        ``start()`` normally resolves the writer when the header is written;
        this is the safety net for records that arrive before ``start()``.
        Never raises.
        """
        try:
            if self._writer is None:
                self.start()
            if self._writer is not None:
                self._writer.write(rec)
        except Exception as exc:
            logger.debug("loop-diagnostics: record dropped (%s)", exc)

    def _count_at_cap(self) -> bool:
        if self._writer is not None:
            return self._writer._count >= self._writer.max_events_per_run
        return False

    # -- edge derivation --------------------------------------------------

    def _emit_edges(self, action_id: str) -> None:
        """Derive and write dependency edges for a completed action.

        Rules (contract §5, recorder simplification):
          * loop actions (loop_id set) get a ``loop`` edge from the
            previous iteration of the same loop_id.
          * retries (same tool as the immediately-previous action in the
            same turn, and the previous one errored) get a ``retry`` edge.
          * producer/consumer pairs get a ``data`` edge.
          * otherwise a ``causal`` edge from the last completed action in
            the same turn (or the last global action when turn is unknown).
        Edges are emitted only after the consumer's action_end, so a
        cancelled action never emits a spurious edge.
        """
        try:
            with self._lock:
                node = self._nodes.get(action_id)
                if node is None:
                    return
                loop_id = node.get("loop_id")
                turn_id = node.get("turn_id")
                tool_name = node.get("tool_name")
                # Subagent children spawned by an in-flight parent action
                # (delegate_task) fan out from the PARENT, not from the
                # previous sibling.  A child is an independent branch; the
                # graph must show parent -> child edges, never a sequential
                # sibling chain that would collapse concurrent failures into
                # one causal lineage.
                parent_id = node.get("parent_action_id")
                if parent_id is not None and parent_id in self._nodes:
                    edge_kind = self._classify_edge(parent_id, action_id)
                    self._write_edge(parent_id, action_id, edge_kind)
                    # Do NOT advance _last_by_turn / _last_global here: the
                    # parent action (delegate_task) is still in flight and
                    # must chain from ITS predecessor when it completes, not
                    # from a child. Siblings fan out from the shared parent;
                    # the parent remains the turn anchor.
                    return
                prev: Optional[str] = None

                if loop_id is not None:
                    # Link from the previous iteration of this loop.
                    prev = self._last_in_loop.get(loop_id)
                    self._last_in_loop[loop_id] = action_id

                if prev is None:
                    prev = self._last_by_turn.get(turn_id) if turn_id else None

                if prev is None:
                    # First action of a new turn (or turn unknown): the causal
                    # predecessor is the last action anywhere in the run, so
                    # cross-turn chains stay connected.
                    prev = self._last_global

                from_id = prev
                if from_id is not None and from_id != action_id:
                    edge_kind = self._classify_edge(from_id, action_id)
                    self._write_edge(from_id, action_id, edge_kind)

                if turn_id:
                    self._last_by_turn[turn_id] = action_id
                self._last_global = action_id
        except Exception as exc:
            logger.debug("loop-diagnostics: edge derivation failed (%s)", exc)

    def _classify_edge(self, from_id: str, to_id: str) -> str:
        frm = self._nodes.get(from_id)
        to = self._nodes.get(to_id)
        if frm is None or to is None:
            return "causal"
        f_tool = frm.get("tool_name")
        t_tool = to.get("tool_name")
        # Loop: same loop instance AND the target is a later iteration.  Two
        # actions with the same loop_id but the same iteration are steps of
        # one iteration, so they stay causal/data — never a loop edge.  This
        # is the no-conflation rule from the contract.  Checked BEFORE retry:
        # an explicit loop iteration is a loop edge even when the tool is
        # unchanged and the previous iteration errored (a repeating search /
        # retry-with-budget loop), so the engine can classify loop_repeated
        # vs retry_exhausted from the edge kind.
        f_loop = frm.get("loop_id")
        f_iter = frm.get("iteration")
        t_iter = to.get("iteration")
        if (
            f_loop is not None
            and f_loop == to.get("loop_id")
            and f_iter is not None
            and t_iter is not None
            and t_iter != f_iter
        ):
            return "loop"
        # Retry: same tool, consecutive, previous errored.  Does NOT depend
        # on loop_id so ordinary sequential same-tool calls after a failure
        # are recognised as retries without being conflated into a loop.
        if f_tool and f_tool == t_tool and frm.get("status") == "error":
            return "retry"
        # Data: producer -> consumer pair.
        if f_tool in _PRODUCER_TOOLS and t_tool in _CONSUMER_TOOLS:
            return "data"
        return "causal"

    def _write_edge(self, from_id: str, to_id: str, edge_kind: str) -> None:
        rec = self._base_record(
            "edge",
            from_action_id=from_id,
            to_action_id=to_id,
            edge_kind=edge_kind,
            ts=int(time.time()),
        )
        self._write_or_drop(rec)

    # -- hook entry points ------------------------------------------------

    def on_pre_tool_call(self, **kwargs: Any) -> None:
        try:
            if not self.enabled:
                return
            tool_name = kwargs.get("tool_name") or ""
            args = kwargs.get("args")
            summary = _summarize_tool_args(tool_name, args)
            action_id = self.record_action_start(
                action_kind="tool_call",
                tool_name=tool_name,
                summary=summary,
                turn_id=kwargs.get("turn_id"),
                tool_call_id=kwargs.get("tool_call_id"),
                api_request_id=kwargs.get("api_request_id"),
                loop_id=kwargs.get("loop_id"),
                iteration=kwargs.get("iteration"),
            )
            if action_id is not None:
                self._tool_action_ids[kwargs.get("tool_call_id") or ""] = action_id
                self._pending_by_tool[(tool_name, kwargs.get("turn_id") or "")] = action_id
        except Exception as exc:
            logger.debug("loop-diagnostics: pre_tool_call failed (%s)", exc)

    def on_post_tool_call(self, **kwargs: Any) -> None:
        try:
            if not self.enabled:
                return
            action_id = self._action_id_for_post(kwargs)
            if action_id is None:
                return
            status = kwargs.get("status") or "ok"
            result = kwargs.get("result")
            self.record_action_end(
                action_id=action_id,
                status=status,
                duration_ms=int(kwargs.get("duration_ms") or 0),
                summary=self._result_summary(status, result),
                error_type=kwargs.get("error_type"),
                error_message=kwargs.get("error_message"),
                result_text=result if status != "ok" else None,
            )
        except Exception as exc:
            logger.debug("loop-diagnostics: post_tool_call failed (%s)", exc)

    def _action_id_for_post(self, kwargs: Dict[str, Any]) -> Optional[str]:
        tool_call_id = kwargs.get("tool_call_id") or ""
        if tool_call_id and tool_call_id in self._tool_action_ids:
            return self._tool_action_ids.pop(tool_call_id, None)
        tool_name = kwargs.get("tool_name") or ""
        turn_id = kwargs.get("turn_id") or ""
        key = (tool_name, turn_id)
        if key in self._pending_by_tool:
            return self._pending_by_tool.pop(key, None)
        return None

    def on_llm_turn_end(self, **kwargs: Any) -> None:
        """Record a synthetic llm_call node at each agent turn end.

        Gives the goal loop a per-turn anchor so the diagnosis engine can
        see turn boundaries even when no tool ran.  Cheap and bounded; the
        node has no result body.
        """
        try:
            if not self.enabled:
                return
            turn_id = kwargs.get("turn_id") or ""
            if not turn_id:
                return
            action_id = self.record_action_start(
                action_kind="llm_call",
                summary="agent turn",
                turn_id=turn_id,
                model=kwargs.get("model"),
            )
            if action_id is not None:
                self.record_action_end(
                    action_id=action_id,
                    status="ok",
                    duration_ms=int(kwargs.get("duration_ms") or 0),
                    summary="agent turn complete",
                )
        except Exception as exc:
            logger.debug("loop-diagnostics: llm_turn_end failed (%s)", exc)

    def on_subagent_start(self, **kwargs: Any) -> None:
        try:
            if not self.enabled:
                return
            parent_turn_id = kwargs.get("parent_turn_id") or ""
            # The subagent is spawned by an in-flight delegate_task tool
            # call; link the child to that action so the graph shows the
            # real parent->child branch, not a sequential sibling chain.
            parent_action_id = self._pending_by_tool.get(
                ("delegate_task", parent_turn_id)
            )
            action_id = self.record_action_start(
                action_kind="subagent",
                summary="spawn subagent",
                parent_action_id=parent_action_id,
                turn_id=parent_turn_id,
                loop_id=kwargs.get("loop_id"),
                iteration=kwargs.get("iteration"),
            )
            if action_id is not None:
                self._subagent_action_ids[kwargs.get("child_session_id") or ""] = action_id
        except Exception as exc:
            logger.debug("loop-diagnostics: subagent_start failed (%s)", exc)

    def on_subagent_stop(self, **kwargs: Any) -> None:
        try:
            if not self.enabled:
                return
            action_id = self._subagent_action_ids.pop(
                kwargs.get("child_session_id") or "", None
            )
            if action_id is None:
                return
            status = "ok" if str(kwargs.get("child_status") or "").lower() in ("ok", "done", "completed") else "error"
            summary = kwargs.get("child_summary")
            self.record_action_end(
                action_id=action_id,
                status=status,
                duration_ms=int(kwargs.get("duration_ms") or 0),
                summary=_redact_summary(str(summary)[:200]) if summary else "subagent finished",
                result_text=None,
            )
        except Exception as exc:
            logger.debug("loop-diagnostics: subagent_stop failed (%s)", exc)

    # -- helpers ---------------------------------------------------------

    def _result_summary(self, status: str, result: Any) -> str:
        if status == "ok":
            return "ok"
        return "failed"


def load_recorder_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read the loop-diagnostics config block with defaults.

    Accepts an already-loaded config dict, or loads via
    ``hermes_cli.config.load_config`` when None.
    """
    cfg: Dict[str, Any] = {}
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception:
            config = {}
    kb = (config or {}).get("kanban") or {}
    ld = kb.get("loop_diagnostics") or {}
    try:
        enabled = bool(ld.get("enabled", DEFAULT_ENABLED))
    except Exception:
        enabled = DEFAULT_ENABLED
    try:
        max_events = int(ld.get("max_events_per_run", DEFAULT_MAX_EVENTS_PER_RUN))
    except (TypeError, ValueError):
        max_events = DEFAULT_MAX_EVENTS_PER_RUN
    try:
        retain = int(ld.get("retain_runs", DEFAULT_RETAIN_RUNS))
    except (TypeError, ValueError):
        retain = DEFAULT_RETAIN_RUNS
    # Independent gate for the failure-time diagnosis step. Defaults to True
    # so enabling the recorder automatically enables diagnosis; set False to
    # keep recording traces while disabling the diagnostic attach on failure.
    try:
        diagnose_on_failure = bool(ld.get("diagnose_on_failure", True))
    except Exception:
        diagnose_on_failure = True
    return {
        "enabled": enabled,
        "diagnose_on_failure": diagnose_on_failure,
        "max_events_per_run": max(1, max_events),
        "retain_runs": max(1, retain),
    }
