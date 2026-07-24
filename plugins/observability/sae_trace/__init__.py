"""sae_trace — Hermes plugin for SAE feature-trace observability.

Correlates Hermes agent turns with the same-inference SAE (sparse
autoencoder) feature-trace records emitted by an SAE-hooked local
OpenAI-compatible model server, giving a per-session interpretability
trace of what the local model's internals were doing during each turn.

How it works: an SAE-instrumented server (reference implementation:
https://github.com/SolshineCode/hermes-sae) wraps ``model.generate()``
with forward hooks and appends one JSONL record per request to a sidecar
file, in the same inference pass that produced the completion Hermes
receives. This plugin tail-reads that sidecar and, on every
``post_api_request`` observer hook (hermes.observer.v1 contract), matches
new sidecar records against the request's identity and timing, then
writes correlated per-turn records under
``$HERMES_SAE_TRACE_OUT_DIR/<session_id>.jsonl``.

Activation is handled by the Hermes plugin system — standalone plugins
only load when listed in ``plugins.enabled`` (via
``hermes plugins enable observability/sae_trace``). At runtime the
plugin also requires ``HERMES_SAE_TRACE_FILE``; without it the hooks are
inert (fail-open, like the langfuse plugin).

Required env vars (set via ~/.hermes/.env):
  HERMES_SAE_TRACE_FILE     - path to the SAE server's sidecar JSONL
                              (e.g. activations.jsonl / sae_history.jsonl)

Optional env vars:
  HERMES_SAE_TRACE_OUT_DIR  - output directory for correlated per-session
                              traces (default: $HERMES_HOME/sae_trace)
  HERMES_SAE_TRACE_SKEW     - time-window slack in seconds around each
                              API request when matching by timestamp
                              (default: 10)
  HERMES_SAE_TRACE_DEBUG    - set to "true" for verbose logging

Everything here is stdlib-only, read-only with respect to the sidecar,
and local-files-only (no network).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    # In-tree install: shared thread-safety helper from plugins/plugin_utils.py.
    from plugins.plugin_utils import SingletonSlot
except Exception:  # pragma: no cover - exercised via the standalone-load test
    # Standalone install (~/.hermes/plugins/sae_trace/): the plugin loader
    # imports this module as ``hermes_plugins.sae_trace`` from its own
    # directory, where ``plugins.plugin_utils`` may not be importable.
    # Minimal drop-in fallback with the same semantics (double-checked
    # locking; a raising factory caches nothing).
    class SingletonSlot:  # type: ignore[no-redef]
        __slots__ = ("_lock", "_value", "_set")

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._value: Any = None
            self._set = False

        def get(self, factory):
            if self._set:
                return self._value
            with self._lock:
                if self._set:
                    return self._value
                value = factory()
                self._value = value
                self._set = True
                return value

        def peek(self):
            return self._value if self._set else None

        def reset(self) -> None:
            with self._lock:
                self._value = None
                self._set = False


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_DEFAULT_SKEW_SECONDS = 10.0
# Cap on bytes consumed from the sidecar per poll — bounds memory if some
# other process dumps a huge backlog between our reads.
_MAX_POLL_BYTES = 8 * 1024 * 1024
# Unclaimed sidecar records are held for late matching, then dropped.
_PENDING_MAX_RECORDS = 512
_PENDING_TTL_SECONDS = 600.0
# Per-record summary bounds.
_TOP_FEATURES_PER_LAYER = 10
_TEXT_PREVIEW_CHARS = 200


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if _env_bool("HERMES_SAE_TRACE_DEBUG"):
        logger.info("SAE trace: %s", message)
    else:
        logger.debug("SAE trace: %s", message)


# ---------------------------------------------------------------------------
# Config (lazy, cached — a miss is cached too, matching the langfuse plugin's
# runtime-availability gate; tests reset by reloading the module)
# ---------------------------------------------------------------------------

@dataclass
class _Config:
    trace_file: Path
    out_dir: Path
    skew: float


_CONFIG_SLOT: SingletonSlot = SingletonSlot()


def _default_out_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "sae_trace"
    except Exception:
        # Standalone / degraded path: fall back to the same convention
        # get_hermes_home() itself uses.
        home = _env("HERMES_HOME")
        base = Path(home) if home else (Path.home() / ".hermes")
        return base / "sae_trace"


def _build_config() -> Optional[_Config]:
    trace_file = _env("HERMES_SAE_TRACE_FILE")
    if not trace_file:
        _debug("HERMES_SAE_TRACE_FILE not set — hooks are inert")
        return None
    out_dir_raw = _env("HERMES_SAE_TRACE_OUT_DIR")
    out_dir = Path(out_dir_raw).expanduser() if out_dir_raw else _default_out_dir()
    skew = _DEFAULT_SKEW_SECONDS
    raw_skew = _env("HERMES_SAE_TRACE_SKEW")
    if raw_skew:
        try:
            skew = max(0.0, float(raw_skew))
        except ValueError:
            logger.warning("Invalid HERMES_SAE_TRACE_SKEW=%r", raw_skew)
    return _Config(
        trace_file=Path(trace_file).expanduser(),
        out_dir=out_dir,
        skew=skew,
    )


def _get_config() -> Optional[_Config]:
    return _CONFIG_SLOT.get(_build_config)


# ---------------------------------------------------------------------------
# Sidecar tailer — offset-remembering, rotation-aware JSONL reader
# ---------------------------------------------------------------------------

class _SidecarTailer:
    """Tail-read the SAE server's sidecar JSONL efficiently.

    Remembers the byte offset between polls so each poll only reads bytes
    appended since the last one. Handles rotation/truncation (file
    replaced or shrunk → restart from offset 0) and partial trailing
    lines (a record the server is mid-write → left unconsumed until the
    newline arrives). All access is lock-guarded; hooks can fire
    concurrently.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._offset: Optional[int] = None
        self._inode: Optional[int] = None
        self._primed = False
        # Parsed-but-unclaimed records: (wall-clock insert time, record).
        self._pending: Deque[Tuple[float, Dict[str, Any]]] = deque()
        self.records_seen = 0
        self.malformed_lines = 0

    # -- lifecycle ---------------------------------------------------------

    def prime(self) -> None:
        """Skip sidecar history: start tailing from the current EOF.

        Called from ``pre_api_request`` so that only records appended
        *after* the request started are candidates. After the first call
        this is a no-op boolean check.
        """
        if self._primed:
            return
        with self._lock:
            if self._primed:
                return
            try:
                stat = self.path.stat()
                self._offset = stat.st_size
                self._inode = getattr(stat, "st_ino", None) or None
            except OSError:
                # Not created yet — when it appears, read from the start
                # (a fresh file only contains new records).
                self._offset = 0
                self._inode = None
            self._primed = True

    # -- reading -----------------------------------------------------------

    def poll(self) -> None:
        """Consume newly appended sidecar lines into the pending queue."""
        with self._lock:
            self.prime()
            try:
                stat = self.path.stat()
            except OSError:
                return  # fail-open: sidecar missing
            inode = getattr(stat, "st_ino", None) or None
            offset = self._offset or 0
            if stat.st_size < offset or (
                inode is not None and self._inode is not None and inode != self._inode
            ):
                # Rotated or truncated — the new file is all-new content.
                _debug(f"sidecar rotation detected ({self.path}), rereading from 0")
                offset = 0
            self._inode = inode
            if stat.st_size <= offset:
                self._offset = offset
                return
            try:
                with open(self.path, "rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read(_MAX_POLL_BYTES)
            except OSError as exc:
                _debug(f"sidecar read failed: {exc}")
                return
            # Only consume through the last complete line; a partial
            # trailing line is re-read on the next poll.
            newline = chunk.rfind(b"\n")
            if newline < 0:
                self._offset = offset
                return
            consumed = chunk[: newline + 1]
            self._offset = offset + len(consumed)
            now = time.time()
            for raw_line in consumed.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeError):
                    self.malformed_lines += 1
                    _debug("skipping malformed sidecar line")
                    continue
                if not isinstance(record, dict):
                    self.malformed_lines += 1
                    continue
                self.records_seen += 1
                self._pending.append((now, record))
            self._trim_locked(now)

    def _trim_locked(self, now: float) -> None:
        while self._pending and (
            len(self._pending) > _PENDING_MAX_RECORDS
            or now - self._pending[0][0] > _PENDING_TTL_SECONDS
        ):
            self._pending.popleft()

    # -- matching ----------------------------------------------------------

    def claim_matches(
        self,
        *,
        api_request_id: str,
        session_id: str,
        started_at: float,
        ended_at: float,
        skew: float,
        models: Tuple[str, ...],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Pop and return sidecar records correlated with one API request.

        Returns ``(records, confidence)`` where confidence is the
        strongest evidence tier among the matched records:
        ``"request_id"`` > ``"session_id"`` > ``"time_window"``.
        """
        lo = started_at - skew
        hi = ended_at + skew
        matched: List[Tuple[int, Dict[str, Any], str]] = []
        with self._lock:
            remaining: Deque[Tuple[float, Dict[str, Any]]] = deque()
            for item in self._pending:
                _, record = item
                tier = self._match_tier(
                    record,
                    api_request_id=api_request_id,
                    session_id=session_id,
                    lo=lo,
                    hi=hi,
                    models=models,
                )
                if tier is None:
                    remaining.append(item)
                else:
                    matched.append((_TIER_RANK[tier], record, tier))
            self._pending = remaining
        if not matched:
            return [], None
        best = min(rank for rank, _, _ in matched)
        confidence = _TIER_BY_RANK[best]
        return [record for _, record, _ in matched], confidence

    @staticmethod
    def _match_tier(
        record: Dict[str, Any],
        *,
        api_request_id: str,
        session_id: str,
        lo: float,
        hi: float,
        models: Tuple[str, ...],
    ) -> Optional[str]:
        ts = _parse_record_ts(record)
        in_window = ts is not None and lo <= ts <= hi
        rec_request_id = record.get("request_id")
        if api_request_id and rec_request_id and rec_request_id == api_request_id:
            return "request_id"
        rec_session_id = record.get("session_id")
        if session_id and rec_session_id and rec_session_id == session_id:
            # When the record is timestamped, it must also fall inside the
            # window — same-session records from *other* turns should be
            # claimed by their own turn, not this one.
            if ts is None or in_window:
                return "session_id"
            return None
        if in_window and _model_matches(record, models):
            return "time_window"
        return None

    # -- introspection -----------------------------------------------------

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


_TIER_RANK = {"request_id": 0, "session_id": 1, "time_window": 2}
_TIER_BY_RANK = {rank: tier for tier, rank in _TIER_RANK.items()}

_TAILER_SLOT: SingletonSlot = SingletonSlot()


def _get_tailer() -> Optional[_SidecarTailer]:
    config = _get_config()
    if config is None:
        return None
    return _TAILER_SLOT.get(lambda: _SidecarTailer(config.trace_file))


# ---------------------------------------------------------------------------
# Record parsing helpers
# ---------------------------------------------------------------------------

def _parse_record_ts(record: Dict[str, Any]) -> Optional[float]:
    """Parse a sidecar record timestamp to a POSIX epoch float.

    Supports both documented field names: ``timestamp`` (activation-capture
    servers, naive local ISO) and ``ts`` (feature-history servers, UTC ISO
    with a ``Z`` suffix). Numeric values pass through as epoch seconds.
    """
    value = record.get("timestamp", record.get("ts"))
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # Naive timestamps are interpreted as local time — the sidecar is
    # written on the same machine as the agent.
    return parsed.timestamp()


def _model_matches(record: Dict[str, Any], models: Tuple[str, ...]) -> bool:
    rec_model = record.get("model")
    if not isinstance(rec_model, str) or not rec_model:
        return True  # record carries no model — window evidence stands alone
    rec_model_cf = rec_model.casefold()
    return any(m and m.casefold() == rec_model_cf for m in models)


def _summarize_top_features(feats_topk: Any) -> Optional[Dict[str, Any]]:
    """Reduce per-token top-k feature events to a per-layer summary.

    Input shape (sidecar contract): ``{layer: [[token_pos, feature_id,
    activation], ...]}``. Output: ``{layer: [[feature_id, max_activation,
    n_tokens], ...]}`` — top features by max activation, bounded.
    """
    if not isinstance(feats_topk, dict):
        return None
    summary: Dict[str, Any] = {}
    for layer, events in feats_topk.items():
        if not isinstance(events, list):
            continue
        per_feature: Dict[int, Tuple[float, int]] = {}
        for event in events:
            if not isinstance(event, (list, tuple)) or len(event) < 3:
                continue
            try:
                feature_id = int(event[1])
                activation = float(event[2])
            except (TypeError, ValueError):
                continue
            prev = per_feature.get(feature_id)
            if prev is None:
                per_feature[feature_id] = (activation, 1)
            else:
                per_feature[feature_id] = (max(prev[0], activation), prev[1] + 1)
        top = sorted(per_feature.items(), key=lambda kv: kv[1][0], reverse=True)
        summary[str(layer)] = [
            [feature_id, round(max_act, 4), count]
            for feature_id, (max_act, count) in top[:_TOP_FEATURES_PER_LAYER]
        ]
    return summary or None


def _summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compact one sidecar record for the correlated output line.

    Keeps identity/shape scalars, summarizes ``feats_topk`` into top
    features when present, else keeps the ``npz_path`` pointer to the raw
    activations. Never includes bulky payloads (``allf``, full ``gen_text``).
    """
    summary: Dict[str, Any] = {}
    for key in (
        "request_id",
        "session_id",
        "timestamp",
        "ts",
        "model",
        "layer",
        "layers",
        "d_model",
        "n_records",
        "n_gen_tokens",
        "prompt_len",
        "gen_len",
        "npz_path",
        "same_inference",
    ):
        if key in record:
            summary[key] = record[key]
    top_features = _summarize_top_features(record.get("feats_topk"))
    if top_features is not None:
        summary["top_features"] = top_features
    if "allf" in record:
        summary["has_all_features"] = True
    preview = record.get("gen_text_preview")
    if not isinstance(preview, str):
        gen_text = record.get("gen_text")
        preview = gen_text if isinstance(gen_text, str) else None
    if preview:
        summary["gen_text_preview"] = preview[:_TEXT_PREVIEW_CHARS]
    return summary


def _safe_session_filename(session_id: str) -> str:
    """Collapse a session ID to a single traversal-free path segment."""
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")
    return sanitized[:96] or "sessionless"


# ---------------------------------------------------------------------------
# In-process stats for the /sae slash command
# ---------------------------------------------------------------------------

@dataclass
class _Stats:
    turns_matched: int = 0
    turns_unmatched: int = 0
    records_matched: int = 0
    last_output: Optional[Dict[str, Any]] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATS = _Stats()
_WRITE_LOCK = threading.Lock()


def _write_output_line(out_dir: Path, session_id: str, payload: Dict[str, Any]) -> None:
    out_path = out_dir / f"{_safe_session_filename(session_id)}.jsonl"
    with _WRITE_LOCK:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Observer hooks (hermes.observer.v1 — read-only, kwargs-based, fail-open)
# ---------------------------------------------------------------------------

def on_pre_api_request(**_: Any) -> None:
    """Prime the sidecar offset so history is never mis-correlated.

    First call records the sidecar's current EOF; records appended before
    the agent's first request in this process are never candidates. After
    that this hook is a boolean no-op, preserving the cheap-by-default
    property of the uninstrumented path.
    """
    try:
        tailer = _get_tailer()
        if tailer is not None:
            tailer.prime()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"pre_api_request failed: {exc}")


def on_post_api_request(
    *,
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    session_id: str = "",
    model: str = "",
    response_model: Any = None,
    api_duration: float = 0.0,
    started_at: float = 0.0,
    ended_at: float = 0.0,
    **_: Any,
) -> None:
    """Correlate this provider request with new sidecar records."""
    try:
        config = _get_config()
        tailer = _get_tailer()
        if config is None or tailer is None:
            return
        tailer.poll()
        now = time.time()
        if not ended_at:
            ended_at = now
        if not started_at:
            started_at = ended_at - (api_duration or 0.0)
        models = tuple(
            m for m in (model, response_model if isinstance(response_model, str) else "")
            if m
        )
        records, confidence = tailer.claim_matches(
            api_request_id=api_request_id,
            session_id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            skew=config.skew,
            models=models,
        )
        if not records:
            with _STATS.lock:
                _STATS.turns_unmatched += 1
            _debug(
                f"no sidecar match for api_request (session={session_id!r}, "
                f"model={model!r})"
            )
            return
        payload = {
            "ts": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
            "session_id": session_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "api_request_id": api_request_id,
            "model": model,
            "api_duration": round(api_duration, 3) if api_duration else None,
            "match_confidence": confidence,
            "records": [_summarize_record(r) for r in records],
        }
        _write_output_line(config.out_dir, session_id, payload)
        with _STATS.lock:
            _STATS.turns_matched += 1
            _STATS.records_matched += len(records)
            _STATS.last_output = payload
        _debug(
            f"matched {len(records)} sidecar record(s) "
            f"(confidence={confidence}, session={session_id!r})"
        )
    except Exception as exc:  # pragma: no cover - fail-open
        logger.debug("SAE trace post_api_request failed: %s", exc)


# ---------------------------------------------------------------------------
# /sae slash command
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
/sae — SAE feature-trace status

Subcommands:
  status     Sidecar path, records seen/matched this session, last match
  last       Most recent turn's correlated feature summary
  dashboard  Where to find the zero-install session dashboard (dashboard.html)

Configure via HERMES_SAE_TRACE_FILE (sidecar JSONL) and
HERMES_SAE_TRACE_OUT_DIR (default: $HERMES_HOME/sae_trace).
"""


def _format_last_features(payload: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for record in payload.get("records", []):
        top_features = record.get("top_features")
        if top_features:
            for layer, features in sorted(top_features.items()):
                feats = ", ".join(
                    f"#{fid} (max {act}, {n} tok)" for fid, act, n in features[:5]
                )
                lines.append(f"  layer {layer}: {feats}")
        elif record.get("npz_path"):
            lines.append(f"  raw activations: {record['npz_path']}")
        preview = record.get("gen_text_preview")
        if preview:
            lines.append(f"  text: {preview[:80]!r}")
    return lines


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = (raw_args or "").strip().split()
    sub = argv[0] if argv else "status"
    if sub in {"help", "-h", "--help"}:
        return _HELP_TEXT

    config = _get_config()
    if config is None:
        return (
            "[sae] Not configured. Set HERMES_SAE_TRACE_FILE to your SAE "
            "server's sidecar JSONL (then restart Hermes) to enable "
            "correlation.\n\n" + _HELP_TEXT
        )

    if sub == "status":
        tailer = _get_tailer()
        with _STATS.lock:
            matched = _STATS.turns_matched
            unmatched = _STATS.turns_unmatched
            records = _STATS.records_matched
            last = _STATS.last_output
        lines = [
            "[sae] SAE feature-trace status",
            f"  sidecar : {config.trace_file}"
            + ("" if config.trace_file.exists() else "  (missing)"),
            f"  out dir : {config.out_dir}",
            f"  records seen: {tailer.records_seen if tailer else 0} "
            f"(pending: {tailer.pending_count() if tailer else 0}, "
            f"malformed: {tailer.malformed_lines if tailer else 0})",
            f"  turns matched: {matched} ({records} record(s)); "
            f"unmatched: {unmatched}",
        ]
        if last:
            lines.append(
                f"  last match: {last.get('ts')} "
                f"confidence={last.get('match_confidence')}"
            )
            lines.extend(_format_last_features(last))
        return "\n".join(lines)

    if sub == "dashboard":
        dash = Path(__file__).resolve().parent / "dashboard.html"
        lines = [
            "[sae] Zero-install session dashboard",
            f"  page   : file://{dash}",
            f"  traces : {config.out_dir}",
        ]
        try:
            newest = max(
                config.out_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime
            )
            lines.append(f"  latest : {newest}")
        except (ValueError, OSError):
            lines.append("  latest : (no session traces written yet)")
        lines.append(
            "  Open the page in any browser and load the trace (file picker or"
            " drag & drop). 'Follow live' re-reads it every 2s on"
            " Chromium-based browsers. No server, no install, no network."
        )
        return "\n".join(lines)

    if sub == "last":
        with _STATS.lock:
            last = _STATS.last_output
        if not last:
            return "[sae] No correlated turn yet in this session."
        lines = [
            f"[sae] Last correlated turn ({last.get('ts')}, "
            f"confidence={last.get('match_confidence')}, "
            f"model={last.get('model')})",
        ]
        lines.extend(_format_last_features(last) or ["  (no feature summary)"])
        return "\n".join(lines)

    return f"Unknown subcommand: {sub}\n\n{_HELP_TEXT}"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_command(
        "sae",
        handler=_handle_slash,
        description="Show SAE feature-trace correlation status for this session.",
        args_hint="status|last",
    )
