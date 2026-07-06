"""Plugin fire ledger — local-only AGGREGATED telemetry for the customization audit.

Records how often each plugin hook / middleware *fires* through the two core
dispatch loops, so a later audit can answer "is this guard helping, dead, or
net-negative?".

v2 (aggregate-at-write): the v1 event-log wrote one JSONL row per fire, which at
observed volume was ~75k rows / ~50MB per day (99.7% of them `noop` observer
passes) — the 30-day audit window could not fit in any sane disk budget. v2 keeps
an IN-MEMORY counter per process and flushes a compact per-(day,plugin,channel,
decision) COUNT to disk. 75k daily rows collapse to a few dozen counter records,
so a 30-day window fits in well under a megabyte.

Design constraints (from the reviewed plan + the volume rework, SPEC AC-1/AC-8):

- **Local-only, no outbound.** Imports nothing that touches the network; a
  socket-deny test proves flush still succeeds.
- **Race-free by construction, no rotate/append race (Blocker-1 still honored).**
  Each *process* owns exactly one file per day: `<YYYY-MM-DD>.<pid>.json`. Only
  that pid ever writes that file, via a temp-file + atomic `os.replace`. No other
  actor rewrites it, so there is no cross-process race — unlike a shared active
  file that a separate pruner rewrites. Pruning deletes whole CLOSED day-files
  (an earlier day, or a pid no longer alive); it never touches a live file.
- **Safe metadata only.** No raw command text, args, tool results, prompt text,
  or provider payloads. Counters + coarse tool-class SET + decision only.
- **Fail-open everywhere.** Any counter/flush error is swallowed so the ledger
  can never break plugin dispatch.
- **Cheap hot path.** A fire is an in-memory dict bump; disk I/O is a throttled
  flush (time- or count-triggered) + atexit — not one write per fire.
- **Config-gated in config.yaml** (`plugins.fire_ledger.*`), not an env var.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # never let an import failure break dispatch
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - defensive
    def get_hermes_home() -> Path:  # type: ignore
        return Path(os.path.expanduser("~/.hermes"))

__all__ = [
    "FIRE_LEDGER_SCHEMA_VERSION",
    "FireLedgerConfig",
    "load_fire_ledger_config",
    "record_hook_fire",
    "record_middleware_fire",
    "classify_hook_decision",
    "classify_middleware_decision",
    "read_counters",
    "summarize_window",
    "prune_closed_files",
    "flush",
    "FireLedgerReadResult",
]

FIRE_LEDGER_SCHEMA_VERSION = "hermes.plugin_fire.v2"

DEFAULT_RELATIVE_DIR = "metrics/plugin-fire-ledger"
DEFAULT_RETENTION_DAYS = 30
DEFAULT_MAX_MB = 25
DEFAULT_MIN_OBSERVATION_DAYS = 30

_MAX_TOOL_CLASSES = 32  # bound the per-counter tool-class set
_MAX_STR_FIELD = 120

# Flush throttle: flush after this many increments OR this many seconds,
# whichever comes first. Also flushed at process exit via atexit.
_FLUSH_EVERY_N = 500
_FLUSH_EVERY_S = 30.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireLedgerConfig:
    enabled: bool
    dir: Path
    retention_days: int
    max_mb: int
    min_observation_days: int


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def load_fire_ledger_config(cfg: Mapping[str, Any] | None = None) -> FireLedgerConfig:
    """Load ledger config from ``plugins.fire_ledger.*``. Fail-open to disabled."""
    disabled = FireLedgerConfig(
        enabled=False,
        dir=get_hermes_home() / DEFAULT_RELATIVE_DIR,
        retention_days=DEFAULT_RETENTION_DAYS,
        max_mb=DEFAULT_MAX_MB,
        min_observation_days=DEFAULT_MIN_OBSERVATION_DAYS,
    )
    try:
        if cfg is None:
            from hermes_cli.config import load_config  # lazy; avoid import cycles
            cfg = load_config()
        from hermes_cli.config import cfg_get

        enabled = bool(cfg_get(cfg, "plugins", "fire_ledger", "enabled", default=False))
        raw_path = cfg_get(cfg, "plugins", "fire_ledger", "path", default=None)
        if raw_path:
            ledger_dir = Path(os.path.expanduser(str(raw_path)))
            if ledger_dir.suffix in (".jsonl", ".json"):
                ledger_dir = ledger_dir.parent
        else:
            ledger_dir = get_hermes_home() / DEFAULT_RELATIVE_DIR

        return FireLedgerConfig(
            enabled=enabled,
            dir=ledger_dir,
            retention_days=_clamp_int(
                cfg_get(cfg, "plugins", "fire_ledger", "retention_days", default=DEFAULT_RETENTION_DAYS),
                DEFAULT_RETENTION_DAYS, 1, 3650,
            ),
            max_mb=_clamp_int(
                cfg_get(cfg, "plugins", "fire_ledger", "max_mb", default=DEFAULT_MAX_MB),
                DEFAULT_MAX_MB, 1, 100000,
            ),
            min_observation_days=_clamp_int(
                cfg_get(cfg, "plugins", "fire_ledger", "min_observation_days", default=DEFAULT_MIN_OBSERVATION_DAYS),
                DEFAULT_MIN_OBSERVATION_DAYS, 1, 3650,
            ),
        )
    except Exception:
        return disabled


# ---------------------------------------------------------------------------
# Decision classifiers (unchanged from v1)
# ---------------------------------------------------------------------------

_TRANSFORM_HOOKS = {
    "transform_tool_result",
    "transform_llm_output",
    "transform_terminal_output",
}


def _result_is_block(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    action = result.get("action")
    decision = result.get("decision")
    return action == "block" or decision == "block" or action == "skip"


def classify_hook_decision(hook_name: str, result: Any, error: BaseException | None) -> str:
    if error is not None:
        return "error"
    if _result_is_block(result):
        return "block"
    if hook_name in _TRANSFORM_HOOKS and isinstance(result, str) and result:
        return "transform"
    if hook_name == "pre_llm_call":
        if isinstance(result, str) and result:
            return "context"
        if isinstance(result, Mapping) and result.get("context"):
            return "context"
    if hook_name == "pre_verify" and isinstance(result, Mapping):
        if result.get("action") == "continue" or result.get("decision") == "block":
            return "continue"
    if isinstance(result, Mapping):
        action = result.get("action")
        if action == "rewrite":
            return "transform"
        if action == "allow":
            return "allow"
    return "noop"


def classify_middleware_decision(kind: str, kwargs: Mapping[str, Any], result: Any, error: BaseException | None) -> str:
    if error is not None:
        return "error"
    if _result_is_block(result):
        return "block"
    if isinstance(result, Mapping):
        if any(k in result for k in ("args", "request", "command", "text")):
            return "transform"
        if result.get("action") in ("rewrite", "transform"):
            return "transform"
        if result.get("action") == "allow":
            return "allow"
    return "noop"


# ---------------------------------------------------------------------------
# Safe field extraction
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value)[:_MAX_STR_FIELD]
    except Exception:
        return None


def _tool_class(kwargs: Mapping[str, Any]) -> str | None:
    tool_name = kwargs.get("tool_name") or kwargs.get("tool")
    if isinstance(tool_name, str) and tool_name:
        return tool_name[:_MAX_STR_FIELD]
    return None


# ---------------------------------------------------------------------------
# In-memory aggregator (per process)
# ---------------------------------------------------------------------------


@dataclass
class _Agg:
    count: int = 0
    first_ts: str = ""
    last_ts: str = ""
    sum_duration_ms: int = 0
    tool_classes: set = field(default_factory=set)
    error_types: set = field(default_factory=set)


class _Aggregator:
    """Thread-safe per-process fire counter with throttled disk flush."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # day -> key(tuple) -> _Agg
        self._buckets: dict[str, dict[tuple, _Agg]] = {}
        self._dirty_days: set[str] = set()
        self._since_flush = 0
        self._last_flush_monotonic = time.monotonic()
        self._atexit_registered = False

    def record(
        self,
        *,
        dispatch_kind: str,
        plugin_name: str,
        channel: str | None,
        decision: str,
        tool_class: str | None,
        duration_ms: int,
        error_type: str | None,
        cfg: FireLedgerConfig,
    ) -> None:
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        day = now.strftime("%Y-%m-%d")
        key = (plugin_name, dispatch_kind, channel or "?", decision)
        with self._lock:
            bucket = self._buckets.setdefault(day, {})
            agg = bucket.get(key)
            if agg is None:
                agg = _Agg(first_ts=ts)
                bucket[key] = agg
            agg.count += 1
            agg.last_ts = ts
            agg.sum_duration_ms += max(0, int(duration_ms or 0))
            if tool_class and len(agg.tool_classes) < _MAX_TOOL_CLASSES:
                agg.tool_classes.add(tool_class)
            if error_type and len(agg.error_types) < _MAX_TOOL_CLASSES:
                agg.error_types.add(error_type)
            self._dirty_days.add(day)
            self._since_flush += 1
            if not self._atexit_registered:
                # register once; flush our counters on clean process exit
                try:
                    atexit.register(self._atexit_flush)
                    self._atexit_registered = True
                except Exception:
                    pass
            due = (
                self._since_flush >= _FLUSH_EVERY_N
                or (time.monotonic() - self._last_flush_monotonic) >= _FLUSH_EVERY_S
            )
        if due:
            self.flush(cfg)

    def _atexit_flush(self) -> None:
        try:
            self.flush(load_fire_ledger_config())
        except Exception:
            pass

    def flush(self, cfg: FireLedgerConfig) -> bool:
        """Atomically write each dirty day's counters to <day>.<pid>.json.

        Race-free: only THIS pid ever writes THIS file; temp + os.replace makes
        each file complete-or-old for readers, never partial.
        """
        try:
            if not cfg.enabled:
                return False
            with self._lock:
                dirty = list(self._dirty_days)
                snapshot: dict[str, list[dict[str, Any]]] = {}
                for day in dirty:
                    rows = []
                    for (plugin, kind, channel, decision), agg in self._buckets.get(day, {}).items():
                        rows.append({
                            "plugin_name": plugin,
                            "dispatch_kind": kind,
                            "channel": channel,
                            "decision": decision,
                            "count": agg.count,
                            "first_ts": agg.first_ts,
                            "last_ts": agg.last_ts,
                            "sum_duration_ms": agg.sum_duration_ms,
                            "tool_classes": sorted(agg.tool_classes),
                            "error_types": sorted(agg.error_types),
                        })
                    snapshot[day] = rows
                self._dirty_days.clear()
                self._since_flush = 0
                self._last_flush_monotonic = time.monotonic()
            cfg.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            pid = os.getpid()
            for day, rows in snapshot.items():
                if not rows:
                    continue
                payload = {
                    "schema": FIRE_LEDGER_SCHEMA_VERSION,
                    "day": day,
                    "pid": pid,
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "rows": rows,
                }
                final = cfg.dir / f"{day}.{pid}.json"
                tmp = cfg.dir / f".{day}.{pid}.{os.getpid()}.tmp"
                data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                os.replace(str(tmp), str(final))  # atomic; only this pid writes this file
            _maybe_prune(cfg)
            return True
        except Exception:
            return False


_AGG = _Aggregator()


def flush(cfg: FireLedgerConfig | None = None) -> bool:
    """Force a flush of the in-memory counters (for tests / shutdown)."""
    if cfg is None:
        cfg = load_fire_ledger_config()
    return _AGG.flush(cfg)


# ---------------------------------------------------------------------------
# Public record API — signatures unchanged from v1 (plugins.py calls these)
# ---------------------------------------------------------------------------


def record_hook_fire(
    *,
    plugin_name: str,
    hook_name: str,
    kwargs: Mapping[str, Any],
    result: Any = None,
    error: BaseException | None = None,
    duration_ms: int = 0,
    cfg: FireLedgerConfig | None = None,
) -> bool:
    try:
        if cfg is None:
            cfg = load_fire_ledger_config()
        if not cfg.enabled:
            return False
        _AGG.record(
            dispatch_kind="hook",
            plugin_name=_safe_str(plugin_name) or "unknown",
            channel=_safe_str(hook_name),
            decision=classify_hook_decision(hook_name, result, error),
            tool_class=_tool_class(kwargs),
            duration_ms=duration_ms,
            error_type=type(error).__name__ if error is not None else None,
            cfg=cfg,
        )
        return True
    except Exception:
        return False


def record_middleware_fire(
    *,
    plugin_name: str,
    kind: str,
    kwargs: Mapping[str, Any],
    result: Any = None,
    error: BaseException | None = None,
    duration_ms: int = 0,
    cfg: FireLedgerConfig | None = None,
) -> bool:
    try:
        if cfg is None:
            cfg = load_fire_ledger_config()
        if not cfg.enabled:
            return False
        _AGG.record(
            dispatch_kind="middleware",
            plugin_name=_safe_str(plugin_name) or "unknown",
            channel=_safe_str(kind),
            decision=classify_middleware_decision(kind, kwargs, result, error),
            tool_class=_tool_class(kwargs),
            duration_ms=duration_ms,
            error_type=type(error).__name__ if error is not None else None,
            cfg=cfg,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Maintenance — prune CLOSED day-files only
# ---------------------------------------------------------------------------

_last_maint_monotonic = 0.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _parse_file_name(path: Path) -> tuple[str, int] | None:
    name = path.name
    if not name.endswith(".json") or name.startswith("."):
        return None
    stem = name[: -len(".json")]
    parts = stem.rsplit(".", 1)
    if len(parts) != 2:
        return None
    day, pid_s = parts
    try:
        return day, int(pid_s)
    except ValueError:
        return None


def _is_closed_file(path: Path, today: str) -> bool:
    parsed = _parse_file_name(path)
    if parsed is None:
        return False
    day, pid = parsed
    if day < today:
        return True
    if day == today and pid == os.getpid():
        return False  # our own live file
    return not _pid_alive(pid)


def prune_closed_files(cfg: FireLedgerConfig, *, now: datetime | None = None) -> dict[str, Any]:
    result = {"deleted": 0, "kept": 0, "bytes_after": 0}
    try:
        now = now or datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        cutoff_day = _day_n_ago(now, cfg.retention_days)
        closed: list[tuple[float, Path, int]] = []
        for p in sorted(cfg.dir.glob("*.json")):
            parsed = _parse_file_name(p)
            if parsed is None:
                continue
            day, _pid = parsed
            try:
                st = p.stat()
            except OSError:
                continue
            if _is_closed_file(p, today):
                if day < cutoff_day:  # age by DAY label, not mtime
                    try:
                        p.unlink()
                        result["deleted"] += 1
                        continue
                    except OSError:
                        pass
                closed.append((st.st_mtime, p, st.st_size))
            else:
                result["kept"] += 1
        # size-based prune: oldest closed files first until under max_mb
        max_bytes = cfg.max_mb * 1024 * 1024
        total = sum(s for _, _, s in closed)
        closed.sort()
        idx = 0
        while total > max_bytes and idx < len(closed):
            _, p, size = closed[idx]
            try:
                p.unlink()
                result["deleted"] += 1
                total -= size
            except OSError:
                pass
            idx += 1
        result["bytes_after"] = max(0, total)
        return result
    except Exception:
        return result


def _day_n_ago(now: datetime, n: int) -> str:
    return datetime.fromtimestamp(now.timestamp() - n * 86400, tz=timezone.utc).strftime("%Y-%m-%d")


def _maybe_prune(cfg: FireLedgerConfig) -> None:
    global _last_maint_monotonic
    try:
        now = time.monotonic()
        if now - _last_maint_monotonic < 300.0:
            return
        _last_maint_monotonic = now
        prune_closed_files(cfg)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reader — glob day-files, SUM counters across pids/days
# ---------------------------------------------------------------------------


@dataclass
class FireLedgerReadResult:
    counters: list[dict[str, Any]]  # merged, one per (plugin,kind,channel,decision)
    skipped_corrupt: int
    files_read: int


def read_counters(*, cfg: FireLedgerConfig | None = None, since_day: str | None = None) -> FireLedgerReadResult:
    """Read all day-files, merge counters across pids and days by summing.

    Also merges THIS process's un-flushed in-memory counters so a reader in the
    same process sees live counts without forcing a disk flush.
    """
    if cfg is None:
        cfg = load_fire_ledger_config()
    merged: dict[tuple, dict[str, Any]] = {}
    skipped = 0
    files_read = 0

    def _merge(row: dict[str, Any]) -> None:
        key = (row.get("plugin_name"), row.get("dispatch_kind"), row.get("channel"), row.get("decision"))
        m = merged.get(key)
        if m is None:
            m = {
                "plugin_name": row.get("plugin_name"),
                "dispatch_kind": row.get("dispatch_kind"),
                "channel": row.get("channel"),
                "decision": row.get("decision"),
                "count": 0,
                "first_ts": row.get("first_ts") or "",
                "last_ts": row.get("last_ts") or "",
                "sum_duration_ms": 0,
                "tool_classes": set(),
                "error_types": set(),
            }
            merged[key] = m
        m["count"] += int(row.get("count", 0))
        m["sum_duration_ms"] += int(row.get("sum_duration_ms", 0))
        ft, lt = row.get("first_ts") or "", row.get("last_ts") or ""
        if ft and (not m["first_ts"] or ft < m["first_ts"]):
            m["first_ts"] = ft
        if lt and lt > m["last_ts"]:
            m["last_ts"] = lt
        for tc in row.get("tool_classes", []) or []:
            m["tool_classes"].add(tc)
        for et in row.get("error_types", []) or []:
            m["error_types"].add(et)

    try:
        files = sorted(cfg.dir.glob("*.json"))
    except Exception:
        files = []
    my_pid = os.getpid()
    with _AGG._lock:
        have_memory = bool(_AGG._buckets)
    for path in files:
        parsed = _parse_file_name(path)
        if parsed is None:
            continue
        day, pid = parsed
        if since_day and day < since_day:
            continue
        # Our own flushed file is a cumulative snapshot of the SAME counts we
        # still hold in _AGG. Skip it ONLY when memory is non-empty (memory is
        # fresher-or-equal). If memory is empty — a freshly started/restarted
        # process — the disk file is authoritative, so read it.
        if pid == my_pid and have_memory:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            files_read += 1
        except (OSError, ValueError):
            skipped += 1
            continue
        if not isinstance(obj, dict) or obj.get("schema") != FIRE_LEDGER_SCHEMA_VERSION:
            skipped += 1
            continue
        for row in obj.get("rows", []):
            if isinstance(row, dict):
                _merge(row)

    # fold this process's un-flushed in-memory counters
    try:
        with _AGG._lock:
            for day, bucket in _AGG._buckets.items():
                if since_day and day < since_day:
                    continue
                for (plugin, kind, channel, decision), agg in bucket.items():
                    _merge({
                        "plugin_name": plugin, "dispatch_kind": kind, "channel": channel,
                        "decision": decision, "count": agg.count, "first_ts": agg.first_ts,
                        "last_ts": agg.last_ts, "sum_duration_ms": agg.sum_duration_ms,
                        "tool_classes": list(agg.tool_classes), "error_types": list(agg.error_types),
                    })
    except Exception:
        pass

    out = []
    for m in merged.values():
        m["tool_classes"] = sorted(m["tool_classes"])
        m["error_types"] = sorted(m["error_types"])
        out.append(m)
    out.sort(key=lambda r: (-r["count"], str(r["plugin_name"])))
    return FireLedgerReadResult(counters=out, skipped_corrupt=skipped, files_read=files_read)


def summarize_window(days: int = 30, *, cfg: FireLedgerConfig | None = None) -> dict[str, Any]:
    """Per-plugin summary for the audit collector, from aggregated counters."""
    if cfg is None:
        cfg = load_fire_ledger_config()
    since_day = _day_n_ago(datetime.now(timezone.utc), days)
    res = read_counters(cfg=cfg, since_day=since_day)
    plugins: dict[str, dict[str, Any]] = {}
    for row in res.counters:
        name = row["plugin_name"]
        p = plugins.setdefault(name, {
            "fires": 0, "decisions": {}, "channels": {},
            "first_seen": None, "last_seen": None,
        })
        p["fires"] += row["count"]
        d = row["decision"]
        p["decisions"][d] = p["decisions"].get(d, 0) + row["count"]
        ch = row["channel"] or "?"
        p["channels"][ch] = p["channels"].get(ch, 0) + row["count"]
        ft, lt = row.get("first_ts"), row.get("last_ts")
        if ft and (p["first_seen"] is None or ft < p["first_seen"]):
            p["first_seen"] = ft
        if lt and (p["last_seen"] is None or lt > p["last_seen"]):
            p["last_seen"] = lt
    return {
        "schema": FIRE_LEDGER_SCHEMA_VERSION,
        "window_days": days,
        "total_fires": sum(p["fires"] for p in plugins.values()),
        "distinct_counters": len(res.counters),
        "skipped_corrupt": res.skipped_corrupt,
        "files_read": res.files_read,
        "plugins": plugins,
    }
