"""Local, content-free provider observations; not a provider billing ledger."""
from dataclasses import astuple, dataclass
import logging
import time
import threading

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageEvent:
    attempt_id: str
    provider: str
    model: str
    profile: str
    completed_at_us: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    retry_count: int | None = None
    status: str | None = None
    usage_state: str = "reported"


TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
BIN_US = 15 * 60 * 1_000_000
WINDOW_US = 24 * BIN_US
RETENTION_US = 7 * 24 * 3600 * 1_000_000
_RETENTION_BATCH = 256
_FAILURE_KEY = "usage_events_recording_incomplete"
_failure_lock = threading.Lock()
_failures: dict[str, int] = {}


def note_recording_failure(db) -> None:
    key = hermes_home_key(db.db_path.parent)
    with _failure_lock:
        _failures[key] = _failures.get(key, 0) + 1
    # Deliberately no exception text, model, ids, paths or response payload.
    logger.warning("usage_event_recording_failed; local usage coverage is incomplete")


def _failure_count(db) -> int:
    with _failure_lock:
        return _failures.get(hermes_home_key(db.db_path.parent), 0)


def _prune(conn, now_us: int) -> int:
    return conn.execute(
        "DELETE FROM usage_events WHERE attempt_id IN "
        "(SELECT attempt_id FROM usage_events WHERE completed_at_us < ? "
        "ORDER BY completed_at_us LIMIT ?)", (now_us - RETENTION_US, _RETENTION_BATCH),
    ).rowcount


def _empty_usage():
    return dict.fromkeys((*TOKEN_FIELDS, *("unknown_" + f for f in TOKEN_FIELDS),
                          "processed_tokens", "events", "missing_usage_events", "invalid_usage_events"), 0)


class SessionUsageEventsMixin:
    def codex_usage_timeline(self, *, profile: str | None = None) -> dict:
        """Backend-selected UTC, exact [as_of-6h, as_of); no maintenance or model calls.

        Numeric sums are known-token subtotals. Paired unknown counters MUST travel
        with them: a subtotal of zero is not evidence of zero provider consumption.
        """
        end = time.time_ns() // 1000
        start = end - WINDOW_US
        bins = [{"start_us": start + i * BIN_US, "end_us": start + (i + 1) * BIN_US,
                 "usage": _empty_usage()} for i in range(24)]
        total = _empty_usage()
        coverage = {"status": "partial", "scope": "local_profile_recorded_responses",
                    "reason": "post_api_request_only_no_crash_recovery"}
        coverage["recording_failures_in_process"] = _failure_count(self)
        try:
            coverage["persisted_recording_failure"] = self._read_one(
                "SELECT 1 FROM state_meta WHERE key=?", (_FAILURE_KEY,),
            ) is not None
            # Identifiers come only from the constant allowlist above, never the caller.
            sums = ", ".join(f"COALESCE(SUM({f}), 0) AS {f}, "
                             f"SUM({f} IS NULL) AS unknown_{f}" for f in TOKEN_FIELDS)
            rows = self._read_all(
                "SELECT (completed_at_us - ?) / ? AS bin, " + sums + ", "
                "COUNT(*) AS events, "
                "SUM(usage_state='missing') AS missing_usage_events, "
                "SUM(usage_state='invalid') AS invalid_usage_events "
                "FROM usage_events WHERE provider=? AND completed_at_us>=? AND completed_at_us<? "
                + ("AND profile=? " if profile is not None else "")
                + "GROUP BY bin", (start, BIN_US, "openai-codex", start, end)
                + ((profile,) if profile is not None else ()),
            )
            for row in rows:
                usage = bins[row["bin"]]["usage"]
                for field in usage:
                    if field != "processed_tokens":
                        usage[field] = row[field]
                usage["processed_tokens"] = usage["input_tokens"] + usage["output_tokens"]
                for field in total:
                    total[field] += usage[field]
        except Exception:
            coverage.update(status="unavailable", reason="usage_event_query_failed")
            for b in bins:
                b["usage"] = None
        return {"as_of_us": end, "start_us": start, "provider": "openai-codex",
                "bins": bins, "total": total if coverage["status"] != "unavailable" else None,
                "coverage": coverage}

    def record_usage_event(self, event: UsageEvent, *, now_us: int | None = None) -> bool:
        """True includes duplicate observation. Failures never escape into the turn."""
        try:
            now = time.time_ns() // 1000 if now_us is None else now_us
            def write(conn):
                if event.completed_at_us >= now - RETENTION_US:
                    conn.execute(
                        "INSERT INTO usage_events (attempt_id, provider, model, profile, completed_at_us, "
                        "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, "
                        "retry_count, status, usage_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(attempt_id) DO NOTHING", astuple(event),
                    )
                if _failure_count(self):
                    conn.execute("INSERT OR IGNORE INTO state_meta(key, value) VALUES (?, ?)",
                                 (_FAILURE_KEY, "1"))
                _prune(conn, now)
            self._execute_write(write, patience_s=0.1)
            return True
        except Exception:
            note_recording_failure(self)
            return False

    def prune_usage_events(self, *, now_us: int | None = None) -> int | None:
        """One bounded maintenance batch; also run on ingestion, NEVER on timeline reads.

        Idle stores retain expired rows until ingestion or explicit maintenance resumes.
        None signals a failed batch; callers must not spin on contention.
        """
        now = time.time_ns() // 1000 if now_us is None else now_us
        try:
            return self._execute_write(lambda conn: _prune(conn, now), patience_s=0.1)
        except Exception:
            note_recording_failure(self)
            return None
