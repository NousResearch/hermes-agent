"""Offline support for Himalaya CLI v2.1.0 Gmail scans; no mailbox I/O.

Import collect_pages with an explicit fetch/stop adapter, or run `windows`.
Result `exhausted` describes pagination, not body reading or classification.
"""
import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ScanScope:
    account: str
    labels: tuple
    query: str = ""
    config: Optional[str] = None
    include_spam_trash: bool = False
    page_size: int = 50

    def __post_init__(self):
        if not isinstance(self.account, str) or not self.account.strip():
            raise ValueError("An explicit account is required")
        if not isinstance(self.labels, tuple) or any(
            not isinstance(label, str) or not label for label in self.labels
        ):
            raise ValueError("labels must be a tuple of nonempty label IDs")
        if type(self.page_size) is not int or not 1 <= self.page_size <= 500:
            raise ValueError("page_size must be between 1 and 500")
        if not isinstance(self.query, str):
            raise ValueError("query must be a string")


def list_argv(scope, token=None, executable="himalaya"):
    """Construct only native read-only listing; pass to subprocess with shell=False."""
    if not isinstance(executable, str) or not executable or '\x00' in executable:
        raise ValueError("Executable must be a nonempty path or name")
    argv = [executable]
    if scope.config is not None:
        argv += ["--config", scope.config]
    argv += ["--account", scope.account, "--json", "gmail", "messages", "list"]
    for label in scope.labels:
        argv += ["--label", label]
    if scope.query:
        argv += ["--query", scope.query]
    argv += ["--max-results", str(scope.page_size)]
    if scope.include_spam_trash:
        argv.append("--include-spam-trash")
    if token is not None:
        if not isinstance(token, str) or not token:
            raise ValueError("A continuation token must be a nonempty string")
        argv += ["--page-token", token]
    return argv


def parse_page(payload):
    """Validate the complete page before accepting any IDs. Optional fields ignored."""
    if not isinstance(payload, dict) or not isinstance(payload.get("ids"), list):
        raise ValueError("Expected Himalaya native listing object with an ids array")
    if any(key in payload for key in ("nextPageToken", "next_page_token", "messages")):
        raise ValueError("Provider-shaped JSON does not match the pinned CLI schema")
    ids = []
    for row in payload["ids"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each ids row needs a nonempty string id")
        ids.append(row["id"])
    token = payload.get("next_page")
    if token is not None and (not isinstance(token, str) or not token):
        raise ValueError("next_page must be absent, null, or a nonempty string")
    return ids, token


@dataclass
class ScanResult:
    scope: ScanScope
    ids: list = field(default_factory=list)
    pages: int = 0
    exhausted: bool = False
    reason: str = "not_started"
    error: Optional[str] = None
    next_page: Optional[str] = None
    # If a result limit ends mid-page, these IDs must not be lost on continuation.
    pending_ids: list = field(default_factory=list)


def collect_pages(scope: ScanScope, fetch_page: Callable, should_stop: Callable,
                  max_pages=100, limit=None):
    """Fetch callback(scope, token) returns parsed JSON or raises on command failure.

    Use a finite timeout in that adapter. No automatic retries or resume occurs.
    should_stop must read the real task cancellation state. This function cannot
    interrupt a running callback; it records a returned page before stopping.
    """
    if type(max_pages) is not int or max_pages < 1:
        raise ValueError("max_pages must be a positive integer")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    result = ScanResult(scope)
    seen_ids, requested_tokens = set(), set()
    token = None
    for _ in range(max_pages):
        if should_stop():
            result.reason = "stopped"
            return result
        if token in requested_tokens:
            result.reason = "repeated_cursor"
            return result
        requested_tokens.add(token)
        try:
            ids, next_token = parse_page(fetch_page(scope, token))
        except Exception as exc:
            result.reason = "stopped" if should_stop() else "error"
            result.error = str(exc)  # Adapter should redact stderr before raising.
            result.next_page = token
            return result
        result.pages += 1
        fresh = []
        for message_id in ids:
            if message_id not in seen_ids:
                fresh.append(message_id)
                seen_ids.add(message_id)
        remaining = len(fresh) if limit is None else max(0, limit - len(result.ids))
        result.ids.extend(fresh[:remaining])
        result.pending_ids = fresh[remaining:]
        result.next_page = next_token
        result.exhausted = next_token is None
        if should_stop():
            result.reason = "stopped"
            return result
        if limit is not None and len(result.ids) >= limit:
            result.reason = "requested_limit"
            return result
        if result.exhausted:
            result.reason = "exhausted"
            return result
        if next_token in requested_tokens:
            result.reason = "repeated_cursor"
            return result
        token = next_token
    result.reason = "page_budget"
    return result


def month_windows(start: date, end: date):
    """Newest-first, contiguous half-open calendar windows clipped to [start,end)."""
    if start >= end:
        raise ValueError("start must be earlier than exclusive end")
    upper = end
    windows = []
    while upper > start:
        first = upper.replace(day=1)
        lower = (first - timedelta(days=1)).replace(day=1) if upper == first else first
        lower = max(start, lower)
        windows.append((lower, upper))
        upper = lower
    return windows


def epoch_bounds(start, end, zone_name):
    if start >= end:
        raise ValueError("start must be earlier than exclusive end")
    zone = timezone.utc if zone_name == "UTC" else ZoneInfo(zone_name)
    lower = int(datetime.combine(start, time.min, zone).timestamp())
    upper = int(datetime.combine(end, time.min, zone).timestamp())
    if lower >= upper:
        raise ValueError("Timezone conversion did not produce an increasing interval")
    return lower, upper


def in_internal_date_window(internal_date, lower_s, upper_s):
    """Apply exact half-open bounds to Gmail metadata internal-date milliseconds."""
    if lower_s >= upper_s:
        raise ValueError("Invalid epoch interval")
    if isinstance(internal_date, bool) or not isinstance(internal_date, (str, int)):
        raise ValueError("internal-date must be integer milliseconds")
    return lower_s * 1000 <= int(internal_date) < upper_s * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    win = sub.add_parser("windows", help="Generate calendar windows without mailbox access")
    win.add_argument("--start", type=date.fromisoformat, required=True)
    win.add_argument("--end", type=date.fromisoformat, required=True)
    win.add_argument("--timezone", required=True)
    args = parser.parse_args()
    try:
        rows = []
        for lower, upper in month_windows(args.start, args.end):
            low_s, high_s = epoch_bounds(lower, upper, args.timezone)
            rows.append({"start": lower.isoformat(), "end_exclusive": upper.isoformat(),
                         "timezone": args.timezone, "start_epoch": low_s, "end_epoch": high_s,
                         "gmail_candidate_query": f"after:{low_s - 1} before:{high_s}"})
    except (ValueError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
