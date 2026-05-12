#!/usr/bin/env python3
"""Diagnose gateway flush-without-inbound incidents.

Looks for the May-2026 failure shape: Telegram adapter flushes a text batch,
then no normal gateway inbound/session processing follows before a restart or
send failure. Outputs a human summary plus machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - script can run from partial checkouts
    def get_hermes_home() -> Path:  # type: ignore
        return Path.home() / ".hermes"


TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{1,6})?)")
FLUSH_RE = re.compile(r"Flushing text batch", re.I)
INBOUND_RE = re.compile(r"\binbound message\b", re.I)
STALE_RE = re.compile(r"stale[- ]code|code was updated|requesting gateway restart after stale", re.I)
SEND_FAILURE_RE = re.compile(r"failed to send|response send failed|send returned success=false|invalid .*target|invalid-chat-id", re.I)


@dataclass
class LogHit:
    line_no: int
    timestamp: Optional[str]
    line: str


@dataclass
class FlushGap:
    flush: LogHit
    matched_inbound: Optional[LogHit]
    stale_restarts: list[LogHit]
    send_failures: list[LogHit]


def _parse_timestamp(text: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.match(text)
    if not match:
        return None
    raw = match.group("ts").replace("T", " ").replace(",", ".")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _parse_time_expr(value: str, *, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    text = value.strip()
    if text.startswith("now-"):
        amount_unit = text[4:]
        match = re.fullmatch(r"(\d+)(s|sec|secs|m|min|mins|h|hr|hrs|d|day|days)", amount_unit)
        if not match:
            raise ValueError(f"unsupported relative time: {value!r}")
        amount = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("s"):
            delta = timedelta(seconds=amount)
        elif unit.startswith("m"):
            delta = timedelta(minutes=amount)
        elif unit.startswith("h"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return now - delta
    if text == "now":
        return now
    try:
        return datetime.fromisoformat(text.replace("T", " "))
    except ValueError as exc:
        raise ValueError(f"unsupported timestamp: {value!r}") from exc


def _line_matches_chat(line: str, chat_id: Optional[str]) -> bool:
    if not chat_id:
        return True
    return str(chat_id) in line


def _hit(line_no: int, line: str) -> LogHit:
    ts = _parse_timestamp(line)
    return LogHit(line_no=line_no, timestamp=ts.isoformat(sep=" ") if ts else None, line=line.rstrip("\n"))


def scan_gateway_log(
    log_path: Path,
    *,
    since: datetime,
    until: datetime,
    chat_id: Optional[str] = None,
    window_seconds: int = 20,
) -> dict:
    flushes: list[LogHit] = []
    inbounds: list[LogHit] = []
    stale_restarts: list[LogHit] = []
    send_failures: list[LogHit] = []

    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                ts = _parse_timestamp(line)
                if ts is not None and (ts < since or ts > until):
                    continue
                if chat_id and not _line_matches_chat(line, chat_id):
                    # Stale restart lines may not include the chat, but they
                    # are only useful when temporally near a matching flush.
                    if not STALE_RE.search(line) and not SEND_FAILURE_RE.search(line):
                        continue
                if FLUSH_RE.search(line) and _line_matches_chat(line, chat_id):
                    flushes.append(_hit(line_no, line))
                if INBOUND_RE.search(line) and _line_matches_chat(line, chat_id):
                    inbounds.append(_hit(line_no, line))
                if STALE_RE.search(line):
                    stale_restarts.append(_hit(line_no, line))
                if SEND_FAILURE_RE.search(line):
                    send_failures.append(_hit(line_no, line))

    gaps: list[FlushGap] = []
    for flush in flushes:
        flush_ts = _parse_timestamp(flush.line)
        matched_inbound: Optional[LogHit] = None
        near_stale: list[LogHit] = []
        near_send_failures: list[LogHit] = []
        for inbound in inbounds:
            inbound_ts = _parse_timestamp(inbound.line)
            if flush_ts and inbound_ts and timedelta(0) <= inbound_ts - flush_ts <= timedelta(seconds=window_seconds):
                matched_inbound = inbound
                break
        for candidate, target in ((stale_restarts, near_stale), (send_failures, near_send_failures)):
            for hit in candidate:
                hit_ts = _parse_timestamp(hit.line)
                if flush_ts and hit_ts and abs((hit_ts - flush_ts).total_seconds()) <= window_seconds:
                    target.append(hit)
        if matched_inbound is None:
            gaps.append(FlushGap(flush=flush, matched_inbound=None, stale_restarts=near_stale, send_failures=near_send_failures))

    return {
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "flushes": [asdict(h) for h in flushes],
        "inbounds": [asdict(h) for h in inbounds],
        "stale_restarts": [asdict(h) for h in stale_restarts],
        "send_failures": [asdict(h) for h in send_failures],
        "flush_without_inbound": [
            {
                "flush": asdict(g.flush),
                "matched_inbound": None,
                "near_stale_restarts": [asdict(h) for h in g.stale_restarts],
                "near_send_failures": [asdict(h) for h in g.send_failures],
            }
            for g in gaps
        ],
    }


def count_state_rows(db_path: Path, *, since: datetime, until: datetime) -> dict:
    result = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "telegram_message_rows_in_window": None,
        "note": "state.db sessions do not reliably store Telegram chat_id; count is source=telegram only",
        "error": None,
    }
    if not db_path.exists():
        return result
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT COUNT(*)
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.source = 'telegram'
                  AND m.timestamp >= ?
                  AND m.timestamp <= ?
                """,
                (since.timestamp(), until.timestamp()),
            ).fetchone()
            result["telegram_message_rows_in_window"] = int(rows[0] if rows else 0)
        finally:
            conn.close()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def build_summary(scan: dict, state: dict, *, chat_id: Optional[str]) -> list[str]:
    gaps = scan["flush_without_inbound"]
    lines = []
    subject = f"chat {chat_id}" if chat_id else "all chats"
    if gaps:
        lines.append(f"Potential flush-without-inbound incident detected for {subject}: {len(gaps)} unmatched flush(es).")
    else:
        lines.append(f"No flush-without-inbound gap detected for {subject} in the selected window.")
    lines.append(f"Flushes: {len(scan['flushes'])}; inbound lines: {len(scan['inbounds'])}; stale-code restarts: {len(scan['stale_restarts'])}; send failures: {len(scan['send_failures'])}.")
    if state.get("telegram_message_rows_in_window") is not None:
        lines.append(f"state.db Telegram message rows in window: {state['telegram_message_rows_in_window']}.")
    elif state.get("error"):
        lines.append(f"state.db check failed: {state['error']}.")
    else:
        lines.append("state.db check skipped: database missing.")
    return lines


def diagnose(
    *,
    log_path: Path,
    db_path: Path,
    since: datetime,
    until: datetime,
    chat_id: Optional[str],
    window_seconds: int,
) -> dict:
    scan = scan_gateway_log(log_path, since=since, until=until, chat_id=chat_id, window_seconds=window_seconds)
    state = count_state_rows(db_path, since=since, until=until)
    summary = build_summary(scan, state, chat_id=chat_id)
    return {
        "window": {"since": since.isoformat(sep=" "), "until": until.isoformat(sep=" ")},
        "chat_id": chat_id,
        "incident_detected": bool(scan["flush_without_inbound"]),
        "summary": summary,
        "log_scan": scan,
        "state_db": state,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    home = get_hermes_home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="Start time: ISO-ish, 'now', or 'now-10min'")
    parser.add_argument("--until", default="now", help="End time: ISO-ish, 'now', or 'now-10min'")
    parser.add_argument("--chat", dest="chat_id", default=None, help="Chat ID substring to match in gateway.log")
    parser.add_argument("--log", dest="log_path", type=Path, default=home / "logs" / "gateway.log")
    parser.add_argument("--state-db", dest="db_path", type=Path, default=home / "state.db")
    parser.add_argument("--window-seconds", type=int, default=20)
    parser.add_argument("--json-only", action="store_true", help="Emit JSON only")
    args = parser.parse_args(list(argv) if argv is not None else None)

    now = datetime.now()
    since = _parse_time_expr(args.since, now=now)
    until = _parse_time_expr(args.until, now=now)
    result = diagnose(
        log_path=args.log_path,
        db_path=args.db_path,
        since=since,
        until=until,
        chat_id=args.chat_id,
        window_seconds=args.window_seconds,
    )

    if not args.json_only:
        for line in result["summary"]:
            print(line)
        print()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["incident_detected"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
