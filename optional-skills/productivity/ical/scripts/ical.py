"""iCalendar (RFC 5545) read / write / validate.

Stdlib only. No third-party dependencies.

CLI:
    ical.py read <file.ics> [--json]
    ical.py write <events.json> <out.ics>
    ical.py validate <file.ics>
    ical.py quick <summary> <start> <end> [--location L] [--description D]
                              [--organizer "Name <email>"]

Library:
    from ical import parse_ics, generate_ics, Event

Scope:
    Single-calendar files with flat VEVENTs. RRULE is preserved as a
    raw string but not expanded. EXDATE / RDATE are not handled. METHOD
    is not emitted (add it manually for email-invite semantics).
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


# Lines longer than this get folded (CRLF + space continuation) per RFC 5545.
FOLD_WIDTH = 75

# Properties that take a TEXT value, where commas / semicolons / backslashes
# need escaping on write and unescaping on read.
_TEXT_PROPERTIES = {"SUMMARY", "DESCRIPTION", "LOCATION", "COMMENT"}


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class Event:
    summary: str
    start: str
    end: str
    description: str = ""
    location: str = ""
    uid: str = ""
    organizer: str = ""
    # Raw RRULE string, preserved verbatim (not parsed or expanded).
    rrule: str = ""
    extra: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Escape / unescape for TEXT properties
# ---------------------------------------------------------------------------


def _escape_text(value: str) -> str:
    """Escape per RFC 5545 §3.3.11: backslash, then comma, then semicolon.
    Newlines in the source string are encoded as the two-character sequence
    `\\n` (literal backslash + n) — Python string escapes are NOT applied
    here, so the caller must pass through `\n` in JSON.
    """
    out = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == ";":
            out.append("\\;")
        elif ch == ",":
            out.append("\\,")
        elif ch == "\n":
            out.append("\\n")
        else:
            out.append(ch)
    return "".join(out)


def _unescape_text(value: str) -> str:
    out = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt == "n" or nxt == "N":
                out.append("\n")
            elif nxt == "\\":
                out.append("\\")
            elif nxt == ";":
                out.append(";")
            elif nxt == ",":
                out.append(",")
            else:
                # Unknown escape — keep both chars literally. RFC says we
                # SHOULD drop unknown escapes; in practice keeping them
                # makes round-tripping safer for non-conformant input.
                out.append(ch)
                out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Line folding (RFC 5545 §3.1)
# ---------------------------------------------------------------------------


def _fold_line(line: str) -> str:
    """Fold lines longer than 75 octets. Continuation lines start with a
    single space. CRLF is the line terminator.
    """
    if len(line.encode("utf-8")) <= FOLD_WIDTH:
        return line
    # We split on raw characters but treat multi-byte chars as a single unit
    # by tracking byte length. Simplest correct approach: encode to bytes
    # and split at byte boundaries that align to char boundaries.
    encoded = line.encode("utf-8")
    chunks: list[str] = []
    pos = 0
    while pos < len(encoded):
        end = min(pos + FOLD_WIDTH, len(encoded))
        # Walk back to a char boundary that doesn't split a UTF-8 sequence.
        while end < len(encoded) and (encoded[end] & 0xC0) == 0x80:
            end -= 1
        chunks.append(encoded[pos:end].decode("utf-8"))
        pos = end
    return ("\r\n ".join(chunks))


def _unfold_lines(text: str) -> list[str]:
    """Unfold CRLF+space / CRLF+tab continuation lines per RFC 5545."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    unfolded: list[str] = []
    for line in raw:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return [ln for ln in unfolded if ln]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_ics(text: str) -> list[Event]:
    """Parse a VCALENDAR into a list of Events. Unknown properties land
    in `Event.extra` as raw strings. Returns an empty list for a
    calendar with no VEVENTs.
    """
    events: list[Event] = []
    in_event = False
    event: Event | None = None
    extra: dict[str, str] = {}

    for line in _unfold_lines(text):
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        # Property parameters (e.g. DTSTART;TZID=America/New_York:20260615T123000)
        # land in `name` as "DTSTART;TZID=America/New_York". We drop the
        # parameters for the simple Event dataclass — callers that need
        # TZID / VALUE=DATE can read `extra`.
        prop_name = name.split(";", 1)[0].upper()

        if prop_name == "BEGIN" and value.strip().upper() == "VEVENT":
            in_event = True
            event = Event(summary="", start="", end="")
            extra = {}
            continue
        if prop_name == "END" and value.strip().upper() == "VEVENT":
            assert event is not None
            event.extra = extra
            events.append(event)
            in_event = False
            event = None
            extra = {}
            continue
        if prop_name in ("BEGIN", "END"):
            # VCALENDAR / VTIMEZONE etc. — skip their bodies.
            continue
        if not in_event:
            continue
        assert event is not None

        unescaped = _unescape_text(value) if prop_name in _TEXT_PROPERTIES else value
        if prop_name == "SUMMARY":
            event.summary = unescaped
        elif prop_name == "DTSTART":
            event.start = unescaped
        elif prop_name == "DTEND":
            event.end = unescaped
        elif prop_name == "DESCRIPTION":
            event.description = unescaped
        elif prop_name == "LOCATION":
            event.location = unescaped
        elif prop_name == "UID":
            event.uid = unescaped
        elif prop_name == "ORGANIZER":
            event.organizer = unescaped
        elif prop_name == "RRULE":
            event.rrule = unescaped
        else:
            extra[prop_name] = value  # keep raw (un-unscaped) value

    return events


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _now_utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _default_uid() -> str:
    return f"{uuid.uuid4()}@{socket.gethostname() or 'localhost'}"


def _format_property(name: str, value: str, escape: bool = False) -> str:
    if escape:
        value = _escape_text(value)
    return f"{name}:{value}"


def generate_ics(events: Iterable[Event], *, prodid: str = "-//ical-skill//EN") -> str:
    """Render a list of Events as a VCALENDAR string. Lines are folded
    at 75 octets and terminated with CRLF per RFC 5545.
    """
    out: list[str] = ["BEGIN:VCALENDAR", "VERSION:2.0", f"PRODID:{prodid}", "CALSCALE:GREGORIAN"]
    stamp = _now_utc_stamp()
    for ev in events:
        out.append("BEGIN:VEVENT")
        out.append(_fold_line(_format_property("UID", ev.uid or _default_uid())))
        out.append(_fold_line(_format_property("DTSTAMP", stamp)))
        if ev.start:
            out.append(_fold_line(_format_property("DTSTART", ev.start)))
        if ev.end:
            out.append(_fold_line(_format_property("DTEND", ev.end)))
        if ev.summary:
            out.append(_fold_line(_format_property("SUMMARY", ev.summary, escape=True)))
        if ev.description:
            out.append(_fold_line(_format_property("DESCRIPTION", ev.description, escape=True)))
        if ev.location:
            out.append(_fold_line(_format_property("LOCATION", ev.location, escape=True)))
        if ev.organizer:
            out.append(_fold_line(_format_property("ORGANIZER", ev.organizer)))
        if ev.rrule:
            out.append(_fold_line(_format_property("RRULE", ev.rrule)))
        for name, value in ev.extra.items():
            out.append(_fold_line(_format_property(name, value)))
        out.append("END:VEVENT")
    out.append("END:VCALENDAR")
    return "\r\n".join(out) + "\r\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ics(text: str) -> list[str]:
    """Return a list of human-readable problems. Empty list = OK.

    Checks: VCALENDAR open/close, at least one VEVENT, required fields per
    VEVENT (UID, DTSTAMP, DTSTART). DTEND-vs-DTSTART order is NOT checked
    (clients tolerate both orderings).
    """
    problems: list[str] = []
    lines = _unfold_lines(text)
    if not lines:
        return ["file is empty"]
    if lines[0] != "BEGIN:VCALENDAR":
        problems.append(f"first line must be 'BEGIN:VCALENDAR', got {lines[0]!r}")
    if lines[-1] != "END:VCALENDAR":
        problems.append(f"last line must be 'END:VCALENDAR', got {lines[-1]!r}")

    saw_event = False
    required = {"UID", "DTSTAMP", "DTSTART"}
    seen_required: set[str] = set()
    in_event = False
    for line in lines:
        prop_name = line.split(":", 1)[0].split(";", 1)[0].upper()
        if prop_name == "BEGIN" and line.partition(":")[2].strip().upper() == "VEVENT":
            saw_event = True
            in_event = True
            seen_required = set()
            continue
        if prop_name == "END" and line.partition(":")[2].strip().upper() == "VEVENT":
            missing = required - seen_required
            if missing:
                problems.append(f"VEVENT missing required properties: {sorted(missing)}")
            in_event = False
            continue
        if in_event and prop_name in required:
            seen_required.add(prop_name)
    if not saw_event:
        problems.append("no VEVENT blocks found")
    return problems


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_dt(value: str) -> str:
    """Best-effort humanization: 2026-06-15T12:00:00 -> 2026-06-15 12:00:00.
    Falls back to the raw string if it doesn't look like ISO 8601.
    """
    if "T" in value and len(value) >= 16:
        head, _, tail = value.partition("T")
        body = tail.rstrip("Z")
        if len(body) >= 5 and body[2] == ":":
            return f"{head} {body}"
    return value


def format_table(events: list[Event]) -> str:
    """Render events as a fixed-width table. Best-effort; doesn't
    align perfectly across all widths.
    """
    if not events:
        return "(no events)"
    headers = ("SUMMARY", "START", "END", "LOCATION")
    widths = (40, 20, 20, 24)
    sep = "  "
    rows = [headers]
    for ev in events:
        rows.append((
            _truncate(ev.summary, widths[0]),
            _truncate(_format_dt(ev.start), widths[1]),
            _truncate(_format_dt(ev.end), widths[2]),
            _truncate(ev.location, widths[3]),
        ))
    out_lines = []
    for row in rows:
        cells = [cell.ljust(w) for cell, w in zip(row, widths)]
        out_lines.append(sep.join(cells).rstrip())
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _events_from_json_path(path: Path) -> list[Event]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"event JSON must be a list, got {type(data).__name__}")
    out: list[Event] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise SystemExit("each event must be a JSON object")
        out.append(Event(**{k: v for k, v in entry.items() if k in Event.__dataclass_fields__}))
    return out


def _read_events_from_stdin() -> list[Event]:
    events = parse_ics(sys.stdin.read())
    return events


def cmd_read(args: argparse.Namespace) -> int:
    if args.file == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.file).read_text(encoding="utf-8")
    events = parse_ics(text)
    if args.json:
        json.dump([asdict(e) for e in events], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_table(events) + "\n")
    return 0


def cmd_write(args: argparse.Namespace) -> int:
    events = _events_from_json_path(Path(args.input))
    Path(args.output).write_text(generate_ics(events), encoding="utf-8", newline="")
    print(f"wrote {len(events)} event(s) to {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    problems = validate_ics(text)
    if problems:
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(f"OK: {args.file} parses and contains at least one VEVENT")
    return 0


def cmd_quick(args: argparse.Namespace) -> int:
    event = Event(
        summary=args.summary,
        start=args.start,
        end=args.end,
        location=args.location or "",
        description=args.description or "",
        organizer=args.organizer or "",
    )
    sys.stdout.write(generate_ics([event]))
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    """Internal: parse stdin, re-emit on stdout. Used by the SKILL.md
    verification command."""
    events = _read_events_from_stdin()
    sys.stdout.write(format_table(events) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read, write, and validate iCalendar (.ics) files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_read = sub.add_parser("read", help="Print events from an .ics file as a table or JSON.")
    p_read.add_argument("file")
    p_read.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    p_read.set_defaults(func=cmd_read)

    p_write = sub.add_parser("write", help="Generate an .ics from a JSON event list.")
    p_write.add_argument("input", help="Path to a JSON file containing a list of events.")
    p_write.add_argument("output", help="Path to write the .ics file to.")
    p_write.set_defaults(func=cmd_write)

    p_val = sub.add_parser("validate", help="Parse and report problems; exit 0 on success.")
    p_val.add_argument("file")
    p_val.set_defaults(func=cmd_validate)

    p_quick = sub.add_parser("quick", help="Generate a one-event .ics on stdout.")
    p_quick.add_argument("summary")
    p_quick.add_argument("start", help="ISO-8601 datetime (e.g. 2026-06-15T12:30:00).")
    p_quick.add_argument("end")
    p_quick.add_argument("--location", default="")
    p_quick.add_argument("--description", default="")
    p_quick.add_argument("--organizer", default="")
    p_quick.set_defaults(func=cmd_quick)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
