"""Command-line wrapper around the `caldav` library for a self-hosted
Radicale calendar. JSON output throughout, matching the output contract
other Hermes calendar skills use (e.g. the bundled google-workspace
skill's `calendar` subcommand) so results are easy to parse rather than
eyeballed from plain text.

Reads RADICALE_URL / RADICALE_USERNAME / RADICALE_PASSWORD from the process
environment first, then from `~/.hermes/.env` (where Hermes's secure
setup-on-load prompt actually stores them) and a `.env` in the current
directory, in that order. Pass --env-path to load one specific file instead
(e.g. running this script outside the main Hermes environment).

Deliberately named `reschedule-event` rather than `move-event` - other
CalDAV tools use "move" to mean transferring an event to a *different
calendar*, not changing its date/time. Avoid that ambiguity here:
`reschedule-event` changes when an event happens, nothing else does
calendar-to-calendar transfer (not needed yet - add it if it ever comes
up).

Run: python radicale_cli.py <command> [args]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import caldav


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _dotenv_paths(env_path: str | None) -> list[Path]:
    # An explicit --env-path loads only that file (standalone/outside-Hermes
    # use). Otherwise fall back to the same locations Hermes itself uses -
    # notably ~/.hermes/.env, where the secure setup-on-load prompt actually
    # writes RADICALE_PASSWORD. A skill-local .env is deliberately not part
    # of this list: it would be a second, unmanaged plaintext secret store
    # outside Hermes's own, contradicting the "prompts securely" story.
    if env_path:
        return [Path(env_path)]
    paths = []
    project_env = Path.cwd() / ".env"
    if project_env.exists():
        paths.append(project_env)
    user_env = _hermes_home() / ".env"
    if user_env.exists():
        paths.append(user_env)
    return paths


def _load_dotenv_values(env_path: str | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in _dotenv_paths(env_path):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key:
                    values[key] = value
    return values


def _env_lookup(key: str, dotenv_values: dict[str, str]) -> str | None:
    # Process environment wins over any dotenv file - lets a caller override
    # without editing a file on disk.
    return os.environ.get(key) or dotenv_values.get(key)


def _client(env_path: str | None) -> caldav.DAVClient:
    dotenv_values = _load_dotenv_values(env_path)
    creds = {}
    for var in ("RADICALE_URL", "RADICALE_USERNAME", "RADICALE_PASSWORD"):
        value = _env_lookup(var, dotenv_values)
        if not value:
            _fail(
                f"Missing {var} - set it via `hermes setup`, "
                f"{_hermes_home() / '.env'}, or --env-path"
            )
        creds[var] = value
    return caldav.DAVClient(
        url=creds["RADICALE_URL"],
        username=creds["RADICALE_USERNAME"],
        password=creds["RADICALE_PASSWORD"],
    )


def _fail(message: str) -> None:
    print(json.dumps({"error": message}), file=sys.stderr)
    sys.exit(1)


def _get_calendar(principal, name: str | None):
    calendars = principal.calendars()
    if not calendars:
        _fail("No calendars exist yet - create one first (e.g. via a real CalDAV client)")
    if name is None:
        return calendars[0]
    for cal in calendars:
        if cal.get_display_name() == name:
            return cal
    available = ", ".join(c.get_display_name() or "(unnamed)" for c in calendars)
    _fail(f"No calendar named '{name}'. Available: {available}")


def _parse_dt(value: str) -> datetime:
    # Accept both a bare "Z" suffix and explicit offsets.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now_local() -> datetime:
    # Timezone-aware, not naive - cal.search(..., expand=True) hands back
    # events with aware dtstart/dtend once recurrence expansion kicks in, and
    # comparing those against a naive default silently misses matches (or
    # raises TypeError: can't compare offset-naive and offset-aware
    # datetimes) instead of failing loudly.
    return datetime.now().astimezone()


def _parse_date(value: str) -> date:
    # Plain YYYY-MM-DD, no time component - for --all-day events. A date
    # (not datetime) object is what makes icalendar emit DTSTART;VALUE=DATE
    # instead of a timed DTSTART - confirmed live against Radicale
    # (2026-08-11) rather than assumed from the RFC.
    return date.fromisoformat(value)


def _event_json(event) -> dict:
    comp = event.icalendar_component
    dtstart = comp.get("dtstart")
    dtend = comp.get("dtend")
    rrule = comp.get("rrule")
    return {
        "uid": str(comp.get("uid")),
        "summary": str(comp.get("summary", "")),
        # .dt is a `date` for --all-day events, `datetime` otherwise - both
        # have .isoformat(), so this stays correct either way (a bare
        # "YYYY-MM-DD" for all-day, full timestamp for timed events).
        "start": dtstart.dt.isoformat() if dtstart else None,
        "end": dtend.dt.isoformat() if dtend else None,
        "location": str(comp.get("location", "")) or None,
        "description": str(comp.get("description", "")) or None,
        "calendar": event.parent.get_display_name(),
        "all_day": bool(dtstart and not isinstance(dtstart.dt, datetime)),
        "recurring": rrule.to_ical().decode() if rrule else None,
    }


def cmd_list_calendars(client: caldav.DAVClient, args) -> None:
    principal = client.principal()
    result = [{"name": c.get_display_name(), "url": str(c.url)} for c in principal.calendars()]
    print(json.dumps(result, indent=2))


def cmd_list_events(client: caldav.DAVClient, args) -> None:
    principal = client.principal()
    cal = _get_calendar(principal, args.calendar)
    start = _parse_dt(args.start) if args.start else _now_local()
    end = _parse_dt(args.end) if args.end else start + timedelta(days=7)
    events = cal.search(start=start, end=end, event=True, expand=True)
    print(json.dumps([_event_json(e) for e in events], indent=2))


def cmd_create_event(client: caldav.DAVClient, args) -> None:
    principal = client.principal()
    cal = _get_calendar(principal, args.calendar)
    if args.all_day:
        start = _parse_date(args.start)
        # DTEND is exclusive in iCalendar - a single-day all-day event's
        # end is the NEXT day, not the same day. Default to that unless an
        # explicit --end was given (e.g. for a multi-day all-day event).
        end = _parse_date(args.end) if args.end else start + timedelta(days=1)
    else:
        if not args.end:
            _fail("--end is required unless --all-day is set")
        start = _parse_dt(args.start)
        end = _parse_dt(args.end)
    kwargs = {"dtstart": start, "dtend": end, "summary": args.summary}
    if args.location:
        kwargs["location"] = args.location
    if args.description:
        kwargs["description"] = args.description
    if args.recur == "yearly":
        kwargs["rrule"] = {"FREQ": "YEARLY"}
    event = cal.add_event(**kwargs)
    print(json.dumps({"status": "created", **_event_json(event)}, indent=2))


def cmd_reschedule_event(client: caldav.DAVClient, args) -> None:
    principal = client.principal()
    cal = _get_calendar(principal, args.calendar)
    event = cal.event_by_uid(args.uid)
    if event is None:
        _fail(f"No event with uid '{args.uid}'")
    old = _event_json(event)
    ical = event.icalendar_component
    if args.all_day:
        # Preserve all-day-ness - assigning a `datetime` here (via
        # _parse_dt) would silently turn an all-day event into a timed one.
        start = _parse_date(args.start)
        end = _parse_date(args.end) if args.end else start + timedelta(days=1)
    else:
        if not args.end:
            _fail("--end is required unless --all-day is set")
        start = _parse_dt(args.start)
        end = _parse_dt(args.end)
    ical["dtstart"].dt = start
    ical["dtend"].dt = end
    event.save()
    print(
        json.dumps(
            {
                "status": "rescheduled",
                "old_start": old["start"],
                "old_end": old["end"],
                **_event_json(event),
            },
            indent=2,
        )
    )


def _set_or_clear(ical, field: str, value: str | None) -> None:
    # None (flag not passed) means leave alone. Empty string means clear -
    # delete the property entirely rather than leaving an empty one behind
    # (an empty DESCRIPTION: line is different, and uglier, than no
    # DESCRIPTION property at all).
    if value is None:
        return
    if value == "":
        if field in ical:
            del ical[field]
        return
    ical[field] = value


def cmd_update_event(client: caldav.DAVClient, args) -> None:
    principal = client.principal()
    cal = _get_calendar(principal, args.calendar)
    event = cal.event_by_uid(args.uid)
    if event is None:
        _fail(f"No event with uid '{args.uid}'")
    ical = event.icalendar_component
    _set_or_clear(ical, "summary", args.summary)
    _set_or_clear(ical, "location", args.location)
    _set_or_clear(ical, "description", args.description)
    event.save()
    print(json.dumps({"status": "updated", **_event_json(event)}, indent=2))


def cmd_delete_event(client: caldav.DAVClient, args) -> None:
    principal = client.principal()
    cal = _get_calendar(principal, args.calendar)
    event = cal.event_by_uid(args.uid)
    if event is None:
        _fail(f"No event with uid '{args.uid}'")
    summary = str(event.icalendar_component.get("summary", ""))
    event.delete()
    print(json.dumps({"status": "deleted", "uid": args.uid, "summary": summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="radicale", description="Control a self-hosted Radicale calendar."
    )
    parser.add_argument(
        "--env-path",
        default=None,
        help="Load env vars from this file only, instead of ~/.hermes/.env / ./.env",
    )
    parser.add_argument(
        "--calendar", default=None, help="Calendar name (default: the first/only one)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-calendars").set_defaults(func=cmd_list_calendars)

    p_list = sub.add_parser("list-events", help="Defaults to the next 7 days")
    p_list.add_argument("--start", help="ISO 8601, e.g. 2026-08-12T00:00:00")
    p_list.add_argument("--end", help="ISO 8601")
    p_list.set_defaults(func=cmd_list_events)

    p_create = sub.add_parser("create-event")
    p_create.add_argument("--summary", required=True)
    p_create.add_argument(
        "--start", required=True, help="ISO 8601 datetime, or YYYY-MM-DD with --all-day"
    )
    p_create.add_argument(
        "--end",
        help="ISO 8601, or YYYY-MM-DD with --all-day. Required unless --all-day "
        "(defaults to the day after --start)",
    )
    p_create.add_argument("--location")
    p_create.add_argument("--description")
    p_create.add_argument(
        "--all-day",
        action="store_true",
        help="Whole-day event (e.g. a birthday) - no specific time",
    )
    p_create.add_argument(
        "--recur",
        choices=["yearly"],
        help="Recurrence rule. 'yearly' for annually-repeating events like birthdays",
    )
    p_create.set_defaults(func=cmd_create_event)

    p_resched = sub.add_parser("reschedule-event", help="Change an event's start/end time")
    p_resched.add_argument("--uid", required=True)
    p_resched.add_argument("--start", required=True, help="ISO 8601, or YYYY-MM-DD with --all-day")
    p_resched.add_argument(
        "--end", help="ISO 8601, or YYYY-MM-DD with --all-day. Required unless --all-day"
    )
    p_resched.add_argument(
        "--all-day", action="store_true", help="Keep/make this an all-day event, not a timed one"
    )
    p_resched.set_defaults(func=cmd_reschedule_event)

    p_update = sub.add_parser(
        "update-event", help="Change summary/location/description, not the time"
    )
    p_update.add_argument("--uid", required=True)
    p_update.add_argument("--summary", help="Pass an empty string to clear")
    p_update.add_argument("--location", help="Pass an empty string to clear")
    p_update.add_argument("--description", help="Pass an empty string to clear")
    p_update.set_defaults(func=cmd_update_event)

    p_delete = sub.add_parser("delete-event")
    p_delete.add_argument("--uid", required=True)
    p_delete.set_defaults(func=cmd_delete_event)

    args = parser.parse_args()
    client = _client(args.env_path)
    args.func(client, args)


if __name__ == "__main__":
    main()
