#!/usr/bin/env python3
"""
ds.py — CLI for the desired-state skill.

One dispatcher the agent drives to define goals, track current values, and
read the current-vs-desired gap. Human-readable by default; pass --json for
machine-parseable output (used by the agent and, later, the desktop panel).

Subcommands:
    define   <domain> <goal>        create a goal artifact
    track    <domain> <slug> <val>  record a new measured current value
    edit     <domain> <slug>        change frontmatter fields
    show     <domain> <slug>        print one artifact (raw or --json)
    list                            list goals (filter --domain/--status)
    gap      [<domain> <slug>]      gap for one goal, or all active goals
    report                          per-domain rollup: progress + what's behind
    milestone <domain> <slug>       list / add / check the plan's milestones
    archive  <domain> <slug>        soft-close (achieved | dropped | paused)

Stdlib-only. Python 3.11+.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from _common import (
    DIRECTIONS,
    HORIZONS,
    STATUSES,
    GoalDoc,
    add_milestone,
    desired_root,
    display_root,
    milestones_in,
    set_milestone,
)
from ds_store import (
    GoalExistsError,
    GoalNotFoundError,
    archive_goal,
    create_goal,
    get_goal,
    list_goals,
    set_current,
    update_goal,
)
from gap import GapResult, compute_gap, rollup


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PACE_GLYPH = {"ahead": "↑", "on_track": "→", "behind": "↓", "met": "✓", "unknown": "·"}


def _gap_line(doc: GoalDoc, res: GapResult) -> str:
    glyph = _PACE_GLYPH.get(res.pace, "·")
    return f"  {glyph} {doc.domain}/{doc.slug} — {doc.goal}: {res.summary}"


def _emit(obj: object, as_json: bool, human: str) -> None:
    print(json.dumps(obj, default=_json_default, indent=2) if as_json else human)


def _json_default(o: object) -> object:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _goal_json(doc: GoalDoc, res: GapResult | None = None) -> dict:
    data = {k: getattr(doc, k) for k in GoalDoc._FIELD_ORDER}
    data["slug"] = doc.slug
    data["path"] = str(doc.path) if doc.path else None
    if res is not None:
        data["gap"] = asdict(res)
    return data


# ---------------------------------------------------------------------------
# Field parsing shared by define/edit
# ---------------------------------------------------------------------------

def _add_field_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--horizon", choices=HORIZONS)
    p.add_argument("--status", choices=STATUSES)
    p.add_argument("--direction", choices=DIRECTIONS)
    p.add_argument("--target", dest="target_value")
    p.add_argument("--current", dest="current_value")
    p.add_argument("--baseline", dest="baseline_value")
    p.add_argument("--unit")
    p.add_argument("--target-date", dest="target_date")
    p.add_argument("--start-date", dest="start_date")
    p.add_argument("--source", dest="measurement_source")
    p.add_argument("--project", dest="linked_projects", action="append")
    p.add_argument("--person", dest="linked_people", action="append")
    p.add_argument("--todo", dest="linked_todos", action="append")
    p.add_argument("--tag", dest="tags", action="append")
    p.add_argument("--body", help="markdown body (or use --body-file)")
    p.add_argument("--body-file", help="read markdown body from a file ('-' for stdin)")


_FIELD_KEYS = (
    "horizon", "status", "direction", "target_value", "current_value",
    "baseline_value", "unit", "target_date", "start_date",
    "measurement_source", "linked_projects", "linked_people", "linked_todos", "tags",
)


def _collect_fields(args: argparse.Namespace) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key in _FIELD_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            fields[key] = _numify(key, val)
    body = _resolve_body(args)
    if body is not None:
        fields["body"] = body
    return fields


def _numify(key: str, val: object) -> object:
    if key in ("target_value", "current_value", "baseline_value") and isinstance(val, str):
        try:
            return int(val) if val.strip().lstrip("+-").isdigit() else float(val)
        except ValueError:
            return val
    return val


def _resolve_body(args: argparse.Namespace) -> str | None:
    if getattr(args, "body_file", None):
        if args.body_file == "-":
            return sys.stdin.read()
        try:
            return Path(args.body_file).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"could not read body file {args.body_file!r}: {e.strerror}") from e
    return getattr(args, "body", None)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_define(args: argparse.Namespace) -> int:
    try:
        fields = _collect_fields(args)
        doc = GoalDoc(domain=args.domain, goal=args.goal, body=str(fields.pop("body", "") or ""))
        for k, v in fields.items():
            setattr(doc, k, v)
        saved = create_goal(doc, overwrite=args.force)
    except GoalExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"invalid goal: {e}", file=sys.stderr)
        return 2
    res = compute_gap(saved)
    _emit(_goal_json(saved, res), args.json, f"defined {saved.domain}/{saved.slug}\n{_gap_line(saved, res)}")
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    try:
        doc = set_current(args.domain, args.slug, args.value)
    except (GoalNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = compute_gap(doc)
    _emit(_goal_json(doc, res), args.json, f"tracked {doc.domain}/{doc.slug}\n{_gap_line(doc, res)}")
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    try:
        fields = _collect_fields(args)
        if not fields:
            print("nothing to change: pass at least one field flag", file=sys.stderr)
            return 2
        doc = update_goal(args.domain, args.slug, fields)
    except GoalNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as e:
        print(f"invalid edit: {e}", file=sys.stderr)
        return 2
    res = compute_gap(doc)
    _emit(_goal_json(doc, res), args.json, f"updated {doc.domain}/{doc.slug}\n{_gap_line(doc, res)}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        doc = get_goal(args.domain, args.slug)
    except (GoalNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        _emit(_goal_json(doc, compute_gap(doc)), True, "")
    else:
        print(doc.to_text())
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    docs = list_goals(domain=args.domain)
    if args.status:
        docs = [d for d in docs if d.status == args.status]
    if args.json:
        _emit([_goal_json(d) for d in docs], True, "")
        return 0
    if not docs:
        print(f"no goals in {display_root()}")
        return 0
    for d in docs:
        print(f"  {d.domain}/{d.slug}  [{d.status}]  {d.goal}")
    return 0


def cmd_gap(args: argparse.Namespace) -> int:
    if args.domain and args.slug:
        try:
            doc = get_goal(args.domain, args.slug)
        except (GoalNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        res = compute_gap(doc)
        _emit(_goal_json(doc, res), args.json, _gap_line(doc, res).lstrip())
        return 0
    docs = [d for d in list_goals(domain=args.domain) if d.status == "active"]
    pairs = [(d, compute_gap(d)) for d in docs]
    if args.json:
        _emit([_goal_json(d, r) for d, r in pairs], True, "")
        return 0
    if not pairs:
        print("no active goals")
        return 0
    for d, r in pairs:
        print(_gap_line(d, r))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    pairs = [(d, compute_gap(d)) for d in list_goals(domain=args.domain)]
    rollups = rollup(pairs)
    if args.json:
        _emit([asdict(r) for r in rollups], True, "")
        return 0
    if not rollups:
        print("no goals yet")
        return 0
    for r in rollups:
        status_bits = " ".join(f"{k}:{v}" for k, v in sorted(r.by_status.items()))
        noun = "goal" if r.total == 1 else "goals"
        print(f"{r.domain}  ({r.total} {noun} — {status_bits})")
        for doc, res in pairs:
            if doc.domain == r.domain and doc.status == "active":
                print(_gap_line(doc, res))
        if r.ready_to_close:
            print(f"  ✓ ready to close: {', '.join(r.ready_to_close)}")
    return 0


def cmd_milestone(args: argparse.Namespace) -> int:
    try:
        doc = get_goal(args.domain, args.slug)
        if args.add is not None:
            doc = update_goal(args.domain, args.slug, {"body": add_milestone(doc.body, args.add)})
        elif args.check is not None:
            doc = update_goal(args.domain, args.slug, {"body": set_milestone(doc.body, args.check, True)})
        elif args.uncheck is not None:
            doc = update_goal(args.domain, args.slug, {"body": set_milestone(doc.body, args.uncheck, False)})
    except (GoalNotFoundError, ValueError, KeyError, IndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    steps = milestones_in(doc.body)
    if args.json:
        _emit([{"n": i, "done": c, "text": t} for i, (c, t) in enumerate(steps, 1)], True, "")
        return 0
    if not steps:
        print("no milestones")
        return 0
    for i, (c, t) in enumerate(steps, 1):
        print(f"  {i}. [{'x' if c else ' '}] {t}")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    try:
        doc = archive_goal(args.domain, args.slug, status=args.status)
    except GoalNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _emit(_goal_json(doc), args.json, f"{doc.domain}/{doc.slug} -> {doc.status}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ds", description="Desired-state goal tracking for Hermes.")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("define", help="create a goal")
    d.add_argument("domain")
    d.add_argument("goal")
    d.add_argument("--force", action="store_true", help="overwrite if it exists")
    _add_field_flags(d)
    d.set_defaults(func=cmd_define)

    t = sub.add_parser("track", help="record a new current value")
    t.add_argument("domain")
    t.add_argument("slug")
    t.add_argument("value")
    t.set_defaults(func=cmd_track)

    e = sub.add_parser("edit", help="change goal fields")
    e.add_argument("domain")
    e.add_argument("slug")
    _add_field_flags(e)
    e.set_defaults(func=cmd_edit)

    s = sub.add_parser("show", help="print one goal")
    s.add_argument("domain")
    s.add_argument("slug")
    s.set_defaults(func=cmd_show)

    ls = sub.add_parser("list", help="list goals")
    ls.add_argument("--domain")
    ls.add_argument("--status", choices=STATUSES)
    ls.set_defaults(func=cmd_list)

    g = sub.add_parser("gap", help="current-vs-desired gap")
    g.add_argument("domain", nargs="?")
    g.add_argument("slug", nargs="?")
    g.set_defaults(func=cmd_gap)

    r = sub.add_parser("report", help="per-domain rollup")
    r.add_argument("--domain")
    r.set_defaults(func=cmd_report)

    m = sub.add_parser("milestone", help="list / add / check milestones on a goal")
    m.add_argument("domain")
    m.add_argument("slug")
    mg = m.add_mutually_exclusive_group()
    mg.add_argument("--add", metavar="TEXT", help="append a new milestone")
    mg.add_argument("--check", type=int, metavar="N", help="mark milestone N done")
    mg.add_argument("--uncheck", type=int, metavar="N", help="mark milestone N not done")
    m.set_defaults(func=cmd_milestone)

    a = sub.add_parser("archive", help="soft-close a goal")
    a.add_argument("domain")
    a.add_argument("slug")
    a.add_argument("--status", choices=("achieved", "dropped", "paused"), default="dropped")
    a.set_defaults(func=cmd_archive)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
