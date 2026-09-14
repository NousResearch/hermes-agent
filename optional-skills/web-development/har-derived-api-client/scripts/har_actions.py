#!/usr/bin/env python3
"""Shared --action parser for the HAR capture scripts.

Specs:
  fill:SELECTOR:TEXT | press:SELECTOR:KEY | click:SELECTOR
  goto:URL | sleep:SECONDS
"""
from __future__ import annotations

import time


def parse_action(spec: str) -> tuple[str, list[str]]:
    """Return (kind, args). Raise ValueError on a malformed spec."""
    if not spec:
        raise ValueError("empty action spec")
    parts = spec.split(":", 2)
    kind = parts[0]
    if kind == "fill":
        if len(parts) < 3 or parts[1] == "":
            raise ValueError(f"fill needs fill:SELECTOR:TEXT, got {spec!r}")
        return kind, [parts[1], parts[2]]
    if kind == "press":
        if len(parts) < 3 or parts[1] == "":
            raise ValueError(f"press needs press:SELECTOR:KEY, got {spec!r}")
        return kind, [parts[1], parts[2]]
    if kind == "click":
        if len(parts) < 2 or parts[1] == "":
            raise ValueError(f"click needs click:SELECTOR, got {spec!r}")
        return kind, [parts[1]]
    if kind == "goto":
        if len(parts) < 2 or parts[1] == "":
            raise ValueError(f"goto needs goto:URL, got {spec!r}")
        url = parts[1] + (":" + parts[2] if len(parts) > 2 else "")
        return kind, [url]
    if kind == "sleep":
        if len(parts) != 2 or parts[1] == "":
            raise ValueError(f"sleep needs sleep:SECONDS, got {spec!r}")
        try:
            seconds = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"sleep needs a number of seconds, got {spec!r}") from exc
        return kind, [seconds]
    raise ValueError(f"unknown action: {spec!r}")


def run_action(page, spec: str) -> None:
    kind, args = parse_action(spec)
    if kind == "fill":
        page.fill(args[0], args[1])
    elif kind == "press":
        page.press(args[0], args[1])
    elif kind == "click":
        page.click(args[0])
    elif kind == "goto":
        page.goto(args[0])
    elif kind == "sleep":
        time.sleep(args[0])


def choose_drive_page(context, *, new_page: bool):
    """Pick a page to drive.

    ``new_page=True`` (CDP ``--goto``) always opens a tab so we do not
    navigate a tab Hermes is already using.
    """
    if new_page or not getattr(context, "pages", None):
        return context.new_page()
    return context.pages[-1]


def flush_pending(pending: dict, entries: list, make_entry) -> None:
    """Turn in-flight requests into incomplete HAR entries, then clear ``pending``.

    Leftovers have no completed response. Playwright's ``request.response()``
    blocks until one arrives (no timeout), so it must never be called here.
    An entry with a null response is the correct partial-capture outcome.
    """
    for req in list(pending.values()):
        entries.append(make_entry(req, None))
    pending.clear()
