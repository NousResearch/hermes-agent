"""Browser QA command helpers.

This module intentionally keeps the first-class browser QA surface thin: it
constructs a repeatable QA mission prompt, preloads the dogfood skill, and
then delegates execution to the normal Hermes chat runner. The browser tools remain the source of truth for evidence capture.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Callable, Iterable


_DEFAULT_TOOLSETS = ("browser", "vision", "file")
_DEFAULT_SKILL = "dogfood"


def _split_csv_items(value: object) -> list[str]:
    """Normalize argparse-style comma/list values into a string list."""

    if value is None:
        return []
    raw_items: Iterable[object]
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]

    items: list[str] = []
    for raw in raw_items:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                items.append(item)
    return items


def _append_unique(existing: object, additions: Iterable[str]) -> list[str]:
    items = _split_csv_items(existing)
    seen = {item.lower() for item in items}
    for item in additions:
        normalized = item.strip()
        if normalized and normalized.lower() not in seen:
            items.append(normalized)
            seen.add(normalized.lower())
    return items


def _default_output_dir(url: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in url.lower()).strip("-")
    safe = "-".join(part for part in safe.split("-") if part)[:80] or "site"
    return str(Path.cwd() / "dogfood-output" / safe)


def build_browser_qa_prompt(
    *,
    url: str,
    scope: str | None = None,
    output: str | None = None,
    max_pages: int = 5,
    notes: str | None = None,
) -> str:
    """Return the deterministic browser QA mission prompt used by the CLI."""

    output_dir = os.path.abspath(os.path.expanduser(output or _default_output_dir(url)))
    clean_scope = (scope or "smoke-test the core user journeys").strip()
    clean_notes = (notes or "").strip()

    sections = [
        "Run a browser QA mission for this web app.",
        "",
        f"Target URL: {url}",
        f"Scope: {clean_scope}",
        f"Maximum pages/journeys to inspect: {max_pages}",
        f"Output directory: {output_dir}",
        "",
        "Use the dogfood skill workflow. Be evidence-driven and do not report a clean pass if diagnostics are unavailable or failed.",
        "",
        "Required checks:",
        "1. Navigate to the target URL and capture the initial page state.",
        "2. Exercise the most important visible journeys within scope.",
        "3. After each meaningful action, inspect browser_console so console errors, JavaScript exceptions, failed requests, and diagnostic backend failures are included.",
        "4. Capture screenshots for important states or failures; in CLI reports, write screenshot file paths as plain paths, not MEDIA: tags.",
        "5. Classify issues by severity with reproduction steps, expected vs actual behavior, and supporting evidence.",
        "6. Save a Markdown report under the output directory using the dogfood report format, and include the absolute report path in the final answer.",
        "",
        "Final answer format:",
        "- Overall verdict",
        "- Issue counts by severity",
        "- Absolute report path",
        "- Any diagnostics that could not be captured",
    ]
    if clean_notes:
        sections.extend(["", "Additional tester notes:", clean_notes])
    return "\n".join(sections)


def cmd_browser_qa(args: Namespace, chat_runner: Callable[[Namespace], object]) -> object:
    """Prepare args for ``cmd_chat`` and dispatch a browser QA run."""

    prompt = build_browser_qa_prompt(
        url=args.url,
        scope=getattr(args, "scope", None),
        output=getattr(args, "output", None),
        max_pages=getattr(args, "max_pages", 5),
        notes=getattr(args, "notes", None),
    )

    args.query = prompt
    args.skills = _append_unique(getattr(args, "skills", None), [_DEFAULT_SKILL])
    args.toolsets = ",".join(
        _append_unique(getattr(args, "toolsets", None), _DEFAULT_TOOLSETS)
    )
    args.source = getattr(args, "source", None) or "browser-qa"
    args.image = None
    args.tui = False
    args.tui_dev = False

    # cmd_chat expects these attributes to exist when called through its normal parser.
    for name, default in {
        "verbose": False,
        "quiet": False,
        "resume": None,
        "worktree": False,
        "checkpoints": False,
        "pass_session_id": False,
        "max_turns": None,
        "ignore_rules": False,
        "ignore_user_config": False,
        "yolo": False,
        "continue_last": None,
        "model": None,
        "provider": None,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, default)

    return chat_runner(args)
