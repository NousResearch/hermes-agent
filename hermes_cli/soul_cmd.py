"""``hermes soul`` — testable agent constitution commands.

``hermes soul validate`` loads SOUL.md + its paired probe suite for the current
project and runs the entry lint. ``hermes soul eval`` deterministically scores
probe responses and checks the release gates.

Exit codes: 0 = valid / SHIP, 1 = entry-lint failure or BLOCKED gates,
2 = bad input (no soul file, unreadable or unparseable files).
"""

from __future__ import annotations

import json
import os
import sys


class _BadInput(Exception):
    pass


def _read_file(path: str, label: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        print(f"soul: {label} not found: {path}", file=sys.stderr)
        raise _BadInput


def cmd_validate() -> int:
    from agent.soul_constitution import (
        SoulLoadError,
        entry_lint,
        find_constitution_file,
        is_constitution_content,
        parse_constitution,
    )
    from agent.soul_eval import parse_suite

    start = os.getcwd()
    path = find_constitution_file(start)
    if not path:
        print(f"soul: no SOUL.md found (looked up from {start} and the Hermes home)",
              file=sys.stderr)
        return 2
    try:
        content = _read_file(path, "soul file")
    except _BadInput:
        return 2
    if not is_constitution_content(content):
        print(f"soul: {path} is not a structured constitution "
              f"(no soul_version frontmatter + axioms section); nothing to validate",
              file=sys.stderr)
        return 2
    try:
        soul = parse_constitution(content, path)
    except SoulLoadError as e:
        print(f"soul: cannot parse {path}:", file=sys.stderr)
        for m in e.messages:
            print(f"  {m}", file=sys.stderr)
        return 2
    try:
        suite_text = _read_file(soul.suite_path, "eval suite")
    except _BadInput:
        print("soul: every soul needs a paired probe suite", file=sys.stderr)
        return 2
    try:
        suite = parse_suite(suite_text)
    except Exception as e:
        print(f"soul: cannot parse suite {soul.suite_path}: {e}", file=sys.stderr)
        return 2
    errs = entry_lint(soul, suite)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    print(f"soul valid: {path} ({len(soul.axioms)} axioms, suite {soul.suite_path})")
    return 0


def cmd_eval(args) -> int:
    from agent.soul_eval import parse_responses, parse_suite, score_all

    try:
        suite_text = _read_file(args.suite, "suite")
        responses_text = _read_file(args.responses, "responses")
        baseline_text = _read_file(args.baseline, "baseline") if args.baseline else None
    except _BadInput:
        return 2
    try:
        suite = parse_suite(suite_text)
    except Exception as e:
        print(f"soul: cannot parse suite {args.suite}: {e}", file=sys.stderr)
        return 2
    try:
        responses = parse_responses(responses_text)
    except ValueError as e:
        print(f"soul: cannot parse responses {args.responses}: {e}", file=sys.stderr)
        return 2
    baseline = None
    if baseline_text is not None:
        try:
            baseline = json.loads(baseline_text)
        except json.JSONDecodeError as e:
            print(f"soul: cannot parse baseline {args.baseline}: {e}", file=sys.stderr)
            return 2

    scored = score_all(suite, responses, baseline)
    report = scored["report"]
    print(report)
    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(report + "\n")
        except OSError as e:
            print(f"soul: cannot write report {args.report}: {e}", file=sys.stderr)
            return 2
    blocked = any(g.blocking for g in scored["gates"])
    return 1 if blocked else 0


def soul_command(args) -> int:
    """Dispatch ``hermes soul <validate|eval>``; returns the process exit code."""
    action = getattr(args, "soul_cmd", None)
    if action == "eval":
        return cmd_eval(args)
    return cmd_validate()
