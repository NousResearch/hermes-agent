#!/usr/bin/env python3
"""Unified Hermes-native CLI for Kubrick's deterministic tools.

Primary surface:
  kubrick do <intent> [--action <action>] [flags]

Also:
  kubrick recipe <name>
  kubrick help [<intent>]
  kubrick aliases
  <legacy-command> [flags]   # soft aliases via intent_router
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable

sys.path.insert(0, str(SCRIPTS))
import intent_router as ir  # noqa: E402
from diagnostics import abort, diagnostic  # noqa: E402


def _abort_router(error: ir.RouterError) -> None:
    abort(
        diagnostic(
            status="INVALID_COMMAND",
            code="ROUTER_ERROR",
            exit_code=error.exit_code,
            message=error.message,
            context={"surface": "kubrick"},
        )
    )


def _is_agent() -> bool:
    """Quiet mode for agents / non-TTY (suppress alias deprecation notes)."""
    return os.environ.get("KUBRICK_AGENT") == "1" or not sys.stderr.isatty()


def _dispatch_smoke(call: ir.IntentCall) -> None:
    """check smoke: validate manifest, skill, then pattern corpus (composite)."""
    for script in (
        "validate_manifest.py",
        "validate_hermes_skill.py",
        "validate_pattern_corpus.py",
    ):
        r = subprocess.run([PY, str(SCRIPTS / script), *call.argv], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    print("check smoke: PASS")
    raise SystemExit(0)


def main() -> None:
    argv = sys.argv[1:]

    # Top-level help (prefer entrypoint handling; avoid RouterError("HELP") path)
    if not argv or (argv[0] in {"-h", "--help", "help"} and len(argv) == 1):
        sys.stdout.write(ir.format_top_level_help())
        raise SystemExit(0)

    if argv[0] == "help" and len(argv) >= 2:
        try:
            sys.stdout.write(ir.format_intent_help(argv[1]))
        except ir.RouterError as e:
            _abort_router(e)
        raise SystemExit(0)

    if argv[0] == "aliases":
        sys.stdout.write(ir.format_aliases())
        raise SystemExit(0)

    if argv[0] == "recipe":
        if len(argv) < 2:
            _abort_router(ir.RouterError("usage: kubrick recipe <name>"))
        try:
            argv = ir.resolve_recipe(argv[1])
        except ir.RouterError as e:
            _abort_router(e)

    # do <intent> --help / -h → intent help (do not pass --help into scripts)
    if (
        len(argv) >= 2
        and argv[0] == "do"
        and any(tok in {"-h", "--help"} for tok in argv[2:])
    ):
        try:
            sys.stdout.write(ir.format_intent_help(argv[1]))
        except ir.RouterError as e:
            _abort_router(e)
        raise SystemExit(0)

    try:
        call = ir.resolve(argv)
    except ir.RouterError as e:
        if e.message == "HELP":
            sys.stdout.write(ir.format_top_level_help())
            raise SystemExit(0)
        _abort_router(e)

    if (
        call.legacy_name
        and not _is_agent()
        and os.environ.get("KUBRICK_QUIET") != "1"
    ):
        print(
            f"note: `{call.legacy_name}` is a legacy alias; "
            f"prefer `kubrick do {call.intent} --action {call.action}`",
            file=sys.stderr,
        )

    if call.intent == "check" and call.action == "smoke":
        _dispatch_smoke(call)

    result = subprocess.run([PY, str(call.script), *call.argv], cwd=ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
