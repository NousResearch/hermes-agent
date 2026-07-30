#!/usr/bin/env python3
"""Preflight for the Pencil CLI (this skill's lazy runtime dependency).

Checks `node` + `pencil` presence/version and `pencil status` auth, and reads
`pencil --help` to report which integration paths this build exposes (the CLI's
surface changes over time, so capabilities are discovered at runtime, never
hardcoded). Exit 0 iff `pencil` is installed.

    python pencil_doctor.py [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess


def _probe(argv, runner):
    """Run a short command, swallowing missing-binary/timeout into None."""
    try:
        return runner(argv, text=True, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _out(proc):
    return (proc.stdout or "").strip() if proc else None


def _major(version):
    head = (version or "").lstrip("v").split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _has_subcommand(help_text, word):
    """Whole-word match for a subcommand token in `pencil --help` output."""
    return bool(re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", help_text or ""))


def check(*, which=shutil.which, runner=subprocess.run) -> dict:
    """Structured environment status. ``which``/``runner`` are injectable."""
    node, pencil = which("node"), which("pencil")
    node_ver = _out(_probe([node, "--version"], runner)) if node else None
    pencil_ver = _out(_probe([pencil, "version"], runner)) if pencil else None
    auth = _probe([pencil, "status"], runner) if pencil else None
    # Discover the CLI surface at runtime instead of assuming a fixed set of
    # subcommands — the schema/tools/commands evolve between releases.
    help_text = _out(_probe([pencil, "--help"], runner)) if pencil else None
    return {
        "node": {
            "present": bool(node),
            "version": node_ver,
            "ok": (_major(node_ver) or 0) >= 18,
        },
        "pencil": {"present": bool(pencil), "version": pencil_ver},
        "auth": {"checked": auth is not None, "ok": bool(auth and auth.returncode == 0)},
        "capabilities": {
            "start": _has_subcommand(help_text, "start"),
            "interactive": _has_subcommand(help_text, "interactive"),
        },
    }


def _summary(s: dict) -> str:
    n, p, a, caps = s["node"], s["pencil"], s["auth"], s["capabilities"]
    lines = []

    if not n["present"]:
        lines.append("✗ node: not found — install Node.js (https://nodejs.org)")
    elif not n["ok"]:
        lines.append(
            f"⚠ node: {n['version']} — Pencil needs Node 18+ "
            "(newer builds need 22+; upgrade if you hit ERR_REQUIRE_ESM)"
        )
    else:
        lines.append(f"✓ node: {n['version']}")

    if not p["present"]:
        lines.append("✗ pencil: not found — `npm install -g @pencil.dev/cli`")
    else:
        lines.append(f"✓ pencil: {p['version'] or 'installed'}")

    if p["present"]:
        if not a["checked"]:
            lines.append("⚠ auth: could not run `pencil status`")
        elif a["ok"]:
            lines.append("✓ auth: authenticated")
        else:
            lines.append(
                "✗ auth: not authenticated — `pencil login` (or set "
                "PENCIL_CLI_KEY). REQUIRED: no offline mode, so even a headless "
                "session refuses to start."
            )

        # Recommend the best available integration path from the live surface.
        if caps["start"]:
            lines.append(
                "✓ path: `pencil start` present — prefer the headless MCP path "
                "(`hermes mcp add`). Verify flags with `pencil start --help`."
            )
        elif caps["interactive"]:
            lines.append(
                "✓ path: `pencil interactive` present — use the REPL wrapper "
                "(pencil_repl.py). Learn tools with `pencil interactive --help`."
            )
        else:
            lines.append(
                "⚠ path: neither `start` nor `interactive` seen in `pencil "
                "--help` — run `pencil --help` yourself to find the current one."
            )

    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the Pencil CLI environment.")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args(argv)

    status = check()
    print(json.dumps(status, indent=2) if args.json else _summary(status))
    return 0 if status["pencil"]["present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
