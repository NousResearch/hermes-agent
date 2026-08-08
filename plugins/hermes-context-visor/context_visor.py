#!/usr/bin/env python3
"""Hermes Context Visor / Cockpit — read-only real-time context dashboard.

A standalone sidecar that renders a labeled, glanceable Flight Deck for the
active Hermes session. It NEVER writes to any Hermes file and NEVER calls an
LLM. It only reads:

  * state.db   -> sessions + messages
  * lcm.db     -> compaction telemetry (provider-counted last prompt)
  * context_length_cache.yaml + fallback map -> model window size
  * .skills_prompt_snapshot.json -> skills block size
  * config.yaml / .env -> LCM threshold
  * gateway_state.json / processes.json / live process scan -> freshness

Run:  hermes-context-visor [--profile personal-ops] [--serve] [--json]
      default / --serve  run the graphical browser cockpit on localhost
      --once            render a single Rich frame to stdout and exit
      --terminal        run the legacy Rich terminal fallback loop
      --json            emit status classification + metrics JSON (CI / proof)

Quit: Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Allow `python context_visor.py` when the package sits beside this file.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from context_cockpit.launcher import build_visor_url  # noqa: E402
from context_cockpit.metrics import collect_metrics  # noqa: E402
from context_cockpit.render import build_cockpit  # noqa: E402
from context_cockpit.status import build_status_payload, classify_status  # noqa: E402
from context_cockpit.web import serve_context_cockpit  # noqa: E402

REFRESH_SEC = 10.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hermes Context Cockpit — read-only context health dashboard"
    )
    ap.add_argument(
        "--profile",
        default=os.environ.get("HERMES_VISOR_PROFILE") or os.environ.get("HERMES_PROFILE") or "default",
    )
    ap.add_argument("--hermes-home", default=None, help="override profile directory")
    ap.add_argument("--once", action="store_true", help="render one Rich frame and exit")
    ap.add_argument(
        "--terminal",
        action="store_true",
        help="run the legacy Rich terminal fallback instead of the browser cockpit",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit JSON status+metrics (no Rich live layout)",
    )
    ap.add_argument("--serve", action="store_true", help="run the graphical cockpit server")
    ap.add_argument("--host", default="127.0.0.1", help="server bind host")
    ap.add_argument("--port", type=int, default=None, help="server bind port")
    ap.add_argument(
        "--no-browser",
        action="store_true",
        help="serve without opening a browser tab",
    )
    args = ap.parse_args()

    if args.hermes_home:
        profile_dir = Path(args.hermes_home)
    else:
        profile_dir = Path.home() / ".hermes" / "profiles" / args.profile

    if not profile_dir.exists():
        sys.stderr.write(f"profile dir not found: {profile_dir}\n")
        return 2

    if args.json:
        metrics = collect_metrics(args.profile, profile_dir, {})
        payload = build_status_payload(metrics)
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if args.serve or not args.once and not args.terminal:
        port = args.port or (8421 + (sum(ord(ch) for ch in args.profile) % 200))
        if not args.no_browser:
            sys.stdout.write(
                f"Hermes Context Cockpit: {build_visor_url(args.profile, host=args.host, port=port)}\n"
            )
            sys.stdout.flush()
        return serve_context_cockpit(
            profile=args.profile,
            profile_dir=profile_dir,
            host=args.host,
            port=port,
            open_browser_on_start=not args.no_browser,
        )

    try:
        from rich.console import Console
    except Exception:
        sys.stderr.write(
            "context_visor needs the 'rich' package (Hermes venv has it).\n"
        )
        return 2

    if args.once:
        console = Console()
        metrics = collect_metrics(args.profile, profile_dir, {})
        status = classify_status(metrics)
        console.print(build_cockpit(metrics, status))
        return 0

    console = Console()
    stop = {"flag": False}
    visor_state: dict = {}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not stop["flag"]:
        console.clear()
        metrics = collect_metrics(args.profile, profile_dir, visor_state)
        status = classify_status(metrics)
        console.print(build_cockpit(metrics, status))
        for _ in range(int(REFRESH_SEC * 10)):
            if stop["flag"]:
                break
            time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
