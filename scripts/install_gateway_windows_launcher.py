"""Elevated Hermes gateway install launcher (called from install-gateway-windows.ps1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hermes repo root on sys.path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from hermes_cli import gateway_windows

    if gateway_windows._is_running_as_admin():
        gateway_windows.install(
            force=args.force,
            start_now=True,
            start_on_login=True,
            elevated_handoff=True,
        )
        return 0

    ok = gateway_windows._launch_elevated_install(
        force=args.force,
        start_now=True,
        start_on_login=True,
    )
    if not ok:
        print(
            "Failed to launch elevated install (UAC cancelled or blocked).",
            file=sys.stderr,
        )
        return 1
    print("Elevated install launched — approve the UAC prompt to finish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
