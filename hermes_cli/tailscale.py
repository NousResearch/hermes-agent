"""`/tailscale` command — control + status of the Tailscale daemon.

Reuses the `tailscale` CLI that ships in /usr/bin. Three subcommands:
status, up, down. `up` requires an auth key (env HERMES_TS_AUTHKEY) or
already-authed state; we never persist keys here. Output is plain text,
monospace-safe for Telegram.
"""
from __future__ import annotations

import os
import subprocess
from typing import List

_TIMEOUT = 30


def _run(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["tailscale", *args],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _status_text() -> str:
    rc, out, err = _run(["status"])
    if rc != 0:
        if "tailscaled" in (err + out).lower() and "not running" in (err + out).lower():
            return "Tailscale: daemon detenido. Usá /tailscale up para levantarlo."
        return f"Tailscale: error (rc={rc}): {err or out}"
    # First non-empty line usually has the magic DN; subsequent lines are peers.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    head = lines[0] if lines else "(sin info)"
    return f"Tailscale: {head}"


def format_tailscale(arg: str) -> str:
    """Dispatch for /tailscale [status|up|down]."""
    arg = (arg or "status").strip().lower()
    if arg in ("", "status", "st"):
        return _status_text()
    if arg in ("up", "start"):
        authkey = os.environ.get("HERMES_TS_AUTHKEY", "").strip()
        args = ["up", "--accept-routes"]
        if authkey:
            args.append(f"--authkey={authkey}")
        rc, out, err = _run(args)
        if rc == 0:
            return f"Tailscale: levantado OK.\n{_status_text()}"
        return f"Tailscale: falló al levantar (rc={rc}).\n{err or out}"
    if arg in ("down", "stop"):
        rc, out, err = _run(["down"])
        if rc == 0:
            return "Tailscale: detenido."
        return f"Tailscale: falló al detener (rc={rc}).\n{err or out}"
    return "tailscale: opción inválida. Usá /tailscale (status), /tailscale up, /tailscale down."


if __name__ == "__main__":
    # ponytail: self-check, prints to stdout so `python -m hermes_cli.tailscale status` exercises the read path
    import sys
    print(format_tailscale(sys.argv[1] if len(sys.argv) > 1 else "status"))
