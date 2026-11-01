"""`/apt-update` command — run the system package update + list upgradable.

Wraps /opt/scripts/check-updates.sh (the daily cron) so the same logic is
available on demand from chat. Sync, ~30-60s, run via asyncio.to_thread
in the handler. Output is plain text, monospace-safe.
"""
from __future__ import annotations

import os
import subprocess

_TIMEOUT = 120
_SCRIPT = "/opt/scripts/check-updates.sh"


def _run_script() -> tuple[int, str, str]:
    proc = subprocess.run(
        ["bash", _SCRIPT],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _read_available() -> tuple[str, int]:
    log = "/var/log/available-updates.log"
    if not os.path.exists(log):
        return "", 0
    try:
        with open(log) as fh:
            body = fh.read().strip()
    except OSError:
        return "", 0
    return body, len([ln for ln in body.splitlines() if ln.strip()])


def format_apt_update() -> str:
    rc, out, err = _run_script()
    body, count = _read_available()
    if rc != 0:
        return f"apt-update: falló (rc={rc}). {err or out}"
    if count == 0:
        return "apt-update: sistema al día. 0 paquetes para actualizar."
    head = "\n".join(body.splitlines()[:10])
    more = f"\n… (+{count - 10} más)" if count > 10 else ""
    return f"apt-update: {count} paquetes disponibles.\n{head}{more}"


if __name__ == "__main__":
    # ponytail: self-check
    print(format_apt_update())
