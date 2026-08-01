"""``hermes_cli/_scan_venv_blockers.py`` — Standalone venv-process scan for JSON consumption.

Invoked by the Desktop Electron app::

    venv\\Scripts\\python.exe -m hermes_cli._scan_venv_blockers

Exits 0 for valid clear or blocked results.  Non-zero exit signals probe
failure (the detector itself crashed, psutil unavailable, etc.).  Exactly
one JSON document on stdout; diagnostics on stderr only.
"""

from __future__ import annotations

import json
import sys
from typing import NoReturn

# Long CLI flags whose argument value must be redacted from the cmdline.
_SENSITIVE_LONG_FLAGS: list[str] = [
    "--token",
    "--api-key",
    "--password",
    "--secret",
    "--authorization",
    "--access-key",
    "--private-key",
    "--session-key",
]


def _probe_fail_json(error_msg: str = "") -> str:
    """Return standard probe-failure JSON document (fail-closed contract)."""
    res = {"ok": False, "blocked": False, "processes": []}
    if error_msg:
        res["error"] = error_msg
    return json.dumps(res)


def _emit_probe_fail(diagnostic: str) -> NoReturn:
    """Print structured failure JSON to stdout, diagnostic to stderr, exit 1."""
    print(_probe_fail_json(diagnostic))
    print(diagnostic, file=sys.stderr)
    sys.exit(1)


def _find_flag(text: str, flag: str) -> int:
    """Return the index of *flag* when it starts the string or follows a space.

    Returns -1 when not found.  This avoids matching ``--token`` inside an
    embedded token or path like ``/some--token-thing``.
    """
    low = text.lower()
    fl = flag.lower()
    pos = 0
    while True:
        idx = low.find(fl, pos)
        if idx == -1:
            return -1
        if idx == 0 or text[idx - 1] == " ":
            return idx
        pos = idx + 1


def _redact_sensitive_cmdline(cmdline: str) -> str:
    """Apply generic secret redaction then long-flag redaction.

    If the generic redactor itself fails, return ``"<redacted>"`` — the PID
    and process name still provide actionable diagnostics.
    """
    # Generic pass: the project's shared secret redactor.
    try:
        from agent.redact import redact_sensitive_text  # noqa: PLC0415

        cmdline = redact_sensitive_text(cmdline, force=True)
    except Exception:
        return "<redacted>"

    # Conservative long-flag pass: preserve the flag name, replace the value
    # and everything after it with ``<redacted>``.  Short flags (-t, -k, -p)
    # are intentionally not redacted — they are ambiguous and may be useful
    # diagnostics (toolset, port, profile).
    try:
        earliest = len(cmdline)
        for flag in _SENSITIVE_LONG_FLAGS:
            # --flag=value  →  preserve "--flag="
            idx = _find_flag(cmdline, flag + "=")
            if idx != -1 and idx + len(flag) + 1 < earliest:
                earliest = idx + len(flag) + 1
            # --flag value  →  preserve "--flag "
            idx = _find_flag(cmdline, flag + " ")
            if idx != -1 and idx + len(flag) + 1 < earliest:
                earliest = idx + len(flag) + 1

        if earliest < len(cmdline):
            return cmdline[:earliest] + "<redacted>"
    except Exception:
        return "<redacted>"
    return cmdline


def _is_pausable_gateway(cmdline: str) -> bool:
    """Return True when *cmdline* is a gateway process the updater can pause."""
    try:
        from gateway.status import looks_like_gateway_command_line  # noqa: PLC0415
    except Exception:
        return False
    try:
        return looks_like_gateway_command_line(cmdline)
    except Exception:
        return False


def main() -> None:
    """Entry point.  Prints one JSON doc to stdout.  Exits 0 for valid scan."""
    try:
        import psutil  # noqa: PLC0415, F401
    except Exception as exc:
        _emit_probe_fail(f"psutil is not available: {exc}")

    try:
        from hermes_cli.main import _detect_venv_python_processes  # noqa: PLC0415

        matches = _detect_venv_python_processes()
    except Exception as exc:
        _emit_probe_fail(f"scan aborted: {exc}")

    processes = []
    exempted = 0

    for item in matches:
        try:
            pid, name, cmdline = item
            redacted_cmd = _redact_sensitive_cmdline(cmdline)
            if _is_pausable_gateway(cmdline):
                exempted += 1
            else:
                processes.append({
                    "pid": pid,
                    "name": name,
                    "cmdline": redacted_cmd,
                })
        except Exception as proc_exc:
            print(f"ignoring process entry due to error: {proc_exc}", file=sys.stderr)
            continue

    data = {
        "ok": True,
        "blocked": bool(processes),
        "processes": processes,
        "pausable_gateways": exempted,
    }
    print(json.dumps(data))
    sys.exit(0)


if __name__ == "__main__":
    main()