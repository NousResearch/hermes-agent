"""``hermes_cli/_scan_venv_blockers.py`` — Standalone venv-process scan for JSON consumption.

Invoked by the Desktop Electron app::

    venv\\Scripts\\python.exe -m hermes_cli._scan_venv_blockers

Exits 0 for valid clear or blocked results.  Non-zero exit signals probe
failure (the detector itself crashed, psutil unavailable, etc.).  Exactly
one JSON document on stdout; diagnostics on stderr only.
"""

from __future__ import annotations

import json
import re
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


def _probe_fail_json() -> str:
    """Return the standard probe-failure JSON document."""
    return json.dumps({"ok": False, "blocked": False, "processes": []})


def _emit_probe_fail(diagnostic: str) -> NoReturn:
    """Print one JSON to stdout, diagnostic to stderr, exit non-zero."""
    print(_probe_fail_json())
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
    return cmdline


def _is_pausable_gateway(cmdline: str) -> bool:
    """Return True when *cmdline* is a gateway process the updater can pause.

    A running gateway shows up in the venv-holder scan as one or both halves
    of its launcher/worker chain (``venv\\Scripts\\python.exe -m
    hermes_cli.main gateway run`` and the uv-side interpreter re-running the
    same argv). Reporting those as blockers dead-ends the Desktop update:
    the preflight aborts with ``venv-blocked`` *before* spawning
    ``hermes-setup``, so the CLI updater's own
    ``_pause_windows_gateways_for_update()`` — which exists precisely to
    stop these processes (and is always active: ``hermes-setup`` invokes
    ``hermes update --yes --gateway``) — never gets the chance to run.

    Only gateway invocations are exempted. Anything else running from the
    venv (an operator's REPL, a stray script, a ``serve`` backend that
    survived the desktop's own teardown) has no pause machinery downstream
    and must keep blocking the handoff.

    Delegates to ``gateway.status.looks_like_gateway_command_line`` — the
    canonical ``gateway run`` matcher (profile-selector aware, shlex
    tokenization, ``run``-only) — so this exemption, the pause discovery,
    and the updater's guard fallback all share one parser. A hand-rolled
    token scan here regressed ``--profile gateway gateway run``: the profile
    *value* shadowed the subcommand token. An import failure counts as
    not-pausable — the scan then reports the process as a blocker, which is
    exactly the pre-exemption behavior.
    """
    try:
        from gateway.status import looks_like_gateway_command_line  # noqa: PLC0415
    except Exception:
        return False
    return looks_like_gateway_command_line(cmdline)


def _is_non_hermes_venv_holder(cmdline: str) -> bool:
    """Return True when *cmdline* runs a non-Hermes module from the venv.

    A process that runs ``python -m <stdlib-or-user-module>`` from the venv
    interpreter (e.g. an operator's ``python -m http.server 8077`` or a
    one-off ``-m my_script``) holds the interpreter but does NOT import
    Hermes' native ``.pyd`` extensions — the actual reason the update
    preflight blocks venv holders (a dependency sync mid-update dies with
    access-denied when a ``.pyd`` is mapped). Treating every venv-python
    process as a blocker gives a misleading "Close other Hermes windows"
    error for a process that has nothing to do with Hermes (#77422).

    Only recognizably-Hermes invocations count as real blockers:
    ``-m hermes_cli...``, ``serve``/``dashboard``/``gateway run``, and bare
    interpreters (a bare REPL can import Hermes' native extensions at any
    time, so it stays conservative and keeps blocking).
    """
    low = cmdline.lower()
    # Explicit Hermes invocation markers → real blocker. Check the module /
    # command tokens, not the filesystem path (the venv lives under
    # .../hermes-agent/, which contains "hermes" but is not a Hermes process).
    if "hermes_cli" in low or " -m hermes" in low or "hermes.main" in low:
        return False
    if re.search(r"(^|[\s\"])hermes(\s|$)", low):
        return False
    # A bare interpreter (no -m module) is a REPL — keep blocking
    # (conservative: it can import .pyd at any time).
    if " -m " not in low:
        return False
    # `python -m <module>` where the module is not Hermes (stdlib or user
    # module like http.server) → non-Hermes holder, not a blocker.
    return True


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

    processes = [
        {
            "pid": pid,
            "name": name,
            "cmdline": _redact_sensitive_cmdline(cmdline),
        }
        for pid, name, cmdline in matches
        if not _is_pausable_gateway(cmdline)
        and not _is_non_hermes_venv_holder(cmdline)
    ]
    exempted = sum(1 for _pid, _name, cmdline in matches if _is_pausable_gateway(cmdline))
    non_hermes = sum(
        1 for _pid, _name, cmdline in matches
        if not _is_pausable_gateway(cmdline) and _is_non_hermes_venv_holder(cmdline)
    )
    data = {
        "ok": True,
        "blocked": bool(processes),
        "processes": processes,
        # Diagnostic only: gateway processes present but not counted as
        # blockers because the downstream updater pauses them itself.
        "pausable_gateways": exempted,
        # Diagnostic only: venv-python processes that are not Hermes (stdlib
        # module, REPL, user script) — they hold the interpreter but not the
        # native .pyd extensions, so they do not block the update (#77422).
        "non_hermes_holders": non_hermes,
    }
    print(json.dumps(data))
    sys.exit(0)


if __name__ == "__main__":
    main()