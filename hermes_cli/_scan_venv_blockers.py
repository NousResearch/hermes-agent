"""``hermes_cli/_scan_venv_blockers.py`` — Standalone venv-process scan for JSON consumption.

Invoked by the Desktop Electron app::

    venv\\Scripts\\python.exe -m hermes_cli._scan_venv_blockers

Exits 0 for valid clear or blocked results.  Non-zero exit signals probe
failure (the detector itself crashed, psutil unavailable, etc.).  Exactly
one JSON document on stdout; diagnostics on stderr only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import PureWindowsPath
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


def _probe_fail_json(diagnostic: str = "probe failed") -> str:
    """Return the standard probe-failure JSON document.

    ``ok: false`` plus ``probe_failed: true`` means the detector itself could
    not run — this is *not* a clear scan. Callers must treat
    ``ok is not True`` / non-zero exit as probe failure, never as
    ``blocked: false`` "clear" (#83149).
    """
    return json.dumps(
        {
            "ok": False,
            "probe_failed": True,
            "blocked": False,
            "processes": [],
            "error": diagnostic,
        }
    )


def _emit_probe_fail(diagnostic: str) -> NoReturn:
    """Print one JSON to stdout, diagnostic to stderr, exit non-zero."""
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


def _classify_local_preview_args(args: object) -> dict[str, object]:
    """Return safe UI metadata for an exact ``python -m http.server`` argv.

    The general holder detector intentionally truncates its diagnostic command
    line. Reading argv separately preserves a useful directory label without
    exposing an unbounded command line to the renderer.
    """
    if not isinstance(args, (list, tuple)) or not all(isinstance(arg, str) for arg in args):
        return {}

    # For a real interpreter module invocation, ``-m`` is the first argument
    # after the executable. A later ``-m http.server`` can merely be data passed
    # to an unrelated script and must never authorize termination.
    if len(args) < 3 or args[1] != "-m" or args[2].lower() != "http.server":
        return {}
    module_index = 1

    port = 8000
    if module_index + 2 < len(args):
        candidate = args[module_index + 2]
        if candidate.isdigit() and 0 < int(candidate) <= 65535:
            port = int(candidate)

    label = ""
    try:
        directory_index = args.index("--directory")
        if directory_index + 1 < len(args):
            label = PureWindowsPath(args[directory_index + 1]).name
    except ValueError:
        pass

    metadata: dict[str, object] = {
        "kind": "local-preview",
        "safeToStop": True,
        "port": port,
    }
    if label:
        metadata["label"] = label
    return metadata


def _local_preview_metadata(pid: int, name: str) -> dict[str, object]:
    if name.lower() not in {"python.exe", "pythonw.exe", "python", "pythonw"}:
        return {}
    try:
        import psutil  # noqa: PLC0415

        process = psutil.Process(pid)
        metadata = _classify_local_preview_args(process.cmdline())
        if metadata:
            metadata["createTime"] = process.create_time()
        return metadata
    except Exception:
        return {}


def _terminate_safe_preview(
    pid: int,
    expected_create_time: float,
    *,
    psutil_module: object | None = None,
) -> tuple[bool, str | None]:
    """Terminate one verified local preview process tree.

    A fresh ``psutil.Process`` identity check and exact argv classification occur
    immediately before termination. psutil guards mutating Process methods
    against PID reuse, avoiding taskkill's stale-PID race.
    """
    try:
        if psutil_module is None:
            import psutil as psutil_module  # type: ignore[no-redef]  # noqa: PLC0415

        process = psutil_module.Process(pid)  # type: ignore[attr-defined]
        if abs(process.create_time() - expected_create_time) > 0.001:
            return False, "process identity changed"
        if not _classify_local_preview_args(process.cmdline()):
            return False, "process is no longer a local preview"

        children = process.children(recursive=True)
        targets = [*reversed(children), process]
        for target in targets:
            target.terminate()
        _gone, alive = psutil_module.wait_procs(targets, timeout=3)  # type: ignore[attr-defined]
        for target in alive:
            target.kill()
        if alive:
            psutil_module.wait_procs(alive, timeout=2)  # type: ignore[attr-defined]
        return True, None
    except Exception as exc:
        return False, f"termination failed: {type(exc).__name__}"


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


def _is_updater_owned_backend(pid: int, cmdline: str) -> bool:
    """Return True when *pid* is a Hermes backend the CLI updater can stop.

    The gateway exemption above keeps ``gateway run`` holders out of the
    blocker list because the updater's own pause machinery stops and resumes
    them. ``hermes serve`` / ``hermes dashboard`` backends had no such
    deferral, so a leaked serve child (or a Desktop-owned backend the
    teardown lost track of) dead-ended the hand-off with ``venv-blocked`` —
    or, worse, survived the hand-off and made the shim quarantine fail with
    ``os error 32`` (#98336) — even though the updater downstream owns
    exactly this case with its ledger rungs (`_ledger_reapable_backend_pids`
    reaps dead-spawner orphans; `_ledger_manual_serve_holders` stops manual
    serves and relaunches them on their recorded host/port).

    Positive identity only — never name/substring matching (#90778, and the
    #99558 identity-guard contract):

    - the argv's parsed SUBCOMMAND (token-based) is ``serve``/``dashboard``;
    - the machine spawn ledger has a live-verified ``(pid, create_time)``
      entry for the process with a matching purpose;
    - ownership is provable: the recorded spawner is dead or unrecorded
      (the updater's rungs stop/relaunch those), or the spawner is an
      ancestor of THIS scan — i.e. the Desktop app performing the hand-off,
      which exits before the updater runs, turning the backend into exactly
      the dead-spawner orphan the ledger rung reaps.

    A backend whose recorded spawner is alive and is NOT this hand-off's
    Desktop (a second Desktop window, another supervisor) keeps blocking:
    that supervisor would respawn whatever the updater kills. Anything
    unprovable → not exempt (fail closed, pre-exemption behavior).
    """
    return _updater_owned_backend_entry(pid, cmdline) is not None


def _updater_owned_backend_entry(pid: int, cmdline: str) -> dict | None:
    """Ledger entry for a deferred backend, or ``None`` when it must block.

    Same decision logic as ``_is_updater_owned_backend`` (which delegates
    here); returning the matched ledger entry lets ``main()`` emit sanitized
    decision evidence — structured identity fields only, never argv, which
    can carry tokens or private endpoints (#98350).
    """
    try:
        from hermes_cli.update_cmd import _hermes_holder_subcommand  # noqa: PLC0415

        purpose = _hermes_holder_subcommand(cmdline)
    except Exception:
        return None
    if purpose not in ("serve", "dashboard"):
        return None
    try:
        from hermes_cli.process_identity import (  # noqa: PLC0415
            ledger_entries,
            spawner_is_dead,
        )

        entries = ledger_entries()
    except Exception:
        return None
    for entry in entries:
        if entry.get("pid") != pid:
            continue
        if entry.get("purpose") not in ("serve", "dashboard"):
            return None
        dead = spawner_is_dead(entry)
        if dead is not False:
            # Spawner dead, unrecorded, or unprovable-but-registered: the
            # updater's ledger rungs own this holder (reap or stop+relaunch).
            return entry
        if _spawner_is_this_handoff_desktop(entry):
            return entry
        return None
    return None


def _deferred_backend_evidence(entries: list[dict]) -> list[dict]:
    """Sanitized decision evidence for deferred serve/dashboard backends.

    Structured ledger fields only — pid, purpose, recorded port — never the
    command line, which can carry tokens or private endpoints. Lets the
    scan result explain *why* a holder disappeared from ``processes``
    without echoing argv (#98350).
    """
    evidence = []
    for entry in entries:
        pid = entry.get("pid")
        if not isinstance(pid, int):
            continue
        evidence.append(
            {"pid": pid, "purpose": entry.get("purpose"), "port": entry.get("port")}
        )
    return evidence


def _spawner_is_this_handoff_desktop(entry: dict) -> bool:
    """True when the entry's live spawner is an ancestor of this scan.

    The scan subprocess is spawned by the Desktop app's update preflight, so
    the Desktop performing the hand-off is in our ancestor chain. Identity is
    verified by ``(pid, create_time)`` — a recycled PID cannot forge the pair.
    """
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return False
    try:
        import psutil  # noqa: PLC0415

        for ancestor in psutil.Process().parents():
            if ancestor.pid != spawner_pid:
                continue
            expected = entry.get("spawner_create")
            if expected is None:
                return True
            return abs(float(ancestor.create_time()) - float(expected)) < 2.0
    except Exception:
        return False
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
    exempted_gateways = 0
    deferred_entries: list[dict] = []
    for pid, name, cmdline in matches:
        if _is_pausable_gateway(cmdline):
            exempted_gateways += 1
            continue
        deferred_entry = _updater_owned_backend_entry(pid, cmdline)
        if deferred_entry is not None:
            # Ledger-verified serve/dashboard backend the CLI updater's own
            # rungs stop (and relaunch) downstream — reporting it here would
            # dead-end the hand-off before that machinery can run (#98336).
            deferred_entries.append(deferred_entry)
            continue
        process = {
            "pid": pid,
            "name": name,
            # Truncate for display AFTER the gateway exemption has seen the
            # full cmdline (long managed-runtime interpreter paths would
            # otherwise swallow the `gateway run` argv).
            "cmdline": _redact_sensitive_cmdline(cmdline)[:120],
        }
        process.update(_local_preview_metadata(pid, name))
        processes.append(process)

    data = {
        "ok": True,
        "blocked": bool(processes),
        "processes": processes,
        # Diagnostic only: gateway processes present but not counted as
        # blockers because the downstream updater pauses them itself.
        "pausable_gateways": exempted_gateways,
        # Diagnostic only: ledger-verified serve/dashboard backends deferred
        # to the updater's stop/relaunch rungs (#98336).
        "deferred_backends": len(deferred_entries),
        # Diagnostic only: sanitized evidence (structured ledger identity,
        # never argv) explaining which holders the deferral consumed (#98350).
        "deferred_backend_evidence": _deferred_backend_evidence(deferred_entries),
    }
    print(json.dumps(data))
    sys.exit(0)


def _terminate_safe_main(argv: list[str]) -> NoReturn:
    if len(argv) != 2:
        print(json.dumps({"ok": False, "error": "expected pid and create time"}))
        raise SystemExit(2)
    try:
        pid = int(argv[0])
        create_time = float(argv[1])
        if pid <= 0 or not math.isfinite(create_time) or create_time <= 0:
            raise ValueError
    except ValueError:
        print(json.dumps({"ok": False, "error": "invalid process identity"}))
        raise SystemExit(2)

    stopped, error = _terminate_safe_preview(pid, create_time)
    print(json.dumps({"ok": stopped, "pid": pid, "error": error}))
    raise SystemExit(0 if stopped else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--terminate-safe":
        _terminate_safe_main(sys.argv[2:])
    main()