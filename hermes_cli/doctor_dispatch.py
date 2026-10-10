"""Dispatch-plane freshness check for ``hermes doctor`` (kanban t_50d090c0).

Dispatch modules are imported *inside* the gateway process, so an edit to one is invisible to the live
board until a restart: work gets shipped, reviewed and verified while production runs the old code.
``gateway/code_skew.py`` cannot catch it (an uncommitted working-tree edit never moves the checkout
sha), and the service checks here look at liveness, not code freshness.

This check reads every reachable gateway runtime record (``gateway_state.json``), keeps the ones that
belong to a LIVE gateway, and compares the boot snapshot the record carries against the dispatch
modules on disk *now*. It warns — loudly, and with the restart command — when the file is newer than
what the gateway loaded.

Deliberately not a ``--fix``able finding: doctor must never restart the gateway. The automatic path is
the gateway's own dispatcher (``gateway/dispatch_freshness.py``); this is the surface for a human who
is not watching ``agent.log``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check, warn_on_error

_MAX_NAMED = 3  # files named inline before the list is truncated


def _runtime_candidates() -> list[tuple[str, Path]]:
    """``(label, gateway_state.json path)`` for this home and the host root, deduplicated.

    Multiplex-only: the ONE host gateway serves every profile and a served profile writes no identity
    files of its own (``gateway/status.py::multiplexer_liveness_for_profile``), so the host root's
    record is what a profile's doctor run has to judge.
    """
    from hermes_cli.doctor import HERMES_HOME
    from hermes_constants import get_default_hermes_root

    home = Path(HERMES_HOME)
    try:
        host_root = Path(get_default_hermes_root())
    except Exception:
        host_root = home
    candidates = [(f"{home}", home / "gateway_state.json")]
    if host_root.resolve() != home.resolve():
        candidates.append((f"{host_root} (host gateway)", host_root / "gateway_state.json"))
    seen: set = set()
    unique: list[tuple[str, Path]] = []
    for label, path in candidates:
        key = str(path.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            unique.append((label, path))
    return unique


def _source_note(verdict) -> str:
    """How the boot instant was established — the two paths differ in precision, and say so."""
    if verdict.source == "stamp":
        return "against the boot snapshot the gateway stamped at startup"
    return ("derived from the gateway process start time — this record predates the boot snapshot, so "
            "code touched during startup also reads as stale")


def _stale_issue(pid: Any, profile_note: str, verdict) -> str:
    from gateway.dispatch_freshness import reload_opt_out_hint, restart_command

    named = ", ".join(verdict.stale_files[:6])
    return (f"Stale kanban dispatch plane: the live gateway (PID {pid}{profile_note}) is running dispatch "
            f"code older than the files on disk ({named}) — run `{restart_command()}` to load it. "
            f"The gateway also self-reloads through its dispatcher when idle; to keep only this warning, "
            f"set {reload_opt_out_hint()}.")


@doctor_check("Dispatch-plane freshness check could not run ({e})")
def _check_dispatch_freshness(should_fix: bool, f: Finding) -> None:
    """Warn when a live gateway is serving dispatch code older than the module on disk."""
    from gateway.dispatch_freshness import judge_record, restart_command
    from gateway.status import (
        derive_gateway_drainable,
        profile_name_for_home,
        read_runtime_status,
        runtime_status_pid_is_live,
    )

    live = 0
    candidates = _runtime_candidates()
    for _label, path in candidates:
        with warn_on_error(""):
            record: Optional[dict] = read_runtime_status(path)
        if not isinstance(record, dict) or not runtime_status_pid_is_live(record):
            continue  # no gateway, or a retained record whose PID is gone
        if not derive_gateway_drainable(gateway_running=True, gateway_state=record.get("gateway_state")):
            continue  # recorded as stopped/startup_failed: nothing is serving this code
        live += 1
        pid = record.get("pid")
        profile = profile_name_for_home(path.parent)
        profile_note = f" serving {profile}" if profile and profile != "default" else ""
        verdict = judge_record(record)
        if verdict.status == "stale":
            named = ", ".join(verdict.stale_files[:_MAX_NAMED])
            more = len(verdict.stale_files) - _MAX_NAMED
            check_warn(
                f"Dispatch code on disk is newer than what gateway PID {pid}{profile_note} loaded "
                f"({len(verdict.stale_files)} of {verdict.files} watched file(s): {named}"
                f"{f', +{more} more' if more > 0 else ''})",
                f"(run `{restart_command()}` to load it; {_source_note(verdict)})",
            )
            f.manual_issues.append(_stale_issue(pid, profile_note, verdict))  # doctor must never restart it
        elif verdict.status == "fresh":
            check_ok(f"Dispatch code matches gateway PID {pid}{profile_note}'s boot snapshot "
                     f"({verdict.files} watched file(s))")
        else:
            check_info(f"Dispatch-plane freshness not judged for gateway PID {pid}{profile_note}: "
                       f"{verdict.reason}")
    if live == 0:
        check_info("No live gateway to judge dispatch-plane freshness against "
                   f"({' or '.join(str(p) for _l, p in candidates)})")
