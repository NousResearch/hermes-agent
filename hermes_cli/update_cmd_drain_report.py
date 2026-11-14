"""Name what a draining gateway is waiting on while ``hermes update`` blocks on it.

The gateway's in-band restart (SIGUSR1 → ``request_restart``) defers ``stop()`` until in-flight
work finishes, capped by ``agent.restart_after_turn_timeout`` (30 min by default). From the
updater's side that was a bare "draining (up to 1875s)..." followed by silence, which reads as a
hung update. The gateway publishes each unit it is holding for in ``gateway_state.json``
(``active_work``, written by ``GatewayShutdownMixin._describe_active_work``); this module turns
that into progress lines.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

# Progress cadence: the gateway refreshes ``active_work`` every 30s; printing faster only repeats it.
DRAIN_REPORT_INTERVAL_S = 30.0


def _fmt_elapsed(seconds: object) -> str:
    try:
        total = int(float(seconds))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"
    return f"{total // 60}m{total % 60:02d}s" if total >= 60 else f"{total}s"


def _cron_job_name(job_id: str, home: Optional[Path]) -> Optional[str]:
    """``name`` from the profile's ``jobs.json`` (None when unreadable — the id alone still identifies it)."""
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from cron.jobs import load_jobs

        token = set_hermes_home_override(home) if home else None
        try:
            for job in load_jobs():
                if str(job.get("id")) == job_id:
                    return str(job.get("name") or "") or None
        finally:
            if token is not None:
                reset_hermes_home_override(token)
    except Exception:
        return None
    return None


def describe_active_work_unit(unit: dict, home: Optional[Path] = None) -> str:
    """One human line for one ``active_work`` entry; unknown shapes degrade to their ``kind``."""
    kind = str(unit.get("kind") or "work")
    pid = unit.get("pid")
    pid_part = f" pid {pid}" if pid else ""
    elapsed = unit.get("elapsed_s")
    elapsed_part = f", running {_fmt_elapsed(elapsed)}" if elapsed is not None else ""
    if kind == "cron":
        job_id = str(unit.get("job_id") or "?")
        name = _cron_job_name(job_id, home)
        label = f"cron job {job_id}" + (f" ({name})" if name else "")
        where = f" in external worker{pid_part}" if unit.get("external") else f" in-process{pid_part}"
        return f"{label}{where}{elapsed_part}"
    if kind == "chat":
        session = str(unit.get("session") or "?")
        model = unit.get("model")
        tool = unit.get("current_tool")
        detail = ", ".join(p for p in (f"model {model}" if model else "", f"tool {tool}" if tool else "") if p)
        return f"chat turn {session}{pid_part}{elapsed_part}" + (f" [{detail}]" if detail else "")
    if kind == "delegation":
        goal = str(unit.get("goal") or "").strip()
        return (f"background delegation {unit.get('delegation_id') or '?'}{pid_part}{elapsed_part}"
                + (f" [{goal}]" if goal else ""))
    session = unit.get("session")
    return f"{kind} run" + (f" {session}" if session else "") + f"{pid_part}{elapsed_part}"


def read_active_work(home: Optional[Path] = None) -> Optional[list]:
    """``active_work`` as the gateway last published it, or None (old gateway / not draining / unreadable)."""
    try:
        from gateway.status import read_runtime_status

        record = read_runtime_status(home / "gateway_state.json" if home else None) or {}
        work = record.get("active_work")
        return list(work) if isinstance(work, list) else None
    except Exception:
        return None


def _tui_command(home: Optional[Path]) -> str:
    """The TUI command that opens *home*'s profile: a bare ``hermes`` opens the default one."""
    from hermes_constants import profile_name_for_home

    name = profile_name_for_home(home)
    return f"hermes -p {name} --tui" if name and name != "default" else "hermes --tui"


def format_drain_report(work: Optional[list], *, remaining_s: float, home: Optional[Path] = None) -> str:
    """Multi-line progress block: what the gateway is waiting on plus how to stop waiting."""
    lines = [f"  ⏳ still waiting for idle — {max(remaining_s, 0):.0f}s left of this wait "
             "(live work is never interrupted; past it the gateway keeps its restart pending)"]
    if work is None:
        lines.append("     (gateway did not report what it is waiting on — pre-update gateway or unreadable state file)")
    elif not work:
        lines.append("     (no active work reported; the gateway should exit momentarily)")
    else:
        lines.append(f"     waiting on {len(work)} active work unit(s):")
        lines.extend(f"       • {describe_active_work_unit(u, home)}" for u in work)
        if any(isinstance(u, dict) and u.get("kind") == "chat" for u in work):
            # The draining gateway stays reachable for this (#139041): a chat turn parked on an
            # approval/clarify is answered from a client, not waited out.
            lines.append("     a chat turn waiting on an approval or clarify question can be answered now: "
                         f"open it with `{_tui_command(home)}` (resume the session)")
        lines.append("     finish or kill the work above to restart now, or `hermes update --force` "
                     "to stop it now")
    return "\n".join(lines)


def drain_progress_reporter(home: Optional[Path] = None, *, budget_s: float,
                            interval_s: float = DRAIN_REPORT_INTERVAL_S,
                            emit: Callable[[str], None] = print) -> Callable[[], None]:
    """Return a zero-arg callback for ``_wait_for_pid_exit(on_progress=...)`` that prints the drain
    report every ``interval_s`` while the wait is in progress."""
    started = time.monotonic()
    state = {"last": started}

    def _tick() -> None:
        now = time.monotonic()
        if now - state["last"] < interval_s:
            return
        state["last"] = now
        emit(format_drain_report(read_active_work(home), remaining_s=budget_s - (now - started), home=home))

    return _tick


def restart_left_pending(pid: int, home: Optional[Path] = None) -> bool:
    """True when gateway *pid* outlived the wait still holding the restart SIGUSR1 asked for: alive,
    reachable by that signal (one that could not be sent asked nothing) and not provably wedged (a
    dead loop is not live work; its caller keeps the bounded escalation). Such a gateway restarts
    itself at its first idle moment, so nothing may stop it on the wait's account."""
    import os
    import signal
    if not hasattr(signal, "SIGUSR1") or pid <= 0:
        return False
    try:
        # Signal 0 probes, and fails exactly where SIGUSR1 could not have been delivered.
        os.kill(pid, 0)  # windows-footgun: ok — POSIX only (SIGUSR1-guarded above)
    except OSError:
        return False
    from hermes_cli.gateway import GATEWAY_LOOP_WEDGED, probe_gateway_loop_liveness
    return probe_gateway_loop_liveness(pid, home=home) != GATEWAY_LOOP_WEDGED


def leave_restart_pending(pid: int, label: str, home: Optional[Path] = None, *, supervised: bool = True) -> bool:
    """When :func:`restart_left_pending` holds, say so (:func:`report_restart_left_pending`) and return
    True: the caller must not stop, kill or force-restart *pid*."""
    if not restart_left_pending(pid, home):
        return False
    report_restart_left_pending(label, home, supervised=supervised)
    return True


def report_restart_left_pending(label: str, home: Optional[Path] = None, *, supervised: bool = True,
                                emit: Callable[[str], None] = print) -> None:
    """The one line a caller prints instead of stopping a gateway whose wait ran out."""
    work = read_active_work(home)
    what = "; ".join(describe_active_work_unit(u, home) for u in work if isinstance(u, dict)) if work else ""
    after = ("restarts itself on the new code once idle" if supervised else
             "exits once idle (start it again with `hermes gateway run` if nothing relaunches it)")
    emit(f"  ↻ {label}: still running live work{f' ({what})' if what else ''} — left running; it {after}. "
         "`hermes update --force` stops it now.")
