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
    return f"{kind} run{pid_part}{elapsed_part}"


def read_active_work(home: Optional[Path] = None) -> Optional[list]:
    """``active_work`` as the gateway last published it, or None (old gateway / not draining / unreadable)."""
    try:
        from gateway.status import read_runtime_status

        record = read_runtime_status(home / "gateway_state.json" if home else None) or {}
        work = record.get("active_work")
        return list(work) if isinstance(work, list) else None
    except Exception:
        return None


def format_drain_report(work: Optional[list], *, remaining_s: float, home: Optional[Path] = None) -> str:
    """Multi-line progress block: what the gateway is waiting on plus how to stop waiting."""
    lines = [f"  ⏳ still draining — {int(max(remaining_s, 0))}s left before the forced restart"]
    if work is None:
        lines.append("     (gateway did not report what it is waiting on — pre-update gateway or unreadable state file)")
    elif not work:
        lines.append("     (no active work reported; the gateway should exit momentarily)")
    else:
        lines.append(f"     waiting on {len(work)} active work unit(s):")
        lines.extend(f"       • {describe_active_work_unit(u, home)}" for u in work)
        lines.append("     finish or kill the work above to release the drain now; "
                     "agent.restart_after_turn_timeout in config.yaml caps this wait")
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
