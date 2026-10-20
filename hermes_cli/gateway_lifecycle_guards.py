"""Refusals that keep ``hermes gateway`` lifecycle verbs off gateways they must not take down.

Two rules, one per failure that took a production gateway down:

* **Foreign service definition.** A systemd unit / launchd plist pins the ``HERMES_HOME`` its gateway
  runs on. A lifecycle verb (start/stop/restart/uninstall) on a definition pinning ANOTHER home acts on
  another install's gateway: a ``HOME=<scratch>`` probe resolved the bare ``hermes-gateway`` name and ran
  ``systemctl --user stop`` on the account's real unit. Install/refresh already refused to WRITE such a
  definition (``gateway_service_owner``); the same module now refuses to OPERATE one:
  ``refuse_foreign_systemd_unit`` (salvaged from #141908, checked before any home sync) and
  ``refuse_foreign_launchd_plist``.
* **Live sessions.** ``gateway stop``/``restart`` refuse while the gateway is running turns, cron jobs or
  background delegations, naming each, unless the caller passes ``--interrupt``. The in-band restart
  drains turns, but its stop still interrupts every async delegation (``interrupt_all``), and a stop
  ends the sessions outright.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

INTERRUPT_FLAG = "--interrupt"


def _live_work_from_state(home: Path) -> Optional[dict[str, Any]]:
    """``active_agents`` from a fresh ``gateway_state.json`` whose writer is alive (older gateways)."""
    from gateway.status import parse_active_agents, read_runtime_status, runtime_status_is_stale, runtime_status_pid_is_live

    record = read_runtime_status(home / "gateway_state.json")
    if not record or runtime_status_is_stale(record) or not runtime_status_pid_is_live(record):
        return None
    work = record.get("active_work")
    return {"active_agents": parse_active_agents(record.get("active_agents")),
            "active_work": work if isinstance(work, list) else []}


def live_work(home: Path) -> Optional[dict[str, Any]]:
    """What the gateway serving ``home`` is running now: ``{"active_agents", "active_work"}``, or None
    when no gateway answers. The control socket answers live; the state file covers older gateways."""
    from gateway.control_socket import query_gateway_control

    answer = query_gateway_control(home, "active-work")
    if isinstance(answer, dict) and isinstance(answer.get("active_work"), list):
        return answer
    return _live_work_from_state(home)


def _candidate_homes() -> list[Path]:
    """Homes whose gateway a stop/restart from here takes down, in order: this home's own gateway (a
    ``--force`` standalone one included), then the live host owner that serves it."""
    from hermes_cli import gateway as gw

    homes = [gw.get_hermes_home()]
    owner = gw._host_multiplexer_for_all_verb()
    if owner is not None and Path(owner.home) not in homes:
        homes.append(Path(owner.home))
    return homes


def refusal_lines(verb: str, homes: list[Path], *, flag_hint: str = f"hermes gateway {{verb}} {INTERRUPT_FLAG}") -> list[str]:
    """Refusal text for ``verb`` while the first answering gateway among ``homes`` has live work, else []."""
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit

    for home in homes:
        work = live_work(home)
        if work is not None:
            break
    else:
        return []
    units = [u for u in work.get("active_work") or [] if isinstance(u, dict)]
    count = max(len(units), int(work.get("active_agents") or 0))
    if not count:
        return []
    lines = [f"Refusing to {verb} the gateway: {count} live session(s) would be interrupted."]
    lines += [f"  • {describe_active_work_unit(u, home)}" for u in units]
    if not units:
        lines.append(f"  • {count} active agent run(s) (this gateway does not name them)")
    lines += ["  Let them finish, or interrupt them deliberately:",
              f"    {flag_hint.format(verb=verb)}   # interrupts {count} live session(s)"]
    return lines


def refuse_while_sessions_live(verb: str, args) -> None:
    """Exit 1 naming the live work unless ``--interrupt`` was passed; with it, say what is interrupted."""
    lines = refusal_lines(verb, _candidate_homes())
    if not lines:
        return
    from hermes_cli.gateway import print_error, print_warning

    if getattr(args, "interrupt", False):
        print_warning(lines[0].replace(f"Refusing to {verb} the gateway", f"--interrupt: {verb} proceeds"))
        for line in lines[1:-2]:
            print(line)
        return
    print_error(lines[0])
    for line in lines[1:]:
        print(line)
    sys.exit(1)
