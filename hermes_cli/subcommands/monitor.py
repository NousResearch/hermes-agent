"""``hermes monitor`` — machine-wide, read-only Hermes activity dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from hermes_cli.monitor_processes import MonitorSampler, MonitorSnapshot, ProcessKey


_SORT_KEYS = ("cpu", "rss", "elapsed", "profile", "pid")
_HELPER_KINDS = frozenset({"helper", "mcp-helper"})


@dataclass
class _ViewState:
    sort: str = "cpu"
    show_helpers: bool = False
    selected_key: ProcessKey | None = None
    details: bool = False


def _selectable_rows(state: _ViewState, snapshot: MonitorSnapshot):
    rows = _sorted_processes(snapshot, state.sort)
    if not state.show_helpers:
        rows = [row for row in rows if row.kind not in _HELPER_KINDS]
    return rows


def _apply_key(state: _ViewState, key: str, snapshot: MonitorSnapshot) -> bool:
    if key in {"q", "ctrl-c"}:
        return False
    if key == "s":
        state.sort = _SORT_KEYS[(_SORT_KEYS.index(state.sort) + 1) % len(_SORT_KEYS)]
    elif key == "a":
        state.show_helpers = not state.show_helpers
    elif key in {"up", "down", "k", "j"}:
        rows = _selectable_rows(state, snapshot)
        if not rows:
            state.selected_key = None
            state.details = False
            return True
        keys = [ProcessKey(row.pid, row.create_time) for row in rows]
        if state.selected_key not in keys:
            state.selected_key = keys[-1] if key in {"up", "k"} else keys[0]
        else:
            delta = -1 if key in {"up", "k"} else 1
            index = keys.index(state.selected_key)
            state.selected_key = keys[(index + delta) % len(keys)]
    elif key in {"enter", "right"} and state.selected_key is not None:
        state.details = not state.details
    elif key in {"escape", "left"}:
        state.details = False
    return True


class _Keyboard:
    """Small cross-platform nonblocking keyboard reader for the live dashboard."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved = None

    def __enter__(self):
        if sys.platform != "win32":
            try:
                import termios
                import tty

                self._fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except (OSError, ValueError):
                self._fd = None
        return self

    def __exit__(self, *_exc) -> None:
        if self._fd is not None and self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout: float) -> str | None:
        timeout = max(0.0, timeout)
        if sys.platform == "win32":
            import msvcrt

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char in {"\x00", "\xe0"}:
                        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch())
                    return _key_name(char)
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            return None
        if self._fd is None:
            time.sleep(timeout)
            return None
        import os
        import select

        readable, _, _ = select.select([self._fd], [], [], timeout)
        if not readable:
            return None
        data = os.read(self._fd, 1)
        if data == b"\x1b":
            while select.select([self._fd], [], [], 0.005)[0] and len(data) < 3:
                data += os.read(self._fd, 1)
        return {
            b"\x1b[A": "up", b"\x1b[B": "down", b"\x1b[C": "right", b"\x1b[D": "left",
            b"\x1b": "escape", b"\x03": "ctrl-c", b"\r": "enter", b"\n": "enter",
        }.get(data, data.decode("utf-8", errors="ignore").casefold())


def _key_name(char: str) -> str:
    return {"\r": "enter", "\n": "enter", "\x1b": "escape", "\x03": "ctrl-c"}.get(
        char, char.casefold(),
    )


def _positive_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("interval must be a number") from exc
    if not math.isfinite(interval) or interval < 0.2:
        raise argparse.ArgumentTypeError("interval must be finite and at least 0.2 seconds")
    return interval


def build_monitor_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "monitor",
        help="Monitor every local Hermes runtime",
        description=(
            "Full-screen, read-only dashboard for local Hermes runtimes, profiles, "
            "Kanban workers, gateways, and delegated agents."
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true", help="Render one dashboard snapshot and exit")
    modes.add_argument("--json", action="store_true", help="Emit one redacted JSON snapshot and exit")
    parser.add_argument("--only-profile", help="Show only one profile")
    parser.add_argument(
        "--all", action="store_true",
        help="Expand helper processes instead of summarizing them",
    )
    parser.add_argument(
        "--sort", choices=("cpu", "rss", "elapsed", "profile", "pid"), default="cpu",
        help="Runtime sort key (default: cpu)",
    )
    parser.add_argument(
        "--interval", type=_positive_interval, default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    parser.set_defaults(func=cmd_monitor)


def _duration(seconds: Any) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        return "?"
    minutes, sec = divmod(value, 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _event_age(value: Any, observed_at: float) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            timestamp = parsed.timestamp()
        except (TypeError, ValueError, OverflowError):
            return "?"
    if not math.isfinite(timestamp):
        return "?"
    return _duration(max(0.0, observed_at - timestamp))


def _bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "?"


def _filtered(snapshot: MonitorSnapshot, profile: str | None) -> MonitorSnapshot:
    if not profile:
        return snapshot
    processes = [row for row in snapshot.processes if row.profile == profile]
    delegations = [row for row in snapshot.delegations if row.get("profile", "default") == profile]
    return MonitorSnapshot(snapshot.observed_at, processes, delegations)


def _sorted_processes(snapshot: MonitorSnapshot, sort: str):
    rows = list(snapshot.processes)
    if sort == "cpu":
        return sorted(rows, key=lambda row: (-row.cpu_percent, -row.rss, row.pid))
    if sort == "rss":
        return sorted(rows, key=lambda row: (-row.rss, -row.cpu_percent, row.pid))
    if sort == "elapsed":
        return sorted(rows, key=lambda row: (-row.elapsed, row.pid))
    if sort == "profile":
        return sorted(rows, key=lambda row: (row.profile, row.kind, row.pid))
    return sorted(rows, key=lambda row: row.pid)


def _kind_style(kind: str) -> str:
    return {
        "kanban": "bold magenta",
        "gateway": "bold cyan",
        "backend": "cyan",
        "monitor": "bold green",
        "interactive": "bold yellow",
        "helper": "dim",
        "mcp-helper": "dim",
    }.get(kind, "white")


def _render_screen(
    snapshot: MonitorSnapshot, *, sort: str = "cpu", width: int = 140,
    show_helpers: bool = False, selected_key: ProcessKey | None = None,
    show_details: bool = False,
):
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    title = Text(" HERMES MONITOR ", style="bold black on bright_cyan")
    totals = Text()
    helper_count = sum(row.kind in _HELPER_KINDS for row in snapshot.processes)
    visible_count = snapshot.total_processes if show_helpers else snapshot.total_processes - helper_count
    totals.append(f"{visible_count}", style="bold bright_white")
    totals.append(" runtimes  ", style="dim")
    if helper_count and not show_helpers:
        totals.append(f"+{helper_count}", style="bold white")
        totals.append(" helpers  ", style="dim")
    totals.append(f"{snapshot.total_profiles}", style="bold bright_white")
    totals.append(" profiles  ", style="dim")
    totals.append(f"{snapshot.total_delegations}", style="bold bright_white")
    totals.append(" delegates  ", style="dim")
    totals.append(f"{snapshot.total_cpu:.1f}%", style="bold bright_green")
    totals.append(" CPU  ", style="dim")
    totals.append(_bytes(snapshot.total_rss), style="bold bright_blue")
    totals.append(" RSS", style="dim")
    header.add_row(title, totals)

    runtimes = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    runtimes.add_column("PID", justify="right", no_wrap=True, style="dim")
    runtimes.add_column("TYPE", no_wrap=True)
    runtimes.add_column("PROFILE", no_wrap=True)
    runtimes.add_column("CPU", justify="right", no_wrap=True)
    runtimes.add_column("RSS", justify="right", no_wrap=True)
    runtimes.add_column("UP", justify="right", no_wrap=True)
    if width >= 105:
        runtimes.add_column("TTY", no_wrap=True, style="dim")
        runtimes.add_column("TMUX", no_wrap=True, style="dim")
    runtimes.add_column("ACTIVITY", ratio=2, overflow="ellipsis", no_wrap=True)
    runtime_rows = _sorted_processes(snapshot, sort)
    if not show_helpers:
        runtime_rows = [row for row in runtime_rows if row.kind not in _HELPER_KINDS]
    for row in runtime_rows:
        cells = [
            Text(str(row.pid)),
            Text(row.kind, style=_kind_style(row.kind)),
            Text(row.profile),
            Text(f"{row.cpu_percent:5.1f}%", style="bright_green" if row.cpu_percent else "dim"),
            Text(_bytes(row.rss), style="bright_blue"),
            Text(_duration(row.elapsed)),
        ]
        if width >= 105:
            cells.extend((Text(row.tty), Text(row.tmux)))
        cells.append(Text(row.activity, no_wrap=True, overflow="ellipsis"))
        row_key = ProcessKey(row.pid, row.create_time)
        runtimes.add_row(*cells, style="reverse" if row_key == selected_key else None)
    if not runtime_rows:
        runtimes.add_row("—", "—", "—", "—", "—", "—", *(["—", "—"] if width >= 105 else []),
                         Text("No local Hermes runtimes detected", style="dim"))

    delegates = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    delegates.add_column("OWNER", no_wrap=True)
    delegates.add_column("AGENT", no_wrap=True)
    delegates.add_column("STATE", no_wrap=True)
    delegates.add_column("AGE", justify="right", no_wrap=True)
    delegates.add_column("ACTIVITY", ratio=2, overflow="ellipsis", no_wrap=True)
    for child in sorted(
        snapshot.delegations,
        key=lambda row: (str(row.get("profile") or ""), str(row.get("delegation_id") or ""), int(row.get("task_index") or 0)),
    ):
        owner = child.get("owner_pid") or "reported"
        agent = child.get("subagent_id") or f"{child.get('delegation_id', '?')}#{child.get('task_index', '?')}"
        state = str(child.get("status") or "?")
        cells = [
            Text(f"{child.get('profile', 'default')}:{owner}"),
            Text(str(agent)),
            Text(state, style="bold green" if state.startswith("running") else "yellow"),
            Text(_event_age(child.get("updated_at"), snapshot.observed_at)),
        ]
        cells.append(Text("working", no_wrap=True, overflow="ellipsis"))
        delegates.add_row(*cells)
    if not snapshot.delegations:
        delegates.add_row("—", "—", "—", "—", Text("No active delegated agents", style="dim"))

    sections = [
        Panel(header, border_style="bright_cyan", padding=(0, 1)),
        Panel(runtimes, title="[bold]RUNTIMES[/bold]", border_style="blue"),
        Panel(delegates, title="[bold]DELEGATED AGENTS[/bold]", border_style="magenta"),
    ]
    selected = next((
        row for row in snapshot.processes
        if ProcessKey(row.pid, row.create_time) == selected_key
    ), None)
    if show_details and selected is not None:
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold cyan", no_wrap=True)
        details.add_column(ratio=1)
        details.add_row("IDENTITY", Text(f"PID {selected.pid} · started {selected.create_time:.3f}"))
        details.add_row("RUNTIME", Text(
            f"{selected.kind} · profile {selected.profile} · {selected.status}"
        ))
        details.add_row("LOAD", Text(
            f"{selected.cpu_percent:.1f}% CPU · {_bytes(selected.rss)} RSS · "
            f"up {_duration(selected.elapsed)}"
        ))
        details.add_row("TERMINAL", Text(f"{selected.tty} · tmux {selected.tmux}"))
        details.add_row("ACTIVITY", Text(selected.activity))
        details.add_row("EVIDENCE", Text(selected.confidence))
        sections.append(Panel(details, title="[bold]RUNTIME DETAILS[/bold]", border_style="bright_green"))
    footer = Text(
        f"read-only  •  ↑/↓ select  •  Enter details  •  s sort:{sort}  •  "
        "a helpers  •  q quit",
        style="dim",
    )
    sections.append(footer)
    return Group(*sections)


def _json_payload(snapshot: MonitorSnapshot) -> dict[str, Any]:
    delegation_keys = (
        "owner_pid", "subagent_id", "delegation_id", "task_index", "status",
        "profile", "updated_at", "confidence",
    )
    return {
        "schema_version": 1,
        "observed_at": snapshot.observed_at,
        "totals": {
            "processes": snapshot.total_processes,
            "delegations": snapshot.total_delegations,
            "profiles": snapshot.total_profiles,
        },
        "processes": [asdict(row) for row in snapshot.processes],
        "delegations": [{key: row.get(key) for key in delegation_keys} for row in snapshot.delegations],
    }


def cmd_monitor(args) -> int:
    sampler = MonitorSampler()
    snapshot = sampler.sample()
    one_shot = args.json or args.once or not sys.stdout.isatty()
    if one_shot:
        time.sleep(min(args.interval, 0.2))
        snapshot = sampler.sample()
    snapshot = _filtered(snapshot, args.only_profile)
    if args.json:
        print(json.dumps(_json_payload(snapshot), ensure_ascii=False, sort_keys=True))
        return 0

    from rich.console import Console

    console = Console()
    if args.once or not sys.stdout.isatty():
        console.print(_render_screen(
            snapshot, sort=args.sort, width=console.size.width, show_helpers=args.all,
        ))
        return 0

    from rich.live import Live

    state = _ViewState(sort=args.sort, show_helpers=args.all)
    try:
        with Live(
            _render_screen(
                snapshot, sort=state.sort, width=console.size.width,
                show_helpers=state.show_helpers, selected_key=state.selected_key,
                show_details=state.details,
            ),
            console=console,
            screen=True,
            auto_refresh=False,
        ) as live, _Keyboard() as keyboard:
            next_sample = time.monotonic()
            while True:
                key = keyboard.read(max(0.0, next_sample - time.monotonic()))
                if key is not None and not _apply_key(state, key, snapshot):
                    return 0
                if time.monotonic() >= next_sample:
                    snapshot = _filtered(sampler.sample(), args.only_profile)
                    next_sample = time.monotonic() + args.interval
                live.update(
                    _render_screen(
                        snapshot, sort=state.sort, width=console.size.width,
                        show_helpers=state.show_helpers, selected_key=state.selected_key,
                        show_details=state.details,
                    ),
                    refresh=True,
                )
    except KeyboardInterrupt:
        return 130
