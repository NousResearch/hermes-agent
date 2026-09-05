"""Graceful Desktop shutdown coordination for gateway-owned work."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class DesktopDrainSnapshot:
    gateway_agents: int
    kanban_workers: int

    @property
    def idle(self) -> bool:
        return self.gateway_agents == 0 and self.kanban_workers == 0


def desktop_profile_homes() -> tuple[Path, ...]:
    """Return every local profile home whose gateway Desktop may supervise."""
    from hermes_cli.profiles import list_profiles

    return tuple(Path(profile.path) for profile in list_profiles())


def read_desktop_drain_snapshot(homes: Iterable[Path]) -> tuple[int, int]:
    """Count live gateway turns and board workers without mutating either."""
    from gateway.status import (
        parse_active_agents,
        read_runtime_status,
        runtime_status_pid_is_live,
    )
    from hermes_cli import kanban_db as kb

    gateway_agents = 0
    for home in homes:
        runtime = read_runtime_status(path=Path(home) / "gateway_state.json")
        if runtime_status_pid_is_live(runtime):
            gateway_agents += parse_active_agents((runtime or {}).get("active_agents"))

    kanban_workers = 0
    seen_paths: set[str] = set()
    for metadata in kb.list_boards(include_archived=False):
        slug = metadata.get("slug") or kb.DEFAULT_BOARD
        path = kb.kanban_db_path(board=slug).expanduser()
        try:
            identity = str(path.resolve())
        except OSError:
            identity = str(path)
        if identity in seen_paths or not path.exists():
            continue
        seen_paths.add(identity)
        conn = kb.connect(board=slug)
        try:
            rows = conn.execute(
                "SELECT worker_pid FROM tasks WHERE status = 'running' "
                "AND worker_pid IS NOT NULL"
            ).fetchall()
            kanban_workers += sum(1 for row in rows if kb._pid_alive(row[0]))
        finally:
            conn.close()

    return gateway_agents, kanban_workers


def drain_all_desktop_work() -> DesktopDrainSnapshot:
    """Production entry point used by ``gateway stop --all --drain``."""
    from gateway.drain_control import write_drain_request

    homes = desktop_profile_homes()

    def report(current: DesktopDrainSnapshot) -> None:
        if current.idle:
            print("✓ Hermes drain is idle; stopping supervised services")
        else:
            print(
                "… Draining Hermes: "
                f"{current.gateway_agents} gateway turn(s), "
                f"{current.kanban_workers} Kanban worker(s) still active"
            )

    return wait_for_desktop_drain(
        homes=homes,
        snapshot=lambda: read_desktop_drain_snapshot(homes),
        write_marker=lambda home: write_drain_request(
            home=home,
            principal="desktop-close",
            suppress_notification=True,
        ),
        on_change=report,
    )


def wait_for_desktop_drain(
    *,
    homes: Iterable[Path],
    snapshot: Callable[[], tuple[int, int]],
    write_marker: Callable[[Path], object],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    poll_interval: float = 1.0,
    refresh_interval: float = 30.0,
    idle_samples_required: int = 2,
    on_change: Callable[[DesktopDrainSnapshot], None] | None = None,
) -> DesktopDrainSnapshot:
    """Stop new work and wait without terminating in-flight workers.

    Two consecutive idle samples close the small race between writing the
    marker and each gateway watcher observing it.  There is intentionally no
    deadline: Desktop owns this helper until work finishes naturally.
    """
    profile_homes = tuple(dict.fromkeys(Path(home) for home in homes))

    def refresh_markers() -> None:
        for home in profile_homes:
            write_marker(home)

    refresh_markers()
    refreshed_at = monotonic()
    idle_samples = 0
    previous: DesktopDrainSnapshot | None = None

    while True:
        current = DesktopDrainSnapshot(*snapshot())
        if current != previous and on_change is not None:
            on_change(current)
        previous = current

        if current.idle:
            idle_samples += 1
            if idle_samples >= max(1, idle_samples_required):
                return current
        else:
            idle_samples = 0

        if monotonic() - refreshed_at >= refresh_interval:
            refresh_markers()
            refreshed_at = monotonic()

        sleep(poll_interval)
