"""Detection and removal of pre-rename Hermes systemd units."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from gateway import service_identity
from gateway.systemd_identity import user_unit_dir
from gateway.systemd_runtime import run_systemctl

LEGACY_SERVICE_NAMES = ("hermes.service",)
LEGACY_EXECSTART_MARKERS = (
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
)


def search_paths() -> list[tuple[bool, Path]]:
    return [(False, user_unit_dir()), (True, service_identity.SYSTEM_UNIT_DIR)]


def find_units() -> list[tuple[str, Path, bool]]:
    results: list[tuple[str, Path, bool]] = []
    for is_system, base in search_paths():
        for name in LEGACY_SERVICE_NAMES:
            path = base / name
            try:
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue
            if any(marker in text for marker in LEGACY_EXECSTART_MARKERS):
                results.append((name, path, is_system))
    return results


def has_units() -> bool:
    return bool(find_units())


def remove_units(*, dry_run: bool = False) -> tuple[int, list[Path]]:
    """Stop, disable and unlink legacy units; return removed count and remaining paths."""
    legacy = find_units()
    if dry_run:
        return 0, [path for _, path, _ in legacy]

    removed = 0
    remaining: list[Path] = []

    def remove_scope(units: list[tuple[str, Path]], *, system: bool) -> None:
        nonlocal removed
        for name, path in units:
            try:
                run_systemctl(["stop", name], system=system, check=False, timeout=90)
                run_systemctl(["disable", name], system=system, check=False, timeout=30)
                path.unlink(missing_ok=True)
                removed += 1
            except (OSError, RuntimeError):
                remaining.append(path)
        with contextlib.suppress(RuntimeError):
            run_systemctl(["daemon-reload"], system=system, check=False, timeout=30)

    user_units = [(name, path) for name, path, system in legacy if not system]
    system_units = [(name, path) for name, path, system in legacy if system]
    if user_units:
        remove_scope(user_units, system=False)
    if system_units:
        if os.geteuid() != 0:
            remaining.extend(path for _, path in system_units)
        else:
            remove_scope(system_units, system=True)
    return removed, remaining
