"""Open-file isolation audit for Hermes Lab processes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

from gateway.dev_control.lab_environment import PRODUCTION_ROOTS, lab_paths_from_env


WRITE_FD_SUFFIXES = ("u", "w")
BENIGN_OUTPUT_FDS = {"1w", "2w"}
BENIGN_OUTPUT_ROOTS = tuple(
    Path(root).resolve(strict=False)
    for root in ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")
)


def audit_process_isolation(
    *,
    pids: Iterable[int | str],
    allowed_roots: Optional[Iterable[Path | str]] = None,
    forbidden_roots: Optional[Iterable[Path | str]] = None,
) -> dict[str, Any]:
    """Return whether each regular-file write handle stays inside lab roots."""

    paths = lab_paths_from_env()
    allowed = [Path(root).expanduser().resolve(strict=False) for root in (allowed_roots or [paths["lab_home"]])]
    forbidden = [Path(root).expanduser().resolve(strict=False) for root in (forbidden_roots or PRODUCTION_ROOTS)]
    pid_values = _normalize_pids(pids)
    handles: list[dict[str, Any]] = []
    warnings: list[str] = []
    for pid in pid_values:
        try:
            handles.extend(_write_handles_for_pid(pid))
        except Exception as exc:  # noqa: BLE001 - audit must report uncertainty without crashing callers.
            warnings.append(f"pid {pid}: open-file audit failed: {exc}")

    offending: list[dict[str, Any]] = []
    for handle in handles:
        path = Path(str(handle.get("path") or "")).expanduser().resolve(strict=False)
        in_allowed = any(_is_same_or_child(path, root) for root in allowed)
        in_forbidden = any(_is_same_or_child(path, root) for root in forbidden)
        if not in_forbidden and not in_allowed and _is_benign_output_handle(handle, path):
            continue
        if in_forbidden or not in_allowed:
            offending.append({**handle, "path": str(path), "in_forbidden_root": in_forbidden})

    return {
        "ok": not offending and not warnings,
        "object": "hermes.dev_lab_process_isolation",
        "pids": pid_values,
        "allowed_roots": [str(root) for root in allowed],
        "forbidden_roots": [str(root) for root in forbidden],
        "write_handles": handles,
        "offending_paths": offending,
        "warnings": warnings,
        "authoritative": True,
    }


def audit_current_process_isolation(extra_pids: Optional[Iterable[int | str]] = None) -> dict[str, Any]:
    pids = [os.getpid(), *(extra_pids or [])]
    return audit_process_isolation(pids=pids)


def _normalize_pids(pids: Iterable[int | str]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for raw in pids:
        try:
            pid = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if pid > 0 and pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result


def _write_handles_for_pid(pid: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["lsof", "-nP", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"lsof exited with {result.returncode}")
    return _parse_lsof_table(result.stdout, pid=pid)


def _parse_lsof_table(output: str, *, pid: int) -> list[dict[str, Any]]:
    handles: list[dict[str, Any]] = []
    for line in output.splitlines()[1:]:
        parts = line.split(None, 8)
        if len(parts) < 9:
            continue
        _command, pid_text, _user, fd, file_type, _device, _size, _node, name = parts
        if str(pid_text) != str(pid):
            continue
        if file_type != "REG":
            continue
        if not fd.endswith(WRITE_FD_SUFFIXES):
            continue
        if not name.startswith("/"):
            continue
        handles.append({
            "pid": pid,
            "fd": fd,
            "type": file_type,
            "path": str(Path(name).resolve(strict=False)),
        })
    return handles


def _is_same_or_child(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_benign_output_handle(handle: dict[str, Any], path: Path) -> bool:
    fd = str(handle.get("fd") or "")
    if fd not in BENIGN_OUTPUT_FDS:
        return False
    return any(_is_same_or_child(path, root) for root in BENIGN_OUTPUT_ROOTS)
