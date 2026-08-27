"""Read-only coherence audit for processes launched from Ares releases."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import psutil

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_ROLES = frozenset(
    {
        "controller",
        "cli",
        "tui",
        "tui_gateway",
        "gateway",
        "desktop_backend",
        "profile_backend",
        "mcp_watchdog",
    }
)


@dataclass(frozen=True)
class ManagedRuntimeProcess:
    """One process whose executable/cwd/argv binds it to an Ares release."""

    pid: int
    ppid: int
    role: str
    revision: str | None
    executable: Path | None
    cwd: Path | None
    argv: tuple[str, ...]
    deleted_runtime_mappings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeProcessDrift:
    process: ManagedRuntimeProcess
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeAuditReport:
    processes: tuple[ManagedRuntimeProcess, ...]
    coherent: tuple[ManagedRuntimeProcess, ...]
    stale: tuple[RuntimeProcessDrift, ...]

    @property
    def ok(self) -> bool:
        # A controller running ``ares doctor`` must discover at least itself.
        # Treat an empty projection as an audit failure rather than silently
        # certifying a process iterator/permission failure as coherent.
        return bool(self.processes) and not self.stale

    @property
    def managed_count(self) -> int:
        return len(self.processes)

    def summary(self) -> str:
        roles: dict[str, int] = {}
        for process in self.processes:
            roles[process.role] = roles.get(process.role, 0) + 1
        role_text = ",".join(f"{role}:{roles[role]}" for role in sorted(roles)) or "none"
        stale_text = ",".join(
            f"{finding.process.pid}/{finding.process.role}:{'+'.join(finding.reasons)}"
            for finding in self.stale[:8]
        )
        if len(self.stale) > 8:
            stale_text += f",+{len(self.stale) - 8}"
        return (
            f"managed={self.managed_count} coherent={len(self.coherent)} "
            f"stale={len(self.stale)} roles={role_text}"
            + (f" drift={stale_text}" if stale_text else "")
        )


def _hermes_subcommand(argv: Sequence[str]) -> str | None:
    """Use Hermes's canonical top-level parser-backed process classifier."""

    try:
        from hermes_cli.update_cmd import _hermes_holder_subcommand

        return _hermes_holder_subcommand(shlex.join(argv))
    except Exception:
        return None


def classify_runtime_role(argv: Sequence[str]) -> str | None:
    """Classify only exact Ares/Hermes entry shapes; never scan substrings."""

    values = tuple(str(value) for value in argv if str(value))
    if not values:
        return None
    executable_name = Path(values[0]).name
    if executable_name in {"Ares", "Ares.exe"}:
        return "desktop"
    if any(Path(value).name == "mcp_stdio_watchdog.py" for value in values[1:]):
        return "mcp_watchdog"
    try:
        module_index = values.index("-m")
        module = values[module_index + 1]
    except (ValueError, IndexError):
        return None
    arguments = values[module_index + 2 :]
    if module == "ares_runtime.local_runtime":
        return "controller"
    if module == "tui_gateway.entry":
        return "tui_gateway"
    if module != "hermes_cli.main":
        return None
    if "--tui" in arguments:
        return "tui"
    subcommand = _hermes_subcommand(values)
    if subcommand == "gateway":
        return "gateway"
    if subcommand == "serve":
        return "profile_backend" if "--profile" in arguments else "desktop_backend"
    if subcommand is None:
        return "cli"
    return None


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return _resolved(left) == _resolved(right)


def _release_revision(
    *, cwd: Path | None, argv: Sequence[str], releases_dir: Path
) -> str | None:
    root = _resolved(releases_dir)
    candidates: list[Path] = []
    if cwd is not None:
        candidates.append(cwd)
    for value in argv:
        if value.startswith(os.sep):
            candidates.append(Path(value))
    for candidate in candidates:
        try:
            relative = _resolved(candidate).relative_to(root)
        except (OSError, ValueError):
            continue
        if len(relative.parts) >= 2 and relative.parts[1] == "source":
            revision = relative.parts[0]
            if _REVISION_RE.fullmatch(revision):
                return revision
    return None


def _deleted_runtime_mappings(pid: int, releases_dir: Path) -> tuple[str, ...]:
    maps = Path("/proc") / str(pid) / "maps"
    try:
        lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    root = str(_resolved(releases_dir))
    return tuple(
        line
        for line in lines
        if "(deleted)" in line and root in line and "/.venv/" in line
    )


def discover_managed_runtime_processes(
    releases_dir: Path,
) -> tuple[ManagedRuntimeProcess, ...]:
    """Discover every recognized process bound to a managed release path."""

    discovered: list[ManagedRuntimeProcess] = []
    try:
        processes = psutil.process_iter(["pid", "ppid", "exe", "cwd", "cmdline"])
    except Exception:
        return ()
    for process in processes:
        try:
            info = process.info
            argv = tuple(str(value) for value in (info.get("cmdline") or ()) if value)
            role = classify_runtime_role(argv)
            if role is None:
                continue
            cwd_value = info.get("cwd")
            cwd = Path(cwd_value) if cwd_value else None
            revision = _release_revision(cwd=cwd, argv=argv, releases_dir=releases_dir)
            if revision is None:
                continue
            executable_value = info.get("exe")
            executable = Path(executable_value) if executable_value else None
            pid = int(info["pid"])
            discovered.append(
                ManagedRuntimeProcess(
                    pid=pid,
                    ppid=int(info.get("ppid") or 0),
                    role=role,
                    revision=revision,
                    executable=executable,
                    cwd=cwd,
                    argv=argv,
                    deleted_runtime_mappings=_deleted_runtime_mappings(pid, releases_dir),
                )
            )
        except (psutil.Error, OSError, TypeError, ValueError):
            continue
    return tuple(sorted(discovered, key=lambda item: item.pid))


def audit_process_snapshots(
    processes: Iterable[ManagedRuntimeProcess],
    *,
    active_revision: str,
    active_source: Path,
    expected_python: Path,
) -> RuntimeAuditReport:
    """Compare captured processes with the selected immutable release."""

    if not _REVISION_RE.fullmatch(active_revision):
        raise ValueError("active_revision must be a 40-character lowercase Git revision")
    expected_source = _resolved(active_source)
    releases_root = expected_source.parent.parent
    expected_interpreter = _resolved(expected_python)
    ordered = tuple(sorted(processes, key=lambda item: item.pid))
    coherent: list[ManagedRuntimeProcess] = []
    stale: list[RuntimeProcessDrift] = []
    for process in ordered:
        reasons: list[str] = []
        if process.revision is None:
            reasons.append("release_unknown")
        elif process.revision != active_revision:
            reasons.append("release_mismatch")
        elif process.cwd is not None:
            resolved_cwd = _resolved(process.cwd)
            try:
                resolved_cwd.relative_to(expected_source)
            except ValueError:
                # Hermes may intentionally chdir to an operator workspace.
                # Only a cwd that remains inside a managed release but outside
                # the active source is evidence of runtime-source drift.
                try:
                    resolved_cwd.relative_to(releases_root)
                except ValueError:
                    pass
                else:
                    reasons.append("source_mismatch")
        if process.role in _PYTHON_ROLES:
            if process.executable is None:
                reasons.append("interpreter_unknown")
            elif not _same_path(process.executable, expected_interpreter):
                reasons.append("interpreter_mismatch")
        if process.deleted_runtime_mappings:
            reasons.append("deleted_runtime_mapping")
        if reasons:
            stale.append(RuntimeProcessDrift(process, tuple(dict.fromkeys(reasons))))
        else:
            coherent.append(process)
    return RuntimeAuditReport(ordered, tuple(coherent), tuple(stale))


def audit_managed_runtime_processes(
    *,
    releases_dir: Path,
    active_revision: str,
    active_source: Path,
    expected_python: Path,
) -> RuntimeAuditReport:
    """Discover and audit the live managed-runtime fleet."""

    return audit_process_snapshots(
        discover_managed_runtime_processes(releases_dir),
        active_revision=active_revision,
        active_source=active_source,
        expected_python=expected_python,
    )
