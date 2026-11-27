#!/usr/bin/env python3
"""Catalog CI lint for ``default: true`` plugins that would trigger a PM venv rebuild.

Platform sampling and actionable CI guidance live here in the CI layer while all
workspace-membership rules stay in :func:`pm.plugin_declarations.assess_workspace_membership`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from pm.plugin_declarations import assess_workspace_membership, read_python_declaration

_DEFAULT_REBUILD_GUIDANCE = (
    "default: true plugins install automatically on first launch and updates; triggering a venv "
    "rebuild forces `uv lock`/`uv sync` and a ~1 GB `uv-cache` seed copy on Windows (minutes on "
    "first launch). How to avoid triggering a rebuild: "
    "(1) drop `[build-system]` / `tool.uv.package = true` if the plugin does not need wheel installation; "
    "(2) use dependencies already shipped in Hermes core (e.g. `httpx`, `PyJWT[crypto]`, "
    "`cryptography`, `fastapi`, `pydantic`) instead of introducing new third-party packages; or "
    "(3) if the dependency is genuinely required for a default plugin, add it to Hermes core's "
    "`pyproject.toml` first."
)


def _pinned_python_version(project_root: Path | None = None) -> tuple[str, str]:
    """Return ``(python_full_version, python_version)`` from ``pm/lock.json``, defaulting to ``3.14.0``/``3.14``."""
    from pm import paths

    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(Path(project_root) / "pm" / "lock.json")
    candidates.append(paths.lockfile_path())
    for lock_path in candidates:
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        raw = data.get("packages", {}).get("python", {}).get("version") if isinstance(data, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            continue
        full = raw.partition("+")[0].strip()
        parts = full.split(".")
        if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
            return full, f"{parts[0]}.{parts[1]}"
    return "3.14.0", "3.14"


def _default_plugin_check_targets(
    project_root: Path | None = None,
) -> tuple[tuple[str, str, dict[str, str]], ...]:
    """Representative desktop targets using Hermes's pinned Python version from ``pm/lock.json``."""
    full_ver, minor_ver = _pinned_python_version(project_root)
    return (
        ("linux", "linux-x64", {
            "implementation_name": "cpython",
            "implementation_version": full_ver,
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "platform_release": "6.8.0",
            "platform_system": "Linux",
            "platform_version": "",
            "python_full_version": full_ver,
            "python_version": minor_ver,
            "sys_platform": "linux",
        }),
        ("macos", "darwin-arm64", {
            "implementation_name": "cpython",
            "implementation_version": full_ver,
            "os_name": "posix",
            "platform_machine": "arm64",
            "platform_python_implementation": "CPython",
            "platform_release": "24.0.0",
            "platform_system": "Darwin",
            "platform_version": "",
            "python_full_version": full_ver,
            "python_version": minor_ver,
            "sys_platform": "darwin",
        }),
        ("windows", "win32-x64", {
            "implementation_name": "cpython",
            "implementation_version": full_ver,
            "os_name": "nt",
            "platform_machine": "AMD64",
            "platform_python_implementation": "CPython",
            "platform_release": "11",
            "platform_system": "Windows",
            "platform_version": "10.0.26100",
            "python_full_version": full_ver,
            "python_version": minor_ver,
            "sys_platform": "win32",
        }),
    )


@dataclass(frozen=True)
class DefaultPluginRebuildCheck:
    """Result of checking whether a ``default: true`` plugin would trigger a PM venv rebuild."""

    triggers_rebuild: bool
    reason: str
    unsatisfied_requirements: list[str]
    pyproject_installs: bool
    guidance: str


def check_default_plugin_venv_rebuild(
    plugin_dir: Path,
    *,
    project_root: Path | None = None,
    base_site_packages: Path | None = None,
    target_env: Mapping[str, str] | None = None,
    platforms: Iterable[str] | None = None,
) -> DefaultPluginRebuildCheck:
    """Check whether *plugin_dir* would trigger a PM venv rebuild as a ``default: true`` plugin.

    Delegates every membership rule to :func:`pm.plugin_declarations.assess_workspace_membership`.
    When *target_env* and *base_site_packages* are omitted, checks Linux, macOS, and Windows
    (filtered by *platforms* if provided) so platform-specific rebuild triggers are caught in
    single-OS CI.
    """
    declaration = read_python_declaration(Path(plugin_dir))
    if declaration.external or (not declaration.installs_packages and not declaration.requirements):
        return DefaultPluginRebuildCheck(
            triggers_rebuild=False,
            reason="",
            unsatisfied_requirements=[],
            pyproject_installs=False,
            guidance="",
        )

    if target_env is not None or base_site_packages is not None:
        from packaging.markers import default_environment

        env = dict(default_environment()) if target_env is None else dict(target_env)
        targets: list[tuple[str, Mapping[str, str]]] = [("host", env)]
    else:
        check_targets = _default_plugin_check_targets(project_root)
        allowed = set(platforms) if platforms else None
        targets = [
            (label, env)
            for plat, label, env in check_targets
            if not allowed or plat in allowed
        ]
        if not targets:
            targets = [(label, env) for _plat, label, env in check_targets]

    unsat_ordered: list[str] = []
    unsat_seen: set[str] = set()
    other_reasons: list[str] = []
    any_needs = False
    pyproject_installs = False
    for _label, env in targets:
        assessment = assess_workspace_membership(
            declaration,
            project_root=project_root,
            base_site_packages=base_site_packages,
            target_env=env,
        )
        if not assessment.needs_member:
            continue
        any_needs = True
        pyproject_installs = pyproject_installs or assessment.pyproject_installs
        for spec in assessment.unsatisfied_requirements:
            if spec not in unsat_seen:
                unsat_seen.add(spec)
                unsat_ordered.append(spec)
        for reason_item in assessment.reasons:
            if not reason_item.startswith("unsatisfied Python requirement(s)") and reason_item not in other_reasons:
                other_reasons.append(reason_item)

    if not any_needs:
        return DefaultPluginRebuildCheck(
            triggers_rebuild=False,
            reason="",
            unsatisfied_requirements=[],
            pyproject_installs=False,
            guidance="",
        )

    reasons = list(other_reasons)
    if unsat_ordered:
        reasons.append(
            "unsatisfied Python requirement(s) not in Hermes core base environment: "
            + ", ".join(unsat_ordered)
        )
    reason = "; ".join(reasons) or "triggers PM workspace membership"

    return DefaultPluginRebuildCheck(
        triggers_rebuild=True,
        reason=reason,
        unsatisfied_requirements=unsat_ordered,
        pyproject_installs=pyproject_installs,
        guidance=_DEFAULT_REBUILD_GUIDANCE,
    )
