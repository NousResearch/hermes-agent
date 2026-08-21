"""Runtime-backed validation behind ``hermes plugins doctor``.

The Doctor originated in #46456 / contributor PR #46457 by 峯岸 亮
(@zapabob).  This core command keeps that contribution's manifest/import/
registration validation intent while routing every check through the current
runtime contracts instead of maintaining a parallel scanner.
"""

from __future__ import annotations

import inspect
import os
import shutil
import socket
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import patch

from hermes_constants import get_hermes_home


class _DoctorLoadError(RuntimeError):
    """Raised when the real plugin runtime cannot load the target."""


def _deny_network(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("network access is disabled while Plugin Doctor runs")


@contextmanager
def _doctor_runtime(plugin_path: Path):
    """Load one plugin through the real runtime and restore global state.

    This is deliberately private Doctor machinery, not a standalone plugin
    test framework. Registration code executes under a temporary HERMES_HOME
    with outbound socket connects blocked.
    """
    temporary_home = tempfile.TemporaryDirectory(prefix="hermes-plugin-doctor-")
    stack = ExitStack()
    home = Path(temporary_home.name)
    bundled = home / "bundled-plugins"
    plugins_root = home / "plugins"
    bundled.mkdir(parents=True)
    plugins_root.mkdir(parents=True)
    copied = plugins_root / plugin_path.name
    shutil.copytree(
        plugin_path,
        copied,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )

    stack.enter_context(
        patch.dict(
            os.environ,
            {
                "HERMES_HOME": str(home),
                "HERMES_BUNDLED_PLUGINS": str(bundled),
                "HERMES_ENABLE_PROJECT_PLUGINS": "0",
            },
            clear=False,
        )
    )
    stack.enter_context(patch.object(socket, "create_connection", _deny_network))
    stack.enter_context(patch.object(socket.socket, "connect", _deny_network))
    stack.enter_context(patch.object(socket.socket, "connect_ex", _deny_network))

    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    entries_before = {entry.name: entry for entry in registry._snapshot_entries()}
    policy_before = dict(registry._plugin_override_policy)
    modules_before = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }
    manager = PluginManager()
    try:
        manifests = manager._scan_directory(plugins_root, source="user")
        if not manifests:
            raise _DoctorLoadError(
                f"Hermes discovery found no valid plugin manifest under {copied}"
            )
        if len(manifests) != 1:
            raise _DoctorLoadError(
                f"Expected one plugin manifest, discovered {len(manifests)} under {copied}"
            )
        manifest = manifests[0]
        manager._load_plugin(manifest)
        loaded = manager._plugins.get(manifest.key or manifest.name)
        if loaded is None:
            raise _DoctorLoadError("Plugin registration produced no runtime record")
        if loaded.error:
            raise _DoctorLoadError(f"Plugin registration failed: {loaded.error}")
        if not loaded.enabled:
            raise _DoctorLoadError("Plugin registration did not enable the runtime record")
        yield SimpleNamespace(
            manifest=manifest,
            manager=manager,
            registered_tools=tuple(sorted(loaded.tools_registered)),
            registered_hooks=tuple(loaded.hooks_registered),
        )
    finally:
        entries_after = {entry.name: entry for entry in registry._snapshot_entries()}
        changed_names = {
            name
            for name in set(entries_before) | set(entries_after)
            if entries_after.get(name) is not entries_before.get(name)
        }
        with registry._lock:
            for name in changed_names:
                previous = entries_before.get(name)
                if previous is None:
                    registry._tools.pop(name, None)
                else:
                    registry._tools[name] = previous
            registry._plugin_override_policy.clear()
            registry._plugin_override_policy.update(policy_before)
            if changed_names:
                registry._generation += 1
        for name in list(sys.modules):
            if (
                name not in modules_before
                and (name == "hermes_plugins" or name.startswith("hermes_plugins."))
            ):
                sys.modules.pop(name, None)
        stack.close()
        temporary_home.cleanup()


@dataclass(frozen=True)
class DoctorFinding:
    level: Literal["error", "warning"]
    message: str


@dataclass
class DoctorReport:
    path: Path
    manifest: Any | None = None
    findings: list[DoctorFinding] = field(default_factory=list)
    registered_tools: tuple[str, ...] = ()
    registered_hooks: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(finding.level == "error" for finding in self.findings)

    def error(self, message: str) -> None:
        self.findings.append(DoctorFinding("error", message))

    def warning(self, message: str) -> None:
        self.findings.append(DoctorFinding("warning", message))

    def format_text(self) -> str:
        lines = [f"Plugin Doctor: {self.path}"]
        if self.manifest is not None:
            lines.append(
                f"  manifest: {self.manifest.name} "
                f"{self.manifest.version or '(no version)'} ({self.manifest.kind})"
            )
        for finding in self.findings:
            marker = "ERROR" if finding.level == "error" else "WARN"
            lines.append(f"  {marker}: {finding.message}")
        if self.ok:
            lines.append(
                "  OK: runtime discovery, manifest parsing, import, and registration passed"
            )
        lines.append(
            f"  registrations: {len(self.registered_tools)} tool(s), "
            f"{len(self.registered_hooks)} hook(s)"
        )
        return "\n".join(lines)


_MANIFEST_NAMES = ("plugin.yaml", "plugin.yml", "plugin.json")


def _has_manifest(path: Path) -> bool:
    return any(
        (path / name).exists() or (path / name).is_symlink()
        for name in _MANIFEST_NAMES
    )


def _holds_plugin(path: Path) -> bool:
    """True when plugin discovery would find a manifest under *path*.

    Mirrors ``PluginManager._scan_directory``: a manifest in *path* itself
    (flat layout) or in one immediate subdirectory (category layout, where
    the category directory carries no manifest of its own).

    Doctor copies the resolved directory wholesale before the runtime gets
    to reject it, so an unvalidated resolve is a disk-usage bug, not just a
    confusing error: ``hermes plugins doctor`` with no argument defaults to
    ``.``, and any directory used to satisfy that.
    """
    if not path.is_dir():
        return False
    if _has_manifest(path):
        return True
    try:
        children = sorted(path.iterdir())
    except OSError:
        return False
    return any(child.is_dir() and _has_manifest(child) for child in children)


def _is_plugin_id(raw: str) -> bool:
    """True when *raw* can name an installed plugin rather than a path.

    Ids are relative and may carry one category segment
    (``image_gen/openai``). Dot components are excluded: joining ``.``
    onto a plugins root yields the root itself, which would hand Doctor
    every installed plugin at once instead of one.
    """
    if not raw or PurePath(raw).is_absolute():
        return False
    parts = PurePath(raw).parts
    return bool(parts) and all(part not in {os.curdir, os.pardir} for part in parts)


def resolve_plugin_path(target: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit path or an installed/bundled plugin id."""
    raw = os.fspath(target or ".")
    direct = Path(raw).expanduser()
    direct_is_dir = direct.is_dir()
    if direct_is_dir and _holds_plugin(direct):
        return direct.resolve()

    candidates: list[Path] = []
    if _is_plugin_id(raw):
        user_root = get_hermes_home() / "plugins"
        candidates.append(user_root / raw)
        try:
            from hermes_cli.plugins import get_bundled_plugins_dir

            bundled = get_bundled_plugins_dir()
            candidates.extend(
                [
                    bundled / raw,
                    bundled / "platforms" / raw,
                    bundled / "model-providers" / raw,
                ]
            )
        except Exception:
            pass
        candidates.append(Path.cwd() / ".hermes" / "plugins" / raw)
    for candidate in candidates:
        if _holds_plugin(candidate):
            return candidate.resolve()
    if direct_is_dir:
        raise FileNotFoundError(
            f"{direct.resolve()} holds no plugin manifest "
            f"({', '.join(_MANIFEST_NAMES)}), and {raw!r} is not an installed "
            "plugin id. Point Doctor at a plugin directory."
        )
    raise FileNotFoundError(
        f"Plugin {raw!r} was not found as a path or installed plugin id"
    )


def _accepts_var_kwargs(callback: Any) -> bool:
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _check_manifest_v2(report: "DoctorReport", manifest: Any) -> None:
    """Manifest v2 (#64165) checks: versions, deps, pip declarations, schema."""
    import importlib.metadata
    import re as _re

    from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION

    mv = getattr(manifest, "manifest_version", 1)
    if mv > SUPPORTED_MANIFEST_VERSION:
        report.warning(
            f"manifest_version {mv} is newer than this Hermes supports "
            f"({SUPPORTED_MANIFEST_VERSION}); unknown fields are ignored"
        )

    api_version = getattr(manifest, "api_version", None)
    if api_version is not None and api_version < 1:
        report.warning(f"api_version {api_version} is not a valid API generation (>= 1)")

    for dep in getattr(manifest, "requires_plugins", []) or []:
        dep_id = dep.get("id") if isinstance(dep, dict) else None
        if not dep_id:
            report.warning(f"requires_plugins entry {dep!r} has no plugin id")
            continue
        vr = dep.get("version_range")
        if vr:
            report.warning(
                f"requires plugin {dep_id!r} ({vr}) — version ranges are "
                "advisory; a missing dependency logs a warning at load"
            )

    pydeps = getattr(manifest, "python_dependencies", []) or []
    missing: list[str] = []
    unpinned: list[str] = []
    for req in pydeps:
        dist = _re.split(r"[<>=!~\[;\s]", req, maxsplit=1)[0].strip()
        if not _re.search(r"<|==|~=", req):
            unpinned.append(req)
        if not dist:
            continue
        try:
            importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            missing.append(req)
        except Exception:
            continue
    for req in unpinned:
        report.warning(
            f"python_dependencies entry {req!r} has no upper bound — "
            "pin an upper bound (e.g. 'pkg>=1.0,<2') per the dependency policy"
        )
    if missing:
        report.warning(
            "declared python_dependencies not installed: "
            + ", ".join(missing)
            + " — Hermes never auto-installs plugin dependencies; "
            + "install manually: pip install "
            + " ".join(f"'{m}'" for m in missing)
        )

    schema = getattr(manifest, "config_schema", {}) or {}
    if schema:
        from hermes_cli.plugins import _CONFIG_SCHEMA_TYPES

        for skey, spec in schema.items():
            if not isinstance(spec, dict):
                continue
            stype = spec.get("type")
            if stype is not None and str(stype).lower() not in _CONFIG_SCHEMA_TYPES:
                report.warning(
                    f"config_schema key {skey!r} declares unknown type {stype!r}"
                )


def doctor_plugin(target: str | os.PathLike[str] | None = None) -> DoctorReport:
    """Validate one plugin through Hermes' real scanner and registration path."""
    try:
        path = resolve_plugin_path(target)
    except FileNotFoundError as exc:
        report = DoctorReport(Path(os.fspath(target or ".")).expanduser())
        report.error(str(exc))
        return report

    report = DoctorReport(path)
    try:
        with _doctor_runtime(path) as host:
            report.manifest = host.manifest
            report.registered_tools = host.registered_tools
            report.registered_hooks = host.registered_hooks

            from hermes_cli.plugins import VALID_HOOKS

            declared_hooks = host.manifest.provides_hooks
            declared_tools = host.manifest.provides_tools
            if not isinstance(declared_hooks, list):
                report.error("provides_hooks must be a list")
                declared_hooks = []
            if not isinstance(declared_tools, list):
                report.error("provides_tools must be a list")
                declared_tools = []

            for name in declared_hooks:
                if not isinstance(name, str):
                    report.error("provides_hooks entries must be strings")
                elif name not in VALID_HOOKS:
                    report.error(f"unknown hook {name!r} in provides_hooks")

            for hook_name, callbacks in host.manager._hooks.items():
                if hook_name not in VALID_HOOKS:
                    report.error(f"registered unknown hook {hook_name!r}")
                for callback in callbacks:
                    if not _accepts_var_kwargs(callback):
                        callback_name = getattr(callback, "__name__", repr(callback))
                        report.error(
                            f"hook callback {callback_name!r} for {hook_name!r} "
                            "must accept **kwargs for forward compatibility"
                        )

            declared_hook_names = {name for name in declared_hooks if isinstance(name, str)}
            registered_hook_names = set(host.registered_hooks)
            for name in sorted(declared_hook_names - registered_hook_names):
                report.warning(f"manifest declares hook {name!r} but registration did not add it")
            for name in sorted(registered_hook_names - declared_hook_names):
                report.warning(f"registration adds hook {name!r} not listed in provides_hooks")

            declared_tool_names = {name for name in declared_tools if isinstance(name, str)}
            registered_tool_names = set(host.registered_tools)
            for name in sorted(declared_tool_names - registered_tool_names):
                report.warning(f"manifest declares tool {name!r} but registration did not add it")
            for name in sorted(registered_tool_names - declared_tool_names):
                report.warning(f"registration adds tool {name!r} not listed in provides_tools")

            _check_manifest_v2(report, host.manifest)
    except _DoctorLoadError as exc:
        report.error(str(exc))
    except Exception as exc:
        report.error(f"unexpected validation failure: {type(exc).__name__}: {exc}")
    return report


__all__ = [
    "DoctorFinding",
    "DoctorReport",
    "doctor_plugin",
    "resolve_plugin_path",
]
