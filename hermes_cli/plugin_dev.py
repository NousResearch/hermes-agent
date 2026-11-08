"""Runtime-backed validation behind ``hermes plugins doctor``: every manifest/import/registration
check routes through the real runtime contracts instead of a parallel scanner."""

from __future__ import annotations

import inspect
import json
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


def _stage_copy(source: Path, staged: Path) -> None:
    """Copy *source* to *staged*, refusing any directory that leads into the destination.

    TMPDIR can sit inside the plugin being checked, or a symlink/junction inside the plugin can
    point at TMPDIR; either way the copy walks its own output until the path is too long or the
    disk is full. Paths are resolved so link aliases cannot hide the overlap.
    """
    dest = staged.resolve()
    skipped = shutil.ignore_patterns(*_STAGING_SKIPPED_DIRS, "*.pyc")

    def _refuse_overlap(path: Path) -> None:
        real = path.resolve()
        if dest.is_relative_to(real) or real.is_relative_to(dest):
            raise _DoctorLoadError(
                f"Refusing to stage {source}: {path} leads into Doctor's temporary directory "
                f"{staged.parent}, so the copy would recurse into itself. Point TMPDIR outside "
                "the plugin and remove links to it.")

    def _ignore(directory: str, names: list[str]) -> set[str]:
        for name in names:
            if os.path.isdir(os.path.join(directory, name)):
                _refuse_overlap(Path(directory, name))
        return skipped(directory, names)

    _refuse_overlap(source)
    shutil.copytree(source, staged, ignore=_ignore)


@contextmanager
def _doctor_runtime(plugin_path: Path):
    """Load one plugin through the real runtime and restore global state afterwards.

    Private Doctor machinery, not a plugin test framework. Registration code runs under a temporary
    HERMES_HOME with outbound socket connects blocked.
    """
    stack = ExitStack()
    try:
        # The temp dir enters the stack FIRST so any failure below (ENOSPC / KeyboardInterrupt in
        # copytree) removes it instead of stranding a hermes-plugin-doctor-* directory.
        home = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="hermes-plugin-doctor-")))
        bundled = home / "bundled-plugins"
        plugins_root = home / "plugins"
        bundled.mkdir(parents=True)
        plugins_root.mkdir(parents=True)
        staging_root = plugins_root / plugin_path.name
        if _has_manifest(plugin_path):
            staged = staging_root
            _stage_copy(plugin_path, staged)
        else:
            # A category: stage only its plugin under the category name (discovery keys it
            # ``category/plugin``), never the loose files or skipped dirs beside it.
            members = _category_plugins(plugin_path)
            if len(members) != 1:
                names = ", ".join(member.name for member in members)
                raise _DoctorLoadError(
                    f"{plugin_path} holds {len(members)} plugins ({names}); point Doctor at one of them")
            staged = staging_root / members[0].name
            _stage_copy(members[0], staged)
        stack.enter_context(patch.dict(os.environ, {
            "HERMES_HOME": str(home),
            "HERMES_BUNDLED_PLUGINS": str(bundled),
            "HERMES_ENABLE_PROJECT_PLUGINS": "0",
        }, clear=False))
        stack.enter_context(patch.object(socket, "create_connection", _deny_network))
        stack.enter_context(patch.object(socket.socket, "connect", _deny_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", _deny_network))
    except BaseException:
        stack.close()
        raise

    from hermes_cli.plugins import PluginManager
    from tools.registry import registry
    entries_before = {entry.name: entry for entry in registry._snapshot_entries()}
    policy_before = dict(registry._plugin_override_policy)
    modules_before = {name for name in sys.modules if _is_plugin_module(name)}
    manager = PluginManager()
    try:
        manifests = manager._scan_directory(plugins_root, source="user")
        if not manifests:
            raise _DoctorLoadError(f"Hermes discovery found no valid plugin manifest under {staging_root}")
        if len(manifests) != 1:
            raise _DoctorLoadError(
                f"Expected one plugin manifest, discovered {len(manifests)} under {staging_root}")
        manifest = manifests[0]
        if manifest.kind == "model-provider":
            with _load_model_provider(staged, manifest) as registered:
                yield SimpleNamespace(manifest=manifest, manager=manager, registered_tools=(),
                                      registered_hooks=(), registered_providers=registered)
            return
        manager._load_plugin(manifest)
        loaded = manager._plugins.get(manifest.key or manifest.name)
        if loaded is None:
            raise _DoctorLoadError("Plugin registration produced no runtime record")
        if loaded.error:
            raise _DoctorLoadError(f"Plugin registration failed: {loaded.error}")
        if not loaded.enabled:
            raise _DoctorLoadError("Plugin registration did not enable the runtime record")
        yield SimpleNamespace(
            manifest=manifest, manager=manager, registered_tools=tuple(sorted(loaded.tools_registered)),
            registered_hooks=tuple(loaded.hooks_registered), registered_providers=())
    finally:
        # Dispose the plugin's own registrations FIRST, while the temporary
        # HERMES_HOME still exists. This runs the host-owned ctx.on_unload(...)
        # callbacks — e.g. closing a SQLite handle a context-engine plugin
        # opened under that home. Without it the DB stays open and stack.close()
        # below (TemporaryDirectory removal) fails with WinError 32 on Windows
        # (#99918). Best-effort: the snapshot restore below remains the
        # authoritative registry cleanup, so an unload hiccup never masks it or
        # the original exception.
        try:
            manager.unload()
        except Exception:
            pass
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
            if name not in modules_before and _is_plugin_module(name):
                sys.modules.pop(name, None)
        stack.close()


def _is_plugin_module(name: str) -> bool:
    return name == "hermes_plugins" or name.startswith("hermes_plugins.")


@contextmanager
def _load_model_provider(plugin_dir: Path, manifest):
    """Doctor path for ``kind: model-provider``: those register a ProviderProfile at import through
    providers/ discovery (PluginManager skips the kind), so demanding ``register(ctx)`` would fail
    every valid provider plugin. Registry additions are undone on exit."""
    import providers

    # The live install may already have imported this very plugin (same directory name) during
    # startup discovery; import the copy fresh and put the live module/profiles back afterwards.
    module_name = providers._user_module_name(plugin_dir, "")
    prior_module = sys.modules.pop(module_name, None)
    before = dict(providers._REGISTRY)
    before_aliases = dict(providers._ALIASES)
    try:
        providers._import_plugin_dir(plugin_dir, "user")
        registered = tuple(sorted(
            name for name, profile in providers._REGISTRY.items() if before.get(name) is not profile))
        if not registered:
            raise _DoctorLoadError(
                "model-provider plugin registered no ProviderProfile at import (see the warning above)")
        yield registered
    finally:
        for name in [n for n, p in providers._REGISTRY.items() if before.get(n) is not p]:
            providers._REGISTRY.pop(name)
            providers._SOURCES.pop(name, None)
        providers._REGISTRY.update(before)
        providers._ALIASES.clear()
        providers._ALIASES.update(before_aliases)
        providers._PROVIDER_LIST_CACHE = None
        sys.modules.pop(module_name, None)
        if prior_module is not None:
            sys.modules[module_name] = prior_module


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
    registered_providers: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(finding.level != "error" for finding in self.findings)

    def error(self, message: str) -> None:
        self.findings.append(DoctorFinding("error", message))

    def warning(self, message: str) -> None:
        self.findings.append(DoctorFinding("warning", message))

    def format_text(self) -> str:
        lines = [f"Plugin Doctor: {self.path}"]
        if self.manifest is not None:
            lines.append(
                f"  manifest: {self.manifest.name} "
                f"{self.manifest.version or '(no version)'} ({self.manifest.kind})")
        for finding in self.findings:
            lines.append(f"  {'ERROR' if finding.level == 'error' else 'WARN'}: {finding.message}")
        if self.ok:
            lines.append("  OK: runtime discovery, manifest parsing, import, and registration passed")
        lines.append(
            f"  registrations: {len(self.registered_tools)} tool(s), {len(self.registered_hooks)} hook(s)"
            + (f", provider(s): {', '.join(self.registered_providers)}" if self.registered_providers else ""))
        return "\n".join(lines)


_MANIFEST_NAMES = ("plugin.yaml", "plugin.yml", "plugin.json")


def _has_manifest(path: Path) -> bool:
    return any((path / name).exists() or (path / name).is_symlink() for name in _MANIFEST_NAMES)


# Dirs the staging copy drops, so a category may hold them beside its plugins.
_STAGING_SKIPPED_DIRS = (".git", "__pycache__", ".pytest_cache")


def _candidate_dirs(path: Path) -> list[Path]:
    """Subdirectories of *path* that could be plugins: not skipped by discovery or the staging copy."""
    from hermes_cli.plugins_discovery import is_skipped_plugin_dir

    try:
        return sorted(
            child for child in path.iterdir()
            if child.is_dir()
            and child.name not in _STAGING_SKIPPED_DIRS
            and not is_skipped_plugin_dir(child.name)
        )
    except OSError:
        return []


def _category_plugins(path: Path) -> list[Path]:
    """The plugin subdirectories of a manifest-less *path* when it is a category, else ``[]``.

    A category is a directory whose candidate subdirectories (:func:`_candidate_dirs`) all hold
    a manifest. A plugin checkout next to other directories (``$HOME`` beside ``.hermes``, a repo
    root beside ``tests/``) is not a category."""
    children = _candidate_dirs(path)
    return children if children and all(_has_manifest(child) for child in children) else []


def _holds_plugin(path: Path) -> bool:
    """True when *path* holds a manifest or is a category (:func:`_category_plugins`). Stricter
    than discovery on purpose: resolving an arbitrary directory (e.g. the default ``.``) would
    send Doctor to validate whatever plugin happens to sit inside it."""
    return path.is_dir() and (_has_manifest(path) or bool(_category_plugins(path)))


def _is_plugin_id(raw: str) -> bool:
    """True when *raw* can name an installed plugin (relative, maybe ``category/name``). Dot
    components are excluded: ``.`` joined onto a plugins root would hand Doctor every plugin."""
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
        candidates.append(get_hermes_home() / "plugins" / raw)
        try:
            from hermes_cli.plugins import get_bundled_plugins_dir
            bundled = get_bundled_plugins_dir()
            candidates += [bundled / raw, bundled / "platforms" / raw, bundled / "model-providers" / raw]
        except Exception:
            pass
        candidates.append(Path.cwd() / ".hermes" / "plugins" / raw)
    for candidate in candidates:
        if _holds_plugin(candidate):
            return candidate.resolve()
    if direct_is_dir:
        nested = [os.path.join(raw, c.name) for c in _candidate_dirs(direct) if _has_manifest(c)]
        if nested:
            hint = f"Point Doctor at a plugin directory inside it: {', '.join(nested)}."
        else:
            hint = "Point Doctor at a plugin directory."
        raise FileNotFoundError(
            f"{direct.resolve()} holds no plugin manifest "
            f"({', '.join(_MANIFEST_NAMES)}), and {raw!r} is not an installed "
            f"plugin id. {hint}")
    raise FileNotFoundError(f"Plugin {raw!r} was not found as a path or installed plugin id")


def _accepts_var_kwargs(callback: Any) -> bool:
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _check_manifest_v2(report: DoctorReport, manifest: Any) -> None:
    """Manifest v2 checks: versions, deps, pip declarations, config schema."""
    import importlib.metadata
    import re as _re
    from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION
    mv = getattr(manifest, "manifest_version", 1)
    if mv > SUPPORTED_MANIFEST_VERSION:
        report.warning(
            f"manifest_version {mv} is newer than this Hermes supports "
            f"({SUPPORTED_MANIFEST_VERSION}); unknown fields are ignored")

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
                "advisory; a missing dependency logs a warning at load")

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
            "pin an upper bound (e.g. 'pkg>=1.0,<2') per the dependency policy")
    if missing:
        report.warning(
            "declared python_dependencies not installed: " + ", ".join(missing)
            + " — install and enable the plugin with hermes plugins. "
            "For an enabled plugin, run hermes pm repair, then restart Hermes.")

    schema = getattr(manifest, "config_schema", {}) or {}
    if schema:
        from hermes_cli.plugins import _CONFIG_SCHEMA_TYPES
        for skey, spec in schema.items():
            stype = spec.get("type") if isinstance(spec, dict) else None
            if stype is not None and str(stype).lower() not in _CONFIG_SCHEMA_TYPES:
                report.warning(f"config_schema key {skey!r} declares unknown type {stype!r}")


def _check_desktop_half_copy(report: DoctorReport, path: Path) -> None:
    """Warn when the Desktop app runs a stale copy of this package's desktop ``plugin.js``.

    Electron copies a unified package's desktop half to ``<root>/desktop-plugins/<name>/`` and loads
    THAT copy, so editing the package in place changes nothing on screen until the copy is refreshed.
    The copy carries a ``.hermes-package.json`` marker whose ``source`` names the folder it came from.
    """
    from hermes_cli.desktop_plugin_halves import desktop_source

    source_dir = desktop_source(path)
    if source_dir is None:
        return
    source = source_dir / "plugin.js"
    shown = source.relative_to(path).as_posix()
    from hermes_constants import get_default_hermes_root

    app_root = get_default_hermes_root() / "desktop-plugins"
    try:
        copies = sorted(app_root.glob("*/.hermes-package.json"))
    except OSError:
        return
    for marker_file in copies:
        try:
            marker = json.loads(marker_file.read_text(encoding="utf-8-sig"))
            if Path(marker["source"]).resolve() != source.parent.resolve():
                continue
            if (marker_file.parent / "plugin.js").read_bytes() == source.read_bytes():
                return
        except (OSError, ValueError, KeyError, TypeError):
            continue
        report.warning(
            f"Desktop runs a stale copy of {shown} ({marker_file.parent}); your edits are not "
            "loaded. Refresh it with Capabilities → Plugins → Rescan in the Desktop app (or restart it). "
            "To develop in place, symlink the checkout into plugins/ instead: the Desktop then re-syncs "
            "the copy on every save.")
        return


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
            report.registered_providers = host.registered_providers

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
                            "must accept **kwargs for forward compatibility")

            for kind, declared, registered in (
                ("hook", declared_hooks, host.registered_hooks),
                ("tool", declared_tools, host.registered_tools),
            ):
                declared_names = {name for name in declared if isinstance(name, str)}
                registered_names = set(registered)
                for name in sorted(declared_names - registered_names):
                    report.warning(f"manifest declares {kind} {name!r} but registration did not add it")
                for name in sorted(registered_names - declared_names):
                    report.warning(f"registration adds {kind} {name!r} not listed in provides_{kind}s")

            _check_manifest_v2(report, host.manifest)
    except _DoctorLoadError as exc:
        report.error(str(exc))
    except Exception as exc:
        report.error(f"unexpected validation failure: {type(exc).__name__}: {exc}")
    # Outside the runtime sandbox: it swaps HERMES_HOME for a temp dir, and the copy lives in the real one.
    _check_desktop_half_copy(report, path)
    return report


__all__ = ["DoctorFinding", "DoctorReport", "doctor_plugin", "resolve_plugin_path"]
