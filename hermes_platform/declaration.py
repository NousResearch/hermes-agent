"""Parse and register application requirements shared by MCP and skill offer-time gates."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from hermes_platform.resolver.app import LOCATION_KINDS, AppDef, AppLocation

__all__ = [
    "AppSpec",
    "RequiresSpec",
    "Declaration",
    "DeclarationError",
    "gpu_label",
    "parse_app",
    "parse_requires",
    "parse_declaration",
    "register",
    "unregister",
    "lookup",
    "clear",
]


class DeclarationError(ValueError):
    """Declaration parse/validation failure."""


@dataclass(frozen=True)
class AppSpec:
    """Where the application lives, one ``AppDef`` per OS it is detectable on."""

    per_os: Mapping[str, AppDef] = field(default_factory=dict)

    def for_os(self, os_family: str) -> Optional[AppDef]:
        return self.per_os.get(os_family)


@dataclass(frozen=True)
class RequiresSpec:
    """What the server needs before it is offered. Separate from ``app`` on purpose: one is data, one is policy."""

    app: bool = False
    min_version: Optional[str] = None
    gpu: Optional[str] = None


@dataclass(frozen=True)
class Declaration:
    """A parsed declaration; satisfies ``hermes_platform.resolver.availability._HasRequirements``."""

    name: str
    app: Optional[AppSpec]
    requires: RequiresSpec = RequiresSpec()

    @property
    def requires_app(self) -> bool:
        return self.requires.app

    @property
    def min_version(self) -> Optional[str]:
        return self.requires.min_version

    @property
    def required_gpu(self) -> Optional[str]:
        return self.requires.gpu

    def app_for(self, os_family: str) -> Optional[AppDef]:
        return self.app.for_os(os_family) if self.app else None


_APP_OS_FAMILIES = ("win32", "darwin", "linux")
_APP_PRESENCE = ("executable", "bundle")
_APP_VERSION_KINDS = {"pe_resource": "win32", "uninstall_registry": "win32", "plist": "darwin", "none": None}
_APP_LIVENESS_KINDS = ("server_json", "none")
_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")
# `requires.gpu` values and how a sentence names them. Only NVIDIA: `facts.gpu_class()` reports the
# highest-priority vendor present, which answers "is an NVIDIA GPU here" exactly and would not answer
# it for AMD or Intel on a machine that also has an NVIDIA GPU.
GPU_LABELS = {"nvidia": "an NVIDIA GPU"}


def gpu_label(gpu: Optional[str]) -> str:
    """How a user-facing sentence names a ``requires.gpu`` value."""
    return GPU_LABELS.get(gpu or "", "a supported GPU")


def _require_mapping(where: str, key: str, raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise DeclarationError(f"{where}: '{key}' must be a mapping")
    return raw


def _location_is_rooted(location: str, osf: str) -> bool:
    if ".." in location.replace("\\", "/").split("/") or "://" in location:
        return False
    if location.startswith(("~", "%", "$")):
        return True
    if osf == "win32":
        return bool(re.match(r"^[A-Za-z]:[\\/]", location))
    return location.startswith("/")


def _is_relative_name(value: Any, *, allow_dirs: bool) -> bool:
    """Checks the stripped value, the one later joined: an empty part rejects a leading / or \\, a colon rejects a drive."""
    if not isinstance(value, str) or not value.strip() or "://" in value:
        return False
    parts = value.strip().replace("\\", "/").split("/")
    return ".." not in parts and "" not in parts and (allow_dirs or len(parts) == 1) and ":" not in parts[0]


def _parse_location(where: str, osf: str, presence: str, label: str, raw: Any) -> AppLocation:
    """A string is a path, where `*` may stand for a versioned folder; a mapping names a location kind."""
    if isinstance(raw, str):
        path = raw.strip()
        if not path:
            raise DeclarationError(f"{where}: {label} is required")
        if not _location_is_rooted(path, osf) or "**" in path:
            raise DeclarationError(
                f"{where}: {label} must be absolute or start with ~ / %VAR% / $VAR, without '..', '**' or a URL scheme")
        return AppLocation("path", path)
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str) or raw["kind"] not in LOCATION_KINDS:
        raise DeclarationError(f"{where}: {label} must be a path or a mapping with kind one of {sorted(LOCATION_KINDS)}")
    kind = raw["kind"]
    spec = LOCATION_KINDS[kind]
    if spec.only_on and spec.only_on != osf:
        raise DeclarationError(f"{where}: {label} kind {kind!r} is only valid under app.{spec.only_on}")
    if spec.yields != presence:
        raise DeclarationError(f"{where}: {label} kind {kind!r} needs app.{osf}.presence {spec.yields}")
    value = raw.get(spec.key)
    if not _is_relative_name(value, allow_dirs=False):
        raise DeclarationError(f"{where}: {label}.{spec.key} must be a bare name")
    file = raw.get("file", "")
    if spec.needs_file and not _is_relative_name(file, allow_dirs=True):
        raise DeclarationError(f"{where}: {label}.file must be a relative path inside the install folder")
    return AppLocation(kind, value.strip(), str(file).strip())


def _parse_app_os(where: str, name: str, osf: str, raw: Any) -> AppDef:
    _require_mapping(where, f"app.{osf}", raw)
    presence = raw.get("presence")
    if presence not in _APP_PRESENCE:
        raise DeclarationError(f"{where}: app.{osf}.presence must be one of {_APP_PRESENCE}")
    location = raw.get("location")
    if isinstance(location, list):
        if not location:
            raise DeclarationError(f"{where}: app.{osf}.location is required")
        locations = tuple(_parse_location(where, osf, presence, f"app.{osf}.location[{i}]", item)
                          for i, item in enumerate(location))
    elif isinstance(location, str):
        locations = (_parse_location(where, osf, presence, f"app.{osf}.location", location),)
    else:
        raise DeclarationError(f"{where}: app.{osf}.location is required")
    version = raw.get("version") or {"kind": "none"}
    _require_mapping(where, f"app.{osf}.version", version)
    vkind = version.get("kind", "none")
    if vkind not in _APP_VERSION_KINDS:
        raise DeclarationError(f"{where}: app.{osf}.version.kind must be one of {sorted(_APP_VERSION_KINDS)}")
    only_on = _APP_VERSION_KINDS[vkind]
    if only_on and only_on != osf:
        raise DeclarationError(f"{where}: app.{osf}.version.kind {vkind!r} is only valid under app.{only_on}")
    varg = str(version.get("display_name_prefix") or "")
    if vkind == "uninstall_registry" and not varg:
        raise DeclarationError(f"{where}: app.{osf}.version.display_name_prefix is required for uninstall_registry")
    liveness = raw.get("liveness") or {"kind": "none"}
    _require_mapping(where, f"app.{osf}.liveness", liveness)
    lkind = liveness.get("kind", "none")
    if lkind not in _APP_LIVENESS_KINDS:
        raise DeclarationError(f"{where}: app.{osf}.liveness.kind must be one of {_APP_LIVENESS_KINDS}")
    lpath = str(liveness.get("path") or "")
    if lkind == "server_json" and not lpath:
        raise DeclarationError(f"{where}: app.{osf}.liveness.path is required for server_json")
    return AppDef(
        app_id=name, os_family=osf, presence=presence, locations=locations,
        version_kind=vkind, version_arg=varg,
        liveness_kind=lkind, liveness_path=lpath,
        liveness_pid_key=str(liveness.get("pid_key") or "pid"),
        liveness_url_key=str(liveness.get("url_key") or "http"),
        liveness_token_key=str(liveness.get("token_key") or "token"),
        endpoint_path=str(liveness.get("endpoint_path") or "/mcp"),
    )


def parse_app(raw: Any, *, name: str, where: str) -> Optional[AppSpec]:
    """Parse the ``app:`` block (already-decoded mapping, not YAML text) into an ``AppSpec``."""
    if raw is None:
        return None
    _require_mapping(where, "app", raw)
    unknown = set(raw) - set(_APP_OS_FAMILIES)
    if unknown:
        raise DeclarationError(f"{where}: app has unknown OS keys {sorted(unknown)}; use {_APP_OS_FAMILIES}")
    if not raw:
        raise DeclarationError(f"{where}: app needs at least one OS block")
    return AppSpec(per_os={osf: _parse_app_os(where, name, osf, raw[osf]) for osf in raw})


def parse_requires(raw: Any, app: Optional[AppSpec], *, where: str) -> RequiresSpec:
    """Parse the ``requires:`` block; needs the parsed ``app`` to cross-check version policy."""
    if raw is None:
        return RequiresSpec()
    _require_mapping(where, "requires", raw)
    unknown = set(raw) - {"app", "min_version", "gpu"}
    if unknown:
        raise DeclarationError(f"{where}: requires has unknown keys {sorted(unknown)}")
    gpu = raw.get("gpu")
    if gpu is not None and (not isinstance(gpu, str) or gpu not in GPU_LABELS):
        raise DeclarationError(f"{where}: requires.gpu must be one of {sorted(GPU_LABELS)}")
    needs_app = raw.get("app", False)
    if not isinstance(needs_app, bool):
        raise DeclarationError(f"{where}: requires.app must be a boolean")
    if needs_app and app is None:
        raise DeclarationError(f"{where}: requires.app is true but the declaration has no 'app' block")
    min_version = raw.get("min_version")
    if min_version is not None:
        if not isinstance(min_version, str) or not _VERSION_RE.fullmatch(min_version):
            raise DeclarationError(f"{where}: requires.min_version must be a dotted numeric string")
        if not needs_app:
            raise DeclarationError(f"{where}: requires.min_version needs requires.app: true")
        assert app is not None
        # Linux has no version source, so a Linux block is gated on presence only; Windows and macOS must read
        # one, and a Linux-only declaration has nothing to compare.
        unversioned = [osf for osf in sorted(app.per_os) if app.per_os[osf].version_kind == "none"
                       and (osf != "linux" or len(app.per_os) == 1)]
        if unversioned:
            raise DeclarationError(
                f"{where}: requires.min_version needs a version source under app.{unversioned[0]} (kind is none)")
    return RequiresSpec(app=needs_app, min_version=min_version, gpu=gpu)


def parse_declaration(name: str, raw_app: Any, raw_requires: Any, *, where: str) -> Declaration:
    """Parse both blocks of one declaration. ``where`` is the human label in error messages."""
    app = parse_app(raw_app, name=name, where=where)
    return Declaration(name=name, app=app, requires=parse_requires(raw_requires, app, where=where))


_REGISTRY: dict[str, Declaration] = {}
_REGISTRY_LOCK = threading.Lock()
on_change: Optional[Callable[[], None]] = None


def _changed() -> None:
    if on_change is not None:
        on_change()


def register(server_name: str, decl: Declaration) -> None:
    """Record which application declaration gates *server_name*."""
    with _REGISTRY_LOCK:
        _REGISTRY[server_name] = decl
    _changed()


def unregister(server_name: str) -> None:
    """Remove the declaration registered for *server_name*."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(server_name, None)
    _changed()


def lookup(server_name: str) -> Optional[Declaration]:
    """Return the declaration registered for *server_name*, if any."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(server_name)


def clear() -> None:
    """Drop every registration."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
    _changed()
