"""Hermetic release-local runtime gate for production owner operations.

This module is the only supported entry point for the local production
cutover launcher.  It verifies a canonical whole-tree manifest before loading
the launcher, requires an isolated ``-I -B`` interpreter invocation, rejects
ambient Python configuration, attests every imported module origin, and keeps
an import guard installed for lazy imports during the operation.

It performs no Cloud, database, service, Discord, or filesystem mutation
outside its immutable release.  The release packager is responsible for
building the non-editable virtual environment and writing the manifest while
the tree is still staged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import stat
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "muncho-production-owner-runtime-manifest.v1"
ATTESTATION_SCHEMA = "muncho-production-owner-runtime-attestation.v1"
MANIFEST_NAME = "production-owner-runtime-manifest.json"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_TREE_ENTRIES = 200_000
MAX_TREE_BYTES = 4 * 1024 * 1024 * 1024
TARGET_MODULE = "scripts.canary.production_cutover_owner_launcher"
CANONICAL_MODULE_NAME = "gateway.production_owner_runtime"
REQUIRED_MODULES = (
    "cryptography",
    "yaml",
    CANONICAL_MODULE_NAME,
    "gateway.canonical_writer_production_cutover",
    "gateway.operational_edge_assets",
    "gateway.operational_edge_catalog",
    "gateway.operational_edge_client",
    "gateway.operational_edge_protocol",
    "gateway.operational_edge_readiness",
    "gateway.operational_edge_service",
    "gateway.operational_edge_units",
    "gateway.production_cron_continuity_package",
    "gateway.production_cron_cutover_runtime",
    "ops.muncho.runtime.mechanical_job_rail",
    "ops.muncho.runtime.trusted_cron_collector_rail",
    "scripts.canary.full_canary_owner_launcher",
    "scripts.canary.owner_gate_caddy_cutover",
    "scripts.canary.package_production_cutover_artifacts",
    "scripts.canary.production_cutover_host_authority",
    "scripts.canary.production_cutover_initial_collector",
    "scripts.canary.production_cutover_owner_launcher",
    "scripts.canary.production_cutover_public_stager",
    "scripts.canary.production_database_recovery_gate",
    "scripts.canary.production_database_recovery_probe",
    "scripts.canary.production_os_login_metadata_migration",
    "scripts.canary.stage_production_cron_continuity",
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_ACTIVE_ATTESTATION: Mapping[str, Any] | None = None
ATTESTATION_FIELDS = frozenset({
    "schema",
    "revision",
    "manifest_sha256",
    "tree_sha256",
    "interpreter_sha256",
    "pyvenv_cfg_sha256",
    "sys_path_sha256",
    "required_modules_sha256",
    "module_origins_release_local",
    "ambient_python_environment_present",
    "secret_material_recorded",
    "secret_digest_recorded",
    "attestation_sha256",
})


class ProductionOwnerRuntimeError(RuntimeError):
    """Stable, secret-free owner-runtime failure."""


def _bind_canonical_module_identity() -> None:
    """Make ``python -m`` and canonical imports share one active gate."""

    current = sys.modules.get(__name__)
    package = sys.modules.get("gateway")
    canonical = sys.modules.get(CANONICAL_MODULE_NAME)
    exposed = getattr(package, "production_owner_runtime", None)
    if (
        __name__ not in {"__main__", CANONICAL_MODULE_NAME}
        or not isinstance(current, ModuleType)
        or current.__name__ != __name__
        or not isinstance(current.__spec__, ModuleSpec)
        or current.__spec__.name != CANONICAL_MODULE_NAME
        or current.__spec__.parent != "gateway"
        or not isinstance(package, ModuleType)
        or (canonical is not None and canonical is not current)
        or (exposed is not None and exposed is not current)
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_module_identity_conflict"
        )
    sys.modules[CANONICAL_MODULE_NAME] = current
    setattr(package, "production_owner_runtime", current)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_json_invalid"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_owner_runtime_attestation(
    value: Any,
    *,
    revision: str,
) -> Mapping[str, Any]:
    """Validate the portable, secret-free owner-runtime plan binding."""

    if (
        _REVISION.fullmatch(revision or "") is None
        or not isinstance(value, Mapping)
        or set(value) != ATTESTATION_FIELDS
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_attestation_invalid"
        )
    raw = copy.deepcopy(dict(value))
    unsigned = {
        name: item
        for name, item in raw.items()
        if name != "attestation_sha256"
    }
    if (
        raw.get("schema") != ATTESTATION_SCHEMA
        or raw.get("revision") != revision
        or any(
            _SHA256.fullmatch(str(raw.get(name))) is None
            for name in (
                "manifest_sha256",
                "tree_sha256",
                "interpreter_sha256",
                "pyvenv_cfg_sha256",
                "sys_path_sha256",
                "required_modules_sha256",
                "attestation_sha256",
            )
        )
        or raw.get("module_origins_release_local") is not True
        or raw.get("ambient_python_environment_present") is not False
        or raw.get("secret_material_recorded") is not False
        or raw.get("secret_digest_recorded") is not False
        or raw.get("attestation_sha256")
        != _sha256_bytes(_canonical(unsigned))
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_attestation_invalid"
        )
    return raw


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_unavailable"
        ) from exc
    return digest.hexdigest()


def _runtime_root() -> Path:
    try:
        executable = Path(sys.executable)
        root = executable.parents[2]
    except (IndexError, TypeError, ValueError) as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_interpreter_invalid"
        ) from exc
    if not executable.is_absolute() or not root.is_absolute():
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_interpreter_invalid"
        )
    return root


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _stable_regular(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_manifest_invalid"
            )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except ProductionOwnerRuntimeError:
        raise
    except OSError as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_changed"
        )
    if len(raw) != before.st_size:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_changed"
        )
    return raw


def _decode_manifest(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            if name in result:
                raise ProductionOwnerRuntimeError(
                    "production_owner_runtime_manifest_invalid"
                )
            result[name] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except ProductionOwnerRuntimeError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_invalid"
        )
    return value


def _tree_entry(path: Path, root: Path) -> tuple[dict[str, Any], int]:
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_unavailable"
        ) from exc
    relative = path.relative_to(root).as_posix()
    if not relative or _CONTROL.search(relative) is not None:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_invalid"
        )
    common = {
        "path": relative,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "uid": int(item.st_uid),
        "gid": int(item.st_gid),
    }
    if stat.S_ISDIR(item.st_mode):
        if stat.S_IMODE(item.st_mode) & 0o222:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_tree_writable"
            )
        return {**common, "kind": "directory"}, 0
    if stat.S_ISREG(item.st_mode):
        if item.st_nlink != 1 or stat.S_IMODE(item.st_mode) & 0o222:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_tree_writable"
            )
        return {
            **common,
            "kind": "file",
            "size": int(item.st_size),
            "sha256": _hash_file(path),
        }, int(item.st_size)
    if stat.S_ISLNK(item.st_mode):
        try:
            target = os.readlink(path)
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_tree_invalid"
            ) from exc
        if (
            not target
            or _CONTROL.search(target) is not None
            or not _within(resolved, root)
        ):
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_tree_invalid"
            )
        return {**common, "kind": "symlink", "target": target}, 0
    raise ProductionOwnerRuntimeError("production_owner_runtime_tree_invalid")


def collect_tree_entries(root: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        root_state = os.lstat(root)
    except OSError as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(root_state.st_mode)
        or stat.S_ISLNK(root_state.st_mode)
        or stat.S_IMODE(root_state.st_mode) & 0o222
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_writable"
        )
    entries: list[dict[str, Any]] = []
    total = 0
    for current, directories, files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directories.sort()
        files.sort()
        for name in (*directories, *files):
            path = Path(current) / name
            if path == root / MANIFEST_NAME:
                continue
            entry, size = _tree_entry(path, root)
            entries.append(entry)
            total += size
            if len(entries) > MAX_TREE_ENTRIES or total > MAX_TREE_BYTES:
                raise ProductionOwnerRuntimeError(
                    "production_owner_runtime_tree_oversized"
                )
    entries.sort(key=lambda item: item["path"])
    if len({item["path"] for item in entries}) != len(entries):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_invalid"
        )
    return entries, total


def _module_origin(module: ModuleType) -> Path:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin in {None, "built-in", "frozen"}:
        origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not os.path.isabs(origin):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_module_origin_invalid"
        )
    return Path(origin)


def _origin_record(module: ModuleType, *, root: Path) -> dict[str, Any]:
    origin = _module_origin(module)
    try:
        if origin.resolve(strict=True) != origin or not _within(origin, root):
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_module_origin_invalid"
            )
        item = os.lstat(origin)
    except ProductionOwnerRuntimeError:
        raise
    except OSError as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_module_origin_invalid"
        ) from exc
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or stat.S_IMODE(item.st_mode) & 0o222
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_module_origin_invalid"
        )
    return {
        "origin": str(origin),
        "relative_path": origin.relative_to(root).as_posix(),
        "sha256": _hash_file(origin),
    }


def _runtime_identity(root: Path) -> tuple[Path, Path, Path]:
    interpreter = root / "venv/bin/python"
    pyvenv = root / "venv/pyvenv.cfg"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = root / "venv/lib" / version / "site-packages"
    return interpreter, pyvenv, site_packages


def _validate_pyvenv_cfg(path: Path, *, root: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_pyvenv_invalid"
        ) from exc
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if " = " not in line:
            continue
        name, item = line.split(" = ", 1)
        if name in values:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_pyvenv_invalid"
            )
        values[name] = item
    try:
        home = Path(values["home"])
        executable = Path(values.get("executable", str(home / "python3.11")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_pyvenv_invalid"
        ) from exc
    python_root = root / "python"
    if (
        not home.is_absolute()
        or not executable.is_absolute()
        or not _within(home, python_root)
        or not _within(executable, python_root)
        or values.get("include-system-site-packages", "").casefold()
        != "false"
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_pyvenv_invalid"
        )


def _validate_site_packages(path: Path, *, root: Path) -> None:
    if list(path.glob("*.pth")) or list(path.glob("*.egg-link")):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_dynamic_site_path_forbidden"
        )
    for direct_url in sorted(path.glob("*.dist-info/direct_url.json")):
        try:
            value = json.loads(direct_url.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_direct_url_invalid"
            ) from exc
        directory = value.get("dir_info") if isinstance(value, Mapping) else None
        url = value.get("url") if isinstance(value, Mapping) else None
        if isinstance(directory, Mapping) and directory.get("editable") is True:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_editable_install_forbidden"
            )
        if isinstance(url, str) and url.startswith("file://"):
            local_path = Path(url.removeprefix("file://"))
            if not local_path.is_absolute() or not _within(local_path, root):
                raise ProductionOwnerRuntimeError(
                    "production_owner_runtime_direct_url_invalid"
                )


def build_manifest(revision: str) -> Mapping[str, Any]:
    """Build the canonical manifest after the staged tree has been sealed."""

    if _REVISION.fullmatch(revision or "") is None:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_revision_invalid"
        )
    root = _runtime_root()
    interpreter, pyvenv, site_packages = _runtime_identity(root)
    try:
        executable_realpath = Path(sys.executable).resolve(strict=True)
        interpreter_realpath = interpreter.resolve(strict=True)
        pyvenv_state = os.lstat(pyvenv)
        site_state = os.lstat(site_packages)
    except OSError as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_interpreter_invalid"
        ) from exc
    if (
        executable_realpath != interpreter
        or interpreter_realpath != interpreter
        or not stat.S_ISREG(pyvenv_state.st_mode)
        or stat.S_ISLNK(pyvenv_state.st_mode)
        or pyvenv_state.st_nlink != 1
        or stat.S_IMODE(pyvenv_state.st_mode) & 0o222
        or not stat.S_ISDIR(site_state.st_mode)
        or stat.S_ISLNK(site_state.st_mode)
        or stat.S_IMODE(site_state.st_mode) & 0o222
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_interpreter_invalid"
        )
    _validate_pyvenv_cfg(pyvenv, root=root)
    _validate_site_packages(site_packages, root=root)
    sys_path = list(sys.path)
    if (
        not sys_path
        or any(
            not isinstance(item, str)
            or not item
            or not os.path.isabs(item)
            or not _within(Path(item), root)
            for item in sys_path
        )
        or len(sys_path) != len(set(sys_path))
        or str(site_packages) not in sys_path
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_sys_path_invalid"
        )
    modules = {
        name: importlib.import_module(name)
        for name in REQUIRED_MODULES
    }
    required = {
        name: _origin_record(module, root=root)
        for name, module in sorted(modules.items())
    }
    entries, total = collect_tree_entries(root)
    tree_sha256 = _sha256_bytes(_canonical(entries))
    interpreter_state = os.lstat(interpreter)
    unsigned = {
        "schema": MANIFEST_SCHEMA,
        "revision": revision,
        "artifact_root": str(root),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "interpreter": {
            "path": str(interpreter),
            "realpath": str(interpreter_realpath),
            "mode": f"{stat.S_IMODE(interpreter_state.st_mode):04o}",
            "size": interpreter_state.st_size,
            "sha256": _hash_file(interpreter),
        },
        "pyvenv_cfg": {
            "path": str(pyvenv),
            "mode": f"{stat.S_IMODE(pyvenv_state.st_mode):04o}",
            "size": pyvenv_state.st_size,
            "sha256": _hash_file(pyvenv),
        },
        "site_packages": str(site_packages),
        "sys_path": sys_path,
        "required_modules": required,
        "entries": entries,
        "entry_count": len(entries),
        "tree_bytes": total,
        "tree_sha256": tree_sha256,
        "root_uid": os.lstat(root).st_uid,
        "root_gid": os.lstat(root).st_gid,
        "root_mode": f"{stat.S_IMODE(os.lstat(root).st_mode):04o}",
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "manifest_sha256": _sha256_bytes(_canonical(unsigned)),
    }


_MANIFEST_FIELDS = frozenset({
    "schema",
    "revision",
    "artifact_root",
    "python_version",
    "interpreter",
    "pyvenv_cfg",
    "site_packages",
    "sys_path",
    "required_modules",
    "entries",
    "entry_count",
    "tree_bytes",
    "tree_sha256",
    "root_uid",
    "root_gid",
    "root_mode",
    "secret_material_recorded",
    "secret_digest_recorded",
    "manifest_sha256",
})


def _validate_manifest(value: Mapping[str, Any], revision: str) -> Mapping[str, Any]:
    unsigned = {
        name: item for name, item in value.items() if name != "manifest_sha256"
    }
    root = _runtime_root()
    interpreter, pyvenv, site_packages = _runtime_identity(root)
    entries = value.get("entries")
    required = value.get("required_modules")
    if (
        set(value) != _MANIFEST_FIELDS
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("revision") != revision
        or value.get("artifact_root") != str(root)
        or value.get("python_version")
        != (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        or value.get("site_packages") != str(site_packages)
        or value.get("sys_path") != list(sys.path)
        or not isinstance(entries, list)
        or value.get("entry_count") != len(entries)
        or type(value.get("tree_bytes")) is not int
        or value.get("tree_bytes", -1) < 0
        or value.get("tree_sha256") != _sha256_bytes(_canonical(entries))
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("manifest_sha256") != _sha256_bytes(_canonical(unsigned))
        or not isinstance(required, Mapping)
        or set(required) != set(REQUIRED_MODULES)
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_invalid"
        )
    try:
        root_state = os.lstat(root)
        interpreter_value = value["interpreter"]
        pyvenv_value = value["pyvenv_cfg"]
    except (KeyError, TypeError, OSError) as exc:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_invalid"
        ) from exc
    if (
        value.get("root_uid") != root_state.st_uid
        or value.get("root_gid") != root_state.st_gid
        or value.get("root_mode") != f"{stat.S_IMODE(root_state.st_mode):04o}"
        or stat.S_IMODE(root_state.st_mode) & 0o222
        or not isinstance(interpreter_value, Mapping)
        or set(interpreter_value) != {"path", "realpath", "mode", "size", "sha256"}
        or interpreter_value.get("path") != str(interpreter)
        or interpreter_value.get("realpath") != str(interpreter)
        or not isinstance(pyvenv_value, Mapping)
        or set(pyvenv_value) != {"path", "mode", "size", "sha256"}
        or pyvenv_value.get("path") != str(pyvenv)
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_manifest_invalid"
        )
    _validate_pyvenv_cfg(pyvenv, root=root)
    _validate_site_packages(site_packages, root=root)
    observed_entries, observed_bytes = collect_tree_entries(root)
    if (
        observed_entries != entries
        or observed_bytes != value["tree_bytes"]
        or _hash_file(interpreter) != interpreter_value.get("sha256")
        or _hash_file(pyvenv) != pyvenv_value.get("sha256")
        or os.lstat(interpreter).st_size != interpreter_value.get("size")
        or os.lstat(pyvenv).st_size != pyvenv_value.get("size")
        or f"{stat.S_IMODE(os.lstat(interpreter).st_mode):04o}"
        != interpreter_value.get("mode")
        or f"{stat.S_IMODE(os.lstat(pyvenv).st_mode):04o}"
        != pyvenv_value.get("mode")
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_tree_changed"
        )
    return copy.deepcopy(dict(value))


def _validate_invocation(manifest: Mapping[str, Any]) -> None:
    flags = sys.flags
    if (
        any(name.startswith("PYTHON") for name in os.environ)
        or flags.isolated != 1
        or flags.dont_write_bytecode != 1
        or flags.no_user_site != 1
        or flags.ignore_environment != 1
        or getattr(flags, "safe_path", False) is not True
        or flags.no_site != 0
        or Path(sys.executable) != Path(manifest["interpreter"]["path"])
        or Path(sys.executable).resolve(strict=True) != Path(sys.executable)
    ):
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_invocation_invalid"
        )


class _ManifestImportGuard(MetaPathFinder):
    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self._root = Path(manifest["artifact_root"])
        self._entries = {
            item["path"]: item
            for item in manifest["entries"]
            if isinstance(item, Mapping)
        }
        self._delegates = tuple(sys.meta_path)

    def _validate_path(self, value: str) -> None:
        path = Path(value)
        try:
            if (
                not path.is_absolute()
                or path.resolve(strict=True) != path
                or not _within(path, self._root)
            ):
                raise ProductionOwnerRuntimeError(
                    "production_owner_runtime_module_origin_invalid"
                )
            relative = path.relative_to(self._root).as_posix()
            entry = self._entries.get(relative)
            item = os.lstat(path)
        except ProductionOwnerRuntimeError:
            raise
        except OSError as exc:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_module_origin_invalid"
            ) from exc
        if (
            not isinstance(entry, Mapping)
            or entry.get("kind") not in {"file", "directory"}
            or stat.S_IMODE(item.st_mode) & 0o222
        ):
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_module_origin_invalid"
            )

    def validate_spec(self, spec: ModuleSpec | None) -> None:
        if spec is None:
            return
        origin = spec.origin
        if origin not in {None, "built-in", "frozen"}:
            if not isinstance(origin, str):
                raise ProductionOwnerRuntimeError(
                    "production_owner_runtime_module_origin_invalid"
                )
            self._validate_path(origin)
        locations = spec.submodule_search_locations
        if locations is not None:
            for location in locations:
                self._validate_path(str(location))

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        for finder in self._delegates:
            find_spec = getattr(finder, "find_spec", None)
            if not callable(find_spec):
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                self.validate_spec(spec)
                return spec
        return None


def _audit_loaded_modules(guard: _ManifestImportGuard) -> None:
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        spec = getattr(module, "__spec__", None)
        if isinstance(spec, ModuleSpec):
            guard.validate_spec(spec)


def verify_runtime(revision: str) -> Mapping[str, Any]:
    if _REVISION.fullmatch(revision or "") is None:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_revision_invalid"
        )
    manifest_path = _runtime_root() / MANIFEST_NAME
    manifest = _validate_manifest(
        _decode_manifest(
            _stable_regular(manifest_path, maximum=MAX_MANIFEST_BYTES)
        ),
        revision,
    )
    _validate_invocation(manifest)
    modules = {
        name: importlib.import_module(name)
        for name in REQUIRED_MODULES
    }
    observed_required = {
        name: _origin_record(module, root=Path(manifest["artifact_root"]))
        for name, module in sorted(modules.items())
    }
    if observed_required != manifest["required_modules"]:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_module_origin_changed"
        )
    guard = _ManifestImportGuard(manifest)
    _audit_loaded_modules(guard)
    sys.meta_path.insert(0, guard)
    unsigned = {
        "schema": ATTESTATION_SCHEMA,
        "revision": revision,
        "manifest_sha256": manifest["manifest_sha256"],
        "tree_sha256": manifest["tree_sha256"],
        "interpreter_sha256": manifest["interpreter"]["sha256"],
        "pyvenv_cfg_sha256": manifest["pyvenv_cfg"]["sha256"],
        "sys_path_sha256": _sha256_bytes(_canonical(manifest["sys_path"])),
        "required_modules_sha256": _sha256_bytes(
            _canonical(manifest["required_modules"])
        ),
        "module_origins_release_local": True,
        "ambient_python_environment_present": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "attestation_sha256": _sha256_bytes(_canonical(unsigned)),
    }


def _set_active_attestation(value: Mapping[str, Any]) -> None:
    global _ACTIVE_ATTESTATION
    if _ACTIVE_ATTESTATION is not None and _ACTIVE_ATTESTATION != value:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_attestation_conflict"
        )
    _ACTIVE_ATTESTATION = copy.deepcopy(dict(value))


def require_active_owner_runtime(revision: str) -> Mapping[str, Any]:
    value = _ACTIVE_ATTESTATION
    try:
        validated = validate_owner_runtime_attestation(
            value,
            revision=revision,
        )
    except ProductionOwnerRuntimeError:
        raise ProductionOwnerRuntimeError(
            "production_owner_runtime_not_active"
        ) from None
    return validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and enter the sealed production owner runtime"
    )
    parser.add_argument("action", choices=("manifest", "attest", "run"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _bind_canonical_module_identity()
    arguments = _parser().parse_args(argv)
    if arguments.action == "manifest":
        if arguments.launcher_args:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_argv_invalid"
            )
        print(_canonical(build_manifest(arguments.revision)).decode("ascii"))
        return 0
    attestation = verify_runtime(arguments.revision)
    if arguments.action == "attest":
        if arguments.launcher_args:
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_argv_invalid"
            )
        print(_canonical(attestation).decode("ascii"))
        return 0
    launcher_args = list(arguments.launcher_args)
    if launcher_args[:1] == ["--"]:
        launcher_args = launcher_args[1:]
    _set_active_attestation(attestation)
    launcher = importlib.import_module(TARGET_MODULE)
    try:
        result = int(launcher.main(launcher_args))
        guard = sys.meta_path[0]
        if not isinstance(guard, _ManifestImportGuard):
            raise ProductionOwnerRuntimeError(
                "production_owner_runtime_import_guard_changed"
            )
        _audit_loaded_modules(guard)
        return result
    finally:
        # Keep the attestation active through all launcher cleanup/finally
        # blocks; the process exits immediately after this return.
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProductionOwnerRuntimeError:
        print(
            '{"error_code":"production_owner_runtime_failed","ok":false}',
            file=sys.stderr,
        )
        raise SystemExit(2) from None


__all__ = [
    "ATTESTATION_FIELDS",
    "ATTESTATION_SCHEMA",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "ProductionOwnerRuntimeError",
    "REQUIRED_MODULES",
    "build_manifest",
    "collect_tree_entries",
    "main",
    "require_active_owner_runtime",
    "validate_owner_runtime_attestation",
    "verify_runtime",
]
