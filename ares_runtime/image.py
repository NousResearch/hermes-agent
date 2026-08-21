"""Build a relocatable, immutable Ares runtime image before candidate sealing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from .errors import AresRuntimeError


RELEASE_MANIFEST_SCHEMA = "AresReleaseManifestV1"
_REJECTED_NAMES = {"direct_url.json"}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )


def _rejected(name: str) -> bool:
    return (
        name in _REJECTED_NAMES
        or name.endswith(".pth")
        or name.endswith(".egg-link")
        or name.startswith("__editable__")
        or name == "__pycache__"
        or name.endswith(".pyc")
    )


def _validate_source_tree(root: Path) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise AresRuntimeError("INVALID_RUNTIME_SOURCE", str(root))
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            if os.path.isabs(target):
                raise AresRuntimeError("RUNTIME_SOURCE_SYMLINK", str(path))
            try:
                resolved = (path.parent / target).resolve(strict=True)
            except OSError as exc:
                raise AresRuntimeError("RUNTIME_SOURCE_SYMLINK", str(path)) from exc
            if not resolved.is_relative_to(root):
                raise AresRuntimeError("RUNTIME_SOURCE_SYMLINK", str(path))
            continue
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise AresRuntimeError("INVALID_RUNTIME_SOURCE", str(path))


def _copy_tree(source: Path, destination: Path) -> None:
    _validate_source_tree(source)
    if destination.exists():
        raise AresRuntimeError("RUNTIME_DESTINATION_EXISTS", str(destination))
    shutil.copytree(
        source,
        destination,
        # The source validator permits only relative links whose resolved
        # target remains inside ``source``.  Materializing them as regular
        # files ensures the candidate archive remains compatible with the
        # existing no-symlink verified extractor and cannot retain a source
        # runtime reference.
        symlinks=False,
        ignore=lambda _directory, names: {name for name in names if _rejected(name)},
    )


_BOOTSTRAP = '''"""Release-local Ares bootstrap.  Generated before candidate sealing."""
from __future__ import annotations

import os
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime"
    ares = root / "ares"
    site_packages = runtime / "site-packages"
    python_root = runtime / "python"
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        if os.environ.get(name):
            raise RuntimeError(f"stable runtime rejected {name}")
    allowed = [str(ares), str(site_packages)]
    allowed.extend(
        entry for entry in sys.path if entry and Path(entry).resolve().is_relative_to(python_root)
    )
    sys.path[:] = allowed
    if len(sys.argv) < 2:
        raise RuntimeError("Ares runtime role required")
    role, *args = sys.argv[1:]
    if role not in {"cli", "tui", "desktop", "gateway", "backend", "identity"}:
        raise RuntimeError(f"unknown Ares runtime role: {role}")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CONTEXT_GOVERNOR_BIN"] = str(runtime / "context-governor")
    if role == "identity":
        manifest_path = root / "release-manifest.json"
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
        identity = {
            "schema": "AresRuntimeIdentityV1",
            "sealed_candidate_id": os.environ["ARES_RELEASE_ID"],
            "release_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "runtime_tree_sha256": manifest["runtime_tree_sha256"],
            "resolver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "role": os.environ.get("ARES_RUNTIME_ROLE", "identity"),
            "generation": int(os.environ["ARES_RELEASE_GENERATION"]),
        }
        print(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        return
    from hermes_cli.main import main as hermes_main

    sys.argv = ["hermes", *args]
    hermes_main()


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class RuntimeImage:
    python: Path
    site_packages: Path
    bootstrap: Path


def stage_runtime_image(
    payload: Path, *, python_root: Path, site_packages_root: Path
) -> RuntimeImage:
    """Copy the exact runtime closure without editable import metadata."""

    runtime = payload / "runtime"
    runtime.mkdir(mode=0o755)
    python = runtime / "python"
    site_packages = runtime / "site-packages"
    _copy_tree(python_root.resolve(), python)
    _copy_tree(site_packages_root.resolve(), site_packages)
    bootstrap = payload / "bootstrap" / "ares_bootstrap.py"
    bootstrap.parent.mkdir(mode=0o755)
    bootstrap.write_text(_BOOTSTRAP, encoding="utf-8")
    bootstrap.chmod(0o555)
    return RuntimeImage(python, site_packages, bootstrap)


def write_release_manifest(payload: Path) -> dict[str, object]:
    """Describe every payload byte except this self-referential manifest."""

    manifest_path = payload / "release-manifest.json"
    if manifest_path.exists():
        raise AresRuntimeError("RELEASE_MANIFEST_ALREADY_EXISTS")
    entries: list[dict[str, object]] = []
    for path in sorted(payload.rglob("*")):
        info = path.lstat()
        relative = path.relative_to(payload).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise AresRuntimeError("UNSAFE_RELEASE_OBJECT", str(path))
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise AresRuntimeError("UNSAFE_RELEASE_OBJECT", str(path))
        if _rejected(path.name):
            raise AresRuntimeError("EDITABLE_RUNTIME_METADATA", relative)
        entries.append({
            "kind": "file",
            "path": relative,
            "mode": stat.S_IMODE(info.st_mode),
            "size": info.st_size,
            "sha256": _digest(path),
        })
    runtime_tree_sha256 = hashlib.sha256(_canonical(entries)).hexdigest()
    manifest = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "runtime_tree_sha256": runtime_tree_sha256,
        "files": entries,
    }
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o444)
    return manifest


def verify_release_manifest(
    payload: Path,
    *,
    expected_manifest_sha256: str,
    expected_runtime_tree_sha256: str,
) -> dict[str, object]:
    """Recompute the sealed runtime tree from its release-local manifest."""

    manifest_path = payload / "release-manifest.json"
    try:
        info = manifest_path.lstat()
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise AresRuntimeError(
            "RELEASE_MANIFEST_UNAVAILABLE", str(manifest_path)
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AresRuntimeError("UNSAFE_RELEASE_OBJECT", str(manifest_path))
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise AresRuntimeError("RELEASE_MANIFEST_MISMATCH")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AresRuntimeError("INVALID_RELEASE_MANIFEST") from exc
    if not isinstance(manifest, dict) or _canonical(manifest) != raw:
        raise AresRuntimeError("INVALID_RELEASE_MANIFEST")
    if set(manifest) != {"schema", "runtime_tree_sha256", "files"}:
        raise AresRuntimeError("INVALID_RELEASE_MANIFEST")
    if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
        raise AresRuntimeError("INVALID_RELEASE_MANIFEST")
    if manifest.get("runtime_tree_sha256") != expected_runtime_tree_sha256:
        raise AresRuntimeError("RUNTIME_TREE_MISMATCH")
    expected = manifest.get("files")
    if not isinstance(expected, list):
        raise AresRuntimeError("INVALID_RELEASE_MANIFEST")
    actual: list[dict[str, object]] = []
    for path in sorted(payload.rglob("*")):
        if path == manifest_path:
            continue
        info = path.lstat()
        relative = path.relative_to(payload).as_posix()
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise AresRuntimeError("UNSAFE_RELEASE_OBJECT", relative)
        if _rejected(path.name):
            raise AresRuntimeError("EDITABLE_RUNTIME_METADATA", relative)
        actual.append({
            "kind": "file",
            "path": relative,
            "mode": stat.S_IMODE(info.st_mode),
            "size": info.st_size,
            "sha256": _digest(path),
        })
    if expected != actual:
        raise AresRuntimeError("RUNTIME_TREE_MISMATCH")
    if hashlib.sha256(_canonical(actual)).hexdigest() != expected_runtime_tree_sha256:
        raise AresRuntimeError("RUNTIME_TREE_MISMATCH")
    return manifest
