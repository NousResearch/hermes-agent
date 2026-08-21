"""Fail-closed stable runtime resolution for every Ares entry surface."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import InstalledRuntimePointer, ReleaseReference
from .errors import AresRuntimeError
from .image import verify_release_manifest
from .layout import AresRuntimeLayout

_DEVELOPMENT_OVERRIDES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "HERMES_PYTHON_SRC_ROOT",
    "HERMES_DESKTOP_HERMES_ROOT",
    "VIRTUAL_ENV",
)


def _file_digest(path: Path) -> str:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise AresRuntimeError("UNSAFE_RUNTIME_PATH", str(path))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise AresRuntimeError("RUNTIME_IDENTITY_UNAVAILABLE", str(path)) from exc


@dataclass(frozen=True)
class ResolvedRuntime:
    pointer: InstalledRuntimePointer
    release: ReleaseReference
    release_root: Path
    python: Path
    bootstrap: Path


class AresRuntimeResolver:
    """One resolver used by CLI, TUI, Desktop backend, and gateway."""

    def __init__(self, layout: AresRuntimeLayout) -> None:
        self.layout = layout

    @staticmethod
    def _require_stable_environment(environment: Mapping[str, str]) -> None:
        for name in _DEVELOPMENT_OVERRIDES:
            if environment.get(name, "").strip():
                raise AresRuntimeError("SOURCE_SHADOWING_DETECTED", name)

    def resolve(self, environment: Mapping[str, str] | None = None) -> ResolvedRuntime:
        source = dict(os.environ if environment is None else environment)
        self._require_stable_environment(source)
        pointer = self.layout.read_pointer()
        release_root = self.layout.release_dir(pointer.current.release_id)
        try:
            info = release_root.lstat()
        except FileNotFoundError as exc:
            raise AresRuntimeError(
                "CURRENT_RELEASE_MISSING", str(release_root)
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise AresRuntimeError("UNSAFE_RELEASE_PATH", str(release_root))
        payload_root = release_root / "payload"
        verify_release_manifest(
            payload_root,
            expected_manifest_sha256=pointer.current.release_manifest_sha256,
            expected_runtime_tree_sha256=pointer.current.runtime_tree_sha256,
        )
        python = payload_root / "runtime" / "python" / "bin" / "python"
        bootstrap = payload_root / "bootstrap" / "ares_bootstrap.py"
        _file_digest(python)
        _file_digest(bootstrap)
        return ResolvedRuntime(
            pointer, pointer.current, release_root, python, bootstrap
        )

    def launch_environment(
        self, resolved: ResolvedRuntime, environment: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        source = dict(os.environ if environment is None else environment)
        self._require_stable_environment(source)
        for name in _DEVELOPMENT_OVERRIDES:
            source.pop(name, None)
        source["PYTHONDONTWRITEBYTECODE"] = "1"
        source["PYTHONNOUSERSITE"] = "1"
        source["ARES_RUNTIME_MODE"] = "stable"
        source["ARES_RELEASE_ID"] = resolved.release.release_id
        source["ARES_RELEASE_GENERATION"] = str(resolved.pointer.generation)
        return source

    def launch_argv(self, role: str, resolved: ResolvedRuntime) -> list[str]:
        if role not in {"cli", "tui", "desktop", "gateway", "backend", "identity"}:
            raise AresRuntimeError("UNKNOWN_RUNTIME_ROLE", role)
        return [
            str(resolved.python),
            "-I",
            "-S",
            "-B",
            str(resolved.bootstrap),
            role,
        ]
