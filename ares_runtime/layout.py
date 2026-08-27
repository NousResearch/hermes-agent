"""Filesystem layout and atomic pointer persistence for installed Ares releases."""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .contracts import InstalledRuntimePointer, require_sha256
from .errors import AresRuntimeError


@dataclass(frozen=True)
class AresRuntimeLayout:
    """Paths owned by one installation-scoped Ares runtime root."""

    root: Path

    def __post_init__(self) -> None:
        if not self.root.is_absolute() or ".." in self.root.parts:
            raise AresRuntimeError("INVALID_PATH", "ares root")

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def releases_dir(self) -> Path:
        return self.root / "releases"

    @property
    def staging_dir(self) -> Path:
        return self.root / "staging"

    @property
    def transactions_dir(self) -> Path:
        return self.root / "activation-transactions"

    @property
    def receipts_dir(self) -> Path:
        return self.root / "activation-receipts"

    @property
    def rollback_snapshots_dir(self) -> Path:
        return self.root / "rollback-snapshots"

    @property
    def leases_dir(self) -> Path:
        return self.root / "runtime-leases"

    @property
    def activation_lock_path(self) -> Path:
        return self.root / "activation-lock"

    @property
    def pointer_path(self) -> Path:
        return self.root / "release-state.json"

    def release_dir(self, release_id: str) -> Path:
        if release_id.startswith("legacy-"):
            require_sha256(release_id.removeprefix("legacy-"), "legacy release id")
        else:
            require_sha256(release_id, "release id")
        return self.releases_dir / release_id

    def initialize(self) -> None:
        """Create only layout directories, retaining CandidateStore ownership."""

        for path in (
            self.root,
            self.releases_dir,
            self.staging_dir,
            self.transactions_dir,
            self.receipts_dir,
            self.rollback_snapshots_dir,
            self.leases_dir,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                raise AresRuntimeError("UNSAFE_LAYOUT_PERMISSIONS", str(path))

    def read_pointer(self) -> InstalledRuntimePointer:
        try:
            info = self.pointer_path.lstat()
        except FileNotFoundError as exc:
            raise AresRuntimeError("CURRENT_RELEASE_MISSING") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise AresRuntimeError("UNSAFE_POINTER_PATH", str(self.pointer_path))
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise AresRuntimeError("UNSAFE_POINTER_PERMISSIONS", str(self.pointer_path))
        return InstalledRuntimePointer.parse(self.pointer_path.read_bytes())

    def write_pointer_atomic(self, pointer: InstalledRuntimePointer) -> None:
        """Commit one current/previous descriptor with a same-directory rename."""

        self.initialize()
        if Path(pointer.state_root) != self.root.parent:
            raise AresRuntimeError("POINTER_STATE_ROOT_MISMATCH")
        if self.pointer_path.exists() or self.pointer_path.is_symlink():
            info = self.pointer_path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise AresRuntimeError("UNSAFE_POINTER_PATH", str(self.pointer_path))
        raw = pointer.canonical_bytes()
        fd, temporary = tempfile.mkstemp(
            dir=self.root,
            prefix=".release-state-",
            suffix=".tmp",
            text=False,
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.pointer_path)
            directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise AresRuntimeError("POINTER_COMMIT_FAILED", str(exc)) from exc
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
