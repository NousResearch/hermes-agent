"""Authoritative containment boundaries for isolated cron children."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class BoundaryUnavailable(RuntimeError):
    """The host cannot provide a verifiably-owned hard boundary."""

    def __init__(self, message: str, *, boundary=None, cleanup_failed: bool = False):
        super().__init__(message)
        self.boundary = boundary
        self.cleanup_failed = cleanup_failed


@dataclass(frozen=True)
class CgroupV2Boundary:
    """Opaque identity for one uniquely-owned cgroup-v2 boundary."""

    path: Path
    expected_parent: Path
    expected_parent_inode: int
    inode: int

    def _validate(self) -> None:
        parent_stat = self.expected_parent.stat()
        stat = self.path.stat()
        if (
            self.path.parent != self.expected_parent
            or parent_stat.st_ino != self.expected_parent_inode
            or stat.st_ino != self.inode
        ):
            raise RuntimeError("cron cgroup ownership identity changed")

    def assign_and_verify(self, pid: int) -> None:
        self._validate()
        (self.path / "cgroup.procs").write_text(str(pid), encoding="ascii")
        self._validate()
        members = {
            int(value)
            for value in (self.path / "cgroup.procs").read_text(encoding="ascii").split()
        }
        if pid not in members:
            raise RuntimeError("cron child cgroup membership could not be verified")

    def prove_termination_capability(self) -> None:
        """Prove the owned boundary accepts the authoritative kill operation."""
        self._validate()
        kill_file = self.path / "cgroup.kill"
        if not kill_file.exists():
            raise RuntimeError("owned cron cgroup has no cgroup.kill")
        try:
            kill_file.write_text("1", encoding="ascii")
        except OSError as exc:
            raise RuntimeError(f"owned cron cgroup kill is not writable: {exc}") from exc
        self._validate()
        if (self.path / "cgroup.procs").read_text(encoding="ascii").strip():
            raise RuntimeError("owned cron cgroup kill probe did not leave it empty")

    def terminate(self, *, force: bool, timeout: float) -> bool:
        if not self.path.exists():
            raise RuntimeError("cron cgroup boundary disappeared before teardown")
        self._validate()
        # cgroup.kill is the only supported termination primitive. Never
        # enumerate cgroup.procs and signal individual PIDs: that recreates
        # the race this boundary exists to eliminate.
        (self.path / "cgroup.kill").write_text("1", encoding="ascii")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._validate()
            if not (self.path / "cgroup.procs").read_text(encoding="ascii").strip():
                self.path.rmdir()
                return True
            time.sleep(0.05)
        return False


def _current_cgroup_parent() -> Path:
    if sys.platform != "linux":
        raise BoundaryUnavailable("hard cron containment is unsupported on this platform")
    try:
        mount_point = None
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            before, after = line.split(" - ", 1)
            if after.split()[0] == "cgroup2":
                mount_point = Path(before.split()[4])
                break
        if mount_point is None:
            raise BoundaryUnavailable("cgroup-v2 is not mounted")
        cgroup_line = Path("/proc/self/cgroup").read_text(encoding="ascii").strip()
        relative = cgroup_line.split("::", 1)[1].lstrip("/")
        parent = mount_point / relative
        if not os.access(parent, os.W_OK):
            raise BoundaryUnavailable("writable cgroup-v2 delegation is unavailable")
        if not (parent / "cgroup.procs").exists() or not (parent / "cgroup.subtree_control").exists():
            raise BoundaryUnavailable("cgroup-v2 delegation files are unavailable")
        return parent
    except (OSError, IndexError, ValueError) as exc:
        raise BoundaryUnavailable(f"cgroup-v2 discovery failed: {exc}") from exc


def allocate_boundary(job_id: str) -> CgroupV2Boundary:
    """Create an empty, uniquely-owned cgroup before workload release.

    The parent may be writable while ``cgroup.kill`` is not usable (for
    example, when delegation exposes only a subset of cgroup files). Probe
    create, kill, remove, and recreate on the uniquely owned boundary before
    returning the real handle, so callers never report ``contained`` for a
    boundary that cannot be terminated authoritatively.
    """
    parent = _current_cgroup_parent()
    parent_inode = parent.stat().st_ino
    path = parent / f"hermes-cron-{os.getpid()}-{time.monotonic_ns()}"
    boundary = None
    try:
        path.mkdir(mode=0o700)
        try:
            inode = path.stat().st_ino
        except OSError:
            # Keep an opaque ownership handle even when identity collection
            # fails after mkdir; cleanup must not fall through to fallback.
            inode = -1
        boundary = CgroupV2Boundary(path, parent, parent_inode, inode)
        if inode == -1:
            raise BoundaryUnavailable("allocated cron cgroup identity could not be read", boundary=boundary)
        if (path / "cgroup.procs").read_text(encoding="ascii").strip():
            raise BoundaryUnavailable("allocated cron cgroup was not empty", boundary=boundary)
        # Probe the actual uniquely owned boundary, not a disposable sibling.
        try:
            boundary.prove_termination_capability()
            path.rmdir()
            path.mkdir(mode=0o700)
            boundary = CgroupV2Boundary(
                path,
                parent,
                parent_inode,
                path.stat().st_ino,
            )
            boundary.prove_termination_capability()
        except Exception as exc:
            if isinstance(exc, BoundaryUnavailable):
                raise
            raise BoundaryUnavailable(
                f"owned cgroup create/kill/remove capability unavailable: {exc}",
                boundary=boundary,
            ) from exc
        return boundary
    except BoundaryUnavailable:
        try:
            path.rmdir()
        except OSError:
            if boundary is not None:
                raise BoundaryUnavailable(
                    "allocated cron cgroup cleanup failed",
                    boundary=boundary,
                    cleanup_failed=True,
                ) from None
        raise
    except OSError as exc:
        try:
            path.rmdir()
        except OSError:
            if boundary is not None:
                raise BoundaryUnavailable(
                    f"cgroup boundary allocation failed and cleanup failed: {exc}",
                    boundary=boundary,
                    cleanup_failed=True,
                ) from exc
        raise BoundaryUnavailable(f"cgroup boundary allocation failed: {exc}") from exc
    except BaseException:
        try:
            path.rmdir()
        except OSError:
            pass
        raise
