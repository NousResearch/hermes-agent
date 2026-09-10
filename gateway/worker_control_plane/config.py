"""Constructor-injected test-only settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_PRODUCTION_DB_NAMES = {
    "hermes.db",
    "kanban.db",
    "sessions.db",
    "state.db",
}


def resolve_test_database_path(
    approved_test_root: Path, db_path: Path
) -> tuple[Path, Path]:
    """Resolve and confine a test database path without creating anything."""
    raw_root = Path(approved_test_root)
    raw_db = Path(db_path)
    if ".." in raw_root.parts or ".." in raw_db.parts:
        raise ValueError("database traversal is not allowed")
    try:
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("approved test root must already exist") from None
    if not root.is_dir():
        raise ValueError("approved test root must be a directory")
    try:
        parent = raw_db.parent.resolve(strict=False)
        resolved_db = (parent / raw_db.name).resolve(strict=False)
        relative = resolved_db.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("test database must be inside the approved test root") from None
    if relative == Path("."):
        raise ValueError("database path must name a file below the approved root")
    if ".hermes" in resolved_db.parts or resolved_db.name.lower() in _PRODUCTION_DB_NAMES:
        raise ValueError("production-like Hermes database paths are forbidden")
    return root, resolved_db


@dataclass(frozen=True)
class WorkerControlPlaneSettings:
    enabled: bool
    test_mode: bool
    db_path: Path
    approved_test_root: Path
    token_ttl_seconds: int = 300
    heartbeat_seconds: int = 30
    ack_deadline_seconds: int = 10
    lease_seconds: int = 60
    max_poll_wait_seconds: int = 0
    max_body_bytes: int = 16 * 1024
    max_stdout_bytes: int = 4096
    max_stderr_bytes: int = 4096
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.enabled or not self.test_mode:
            raise ValueError("Worker Control Plane storage requires enabled test mode")
        for name in (
            "token_ttl_seconds", "heartbeat_seconds", "ack_deadline_seconds",
            "lease_seconds", "max_body_bytes", "max_stdout_bytes",
            "max_stderr_bytes", "max_attempts",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_poll_wait_seconds != 0:
            raise ValueError("M2B-1 supports only zero-second polling")
        root, db_path = resolve_test_database_path(
            self.approved_test_root, self.db_path
        )
        object.__setattr__(self, "approved_test_root", root)
        object.__setattr__(self, "db_path", db_path)

    @classmethod
    def for_test(
        cls, db_path: Path, *, approved_test_root: Path
    ) -> "WorkerControlPlaneSettings":
        return cls(
            enabled=True,
            test_mode=True,
            db_path=db_path,
            approved_test_root=approved_test_root,
        )
