from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class IntegrityVerdict(str, Enum):
    ABSENT = "absent"
    EMPTY = "empty"
    VERIFIED = "verified"
    CORRUPT = "corrupt"
    BUSY = "busy"
    ENVIRONMENT_ERROR = "environment_error"
    UNSUPPORTED_OBJECT = "unsupported_object"


@dataclass(frozen=True)
class StateDBFileIdentity:
    device: int
    inode: int

    @classmethod
    def from_stat(cls, result: os.stat_result) -> "StateDBFileIdentity":
        return cls(device=int(result.st_dev), inode=int(result.st_ino))


@dataclass(frozen=True)
class StateDBIntegrityReport:
    path: Path
    verdict: IntegrityVerdict
    checked: str
    problems: tuple[str, ...] = ()
    identity: Optional[StateDBFileIdentity] = None
    may_open_writer: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "verdict": self.verdict.value,
            "checked": self.checked,
            "problems": list(self.problems),
            "identity": (
                {
                    "device": self.identity.device,
                    "inode": self.identity.inode,
                }
                if self.identity is not None
                else None
            ),
            "may_open_writer": self.may_open_writer,
        }


@dataclass(frozen=True)
class StateDBAdmissionProof:
    proof_id: str
    path: Path
    identity: StateDBFileIdentity
    report: StateDBIntegrityReport
    verified_at: float


class StateDBAdmissionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        path: Path,
        report: Optional[StateDBIntegrityReport] = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.report = report


class StateDBIntegrityError(StateDBAdmissionError):
    pass


class StateDBGenerationConflictError(StateDBAdmissionError):
    pass


class StateDBAdmissionBusyError(StateDBAdmissionError):
    pass


def canonical_state_db_path(db_path: Path | str | None = None) -> Path:
    import hermes_state

    raw = (
        Path(db_path)
        if db_path is not None
        else Path(hermes_state._default_db_path())
    )
    return raw.expanduser().resolve(strict=False)


def sqlite_read_only_uri(path: Path | str) -> str:
    """Percent-encode reserved filename characters before adding URI options."""
    return canonical_state_db_path(path).as_uri() + "?mode=ro"


def identity_from_stat(result: os.stat_result) -> StateDBFileIdentity:
    return StateDBFileIdentity.from_stat(result)


def stat_identity(path: Path) -> StateDBFileIdentity:
    return identity_from_stat(path.stat())


def anchor_identity(anchor_fd: int) -> StateDBFileIdentity:
    return identity_from_stat(os.fstat(anchor_fd))


def same_identity(
    left: StateDBFileIdentity,
    right: StateDBFileIdentity,
) -> bool:
    return left.device == right.device and left.inode == right.inode


def problem_verdict(problem: str) -> IntegrityVerdict:
    lowered = problem.lower()
    if "locked" in lowered or "busy" in lowered:
        return IntegrityVerdict.BUSY
    markers = (
        "malformed",
        "not a database",
        "rowid out of order",
        "2nd reference to page",
        "never used",
        "wrong # of entries in index",
        "fts5 read probe failed",
        "fts write",
        "corrupt",
    )
    if any(marker in lowered for marker in markers):
        return IntegrityVerdict.CORRUPT
    return IntegrityVerdict.ENVIRONMENT_ERROR


def is_repairable(report: StateDBIntegrityReport) -> bool:
    text = " ".join(report.problems).lower()
    return any(
        marker in text
        for marker in (
            "malformed database schema",
            "fts5 read probe failed",
            "fts write",
            "wrong # of entries in index",
        )
    )


def format_refusal(report: StateDBIntegrityReport) -> str:
    detail = "; ".join(report.problems[:3]) or "no diagnostic detail"
    return (
        f"state.db writer admission refused for {report.path}: "
        f"verdict={report.verdict.value}, checked={report.checked}; {detail}. "
        "No gateway writer was opened. Run `hermes doctor` and use the "
        "canonical state.db recovery path before retrying."
    )
