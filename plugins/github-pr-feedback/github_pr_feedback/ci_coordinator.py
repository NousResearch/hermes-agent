"""Bounded grouped coordination for isolated exact-head CI audits."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from .ci_runner import CIAuditIdentity, CIAuditReceipt

_MAX_PARALLEL_CI_AUDITS = 4


@dataclass(frozen=True, slots=True)
class CIAuditJob:
    identity: CIAuditIdentity
    worktree: Path
    failure_lanes: tuple[str, ...]

    def __post_init__(self) -> None:
        resolved = Path(self.worktree).resolve()
        if not resolved.is_dir():
            raise ValueError("CI audit worktree must be an existing directory")
        lanes = tuple(
            sorted(
                {
                    lane.strip().casefold()
                    for lane in self.failure_lanes
                    if isinstance(lane, str) and lane.strip()
                }
            )
        )
        if not lanes:
            raise ValueError("failure_lanes must contain at least one lane")
        object.__setattr__(self, "worktree", resolved)
        object.__setattr__(self, "failure_lanes", lanes)


@dataclass(frozen=True, slots=True)
class CIGroupKey:
    repository: str
    base_sha: str
    manifest_digest: str
    failure_lane_fingerprint: str


@dataclass(frozen=True, slots=True)
class CIAuditGroup:
    key: CIGroupKey
    jobs: tuple[CIAuditJob, ...]


@dataclass(frozen=True, slots=True)
class CIAuditOutcome:
    identity: CIAuditIdentity
    receipt: CIAuditReceipt | None
    error: str | None


class ExactHeadAuditRunner(Protocol):
    def run(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt: ...


class GroupedCICoordinator:
    def __init__(
        self,
        runner_factory: Callable[[], ExactHeadAuditRunner],
        *,
        max_parallel: int,
        prepare_group: Callable[[CIAuditGroup], None] | None = None,
    ) -> None:
        if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer")
        self._runner_factory = runner_factory
        self._max_parallel = min(max_parallel, _MAX_PARALLEL_CI_AUDITS)
        self._prepare_group = prepare_group

    @staticmethod
    def group(jobs: Iterable[CIAuditJob]) -> tuple[CIAuditGroup, ...]:
        grouped: OrderedDict[CIGroupKey, list[CIAuditJob]] = OrderedDict()
        worktrees: set[Path] = set()
        identities: set[CIAuditIdentity] = set()
        for job in jobs:
            if not isinstance(job, CIAuditJob):
                raise TypeError("jobs must contain CIAuditJob values")
            if job.worktree in worktrees:
                raise ValueError("each CI audit requires an isolated worktree")
            if job.identity in identities:
                raise ValueError("each exact PR head may be queued only once")
            worktrees.add(job.worktree)
            identities.add(job.identity)
            manifest = job.worktree / "tests/manifests/test_lanes.toml"
            if not manifest.is_file():
                raise ValueError("CI lane manifest is unavailable")
            key = CIGroupKey(
                repository=job.identity.repository,
                base_sha=job.identity.base_sha,
                manifest_digest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                failure_lane_fingerprint=hashlib.sha256(
                    "\0".join(job.failure_lanes).encode("utf-8")
                ).hexdigest(),
            )
            grouped.setdefault(key, []).append(job)
        return tuple(
            CIAuditGroup(key=key, jobs=tuple(group_jobs)) for key, group_jobs in grouped.items()
        )

    def run(self, jobs: Iterable[CIAuditJob]) -> tuple[CIAuditOutcome, ...]:
        queued_jobs = tuple(jobs)
        groups = self.group(queued_jobs)
        outcomes: dict[CIAuditIdentity, CIAuditOutcome] = {}
        receipt_owners: dict[str, CIAuditIdentity] = {}
        for group in groups:
            if self._prepare_group is not None:
                try:
                    self._prepare_group(group)
                except Exception:  # noqa: BLE001 - one failed group must not cancel others.
                    outcomes.update(
                        {
                            job.identity: CIAuditOutcome(
                                identity=job.identity,
                                receipt=None,
                                error="shared_preparation_failed",
                            )
                            for job in group.jobs
                        }
                    )
                    continue
            with ThreadPoolExecutor(
                max_workers=min(self._max_parallel, len(group.jobs))
            ) as executor:
                group_outcomes = tuple(
                    executor.map(
                        lambda job: self._run_one(job, group.key.manifest_digest),
                        group.jobs,
                    )
                )
            for outcome in group_outcomes:
                receipt = outcome.receipt
                if receipt is not None:
                    owner = receipt_owners.get(receipt.receipt_id)
                    if owner is not None and owner != outcome.identity:
                        outcome = CIAuditOutcome(
                            identity=outcome.identity,
                            receipt=None,
                            error="receipt_id_reused",
                        )
                    else:
                        receipt_owners[receipt.receipt_id] = outcome.identity
                outcomes[outcome.identity] = outcome
        return tuple(outcomes[job.identity] for job in queued_jobs)

    def _run_one(self, job: CIAuditJob, expected_manifest_digest: str) -> CIAuditOutcome:
        try:
            receipt = self._runner_factory().run(job.identity, job.worktree)
        except Exception:  # noqa: BLE001 - preserve an outcome for every queued exact head.
            return CIAuditOutcome(job.identity, None, "audit_failed")
        if receipt.identity != job.identity:
            return CIAuditOutcome(job.identity, None, "receipt_identity_mismatch")
        if receipt.manifest_digest != expected_manifest_digest:
            return CIAuditOutcome(job.identity, None, "receipt_manifest_mismatch")
        return CIAuditOutcome(job.identity, receipt, None)
