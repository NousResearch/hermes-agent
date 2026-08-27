from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from github_pr_feedback.ci_coordinator import (
    CIAuditJob,
    GroupedCICoordinator,
)
from github_pr_feedback.ci_runner import (
    CIAuditIdentity,
    CIAuditReceipt,
    CheckState,
)

BASE_SHA = "b" * 40
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def _job(
    root: Path,
    pr_number: int,
    head_character: str,
    *,
    repository: str = "acme/widgets",
    base_sha: str = BASE_SHA,
    manifest: bytes = b"[lanes.unit]\nci_status='required'\n",
    failure_lanes: tuple[str, ...] = ("unit",),
) -> CIAuditJob:
    worktree = root / str(pr_number)
    manifest_path = worktree / "tests/manifests/test_lanes.toml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(manifest)
    return CIAuditJob(
        identity=CIAuditIdentity(repository, pr_number, base_sha, head_character * 40),
        worktree=worktree,
        failure_lanes=failure_lanes,
    )


@dataclass
class RecordingAuditRunner:
    lock: threading.Lock
    active: list[int]
    peak: list[int]
    release: threading.Event

    def run(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt:
        with self.lock:
            self.active[0] += 1
            self.peak[0] = max(self.peak[0], self.active[0])
        self.release.wait(timeout=2)
        with self.lock:
            self.active[0] -= 1
        digest = hashlib.sha256(
            (worktree / "tests/manifests/test_lanes.toml").read_bytes()
        ).hexdigest()
        return CIAuditReceipt(
            receipt_id=hashlib.sha256(
                f"{identity.repository}:{identity.pr_number}:{identity.head_sha}".encode()
            ).hexdigest(),
            identity=identity,
            manifest_digest=digest,
            status="passed",
            started_at=NOW,
            completed_at=NOW,
            actions_state=CheckState(False, True, 0),
            commands=(),
        )


def test_groups_only_identical_repo_base_manifest_and_failure_lane_fingerprint(
    tmp_path: Path,
) -> None:
    jobs = (
        _job(tmp_path, 1, "1", failure_lanes=("static", "unit")),
        _job(tmp_path, 2, "2", failure_lanes=("unit", "static")),
        _job(tmp_path, 3, "3", failure_lanes=("hygiene",)),
        _job(tmp_path, 4, "4", base_sha="c" * 40, failure_lanes=("static", "unit")),
        _job(
            tmp_path,
            5,
            "5",
            manifest=b"[lanes.integration]\nci_status='required'\n",
            failure_lanes=("static", "unit"),
        ),
        _job(
            tmp_path,
            6,
            "6",
            repository="ACME/widgets",
            failure_lanes=("static", "unit"),
        ),
    )

    groups = GroupedCICoordinator.group(jobs)

    assert tuple(tuple(job.identity.pr_number for job in group.jobs) for group in groups) == (
        (1, 2),
        (3,),
        (4,),
        (5,),
        (6,),
    )
    assert groups[0].key.failure_lane_fingerprint == hashlib.sha256(b"static\0unit").hexdigest()


def test_runs_isolated_exact_heads_with_bounded_parallelism_and_one_receipt_each(
    tmp_path: Path,
) -> None:
    jobs = tuple(_job(tmp_path, number, str(number)) for number in range(1, 5))
    lock = threading.Lock()
    active = [0]
    peak = [0]
    release = threading.Event()
    runner = RecordingAuditRunner(lock, active, peak, release)
    coordinator = GroupedCICoordinator(lambda: runner, max_parallel=2)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        outcomes = coordinator.run(jobs)
    finally:
        timer.cancel()

    assert peak[0] == 2
    assert tuple(outcome.identity for outcome in outcomes) == tuple(job.identity for job in jobs)
    assert all(outcome.error is None for outcome in outcomes)
    receipts = tuple(outcome.receipt for outcome in outcomes)
    assert all(receipt is not None for receipt in receipts)
    assert tuple(receipt.identity for receipt in receipts if receipt is not None) == tuple(
        job.identity for job in jobs
    )
    assert len({receipt.receipt_id for receipt in receipts if receipt is not None}) == 4


def test_parallelism_has_a_hard_ceiling_even_if_the_caller_requests_more(
    tmp_path: Path,
) -> None:
    jobs = tuple(_job(tmp_path, number, str(number)) for number in range(1, 7))
    lock = threading.Lock()
    active = [0]
    peak = [0]
    release = threading.Event()
    runner = RecordingAuditRunner(lock, active, peak, release)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        GroupedCICoordinator(lambda: runner, max_parallel=99).run(jobs)
    finally:
        timer.cancel()

    assert peak[0] == 4


def test_rejects_a_cross_pr_receipt_without_using_it_as_another_jobs_proof(
    tmp_path: Path,
) -> None:
    jobs = (_job(tmp_path, 1, "1"), _job(tmp_path, 2, "2"))

    class CrossBindingRunner:
        def run(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt:
            other = jobs[1].identity if identity == jobs[0].identity else identity
            return CIAuditReceipt(
                receipt_id=hashlib.sha256(str(identity.pr_number).encode()).hexdigest(),
                identity=other,
                manifest_digest=hashlib.sha256(
                    (worktree / "tests/manifests/test_lanes.toml").read_bytes()
                ).hexdigest(),
                status="passed",
                started_at=NOW,
                completed_at=NOW,
                actions_state=CheckState(False, True, 0),
                commands=(),
            )

    outcomes = GroupedCICoordinator(CrossBindingRunner, max_parallel=2).run(jobs)

    assert outcomes[0].receipt is None
    assert outcomes[0].error == "receipt_identity_mismatch"
    assert outcomes[1].receipt is not None
    assert outcomes[1].receipt.identity == jobs[1].identity


def test_shared_preparation_runs_once_per_group_but_never_returns_shared_receipts(
    tmp_path: Path,
) -> None:
    jobs = (
        _job(tmp_path, 1, "1"),
        _job(tmp_path, 2, "2"),
        _job(tmp_path, 3, "3", failure_lanes=("static",)),
    )
    prepared: list[tuple[int, ...]] = []
    runner = RecordingAuditRunner(threading.Lock(), [0], [0], threading.Event())
    runner.release.set()

    outcomes = GroupedCICoordinator(
        lambda: runner,
        max_parallel=2,
        prepare_group=lambda group: prepared.append(
            tuple(job.identity.pr_number for job in group.jobs)
        ),
    ).run(jobs)

    assert prepared == [(1, 2), (3,)]
    assert tuple(outcome.receipt.identity for outcome in outcomes if outcome.receipt) == tuple(
        job.identity for job in jobs
    )


def test_outcomes_preserve_queue_order_when_matching_groups_are_interleaved(
    tmp_path: Path,
) -> None:
    jobs = (
        _job(tmp_path, 1, "1", failure_lanes=("unit",)),
        _job(tmp_path, 2, "2", failure_lanes=("static",)),
        _job(tmp_path, 3, "3", failure_lanes=("unit",)),
    )
    runner = RecordingAuditRunner(threading.Lock(), [0], [0], threading.Event())
    runner.release.set()

    outcomes = GroupedCICoordinator(lambda: runner, max_parallel=2).run(jobs)

    assert tuple(outcome.identity.pr_number for outcome in outcomes) == (1, 2, 3)


def test_manifest_change_after_grouping_cannot_rebind_a_grouped_audit(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, 1, "1")
    runner = RecordingAuditRunner(threading.Lock(), [0], [0], threading.Event())
    runner.release.set()

    def mutate_manifest(_group) -> None:
        (job.worktree / "tests/manifests/test_lanes.toml").write_bytes(
            b"[lanes.changed]\nci_status='required'\n"
        )

    outcome = GroupedCICoordinator(
        lambda: runner,
        max_parallel=1,
        prepare_group=mutate_manifest,
    ).run((job,))[0]

    assert outcome.receipt is None
    assert outcome.error == "receipt_manifest_mismatch"
