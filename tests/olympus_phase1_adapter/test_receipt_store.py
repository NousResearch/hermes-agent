from __future__ import annotations

import asyncio
import errno
import fcntl
import multiprocessing
import os
import stat
import threading
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from olympus_phase1_adapter import receipt_store as store_module
from olympus_phase1_adapter.adapter import (
    IndeterminateReceiptUnavailable,
    SealedReceiptUnavailable,
    execute_operation,
)
from olympus_phase1_adapter.contracts import (
    ContractValidationError,
    Operation,
    canonical_bytes,
    canonical_digest,
    parse_operation,
    strict_loads,
    validate_receipt,
)
from olympus_phase1_adapter.receipt_store import (
    OwnerHandle,
    PersistenceError,
    ReceiptStore,
    StoreIndeterminate,
    StoreIndeterminateUnavailable,
    StorePreInvokeUnavailable,
    open_root_set,
)

from conftest import (
    operation_bytes,
    operation_value,
    valid_transports,
)


def _claim_owner(
    roots: dict[str, Path],
    envelope: bytes,
) -> tuple[store_module.RootSet, OwnerHandle]:
    operation = parse_operation(envelope)
    capabilities = open_root_set(
        repository_root=roots["repository"],
        receipt_root=roots["receipt"],
        evidence_root=roots["evidence"],
    )
    try:
        result = ReceiptStore(capabilities).claim(
            operation, fresh_preflight=lambda: None
        )
    except BaseException:
        capabilities.close()
        raise
    assert result.response is None and result.owner is not None
    return capabilities, result.owner


def _recover(
    roots: dict[str, Path],
    envelope: bytes,
) -> bytes:
    operation = parse_operation(envelope)
    with open_root_set(
        repository_root=roots["repository"],
        receipt_root=roots["receipt"],
        evidence_root=roots["evidence"],
    ) as capabilities:
        result = ReceiptStore(capabilities).claim(
            operation,
            fresh_preflight=lambda: pytest.fail(
                "existing key invoked fresh preflight"
            ),
        )
        assert result.owner is None and result.response is not None
        return result.response


def _key_directory(roots: dict[str, Path], operation: Operation) -> Path:
    return (
        roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )


def _snapshot_tree(root: Path) -> dict[str, tuple[str, int, int, int, bytes | None]]:
    snapshot: dict[str, tuple[str, int, int, int, bytes | None]] = {}
    for path in sorted((root, *root.rglob("*"))):
        entry = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        kind = (
            "symlink"
            if stat.S_ISLNK(entry.st_mode)
            else "directory"
            if stat.S_ISDIR(entry.st_mode)
            else "file"
            if stat.S_ISREG(entry.st_mode)
            else "other"
        )
        raw = path.read_bytes() if kind == "file" else None
        snapshot[relative] = (
            kind,
            stat.S_IMODE(entry.st_mode),
            entry.st_ino,
            entry.st_nlink,
            raw,
        )
    return snapshot


def _execute_with_objects(roots: dict[str, Path], envelope: bytes) -> bytes:
    return execute_operation(
        envelope,
        repository_root=roots["repository"],
        receipt_root=roots["receipt"],
        evidence_root=roots["evidence"],
        pair_a=object(),
        pair_b=object(),
    )


@pytest.mark.parametrize("existing_rejection", [False, True])
def test_t08_clean_preclaim_orphan_recovers_without_invocation(
    phase1_roots: dict[str, Path],
    existing_rejection: bool,
) -> None:
    envelope = operation_bytes()
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        if existing_rejection:
            owner.append(
                "PREINVOKE_REJECTED",
                reason_code="CANCELLED_BEFORE_INVOCATION_CLAIM",
            )
    finally:
        owner.close()
        capabilities.close()
    receipt = strict_loads(_recover(phase1_roots, envelope))
    assert receipt["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert receipt["reason_codes"] == [
        (
            "CANCELLED_BEFORE_INVOCATION_CLAIM"
            if existing_rejection
            else "RECOVERED_BEFORE_INVOCATION_CLAIM"
        )
    ]
    operation = parse_operation(envelope)
    record_names = sorted(
        path.name
        for path in (_key_directory(phase1_roots, operation) / "records").iterdir()
    )
    assert sum("PREINVOKE_REJECTED" in name for name in record_names) == 1
    assert record_names[-1].endswith("RECEIPT_FINALIZED.json")


@pytest.mark.parametrize("evidence_kind", ["final", "staging"])
def test_t08_existing_preinvoke_rejection_with_evidence_is_not_mutated(
    phase1_roots: dict[str, Path],
    evidence_kind: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        owner.append(
            "PREINVOKE_REJECTED",
            reason_code="CANCELLED_BEFORE_INVOCATION_CLAIM",
        )
        destination = owner.evidence_destination
    finally:
        owner.close()
        capabilities.close()
    evidence_path = (
        destination
        if evidence_kind == "final"
        else destination.parent / f".{destination.name}.adversarial.staging"
    )
    evidence_path.mkdir(mode=0o700)
    before_receipt = _snapshot_tree(phase1_roots["receipt"])
    before_evidence = _snapshot_tree(phase1_roots["evidence"])

    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        _execute_with_objects(phase1_roots, envelope)

    assert caught.value.__cause__ is None
    assert caught.value.idempotency_key_digest == operation.idempotency_key_digest
    assert caught.value.operation_digest == operation.operation_digest
    assert _snapshot_tree(phase1_roots["receipt"]) == before_receipt
    assert _snapshot_tree(phase1_roots["evidence"]) == before_evidence


@pytest.mark.parametrize("evidence_kind", ["final", "staging", "scan-error"])
def test_t08_different_operation_preclaim_evidence_is_indeterminate(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    evidence_kind: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    destination = owner.evidence_destination
    owner.close()
    capabilities.close()

    scan_tripped = False
    if evidence_kind == "final":
        destination.mkdir(mode=0o700)
    elif evidence_kind == "staging":
        destination.with_name(
            f".{destination.name}.different-operation.staging"
        ).mkdir(mode=0o700)
    else:
        original_scan = store_module._directory_has_matching_name

        def fail_matching_scan(path: Path, predicate: object) -> bool:
            nonlocal scan_tripped
            if path == destination.parent:
                scan_tripped = True
                raise PersistenceError("injected evidence scan failure")
            return original_scan(path, predicate)  # type: ignore[arg-type]

        monkeypatch.setattr(
            store_module, "_directory_has_matching_name", fail_matching_scan
        )

    before_receipt = _snapshot_tree(phase1_roots["receipt"])
    before_evidence = _snapshot_tree(phase1_roots["evidence"])
    different = operation_bytes(
        correlation_id="corr-phase1-different-preclaim-operation"
    )
    current = parse_operation(different)

    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        _execute_with_objects(phase1_roots, different)

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caught.value.idempotency_key_digest == operation.idempotency_key_digest
    assert caught.value.operation_digest == current.operation_digest
    assert scan_tripped is (evidence_kind == "scan-error")
    assert _snapshot_tree(phase1_roots["receipt"]) == before_receipt
    assert _snapshot_tree(phase1_roots["evidence"]) == before_evidence


@pytest.mark.parametrize(
    ("scenario", "expected_reason"),
    [
        ("unsafe-receipt", "UNSAFE_RECEIPT_ROOT"),
        ("overlap", "ROOTS_OVERLAP"),
        ("unsafe-evidence", "UNSAFE_EVIDENCE_ROOT"),
        ("unsafe-repository", "UNSAFE_REPOSITORY_ROOT"),
    ],
)
def test_t13_root_rejection_reason_precedence_is_global_not_argument_order(
    phase1_roots: dict[str, Path],
    scenario: str,
    expected_reason: str,
) -> None:
    repository = phase1_roots["repository"]
    receipt_root = phase1_roots["receipt"]
    evidence_root = phase1_roots["evidence"]
    evidence_argument = evidence_root

    repository.chmod(0o777)
    if scenario == "unsafe-receipt":
        receipt_root.chmod(0o755)
        evidence_argument = receipt_root
    elif scenario == "overlap":
        evidence_argument = receipt_root
    elif scenario == "unsafe-evidence":
        evidence_root.chmod(0o755)

    raw = execute_operation(
        operation_bytes(),
        repository_root=repository,
        receipt_root=receipt_root,
        evidence_root=evidence_argument,
        pair_a=object(),
        pair_b=object(),
    )
    rejection = strict_loads(raw)
    assert rejection["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert rejection["durability"] == "TRANSIENT"
    assert rejection["reason_codes"] == [expected_reason]


@pytest.mark.parametrize(
    ("alias_root", "target_root"),
    [
        ("evidence", "receipt"),
        ("repository", "receipt"),
        ("evidence", "repository"),
        ("repository", "evidence"),
    ],
)
def test_t13_symlink_alias_overlap_precedes_lower_root_safety(
    phase1_roots: dict[str, Path],
    tmp_path: Path,
    alias_root: str,
    target_root: str,
) -> None:
    alias = tmp_path / f"{alias_root}-alias-to-{target_root}"
    alias.symlink_to(phase1_roots[target_root], target_is_directory=True)
    repository_argument = (
        alias if alias_root == "repository" else phase1_roots["repository"]
    )
    evidence_argument = (
        alias if alias_root == "evidence" else phase1_roots["evidence"]
    )
    before_receipt = _snapshot_tree(phase1_roots["receipt"])
    before_evidence = _snapshot_tree(phase1_roots["evidence"])

    raw = execute_operation(
        operation_bytes(),
        repository_root=repository_argument,
        receipt_root=phase1_roots["receipt"],
        evidence_root=evidence_argument,
        pair_a=object(),
        pair_b=object(),
    )

    rejection = strict_loads(raw)
    assert rejection["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert rejection["durability"] == "TRANSIENT"
    assert rejection["reason_codes"] == ["ROOTS_OVERLAP"]
    assert _snapshot_tree(phase1_roots["receipt"]) == before_receipt
    assert _snapshot_tree(phase1_roots["evidence"]) == before_evidence


@pytest.mark.parametrize("hold_unlinked_inode", [False, True])
def test_t13_established_store_never_replaces_missing_global_lock(
    phase1_roots: dict[str, Path],
    hold_unlinked_inode: bool,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    owner.close()
    capabilities.close()
    lock_path = phase1_roots["receipt"] / ".phase1-store.lock"
    old_fd: int | None = None
    if hold_unlinked_inode:
        old_fd = os.open(lock_path, os.O_RDWR)
        fcntl.flock(old_fd, fcntl.LOCK_EX)
    lock_path.unlink()
    before = _snapshot_tree(phase1_roots["receipt"])

    try:
        with pytest.raises(IndeterminateReceiptUnavailable) as caught:
            _execute_with_objects(phase1_roots, envelope)
        assert caught.value.idempotency_key_digest == operation.idempotency_key_digest
        assert not lock_path.exists()
        assert _snapshot_tree(phase1_roots["receipt"]) == before
    finally:
        if old_fd is not None:
            fcntl.flock(old_fd, fcntl.LOCK_UN)
            os.close(old_fd)


@pytest.mark.parametrize("lock_adversary", ["wrong-mode", "hardlink"])
def test_t13_established_store_global_lock_anomaly_is_indeterminate(
    phase1_roots: dict[str, Path],
    lock_adversary: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    owner.close()
    capabilities.close()
    lock_path = phase1_roots["receipt"] / ".phase1-store.lock"
    if lock_adversary == "wrong-mode":
        lock_path.chmod(0o640)
    else:
        os.link(lock_path, phase1_roots["evidence"] / "global-lock-alias")
    before = _snapshot_tree(phase1_roots["receipt"])

    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        _execute_with_objects(phase1_roots, envelope)

    assert caught.value.idempotency_key_digest == operation.idempotency_key_digest
    assert caught.value.operation_digest == operation.operation_digest
    assert _snapshot_tree(phase1_roots["receipt"]) == before


@pytest.mark.parametrize("lock_adversary", ["unlink", "replacement", "wrong-mode"])
def test_t13_t19_active_owner_never_masks_global_lock_anomaly(
    phase1_roots: dict[str, Path],
    lock_adversary: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    lock_path = phase1_roots["receipt"] / ".phase1-store.lock"
    try:
        if lock_adversary == "unlink":
            lock_path.unlink()
        elif lock_adversary == "replacement":
            lock_path.unlink()
            lock_path.write_bytes(b"")
            lock_path.chmod(0o600)
        else:
            lock_path.chmod(0o640)
        before = _snapshot_tree(phase1_roots["receipt"])

        with pytest.raises(IndeterminateReceiptUnavailable) as caught:
            _execute_with_objects(phase1_roots, envelope)

        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True
        assert caught.value.idempotency_key_digest == operation.idempotency_key_digest
        assert caught.value.operation_digest == operation.operation_digest
        assert _snapshot_tree(phase1_roots["receipt"]) == before
    finally:
        owner.close()
        capabilities.close()


@pytest.mark.parametrize(
    "receipt_adversary",
    ["absent", "truncated", "noncanonical", "wrong-mode", "hardlink"],
)
def test_t22_final_record_proves_sealed_receipt_when_bytes_are_unavailable(
    phase1_roots: dict[str, Path],
    receipt_adversary: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        predecessor = owner.append(
            "PREINVOKE_REJECTED",
            reason_code="CANCELLED_BEFORE_INVOCATION_CLAIM",
        )
        raw = owner.seal(
            receipt_state="REJECTED_PRE_INVOKE",
            reason_code="CANCELLED_BEFORE_INVOCATION_CLAIM",
        )
    finally:
        owner.close()
        capabilities.close()
    receipt = strict_loads(raw)
    key = _key_directory(phase1_roots, operation)
    records = key / "records"
    receipt_path = key / "receipt.json"
    key.chmod(0o700)
    records.chmod(0o700)
    receipt_path.chmod(0o600)
    if receipt_adversary == "absent":
        receipt_path.unlink()
    elif receipt_adversary == "truncated":
        receipt_path.write_bytes(raw[:19])
    elif receipt_adversary == "noncanonical":
        receipt_path.write_bytes(raw + b"\n")
    elif receipt_adversary == "wrong-mode":
        receipt_path.chmod(0o640)
    else:
        os.link(receipt_path, phase1_roots["evidence"] / "receipt-alias")
    final_name = max(path.name for path in records.iterdir())
    assert final_name.endswith("RECEIPT_FINALIZED.json")
    assert predecessor["record_digest"] != receipt["receipt_digest"]
    before = _snapshot_tree(phase1_roots["receipt"])

    with pytest.raises(SealedReceiptUnavailable) as caught:
        _execute_with_objects(phase1_roots, envelope)

    error = caught.value
    assert error.__cause__ is None
    assert error.receipt_digest == receipt["receipt_digest"]
    assert error.receipt_state == "REJECTED_PRE_INVOKE"
    assert error.idempotency_key_digest == operation.idempotency_key_digest
    assert error.operation_digest == operation.operation_digest
    assert _snapshot_tree(phase1_roots["receipt"]) == before


@pytest.mark.parametrize(
    "prefix",
    ["INVOCATION_CLAIMED", "ENGINE_EVIDENCE_VERIFIED", "INDETERMINATE_NO_RETRY"],
)
def test_t09_consumed_attempt_orphan_recovers_without_second_invocation(
    phase1_roots: dict[str, Path],
    prefix: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        owner.append("INVOCATION_CLAIMED")
        if prefix in {"ENGINE_EVIDENCE_VERIFIED", "INDETERMINATE_NO_RETRY"}:
            owner.append(
                "ENGINE_EVIDENCE_VERIFIED",
                reason_code="PHASE0_TERMINAL",
                phase0_terminal_report_digest="1" * 64,
                phase0_evidence_manifest_digest="2" * 64,
                phase0_evidence_directory_name=(
                    f"phase0-{operation.idempotency_key_digest}"
                ),
            )
        if prefix == "INDETERMINATE_NO_RETRY":
            owner.append(
                "INDETERMINATE_NO_RETRY",
                reason_code="ENGINE_EXCEPTION_WITHOUT_SEALED_RECEIPT",
            )
    finally:
        owner.close()
        capabilities.close()
    receipt = strict_loads(_recover(phase1_roots, envelope))
    assert receipt["receipt_state"] == "INDETERMINATE_NO_RETRY"
    assert receipt["reason_codes"] == [
        (
            "ENGINE_EXCEPTION_WITHOUT_SEALED_RECEIPT"
            if prefix == "INDETERMINATE_NO_RETRY"
            else "RECOVERED_AFTER_INVOCATION_CLAIM"
        )
    ]
    assert receipt["phase0_terminal_report"] is None
    record_names = sorted(
        path.name
        for path in (_key_directory(phase1_roots, operation) / "records").iterdir()
    )
    assert sum("INDETERMINATE_NO_RETRY" in name for name in record_names) == 1


@pytest.mark.parametrize("evidence_kind", ["final", "staging"])
def test_t09_consumed_recovery_preserves_matching_evidence_opaquely(
    phase1_roots: dict[str, Path],
    evidence_kind: str,
) -> None:
    envelope = operation_bytes()
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        owner.append("INVOCATION_CLAIMED")
        destination = owner.evidence_destination
    finally:
        owner.close()
        capabilities.close()
    evidence_path = (
        destination
        if evidence_kind == "final"
        else destination.parent / f".{destination.name}.preserved.staging"
    )
    evidence_path.mkdir(mode=0o700)
    artifact = evidence_path / "opaque.bin"
    artifact.write_bytes(b"opaque Phase 0 evidence\x00")
    artifact.chmod(0o600)
    before = _snapshot_tree(phase1_roots["evidence"])

    receipt = strict_loads(_recover(phase1_roots, envelope))

    assert receipt["receipt_state"] == "INDETERMINATE_NO_RETRY"
    assert receipt["reason_codes"] == ["RECOVERED_AFTER_INVOCATION_CLAIM"]
    assert _snapshot_tree(phase1_roots["evidence"]) == before


def _cross_process_owner(
    repository: str,
    receipt: str,
    evidence: str,
    envelope: bytes,
    ready: Any,
    release: Any,
    queue: Any,
) -> None:
    os.umask(0o077)
    roots = {
        "repository": Path(repository),
        "receipt": Path(receipt),
        "evidence": Path(evidence),
    }
    capabilities, owner = _claim_owner(roots, envelope)
    try:
        anchor = str(owner.records[0]["record_digest"])
        owner.append("INVOCATION_CLAIMED")
        queue.put(("owner", anchor))
        ready.set()
        if not release.wait(20):
            raise RuntimeError("owner release timed out")
    finally:
        owner.close()
        capabilities.close()


def _cross_process_contender(
    repository: str,
    receipt: str,
    evidence: str,
    envelope: bytes,
    ready: Any,
    queue: Any,
    label: str,
) -> None:
    os.umask(0o077)
    if not ready.wait(20):
        raise RuntimeError("owner readiness timed out")
    roots = {
        "repository": Path(repository),
        "receipt": Path(receipt),
        "evidence": Path(evidence),
    }
    operation = parse_operation(envelope)
    with open_root_set(
        repository_root=roots["repository"],
        receipt_root=roots["receipt"],
        evidence_root=roots["evidence"],
    ) as capabilities:
        result = ReceiptStore(capabilities).claim(
            operation,
            fresh_preflight=lambda: (_ for _ in ()).throw(
                RuntimeError("active key ran preflight")
            ),
        )
        if result.response is None:
            raise RuntimeError("contender unexpectedly became owner")
        queue.put((label, result.response))


def test_t07_real_spawned_process_contention_uses_sequence_one_anchor(
    phase1_roots: dict[str, Path],
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    queue = context.Queue()
    same = operation_bytes()
    different = canonical_bytes(
        operation_value(correlation_id="corr-phase1-process-other")
    )
    arguments = tuple(str(phase1_roots[name]) for name in ("repository", "receipt", "evidence"))
    owner = context.Process(
        target=_cross_process_owner,
        args=(*arguments, same, ready, release, queue),
    )
    owner.start()
    assert ready.wait(20)
    contenders = [
        context.Process(
            target=_cross_process_contender,
            args=(*arguments, envelope, ready, queue, label),
        )
        for envelope, label in ((same, "same"), (different, "different"))
    ]
    for process in contenders:
        process.start()
    results: dict[str, Any] = {}
    try:
        for _ in range(3):
            label, value = queue.get(timeout=20)
            results[label] = value
    except Empty as exc:
        raise AssertionError("spawned contention result timed out") from exc
    finally:
        release.set()
    for process in contenders:
        process.join(20)
        assert process.exitcode == 0
    owner.join(20)
    assert owner.exitcode == 0
    anchor = results["owner"]
    same_receipt = strict_loads(results["same"])
    different_receipt = strict_loads(results["different"])
    assert same_receipt["receipt_state"] == "CONFLICT_IN_PROGRESS"
    assert different_receipt["receipt_state"] == "IDEMPOTENCY_CONFLICT"
    assert same_receipt["ownership_record_digest"] == anchor
    assert different_receipt["ownership_record_digest"] == anchor
    assert same_receipt["bound_operation_digest"] == parse_operation(same).operation_digest
    assert different_receipt["bound_operation_digest"] == parse_operation(same).operation_digest


@pytest.mark.parametrize("adversary", ["unexpected", "torn-record", "hardlink"])
def test_t13_unclassifiable_existing_key_is_never_mutated(
    phase1_roots: dict[str, Path],
    adversary: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    owner.close()
    capabilities.close()
    key = _key_directory(phase1_roots, operation)
    if adversary == "unexpected":
        unexpected = key / "unexpected"
        unexpected.write_bytes(b"adversarial")
        unexpected.chmod(0o600)
    elif adversary == "torn-record":
        first = key / "records" / "000001-OWNERSHIP_ACQUIRED.json"
        first.write_bytes(first.read_bytes() + b"\n")
        first.chmod(0o600)
    else:
        os.link(key / "lock", key / "lock-alias")
    before = {
        path.relative_to(key).as_posix(): (
            path.read_bytes() if path.is_file() else None
        )
        for path in key.rglob("*")
    }
    with pytest.raises(StoreIndeterminate):
        _recover(phase1_roots, envelope)
    after = {
        path.relative_to(key).as_posix(): (
            path.read_bytes() if path.is_file() else None
        )
        for path in key.rglob("*")
    }
    assert after == before


@pytest.mark.parametrize("adversary", ["torn-later-record", "unexpected-entry"])
def test_t13_different_operation_never_masks_existing_store_corruption(
    phase1_roots: dict[str, Path],
    adversary: str,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        owner.append("INVOCATION_CLAIMED")
    finally:
        owner.close()
        capabilities.close()
    key = _key_directory(phase1_roots, operation)
    if adversary == "torn-later-record":
        second = key / "records" / "000002-INVOCATION_CLAIMED.json"
        second.write_bytes(second.read_bytes() + b"\n")
    else:
        unexpected = key / "unexpected"
        unexpected.write_bytes(b"adversarial")
        unexpected.chmod(0o600)
    before = _snapshot_tree(phase1_roots["receipt"])
    different = operation_bytes(correlation_id="corr-phase1-different-operation")

    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        _execute_with_objects(phase1_roots, different)

    current = parse_operation(different)
    assert caught.value.__cause__ is None
    assert caught.value.idempotency_key_digest == current.idempotency_key_digest
    assert caught.value.operation_digest == current.operation_digest
    assert _snapshot_tree(phase1_roots["receipt"]) == before


def test_t13_existing_key_noncontention_flock_failure_is_indeterminate(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    owner.close()
    capabilities.close()
    key_lock = _key_directory(phase1_roots, operation) / "lock"
    key_identity = (key_lock.stat().st_dev, key_lock.stat().st_ino)
    original_flock = store_module.fcntl.flock

    def fail_key_flock(fd: int, flags: int) -> None:
        entry = os.fstat(fd)
        if (
            (entry.st_dev, entry.st_ino) == key_identity
            and flags & fcntl.LOCK_EX
            and flags & fcntl.LOCK_NB
        ):
            raise OSError(errno.EIO, "injected existing-key flock failure")
        original_flock(fd, flags)

    monkeypatch.setattr(store_module.fcntl, "flock", fail_key_flock)
    before = _snapshot_tree(phase1_roots["receipt"])
    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        _execute_with_objects(phase1_roots, envelope)
    assert caught.value.idempotency_key_digest == operation.idempotency_key_digest
    assert caught.value.operation_digest == operation.operation_digest
    assert _snapshot_tree(phase1_roots["receipt"]) == before


@pytest.mark.parametrize(
    "mutation",
    [
        {"durability": "SEALED"},
        {"automatic_engine_retry_permitted": True},
        {"bound_operation_digest": "f" * 64},
        {"reason_codes": ["ACTIVE_OWNER", "PHASE0_TERMINAL"]},
    ],
)
def test_t13_receipt_cross_field_adversaries_fail_schema_or_semantics(
    mutation: dict[str, object],
) -> None:
    receipt = strict_loads(
        store_module.transient_conflict(
            parse_operation(operation_bytes()),
            bound_operation_digest=parse_operation(
                operation_bytes()
            ).operation_digest,
            ownership_record_digest="a" * 64,
        )
    )
    receipt.update(mutation)
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = canonical_digest(
        "phase1-receipt", body
    )
    with pytest.raises(ContractValidationError):
        validate_receipt(receipt)


def test_t13_publication_revalidation_rejects_target_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = tmp_path / "key"
    records = key / "records"
    key.mkdir(mode=0o700)
    records.mkdir(mode=0o700)
    raw = b'{"bounded":"publication"}'
    original_stat = store_module.os.stat
    original_open = store_module.os.open
    original_write = store_module.os.write
    original_close = store_module.os.close
    original_rename = store_module.os.rename
    tripped = False

    def swap_named_target(
        path: object,
        *args: object,
        **kwargs: object,
    ):
        nonlocal tripped
        parent_fd = kwargs.get("dir_fd")
        if path == "target.json" and isinstance(parent_fd, int) and not tripped:
            tripped = True
            original_rename(
                "target.json",
                "target.swapped",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            replacement_fd = original_open(
                "target.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                remaining = memoryview(raw)
                while remaining:
                    written = original_write(replacement_fd, remaining)
                    assert written > 0
                    remaining = remaining[written:]
            finally:
                original_close(replacement_fd)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "stat", swap_named_target)
    attempt = store_module._PublicationAttempt()
    with pytest.raises(PersistenceError, match="revalidation"):
        store_module._publish_no_replace(
            key_directory=key,
            target_parent=records,
            target_name="target.json",
            raw=raw,
            attempt=attempt,
        )

    assert tripped
    assert attempt.final_revalidated is False
    assert (records / "target.json").read_bytes() == raw
    assert (records / "target.swapped").read_bytes() == raw
    assert (records / "target.json").stat().st_ino != (
        records / "target.swapped"
    ).stat().st_ino


def test_t13_file_append_during_read_revalidation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(b'{"sequence":1}')
    path.chmod(0o600)
    original_stat = store_module.os.stat
    original_open = store_module.os.open
    original_write = store_module.os.write
    original_close = store_module.os.close
    tripped = False

    def append_before_named_revalidation(
        value: object,
        *args: object,
        **kwargs: object,
    ):
        nonlocal tripped
        if value == path and not tripped:
            tripped = True
            append_fd = original_open(path, os.O_WRONLY | os.O_APPEND)
            try:
                assert original_write(append_fd, b"\n") == 1
            finally:
                original_close(append_fd)
        return original_stat(value, *args, **kwargs)

    monkeypatch.setattr(
        store_module.os, "stat", append_before_named_revalidation
    )
    with pytest.raises(PersistenceError, match="identity changed"):
        store_module._read_regular(path, modes={0o600}, maximum=1024)
    assert tripped


@pytest.mark.parametrize(
    "site",
    [
        "global-lock",
        "global-lock-file-fsync",
        "global-lock-parent-fsync",
        "receipt-v1-mkdir",
        "receipt-v1-child-fsync",
        "receipt-v1-parent-fsync",
        "receipt-shard-mkdir",
        "receipt-shard-child-fsync",
        "receipt-shard-parent-fsync",
        "evidence-v1-mkdir",
        "evidence-v1-child-fsync",
        "evidence-v1-parent-fsync",
        "evidence-shard-mkdir",
        "evidence-shard-child-fsync",
        "evidence-shard-parent-fsync",
        "key-mkdir",
        "key-child-fsync",
        "key-parent-fsync",
        "key-lock",
        "key-lock-file-fsync",
        "key-lock-parent-fsync",
        "records-mkdir",
        "records-child-fsync",
        "records-parent-fsync",
        "ownership-publication",
    ],
)
def test_t14_fresh_claim_site_specific_faults_never_invoke(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    site: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    operation = parse_operation(operation_bytes())
    receipt_v1 = phase1_roots["receipt"] / "v1"
    receipt_shard = receipt_v1 / operation.idempotency_key_digest[:2]
    key = receipt_shard / operation.idempotency_key_digest
    evidence_v1 = phase1_roots["evidence"] / "v1"
    evidence_shard = evidence_v1 / operation.idempotency_key_digest[:2]
    global_lock = phase1_roots["receipt"] / ".phase1-store.lock"
    key_lock = key / "lock"
    records = key / "records"
    original_create_directory = store_module._create_directory
    original_fsync_directory = store_module._fsync_directory
    original_create_global = store_module._create_lock_file
    original_create_key = store_module._create_permanent_lock
    original_publish = store_module._publish_no_replace
    original_fsync = store_module.os.fsync
    original_fstat = store_module.os.fstat
    tripped = False

    def maybe_fail_directory(parent: Path, name: str) -> Path:
        nonlocal tripped
        matches = {
            "receipt-v1-mkdir": parent == phase1_roots["receipt"] and name == "v1",
            "receipt-shard-mkdir": parent == receipt_v1
            and name == operation.idempotency_key_digest[:2],
            "evidence-v1-mkdir": parent == phase1_roots["evidence"] and name == "v1",
            "evidence-shard-mkdir": parent == evidence_v1
            and name == operation.idempotency_key_digest[:2],
            "key-mkdir": parent == receipt_shard
            and name == operation.idempotency_key_digest,
            "records-mkdir": parent == key and name == "records",
        }.get(site, False)
        if matches and not tripped:
            tripped = True
            raise PersistenceError(f"injected site fault: {site}")
        return original_create_directory(parent, name)

    def maybe_fail_directory_fsync(path: Path, modes: set[int]) -> None:
        nonlocal tripped
        matches = {
            "receipt-v1-child-fsync": path == receipt_v1,
            "receipt-v1-parent-fsync": path == phase1_roots["receipt"]
            and receipt_v1.exists(),
            "receipt-shard-child-fsync": path == receipt_shard,
            "receipt-shard-parent-fsync": path == receipt_v1
            and receipt_shard.exists(),
            "evidence-v1-child-fsync": path == evidence_v1,
            "evidence-v1-parent-fsync": path == phase1_roots["evidence"]
            and evidence_v1.exists(),
            "evidence-shard-child-fsync": path == evidence_shard,
            "evidence-shard-parent-fsync": path == evidence_v1
            and evidence_shard.exists(),
            "key-child-fsync": path == key,
            "key-parent-fsync": path == receipt_shard and key.exists(),
            "records-child-fsync": path == records,
            "records-parent-fsync": path == key and records.exists(),
        }.get(site, False)
        if matches and not tripped:
            tripped = True
            raise PersistenceError(f"injected directory fsync fault: {site}")
        original_fsync_directory(path, modes)

    def maybe_fail_global(*args: object, **kwargs: object) -> int:
        nonlocal tripped
        if site == "global-lock" and not tripped:
            tripped = True
            raise PersistenceError("injected global-lock site fault")
        return original_create_global(*args, **kwargs)

    def maybe_fail_key_lock(*args: object, **kwargs: object) -> int:
        nonlocal tripped
        if site == "key-lock" and not tripped:
            tripped = True
            raise PersistenceError("injected key-lock site fault")
        return original_create_key(*args, **kwargs)

    def maybe_fail_ownership_publish(**kwargs: object) -> None:
        nonlocal tripped
        if (
            site == "ownership-publication"
            and str(kwargs["target_name"]).endswith("OWNERSHIP_ACQUIRED.json")
            and not tripped
        ):
            tripped = True
            raise PersistenceError("injected ownership publication site fault")
        original_publish(**kwargs)

    def fd_matches(fd: int, path: Path) -> bool:
        try:
            held = original_fstat(fd)
            named = path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)

    def maybe_fail_lock_fsync(fd: int) -> None:
        nonlocal tripped
        target = {
            "global-lock-file-fsync": global_lock,
            "global-lock-parent-fsync": phase1_roots["receipt"],
            "key-lock-file-fsync": key_lock,
            "key-lock-parent-fsync": key,
        }.get(site)
        if (
            target is not None
            and target.exists()
            and fd_matches(fd, target)
            and not tripped
        ):
            tripped = True
            raise OSError(errno.EIO, f"injected lock fsync fault: {site}")
        original_fsync(fd)

    monkeypatch.setattr(store_module, "_create_directory", maybe_fail_directory)
    monkeypatch.setattr(
        store_module, "_fsync_directory", maybe_fail_directory_fsync
    )
    monkeypatch.setattr(store_module, "_create_lock_file", maybe_fail_global)
    monkeypatch.setattr(
        store_module, "_create_permanent_lock", maybe_fail_key_lock
    )
    monkeypatch.setattr(
        store_module, "_publish_no_replace", maybe_fail_ownership_publish
    )
    monkeypatch.setattr(store_module.os, "fsync", maybe_fail_lock_fsync)

    receipt = strict_loads(
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )
    )
    assert tripped
    assert receipt["durability"] == "TRANSIENT"
    assert receipt["reason_codes"] == ["PERSISTENCE_CORRUPTION"]
    assert pair_a.call_count == pair_b.call_count == 0


@pytest.mark.parametrize(
    ("directory_kind", "boundary"),
    [
        (directory_kind, boundary)
        for directory_kind in (
            "receipt-v1",
            "receipt-shard",
            "evidence-v1",
            "evidence-shard",
        )
        for boundary in ("child-fsync", "parent-fsync")
    ],
)
def test_t14_retry_refsyncs_interrupted_existing_directory_before_ownership(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    directory_kind: str,
    boundary: str,
) -> None:
    operation = parse_operation(operation_bytes())
    receipt_v1 = phase1_roots["receipt"] / "v1"
    receipt_shard = receipt_v1 / operation.idempotency_key_digest[:2]
    evidence_v1 = phase1_roots["evidence"] / "v1"
    evidence_shard = evidence_v1 / operation.idempotency_key_digest[:2]
    target, parent = {
        "receipt-v1": (receipt_v1, phase1_roots["receipt"]),
        "receipt-shard": (receipt_shard, receipt_v1),
        "evidence-v1": (evidence_v1, phase1_roots["evidence"]),
        "evidence-shard": (evidence_shard, evidence_v1),
    }[directory_kind]
    key = receipt_shard / operation.idempotency_key_digest
    original_fsync_directory = store_module._fsync_directory
    child_seen = False
    tripped = False

    def interrupt_directory_fsync(path: Path, modes: set[int]) -> None:
        nonlocal child_seen, tripped
        if path == target:
            child_seen = True
            if boundary == "child-fsync" and not tripped:
                tripped = True
                raise PersistenceError(
                    f"injected interrupted directory fsync: {directory_kind}"
                )
        if (
            path == parent
            and child_seen
            and boundary == "parent-fsync"
            and not tripped
        ):
            tripped = True
            raise PersistenceError(
                f"injected interrupted parent fsync: {directory_kind}"
            )
        original_fsync_directory(path, modes)

    monkeypatch.setattr(
        store_module, "_fsync_directory", interrupt_directory_fsync
    )
    with open_root_set(
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
    ) as capabilities:
        with pytest.raises(PersistenceError):
            ReceiptStore(capabilities).claim(
                operation, fresh_preflight=lambda: None
            )

    assert tripped
    assert target.is_dir()
    assert not key.exists()

    monkeypatch.undo()
    original_fsync_directory = store_module._fsync_directory
    original_publish = store_module._publish_no_replace
    events: list[str] = []
    retry_child_seen = False

    def observe_directory_fsync(path: Path, modes: set[int]) -> None:
        nonlocal retry_child_seen
        original_fsync_directory(path, modes)
        if path == target:
            retry_child_seen = True
            events.append("existing-child-refsync")
        elif path == parent and retry_child_seen:
            events.append("existing-parent-refsync")

    def observe_publication(**kwargs: object) -> None:
        if str(kwargs["target_name"]).endswith("OWNERSHIP_ACQUIRED.json"):
            events.append("ownership-publication")
        original_publish(**kwargs)

    monkeypatch.setattr(
        store_module, "_fsync_directory", observe_directory_fsync
    )
    monkeypatch.setattr(store_module, "_publish_no_replace", observe_publication)
    with open_root_set(
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
    ) as capabilities:
        result = ReceiptStore(capabilities).claim(
            operation, fresh_preflight=lambda: None
        )
        assert result.response is None and result.owner is not None
        result.owner.close()

    assert events.index("existing-child-refsync") < events.index(
        "existing-parent-refsync"
    )
    assert events.index("existing-parent-refsync") < events.index(
        "ownership-publication"
    )
    record_names = sorted(path.name for path in (key / "records").iterdir())
    assert record_names == ["000001-OWNERSHIP_ACQUIRED.json"]


@pytest.mark.parametrize(
    "boundary",
    ["mkdir", "write", "fsync", "link", "unlink", "flock"],
)
def test_t14_topology_and_publication_faults_fail_before_invocation(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])

    def fail(*_args: object, **_kwargs: object):
        raise OSError(errno.EIO, f"injected {boundary} failure")

    if boundary == "mkdir":
        monkeypatch.setattr(store_module.os, "mkdir", fail)
    elif boundary == "write":
        monkeypatch.setattr(store_module.os, "write", fail)
    elif boundary == "fsync":
        monkeypatch.setattr(store_module.os, "fsync", fail)
    elif boundary == "link":
        monkeypatch.setattr(store_module.os, "link", fail)
    elif boundary == "unlink":
        monkeypatch.setattr(store_module.os, "unlink", fail)
    else:
        monkeypatch.setattr(store_module.fcntl, "flock", fail)
    if boundary == "unlink":
        with pytest.raises(IndeterminateReceiptUnavailable) as caught:
            execute_operation(
                operation_bytes(),
                repository_root=phase1_roots["repository"],
                receipt_root=phase1_roots["receipt"],
                evidence_root=phase1_roots["evidence"],
                pair_a=pair_a,
                pair_b=pair_b,
            )
        assert caught.value.__cause__ is None
        assert pair_a.call_count == pair_b.call_count == 0
        return
    receipt = strict_loads(
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )
    )
    assert receipt["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert receipt["durability"] == "TRANSIENT"
    assert receipt["reason_codes"] == ["PERSISTENCE_CORRUPTION"]
    assert pair_a.call_count == pair_b.call_count == 0


@pytest.mark.parametrize("target_kind", ["ownership", "receipt"])
def test_t14_postlink_unlink_failure_preserves_phase_classification(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    linked_targets: dict[str, str] = {}
    tripped = False

    def remember_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_link(source, target, *args, **kwargs)
        if isinstance(source, str) and source.startswith(".phase1-stage-"):
            linked_targets[source] = target

    def fail_unlink_once(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        target = linked_targets.get(name, "")
        matches = (
            target_kind == "ownership" and target.endswith("OWNERSHIP_ACQUIRED.json")
        ) or (target_kind == "receipt" and target == "receipt.json")
        if matches and not tripped:
            tripped = True
            raise OSError(errno.EIO, "injected post-link unlink failure")
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", remember_link)
    monkeypatch.setattr(store_module.os, "unlink", fail_unlink_once)
    if target_kind == "ownership":
        with pytest.raises(IndeterminateReceiptUnavailable) as caught:
            execute_operation(
                operation_bytes(),
                repository_root=phase1_roots["repository"],
                receipt_root=phase1_roots["receipt"],
                evidence_root=phase1_roots["evidence"],
                pair_a=pair_a,
                pair_b=pair_b,
            )
        assert caught.value.__cause__ is None
        assert pair_a.call_count == pair_b.call_count == 0
        assert list(phase1_roots["receipt"].rglob(".phase1-stage-*"))
        return

    first = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    assert strict_loads(first)["receipt_state"] == "ENGINE_TERMINAL"
    assert tripped
    assert pair_a.call_count == pair_b.call_count == 1
    assert list(phase1_roots["receipt"].rglob(".phase1-stage-*"))

    replay = execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
    )
    assert replay == first
    assert pair_a.call_count == pair_b.call_count == 1
    assert not list(phase1_roots["receipt"].rglob(".phase1-stage-*"))


@pytest.mark.parametrize("boundary", ["link", "unlink"])
@pytest.mark.parametrize("control_flow", [False, True])
def test_t14_publication_side_effect_then_raise_is_at_most_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    control_flow: bool,
) -> None:
    key = tmp_path / "key"
    records = key / "records"
    key.mkdir(mode=0o700)
    records.mkdir(mode=0o700)
    raw = b'{"side_effect":"then-raise"}'
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    primary: BaseException = (
        KeyboardInterrupt(f"{boundary}-after-side-effect")
        if control_flow
        else OSError(errno.EIO, f"{boundary}-after-side-effect")
    )
    counts = {"link": 0, "unlink": 0}
    tripped = False

    def link_then_maybe_raise(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        if source.startswith(".phase1-stage-") and target == "target.json":
            counts["link"] += 1
            original_link(source, target, *args, **kwargs)
            if boundary == "link" and not tripped:
                tripped = True
                raise primary
            return
        original_link(source, target, *args, **kwargs)

    def unlink_then_maybe_raise(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        if name.startswith(".phase1-stage-"):
            counts["unlink"] += 1
            original_unlink(name, *args, **kwargs)
            if boundary == "unlink" and not tripped:
                tripped = True
                raise primary
            return
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", link_then_maybe_raise)
    monkeypatch.setattr(store_module.os, "unlink", unlink_then_maybe_raise)
    attempt = store_module._PublicationAttempt()
    if control_flow:
        with pytest.raises(type(primary)) as caught:
            store_module._publish_no_replace(
                key_directory=key,
                target_parent=records,
                target_name="target.json",
                raw=raw,
                attempt=attempt,
            )
        assert caught.value is primary
    else:
        store_module._publish_no_replace(
            key_directory=key,
            target_parent=records,
            target_name="target.json",
            raw=raw,
            attempt=attempt,
        )

    assert tripped
    assert counts == {"link": 1, "unlink": 1}
    assert attempt.complete
    assert (records / "target.json").read_bytes() == raw
    assert list(key.glob(".phase1-stage-*")) == []


@pytest.mark.parametrize("control_flow", [False, True])
def test_t14_publication_close_fault_preserves_primary_and_closes_every_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_flow: bool,
) -> None:
    key = tmp_path / "key"
    records = key / "records"
    key.mkdir(mode=0o700)
    records.mkdir(mode=0o700)
    original_open = store_module.os.open
    original_close = store_module.os.close
    original_fstat = store_module.os.fstat
    opened: list[int] = []
    close_fault_tripped = False
    primary: BaseException = (
        KeyboardInterrupt("publication-primary")
        if control_flow
        else PersistenceError("publication-primary")
    )

    def remember_open(*args: object, **kwargs: object) -> int:
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail_write(*_args: object, **_kwargs: object) -> int:
        raise primary

    def close_then_raise_once(fd: int) -> None:
        nonlocal close_fault_tripped
        original_close(fd)
        if fd in opened and not close_fault_tripped:
            close_fault_tripped = True
            raise OSError(errno.EIO, "injected publication close failure")

    monkeypatch.setattr(store_module.os, "open", remember_open)
    monkeypatch.setattr(store_module.os, "write", fail_write)
    monkeypatch.setattr(store_module.os, "close", close_then_raise_once)
    with pytest.raises(type(primary)) as caught:
        store_module._publish_no_replace(
            key_directory=key,
            target_parent=records,
            target_name="target.json",
            raw=b"publication-primary",
        )

    assert caught.value is primary
    assert close_fault_tripped
    assert len(opened) == 3
    for fd in opened:
        with pytest.raises(OSError) as closed:
            original_fstat(fd)
        assert closed.value.errno == errno.EBADF


def test_t14_committed_incomplete_claim_record_stops_same_call_mutation(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    linked_targets: dict[str, str] = {}
    tripped = False

    def remember_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_link(source, target, *args, **kwargs)
        if source.startswith(".phase1-stage-"):
            linked_targets[source] = target

    def fail_claim_stage_unlink(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        target = linked_targets.get(name, "")
        if target.endswith("INVOCATION_CLAIMED.json") and not tripped:
            tripped = True
            raise OSError(errno.EIO, "injected committed claim cleanup failure")
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", remember_link)
    monkeypatch.setattr(store_module.os, "unlink", fail_claim_stage_unlink)
    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )

    assert tripped
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert pair_a.call_count == pair_b.call_count == 0
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    record_names = sorted(path.name for path in (key / "records").iterdir())
    assert any(name.endswith("OWNERSHIP_ACQUIRED.json") for name in record_names)
    assert any(name.endswith("INVOCATION_CLAIMED.json") for name in record_names)
    assert not any("PREINVOKE_REJECTED" in name for name in record_names)
    assert not any("INDETERMINATE_NO_RETRY" in name for name in record_names)
    assert not any("RECEIPT_FINALIZED" in name for name in record_names)
    assert not (key / "receipt.json").exists()


def _publication_control_flow_objects() -> list[BaseException]:
    return [
        KeyboardInterrupt("post-link"),
        SystemExit(79),
        asyncio.CancelledError("post-link"),
    ]


@pytest.mark.parametrize("injected", _publication_control_flow_objects())
def test_t23_postlink_ownership_control_flow_is_identical_and_recoverable(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    linked_targets: dict[str, str] = {}
    tripped = False

    def remember_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_link(source, target, *args, **kwargs)
        if isinstance(source, str) and source.startswith(".phase1-stage-"):
            linked_targets[source] = target

    def interrupt_after_link(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        target = linked_targets.get(name, "")
        if target.endswith("OWNERSHIP_ACQUIRED.json") and not tripped:
            tripped = True
            raise injected
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", remember_link)
    monkeypatch.setattr(store_module.os, "unlink", interrupt_after_link)
    with pytest.raises(type(injected)) as caught:
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )
    assert caught.value is injected
    assert tripped
    assert pair_a.call_count == pair_b.call_count == 0
    monkeypatch.undo()
    with pytest.raises(IndeterminateReceiptUnavailable):
        _execute_with_objects(phase1_roots, operation_bytes())


@pytest.mark.parametrize("injected", _publication_control_flow_objects())
def test_t23_postlink_receipt_control_flow_preserves_sealed_outcome(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    linked_targets: dict[str, str] = {}
    tripped = False

    def remember_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_link(source, target, *args, **kwargs)
        if isinstance(source, str) and source.startswith(".phase1-stage-"):
            linked_targets[source] = target

    def interrupt_after_link(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        if linked_targets.get(name) == "receipt.json" and not tripped:
            tripped = True
            raise injected
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", remember_link)
    monkeypatch.setattr(store_module.os, "unlink", interrupt_after_link)
    with pytest.raises(type(injected)) as caught:
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )
    assert caught.value is injected
    assert tripped
    assert pair_a.call_count == pair_b.call_count == 1
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    sealed_raw = (key / "receipt.json").read_bytes()
    assert list(key.rglob(".phase1-stage-*"))
    monkeypatch.undo()
    replay = _execute_with_objects(phase1_roots, operation_bytes())
    assert replay == sealed_raw
    assert list(key.rglob(".phase1-stage-*")) == []
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize("injected", _publication_control_flow_objects())
def test_t23_postlink_final_record_control_flow_repairs_and_replays_exact_bytes(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    linked_targets: dict[str, str] = {}
    tripped = False

    def remember_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_link(source, target, *args, **kwargs)
        if source.startswith(".phase1-stage-"):
            linked_targets[source] = target

    def interrupt_after_link(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        target = linked_targets.get(name, "")
        if target.endswith("RECEIPT_FINALIZED.json") and not tripped:
            tripped = True
            raise injected
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", remember_link)
    monkeypatch.setattr(store_module.os, "unlink", interrupt_after_link)
    with pytest.raises(type(injected)) as caught:
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )
    assert caught.value is injected
    assert tripped
    assert pair_a.call_count == pair_b.call_count == 1
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    sealed_raw = (key / "receipt.json").read_bytes()
    assert list(key.rglob(".phase1-stage-*"))

    monkeypatch.undo()
    replay = _execute_with_objects(phase1_roots, operation_bytes())
    assert replay == sealed_raw
    assert list(key.rglob(".phase1-stage-*")) == []
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize("target_kind", ["receipt", "final-record"])
def test_t22_postlink_sealed_alias_repair_fault_is_never_indeterminate(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    linked_targets: dict[str, str] = {}
    injected = KeyboardInterrupt(f"{target_kind}-post-link")
    tripped = False

    def remember_link(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_link(source, target, *args, **kwargs)
        if source.startswith(".phase1-stage-"):
            linked_targets[source] = target

    def interrupt_target_unlink(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        target = linked_targets.get(name, "")
        matches = (
            target_kind == "receipt" and target == "receipt.json"
        ) or (
            target_kind == "final-record"
            and target.endswith("RECEIPT_FINALIZED.json")
        )
        if matches and not tripped:
            tripped = True
            raise injected
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", remember_link)
    monkeypatch.setattr(store_module.os, "unlink", interrupt_target_unlink)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )
    assert caught.value is injected
    assert tripped
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    sealed = strict_loads((key / "receipt.json").read_bytes())
    assert list(key.rglob(".phase1-stage-*"))

    monkeypatch.undo()
    repair_unlink = store_module.os.unlink

    def fail_alias_repair(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        if name.startswith(".phase1-stage-"):
            raise OSError(errno.EIO, "injected alias repair failure")
        repair_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "unlink", fail_alias_repair)
    with pytest.raises(SealedReceiptUnavailable) as unavailable:
        _execute_with_objects(phase1_roots, operation_bytes())

    error = unavailable.value
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert error.idempotency_key_digest == operation.idempotency_key_digest
    assert error.operation_digest == operation.operation_digest
    assert error.receipt_digest == sealed["receipt_digest"]
    assert error.receipt_state == "ENGINE_TERMINAL"
    assert error.receipt_was_durably_sealed is True
    assert error.automatic_engine_retry_permitted is False
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize(
    "release_boundary", ["global-unlock", "global-close", "root-release"]
)
@pytest.mark.parametrize("claim_outcome", ["owner", "sealed-response"])
def test_t14_t19_fresh_claim_release_fault_has_no_owner_or_fd_leak(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    release_boundary: str,
    claim_outcome: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    operation = parse_operation(operation_bytes())
    receipt_entry = phase1_roots["receipt"].stat()
    root_identity = (receipt_entry.st_dev, receipt_entry.st_ino)
    registry_key = (*root_identity, operation.idempotency_key_digest)
    original_create_global = store_module._create_lock_file
    original_create_key = store_module._create_permanent_lock
    original_flock = store_module.fcntl.flock
    original_close = store_module.os.close
    original_fstat = store_module.os.fstat
    original_root_release = store_module._root_mutex_release
    original_registry_insert = store_module._registry_insert
    global_fds: list[int] = []
    key_fds: list[int] = []
    release_tripped = False
    insert_tripped = False

    def remember_global_fd(*args: object, **kwargs: object) -> int:
        fd = original_create_global(*args, **kwargs)
        global_fds.append(fd)
        return fd

    def remember_key_fd(*args: object, **kwargs: object) -> int:
        fd = original_create_key(*args, **kwargs)
        key_fds.append(fd)
        return fd

    def unlock_then_raise(fd: int, flags: int) -> None:
        nonlocal release_tripped
        if (
            release_boundary == "global-unlock"
            and fd in global_fds
            and flags == fcntl.LOCK_UN
            and not release_tripped
        ):
            original_flock(fd, flags)
            release_tripped = True
            raise OSError(errno.EIO, "injected global unlock failure")
        original_flock(fd, flags)

    def close_then_raise(fd: int) -> None:
        nonlocal release_tripped
        if (
            release_boundary == "global-close"
            and fd in global_fds
            and not release_tripped
        ):
            original_close(fd)
            release_tripped = True
            raise OSError(errno.EIO, "injected global close failure")
        original_close(fd)

    def release_root_then_raise(
        identity: tuple[int, int], lock: threading.Lock
    ) -> None:
        nonlocal release_tripped
        if (
            release_boundary == "root-release"
            and identity == root_identity
            and not release_tripped
        ):
            original_root_release(identity, lock)
            release_tripped = True
            raise PersistenceError("injected root release failure")
        original_root_release(identity, lock)

    def insert_then_maybe_raise(*args: object, **kwargs: object) -> None:
        nonlocal insert_tripped
        original_registry_insert(*args, **kwargs)
        if claim_outcome == "sealed-response" and not insert_tripped:
            insert_tripped = True
            raise PersistenceError("injected post-insert failure")

    monkeypatch.setattr(store_module, "_create_lock_file", remember_global_fd)
    monkeypatch.setattr(store_module, "_create_permanent_lock", remember_key_fd)
    monkeypatch.setattr(store_module.fcntl, "flock", unlock_then_raise)
    monkeypatch.setattr(store_module.os, "close", close_then_raise)
    monkeypatch.setattr(
        store_module, "_root_mutex_release", release_root_then_raise
    )
    monkeypatch.setattr(store_module, "_registry_insert", insert_then_maybe_raise)

    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    receipt = strict_loads(raw)
    assert release_tripped
    assert insert_tripped is (claim_outcome == "sealed-response")
    assert receipt["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert receipt["durability"] == "SEALED"
    assert receipt["reason_codes"] == ["PERSISTENCE_CORRUPTION"]
    assert pair_a.call_count == pair_b.call_count == 0
    assert store_module._registry_lookup(registry_key) is None
    assert root_identity not in store_module._ROOT_MUTEXES
    for fd in [*global_fds, *key_fds]:
        with pytest.raises(OSError) as closed:
            original_fstat(fd)
        assert closed.value.errno == errno.EBADF

    monkeypatch.undo()
    assert _execute_with_objects(phase1_roots, operation_bytes()) == raw
    assert pair_a.call_count == pair_b.call_count == 0


def test_t19_real_same_process_threads_wait_then_bind_durable_anchor(
    phase1_roots: dict[str, Path],
) -> None:
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    preflight_started = threading.Event()
    allow_preflight = threading.Event()
    owner_ready = threading.Event()
    allow_owner_close = threading.Event()
    contender_done = threading.Event()
    shared: dict[str, Any] = {}

    def owner_worker() -> None:
        capabilities = open_root_set(
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
        )

        def preflight() -> None:
            preflight_started.set()
            if not allow_preflight.wait(10):
                raise RuntimeError("threaded preflight timed out")

        result = ReceiptStore(capabilities).claim(
            operation, fresh_preflight=preflight
        )
        assert result.owner is not None
        owner = result.owner
        shared["anchor"] = str(owner.records[0]["record_digest"])
        owner.append("INVOCATION_CLAIMED")
        owner_ready.set()
        assert allow_owner_close.wait(10)
        owner.close()
        capabilities.close()

    def contender_worker() -> None:
        capabilities = open_root_set(
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
        )
        result = ReceiptStore(capabilities).claim(
            operation,
            fresh_preflight=lambda: pytest.fail(
                "thread contender ran fresh preflight"
            ),
        )
        shared["contender"] = result.response
        capabilities.close()
        contender_done.set()

    first = threading.Thread(target=owner_worker)
    second = threading.Thread(target=contender_worker)
    first.start()
    assert preflight_started.wait(10)
    second.start()
    assert not contender_done.wait(0.1)
    allow_preflight.set()
    assert owner_ready.wait(10)
    assert contender_done.wait(10)
    contender = strict_loads(shared["contender"])
    assert contender["receipt_state"] == "CONFLICT_IN_PROGRESS"
    assert contender["ownership_record_digest"] == shared["anchor"]

    different_same_key = parse_operation(
        operation_bytes(correlation_id="corr-phase1-thread-other")
    )
    capabilities = open_root_set(
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
    )
    different_conflict = ReceiptStore(capabilities).claim(
        different_same_key,
        fresh_preflight=lambda: pytest.fail(
            "different-operation thread contender ran preflight"
        ),
    )
    capabilities.close()
    assert different_conflict.response is not None
    different_receipt = strict_loads(different_conflict.response)
    assert different_receipt["receipt_state"] == "IDEMPOTENCY_CONFLICT"
    assert different_receipt["ownership_record_digest"] == shared["anchor"]
    assert different_receipt["bound_operation_digest"] == operation.operation_digest

    different = parse_operation(
        operation_bytes(idempotency_key="idem-phase1-independent")
    )
    capabilities = open_root_set(
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
    )
    independent = ReceiptStore(capabilities).claim(
        different, fresh_preflight=lambda: None
    )
    assert independent.owner is not None
    independent.owner.close()
    capabilities.close()
    allow_owner_close.set()
    first.join(10)
    second.join(10)
    assert not first.is_alive() and not second.is_alive()


@pytest.mark.parametrize("injected", _publication_control_flow_objects())
def test_t19_t23_owner_cleanup_preserves_primary_and_removes_released_token(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    envelope = operation_bytes()
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    owner_fd = owner.key_lock_fd
    registry_key = owner.registry_key
    original_flock = store_module.fcntl.flock
    tripped = False

    def fail_unlock_once(fd: int, flags: int) -> None:
        nonlocal tripped
        if fd == owner_fd and flags == fcntl.LOCK_UN and not tripped:
            tripped = True
            raise OSError(errno.EIO, "injected owner unlock failure")
        original_flock(fd, flags)

    monkeypatch.setattr(store_module.fcntl, "flock", fail_unlock_once)
    try:
        with pytest.raises(type(injected)) as caught:
            with owner:
                raise injected
        assert caught.value is injected
        assert tripped
        assert store_module._registry_lookup(registry_key) is None
    finally:
        capabilities.close()


@pytest.mark.parametrize(
    ("consumed", "failure_type"),
    [
        (False, StorePreInvokeUnavailable),
        (True, StoreIndeterminateUnavailable),
    ],
)
def test_t20_t21_recovery_append_failure_uses_exact_private_outcome(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    consumed: bool,
    failure_type: type[BaseException],
) -> None:
    envelope = operation_bytes()
    capabilities, owner = _claim_owner(phase1_roots, envelope)
    try:
        if consumed:
            owner.append("INVOCATION_CLAIMED")
    finally:
        owner.close()
        capabilities.close()
    original_append = OwnerHandle.append

    def fail_recovery_append(
        self: OwnerHandle, state: str, **kwargs: object
    ):
        if state in {"PREINVOKE_REJECTED", "INDETERMINATE_NO_RETRY"}:
            raise PersistenceError("injected recovery append failure")
        return original_append(self, state, **kwargs)

    monkeypatch.setattr(OwnerHandle, "append", fail_recovery_append)
    with pytest.raises(failure_type):
        _recover(phase1_roots, envelope)


@pytest.mark.parametrize(
    "boundary",
    [
        "final-record",
        "record-1-chmod",
        "record-2-chmod",
        "record-3-chmod",
        "record-4-chmod",
        "receipt-chmod",
        "records-directory-chmod",
        "key-directory-chmod",
    ],
)
def test_t22_postseal_failure_returns_safe_bytes_and_replay_finalizes(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    tripped = False
    original_publish = store_module._publish_no_replace
    original_file_chmod = store_module._chmod_file_and_sync
    original_directory_chmod = store_module._chmod_directory_and_sync

    def publish(**kwargs: object) -> None:
        nonlocal tripped
        target_name = str(kwargs["target_name"])
        if (
            boundary == "final-record"
            and "RECEIPT_FINALIZED" in target_name
            and not tripped
        ):
            tripped = True
            raise PersistenceError("injected final-record failure")
        original_publish(**kwargs)

    def file_chmod(path: Path, mode: int, parent: Path) -> None:
        nonlocal tripped
        record_boundary = (
            boundary.startswith("record-") and boundary.endswith("-chmod")
        )
        record_sequence = (
            int(boundary.removeprefix("record-").removesuffix("-chmod"))
            if record_boundary
            else None
        )
        matches = (
            record_sequence is not None
            and parent.name == "records"
            and path.name.startswith(f"{record_sequence:06d}-")
        ) or (boundary == "receipt-chmod" and path.name == "receipt.json")
        if matches and not tripped:
            tripped = True
            raise PersistenceError("injected file chmod failure")
        original_file_chmod(path, mode, parent)

    def directory_chmod(path: Path, mode: int, parent: Path) -> None:
        nonlocal tripped
        matches = (
            boundary == "records-directory-chmod" and path.name == "records"
        ) or (
            boundary == "key-directory-chmod" and path.name != "records"
        )
        if matches and not tripped:
            tripped = True
            raise PersistenceError("injected directory chmod failure")
        original_directory_chmod(path, mode, parent)

    monkeypatch.setattr(store_module, "_publish_no_replace", publish)
    monkeypatch.setattr(store_module, "_chmod_file_and_sync", file_chmod)
    monkeypatch.setattr(
        store_module, "_chmod_directory_and_sync", directory_chmod
    )
    first = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    assert tripped
    assert strict_loads(first)["receipt_state"] == "ENGINE_TERMINAL"
    monkeypatch.undo()
    replay = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=object(),
        pair_b=object(),
    )
    assert replay == first
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    assert (key.stat().st_mode & 0o7777) == 0o500
    assert ((key / "records").stat().st_mode & 0o7777) == 0o500
    assert ((key / "receipt.json").stat().st_mode & 0o7777) == 0o400
    assert all(
        (path.stat().st_mode & 0o7777) == 0o400
        for path in (key / "records").iterdir()
    )
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "stage-create",
        "stage-fsync",
        "link",
        "target-parent-fsync",
        "unlink",
        "stage-parent-fsync",
    ],
)
def test_t22_final_record_publication_syscall_matrix_replays_exact_bytes(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_publish = store_module._publish_no_replace
    original_open = store_module.os.open
    original_fsync = store_module.os.fsync
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    active_final_record = False
    tripped = False
    fault_calls = 0
    fsync_ordinal = 0
    fault_fd: int | None = None

    def scoped_publish(**kwargs: object) -> None:
        nonlocal active_final_record
        previous = active_final_record
        active_final_record = str(kwargs["target_name"]).endswith(
            "RECEIPT_FINALIZED.json"
        )
        try:
            original_publish(**kwargs)
        finally:
            active_final_record = previous

    def fail_stage_create_once(*args: object, **kwargs: object) -> int:
        nonlocal tripped, fault_calls
        path = args[0] if args else kwargs.get("path")
        if (
            active_final_record
            and boundary == "stage-create"
            and isinstance(path, str)
            and path.startswith(".phase1-stage-")
        ):
            fault_calls += 1
            if not tripped:
                tripped = True
                raise OSError(errno.EIO, "injected final-record stage create failure")
        return original_open(*args, **kwargs)

    def fail_fsync_once(fd: int) -> None:
        nonlocal tripped, fault_calls, fsync_ordinal, fault_fd
        if active_final_record:
            fsync_ordinal += 1
            wanted = {
                "stage-fsync": 1,
                "target-parent-fsync": 2,
                "stage-parent-fsync": 3,
            }.get(boundary)
            if fault_fd == fd:
                fault_calls += 1
            if wanted == fsync_ordinal and not tripped:
                tripped = True
                fault_fd = fd
                fault_calls = 1
                raise OSError(errno.EIO, f"injected final-record {boundary}")
        original_fsync(fd)

    def fail_link_once(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped, fault_calls
        if active_final_record and boundary == "link":
            fault_calls += 1
            if not tripped:
                tripped = True
                raise OSError(errno.EIO, "injected final-record link failure")
        original_link(source, target, *args, **kwargs)

    def fail_unlink_once(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped, fault_calls
        if (
            active_final_record
            and boundary == "unlink"
            and name.startswith(".phase1-stage-")
        ):
            fault_calls += 1
            if not tripped:
                tripped = True
                raise OSError(errno.EIO, "injected final-record unlink failure")
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module, "_publish_no_replace", scoped_publish)
    supported_dir_fd = set(store_module.os.supports_dir_fd)
    for original, replacement in (
        (original_open, fail_stage_create_once),
        (original_link, fail_link_once),
        (original_unlink, fail_unlink_once),
    ):
        if original in store_module.os.supports_dir_fd:
            supported_dir_fd.add(replacement)
    monkeypatch.setattr(store_module.os, "supports_dir_fd", supported_dir_fd)
    monkeypatch.setattr(store_module.os, "open", fail_stage_create_once)
    monkeypatch.setattr(store_module.os, "fsync", fail_fsync_once)
    monkeypatch.setattr(store_module.os, "link", fail_link_once)
    monkeypatch.setattr(store_module.os, "unlink", fail_unlink_once)

    first = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    assert tripped
    assert fault_calls == 1
    assert strict_loads(first)["receipt_state"] == "ENGINE_TERMINAL"
    assert pair_a.call_count == pair_b.call_count == 1

    monkeypatch.undo()
    replay = _execute_with_objects(phase1_roots, operation_bytes())
    assert replay == first
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    assert list(key.rglob(".phase1-stage-*")) == []
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "stage-create",
        "stage-fsync",
        "link",
        "target-parent-fsync",
        "unlink",
        "stage-parent-fsync",
    ],
)
def test_t22_receipt_publication_syscall_matrix_preserves_exact_outcome(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_publish = store_module._publish_no_replace
    original_open = store_module.os.open
    original_fsync = store_module.os.fsync
    original_link = store_module.os.link
    original_unlink = store_module.os.unlink
    active_receipt = False
    tripped = False
    fsync_ordinal = 0
    engine_receipt_raw: bytes | None = None

    def scoped_publish(**kwargs: object) -> None:
        nonlocal active_receipt, engine_receipt_raw
        previous = active_receipt
        active_receipt = kwargs["target_name"] == "receipt.json"
        if active_receipt and engine_receipt_raw is None:
            value = kwargs["raw"]
            assert isinstance(value, bytes)
            engine_receipt_raw = value
        try:
            original_publish(**kwargs)
        finally:
            active_receipt = previous

    def fail_stage_create_once(*args: object, **kwargs: object) -> int:
        nonlocal tripped
        path = args[0] if args else kwargs.get("path")
        if (
            active_receipt
            and boundary == "stage-create"
            and isinstance(path, str)
            and path.startswith(".phase1-stage-")
            and not tripped
        ):
            tripped = True
            raise OSError(errno.EIO, "injected receipt stage create failure")
        return original_open(*args, **kwargs)

    def fail_fsync_once(fd: int) -> None:
        nonlocal fsync_ordinal, tripped
        if active_receipt:
            fsync_ordinal += 1
            wanted = {
                "stage-fsync": 1,
                "target-parent-fsync": 2,
                "stage-parent-fsync": 3,
            }.get(boundary)
            if wanted == fsync_ordinal and not tripped:
                tripped = True
                raise OSError(errno.EIO, f"injected receipt {boundary}")
        original_fsync(fd)

    def fail_link_once(
        source: str,
        target: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        if active_receipt and boundary == "link" and not tripped:
            tripped = True
            raise OSError(errno.EIO, "injected receipt link failure")
        original_link(source, target, *args, **kwargs)

    def fail_unlink_once(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal tripped
        if (
            active_receipt
            and boundary == "unlink"
            and name.startswith(".phase1-stage-")
            and not tripped
        ):
            tripped = True
            raise OSError(errno.EIO, "injected receipt unlink failure")
        original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(store_module, "_publish_no_replace", scoped_publish)
    supported_dir_fd = set(store_module.os.supports_dir_fd)
    for original, replacement in (
        (original_open, fail_stage_create_once),
        (original_link, fail_link_once),
        (original_unlink, fail_unlink_once),
    ):
        if original in store_module.os.supports_dir_fd:
            supported_dir_fd.add(replacement)
    monkeypatch.setattr(store_module.os, "supports_dir_fd", supported_dir_fd)
    monkeypatch.setattr(store_module.os, "open", fail_stage_create_once)
    monkeypatch.setattr(store_module.os, "fsync", fail_fsync_once)
    monkeypatch.setattr(store_module.os, "link", fail_link_once)
    monkeypatch.setattr(store_module.os, "unlink", fail_unlink_once)

    if boundary == "target-parent-fsync":
        with pytest.raises(IndeterminateReceiptUnavailable) as unavailable:
            execute_operation(
                operation_bytes(),
                repository_root=phase1_roots["repository"],
                receipt_root=phase1_roots["receipt"],
                evidence_root=phase1_roots["evidence"],
                pair_a=pair_a,
                pair_b=pair_b,
            )
        assert unavailable.value.__cause__ is None
        first = None
    else:
        first = execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=pair_a,
            pair_b=pair_b,
        )

    assert tripped
    assert engine_receipt_raw is not None
    assert pair_a.call_count == pair_b.call_count == 1
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    receipt_path = key / "receipt.json"
    stages = list(key.glob(".phase1-stage-*"))

    if boundary in {"stage-create", "stage-fsync", "link"}:
        assert first is not None
        receipt = strict_loads(first)
        assert receipt["receipt_state"] == "INDETERMINATE_NO_RETRY"
        assert receipt["reason_codes"] == ["PERSISTENCE_CORRUPTION"]
        assert first != engine_receipt_raw
        assert stages == []
    elif boundary in {"target-parent-fsync", "unlink"}:
        assert receipt_path.read_bytes() == engine_receipt_raw
        assert len(stages) == 1
        receipt_entry = receipt_path.stat(follow_symlinks=False)
        stage_entry = stages[0].stat(follow_symlinks=False)
        assert (receipt_entry.st_dev, receipt_entry.st_ino) == (
            stage_entry.st_dev,
            stage_entry.st_ino,
        )
        assert receipt_entry.st_nlink == stage_entry.st_nlink == 2
        if boundary == "unlink":
            assert first == engine_receipt_raw
    else:
        assert boundary == "stage-parent-fsync"
        assert first == engine_receipt_raw
        assert receipt_path.read_bytes() == engine_receipt_raw
        assert stages == []

    monkeypatch.undo()
    if first is not None:
        replay = _execute_with_objects(phase1_roots, operation_bytes())
        assert replay == first
    else:
        replay = _execute_with_objects(phase1_roots, operation_bytes())
        assert replay == engine_receipt_raw
    assert list(key.glob(".phase1-stage-*")) == []
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "record-1-file-refsync",
        "record-2-file-refsync",
        "record-3-file-refsync",
        "record-4-file-refsync",
        "record-1-parent-refsync",
        "record-2-parent-refsync",
        "record-3-parent-refsync",
        "record-4-parent-refsync",
        "receipt-file-refsync",
        "receipt-parent-refsync",
        "records-directory-refsync",
        "records-parent-refsync",
        "key-directory-refsync",
        "key-parent-refsync",
    ],
)
def test_t22_recovery_refsync_failure_returns_exact_sealed_bytes(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    records = key / "records"
    shard = key.parent
    original_finalize = store_module._finalize_metadata
    original_file_refsync = store_module._fsync_file_and_parent
    original_directory_refsync = store_module._fsync_directory
    active_finalization = False
    tripped = False
    directory_counts: dict[Path, int] = {}

    def scoped_finalize(**kwargs: object) -> None:
        nonlocal active_finalization
        previous = active_finalization
        active_finalization = True
        try:
            original_finalize(**kwargs)
        finally:
            active_finalization = previous

    def fail_file_refsync_once(
        path: Path, parent: Path, modes: set[int]
    ) -> None:
        nonlocal tripped
        matches = boundary == "receipt-file-refsync" and path.name == "receipt.json"
        if boundary.startswith("record-") and boundary.endswith("-file-refsync"):
            sequence = int(
                boundary.removeprefix("record-").removesuffix("-file-refsync")
            )
            matches = parent == records and path.name.startswith(f"{sequence:06d}-")
        if active_finalization and matches and not tripped:
            tripped = True
            raise PersistenceError(f"injected recovery refsync fault: {boundary}")
        original_file_refsync(path, parent, modes)

    def fail_directory_refsync_once(path: Path, modes: set[int]) -> None:
        nonlocal tripped
        if active_finalization:
            directory_counts[path] = directory_counts.get(path, 0) + 1
            occurrence = directory_counts[path]
            wanted: tuple[Path, int] | None = None
            if boundary.startswith("record-") and boundary.endswith(
                "-parent-refsync"
            ):
                sequence = int(
                    boundary.removeprefix("record-").removesuffix(
                        "-parent-refsync"
                    )
                )
                wanted = (records, sequence)
            else:
                wanted = {
                    "receipt-parent-refsync": (key, 1),
                    "records-directory-refsync": (records, 5),
                    "records-parent-refsync": (key, 2),
                    "key-directory-refsync": (key, 3),
                    "key-parent-refsync": (shard, 1),
                }.get(boundary)
            if wanted == (path, occurrence) and not tripped:
                tripped = True
                raise PersistenceError(
                    f"injected recovery directory refsync fault: {boundary}"
                )
        original_directory_refsync(path, modes)

    monkeypatch.setattr(store_module, "_finalize_metadata", scoped_finalize)
    monkeypatch.setattr(
        store_module, "_fsync_file_and_parent", fail_file_refsync_once
    )
    monkeypatch.setattr(
        store_module, "_fsync_directory", fail_directory_refsync_once
    )
    replay = _execute_with_objects(phase1_roots, operation_bytes())
    assert tripped
    assert replay == raw
    assert pair_a.call_count == pair_b.call_count == 1

    monkeypatch.undo()
    assert _execute_with_objects(phase1_roots, operation_bytes()) == raw
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize(
    "node",
    [
        "record-1",
        "record-2",
        "record-3",
        "record-4",
        "receipt",
        "records-directory",
        "key-directory",
    ],
)
@pytest.mark.parametrize(
    "boundary", ["fchmod", "inode-fsync", "parent-fsync"]
)
def test_t22_partial_prefix_syscall_fault_replays_without_relabel_or_invocation(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    node: str,
    boundary: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    operation = parse_operation(operation_bytes())
    key = _key_directory(phase1_roots, operation)
    records_directory = key / "records"
    receipt_path = key / "receipt.json"
    records = sorted(records_directory.iterdir())
    assert len(records) == 4
    original_names = [path.name for path in records]
    assert original_names[-1].endswith("RECEIPT_FINALIZED.json")

    key.chmod(0o700)
    records_directory.chmod(0o700)
    receipt_path.chmod(0o600)
    for record in records:
        record.chmod(0o600)

    if node.startswith("record-"):
        sequence = int(node.removeprefix("record-"))
        for record in records[: sequence - 1]:
            record.chmod(0o400)
        target = records[sequence - 1]
        parent = records_directory
        final_mode = 0o400
    elif node == "receipt":
        for record in records:
            record.chmod(0o400)
        target = receipt_path
        parent = key
        final_mode = 0o400
    elif node == "records-directory":
        for record in records:
            record.chmod(0o400)
        receipt_path.chmod(0o400)
        target = records_directory
        parent = key
        final_mode = 0o500
    else:
        assert node == "key-directory"
        for record in records:
            record.chmod(0o400)
        receipt_path.chmod(0o400)
        records_directory.chmod(0o500)
        target = key
        parent = key.parent
        final_mode = 0o500

    target_entry = target.stat(follow_symlinks=False)
    parent_entry = parent.stat(follow_symlinks=False)
    target_identity = (target_entry.st_dev, target_entry.st_ino)
    parent_identity = (parent_entry.st_dev, parent_entry.st_ino)
    original_fchmod = store_module.os.fchmod
    original_fsync = store_module.os.fsync
    original_fstat = store_module.os.fstat
    target_started = False
    target_synced = False
    tripped = False

    def fd_identity(fd: int) -> tuple[int, int]:
        entry = original_fstat(fd)
        return (entry.st_dev, entry.st_ino)

    def fchmod_then_interrupt(fd: int, mode: int) -> None:
        nonlocal target_started, tripped
        if fd_identity(fd) == target_identity and mode == final_mode:
            original_fchmod(fd, mode)
            target_started = True
            if boundary == "fchmod" and not tripped:
                tripped = True
                raise OSError(errno.EIO, f"injected {node} fchmod failure")
            return
        original_fchmod(fd, mode)

    def fsync_then_interrupt(fd: int) -> None:
        nonlocal target_synced, tripped
        identity = fd_identity(fd)
        if target_started and identity == target_identity:
            original_fsync(fd)
            target_synced = True
            if boundary == "inode-fsync" and not tripped:
                tripped = True
                raise OSError(errno.EIO, f"injected {node} inode fsync failure")
            return
        if target_synced and identity == parent_identity:
            original_fsync(fd)
            if boundary == "parent-fsync" and not tripped:
                tripped = True
                raise OSError(errno.EIO, f"injected {node} parent fsync failure")
            return
        original_fsync(fd)

    monkeypatch.setattr(store_module.os, "fchmod", fchmod_then_interrupt)
    monkeypatch.setattr(store_module.os, "fsync", fsync_then_interrupt)
    replay = _execute_with_objects(phase1_roots, operation_bytes())
    assert tripped
    assert replay == raw
    assert receipt_path.read_bytes() == raw
    assert [path.name for path in sorted(records_directory.iterdir())] == (
        original_names
    )
    assert pair_a.call_count == pair_b.call_count == 1

    monkeypatch.undo()
    assert _execute_with_objects(phase1_roots, operation_bytes()) == raw
    assert stat.S_IMODE(key.stat().st_mode) == 0o500
    assert stat.S_IMODE(records_directory.stat().st_mode) == 0o500
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o400 for path in records)
    assert [path.name for path in sorted(records_directory.iterdir())] == (
        original_names
    )
    assert pair_a.call_count == pair_b.call_count == 1


def test_t22_invalid_postseal_permission_order_raises_exact_typed_error(
    phase1_roots: dict[str, Path],
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    receipt = strict_loads(raw)
    operation = parse_operation(operation_bytes())
    receipt_path = _key_directory(phase1_roots, operation) / "receipt.json"
    receipt_path.chmod(0o600)
    with pytest.raises(SealedReceiptUnavailable) as caught:
        execute_operation(
            operation_bytes(),
            repository_root=phase1_roots["repository"],
            receipt_root=phase1_roots["receipt"],
            evidence_root=phase1_roots["evidence"],
            pair_a=object(),
            pair_b=object(),
        )
    error = caught.value
    assert error.__cause__ is None
    assert str(error) == (
        "A Phase 1 receipt was durably sealed but cannot be safely returned; "
        "automatic retry is forbidden"
    )
    assert error.receipt_digest == receipt["receipt_digest"]
    assert error.receipt_state == "ENGINE_TERMINAL"
    assert error.reason_code == "PERSISTENCE_CORRUPTION"
    assert error.receipt_was_durably_sealed is True
    assert error.automatic_engine_retry_permitted is False
