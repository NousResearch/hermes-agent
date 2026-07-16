"""TDD contract tests for safe research protocol artifact persistence."""

import hashlib
import json
import multiprocessing
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from plugins.research_protocol.models import (
    BudgetLimits,
    Capability,
    CapabilityGrant,
    OutputRequirement,
    PlanV1,
)
from plugins.research_protocol.storage.artifacts import (
    ArtifactStore,
    ArtifactSecurityError,
    canonical_json_bytes,
)


SHA_A = "a" * 64


def plan_payload() -> dict:
    return {
        "schema_version": "plan.v1",
        "producer_version": "test",
        "run_id": "run-001",
        "created_at": datetime(2026, 7, 13, 20, 0, tzinfo=UTC),
        "objective": "Offline test plan",
        "constraints": ("offline",),
        "inputs": (),
        "outputs": (
            OutputRequirement(
                artifact_id="manifest-001",
                artifact_type="manifest",
            ),
        ),
        "budgets": BudgetLimits(
            max_duration_seconds=60,
            max_executions=1,
            max_records=10,
            max_bytes=1000,
            max_external_calls=0,
        ).model_dump(),
        "capabilities": (
            CapabilityGrant(
                capability=Capability.RESEARCH_COLLECT,
                input_hashes=(SHA_A,),
                external_rights=(),
            ),
        ),
    }


def _concurrent_persist_worker(root: str, start, results) -> None:
    store = ArtifactStore(root)
    start.wait(timeout=5)
    try:
        receipt = store.persist("plan", "plan-concurrent", plan_payload())
        results.put(("persisted", receipt.sha256))
    except FileExistsError:
        results.put(("exists", None))


def test_canonical_json_is_utf8_sorted_compact_and_hashes_exact_bytes():
    value = {"z": "é", "a": [1, 2]}

    encoded = canonical_json_bytes(value)

    assert encoded == b'{"a":[1,2],"z":"\xc3\xa9"}'
    assert hashlib.sha256(encoded).hexdigest() == (
        "69ed5fc8c2388fea22214cceadcebfe6171cbb85cb913d7adb3281efe7431141"
    )


def test_new_storage_subdirectories_are_made_durable_in_the_root(
    tmp_path,
    monkeypatch,
):
    store = ArtifactStore(tmp_path)
    root_stat = os.stat(tmp_path)
    real_fsync = os.fsync
    root_fsyncs = 0

    def track_fsync(fd):
        nonlocal root_fsyncs
        descriptor_stat = os.fstat(fd)
        if (
            descriptor_stat.st_dev == root_stat.st_dev
            and descriptor_stat.st_ino == root_stat.st_ino
        ):
            root_fsyncs += 1
        return real_fsync(fd)

    monkeypatch.setattr(
        "plugins.research_protocol.storage.artifacts.os.fsync",
        track_fsync,
    )

    store.persist("plan", "plan-root-fsync", plan_payload())

    assert root_fsyncs == 3


def test_canonical_json_rejects_non_finite_numbers():
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_store_rejects_group_or_world_accessible_root(tmp_path):
    os.chmod(tmp_path, 0o777)
    try:
        with pytest.raises(ArtifactSecurityError, match="private"):
            ArtifactStore(tmp_path)
    finally:
        os.chmod(tmp_path, 0o700)


def test_store_rejects_root_not_owned_by_effective_user(tmp_path, monkeypatch):
    owner_uid = os.stat(tmp_path).st_uid
    monkeypatch.setattr(
        "plugins.research_protocol.storage.artifacts.os.geteuid",
        lambda: owner_uid + 1,
    )

    with pytest.raises(ArtifactSecurityError, match="owned"):
        ArtifactStore(tmp_path)


def test_store_validates_model_and_returns_readback_hash_and_length(tmp_path):
    store = ArtifactStore(tmp_path)
    payload = plan_payload()

    receipt = store.persist("plan", "plan-001", payload)

    expected = canonical_json_bytes(
        PlanV1.model_validate(payload).model_dump(mode="json")
    )
    assert receipt.artifact_id == "plan-001"
    assert receipt.path_relative == "plans/plan-001.json"
    assert receipt.sha256 == hashlib.sha256(expected).hexdigest()
    assert receipt.byte_length == len(expected)
    assert store.read_bytes("plan", "plan-001") == expected
    assert json.loads(
        (tmp_path / receipt.path_relative).read_text(encoding="utf-8")
    ) == json.loads(expected)


@pytest.mark.parametrize(
    "artifact_id",
    ["../escape", "a/b", "/absolute", "", ".", "a\\b", "a//b"],
)
def test_store_rejects_traversal_absolute_and_ambiguous_identifiers(
    tmp_path, artifact_id
):
    with pytest.raises(ArtifactSecurityError):
        ArtifactStore(tmp_path).persist("plan", artifact_id, plan_payload())


def test_store_rejects_unknown_artifact_type(tmp_path):
    with pytest.raises(ArtifactSecurityError):
        ArtifactStore(tmp_path).persist("arbitrary", "id-001", {})


def test_store_rejects_symlinked_directory_and_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    root_link = tmp_path / "link"
    root_link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ArtifactSecurityError):
        ArtifactStore(root_link)

    (tmp_path / "root").mkdir(mode=0o700)
    store = ArtifactStore(tmp_path / "root")
    (tmp_path / "root" / "plans").symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactSecurityError):
        store.persist("plan", "plan-001", plan_payload())


def test_store_never_overwrites_existing_target(tmp_path):
    store = ArtifactStore(tmp_path)
    receipt = store.persist("plan", "plan-001", plan_payload())
    original = (tmp_path / receipt.path_relative).read_bytes()

    with pytest.raises(FileExistsError):
        store.persist("plan", "plan-001", plan_payload())

    assert (tmp_path / receipt.path_relative).read_bytes() == original


def test_failed_atomic_publish_does_not_publish_partial_artifact(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)

    def fail_link(*_args, **_kwargs):
        raise OSError("simulated crash before publish")

    monkeypatch.setattr(
        "plugins.research_protocol.storage.artifacts.os.link", fail_link
    )
    with pytest.raises(OSError, match="simulated crash"):
        store.persist("plan", "plan-001", plan_payload())

    assert not (tmp_path / "plans" / "plan-001.json").exists()
    assert list((tmp_path / "plans").iterdir()) == []


def test_failure_after_receipt_publish_rolls_back_both_names(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    real_link = __import__("os").link
    calls = 0

    def fail_second_link(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated crash before artifact publish")
        return real_link(*args, **kwargs)

    monkeypatch.setattr(
        "plugins.research_protocol.storage.artifacts.os.link",
        fail_second_link,
    )

    with pytest.raises(OSError, match="simulated crash"):
        store.persist("plan", "plan-001", plan_payload())

    assert list((tmp_path / "plans").iterdir()) == []
    assert list((tmp_path / "receipts").iterdir()) == []


def test_retry_recovers_receipt_orphan_left_by_process_termination(
    tmp_path,
    monkeypatch,
):
    store = ArtifactStore(tmp_path)
    real_publish = store._publish_temporary
    calls = 0

    def terminate_after_first_publish(*args, **kwargs):
        nonlocal calls
        calls += 1
        real_publish(*args, **kwargs)
        if calls == 1:
            raise SystemExit("simulated process termination")

    monkeypatch.setattr(store, "_publish_temporary", terminate_after_first_publish)
    with pytest.raises(SystemExit, match="simulated process termination"):
        store.persist("plan", "plan-001", plan_payload())

    assert not (tmp_path / "plans" / "plan-001.json").exists()
    assert (tmp_path / "receipts" / "plan-plan-001.receipt.json").is_file()

    monkeypatch.setattr(store, "_publish_temporary", real_publish)
    receipt = store.persist("plan", "plan-001", plan_payload())

    assert store.read_verified(
        "plan",
        "plan-001",
        expected_sha256=receipt.sha256,
    )


def test_concurrent_processes_publish_exactly_once(tmp_path):
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_concurrent_persist_worker,
            args=(str(tmp_path), start, results),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
            pytest.fail("concurrent artifact writer hung")
        assert worker.exitcode == 0

    outcomes = sorted(results.get(timeout=2) for _ in workers)
    assert [outcome for outcome, _digest in outcomes] == ["exists", "persisted"]

    digest = next(digest for outcome, digest in outcomes if outcome == "persisted")
    assert ArtifactStore(tmp_path).read_verified(
        "plan",
        "plan-concurrent",
        expected_sha256=digest,
    )


def test_store_does_not_use_replacing_publish_primitive(tmp_path, monkeypatch):
    def reject_replace(*_args, **_kwargs):
        raise AssertionError("os.replace must not be used for no-clobber persistence")

    monkeypatch.setattr(
        "plugins.research_protocol.storage.artifacts.os.replace",
        reject_replace,
    )

    receipt = ArtifactStore(tmp_path).persist("plan", "plan-001", plan_payload())

    assert (tmp_path / receipt.path_relative).is_file()


def test_store_detects_directory_symlink_swap_during_publish(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    plans = tmp_path / "artifacts" / "plans"
    moved = tmp_path / "plans-before-swap"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    real_link = __import__("os").link
    swapped = False

    def swap_then_link(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            plans.rename(moved)
            plans.symlink_to(attacker, target_is_directory=True)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(
        "plugins.research_protocol.storage.artifacts.os.link",
        swap_then_link,
    )

    with pytest.raises(ArtifactSecurityError, match="changed during operation"):
        store.persist("plan", "plan-001", plan_payload())

    assert not (attacker / "plan-001.json").exists()
    assert not (moved / "plan-001.json").exists()
    assert list((tmp_path / "artifacts" / "receipts").iterdir()) == []


def test_readback_detects_modified_bytes(tmp_path):
    store = ArtifactStore(tmp_path)
    receipt = store.persist("plan", "plan-001", plan_payload())
    path = tmp_path / receipt.path_relative
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(ArtifactSecurityError, match="hash"):
        store.read_verified("plan", "plan-001", expected_sha256=receipt.sha256)
