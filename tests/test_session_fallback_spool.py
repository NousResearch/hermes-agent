import hashlib
import json
import errno
import os
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDBBatchMessage
import session_fallback_spool as spool

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows-only fallback
    fcntl = None


@pytest.fixture()
def spool_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _bootstrap() -> spool.SessionSpoolBootstrap:
    return spool.SessionSpoolBootstrap(
        session_id="session-123",
        source="cli",
        started_at=123.456,
        model="gpt-test",
        model_config={"max_iterations": 2},
        system_prompt="system prompt",
        parent_session_id=None,
        cwd="/tmp/project",
        profile_name=None,
        user_id="user-1",
        session_key="session-key",
        chat_id="chat-1",
        chat_type="group",
        thread_id="thread-1",
    )


def _batch_messages(
    unit_id: str = "unit-1",
    *,
    contents: tuple[str, ...] = ("hello",),
    timestamp: float = 100.0,
) -> tuple[SessionDBBatchMessage, ...]:
    return tuple(
        SessionDBBatchMessage(
            persistence_unit_id=unit_id,
            persistence_message_key=f"{unit_id}-key-{idx}",
            persistence_ordinal=idx,
            role="assistant" if idx else "user",
            content=content,
            timestamp=timestamp + idx,
        )
        for idx, content in enumerate(contents)
    )


def _record(
    unit_id: str = "unit-1",
    *,
    attempt_index: int = 0,
    contents: tuple[str, ...] = ("hello",),
) -> spool.SessionSpoolRecord:
    return spool.SessionSpoolRecord(
        bootstrap=_bootstrap(),
        persist_attempt_id="a" * 32,
        persist_attempt_unit_index=attempt_index,
        canonical_failure={
            "stage": "append_messages_batch",
            "error_class": "RuntimeError",
            "error_message": "db down",
            "session_row_created": True,
        },
        batch_messages=_batch_messages(unit_id, contents=contents),
    )


def _paths(home: Path) -> tuple[Path, Path, Path, Path]:
    root = home / "session_fallback_spool"
    return root, root / "active.spool", root / "append.lock", root / "quarantine"


def _fd_count() -> int:
    import psutil

    return psutil.Process().num_fds()


def test_frame_bytes_are_deterministic_and_schema_validation_is_strict(spool_home):
    record = _record()
    frame_one = spool._frame_bytes_for_record(record)
    frame_two = spool._frame_bytes_for_record(record)

    assert frame_one == frame_two
    assert frame_one[:4] == b"HSPL"
    assert int.from_bytes(frame_one[8:16], "big") == len(frame_one) - 32
    assert frame_one[16:32].hex() == hashlib.blake2s(
        frame_one[4:16] + frame_one[32:], digest_size=16
    ).hexdigest()

    _, active_path, _, _ = _paths(spool_home)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(spool._frame_from_payload_bytes(b'{"schema_version":1}'))

    scan = spool.scan_spool(active_path)
    assert scan.tail_status is spool.SpoolTailStatus.INVALID_SCHEMA
    assert scan.valid_prefix_bytes == 0
    assert scan.frame_count == 0


def test_append_creates_private_profile_local_layout_and_modes(spool_home):
    result = spool.append_records((_record(),))
    receipt = result.unit_results[0].receipt
    root, active_path, lock_path, quarantine_path = _paths(spool_home)

    assert root == spool_home / "session_fallback_spool"
    assert receipt.path == str(active_path)
    assert active_path.exists()
    assert lock_path.exists()
    assert quarantine_path.exists()
    assert str(root).startswith(str(spool_home))

    if os.name == "posix":
        assert (root.stat().st_mode & 0o777) == 0o700
        assert (quarantine_path.stat().st_mode & 0o777) == 0o700
        assert (active_path.stat().st_mode & 0o777) == 0o600
        assert (lock_path.stat().st_mode & 0o777) == 0o600

    scan = spool.scan_spool(active_path)
    assert scan.tail_status is spool.SpoolTailStatus.CLEAN
    assert scan.frame_count == 1
    assert receipt.offset == 0


@pytest.mark.skipif(os.name != "posix", reason="symlink security is POSIX-only")
def test_symlinked_root_is_refused(spool_home):
    root, _, _, _ = _paths(spool_home)
    target = spool_home / "other-root"
    target.mkdir()
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(spool.SpoolPathSecurityError):
        spool.append_records((_record(),))

    assert list(target.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="symlink race schedule is POSIX-only")
def test_root_swap_after_preflight_is_rejected(spool_home, monkeypatch):
    external = spool_home / "external-root"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    real_assert = spool._assert_entry_matches_fd
    swapped = {"done": False}

    def _swap(parent_fd, name, fd, *, expect, label):
        if (
            not swapped["done"]
            and name == spool.SPOOL_ROOT_NAME
            and label == str(spool._spool_root())
        ):
            swapped["done"] = True
            root = spool._spool_root()
            parked = root.with_name(root.name + ".real")
            os.replace(root, parked)
            os.mkdir(root, mode=0o755)
        return real_assert(parent_fd, name, fd, expect=expect, label=label)

    monkeypatch.setattr(spool, "_assert_entry_matches_fd", _swap)

    with pytest.raises(spool.SessionFallbackSpoolError):
        spool.append_records((_record(),))

    replacement_root = spool._spool_root()
    assert replacement_root.is_dir()
    if os.name == "posix":
        assert (replacement_root.stat().st_mode & 0o777) == 0o755
    assert not (replacement_root / "quarantine").exists()
    assert not (replacement_root / "active.spool").exists()
    assert sentinel.read_text(encoding="utf-8") == "outside"


@pytest.mark.skipif(os.name != "posix", reason="quarantine race schedule is POSIX-only")
def test_quarantine_dir_swap_after_preflight_is_rejected(spool_home, monkeypatch):
    root, active_path, _, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    clean_frame = spool._frame_bytes_for_record(_record("unit-clean"))
    active_path.write_bytes(clean_frame[:-1])
    external = spool_home / "external-quarantine"
    external.mkdir()

    real_next = spool._next_quarantine_sequence
    swapped = {"done": False}

    def _swap(path: Path) -> int:
        seq = real_next(path)
        if not swapped["done"]:
            swapped["done"] = True
            parked = path.with_name(path.name + ".real")
            os.replace(path, parked)
            os.symlink(external, path, target_is_directory=True)
        return seq

    monkeypatch.setattr(spool, "_next_quarantine_sequence", _swap)

    with pytest.raises(spool.SessionFallbackSpoolError):
        spool.append_records((_record("unit-fresh"),))

    assert list(external.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="active rename race schedule is POSIX-only")
def test_active_name_swap_before_receipt_is_rejected(spool_home, monkeypatch):
    root, active_path, _, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    active_path.write_bytes(b"")
    external_target = spool_home / "external-active"
    external_target.write_bytes(b"")

    real_fsync = spool._fsync_fd
    seen = {"count": 0}

    def _swap(fd: int) -> None:
        seen["count"] += 1
        if seen["count"] == 1:
            parked = active_path.with_name(active_path.name + ".real")
            os.replace(active_path, parked)
            os.symlink(external_target, active_path)
        real_fsync(fd)

    monkeypatch.setattr(spool, "_fsync_fd", _swap)

    with pytest.raises(spool.SessionFallbackSpoolError):
        spool.append_records((_record("unit-live"),))

    assert external_target.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="sidecar race schedule is POSIX-only")
def test_sidecar_destination_swap_is_rejected(spool_home, monkeypatch):
    root, active_path, _, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    clean_frame = spool._frame_bytes_for_record(_record("unit-clean"))
    active_path.write_bytes(clean_frame[:-1])
    external_target = spool_home / "external-sidecar"
    external_target.write_text("outside", encoding="utf-8")

    real_fsync = spool._fsync_fd
    final_sidecar = quarantine_path / "000001-incomplete_eof-vp0.json"
    injected = {"done": False}

    def _swap(fd: int) -> None:
        if not injected["done"]:
            temp_files = list(quarantine_path.glob("*.tmp"))
            if temp_files:
                injected["done"] = True
                os.symlink(external_target, final_sidecar)
        real_fsync(fd)

    monkeypatch.setattr(spool, "_fsync_fd", _swap)

    with pytest.raises(spool.SessionFallbackSpoolError):
        spool.append_records((_record("unit-fresh"),))

    assert external_target.read_text(encoding="utf-8") == "outside"


def test_unknown_record_kind_is_rejected_by_scan(spool_home):
    payload = spool._payload_bytes_for_record(_record())
    _, active_path, _, _ = _paths(spool_home)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(spool._frame_from_payload_bytes(payload, record_kind=0x02))

    scan = spool.scan_spool(active_path)
    assert scan.tail_status is spool.SpoolTailStatus.BAD_RECORD_KIND


def test_nonzero_reserved_header_bytes_are_rejected_by_scan(spool_home):
    payload = spool._payload_bytes_for_record(_record())
    payload_len = len(payload)
    header_prefix = bytes(
        [spool.FRAME_VERSION, spool.RECORD_KIND_SESSION_PERSISTENCE_UNIT, 0x12, 0x34]
    ) + payload_len.to_bytes(8, "big")
    digest = hashlib.blake2s(header_prefix + payload, digest_size=16).digest()
    frame = spool.HEADER_MAGIC + header_prefix + digest + payload
    _, active_path, _, _ = _paths(spool_home)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(frame)

    scan = spool.scan_spool(active_path)
    assert scan.tail_status is spool.SpoolTailStatus.NONZERO_RESERVED


@pytest.mark.skipif(fcntl is None, reason="POSIX flock required")
def test_non_contention_lock_errors_fail_immediately(spool_home, monkeypatch):
    monkeypatch.setattr(spool, "LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(spool, "LOCK_RETRY_SECONDS", 0.01)

    def _boom(_fd: int, _op: int) -> None:
        raise OSError(errno.EBADF, "bad fd")

    monkeypatch.setattr(spool.fcntl, "flock", _boom)

    with pytest.raises(spool.SpoolDurabilityError):
        spool.append_records((_record(),))


def test_invalid_ordinal_record_is_rejected_before_receipt(spool_home):
    invalid = spool.SessionSpoolRecord(
        bootstrap=_bootstrap(),
        persist_attempt_id="b" * 32,
        persist_attempt_unit_index=0,
        canonical_failure={
            "stage": "append_messages_batch",
            "error_class": "RuntimeError",
            "error_message": "db down",
            "session_row_created": True,
        },
        batch_messages=(
            SessionDBBatchMessage(
                persistence_unit_id="unit-invalid",
                persistence_message_key="key-invalid",
                persistence_ordinal=7,
                role="user",
                content="bad order",
                timestamp=1.0,
            ),
        ),
    )

    with pytest.raises(spool.SessionFallbackSpoolError):
        spool.append_records((invalid,))

    _, active_path, _, _ = _paths(spool_home)
    assert not active_path.exists() or active_path.stat().st_size == 0


def test_scan_budget_exceeded_is_not_reported_clean(spool_home):
    frame_one = spool._frame_bytes_for_record(_record("unit-a"))
    frame_two = spool._frame_bytes_for_record(_record("unit-b"))
    _, active_path, _, _ = _paths(spool_home)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(frame_one + frame_two)

    scan = spool.scan_spool(active_path, max_file_bytes=len(frame_one))

    assert scan.tail_status is not spool.SpoolTailStatus.CLEAN
    assert scan.valid_prefix_bytes == len(frame_one)
    assert scan.tail_offset == len(frame_one)


def test_missing_sidecar_is_reconciled_before_new_receipt(spool_home, monkeypatch):
    root, active_path, _, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    clean_frame = spool._frame_bytes_for_record(_record("unit-clean"))
    active_path.write_bytes(clean_frame[:-1])

    def _boom(*_args, **_kwargs):
        raise OSError("sidecar write failed")

    real_write_sidecar_json = spool._write_sidecar_json
    monkeypatch.setattr(spool, "_write_sidecar_json", _boom)
    with pytest.raises(OSError):
        spool.append_records((_record("unit-first"),))

    monkeypatch.setattr(spool, "_write_sidecar_json", real_write_sidecar_json)
    result = spool.append_records((_record("unit-second"),))

    assert len(result.unit_results) == 1
    assert sorted(quarantine_path.glob("*.spool"))
    assert sorted(quarantine_path.glob("*.json"))


def test_parent_fsync_failure_on_created_active_must_be_reestablished_before_receipt(
    spool_home, monkeypatch
):
    root, active_path, lock_path, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path.write_bytes(b"")
    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(quarantine_path, 0o700)
        os.chmod(lock_path, 0o600)

    real_dir_fsync = spool._fsync_directory_fd
    fail_parent = {"enabled": True}
    root_fsync_calls = {"count": 0}

    def _fail_root(fd: int, label):
        if str(label) == str(root):
            root_fsync_calls["count"] += 1
        if (
            fail_parent["enabled"]
            and str(label) == str(root)
            and root_fsync_calls["count"] >= 3
        ):
            raise OSError("simulated parent fsync failure")
        return real_dir_fsync(fd, label)

    monkeypatch.setattr(spool, "_fsync_directory_fd", _fail_root)

    with pytest.raises(spool.SpoolDurabilityError):
        spool.append_records((_record("unit-first"),))

    assert active_path.exists()
    assert active_path.stat().st_size == 0

    with pytest.raises(spool.SpoolDurabilityError):
        spool.append_records((_record("unit-second"),))

    fail_parent["enabled"] = False
    result = spool.append_records((_record("unit-third"),))
    assert len(result.unit_results) == 1
    assert result.unit_results[0].receipt.offset == 0


@pytest.mark.skipif(os.name != "posix", reason="fd-count regression is POSIX-only")
def test_repeated_parent_fsync_failures_do_not_leak_fds_on_lock_open(spool_home, monkeypatch):
    root, _, lock_path, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path.write_bytes(b"")
    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(quarantine_path, 0o700)
        os.chmod(lock_path, 0o600)

    baseline = _fd_count()
    root_stat = root.stat()
    real_fsync = spool.os.fsync

    def _boom(fd: int):
        fd_stat = os.fstat(fd)
        if stat.S_ISDIR(fd_stat.st_mode) and (
            fd_stat.st_dev == root_stat.st_dev and fd_stat.st_ino == root_stat.st_ino
        ):
            raise OSError("simulated root-directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(spool.os, "fsync", _boom)

    for attempt in range(25):
        with pytest.raises(spool.SpoolDurabilityError):
            spool.append_records((_record(f"unit-lock-{attempt}"),))
        assert _fd_count() == baseline


@pytest.mark.skipif(os.name != "posix", reason="fd-count regression is POSIX-only")
def test_repeated_parent_fsync_failures_do_not_leak_fds_on_quarantine_open(
    spool_home, monkeypatch
):
    root, active_path, lock_path, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path.write_bytes(b"")
    active_path.write_bytes(b"")
    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(quarantine_path, 0o700)
        os.chmod(lock_path, 0o600)
        os.chmod(active_path, 0o600)

    baseline = _fd_count()
    root_stat = root.stat()
    real_fsync = spool.os.fsync

    for attempt in range(25):
        root_fsync_calls = {"count": 0}

        def _boom(fd: int):
            fd_stat = os.fstat(fd)
            if stat.S_ISDIR(fd_stat.st_mode) and (
                fd_stat.st_dev == root_stat.st_dev and fd_stat.st_ino == root_stat.st_ino
            ):
                root_fsync_calls["count"] += 1
                if root_fsync_calls["count"] >= 2:
                    raise OSError("simulated quarantine-parent fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(spool.os, "fsync", _boom)
        with pytest.raises(spool.SpoolDurabilityError):
            spool.append_records((_record(f"unit-dir-{attempt}"),))
        assert _fd_count() == baseline


def test_file_fsync_failure_produces_no_receipt(spool_home, monkeypatch):
    real_fsync = spool._fsync_fd
    seen = {"file": 0}

    def _boom(fd: int) -> None:
        seen["file"] += 1
        if seen["file"] >= 2:
            raise OSError("fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(spool, "_fsync_fd", _boom)

    with pytest.raises(spool.SpoolDurabilityError):
        spool.append_records((_record(),))

    _, active_path, _, _ = _paths(spool_home)
    assert active_path.exists()
    assert active_path.stat().st_size > 0


def test_directory_fsync_failure_on_create_produces_no_receipt(spool_home, monkeypatch):
    def _boom(_fd: int, _label) -> None:
        raise OSError("dir fsync failed")

    monkeypatch.setattr(spool, "_fsync_directory_fd", _boom)

    with pytest.raises(spool.SpoolDurabilityError):
        spool.append_records((_record(),))

    root, _, _, _ = _paths(spool_home)
    assert root.exists()


@pytest.mark.skipif(fcntl is None, reason="POSIX flock required")
def test_lock_timeout_is_bounded(spool_home, monkeypatch):
    root, _, lock_path, _ = _paths(spool_home)
    root.mkdir(mode=0o700)
    lock_path.touch(mode=0o600)

    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(spool, "LOCK_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(spool, "LOCK_RETRY_SECONDS", 0.01)
        with pytest.raises(spool.SpoolLockTimeoutError):
            spool.append_records((_record(),))


@pytest.mark.skipif(fcntl is None, reason="POSIX flock required")
def test_concurrent_writers_serialize_and_do_not_overlap_offsets(spool_home):
    records = [_record("unit-a"), _record("unit-b")]
    results = []
    errors = []

    def _worker(item):
        try:
            results.append(spool.append_records((item,)).unit_results[0].receipt)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(item,)) for item in records]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert len(results) == 2
    ordered = sorted(results, key=lambda receipt: receipt.offset)
    assert ordered[0].offset == 0
    assert ordered[1].offset == ordered[0].frame_length

    _, active_path, _, _ = _paths(spool_home)
    scan = spool.scan_spool(active_path)
    assert scan.tail_status is spool.SpoolTailStatus.CLEAN
    assert scan.frame_count == 2


def test_capacity_includes_quarantine_bytes_and_refuses_before_append(spool_home, monkeypatch):
    first = spool.append_records((_record("unit-a"),)).unit_results[0].receipt
    root, active_path, _, quarantine_path = _paths(spool_home)
    quarantine_spool = quarantine_path / "000001-clean-vp0.spool"
    quarantine_bytes = spool._frame_bytes_for_record(_record("unit-q"))
    quarantine_spool.write_bytes(quarantine_bytes)
    (quarantine_path / "000001-clean-vp0.json").write_text(
        json.dumps(
            {
                "sequence": 1,
                "tail_status": "clean",
                "valid_prefix_bytes": len(quarantine_bytes),
                "original_size": len(quarantine_bytes),
                "quarantined_at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    if os.name == "posix":
        os.chmod(quarantine_spool, 0o600)

    second_frame = spool._frame_bytes_for_record(_record("unit-b"))
    monkeypatch.setattr(
        spool,
        "TOTAL_CAP_BYTES",
        active_path.stat().st_size + quarantine_spool.stat().st_size + len(second_frame) - 1,
    )
    before = active_path.read_bytes()

    with pytest.raises(spool.SpoolCapacityError):
        spool.append_records((_record("unit-b"),))

    assert active_path.read_bytes() == before
    assert first.frame_length == len(before)


def _capacity_artifact_payload(*, family: str) -> bytes:
    if family in {"clean_sealed_spool", "prefix_sealed_spool"}:
        return b"X" * 257
    if family == "ack_json":
        return json.dumps(
            {
                "acked_prefix_bytes": 5,
                "last_frame_checksum_hex": "1" * 32,
                "last_frame_length": 5,
                "last_frame_offset": 0,
                "schema_version": 1,
                "segment_kind": "clean",
                "segment_name": "00000000000000000001.spool",
                "segment_sequence": "00000000000000000001",
                "segment_size_bytes": 5,
                "tail_status": "clean",
                "valid_prefix_bytes": 5,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    if family == "blocker_json":
        return json.dumps(
            {
                "acked_prefix_bytes": 0,
                "blocking_offset": 0,
                "evidence_sidecar_name": "seq-00000000000000000001-invalid_json-vp0.json",
                "evidence_spool_name": "seq-00000000000000000001-invalid_json-vp0.spool",
                "original_size_bytes": 5,
                "prefix_segment_name": None,
                "schema_version": 1,
                "segment_sequence": "00000000000000000001",
                "source_kind": "sealed",
                "tail_status": "invalid_json",
                "valid_prefix_bytes": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    if family == "highwater_json":
        return json.dumps(
            {
                "last_reserved_sequence": "00000000000000000001",
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    if family in {"replay_quarantine_json", "legacy_quarantine_json"}:
        return json.dumps(
            {
                "original_size": 5,
                "quarantined_at": 1.0,
                "sequence": 1,
                "tail_status": "clean",
                "valid_prefix_bytes": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    if family in {"replay_quarantine_spool", "legacy_quarantine_spool"}:
        return b"Q" * 257
    raise AssertionError(f"unknown capacity artifact family: {family}")


@pytest.mark.parametrize(
    ("family", "artifact_parts"),
    [
        pytest.param(
            "clean_sealed_spool",
            (spool.SEALED_DIR_NAME, "00000000000000000001.spool"),
            id="clean_sealed_spool",
        ),
        pytest.param(
            "prefix_sealed_spool",
            (spool.SEALED_DIR_NAME, "00000000000000000001.prefix.spool"),
            id="prefix_sealed_spool",
        ),
        pytest.param(
            "ack_json",
            (
                spool.SEALED_DIR_NAME,
                spool.ACKS_DIR_NAME,
                "00000000000000000001.spool.ap00000000000000000005.json",
            ),
            id="ack_json",
        ),
        pytest.param(
            "blocker_json",
            (
                spool.SEALED_DIR_NAME,
                spool.BLOCKERS_DIR_NAME,
                "00000000000000000001.blocker.json",
            ),
            id="blocker_json",
        ),
        pytest.param(
            "highwater_json",
            (spool.HIGHWATER_FILE_NAME,),
            id="highwater_json",
        ),
        pytest.param(
            "replay_quarantine_spool",
            (spool.QUARANTINE_DIR_NAME, "seq-00000000000000000001-clean-vp0.spool"),
            id="replay_quarantine_spool",
        ),
        pytest.param(
            "replay_quarantine_json",
            (spool.QUARANTINE_DIR_NAME, "seq-00000000000000000001-clean-vp0.json"),
            id="replay_quarantine_json",
        ),
        pytest.param(
            "legacy_quarantine_spool",
            (spool.QUARANTINE_DIR_NAME, "000001-clean-vp0.spool"),
            id="legacy_quarantine_spool",
        ),
        pytest.param(
            "legacy_quarantine_json",
            (spool.QUARANTINE_DIR_NAME, "000001-clean-vp0.json"),
            id="legacy_quarantine_json",
        ),
    ],
)
def test_capacity_artifact_family_is_counted(
    spool_home,
    monkeypatch,
    family,
    artifact_parts,
):
    root, active_path, _, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    active_path.write_bytes(b"")

    artifact_path = root.joinpath(*artifact_parts)
    artifact_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    artifact_bytes = _capacity_artifact_payload(family=family)
    artifact_path.write_bytes(artifact_bytes)
    if family in {"replay_quarantine_spool", "legacy_quarantine_spool"}:
        artifact_path.with_suffix(".json").write_bytes(b"")

    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(active_path, 0o600)
        current = artifact_path.parent
        while current != root:
            os.chmod(current, 0o700)
            current = current.parent
        os.chmod(artifact_path, 0o600)
        if family in {"replay_quarantine_spool", "legacy_quarantine_spool"}:
            os.chmod(artifact_path.with_suffix(".json"), 0o600)

    requested_frame = spool._frame_bytes_for_record(_record(f"unit-{family}"))
    monkeypatch.setattr(
        spool,
        "TOTAL_CAP_BYTES",
        len(artifact_bytes) + len(requested_frame) - 1,
    )
    before = active_path.read_bytes()

    with pytest.raises(spool.SpoolCapacityError):
        spool.append_records((_record(f"unit-{family}"),))

    assert before == b""
    assert active_path.read_bytes() == b""


def test_capacity_inventory_excludes_lock_and_protocol_temp(spool_home, monkeypatch):
    root, active_path, lock_path, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    active_path.write_bytes(b"")
    lock_path.write_bytes(b"L" * 4096)
    owner_lock_path = root / spool.REPLAY_OWNER_LOCK_NAME
    owner_lock_path.write_bytes(b"O" * 4096)
    protocol_temp = root / ".segment-sequence.highwater.json.123.456.tmp"
    highwater_payload = json.dumps(
        {
            "last_reserved_sequence": "00000000000000000001",
            "schema_version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    protocol_temp.write_bytes(highwater_payload + (b" " * 5000))

    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(active_path, 0o600)
        os.chmod(lock_path, 0o600)
        os.chmod(owner_lock_path, 0o600)
        os.chmod(protocol_temp, 0o600)

    requested_frame = spool._frame_bytes_for_record(_record("unit-capacity-exclude"))
    monkeypatch.setattr(spool, "TOTAL_CAP_BYTES", len(requested_frame))

    result = spool.append_records((_record("unit-capacity-exclude"),))

    assert len(result.unit_results) == 1
    assert result.unit_results[0].receipt.offset == 0
    assert result.unit_results[0].receipt.frame_length == len(requested_frame)
    assert active_path.read_bytes() == requested_frame


@pytest.mark.parametrize(
    ("anomaly", "expected_exc", "match"),
    [
        pytest.param(
            "sealed_symlink",
            spool.SpoolPathSecurityError,
            "symlinked fallback spool path refused",
            id="sealed_symlink",
        ),
        pytest.param(
            "sealed_fifo",
            spool.SpoolPathSecurityError,
            "not a regular file",
            id="sealed_fifo",
            marks=pytest.mark.skipif(os.name != "posix", reason="FIFO anomaly is POSIX-only"),
        ),
        pytest.param(
            "sealed_unexpected_dir",
            spool.SpoolPathSecurityError,
            "unexpected fallback spool directory encountered during sequence inventory",
            id="sealed_unexpected_dir",
        ),
        pytest.param(
            "sealed_unrecognized_file",
            spool.SpoolDurabilityError,
            "unrecognized sealed segment artifact",
            id="sealed_unrecognized_file",
        ),
        pytest.param(
            "root_unrecognized_file",
            spool.SpoolDurabilityError,
            "unrecognized sequence-bearing artifact",
            id="root_unrecognized_file",
        ),
    ],
)
def test_capacity_inventory_fails_closed(spool_home, anomaly, expected_exc, match):
    root, active_path, _, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    active_path.write_bytes(b"")

    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    anomaly_path = (
        root / "unexpected.bin"
        if anomaly == "root_unrecognized_file"
        else sealed_dir
        / (
            "unexpected"
            if anomaly == "sealed_unexpected_dir"
            else "unexpected.bin"
            if anomaly == "sealed_unrecognized_file"
            else "00000000000000000001.spool"
        )
    )

    if anomaly == "sealed_symlink":
        outside = spool_home / "outside-target.spool"
        outside.write_bytes(b"outside")
        anomaly_path.symlink_to(outside)
    elif anomaly == "sealed_fifo":
        os.mkfifo(anomaly_path)
    elif anomaly == "sealed_unexpected_dir":
        anomaly_path.mkdir(mode=0o700)
    else:
        anomaly_path.write_bytes(b"unexpected")

    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(active_path, 0o600)
        os.chmod(sealed_dir, 0o700)
        if anomaly in {"sealed_unexpected_dir"}:
            os.chmod(anomaly_path, 0o700)
        elif anomaly not in {"sealed_symlink", "sealed_fifo"}:
            os.chmod(anomaly_path, 0o600)

    before = active_path.read_bytes()

    with pytest.raises(expected_exc, match=match):
        spool.append_records((_record(f"unit-{anomaly}"),))

    assert before == b""
    assert active_path.read_bytes() == b""


_CORRUPT_TAIL_CASES = (
    (
        spool.SpoolTailStatus.INCOMPLETE_EOF,
        lambda frame: frame[:-1],
    ),
    (
        spool.SpoolTailStatus.BAD_MAGIC,
        lambda frame: b"NOPE" + frame[4:],
    ),
    (
        spool.SpoolTailStatus.BAD_VERSION,
        lambda frame: frame[:4] + b"\x02" + frame[5:],
    ),
    (
        spool.SpoolTailStatus.OVERSIZED_LENGTH,
        lambda frame: frame[:8]
        + (spool.MAX_PAYLOAD_BYTES + 1).to_bytes(8, "big")
        + frame[16:],
    ),
    (
        spool.SpoolTailStatus.CHECKSUM_MISMATCH,
        lambda frame: frame[:20] + bytes([frame[20] ^ 0xFF]) + frame[21:],
    ),
    (
        spool.SpoolTailStatus.INVALID_JSON,
        lambda _frame: spool._frame_from_payload_bytes(b"{"),
    ),
    (
        spool.SpoolTailStatus.INVALID_SCHEMA,
        lambda _frame: spool._frame_from_payload_bytes(b'{"schema_version":1}'),
    ),
)


@pytest.mark.parametrize(("status", "corrupt"), _CORRUPT_TAIL_CASES)
def test_corrupt_tails_are_quarantined_with_valid_prefix(
    spool_home,
    status,
    corrupt,
):
    root, active_path, _, quarantine_path = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(root, 0o700)
        os.chmod(quarantine_path, 0o700)

    clean_frame = spool._frame_bytes_for_record(_record("unit-clean"))
    corrupt_frame = corrupt(spool._frame_bytes_for_record(_record("unit-bad")))
    original_bytes = clean_frame + corrupt_frame
    active_path.write_bytes(original_bytes)
    if os.name == "posix":
        os.chmod(active_path, 0o600)

    result = spool.append_records((_record("unit-fresh"),))
    receipt = result.unit_results[0].receipt

    quarantine_spools = sorted(quarantine_path.glob("*.spool"))
    quarantine_sidecars = sorted(quarantine_path.glob("*.json"))
    assert len(quarantine_spools) == 1
    assert len(quarantine_sidecars) == 1
    assert quarantine_spools[0].read_bytes() == original_bytes
    assert f"vp{len(clean_frame)}" in quarantine_spools[0].name

    meta = json.loads(quarantine_sidecars[0].read_text(encoding="utf-8"))
    assert meta["sequence"] == 1
    assert meta["tail_status"] == status.value
    assert meta["valid_prefix_bytes"] == len(clean_frame)
    assert meta["original_size"] == len(original_bytes)
    assert "quarantined_at" in meta

    scan = spool.scan_spool(active_path)
    assert scan.tail_status is spool.SpoolTailStatus.CLEAN
    assert scan.frame_count == 1
    assert receipt.offset == 0


def test_existing_active_append_does_not_require_directory_fsync(spool_home, monkeypatch):
    first = spool.append_records((_record("unit-a"),)).unit_results[0].receipt

    def _boom(_path: Path) -> None:  # pragma: no cover - should never fire
        raise AssertionError("directory fsync should not run for an existing active append")

    monkeypatch.setattr(spool, "_fsync_directory", _boom)
    second = spool.append_records((_record("unit-b"),)).unit_results[0].receipt

    assert second.offset == first.frame_length


def test_multi_unit_partial_append_returns_durable_prefix_only(spool_home, monkeypatch):
    real_write = spool.os.write
    second_unit = {"active": False, "started": False}

    def _flaky_write(fd: int, data: bytes) -> int:
        if not second_unit["started"]:
            if data.startswith(b"HSPL") and second_unit["active"]:
                second_unit["started"] = True
                wrote = real_write(fd, data[:7])
                raise OSError("interrupted write")
            if data.startswith(b"HSPL"):
                second_unit["active"] = True
        return real_write(fd, data)

    monkeypatch.setattr(spool.os, "write", _flaky_write)

    with pytest.raises(spool.SpoolAppendAttemptPartialError) as excinfo:
        spool.append_records((_record("unit-a"), _record("unit-b", attempt_index=1)))

    err = excinfo.value
    assert len(err.durable_results) == 1
    assert err.durable_results[0].persistence_unit_id == "unit-a"

    _, active_path, _, _ = _paths(spool_home)
    scan = spool.scan_spool(active_path)
    assert scan.valid_prefix_bytes == err.durable_results[0].receipt.frame_length
    assert scan.tail_status is spool.SpoolTailStatus.INCOMPLETE_EOF


def _close_runtime(runtime) -> None:
    spool._close_fd_quietly(runtime.lock_fd)
    spool._close_fd_quietly(runtime.root_fd)
    spool._close_fd_quietly(runtime.home_fd)


def test_replay_owner_lock_is_exclusive_and_supports_takeover_after_release(spool_home):
    first_runtime = spool._open_locked_runtime()
    second_runtime = spool._open_locked_runtime()
    try:
        owner = spool._try_acquire_replay_owner(first_runtime)
        assert owner is not None
        assert spool._try_acquire_replay_owner(second_runtime) is None

        spool._close_fd_quietly(owner.fd)
        takeover = spool._try_acquire_replay_owner(second_runtime)
        assert takeover is not None
        spool._close_fd_quietly(takeover.fd)
    finally:
        _close_runtime(second_runtime)
        _close_runtime(first_runtime)


def test_segment_sequence_highwater_never_reuses_reserved_values(spool_home):
    runtime = spool._open_locked_runtime()
    sealed_fd = -1
    try:
        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            sealed_fd, _ = spool._open_dir_at(
                runtime.root_fd,
                spool.SEALED_DIR_NAME,
                full_path=spool._sealed_dir(),
                mode=spool.ROOT_MODE,
                create=True,
                parent_label=runtime.root_path,
                fsync_parent_on_open_existing=True,
            )
            first = spool._allocate_next_segment_sequence(
                runtime=runtime,
                root_fd=runtime.root_fd,
            )
            assert first == 1
            (spool._sealed_dir() / f"{first:020d}.spool").write_bytes(b"reserved-gap")

            second = spool._allocate_next_segment_sequence(
                runtime=runtime,
                root_fd=runtime.root_fd,
            )
            assert second == 2

        highwater = json.loads(
            spool._segment_sequence_highwater_path().read_text(encoding="utf-8")
        )
        assert highwater == {
            "last_reserved_sequence": "00000000000000000002",
            "schema_version": 1,
        }
    finally:
        if sealed_fd >= 0:
            spool._close_fd_quietly(sealed_fd)
        _close_runtime(runtime)


def test_decode_spool_segment_returns_full_record_and_frame_metadata(spool_home):
    record = _record("unit-decode", contents=("hello", "world"))
    frame = spool._frame_bytes_for_record(record)
    segment_path = spool_home / "segment.spool"
    segment_path.write_bytes(frame)

    decoded = spool.decode_spool_segment(segment_path)

    assert decoded.valid_prefix_bytes == len(frame)
    assert decoded.tail_status is spool.SpoolTailStatus.CLEAN
    assert decoded.tail_offset is None
    assert len(decoded.prefix_frames) == 1

    decoded_frame = decoded.prefix_frames[0]
    assert decoded_frame.record == record
    assert decoded_frame.frame_offset == 0
    assert decoded_frame.frame_length == len(frame)
    assert decoded_frame.payload_length == len(frame) - spool.HEADER_SIZE
    assert decoded_frame.checksum_hex == frame[16:32].hex()


def test_allocate_next_segment_sequence_reconstructs_from_all_artifact_families(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        root = spool._spool_root()
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        blockers_dir = spool._blockers_dir()
        quarantine_dir = spool._quarantine_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        blockers_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        (sealed_dir / "00000000000000000003.spool").write_bytes(b"clean")
        (sealed_dir / "00000000000000000004.prefix.spool").write_bytes(b"prefix")
        (acks_dir / "00000000000000000007.spool.ap00000000000000000099.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (blockers_dir / "00000000000000000008.blocker.json").write_text("{}", encoding="utf-8")
        (quarantine_dir / "seq-00000000000000000009-checksum_mismatch-vp10.spool").write_bytes(
            b"evidence"
        )
        (quarantine_dir / "seq-00000000000000000010-invalid_json-vp0.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (root / ".segment-sequence.highwater.json.123.456.tmp").write_text(
            json.dumps(
                {
                    "last_reserved_sequence": "00000000000000000011",
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        spool._segment_sequence_highwater_path().write_text(
            json.dumps(
                {
                    "last_reserved_sequence": "00000000000000000002",
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            next_sequence = spool._allocate_next_segment_sequence(
                runtime=runtime,
                root_fd=runtime.root_fd,
            )

        assert next_sequence == 12
    finally:
        _close_runtime(runtime)


def test_malformed_sequence_bearing_name_blocks_replay(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        (sealed_dir / "not-a-sequence.spool").write_bytes(b"bad")

        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            with pytest.raises(spool.SpoolDurabilityError, match="unrecognized sealed segment artifact"):
                spool._allocate_next_segment_sequence(
                    runtime=runtime,
                    root_fd=runtime.root_fd,
                )
    finally:
        _close_runtime(runtime)


def test_malformed_sequence_bearing_protocol_temp_blocks_replay(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        root = spool._spool_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / ".mystery-sequence-00000000000000000012.123.456.tmp").write_text(
            "bad-temp",
            encoding="utf-8",
        )

        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            with pytest.raises(spool.SpoolDurabilityError, match="unrecognized protocol temp artifact"):
                spool._allocate_next_segment_sequence(
                    runtime=runtime,
                    root_fd=runtime.root_fd,
                )
    finally:
        _close_runtime(runtime)


@pytest.mark.skipif(os.name != "posix", reason="directory-swap security checks are POSIX-only")
@pytest.mark.parametrize(
    ("target", "target_parts"),
    [
        ("root_swap", ()),
        ("sealed_swap", (spool.SEALED_DIR_NAME,)),
        ("acks_swap", (spool.SEALED_DIR_NAME, spool.ACKS_DIR_NAME)),
        ("blockers_swap", (spool.SEALED_DIR_NAME, spool.BLOCKERS_DIR_NAME)),
        ("quarantine_swap", (spool.QUARANTINE_DIR_NAME,)),
    ],
)
def test_replay_security_checks_reject_root_swap_for_sealed_acks_blockers_and_quarantine(
    spool_home,
    target,
    target_parts,
):
    runtime = spool._open_locked_runtime()
    try:
        root = spool._spool_root()
        sealed = spool._sealed_dir()
        acks = spool._acks_dir()
        blockers = spool._blockers_dir()
        quarantine = spool._quarantine_dir()
        sealed.mkdir(parents=True, exist_ok=True)
        acks.mkdir(parents=True, exist_ok=True)
        blockers.mkdir(parents=True, exist_ok=True)
        quarantine.mkdir(parents=True, exist_ok=True)

        external = spool_home / f"{target}-external"
        external.mkdir()
        target_path = root.joinpath(*target_parts) if target_parts else root
        parked = target_path.with_name(target_path.name + ".real")
        os.replace(target_path, parked)
        os.symlink(external, target_path, target_is_directory=True)

        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            with pytest.raises(spool.SpoolPathSecurityError):
                spool._allocate_next_segment_sequence(
                    runtime=runtime,
                    root_fd=runtime.root_fd,
                )

        assert not (external / spool.HIGHWATER_FILE_NAME).exists()
    finally:
        _close_runtime(runtime)


def test_inventory_fd_repeated_exceptions_return_to_baseline(spool_home):
    baseline = _fd_count()
    root = spool._spool_root()
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        runtime = spool._open_locked_runtime()
        temp_path = root / f".mystery-sequence-0000000000000000{attempt:04d}.123.456.tmp"
        temp_path.write_text("bad-temp", encoding="utf-8")
        try:
            with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
                with pytest.raises(spool.SpoolDurabilityError, match="unrecognized protocol temp artifact"):
                    spool._allocate_next_segment_sequence(
                        runtime=runtime,
                        root_fd=runtime.root_fd,
                    )
        finally:
            _close_runtime(runtime)
            temp_path.unlink(missing_ok=True)
    assert _fd_count() == baseline


def _ack_payload(*, segment_sequence: int, segment_name: str, acked_prefix_bytes: int, valid_prefix_bytes: int, tail_status: str = "clean", segment_kind: str = "clean"):
    return {
        "schema_version": 1,
        "segment_sequence": f"{segment_sequence:020d}",
        "segment_name": segment_name,
        "segment_kind": segment_kind,
        "segment_size_bytes": valid_prefix_bytes,
        "acked_prefix_bytes": acked_prefix_bytes,
        "valid_prefix_bytes": valid_prefix_bytes,
        "tail_status": tail_status,
        "last_frame_offset": 0,
        "last_frame_length": acked_prefix_bytes,
        "last_frame_checksum_hex": "1" * 32,
    }


def test_publish_ack_same_prefix_same_content_is_idempotent_success(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(b"alpha")
        payload = _ack_payload(
            segment_sequence=1,
            segment_name=segment_path.name,
            acked_prefix_bytes=5,
            valid_prefix_bytes=5,
        )

        spool._publish_ack_sidecar_strict(
            runtime,
            segment_sequence=1,
            segment_path=segment_path,
            ack_payload=payload,
        )
        spool._publish_ack_sidecar_strict(
            runtime,
            segment_sequence=1,
            segment_path=segment_path,
            ack_payload=payload,
        )

        assert sorted(acks_dir.glob("*.json")) == [acks_dir / "00000000000000000001.spool.ap00000000000000000005.json"]
    finally:
        _close_runtime(runtime)


def test_publish_ack_same_prefix_different_content_blocks_integrity(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(b"alpha")
        spool._write_sidecar_json(
            acks_dir / "00000000000000000001.spool.ap00000000000000000005.json",
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=5,
                valid_prefix_bytes=5,
            ),
        )

        with pytest.raises(spool.SpoolDurabilityError, match="conflicting ack sidecar"):
            spool._publish_ack_sidecar_strict(
                runtime,
                segment_sequence=1,
                segment_path=segment_path,
                ack_payload=_ack_payload(
                    segment_sequence=1,
                    segment_name=segment_path.name,
                    acked_prefix_bytes=5,
                    valid_prefix_bytes=5,
                    tail_status="checksum_mismatch",
                ),
            )
    finally:
        _close_runtime(runtime)


def test_malformed_or_oversized_ack_sidecar_blocks_integrity(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(b"alpha")
        malformed = acks_dir / "00000000000000000001.spool.ap00000000000000000005.json"
        malformed.write_text("{}", encoding="utf-8")

        with pytest.raises(spool.SpoolDurabilityError, match="invalid ack sidecar"):
            spool._load_ack_sidecar_winner(runtime=runtime, segment_path=segment_path)

        malformed.unlink()
        oversized = acks_dir / "00000000000000000001.spool.ap00000000000000000005.json"
        oversized.write_text("x" * 3000, encoding="utf-8")
        with pytest.raises(spool.SpoolDurabilityError, match="invalid ack sidecar"):
            spool._load_ack_sidecar_winner(runtime=runtime, segment_path=segment_path)
    finally:
        _close_runtime(runtime)


def test_highest_valid_ack_winner_ignores_lower_or_mismatched_sidecars(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(b"alpha")
        spool._write_sidecar_json(
            acks_dir / "00000000000000000001.spool.ap00000000000000000003.json",
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=3,
                valid_prefix_bytes=5,
            ),
        )
        spool._write_sidecar_json(
            acks_dir / "00000000000000000001.spool.ap00000000000000000005.json",
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=5,
                valid_prefix_bytes=5,
            ),
        )
        spool._write_sidecar_json(
            acks_dir / "00000000000000000009.spool.ap00000000000000000099.json",
            _ack_payload(
                segment_sequence=9,
                segment_name="00000000000000000009.spool",
                acked_prefix_bytes=99,
                valid_prefix_bytes=99,
            ),
        )

        winner = spool._load_ack_sidecar_winner(runtime=runtime, segment_path=segment_path)

        assert winner is not None
        assert winner["acked_prefix_bytes"] == 5
    finally:
        _close_runtime(runtime)


def test_ack_sidecar_count_above_64_blocks_integrity(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(b"a" * 65)
        for idx in range(1, 66):
            spool._write_sidecar_json(
                acks_dir / f"00000000000000000001.spool.ap{idx:020d}.json",
                _ack_payload(
                    segment_sequence=1,
                    segment_name=segment_path.name,
                    acked_prefix_bytes=idx,
                    valid_prefix_bytes=65,
                ),
            )

        with pytest.raises(spool.SpoolDurabilityError, match="too many ack sidecars"):
            spool._load_ack_sidecar_winner(runtime=runtime, segment_path=segment_path)
    finally:
        _close_runtime(runtime)


def test_replay_to_session_db_uses_strict_ack_publication(spool_home, monkeypatch):
    calls = []

    def _strict(runtime, *, segment_sequence, segment_path, ack_payload):
        calls.append(
            {
                "runtime": runtime,
                "segment_sequence": segment_sequence,
                "segment_path": segment_path,
                "ack_payload": ack_payload,
            }
        )

    monkeypatch.setattr(spool, "_publish_ack_sidecar_strict", _strict)
    monkeypatch.setattr(spool, "_delete_fully_acked_segment", lambda *_args, **_kwargs: None)

    from hermes_state import SessionDB

    db = SessionDB(db_path=spool_home / "state.db")
    try:
        monkeypatch.setenv("HERMES_HOME", str(spool_home / ".hermes"))
        hermes_home = spool_home / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        sealed_dir = hermes_home / spool.SPOOL_ROOT_NAME / spool.SEALED_DIR_NAME
        sealed_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(spool._frame_bytes_for_record(_record()))

        result = spool.replay_to_session_db(db, trigger="startup")

        assert result.state is spool.ReplayRunState.REPLAYED
        assert len(calls) == 1
        assert calls[0]["segment_sequence"] == 1
        assert calls[0]["segment_path"].name == "00000000000000000001.spool"
    finally:
        db.close()


def test_stale_lower_ack_sidecars_cleanup_only_after_durable_higher_winner(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / "00000000000000000001.spool"
        segment_path.write_bytes(b"alpha")
        spool._write_sidecar_json(
            acks_dir / "00000000000000000001.spool.ap00000000000000000003.json",
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=3,
                valid_prefix_bytes=5,
            ),
        )

        spool._publish_ack_sidecar_strict(
            runtime,
            segment_sequence=1,
            segment_path=segment_path,
            ack_payload=_ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=5,
                valid_prefix_bytes=5,
            ),
        )

        assert sorted(path.name for path in acks_dir.glob("*.json")) == [
            "00000000000000000001.spool.ap00000000000000000005.json"
        ]
    finally:
        _close_runtime(runtime)


def test_corrupt_active_with_valid_prefix_publishes_prefix_evidence_and_blocker(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        root = spool._spool_root()
        sealed_dir = spool._sealed_dir()
        blockers_dir = spool._blockers_dir()
        quarantine_dir = spool._quarantine_dir()
        root.mkdir(parents=True, exist_ok=True)
        sealed_dir.mkdir(parents=True, exist_ok=True)
        blockers_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        clean_frame = spool._frame_bytes_for_record(_record("unit-clean"))
        corrupt_frame = bytearray(spool._frame_bytes_for_record(_record("unit-bad", attempt_index=1)))
        corrupt_frame[-1] ^= 0x01
        active_path = root / spool.ACTIVE_SPOOL_NAME
        active_path.write_bytes(clean_frame + bytes(corrupt_frame))

        runtime = spool._open_locked_runtime()
        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            result = spool._reconcile_active_spool_for_replay(runtime)

        assert result["tail_status"] is spool.SpoolTailStatus.CHECKSUM_MISMATCH
        assert result["valid_prefix_bytes"] == len(clean_frame)
        assert result["segment_sequence"] == 1
        assert result["prefix_segment_name"] == "00000000000000000001.prefix.spool"
        assert (sealed_dir / "00000000000000000001.prefix.spool").read_bytes() == clean_frame
        evidence_spools = sorted(quarantine_dir.glob("seq-00000000000000000001-*.spool"))
        assert len(evidence_spools) == 1
        assert evidence_spools[0].read_bytes() == clean_frame + bytes(corrupt_frame)
        blocker = json.loads((blockers_dir / "00000000000000000001.blocker.json").read_text(encoding="utf-8"))
        assert blocker["prefix_segment_name"] == "00000000000000000001.prefix.spool"
        assert blocker["blocking_offset"] == len(clean_frame)
        assert active_path.exists()
        assert active_path.read_bytes() == b""
    finally:
        _close_runtime(runtime)


def test_corrupt_active_with_zero_prefix_publishes_evidence_and_blocker(spool_home):
    runtime = spool._open_locked_runtime()
    try:
        root = spool._spool_root()
        blockers_dir = spool._blockers_dir()
        quarantine_dir = spool._quarantine_dir()
        root.mkdir(parents=True, exist_ok=True)
        blockers_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        corrupt_frame = bytearray(spool._frame_bytes_for_record(_record("unit-bad")))
        corrupt_frame[0] = 0x00
        active_path = root / spool.ACTIVE_SPOOL_NAME
        active_path.write_bytes(bytes(corrupt_frame))

        runtime = spool._open_locked_runtime()
        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            result = spool._reconcile_active_spool_for_replay(runtime)

        assert result["tail_status"] is spool.SpoolTailStatus.BAD_MAGIC
        assert result["valid_prefix_bytes"] == 0
        assert result["segment_sequence"] == 1
        assert result["prefix_segment_name"] is None
        evidence_spools = sorted(quarantine_dir.glob("seq-00000000000000000001-*.spool"))
        assert len(evidence_spools) == 1
        assert evidence_spools[0].read_bytes() == bytes(corrupt_frame)
        blocker = json.loads((blockers_dir / "00000000000000000001.blocker.json").read_text(encoding="utf-8"))
        assert blocker["prefix_segment_name"] is None
        assert blocker["blocking_offset"] == 0
        assert active_path.exists()
        assert active_path.read_bytes() == b""
    finally:
        _close_runtime(runtime)


def _write_status_blocker_state(home: Path) -> dict[str, object]:
    root = home / spool.SPOOL_ROOT_NAME
    sealed = root / spool.SEALED_DIR_NAME
    blockers = sealed / spool.BLOCKERS_DIR_NAME
    quarantine = root / spool.QUARANTINE_DIR_NAME
    lock_path = root / spool.LOCK_FILE_NAME
    root.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    blockers.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")

    clean_frame = spool._frame_bytes_for_record(_record("status-blocked"))
    corrupt_frame = bytearray(spool._frame_bytes_for_record(_record("status-bad", attempt_index=1)))
    corrupt_frame[-1] ^= 0x01
    prefix_path = sealed / "00000000000000000001.prefix.spool"
    prefix_path.write_bytes(clean_frame)

    evidence_base = f"seq-{1:020d}-checksum_mismatch-vp{len(clean_frame)}"
    evidence_spool = quarantine / f"{evidence_base}.spool"
    evidence_sidecar = quarantine / f"{evidence_base}.json"
    evidence_spool.write_bytes(clean_frame + bytes(corrupt_frame))
    evidence_sidecar.write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": f"{1:020d}",
                "source_kind": "sealed",
                "tail_status": "checksum_mismatch",
                "valid_prefix_bytes": len(clean_frame),
                "original_size_bytes": len(clean_frame) + len(corrupt_frame),
                "evidence_spool_name": evidence_spool.name,
            }
        )
    )
    blocker_path = blockers / "00000000000000000001.blocker.json"
    blocker_path.write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": f"{1:020d}",
                "source_kind": "sealed",
                "tail_status": "checksum_mismatch",
                "valid_prefix_bytes": len(clean_frame),
                "acked_prefix_bytes": 0,
                "blocking_offset": len(clean_frame),
                "prefix_segment_name": prefix_path.name,
                "evidence_spool_name": evidence_spool.name,
                "evidence_sidecar_name": evidence_sidecar.name,
                "original_size_bytes": len(clean_frame) + len(corrupt_frame),
            }
        )
    )
    return {
        "prefix_path": prefix_path,
        "blocker_path": blocker_path,
        "evidence_spool": evidence_spool,
        "evidence_sidecar": evidence_sidecar,
        "blocking_offset": len(clean_frame),
    }


def _status_scandir_matches(target: object, expected_path: Path) -> bool:
    return isinstance(target, int) and spool._same_file_stat(os.fstat(target), expected_path.stat())


class _StatusFaultingDirEntry:
    def __init__(
        self,
        wrapped,
        *,
        stat_exc=None,
        symlink_exc=None,
    ):
        self._wrapped = wrapped
        self._stat_exc = stat_exc
        self._symlink_exc = symlink_exc
        self.name = wrapped.name

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def is_symlink(self):
        if self._symlink_exc is not None:
            raise self._symlink_exc
        return self._wrapped.is_symlink()

    def stat(self, *, follow_symlinks=True):
        if self._stat_exc is not None:
            raise self._stat_exc
        return self._wrapped.stat(follow_symlinks=follow_symlinks)


class _StatusFaultingScandir:
    def __init__(
        self,
        entries,
        *,
        iteration_exc=None,
        raise_after: int = 0,
    ):
        self._entries = iter(entries)
        self._iteration_exc = iteration_exc
        self._raise_after = raise_after
        self._yielded = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._iteration_exc is not None and self._yielded >= self._raise_after:
            raise self._iteration_exc
        entry = next(self._entries)
        self._yielded += 1
        return entry

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        return None


def test_collect_session_fallback_spool_status_missing_root_is_empty_and_non_creating(
    spool_home, monkeypatch
):
    before = sorted(path.relative_to(spool_home) for path in spool_home.rglob("*"))

    monkeypatch.setattr(
        spool.os,
        "fstatvfs",
        lambda _fd: SimpleNamespace(f_bavail=0, f_blocks=10, f_frsize=1),
    )

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.schema_version == 1
    assert status.state == "empty"
    assert status.reasons == ()
    assert status.pending_units == 0
    assert status.pending_frames == 0
    assert status.pending_bytes == 0
    assert status.oldest_pending_age_seconds is None
    assert status.retry_pending is False
    assert status.ack_pending is False
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_offset is None
    assert status.blocker_reason_class is None
    assert status.blocker_source_kind is None
    assert status.inspection_error_class is None
    assert not (spool_home / spool.SPOOL_ROOT_NAME).exists()
    after = sorted(path.relative_to(spool_home) for path in spool_home.rglob("*"))
    assert after == before


def test_collect_session_fallback_spool_status_counts_clean_backlog_age_and_capacity(spool_home):
    root, active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")

    first = spool._frame_bytes_for_record(_record("status-unit-a"))
    second = spool._frame_bytes_for_record(_record("status-unit-b", attempt_index=1))
    active = spool._frame_bytes_for_record(_record("status-unit-c", attempt_index=2))
    segment_path = sealed_dir / "00000000000000000001.spool"
    segment_path.write_bytes(first + second)
    active_path.write_bytes(active)

    os.utime(segment_path, (75.0, 75.0))
    os.utime(active_path, (90.0, 90.0))

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.reasons == ("pending_backlog",)
    assert status.pending_units == 3
    assert status.pending_frames == 3
    assert status.pending_bytes == len(first + second) + len(active)
    assert status.oldest_pending_age_seconds == pytest.approx(25.0)
    assert status.capacity_used_bytes == len(first + second) + len(active)
    assert status.capacity_cap_bytes == spool.TOTAL_CAP_BYTES
    assert status.capacity_remaining_bytes == spool.TOTAL_CAP_BYTES - (len(first + second) + len(active))
    assert status.capacity_state == "ok"
    assert status.blocker_present is False
    assert status.retry_pending is False
    assert status.ack_pending is False


def test_collect_session_fallback_spool_status_reports_blocker_metadata_without_mutation(spool_home):
    paths = _write_status_blocker_state(spool_home)
    before = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in paths.items()
        if isinstance(path, Path)
    }

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.blocker_present is True
    assert status.blocker_sequence == 1
    assert status.blocker_offset == paths["blocking_offset"]
    assert status.blocker_reason_class == "checksum_mismatch"
    assert status.blocker_source_kind == "sealed"
    assert "blocker" in status.reasons
    assert "pending_backlog" in status.reasons

    after = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in paths.items()
        if isinstance(path, Path)
    }
    assert after == before


def test_collect_session_fallback_spool_status_missing_append_lock_is_inspection_only(
    spool_home,
):
    root, active_path, _lock_path, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(spool._frame_bytes_for_record(_record("missing-lock")))

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "missing_append_lock"
    assert "inspection_error" in status.reasons
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_offset is None
    assert status.blocker_reason_class is None
    assert status.blocker_source_kind is None


def test_collect_session_fallback_spool_status_fails_closed_for_invalid_blocker_json(spool_home):
    paths = _write_status_blocker_state(spool_home)
    blocker_path = Path(str(paths["blocker_path"]))
    blocker_path.write_text('{"schema_version":1}\n', encoding="utf-8")

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert "inspection_error" in status.reasons
    assert status.inspection_error_class == "invalid_blocker_json"
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_offset is None
    assert status.blocker_reason_class is None
    assert status.blocker_source_kind is None


def test_collect_session_fallback_spool_status_reports_capacity_and_disk_from_home_descriptor(
    spool_home, monkeypatch
):
    root, active_path, lock_path, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    active = spool._frame_bytes_for_record(_record("status-capacity"))
    active_path.write_bytes(active)
    captured = {}

    def _fake_fstatvfs(fd):
        captured["fd"] = fd
        return SimpleNamespace(f_bavail=0, f_blocks=10, f_frsize=8)

    monkeypatch.setattr(spool.os, "fstatvfs", _fake_fstatvfs)
    monkeypatch.setattr(spool, "TOTAL_CAP_BYTES", len(active) + 1)

    constrained = spool.collect_session_fallback_spool_status(now=100.0)

    assert isinstance(captured["fd"], int)
    assert constrained.capacity_remaining_bytes == 1
    assert constrained.capacity_state == "constrained"
    assert constrained.disk_free_bytes == 0
    assert constrained.disk_total_bytes == 80
    assert constrained.disk_headroom_threshold_bytes == 1
    assert constrained.disk_state == "low"

    monkeypatch.setattr(spool, "TOTAL_CAP_BYTES", len(active))
    full = spool.collect_session_fallback_spool_status(now=100.0)
    assert full.capacity_remaining_bytes == 0
    assert full.capacity_state == "full"


def test_collect_session_fallback_spool_status_degrades_when_disk_probe_fails(spool_home, monkeypatch):
    root, active_path, lock_path, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    active_path.write_bytes(spool._frame_bytes_for_record(_record("status-disk-fail")))

    def _boom(_fd):
        raise OSError(errno.EIO, "boom disk path should stay hidden")

    monkeypatch.setattr(spool.os, "fstatvfs", _boom)
    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert "inspection_error" in status.reasons
    assert status.inspection_error_class == "disk_probe_failed"
    assert status.disk_state == "unknown"
    assert status.disk_free_bytes is None
    assert status.disk_total_bytes is None
    assert "boom disk path should stay hidden" not in repr(status)


def test_collect_session_fallback_spool_status_valid_blocker_survives_later_disk_probe_failure(
    spool_home, monkeypatch
):
    _write_status_blocker_state(spool_home)

    def _boom(_fd):
        raise OSError(errno.EIO, "later disk probe failure")

    monkeypatch.setattr(spool.os, "fstatvfs", _boom)
    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "disk_probe_failed"
    assert status.blocker_present is True
    assert status.blocker_sequence == 1
    assert status.blocker_reason_class == "checksum_mismatch"
    assert status.blocker_source_kind == "sealed"


def test_collect_session_fallback_spool_status_artifact_bound_exhaustion_is_degraded(spool_home, monkeypatch):
    root, active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    (root / spool.HIGHWATER_FILE_NAME).write_text(
        '{"last_reserved_sequence":"00000000000000000001","schema_version":1}',
        encoding="utf-8",
    )
    (sealed_dir / "00000000000000000001.spool").write_bytes(
        spool._frame_bytes_for_record(_record("status-bound-sealed", attempt_index=1))
    )

    monkeypatch.setattr(spool, "STATUS_SCAN_ARTIFACT_LIMIT", 1)
    monkeypatch.setattr(
        spool,
        "_durable_capacity_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("status bound should preflight before capacity inventory")),
    )
    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "artifact_limit_exceeded"
    assert "inspection_error" in status.reasons
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_offset is None
    assert status.blocker_reason_class is None
    assert status.blocker_source_kind is None


def test_collect_session_fallback_spool_status_fully_acked_segment_reports_ack_pending_without_backlog(
    spool_home,
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    segment_path = sealed_dir / "00000000000000000001.spool"
    frame_bytes = spool._frame_bytes_for_record(_record("ack-only"))
    segment_path.write_bytes(frame_bytes)
    os.utime(segment_path, (75.0, 75.0))
    decoded = spool.decode_spool_segment(segment_path)
    frame = decoded.prefix_frames[0]
    acks_dir = sealed_dir / spool.ACKS_DIR_NAME
    acks_dir.mkdir(parents=True, exist_ok=True)
    ack_name = f"{segment_path.name}.ap{decoded.valid_prefix_bytes:020d}.json"
    (acks_dir / ack_name).write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": "00000000000000000001",
                "segment_name": segment_path.name,
                "segment_kind": "clean",
                "segment_size_bytes": decoded.valid_prefix_bytes,
                "acked_prefix_bytes": decoded.valid_prefix_bytes,
                "valid_prefix_bytes": decoded.valid_prefix_bytes,
                "tail_status": decoded.tail_status.value,
                "last_frame_offset": frame.frame_offset,
                "last_frame_length": frame.frame_length,
                "last_frame_checksum_hex": frame.checksum_hex,
            }
        )
    )

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.pending_units == 0
    assert status.pending_frames == 0
    assert status.pending_bytes == 0
    assert "pending_backlog" not in status.reasons
    assert status.ack_pending is True
    assert status.oldest_pending_age_seconds == pytest.approx(25.0)


def test_collect_session_fallback_spool_status_partial_ack_subtracts_exact_prefix(
    spool_home,
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    first = spool._frame_bytes_for_record(_record("partial-a"))
    second = spool._frame_bytes_for_record(_record("partial-b", attempt_index=1))
    segment_path = sealed_dir / "00000000000000000001.spool"
    segment_path.write_bytes(first + second)
    os.utime(segment_path, (75.0, 75.0))
    acks_dir = sealed_dir / spool.ACKS_DIR_NAME
    acks_dir.mkdir(parents=True, exist_ok=True)
    ack_name = f"{segment_path.name}.ap{len(first):020d}.json"
    (acks_dir / ack_name).write_bytes(
        spool._canonical_json_bytes(
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=len(first),
                valid_prefix_bytes=len(first + second),
            )
        )
    )

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.pending_units == 1
    assert status.pending_frames == 1
    assert status.pending_bytes == len(second)
    assert status.ack_pending is True
    assert status.oldest_pending_age_seconds == pytest.approx(25.0)


def test_collect_session_fallback_spool_status_blocker_prefix_subtracts_acked_prefix_once(
    spool_home,
):
    first = spool._frame_bytes_for_record(_record("blocker-a"))
    second = spool._frame_bytes_for_record(_record("blocker-b", attempt_index=1))
    corrupt_tail = bytearray(spool._frame_bytes_for_record(_record("blocker-c", attempt_index=2)))
    corrupt_tail[-1] ^= 0x01
    state = _write_status_blocker_state(spool_home)
    prefix_path = Path(str(state["prefix_path"]))
    evidence_spool = Path(str(state["evidence_spool"]))
    evidence_sidecar = Path(str(state["evidence_sidecar"]))
    prefix_path.write_bytes(first + second)
    blocker_path = Path(str(state["blocker_path"]))
    payload = json.loads(blocker_path.read_text(encoding="utf-8"))
    evidence_base = f"seq-{1:020d}-checksum_mismatch-vp{len(first + second)}"
    new_evidence_spool = evidence_spool.with_name(f"{evidence_base}.spool")
    new_evidence_sidecar = evidence_sidecar.with_name(f"{evidence_base}.json")
    evidence_spool.unlink()
    evidence_sidecar.unlink()
    new_evidence_spool.write_bytes(first + second + bytes(corrupt_tail))
    new_evidence_sidecar.write_bytes(
        spool._canonical_json_bytes(
            {
                "schema_version": 1,
                "segment_sequence": f"{1:020d}",
                "source_kind": "sealed",
                "tail_status": "checksum_mismatch",
                "valid_prefix_bytes": len(first + second),
                "original_size_bytes": len(first + second + bytes(corrupt_tail)),
                "evidence_spool_name": new_evidence_spool.name,
            }
        )
    )
    payload["valid_prefix_bytes"] = len(first + second)
    payload["acked_prefix_bytes"] = len(first)
    payload["blocking_offset"] = len(first + second)
    payload["evidence_spool_name"] = new_evidence_spool.name
    payload["evidence_sidecar_name"] = new_evidence_sidecar.name
    payload["original_size_bytes"] = len(first + second + bytes(corrupt_tail))
    blocker_path.write_bytes(spool._canonical_json_bytes(payload))

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.blocker_present is True
    assert status.pending_units == 1
    assert status.pending_frames == 1
    assert status.pending_bytes == len(second)
    assert status.capacity_used_bytes >= len(first + second)


def test_collect_session_fallback_spool_status_invalid_ack_is_inspection_degradation(
    spool_home,
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    segment_path = sealed_dir / "00000000000000000001.spool"
    frame_bytes = spool._frame_bytes_for_record(_record("invalid-ack"))
    segment_path.write_bytes(frame_bytes)
    acks_dir = sealed_dir / spool.ACKS_DIR_NAME
    acks_dir.mkdir(parents=True, exist_ok=True)
    (acks_dir / f"{segment_path.name}.ap{len(frame_bytes):020d}.json").write_text(
        '{"schema_version":1}\n',
        encoding="utf-8",
    )

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "invalid_ack_json"
    assert "inspection_error" in status.reasons
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_source_kind is None


@pytest.mark.skipif(fcntl is None, reason="POSIX flock required")
def test_collect_session_fallback_spool_status_lock_contention_is_bounded_and_non_mutating(spool_home):
    assert fcntl is not None
    root, active_path, lock_path, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    active_path.write_bytes(spool._frame_bytes_for_record(_record("status-lock")))
    before = active_path.read_bytes()

    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "append_lock_busy"
    assert "inspection_error" in status.reasons
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_offset is None
    assert status.blocker_reason_class is None
    assert status.blocker_source_kind is None
    assert active_path.read_bytes() == before


def _status_write_ack_sidecar(segment_path: Path, *, acked_prefix_bytes: int) -> None:
    acks_dir = segment_path.parent / spool.ACKS_DIR_NAME
    acks_dir.mkdir(parents=True, exist_ok=True)
    ack_name = f"{segment_path.name}.ap{acked_prefix_bytes:020d}.json"
    (acks_dir / ack_name).write_bytes(
        spool._canonical_json_bytes(
            _ack_payload(
                segment_sequence=int(segment_path.name[:20]),
                segment_name=segment_path.name,
                acked_prefix_bytes=acked_prefix_bytes,
                valid_prefix_bytes=segment_path.stat().st_size,
                segment_kind=(
                    "prefix" if segment_path.name.endswith(".prefix.spool") else "clean"
                ),
            )
        )
    )


def _mutate_status_blocker_case(paths: dict[str, object], case_name: str) -> None:
    blocker_path = Path(str(paths["blocker_path"]))
    prefix_path = Path(str(paths["prefix_path"]))
    evidence_spool = Path(str(paths["evidence_spool"]))
    evidence_sidecar = Path(str(paths["evidence_sidecar"]))
    payload = json.loads(blocker_path.read_text(encoding="utf-8"))

    if case_name == "negative_acked_prefix":
        payload["acked_prefix_bytes"] = -1
        blocker_path.write_bytes(spool._canonical_json_bytes(payload))
        return
    if case_name == "acked_prefix_gt_valid_prefix":
        payload["acked_prefix_bytes"] = payload["valid_prefix_bytes"] + 1
        blocker_path.write_bytes(spool._canonical_json_bytes(payload))
        return
    if case_name == "prefix_name_none_with_nonzero_offsets":
        payload["prefix_segment_name"] = None
        blocker_path.write_bytes(spool._canonical_json_bytes(payload))
        return
    if case_name == "wrong_expected_prefix_name":
        payload["prefix_segment_name"] = f"{1:020d}.spool"
        blocker_path.write_bytes(spool._canonical_json_bytes(payload))
        return
    if case_name == "wrong_expected_evidence_names":
        payload["evidence_spool_name"] = "wrong-evidence.spool"
        payload["evidence_sidecar_name"] = "wrong-evidence.json"
        blocker_path.write_bytes(spool._canonical_json_bytes(payload))
        return
    if case_name == "wrong_evidence_metadata":
        evidence_payload = json.loads(evidence_sidecar.read_text(encoding="utf-8"))
        evidence_payload["valid_prefix_bytes"] = evidence_payload["valid_prefix_bytes"] + 1
        evidence_sidecar.write_bytes(spool._canonical_json_bytes(evidence_payload))
        return
    if case_name == "missing_prefix_artifact":
        prefix_path.unlink()
        return
    if case_name == "nonregular_evidence_artifact":
        evidence_spool.unlink()
        evidence_spool.mkdir()
        return
    raise AssertionError(f"unknown blocker case: {case_name}")


def test_collect_session_fallback_spool_status_decodes_sealed_segments_from_fds_only(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    segment_path = sealed_dir / "00000000000000000001.spool"
    segment_path.write_bytes(spool._frame_bytes_for_record(_record("status-fd-only")))

    called = []

    def _forbidden(path, **_kwargs):
        called.append(str(path))
        raise AssertionError("status path re-resolution")

    monkeypatch.setattr(spool, "decode_spool_segment", _forbidden)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert called == []
    assert status.inspection_error_class is None
    assert status.pending_units == 1
    assert status.pending_frames == 1
    assert status.pending_bytes == segment_path.stat().st_size


def test_collect_session_fallback_spool_status_ack_directory_scan_count_is_constant(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    acks_dir = sealed_dir / spool.ACKS_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    acks_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")

    for sequence in range(1, 4):
        segment_path = sealed_dir / f"{sequence:020d}.spool"
        segment_path.write_bytes(
            spool._frame_bytes_for_record(
                _record(f"status-ack-scan-{sequence}", attempt_index=sequence - 1)
            )
        )
        _status_write_ack_sidecar(
            segment_path,
            acked_prefix_bytes=segment_path.stat().st_size,
        )

    ack_dir_stat = acks_dir.stat()
    counts = {"scans": 0, "entries": 0}
    original_scandir = spool.os.scandir

    class _CountingScandir:
        def __init__(self, iterator, *, count_entries: bool):
            self._iterator = iterator
            self._count_entries = count_entries

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self._iterator)
            if self._count_entries:
                counts["entries"] += 1
            return entry

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._iterator.__exit__(exc_type, exc, tb)

        def close(self):
            return self._iterator.close()

    def _counting_scandir(target):
        iterator = original_scandir(target)
        count_entries = False
        if isinstance(target, int) and spool._same_file_stat(os.fstat(target), ack_dir_stat):
            counts["scans"] += 1
            count_entries = True
        return _CountingScandir(iterator, count_entries=count_entries)

    monkeypatch.setattr(spool.os, "scandir", _counting_scandir)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.ack_pending is True
    assert counts["scans"] == 3
    assert counts["entries"] == 9


def test_collect_session_fallback_spool_status_oldest_age_uses_zero_prefix_blocker_queue_head(
    spool_home,
):
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        corrupt_frame = bytearray(spool._frame_bytes_for_record(_record("zero-prefix-head")))
        corrupt_frame[0] = 0x00
        spool._active_spool_path().write_bytes(bytes(corrupt_frame))
        with spool._append_lock(runtime.lock_fd, str(spool._lock_path())):
            published = spool._reconcile_active_spool_for_replay(runtime)
    finally:
        _close_runtime(runtime)

    assert published is not None
    assert published["valid_prefix_bytes"] == 0

    blocker_path = spool._blockers_dir() / "00000000000000000001.blocker.json"
    segment_path = sealed_dir / "00000000000000000002.spool"
    segment_path.write_bytes(spool._frame_bytes_for_record(_record("later-segment")))
    now = time.time()
    os.utime(blocker_path, (now - 3600, now - 3600))
    os.utime(segment_path, (now - 10, now - 10))

    status = spool.collect_session_fallback_spool_status(now=now)

    assert status.blocker_present is True
    assert status.pending_frames == 1
    assert status.oldest_pending_age_seconds == pytest.approx(3600.0, abs=1.0)


def test_collect_session_fallback_spool_status_oldest_age_uses_nonzero_prefix_queue_head(
    spool_home,
):
    paths = _write_status_blocker_state(spool_home)
    blocker_path = Path(str(paths["blocker_path"]))
    prefix_path = Path(str(paths["prefix_path"]))
    now = time.time()
    os.utime(blocker_path, (now - 3600, now - 3600))
    os.utime(prefix_path, (now - 10, now - 10))

    status = spool.collect_session_fallback_spool_status(now=now)

    assert status.blocker_present is True
    assert status.pending_frames == 1
    assert status.oldest_pending_age_seconds == pytest.approx(10.0, abs=1.0)


def test_collect_session_fallback_spool_status_blocker_preflight_direntry_stat_race_is_exception_safe(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    blockers_dir = root / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME
    blockers_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    (blockers_dir / "00000000000000000001.blocker.json").write_bytes(b"not-json\n")

    original_scandir = spool.os.scandir
    injected = {"done": False}

    def _racing_scandir(target):
        iterator = original_scandir(target)
        if not injected["done"] and _status_scandir_matches(target, blockers_dir):
            entries = list(iterator)
            iterator.close()
            injected["done"] = True
            return _StatusFaultingScandir(
                [
                    _StatusFaultingDirEntry(
                        entries[0],
                        stat_exc=FileNotFoundError("raced away"),
                    ),
                    *entries[1:],
                ]
            )
        return iterator

    monkeypatch.setattr(spool.os, "scandir", _racing_scandir)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "entry_replaced"
    assert "inspection_error" in status.reasons
    assert "raced away" not in repr(status)


def test_collect_session_fallback_spool_status_segment_direntry_stat_race_after_preflight_is_exception_safe(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    (sealed_dir / "00000000000000000001.spool").write_bytes(
        spool._frame_bytes_for_record(_record("segment-stat-race"))
    )

    original_scandir = spool.os.scandir
    sealed_scans = {"count": 0}

    def _racing_scandir(target):
        iterator = original_scandir(target)
        if _status_scandir_matches(target, sealed_dir):
            sealed_scans["count"] += 1
            if sealed_scans["count"] == 2:
                entries = list(iterator)
                iterator.close()
                return _StatusFaultingScandir(
                    [
                        _StatusFaultingDirEntry(
                            entry,
                            stat_exc=(
                                FileNotFoundError("segment replaced")
                                if entry.name.endswith(".spool")
                                else None
                            ),
                        )
                        for entry in entries
                    ]
                )
        return iterator

    monkeypatch.setattr(spool.os, "scandir", _racing_scandir)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "entry_replaced"
    assert "inspection_error" in status.reasons
    assert "segment replaced" not in repr(status)


def test_collect_session_fallback_spool_status_nested_scandir_iteration_oserror_is_exception_safe(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    blockers_dir = root / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME
    blockers_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    (blockers_dir / "00000000000000000001.blocker.json").write_bytes(b"not-json\n")

    original_scandir = spool.os.scandir
    injected = {"done": False}

    def _racing_scandir(target):
        iterator = original_scandir(target)
        if not injected["done"] and _status_scandir_matches(target, blockers_dir):
            iterator.close()
            injected["done"] = True
            return _StatusFaultingScandir(
                [],
                iteration_exc=OSError(errno.EIO, "nested iteration boom"),
            )
        return iterator

    monkeypatch.setattr(spool.os, "scandir", _racing_scandir)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "inspection_error"
    assert "inspection_error" in status.reasons
    assert "nested iteration boom" not in repr(status)


def test_collect_session_fallback_spool_status_symlink_shaped_metadata_race_maps_to_symlink_refused(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    (sealed_dir / "00000000000000000001.spool").write_bytes(
        spool._frame_bytes_for_record(_record("segment-eloop-race"))
    )

    original_scandir = spool.os.scandir
    sealed_scans = {"count": 0}

    def _racing_scandir(target):
        iterator = original_scandir(target)
        if _status_scandir_matches(target, sealed_dir):
            sealed_scans["count"] += 1
            if sealed_scans["count"] == 2:
                entries = list(iterator)
                iterator.close()
                return _StatusFaultingScandir(
                    [
                        _StatusFaultingDirEntry(
                            entry,
                            stat_exc=(
                                OSError(errno.ELOOP, "symlink loop should stay hidden")
                                if entry.name.endswith(".spool")
                                else None
                            ),
                        )
                        for entry in entries
                    ]
                )
        return iterator

    monkeypatch.setattr(spool.os, "scandir", _racing_scandir)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "symlink_refused"
    assert "inspection_error" in status.reasons
    assert "symlink loop should stay hidden" not in repr(status)


def test_collect_session_fallback_spool_status_home_open_race_is_exception_safe(
    spool_home, monkeypatch
):
    monkeypatch.setattr(
        spool,
        "_open_home_dir_fd",
        lambda _home_path: (_ for _ in ()).throw(FileNotFoundError("home vanished")),
    )

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.reasons == ("inspection_error",)
    assert status.inspection_error_class == "entry_replaced"
    assert "home vanished" not in repr(status)


def test_collect_session_fallback_spool_status_segment_replacement_after_open_fails_closed(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    segment_path = sealed_dir / "00000000000000000001.spool"
    segment_path.write_bytes(spool._frame_bytes_for_record(_record("status-swap-a")))
    replacement_frame = spool._frame_bytes_for_record(
        _record("status-swap-b", attempt_index=1)
    )
    original_read_exact = spool._read_exact_from_fd
    swapped = {"done": False}

    def _swap(fd: int, *, offset: int, length: int) -> bytes:
        if not swapped["done"] and length == spool.HEADER_SIZE:
            swapped["done"] = True
            parked = segment_path.with_name(segment_path.name + ".parked")
            os.replace(segment_path, parked)
            segment_path.write_bytes(replacement_frame)
        return original_read_exact(fd, offset=offset, length=length)

    monkeypatch.setattr(spool, "_read_exact_from_fd", _swap)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "entry_replaced"


def test_collect_session_fallback_spool_status_active_replacement_during_scan_fails_closed(
    spool_home, monkeypatch
):
    root, active_path, lock_path, _ = _paths(spool_home)
    root.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    active_path.write_bytes(spool._frame_bytes_for_record(_record("status-active-swap-old")))
    replacement_frame = spool._frame_bytes_for_record(
        _record(
            "status-active-swap-new",
            attempt_index=1,
            contents=("replacement-is-longer",),
        )
    )
    parked_path = root / "held-active.spool"
    original_scan = spool._scan_fd
    swapped = {"done": False}

    def _swap(fd: int):
        if not swapped["done"]:
            swapped["done"] = True
            os.replace(active_path, parked_path)
            active_path.write_bytes(replacement_frame)
        return original_scan(fd)

    monkeypatch.setattr(spool, "_scan_fd", _swap)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "entry_replaced"
    assert "inspection_error" in status.reasons
    assert status.pending_units == 0
    assert status.pending_frames == 0
    assert status.pending_bytes == 0
    assert active_path.read_bytes() == replacement_frame


def test_collect_session_fallback_spool_status_active_file_stat_oserror_is_exception_safe(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    segment_path = sealed_dir / "00000000000000000001.spool"
    corrupt_frame = bytearray(spool._frame_bytes_for_record(_record("status-bad-tail")))
    corrupt_frame[0] = 0x00
    segment_path.write_bytes(bytes(corrupt_frame))

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == spool.SpoolTailStatus.BAD_MAGIC.value


@pytest.mark.skipif(os.name != "posix", reason="symlink security is POSIX-only")
def test_collect_session_fallback_spool_status_segment_symlink_fails_closed(spool_home):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    target = spool_home / "symlink-target.spool"
    target.write_bytes(spool._frame_bytes_for_record(_record("status-symlink-target")))
    (sealed_dir / "00000000000000000001.spool").symlink_to(target)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "symlink_refused"


def test_collect_session_fallback_spool_status_segment_nonregular_artifact_fails_closed(
    spool_home,
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    (sealed_dir / "00000000000000000001.spool").mkdir()

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "unexpected_artifact"


@pytest.mark.parametrize(
    "case_name",
    [
        "negative_acked_prefix",
        "acked_prefix_gt_valid_prefix",
        "prefix_name_none_with_nonzero_offsets",
        "wrong_expected_prefix_name",
        "wrong_expected_evidence_names",
        "wrong_evidence_metadata",
        "missing_prefix_artifact",
        "nonregular_evidence_artifact",
    ],
)
def test_collect_session_fallback_spool_status_invalid_blocker_relationships_fail_closed(
    spool_home, case_name
):
    paths = _write_status_blocker_state(spool_home)
    _mutate_status_blocker_case(paths, case_name)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "invalid_blocker_json"
    assert "inspection_error" in status.reasons
    assert status.blocker_present is False
    assert status.blocker_sequence is None
    assert status.blocker_offset is None
    assert status.blocker_reason_class is None
    assert status.blocker_source_kind is None


def _publish_status_ack_tombstone(
    *,
    sequence: int,
    unit_id: str,
    attempt_index: int = 0,
) -> tuple[Path, Path, bytes]:
    runtime = spool._open_locked_runtime()
    try:
        sealed_dir = spool._sealed_dir()
        acks_dir = spool._acks_dir()
        sealed_dir.mkdir(parents=True, exist_ok=True)
        acks_dir.mkdir(parents=True, exist_ok=True)
        segment_path = sealed_dir / f"{sequence:020d}.spool"
        frame_bytes = spool._frame_bytes_for_record(
            _record(unit_id, attempt_index=attempt_index)
        )
        segment_path.write_bytes(frame_bytes)
        decoded = spool.decode_spool_segment(segment_path)
        frame = decoded.prefix_frames[0]
        spool._publish_ack_sidecar_strict(
            runtime,
            segment_sequence=sequence,
            segment_path=segment_path,
            ack_payload={
                "schema_version": 1,
                "segment_sequence": f"{sequence:020d}",
                "segment_name": segment_path.name,
                "segment_kind": "clean",
                "segment_size_bytes": decoded.valid_prefix_bytes,
                "acked_prefix_bytes": decoded.valid_prefix_bytes,
                "valid_prefix_bytes": decoded.valid_prefix_bytes,
                "tail_status": decoded.tail_status.value,
                "last_frame_offset": frame.frame_offset,
                "last_frame_length": frame.frame_length,
                "last_frame_checksum_hex": frame.checksum_hex,
            },
        )
        segment_path.unlink()
    finally:
        _close_runtime(runtime)
    ack_path = next(spool._acks_dir().glob(f"{sequence:020d}.spool.ap*.json"))
    return segment_path, ack_path, frame_bytes


def test_collect_session_fallback_spool_status_preflight_beats_blocker_parse(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    blockers_dir = root / spool.SEALED_DIR_NAME / spool.BLOCKERS_DIR_NAME
    blockers_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    for sequence in (1, 2):
        (blockers_dir / f"{sequence:020d}.blocker.json").write_bytes(b"not-json\n")

    parsed = {"count": 0}
    real_load = spool._load_canonical_json_entry

    def _counting_load(**kwargs):
        parsed["count"] += 1
        return real_load(**kwargs)

    monkeypatch.setattr(spool, "STATUS_SCAN_ARTIFACT_LIMIT", 1)
    monkeypatch.setattr(spool, "_load_canonical_json_entry", _counting_load)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "artifact_limit_exceeded"
    assert parsed["count"] == 0


def test_collect_session_fallback_spool_status_orphan_ack_tombstone_reports_ack_cleanup(
    spool_home,
):
    _segment_path, ack_path, _frame_bytes = _publish_status_ack_tombstone(
        sequence=1,
        unit_id="ack-tombstone-cleanup",
    )
    now = time.time()
    os.utime(ack_path, (now - 1200, now - 1200))

    status = spool.collect_session_fallback_spool_status(now=now)

    assert status.state == "degraded"
    assert status.reasons == ("ack_pending",)
    assert status.pending_units == 0
    assert status.pending_frames == 0
    assert status.pending_bytes == 0
    assert status.ack_pending is True
    assert status.oldest_pending_age_seconds == pytest.approx(1200.0, abs=1.0)
    assert status.inspection_error_class is None


def test_collect_session_fallback_spool_status_orphan_tombstone_keeps_fifo_age_over_later_segment(
    spool_home,
):
    _segment_path, ack_path, _frame_bytes = _publish_status_ack_tombstone(
        sequence=1,
        unit_id="ack-tombstone-fifo",
    )
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    lock_path.write_bytes(b"")
    later_segment = sealed_dir / "00000000000000000002.spool"
    later_bytes = spool._frame_bytes_for_record(
        _record("later-clean-segment", attempt_index=1)
    )
    later_segment.write_bytes(later_bytes)
    now = time.time()
    os.utime(ack_path, (now - 1200, now - 1200))
    os.utime(later_segment, (now - 10, now - 10))

    status = spool.collect_session_fallback_spool_status(now=now)

    assert status.ack_pending is True
    assert status.pending_units == 1
    assert status.pending_frames == 1
    assert status.pending_bytes == len(later_bytes)
    assert status.oldest_pending_age_seconds == pytest.approx(1200.0, abs=1.0)


def test_collect_session_fallback_spool_status_orphan_ack_winner_beats_lower_partial_candidate(
    spool_home,
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    acks_dir = sealed_dir / spool.ACKS_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    acks_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    first = spool._frame_bytes_for_record(_record("orphan-ack-a"))
    second = spool._frame_bytes_for_record(_record("orphan-ack-b", attempt_index=1))
    segment_path = sealed_dir / "00000000000000000001.spool"
    segment_path.write_bytes(first + second)
    partial_name = f"{segment_path.name}.ap{len(first):020d}.json"
    full_name = f"{segment_path.name}.ap{len(first + second):020d}.json"
    (acks_dir / partial_name).write_bytes(
        spool._canonical_json_bytes(
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=len(first),
                valid_prefix_bytes=len(first + second),
            )
        )
    )
    (acks_dir / full_name).write_bytes(
        spool._canonical_json_bytes(
            _ack_payload(
                segment_sequence=1,
                segment_name=segment_path.name,
                acked_prefix_bytes=len(first + second),
                valid_prefix_bytes=len(first + second),
            )
        )
    )
    segment_path.unlink()
    now = time.time()
    os.utime(acks_dir / full_name, (now - 1200, now - 1200))

    status = spool.collect_session_fallback_spool_status(now=now)

    assert status.state == "degraded"
    assert status.reasons == ("ack_pending",)
    assert status.pending_units == 0
    assert status.pending_frames == 0
    assert status.pending_bytes == 0
    assert status.ack_pending is True
    assert status.inspection_error_class is None
    assert status.oldest_pending_age_seconds == pytest.approx(1200.0, abs=1.0)


def test_collect_session_fallback_spool_status_ack_open_race_is_exception_safe(
    spool_home, monkeypatch
):
    root, _active_path, lock_path, _ = _paths(spool_home)
    sealed_dir = root / spool.SEALED_DIR_NAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    segment_path = sealed_dir / "00000000000000000001.spool"
    frame_bytes = spool._frame_bytes_for_record(_record("ack-race"))
    segment_path.write_bytes(frame_bytes)
    _status_write_ack_sidecar(segment_path, acked_prefix_bytes=len(frame_bytes))

    def _boom(**_kwargs):
        raise FileNotFoundError("ack vanished")

    monkeypatch.setattr(spool, "_load_ack_payload_from_fd", _boom)

    status = spool.collect_session_fallback_spool_status(now=100.0)

    assert status.state == "degraded"
    assert status.inspection_error_class == "entry_replaced"
    assert "inspection_error" in status.reasons
