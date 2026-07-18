from __future__ import annotations

import json
import multiprocessing
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from scripts.canary import owner_gate_foundation_journal as journal_module


TRANSACTION = "a" * 64


def _journal_at(root: Path) -> journal_module.FoundationApplyJournal:
    root.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(root.parent, 0o700)
    os.chown(root.parent, os.geteuid(), os.getegid())
    return journal_module.FoundationApplyJournal(
        _root=root,
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )


def _lease_worker(root: str, connection: Any) -> None:
    journal = _journal_at(Path(root))
    connection.send("attempting")
    with journal.transaction_lease(TRANSACTION):
        assert journal.read(TRANSACTION, "manifest") == {
            "schema": "test.v1"
        }
        connection.send("acquired")
        if connection.recv() != "release":
            raise RuntimeError("unexpected test control message")
    connection.send("released")
    connection.close()


def _pending_kill_worker(
    root: str,
    name: str,
    link_final: bool,
    connection: Any,
) -> None:
    journal = _journal_at(Path(root))
    raw = journal_module._canonical_bytes({"phase": "intent"})
    with journal.transaction_lease(TRANSACTION):
        transaction = journal.root / TRANSACTION
        pending = transaction / f".{name}.pending"
        journal._write_pending(pending, raw)
        journal._fsync_directory(transaction)
        if link_final:
            os.link(
                pending,
                transaction / f"{name}.json",
                follow_symlinks=False,
            )
            journal._fsync_directory(transaction)
        connection.send("crash-point-ready")
        connection.recv()


def _receive(connection: Any, *, timeout: float = 5.0) -> Any:
    assert connection.poll(timeout), "journal worker timed out"
    return connection.recv()


def _journal(tmp_path: Path) -> journal_module.FoundationApplyJournal:
    return _journal_at(tmp_path / "foundation-journal")


def test_publish_is_canonical_no_replace_and_idempotent(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    value = {"schema": "test.v1", "sequence": 1}

    assert journal.publish(TRANSACTION, "manifest", value) == value
    assert journal.publish(TRANSACTION, "manifest", value) == value
    assert journal.read(TRANSACTION, "manifest") == value
    artifact = journal.root / TRANSACTION / "manifest.json"
    assert artifact.read_bytes() == json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert artifact.stat().st_mode & 0o777 == 0o600

    with pytest.raises(
        RuntimeError,
        match="owner_gate_foundation_journal_artifact_diverged",
    ):
        journal.publish(
            TRANSACTION,
            "manifest",
            {"schema": "test.v1", "sequence": 2},
        )


def test_pending_hard_link_publication_recovers_after_crash(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    root = journal.root / TRANSACTION
    pending = root / ".s0-intent.pending"
    pending.write_bytes(b'{"phase":"intent"}')
    os.chmod(pending, 0o600)

    assert journal.read(TRANSACTION, "s0-intent") == {"phase": "intent"}
    assert not pending.exists()
    assert (root / "s0-intent.json").stat().st_nlink == 1


def test_strict_read_never_recovers_or_writes_pending_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    root = journal.root / TRANSACTION
    pending = root / ".success.pending"
    pending.write_bytes(b'{"phase":"success"}')
    os.chmod(pending, 0o600)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("strict journal read attempted a write")

    monkeypatch.setattr(os, "link", forbidden)
    monkeypatch.setattr(os, "unlink", forbidden)
    with pytest.raises(
        RuntimeError,
        match="owner_gate_foundation_journal_pending_requires_recovery",
    ):
        journal.read_strict(TRANSACTION, "success")

    assert pending.read_bytes() == b'{"phase":"success"}'
    assert not (root / "success.json").exists()


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b'{"phase":', "owner_gate_foundation_journal_json_invalid"),
        (b"", "owner_gate_foundation_journal_artifact_invalid"),
    ],
)
def test_partial_pending_is_preserved_and_fails_closed(
    tmp_path: Path,
    raw: bytes,
    error: str,
) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    root = journal.root / TRANSACTION
    pending = root / ".s0-operation.pending"
    pending.write_bytes(raw)
    os.chmod(pending, 0o600)

    with pytest.raises((RuntimeError, PermissionError), match=error):
        journal.read(TRANSACTION, "s0-operation")

    assert pending.read_bytes() == raw
    assert not (root / "s0-operation.json").exists()


@pytest.mark.parametrize(
    ("name", "link_final"),
    [("s1-intent", False), ("s2-intent", True)],
)
def test_sigkill_recovers_each_durable_pending_publication_crash_point(
    tmp_path: Path,
    name: str,
    link_final: bool,
) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_pending_kill_worker,
        args=(str(journal.root), name, link_final, child),
    )
    process.start()
    try:
        assert _receive(parent) == "crash-point-ready"
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == -signal.SIGKILL

        assert journal.read(TRANSACTION, name) == {"phase": "intent"}
        transaction = journal.root / TRANSACTION
        pending = transaction / f".{name}.pending"
        final = transaction / f"{name}.json"
        assert not pending.exists()
        assert final.read_bytes() == b'{"phase":"intent"}'
        assert final.stat().st_nlink == 1
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


def test_two_processes_block_and_sigkill_releases_full_transaction_lease(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    context = multiprocessing.get_context("spawn")
    first_parent, first_child = context.Pipe()
    second_parent, second_child = context.Pipe()
    first = context.Process(
        target=_lease_worker,
        args=(str(journal.root), first_child),
    )
    second = context.Process(
        target=_lease_worker,
        args=(str(journal.root), second_child),
    )
    first.start()
    try:
        assert _receive(first_parent) == "attempting"
        assert _receive(first_parent) == "acquired"
        second.start()
        assert _receive(second_parent) == "attempting"
        assert not second_parent.poll(0.5)

        os.kill(first.pid, signal.SIGKILL)
        first.join(timeout=5)
        assert not first.is_alive()
        assert first.exitcode == -signal.SIGKILL

        assert _receive(second_parent) == "acquired"
        second_parent.send("release")
        assert _receive(second_parent) == "released"
        second.join(timeout=5)
        assert not second.is_alive()
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
        first_parent.close()
        first_child.close()
        second_parent.close()
        second_child.close()


def test_transaction_directory_replacement_breaks_active_lease(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    transaction = journal.root / TRANSACTION
    displaced = journal.root / ("b" * 64)

    with journal.transaction_lease(TRANSACTION):
        transaction.rename(displaced)
        transaction.mkdir(mode=0o700)
        with pytest.raises(
            RuntimeError,
            match="owner_gate_foundation_journal_directory_changed",
        ):
            journal.read(TRANSACTION, "manifest")


def test_concurrent_transaction_directory_creation_joins_same_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _journal(tmp_path)
    journal._ensure_root()
    transaction = journal.root / TRANSACTION
    real_mkdir = os.mkdir
    raced = False

    def competing_mkdir(path: str | bytes | os.PathLike[str], mode: int) -> None:
        nonlocal raced
        if Path(path) == transaction and not raced:
            raced = True
            real_mkdir(path, mode)
            raise FileExistsError(path)
        real_mkdir(path, mode)

    monkeypatch.setattr(os, "mkdir", competing_mkdir)

    with journal.transaction_lease(TRANSACTION):
        assert journal.publish(
            TRANSACTION,
            "manifest",
            {"schema": "test.v1"},
        ) == {"schema": "test.v1"}
    assert raced is True


def test_closed_inventory_and_symlink_fail_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.publish(TRANSACTION, "manifest", {"schema": "test.v1"})
    root = journal.root / TRANSACTION
    (root / "surprise").write_text("x", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="owner_gate_foundation_journal_inventory_invalid",
    ):
        journal.list(TRANSACTION)

    (root / "surprise").unlink()
    (root / "s0-intent.json").symlink_to(root / "manifest.json")
    with pytest.raises(
        PermissionError,
        match="owner_gate_foundation_journal_artifact_invalid",
    ):
        journal.read(TRANSACTION, "s0-intent")


def test_artifact_names_are_bounded_to_nine_signed_steps(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    for name in ("s9-intent", "s0-delete", "resume", "step-0-intent"):
        with pytest.raises(
            RuntimeError,
            match="owner_gate_foundation_journal_artifact_name_invalid",
        ):
            journal.publish(TRANSACTION, name, {"phase": "intent"})
