from __future__ import annotations

import json
import multiprocessing
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from scripts.canary import owner_gate_bootstrap_journal as journal_module


def _journal_at(path: Path) -> journal_module.BootstrapInstallJournal:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    os.chown(path.parent, os.geteuid(), os.getegid())
    return journal_module.BootstrapInstallJournal(
        path,
        owner_uid=os.geteuid(),
        owner_gid=os.getegid(),
    )


def _receive(connection: Any, *, timeout: float = 5.0) -> Any:
    assert connection.poll(timeout), "bootstrap journal worker timed out"
    return connection.recv()


def _partial_write_worker(path: str, connection: Any) -> None:
    journal = _journal_at(Path(path))
    with journal.transaction_lease(create=True):
        scratch = journal.root / (
            f".manifest.{os.getpid()}.{'a' * 32}.pending"
        )
        descriptor = os.open(
            scratch,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            journal_module.ARTIFACT_MODE,
        )
        os.write(descriptor, b'{"schema":')
        connection.send("partial-written")
        connection.recv()
        os.close(descriptor)


def _durable_publication_worker(
    path: str,
    link_final: bool,
    connection: Any,
) -> None:
    journal = _journal_at(Path(path))
    with journal.transaction_lease(create=True):
        raw = journal_module.canonical_bytes(
            {"schema": "bootstrap-test-intent.v1"}
        )
        scratch = journal.root / (
            f".p0-intent.{os.getpid()}.{'b' * 32}.pending"
        )
        descriptor = os.open(
            scratch,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            journal_module.ARTIFACT_MODE,
        )
        try:
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        journal._fsync_directory(journal.root)
        if link_final:
            os.link(
                scratch,
                journal.root / "p0-intent.json",
                follow_symlinks=False,
            )
            journal._fsync_directory(journal.root)
        connection.send("durable-checkpoint")
        connection.recv()


def _lease_worker(path: str, connection: Any) -> None:
    journal = _journal_at(Path(path))
    try:
        with journal.transaction_lease(create=True):
            connection.send("acquired")
            connection.recv()
    except BaseException as exc:
        connection.send((type(exc).__name__, str(exc)))


def _journal(tmp_path: Path) -> journal_module.BootstrapInstallJournal:
    transaction = tmp_path / "state/transaction-a.json"
    return _journal_at(transaction)


def test_publish_is_canonical_immutable_and_idempotent(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    value = {"schema": "bootstrap-test-manifest.v1", "sequence": 1}
    with journal.transaction_lease(create=True):
        assert journal.publish("manifest", value) == value
        assert journal.publish("manifest", value) == value
        assert journal.read("manifest") == value
        with pytest.raises(
            journal_module.BootstrapJournalError,
            match="owner_gate_bootstrap_journal_artifact_diverged",
        ):
            journal.publish(
                "manifest",
                {"schema": "bootstrap-test-manifest.v1", "sequence": 2},
            )
    artifact = journal.root / "manifest.json"
    assert artifact.read_bytes() == json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert artifact.stat().st_nlink == 1
    assert artifact.stat().st_mode & 0o777 == 0o600


@pytest.mark.live_system_guard_bypass
def test_sigkill_mid_write_discards_only_unpublished_scratch(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_partial_write_worker,
        args=(str(journal._transaction_path), child),
    )
    process.start()
    try:
        assert _receive(parent) == "partial-written"
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        value = {"schema": "bootstrap-test-manifest.v1"}
        with journal.transaction_lease(create=True):
            assert journal.publish("manifest", value) == value
        assert not list(journal.root.glob(".*.pending"))
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("link_final", [False, True])
def test_sigkill_recovers_each_durable_no_replace_checkpoint(
    tmp_path: Path,
    link_final: bool,
) -> None:
    journal = _journal(tmp_path)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_durable_publication_worker,
        args=(str(journal._transaction_path), link_final, child),
    )
    process.start()
    try:
        assert _receive(parent) == "durable-checkpoint"
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        with journal.transaction_lease(create=True):
            assert journal.read("p0-intent") == {
                "schema": "bootstrap-test-intent.v1"
            }
        final = journal.root / "p0-intent.json"
        assert final.stat().st_nlink == 1
        assert not list(journal.root.glob(".*.pending"))
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


@pytest.mark.live_system_guard_bypass
def test_concurrent_process_is_excluded_and_sigkill_releases_lease(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    context = multiprocessing.get_context("spawn")
    first_parent, first_child = context.Pipe()
    second_parent, second_child = context.Pipe()
    first = context.Process(
        target=_lease_worker,
        args=(str(journal._transaction_path), first_child),
    )
    second = context.Process(
        target=_lease_worker,
        args=(str(journal._transaction_path), second_child),
    )
    first.start()
    try:
        assert _receive(first_parent) == "acquired"
        second.start()
        error = _receive(second_parent)
        assert error == (
            "BootstrapJournalError",
            "owner_gate_bootstrap_transaction_locked",
        )
        second.join(timeout=5)
        assert second.exitcode == 0

        os.kill(first.pid, signal.SIGKILL)
        first.join(timeout=5)
        assert first.exitcode == -signal.SIGKILL
        with journal.transaction_lease(create=True):
            assert journal.publish("manifest", {"schema": "recovered.v1"})
    finally:
        for process in (first, second):
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
        first_parent.close()
        first_child.close()
        second_parent.close()
        second_child.close()


def test_tamper_and_divergent_pending_fail_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with journal.transaction_lease(create=True):
        journal.publish("manifest", {"schema": "original.v1"})
    artifact = journal.root / "manifest.json"
    artifact.write_bytes(b'{ "schema":"altered.v1"}')
    with pytest.raises(
        journal_module.BootstrapJournalError,
        match="owner_gate_bootstrap_journal_canonical_invalid",
    ):
        with journal.transaction_lease(create=False):
            journal.read("manifest")

    artifact.write_bytes(b'{"schema":"original.v1"}')
    scratch = journal.root / f".manifest.{os.getpid()}.{'c' * 32}.pending"
    scratch.write_bytes(b'{"schema":"different.v1"}')
    scratch.chmod(0o600)
    with pytest.raises(
        journal_module.BootstrapJournalError,
        match="owner_gate_bootstrap_journal_pending_diverged",
    ):
        with journal.transaction_lease(create=False):
            pass


def test_existing_transaction_is_required_for_strict_rollback_load(
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "state/transaction-a.json"
    transaction.parent.mkdir(mode=0o700)
    os.chown(transaction.parent, os.geteuid(), os.getegid())
    journal = _journal_at(transaction)
    with pytest.raises(
        journal_module.BootstrapJournalError,
        match="owner_gate_bootstrap_transaction_missing",
    ):
        with journal.transaction_lease(create=False):
            pass
    assert not journal.root.exists()
