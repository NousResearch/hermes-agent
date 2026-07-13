"""Focused contracts for the root-only full-canary coordinator."""

from __future__ import annotations

import os
import signal
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

import gateway.canonical_full_canary_coordinator as coordinator
import scripts.canary.full_canary_owner_launcher as owner_launcher


APPROVAL_SHA256 = "a" * 64
ADMIN_USERNAME = "muncho_canary_admin_" + "a" * 16
ADMIN_PASSWORD = b"owner-generated-password-123456"


def _frame(
    *,
    username: str = ADMIN_USERNAME,
    password: bytes = ADMIN_PASSWORD,
    trailing: bytes = b"",
    magic: bytes = b"MCA2",
) -> bytes:
    username_raw = username.encode("utf-8")
    return (
        struct.pack("!4sHI", magic, len(username_raw), len(password))
        + username_raw
        + password
        + trailing
    )


def _pipe_frame(payload: bytes) -> int:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, payload)
    finally:
        os.close(write_fd)
    return read_fd


def _fake_stat(*, inode: int = 1, size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        st_dev=1,
        st_ino=inode,
        st_mode=0o100400,
        st_nlink=1,
        st_uid=0,
        st_gid=0,
        st_size=size,
        st_mtime_ns=1,
        st_ctime_ns=1,
    )


def test_ephemeral_admin_username_is_exact_approval_derivation() -> None:
    assert (
        coordinator.derive_ephemeral_admin_username(APPROVAL_SHA256) == ADMIN_USERNAME
    )
    with pytest.raises(
        coordinator.CoordinatorError,
        match="credential_prepare_approval_digest_invalid",
    ):
        coordinator.derive_ephemeral_admin_username("not-a-digest")


def test_opaque_admin_frame_is_one_shot_redacted_and_zeroized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd = _pipe_frame(_frame())
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda fd: False)
    try:
        frame = coordinator.OpaqueStdinAdminFrame.read(
            expected_username=ADMIN_USERNAME,
            fd=read_fd,
        )
    finally:
        os.close(read_fd)
    assert ADMIN_PASSWORD.decode() not in repr(frame)
    password = frame.consume_password()
    assert bytes(password) == ADMIN_PASSWORD
    with pytest.raises(
        coordinator.CoordinatorError,
        match="admin_frame_replay_forbidden",
    ):
        frame.consume_password()
    coordinator._zeroize(password)
    assert password == bytearray(len(ADMIN_PASSWORD))
    frame.close()


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (_frame(magic=b"BAD!"), "admin_frame_magic_invalid"),
        (
            _frame(username="muncho_canary_admin_" + "b" * 16),
            "admin_frame_username_mismatch",
        ),
        (_frame(trailing=b"x"), "admin_frame_trailing_data"),
        (_frame(password=b"short"), "admin_frame_password_bound_invalid"),
        (_frame(password=b"x" * 24 + b"\n"), "admin_frame_password_invalid"),
    ],
)
def test_opaque_admin_frame_rejects_malformed_or_replayed_transport(
    payload: bytes,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd = _pipe_frame(payload)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda fd: False)
    try:
        with pytest.raises(coordinator.CoordinatorError, match=code):
            coordinator.OpaqueStdinAdminFrame.read(
                expected_username=ADMIN_USERNAME,
                fd=read_fd,
            )
    finally:
        os.close(read_fd)


def test_opaque_admin_frame_rejects_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: True)
    with pytest.raises(
        coordinator.CoordinatorError,
        match="admin_frame_tty_forbidden",
    ):
        coordinator.OpaqueStdinAdminFrame.read(
            expected_username=ADMIN_USERNAME,
        )


def test_read_exact_zeroizes_partial_secret_when_read_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = iter((b"secret-prefix", OSError("transport lost")))
    zeroized: list[bytes] = []
    real_zeroize = coordinator._zeroize

    def read(_fd: int, _size: int) -> bytes:
        result = next(reads)
        if isinstance(result, OSError):
            raise result
        return result

    def observe_zeroize(value):
        if value:
            zeroized.append(bytes(value))
        real_zeroize(value)

    monkeypatch.setattr(coordinator.os, "read", read)
    monkeypatch.setattr(coordinator, "_zeroize", observe_zeroize)

    with pytest.raises(coordinator.CoordinatorError, match="admin_frame_read_failed"):
        coordinator._read_exact(0, 32)

    assert zeroized == [b"secret-prefix"]


class _Wire:
    def __init__(self, protected: object) -> None:
        self.protected = protected
        self.closed = False

    def query(self, sql: str, *, maximum_rows: int):
        return SimpleNamespace(sql=sql, maximum_rows=maximum_rows)

    def close(self) -> None:
        self.closed = True


def test_verified_tls_session_consumes_frame_and_pins_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = bytearray(ADMIN_PASSWORD)
    frame = coordinator.OpaqueStdinAdminFrame(
        username=ADMIN_USERNAME,
        password=password,
    )
    protected = SimpleNamespace(close=lambda: None)
    observed: dict[str, object] = {}
    peer_sha256 = "b" * 64

    monkeypatch.setattr(
        coordinator,
        "_open_verified_tls_connection",
        lambda config: (protected, peer_sha256),
    )
    monkeypatch.setattr(
        coordinator,
        "_send_startup_message",
        lambda connection, **kwargs: observed.update(
            connection=connection, startup=kwargs
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_authenticate",
        lambda connection, **kwargs: observed.update(
            auth_connection=connection, auth=kwargs
        ),
    )
    monkeypatch.setattr(coordinator, "_PostgresWireSession", _Wire)

    session = coordinator.VerifiedTLSBootstrapAdminSession.open(
        frame=frame,
        tls_server_name="db.europe-west3.sql.goog",
        expected_tls_peer_certificate_sha256=peer_sha256,
    )

    assert frame.consumed is True
    assert password == bytearray(len(ADMIN_PASSWORD))
    assert observed["auth"] == {
        "user": ADMIN_USERNAME,
        "password": ADMIN_PASSWORD.decode(),
    }
    assert ADMIN_PASSWORD.decode() not in repr(session)
    result = session.query("SELECT 1", maximum_rows=1)
    assert result.sql == "SELECT 1"
    session.close()
    session.close()


def test_verified_tls_session_rejects_peer_drift_and_closes_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = coordinator.OpaqueStdinAdminFrame(
        username=ADMIN_USERNAME,
        password=bytearray(ADMIN_PASSWORD),
    )
    closed: list[bool] = []
    protected = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(
        coordinator,
        "_open_verified_tls_connection",
        lambda _config: (protected, "c" * 64),
    )
    with pytest.raises(
        coordinator.CoordinatorError,
        match="admin_tls_peer_certificate_mismatch",
    ):
        coordinator.VerifiedTLSBootstrapAdminSession.open(
            frame=frame,
            tls_server_name="db.europe-west3.sql.goog",
            expected_tls_peer_certificate_sha256="d" * 64,
        )
    assert frame.consumed is True
    assert closed == [True]


def _secret_test_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    constant: str,
) -> tuple[Path, int, int]:
    target = tmp_path / ("discord-token" if constant == "discord" else "bootstrap")
    monkeypatch.setattr(
        coordinator,
        "DISCORD_TOKEN_PATH"
        if constant == "discord"
        else "CANARY_BOOTSTRAP_CREDENTIAL_PATH",
        target,
    )
    real_lstat = Path.lstat

    def root_parent_lstat(path: Path):
        item = real_lstat(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=item.st_mode, st_uid=0)
        return item

    monkeypatch.setattr(Path, "lstat", root_parent_lstat)
    uid = os.getuid() or 1
    gid = os.getgid() or 1
    return target, uid, gid


@pytest.mark.parametrize("constant", ["discord", "bootstrap"])
@pytest.mark.parametrize("failure", ["directory_fsync", "readback"])
def test_atomic_secret_install_removes_published_path_after_post_link_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    constant: str,
    failure: str,
) -> None:
    target, uid, gid = _secret_test_path(
        monkeypatch,
        tmp_path,
        constant=constant,
    )
    real_lstat = Path.lstat
    fsync_calls = 0
    target_reads = 0

    def bounded_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if failure == "directory_fsync" and fsync_calls == 1:
            raise OSError("injected publication fsync failure")

    def bounded_lstat(path: Path):
        nonlocal target_reads
        item = real_lstat(path)
        if failure == "readback" and path == target:
            target_reads += 1
            if target_reads == 1:
                return SimpleNamespace(
                    st_mode=item.st_mode,
                    st_nlink=2,
                    st_uid=item.st_uid,
                    st_gid=item.st_gid,
                    st_size=item.st_size,
                    st_dev=item.st_dev,
                    st_ino=item.st_ino,
                )
        return item

    monkeypatch.setattr(coordinator, "_fsync_directory", bounded_fsync)
    monkeypatch.setattr(Path, "lstat", bounded_lstat)

    with pytest.raises((OSError, coordinator.CoordinatorError)):
        coordinator._atomic_install_secret(
            target,
            bytearray(b"x" * 32),
            uid=uid,
            gid=gid,
        )

    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".*.install.*"))


def test_atomic_secret_install_close_failure_still_cleans_every_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target, uid, gid = _secret_test_path(
        monkeypatch,
        tmp_path,
        constant="discord",
    )
    real_open = coordinator.os.open
    real_close = coordinator.os.close
    created_fd: list[int] = []
    close_attempts = 0

    def tracked_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if ".install." in os.fspath(path):
            created_fd.append(descriptor)
        return descriptor

    def fail_first_close(descriptor: int) -> None:
        nonlocal close_attempts
        if created_fd and descriptor == created_fd[0] and close_attempts < 2:
            close_attempts += 1
            if close_attempts == 1:
                raise OSError("injected close failure")
        real_close(descriptor)

    monkeypatch.setattr(coordinator.os, "open", tracked_open)
    monkeypatch.setattr(coordinator.os, "close", fail_first_close)

    with pytest.raises(OSError, match="injected close failure"):
        coordinator._atomic_install_secret(
            target,
            bytearray(b"x" * 32),
            uid=uid,
            gid=gid,
        )

    assert close_attempts == 2
    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".*.install.*"))


def test_atomic_secret_install_temp_unlink_failures_still_remove_published_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target, uid, gid = _secret_test_path(
        monkeypatch,
        tmp_path,
        constant="bootstrap",
    )
    real_unlink = Path.unlink
    temp_attempts = 0

    def fail_temp_unlinks(path: Path, *args, **kwargs) -> None:
        nonlocal temp_attempts
        if ".install." in path.name:
            temp_attempts += 1
            if temp_attempts <= 2:
                raise OSError("injected temp unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_unlinks)

    with pytest.raises(
        coordinator.CoordinatorCleanupBlocked,
        match="secret_install_cleanup_blocked",
    ):
        coordinator._atomic_install_secret(
            target,
            bytearray(b"x" * 32),
            uid=uid,
            gid=gid,
        )

    assert temp_attempts == 2
    assert not os.path.lexists(target)
    leftovers = list(tmp_path.glob(".*.install.*"))
    assert len(leftovers) == 1
    real_unlink(leftovers[0])


def test_atomic_secret_install_success_has_no_fallible_post_success_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target, uid, gid = _secret_test_path(
        monkeypatch,
        tmp_path,
        constant="discord",
    )
    fsync_calls = 0

    def one_success_only(_path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls > 1:
            raise OSError("unexpected post-success finalization")

    monkeypatch.setattr(coordinator, "_fsync_directory", one_success_only)
    installed = coordinator._atomic_install_secret(
        target,
        bytearray(b"x" * 32),
        uid=uid,
        gid=gid,
    )

    assert fsync_calls == 1
    assert (installed.st_dev, installed.st_ino) == (
        target.lstat().st_dev,
        target.lstat().st_ino,
    )
    target.unlink()


def test_atomic_secret_install_zero_progress_write_is_bounded_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target, uid, gid = _secret_test_path(
        monkeypatch,
        tmp_path,
        constant="bootstrap",
    )
    monkeypatch.setattr(coordinator.os, "write", lambda _fd, _payload: 0)

    with pytest.raises(
        coordinator.CoordinatorError,
        match="secret_install_write_stalled",
    ):
        coordinator._atomic_install_secret(
            target,
            bytearray(b"x" * 32),
            uid=uid,
            gid=gid,
        )

    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".*.install.*"))


def test_owner_approval_request_is_one_exact_cross_module_contract() -> None:
    unsigned = {
        "schema": coordinator.OWNER_APPROVAL_REQUEST_SCHEMA,
        "ok": True,
        "state": "awaiting_final_owner_approval",
        "release_sha": "1" * 40,
        "coordinator_input_sha256": "2" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA256,
        "owner_subject_sha256": "3" * 64,
        "approval_source_sha256": "4" * 64,
        "ephemeral_admin_username": ADMIN_USERNAME,
        "full_canary_plan_sha256": "5" * 64,
        "staged_plan_path": str(coordinator.DEFAULT_STAGED_PLAN_PATH),
        "staged_plan_file_sha256": "6" * 64,
        "approval_request_path": str(coordinator.OWNER_APPROVAL_REQUEST_PATH),
        "approval_path": str(coordinator.DEFAULT_APPROVAL_PATH),
        "hba_receipt_sha256": "7" * 64,
        "hba_expires_at_unix": 1_300,
        "fixture_expires_at_unix": 1_400,
        "credential_approval_expires_at_unix": 1_300,
        "requested_at_unix": 1_000,
        "approval_deadline_unix": 1_200,
        "owner_input_cutoff_unix": 1_170,
        "final_approval_transmit_margin_seconds": 30,
        "max_wait_seconds": 200,
        "prior_approval_file_sha256": None,
        "final_approval_frame_schema": coordinator.FINAL_APPROVAL_FRAME_SCHEMA,
    }
    request = {
        **unsigned,
        "request_sha256": coordinator._sha256_json(unsigned),
    }
    parsed = coordinator.OwnerApprovalRequest.from_mapping(request)
    gate = {
        "release_sha": unsigned["release_sha"],
        "coordinator_input_sha256": unsigned["coordinator_input_sha256"],
        "credential_prepare_approval_sha256": unsigned[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": unsigned["owner_subject_sha256"],
        "admin_username": unsigned["ephemeral_admin_username"],
        "expires_at_unix": unsigned["credential_approval_expires_at_unix"],
    }
    observed = owner_launcher.validate_owner_approval_request(
        request,
        gate=gate,
        now_unix=1_000,
    )
    assert observed == parsed.value


def test_credential_prepare_approval_rejects_floating_source() -> None:
    approval = coordinator.CredentialPrepareApproval(
        value={
            "coordinator_input_sha256": "1" * 64,
            "release_sha": "2" * 40,
            "approval_source_sha256": "3" * 64,
            "approved_at_unix": 10,
            "expires_at_unix": 20,
        }
    )
    coordinator_input = SimpleNamespace(
        sha256="1" * 64,
        revision="2" * 40,
        value={
            "writer_config": {
                "canary_scope_preapproval": {
                    "approval_source_sha256": "4" * 64,
                }
            }
        },
    )
    with pytest.raises(
        coordinator.CoordinatorError,
        match="credential_prepare_approval_not_fresh_or_bound",
    ):
        approval.require(coordinator_input=coordinator_input, now_unix=15)


def test_credential_prepare_approval_loader_reads_only_its_fixed_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = SimpleNamespace(
        sha256="1" * 64,
        revision="2" * 40,
        value={
            "writer_config": {
                "canary_scope_preapproval": {
                    "approval_source_sha256": "4" * 64,
                }
            }
        },
    )
    payload = {
        "schema": coordinator.CREDENTIAL_PREPARE_APPROVAL_SCHEMA,
        "scope": "full_canary_ephemeral_admin_prepare",
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "coordinator_input_sha256": coordinator_input.sha256,
        "release_sha": coordinator_input.revision,
        "owner_subject_sha256": "3" * 64,
        "approval_source_sha256": "4" * 64,
        "nonce_sha256": "5" * 64,
        "approved_at_unix": 100,
        "expires_at_unix": 200,
    }
    observed: list[Path] = []

    def read(path: Path, *, maximum: int) -> bytes:
        observed.append(path)
        assert maximum == coordinator.MAX_OWNER_APPROVAL_BYTES
        return coordinator._canonical_bytes(payload)

    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(coordinator, "_stable_root_read", read)

    approval = coordinator.load_credential_prepare_approval(
        coordinator_input,
        now_unix=150,
    )

    assert approval.value == payload
    assert observed == [coordinator.CREDENTIAL_PREPARE_APPROVAL_PATH]


def test_run_orchestration_uses_credential_prepare_approval_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    calls: list[str] = []
    emitted: list[Mapping[str, object]] = []
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "load_coordinator_input",
        lambda: coordinator_input,
    )
    monkeypatch.setattr(
        coordinator,
        "load_discord_token_install_receipt",
        lambda _input: ({}, None, None),
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )

    def load_credential(_input):
        calls.append("credential_prepare_approval")
        raise coordinator.CoordinatorError(
            "credential_prepare_approval_sentinel",
            phase="credential_prepare_approval",
        )

    monkeypatch.setattr(
        coordinator,
        "load_credential_prepare_approval",
        load_credential,
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)

    result = coordinator.run_full_canary(
        frame_emitter=lambda value: emitted.append(dict(value)),
    )

    assert calls == ["credential_prepare_approval"]
    assert emitted == []
    assert result["ok"] is False
    assert result["error_code"] == "credential_prepare_approval_sentinel"
    assert result["credential_prepare_approval_sha256"] is None


def test_failed_driver_cleanup_stops_when_durable_evidence_is_tampered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace()
    events: list[str] = []

    def load(_plan):
        assert _plan is plan
        events.append("load_evidence")
        raise RuntimeError("bootstrap evidence file digest drifted")

    monkeypatch.setattr(coordinator, "load_bootstrap_evidence_envelope", load)
    monkeypatch.setattr(
        coordinator,
        "mechanically_stop_full_canary_services",
        lambda: events.append("mechanical_stop")
        or (
            coordinator.GATEWAY_UNIT_NAME,
            coordinator.WRITER_UNIT_NAME,
            coordinator.EDGE_UNIT_NAME,
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe_preclaim") or None,
    )

    with pytest.raises(RuntimeError, match="file digest drifted"):
        coordinator._stop_failed_driver_services(plan)

    assert events == ["mechanical_stop", "observe_preclaim", "load_evidence"]


def test_secret_removal_retries_after_unlink_succeeds_but_fsync_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target, uid, gid = _secret_test_path(
        monkeypatch,
        tmp_path,
        constant="discord",
    )
    installed = coordinator._atomic_install_secret(
        target,
        bytearray(b"x" * 32),
        uid=uid,
        gid=gid,
    )
    state = coordinator._SecretRemovalState()
    calls = 0

    def fail_once(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")

    monkeypatch.setattr(coordinator, "_fsync_directory", fail_once)
    with pytest.raises(
        coordinator.CoordinatorCleanupBlocked,
        match="secret_cleanup_blocked",
    ):
        coordinator._remove_exact_secret(target, installed, state=state)
    assert state.unlinked is True
    assert not os.path.lexists(target)

    coordinator._remove_exact_secret(target, installed, state=state)
    assert calls == 2


def test_install_discord_hardens_before_loading_or_emitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        coordinator,
        "_harden_secret_process",
        lambda: order.append("harden"),
    )

    def fail_load():
        order.append("load")
        raise coordinator.CoordinatorError("stop")

    monkeypatch.setattr(coordinator, "load_coordinator_input", fail_load)
    with pytest.raises(coordinator.CoordinatorError, match="stop"):
        coordinator.install_discord_token(
            gate_emitter=lambda _gate: order.append("emit")
        )
    assert order == ["harden", "load"]


def test_install_discord_rechecks_stopped_services_inside_lifecycle_lock_after_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    coordinator_input = SimpleNamespace(
        sha256="1" * 64,
        revision="2" * 40,
        identities=SimpleNamespace(edge_uid=1001, edge_gid=1002),
        value={
            "writer_config": {
                "canary_scope_preapproval": {
                    "approval_source_sha256": "3" * 64,
                }
            }
        },
    )

    class Approval:
        sha256 = "4" * 64
        value = {
            "owner_subject_sha256": "5" * 64,
            "approval_source_sha256": "3" * 64,
            "expires_at_unix": 9_999_999_999,
        }

        def require(self, **_kwargs: object) -> None:
            events.append("approval")

    class LifecycleLock:
        def __enter__(self) -> None:
            events.append("lock_enter")

        def __exit__(self, *_args: object) -> None:
            events.append("lock_exit")

    service_results = iter((True, False))

    def service_state() -> bool:
        result = next(service_results)
        events.append("services_stopped" if result else "services_live")
        return result

    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "load_coordinator_input",
        lambda: coordinator_input,
    )
    monkeypatch.setattr(
        coordinator,
        "_consume_terminal_discord_retirement",
        lambda _input: None,
    )
    monkeypatch.setattr(
        coordinator,
        "load_discord_token_install_approval",
        lambda _input: Approval(),
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        service_state,
    )
    monkeypatch.setattr(coordinator, "_lifecycle_lock", LifecycleLock)
    monkeypatch.setattr(
        coordinator,
        "_atomic_install_secret",
        lambda *_args, **_kwargs: events.append("install"),
    )

    def read_frame() -> coordinator.OpaqueDiscordTokenFrame:
        events.append("frame")
        return coordinator.OpaqueDiscordTokenFrame(bytearray(b"x" * 32))

    with pytest.raises(
        coordinator.CoordinatorError,
        match="full_canary_services_not_stopped_for_token_install",
    ):
        coordinator.install_discord_token(
            gate_emitter=lambda _gate: events.append("gate"),
            frame_reader=read_frame,
        )

    assert "install" not in events
    assert events.index("frame") < events.index("lock_enter")
    assert events.index("lock_enter") < events.index("services_live")
    assert events.index("services_live") < events.index("lock_exit")


def test_wait_emits_approval_request_only_after_baseline_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import gateway.canonical_full_canary_live_driver as live

    plan = SimpleNamespace(sha256="8" * 64)
    monkeypatch.setattr(live, "FullCanaryPlan", SimpleNamespace)
    path = tmp_path / "approval.json"
    monkeypatch.setattr(live, "DEFAULT_APPROVAL_PATH", path)
    events: list[str] = []

    def missing_lstat(_path: Path):
        events.append("baseline")
        raise FileNotFoundError

    monotonic_values = iter((0.0, 2.0))
    monkeypatch.setattr(Path, "lstat", missing_lstat)
    with pytest.raises(live.LiveCanaryError, match="owner_approval_wait_timeout"):
        live.wait_for_fresh_owner_approval(
            plan,
            path=path,
            timeout_seconds=1,
            monotonic=lambda: next(monotonic_values),
            now=lambda: 100.0,
            sleeper=lambda _seconds: None,
            ready_callback=lambda: events.append("emit"),
            process_guard=lambda: None,
        )
    assert events[:2] == ["baseline", "emit"]


def _recovery_gate_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    discord_token_state: str = "installed",
    discord_token_install_receipt_sha256: str = "9" * 64,
    token_device: int | None = 123,
    token_inode: int | None = 456,
    discord_retirement_receipt_sha256: str | None = None,
) -> tuple[dict[str, object], SimpleNamespace]:
    release = "1" * 40
    coordinator_input = SimpleNamespace(
        revision=release,
        sha256="2" * 64,
    )
    lease = {
        "lease_sha256": "3" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA256,
        "owner_subject_sha256": "4" * 64,
        "ephemeral_admin_username": ADMIN_USERNAME,
        "pid": 4321,
        "process_start_time_ticks": 9876,
        "boot_id_sha256": "5" * 64,
        "boot_time_ns": 100,
        "module_origin": (
            f"/opt/muncho-canary-releases/{release}/venv/lib/python3.13/"
            "site-packages/gateway/canonical_full_canary_coordinator.py"
        ),
        "module_sha256": "6" * 64,
        "process_exe_sha256": "7" * 64,
        "process_cmdline_sha256": "8" * 64,
    }
    causal_unsigned = {
        "schema": coordinator._RECOVERY_CAUSAL_STATE_SCHEMA,
        "discord_token_state": discord_token_state,
        "discord_token_install_receipt_sha256": (discord_token_install_receipt_sha256),
        "token_device": token_device,
        "token_inode": token_inode,
        "discord_retirement_receipt_sha256": (discord_retirement_receipt_sha256),
    }
    causal = {
        **causal_unsigned,
        "causal_state_sha256": coordinator._sha256_json(causal_unsigned),
    }
    predecessor = coordinator._RecoveryPredecessor(
        kind="run_process_lease",
        schema="muncho-full-canary-coordinator-process-lease.v1",
        generation=0,
        value=lease,
        snapshot=SimpleNamespace(sha256="3" * 64),
        original_run_lease=lease,
        causal_state=causal,
        target=coordinator._run_lease_recovery_target(lease),
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "load_coordinator_input",
        lambda: coordinator_input,
    )
    monkeypatch.setattr(
        coordinator,
        "_load_recovery_predecessor",
        lambda _input: predecessor,
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_target_is_exactly_alive",
        lambda *_args, **_kwargs: False,
    )
    gate = dict(coordinator.recovery_gate(now_unix=1_000))
    return gate, coordinator_input


def _recovery_worker_identity() -> dict[str, object]:
    return {
        "recovery_worker_pid": 5432,
        "recovery_worker_process_start_time_ticks": 222,
        "recovery_worker_boot_id_sha256": "a" * 64,
        "recovery_worker_boot_time_ns": 333,
        "recovery_worker_uid": 0,
        "recovery_worker_gid": 0,
        "recovery_worker_module_origin": "/sealed/recovery/coordinator.py",
        "recovery_worker_module_sha256": "b" * 64,
        "recovery_worker_process_exe_sha256": "c" * 64,
        "recovery_worker_process_cmdline_sha256": "d" * 64,
    }


def _recovery_completion() -> tuple[dict[str, object], SimpleNamespace]:
    coordinator_input = SimpleNamespace(revision="1" * 40, sha256="2" * 64)
    original = {
        "lease_sha256": "3" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA256,
        "owner_subject_sha256": "4" * 64,
        "ephemeral_admin_username": ADMIN_USERNAME,
    }
    unsigned: dict[str, object] = {
        "schema": coordinator.RECOVERY_WORKER_COMPLETION_SCHEMA,
        "ok": False,
        "state": "cleanup_complete_awaiting_worker_exit",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": APPROVAL_SHA256,
        "owner_subject_sha256": "4" * 64,
        "ephemeral_admin_username": ADMIN_USERNAME,
        "original_run_process_lease": original,
        "original_run_process_lease_sha256": original["lease_sha256"],
        "causal_recovery_state_sha256": "5" * 64,
        "predecessor_kind": "run_process_lease",
        "predecessor_journal_sha256": "6" * 64,
        "predecessor_generation": 0,
        "recovery_generation": 1,
        "recovery_takeover_gate_sha256": "7" * 64,
        "owner_recovery_takeover_ack_sha256": "8" * 64,
        "recovery_worker_lease_sha256": "9" * 64,
        **_recovery_worker_identity(),
        "predecessor_termination_proven": True,
        "predecessor_process_lock_acquired": True,
        "predecessor_journal_replaced": True,
        "canonical_stop_receipt_sha256": None,
        "preplan_stopped_report_sha256": "e" * 64,
        "preclaim_reconciliation_receipt_sha256": None,
        "preclaim_reconciliation_state": None,
        "admin_frame_zeroized": True,
        "admin_session_closed": True,
        "migration_owner_membership_removed": True,
        "bootstrap_login_password_disabled": True,
        "bootstrap_credential_removed": True,
        "discord_token_removed": True,
        "discord_install_receipt_removed": True,
        "discord_retirement_receipt_sha256": "f" * 64,
        "services_stopped_proven": True,
        "services_enabled": False,
        "recovery_worker_exit_proven": False,
        "safe_to_delete_temporary_admin": False,
        "cleanup_completed_at_unix": 1_100,
    }
    completion = {
        **unsigned,
        "completion_sha256": coordinator._sha256_json(unsigned),
    }
    assert set(completion) == coordinator._RECOVERY_WORKER_COMPLETION_FIELDS
    return completion, coordinator_input


def _completion_snapshot(completion: dict[str, object]) -> SimpleNamespace:
    raw = coordinator._canonical_bytes(completion)
    return SimpleNamespace(
        path=coordinator.COORDINATOR_PROCESS_LEASE_PATH,
        raw=raw,
        sha256=coordinator._sha256_bytes(raw),
        item=_fake_stat(inode=90, size=len(raw)),
    )


def _legacy_recovery_receipt() -> tuple[dict[str, object], SimpleNamespace]:
    coordinator_input = SimpleNamespace(revision="1" * 40, sha256="2" * 64)
    unsigned: dict[str, object] = {
        "schema": coordinator.LEGACY_RECOVERY_RECEIPT_SCHEMA,
        "ok": True,
        "state": "recovered",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": APPROVAL_SHA256,
        "owner_subject_sha256": "4" * 64,
        "ephemeral_admin_username": ADMIN_USERNAME,
        "recovery_gate_sha256": "5" * 64,
        "owner_recovery_ack_sha256": "6" * 64,
        "stale_process_lease_sha256": "7" * 64,
        "process_termination_proven": True,
        "process_lock_acquired": True,
        "process_lease_removed": True,
        "canonical_stop_receipt_sha256": None,
        "preplan_stopped_report_sha256": "8" * 64,
        "preclaim_reconciliation_receipt_sha256": None,
        "preclaim_reconciliation_state": None,
        "admin_session_closed": True,
        "migration_owner_membership_removed": True,
        "bootstrap_login_password_disabled": True,
        "bootstrap_credential_removed": True,
        "discord_token_removed": True,
        "discord_install_receipt_removed": True,
        "discord_retirement_receipt_sha256": "9" * 64,
        "services_stopped_proven": True,
        "services_enabled": False,
        "safe_to_delete_temporary_admin": True,
        "completed_at_unix": 1_000,
    }
    return (
        {**unsigned, "receipt_sha256": coordinator._sha256_json(unsigned)},
        coordinator_input,
    )


def _stage2_recovery_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    dict[str, object],
    SimpleNamespace,
    SimpleNamespace,
    list[dict[str, object]],
    list[int],
]:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    predecessor = coordinator._load_recovery_predecessor(coordinator_input)
    claim = SimpleNamespace(
        lock_fd=55,
        lease={
            "state": "admin_authority_may_be_in_use",
            "transition_seq": 2,
            "original_run_process_lease": predecessor.original_run_lease,
            "causal_recovery_state": predecessor.causal_state,
        },
    )
    secret_unsigned: dict[str, object] = {
        "schema": coordinator.RECOVERY_SECRET_GATE_SCHEMA,
        "state": "awaiting_recovery_admin_credential",
        "ephemeral_admin_username": ADMIN_USERNAME,
        "admin_frame_schema": coordinator.RECOVERY_ADMIN_FRAME_SCHEMA,
        "gate_nonce_sha256": "b" * 64,
        "expires_at_unix": 1_200,
        "tls_server_name": "canary.europe-west3.sql.goog",
        "tls_peer_certificate_sha256": "c" * 64,
    }
    secret_gate = {
        **secret_unsigned,
        "gate_sha256": coordinator._sha256_json(secret_unsigned),
    }
    emitted: list[dict[str, object]] = []
    closed: list[int] = []
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_load_recovery_predecessor", lambda _input: predecessor
    )
    monkeypatch.setattr(coordinator, "recovery_gate", lambda: gate)
    monkeypatch.setattr(coordinator, "_claim_recovery_worker", lambda **_kwargs: claim)
    monkeypatch.setattr(
        coordinator,
        "_transition_recovery_worker_to_admin_authority",
        lambda **_kwargs: claim,
    )
    monkeypatch.setattr(
        coordinator, "_revalidate_recovery_worker_snapshot", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        coordinator, "_build_recovery_secret_gate", lambda **_kwargs: secret_gate
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_100)
    monkeypatch.setattr(
        coordinator, "_close_recovery_process_lock", lambda fd: closed.append(fd)
    )
    return gate, secret_gate, coordinator_input, claim, emitted, closed


def test_recovery_gate_and_ack_are_exact_cross_boundary_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    validated = owner_launcher.validate_recovery_gate(
        gate,
        expected_release_sha=coordinator_input.revision,
        owner_gate=None,
        now_unix=1_000,
    )
    assert validated == gate
    ack = owner_launcher.build_recovery_ack(
        gate,
        now_unix=1_001,
        nonce=b"n" * 32,
    )
    frame = owner_launcher.build_recovery_ack_frame(
        gate,
        ack,
    )
    read_fd = _pipe_frame(frame)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_001)
    try:
        observed = coordinator._read_recovery_ack(gate=gate, fd=read_fd)
    finally:
        os.close(read_fd)
    assert observed == ack


def test_recovery_ack_rejects_approval_before_gate_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    ack = dict(
        owner_launcher.build_recovery_ack(
            gate,
            now_unix=1_001,
            nonce=b"fresh-recovery-ack-nonce",
        )
    )
    ack["approved_at_unix"] = gate["observed_at_unix"] - 1
    unsigned = {key: value for key, value in ack.items() if key != "ack_sha256"}
    ack["ack_sha256"] = coordinator._sha256_json(unsigned)
    frame = owner_launcher.build_recovery_ack_frame(gate, ack)
    read_fd = _pipe_frame(frame)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_001)
    try:
        with pytest.raises(
            coordinator.CoordinatorError, match="recovery_ack_not_bound"
        ):
            coordinator._read_recovery_ack(gate=gate, fd=read_fd)
    finally:
        os.close(read_fd)

    assert coordinator_input.revision == gate["release_sha"]


def test_recovery_admin_reader_rejects_legacy_mca2_after_only_magic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _frame()
    read_fd = _pipe_frame(payload)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    try:
        with pytest.raises(
            coordinator.CoordinatorError,
            match="recovery_admin_frame_magic_invalid",
        ):
            coordinator._read_recovery_admin_frame(
                expected_username=ADMIN_USERNAME,
                expected_gate_sha256="a" * 64,
                expected_gate_nonce_sha256="b" * 64,
                fd=read_fd,
            )
        assert os.read(read_fd, len(payload)) == payload[4:]
    finally:
        os.close(read_fd)


def test_recovery_admin_reader_rejects_wrong_gate_before_credential_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    username_raw = ADMIN_USERNAME.encode("utf-8")
    credential = username_raw + ADMIN_PASSWORD
    payload = (
        struct.pack(
            "!4s32s32sHI",
            coordinator.RECOVERY_ADMIN_FRAME_MAGIC,
            bytes.fromhex("c" * 64),
            bytes.fromhex("b" * 64),
            len(username_raw),
            len(ADMIN_PASSWORD),
        )
        + credential
    )
    read_fd = _pipe_frame(payload)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    try:
        with pytest.raises(
            coordinator.CoordinatorError,
            match="recovery_admin_frame_gate_not_bound",
        ):
            coordinator._read_recovery_admin_frame(
                expected_username=ADMIN_USERNAME,
                expected_gate_sha256="a" * 64,
                expected_gate_nonce_sha256="b" * 64,
                fd=read_fd,
            )
        assert os.read(read_fd, len(credential)) == credential
    finally:
        os.close(read_fd)


def test_recovery_gate_accepts_retired_token_with_null_inode_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(
        monkeypatch,
        discord_token_state="retired",
        discord_token_install_receipt_sha256="d" * 64,
        token_device=None,
        token_inode=None,
        discord_retirement_receipt_sha256="e" * 64,
    )

    assert gate["token_device"] is None
    assert gate["token_inode"] is None
    assert (
        owner_launcher.validate_recovery_gate(
            gate,
            expected_release_sha=coordinator_input.revision,
            owner_gate=None,
            now_unix=1_000,
        )
        == gate
    )


def test_prepared_retirement_recovery_is_exact_cross_boundary_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_retirement()
    gate, coordinator_input = _recovery_gate_fixture(
        monkeypatch,
        discord_token_state="retirement_prepared",
        discord_token_install_receipt_sha256=prepared[
            "discord_token_install_receipt_sha256"
        ],
        token_device=prepared["token_device"],
        token_inode=prepared["token_inode"],
        discord_retirement_receipt_sha256=prepared["receipt_sha256"],
    )

    validated = owner_launcher.validate_recovery_gate(
        gate,
        expected_release_sha=coordinator_input.revision,
        owner_gate=None,
        now_unix=1_000,
    )
    ack = owner_launcher.build_recovery_ack(
        validated,
        now_unix=1_001,
        nonce=b"prepared-retirement-recovery-nonce",
    )
    frame = owner_launcher.build_recovery_ack_frame(
        validated,
        ack,
    )
    read_fd = _pipe_frame(frame)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_001)
    try:
        observed = coordinator._read_recovery_ack(gate=gate, fd=read_fd)
    finally:
        os.close(read_fd)

    assert gate["discord_token_state"] == "retirement_prepared"
    assert gate["discord_retirement_receipt_sha256"] == prepared["receipt_sha256"]
    assert observed == ack
    assert observed["discord_token_state"] == "retirement_prepared"


def test_writer_publication_retry_survives_unlink_then_fsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "writer.json"
    target.write_bytes(b"{}")
    item = target.lstat()
    publication = coordinator._WriterPublication(
        writer_gid=os.getgid(),
        before=None,
        after=coordinator._WriterSnapshot(raw=b"{}", item=item),
    )
    monkeypatch.setattr(coordinator, "DEFAULT_WRITER_CONFIG_SOURCE", target)
    monkeypatch.setattr(
        coordinator,
        "_capture_writer_snapshot",
        lambda _gid: (
            None
            if not os.path.lexists(target)
            else coordinator._WriterSnapshot(
                raw=target.read_bytes(), item=target.lstat()
            )
        ),
    )
    calls = 0

    def fail_once(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fsync failure")

    monkeypatch.setattr(coordinator, "_fsync_directory", fail_once)
    with pytest.raises(
        coordinator.CoordinatorCleanupBlocked,
        match="staged_writer_cleanup_fsync_blocked",
    ):
        publication.rollback()
    assert not os.path.lexists(target)
    publication.rollback()
    assert publication._rolled_back is True


def test_signal_fence_suppresses_second_signal_during_unwind() -> None:
    fence = coordinator._SignalFence()
    with pytest.raises(
        coordinator.CoordinatorError,
        match="coordinator_graceful_termination_requested",
    ):
        fence._handle(signal.SIGTERM, None)
    assert fence.cleaning is True
    fence._handle(signal.SIGTERM, None)


class _TokenApproval:
    sha256 = "a" * 64
    value = {
        "owner_subject_sha256": "b" * 64,
        "approval_source_sha256": "c" * 64,
        "coordinator_input_sha256": "2" * 64,
        "release_sha": "1" * 40,
        "expires_at_unix": 9_999_999_999,
    }

    def require(self, **_kwargs: object) -> None:
        return None


def _token_input() -> SimpleNamespace:
    return SimpleNamespace(
        revision="1" * 40,
        sha256="2" * 64,
        identities=SimpleNamespace(edge_uid=1001, edge_gid=1002),
        value={
            "writer_config": {
                "canary_scope_preapproval": {
                    "approval_source_sha256": "c" * 64,
                }
            }
        },
    )


def test_recovery_reconciliation_uses_sealed_adapter_without_closing_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    plan = SimpleNamespace(
        sha256="1" * 64,
        artifacts={"writer_config": object()},
        identities=SimpleNamespace(),
    )
    approval = SimpleNamespace(
        value={"approval_source_sha256": "2" * 64},
    )
    request = object()
    evidence = object()

    class Admin:
        tls_peer_certificate_sha256 = "3" * 64
        closed = False

    class Provisioner:
        def __init__(self, session, **kwargs):
            self.session = session
            assert kwargs == {"tls_peer_certificate_sha256": "3" * 64}

        def reconcile(self, observed_request, provisioning_receipt):
            assert observed_request is request
            assert provisioning_receipt is None
            events.append("sealed_reconciliation")
            self.session.close()
            return {"raw": "reconciliation"}

    monkeypatch.setattr(
        coordinator,
        "load_full_canary_approval",
        lambda: approval,
    )
    monkeypatch.setattr(
        coordinator,
        "_validate_artifact_source",
        lambda *_args, **_kwargs: b"writer",
    )
    monkeypatch.setattr(
        coordinator,
        "_validate_writer_config",
        lambda *_args, **_kwargs: {"writer": "validated"},
    )
    monkeypatch.setattr(
        coordinator,
        "_build_canary_bootstrap_provisioning_request",
        lambda *_args, **_kwargs: request,
    )
    monkeypatch.setattr(
        coordinator,
        "PreopenedSessionBootstrapProvisioner",
        Provisioner,
    )
    monkeypatch.setattr(
        coordinator,
        "_validate_canary_bootstrap_reconciliation_receipt",
        lambda value, **kwargs: (
            events.append("validated")
            or {
                "validated": value,
                "session_continuity": kwargs["expected_session_continuity"],
            }
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "validated_bootstrap_reconciliation_evidence",
        lambda **kwargs: (
            evidence
            if kwargs["request"] is request
            and kwargs["expected_session_continuity"] == "recovery_session"
            else (_ for _ in ()).throw(AssertionError("evidence binding drifted"))
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "persist_bootstrap_evidence_envelope",
        lambda _plan, observed: observed,
    )
    admin = Admin()

    observed = coordinator._reconcile_bootstrap_authority_for_recovery(
        plan=plan,
        admin_session=admin,
    )

    assert observed is evidence
    assert events == ["sealed_reconciliation", "validated"]
    assert admin.closed is False


def test_runtime_plan_recovery_reconciles_before_disabling_or_removing_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    credential_present = True
    coordinator_input = _token_input()
    coordinator_input.identities.writer_uid = 1003
    coordinator_input.identities.writer_gid = 1004
    plan = SimpleNamespace(revision=coordinator_input.revision, sha256="4" * 64)
    evidence = SimpleNamespace(
        reconciliation_receipt={"receipt_sha256": "5" * 64},
        outcome="retired",
    )

    class Admin:
        closed = False

        def query(self, sql: str, *, maximum_rows: int):
            assert sql == coordinator._BOOTSTRAP_ROLE_DISABLE_SQL
            assert maximum_rows == 0
            events.append("disable_login")
            return SimpleNamespace(command_tag="DO", rows=(), columns=())

        def close(self) -> None:
            events.append("close_admin")
            self.closed = True

    class Lifecycle:
        def __init__(self, observed_plan, **kwargs):
            assert observed_plan is plan
            assert kwargs["bootstrap_reconciliation_evidence"] is evidence

        def attest_stopped_after_mechanical_stop(self, *, reason: str, **kwargs):
            assert reason == "operator_requested"
            assert kwargs["stopped"] == (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            )
            events.append("stop_and_attest")
            return {"stop": "receipt"}

    def capture(path, **_kwargs):
        if path == coordinator.DEFAULT_PLAN_PATH:
            return SimpleNamespace()
        return None

    def lexists(path) -> bool:
        if path == coordinator.CANARY_BOOTSTRAP_CREDENTIAL_PATH:
            return credential_present
        return False

    def remove_secret(*_args, **_kwargs) -> None:
        nonlocal credential_present
        events.append("remove_credential")
        credential_present = False

    causal_state = {
        "discord_token_install_receipt_sha256": "6" * 64,
        "token_device": None,
        "token_inode": None,
    }
    admin = Admin()
    monkeypatch.setattr(coordinator, "_capture_root_snapshot", capture)
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(
            None,
            (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            ),
            None,
        ),
    )
    monkeypatch.setattr(coordinator, "load_full_canary_plan", lambda: plan)
    monkeypatch.setattr(
        coordinator,
        "_load_or_reconcile_bootstrap_authority",
        lambda **_kwargs: events.append("sealed_reconciliation") or evidence,
    )
    monkeypatch.setattr(
        coordinator,
        "_validate_secret_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(coordinator, "_remove_exact_secret", remove_secret)
    monkeypatch.setattr(coordinator.os.path, "lexists", lexists)
    monkeypatch.setattr(coordinator, "FullCanaryLifecycle", Lifecycle)
    monkeypatch.setattr(
        coordinator,
        "_validate_recovery_stop_receipt",
        lambda _value, **kwargs: (
            (
                "7" * 64,
                "8" * 64,
                "retired",
            )
            if kwargs["expected_bootstrap_reconciliation"] is evidence
            else (_ for _ in ()).throw(AssertionError("reconciliation not bound"))
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_remove_recovery_root_artifact",
        lambda _path: None,
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_discord_retirement_inputs",
        lambda **_kwargs: (None, None, None),
    )
    monkeypatch.setattr(
        coordinator,
        "_retire_discord_token_lease",
        lambda **_kwargs: {
            "state": "retired",
            "discord_token_install_receipt_sha256": "6" * 64,
            "token_device": None,
            "token_inode": None,
            "receipt_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )

    result = coordinator._perform_recovery_cleanup(
        coordinator_input=coordinator_input,
        original_run_lease={},
        causal_state=causal_state,
        admin_session=admin,
    )

    assert events == [
        "mechanical_stop",
        "sealed_reconciliation",
        "disable_login",
        "remove_credential",
        "close_admin",
        "stop_and_attest",
    ]
    assert result["migration_owner_membership_removed"] is True
    assert result["bootstrap_login_password_disabled"] is True
    assert result["bootstrap_credential_removed"] is True


@pytest.mark.parametrize("outcome", ["not_authorized", "retired"])
def test_staged_plan_recovery_reconciles_then_retires_before_bound_stop(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    events: list[str] = []
    credential_present = True
    coordinator_input = _token_input()
    coordinator_input.identities.writer_uid = 1003
    coordinator_input.identities.writer_gid = 1004
    plan = SimpleNamespace(revision=coordinator_input.revision, sha256="4" * 64)
    staged_snapshot = SimpleNamespace()
    evidence = SimpleNamespace(
        reconciliation_receipt={"receipt_sha256": "5" * 64},
        outcome=outcome,
    )

    class Admin:
        closed = False

        def query(self, sql: str, *, maximum_rows: int):
            assert sql == coordinator._BOOTSTRAP_ROLE_DISABLE_SQL
            assert maximum_rows == 0
            events.append("disable_login")
            return SimpleNamespace(command_tag="DO", rows=(), columns=())

        def close(self) -> None:
            events.append("close_admin")
            self.closed = True

    admin = Admin()

    class Lifecycle:
        def __init__(self, observed_plan, **kwargs):
            assert observed_plan is plan
            assert kwargs["bootstrap_reconciliation_evidence"] is evidence

        def attest_stopped_after_mechanical_stop(self, *, reason: str, **kwargs):
            assert reason == "operator_requested"
            assert kwargs["stopped"] == (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            )
            assert admin.closed is True
            assert credential_present is False
            events.append("stop_with_bound_reconciliation")
            return {"stop": "receipt"}

    def capture(path, **_kwargs):
        if path == coordinator.DEFAULT_STAGED_PLAN_PATH:
            return staged_snapshot
        return None

    def lexists(path) -> bool:
        if path == coordinator.DEFAULT_APPROVAL_PATH:
            return True
        if path == coordinator.CANARY_BOOTSTRAP_CREDENTIAL_PATH:
            return credential_present
        return False

    def remove_secret(*_args, **_kwargs) -> None:
        nonlocal credential_present
        events.append("remove_credential")
        credential_present = False

    causal_state = {
        "discord_token_install_receipt_sha256": "6" * 64,
        "token_device": None,
        "token_inode": None,
    }
    monkeypatch.setattr(coordinator, "_capture_root_snapshot", capture)
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(
            None,
            (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_staged_plan_for_recovery",
        lambda snapshot, **_kwargs: (
            plan
            if snapshot is staged_snapshot
            else (_ for _ in ()).throw(AssertionError("wrong staged snapshot"))
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_or_reconcile_bootstrap_authority",
        lambda **_kwargs: events.append("sealed_reconciliation") or evidence,
    )
    monkeypatch.setattr(
        coordinator,
        "_validate_secret_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(coordinator, "_remove_exact_secret", remove_secret)
    monkeypatch.setattr(coordinator.os.path, "lexists", lexists)
    monkeypatch.setattr(coordinator, "FullCanaryLifecycle", Lifecycle)
    monkeypatch.setattr(
        coordinator,
        "_validate_recovery_stop_receipt",
        lambda _value, **kwargs: (
            ("7" * 64, "8" * 64, "not_preapproved")
            if kwargs["expected_bootstrap_reconciliation"] is evidence
            else (_ for _ in ()).throw(AssertionError("reconciliation not bound"))
        ),
    )

    def remove_root(path):
        if path == coordinator.DEFAULT_STAGED_PLAN_PATH:
            events.append("remove_staged_plan")

    monkeypatch.setattr(coordinator, "_remove_recovery_root_artifact", remove_root)
    monkeypatch.setattr(
        coordinator,
        "_remove_recovery_writer_artifact",
        lambda _gid: events.append("remove_staged_writer"),
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_discord_retirement_inputs",
        lambda **_kwargs: (None, None, None),
    )
    monkeypatch.setattr(
        coordinator,
        "_retire_discord_token_lease",
        lambda **_kwargs: {
            "state": "retired",
            "discord_token_install_receipt_sha256": "6" * 64,
            "token_device": None,
            "token_inode": None,
            "receipt_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )

    result = coordinator._perform_recovery_cleanup(
        coordinator_input=coordinator_input,
        original_run_lease={},
        causal_state=causal_state,
        admin_session=admin,
    )

    assert events == [
        "mechanical_stop",
        "sealed_reconciliation",
        "disable_login",
        "remove_credential",
        "close_admin",
        "stop_with_bound_reconciliation",
        "remove_staged_plan",
        "remove_staged_writer",
    ]
    assert result["canonical_stop_receipt_sha256"] == "7" * 64
    assert result["bootstrap_credential_removed"] is True


def test_staged_plan_recovery_rejects_impossible_consumed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    plan = SimpleNamespace(revision=coordinator_input.revision, sha256="4" * 64)
    staged_snapshot = SimpleNamespace()
    class Admin:
        closed = False

        def query(self, _sql, *, maximum_rows):
            assert maximum_rows == 0
            return SimpleNamespace(command_tag="DO", rows=(), columns=())

        def close(self):
            self.closed = True

    admin = Admin()
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda path, **_kwargs: (
            staged_snapshot
            if path == coordinator.DEFAULT_STAGED_PLAN_PATH
            else None
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_staged_plan_for_recovery",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        coordinator,
        "_load_or_reconcile_bootstrap_authority",
        lambda **_kwargs: SimpleNamespace(outcome="consumed"),
    )
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: coordinator._MechanicalStopAttempt(
            None,
            (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path == coordinator.DEFAULT_APPROVAL_PATH,
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_staged_bootstrap_outcome_invalid",
    ):
        coordinator._perform_recovery_cleanup(
            coordinator_input=coordinator_input,
            original_run_lease={},
            causal_state={},
            admin_session=admin,
        )


def test_staged_only_recovery_uses_distinct_never_authorized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    credential_present = True
    coordinator_input = _token_input()
    coordinator_input.identities.writer_uid = 1003
    coordinator_input.identities.writer_gid = 1004
    plan = SimpleNamespace(revision=coordinator_input.revision, sha256="4" * 64)
    staged_snapshot = SimpleNamespace(sha256="5" * 64)
    preclaim = {
        "receipt_sha256": "6" * 64,
        "result": {"outcome": "not_preapproved"},
    }

    class Admin:
        closed = False

        def query(self, _sql, *, maximum_rows):
            assert maximum_rows == 0
            events.append("disable_login")
            return SimpleNamespace(command_tag="DO", rows=(), columns=())

        def close(self):
            events.append("close_admin")
            self.closed = True

    def capture(path, **_kwargs):
        events.append(
            "snapshot_staged"
            if path == coordinator.DEFAULT_STAGED_PLAN_PATH
            else "snapshot_runtime"
        )
        return (
            staged_snapshot
            if path == coordinator.DEFAULT_STAGED_PLAN_PATH
            else None
        )

    def lexists(path):
        if path == coordinator.CANARY_BOOTSTRAP_CREDENTIAL_PATH:
            return credential_present
        return False

    def remove_secret(*_args, **_kwargs):
        nonlocal credential_present
        events.append("remove_credential")
        credential_present = False

    monkeypatch.setattr(coordinator, "_capture_root_snapshot", capture)
    monkeypatch.setattr(
        coordinator,
        "_load_staged_plan_for_recovery",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(
            (1,),
            (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            ),
            None,
        ),
    )

    def no_envelope(_plan):
        events.append("prove_no_provisioning")
        raise coordinator.BootstrapEvidenceUnavailable("absent")

    monkeypatch.setattr(
        coordinator,
        "load_bootstrap_evidence_envelope",
        no_envelope,
    )
    monkeypatch.setattr(
        coordinator,
        "load_bootstrap_never_authorized_evidence",
        lambda _plan: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_or_reconcile_bootstrap_authority",
        lambda **_kwargs: pytest.fail("must not fabricate owner approval"),
    )
    monkeypatch.setattr(
        coordinator,
        "load_full_canary_approval",
        lambda: pytest.fail("must not load absent final approval"),
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lexists)
    monkeypatch.setattr(
        coordinator,
        "_validate_secret_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(coordinator, "_remove_exact_secret", remove_secret)
    monkeypatch.setattr(
        coordinator,
        "mechanically_reconcile_never_authorized_preclaim",
        lambda *_args, **_kwargs: events.append("reconcile_preclaim")
        or preclaim,
    )
    marker = SimpleNamespace(
        value={"receipt_sha256": "7" * 64},
        path=Path("/evidence/never.json"),
        file_sha256="8" * 64,
    )
    monkeypatch.setattr(
        coordinator,
        "persist_bootstrap_never_authorized_evidence",
        lambda *_args, **_kwargs: events.append("persist_never_authorized")
        or marker,
    )
    monkeypatch.setattr(
        coordinator,
        "_remove_recovery_root_artifact",
        lambda path: events.append("remove_staged")
        if path == coordinator.DEFAULT_STAGED_PLAN_PATH
        else None,
    )
    monkeypatch.setattr(
        coordinator,
        "_remove_recovery_writer_artifact",
        lambda _gid: events.append("remove_writer"),
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_discord_retirement_inputs",
        lambda **_kwargs: (None, None, None),
    )
    monkeypatch.setattr(
        coordinator,
        "_retire_discord_token_lease",
        lambda **_kwargs: {
            "state": "retired",
            "discord_token_install_receipt_sha256": "9" * 64,
            "token_device": None,
            "token_inode": None,
            "receipt_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )
    causal = {
        "discord_token_install_receipt_sha256": "9" * 64,
        "token_device": None,
        "token_inode": None,
    }

    result = coordinator._perform_recovery_cleanup(
        coordinator_input=coordinator_input,
        original_run_lease={},
        causal_state=causal,
        admin_session=Admin(),
    )

    assert events[:9] == [
        "mechanical_stop",
        "snapshot_runtime",
        "snapshot_staged",
        "prove_no_provisioning",
        "disable_login",
        "remove_credential",
        "close_admin",
        "reconcile_preclaim",
        "persist_never_authorized",
    ]
    assert result["canonical_stop_receipt_sha256"] == "7" * 64
    assert result["preclaim_reconciliation_state"] == "not_preapproved"


def test_staged_only_recovery_reuses_existing_never_authorized_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    coordinator_input = _token_input()
    coordinator_input.identities.writer_gid = 1004
    plan = SimpleNamespace(revision=coordinator_input.revision, sha256="4" * 64)
    staged_snapshot = SimpleNamespace(sha256="5" * 64)
    stop_order = [
        coordinator.GATEWAY_UNIT_NAME,
        coordinator.WRITER_UNIT_NAME,
        coordinator.EDGE_UNIT_NAME,
    ]
    preclaim = {
        "receipt_sha256": "6" * 64,
        "result": {"outcome": "not_preapproved"},
    }
    marker = SimpleNamespace(
        value={
            "receipt_sha256": "7" * 64,
            "staged_plan_file_sha256": staged_snapshot.sha256,
            "mechanical_stop_order": stop_order,
            "preclaim_reconciliation": preclaim,
        },
        path=Path("/evidence/never.json"),
        file_sha256="8" * 64,
    )

    class Admin:
        closed = False

        def query(self, _sql, *, maximum_rows):
            pytest.fail("durable never-authorized retry must not rerun SQL")

        def close(self):
            events.append("close_admin")
            self.closed = True

    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda path, **_kwargs: staged_snapshot
        if path == coordinator.DEFAULT_STAGED_PLAN_PATH
        else None,
    )
    monkeypatch.setattr(
        coordinator,
        "_load_staged_plan_for_recovery",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(
            (1,),
            tuple(stop_order),
            None,
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "load_bootstrap_never_authorized_evidence",
        lambda _plan: events.append("load_never_authorized") or marker,
    )
    monkeypatch.setattr(
        coordinator,
        "load_bootstrap_evidence_envelope",
        lambda _plan: (_ for _ in ()).throw(
            coordinator.BootstrapEvidenceUnavailable("absent")
        ),
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda _path: False,
    )
    monkeypatch.setattr(
        coordinator,
        "mechanically_reconcile_never_authorized_preclaim",
        lambda *_args, **_kwargs: pytest.fail("must not rerun preclaim SQL"),
    )
    monkeypatch.setattr(
        coordinator,
        "persist_bootstrap_never_authorized_evidence",
        lambda *_args, **_kwargs: pytest.fail("must not republish with new time"),
    )
    monkeypatch.setattr(
        coordinator,
        "_remove_recovery_root_artifact",
        lambda _path: None,
    )
    monkeypatch.setattr(
        coordinator,
        "_remove_recovery_writer_artifact",
        lambda _gid: None,
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_discord_retirement_inputs",
        lambda **_kwargs: (None, None, None),
    )
    monkeypatch.setattr(
        coordinator,
        "_retire_discord_token_lease",
        lambda **_kwargs: {
            "state": "retired",
            "discord_token_install_receipt_sha256": "9" * 64,
            "token_device": None,
            "token_inode": None,
            "receipt_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )

    result = coordinator._perform_recovery_cleanup(
        coordinator_input=coordinator_input,
        original_run_lease={},
        causal_state={
            "discord_token_install_receipt_sha256": "9" * 64,
            "token_device": None,
            "token_inode": None,
        },
        admin_session=Admin(),
    )

    assert events == [
        "mechanical_stop",
        "load_never_authorized",
        "close_admin",
    ]
    assert result["canonical_stop_receipt_sha256"] == "7" * 64
    assert result["preclaim_reconciliation_state"] == "not_preapproved"


def test_recovery_preserves_mechanical_stop_and_evidence_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    plan = SimpleNamespace(revision=coordinator_input.revision)
    events: list[str] = []

    class Admin:
        closed = False

        def query(self, _sql, *, maximum_rows):
            assert maximum_rows == 0
            events.append("disable_login")
            return SimpleNamespace(command_tag="DO", rows=(), columns=())

        def close(self):
            events.append("close_admin")
            self.closed = True

    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda path, **_kwargs: SimpleNamespace()
        if path == coordinator.DEFAULT_PLAN_PATH
        else None,
    )
    monkeypatch.setattr(coordinator, "load_full_canary_plan", lambda: plan)
    stop_error = RuntimeError("mechanical stop failed")
    evidence_error = RuntimeError("evidence digest drifted")
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(None, (), stop_error),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_or_reconcile_bootstrap_authority",
        lambda **_kwargs: events.append("load_evidence")
        or (_ for _ in ()).throw(evidence_error),
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)

    with pytest.raises(BaseExceptionGroup) as captured:
        coordinator._perform_recovery_cleanup(
            coordinator_input=coordinator_input,
            original_run_lease={},
            causal_state={},
            admin_session=Admin(),
        )
    assert captured.value.exceptions[:2] == (stop_error, evidence_error)
    assert events == [
        "mechanical_stop",
        "load_evidence",
        "disable_login",
        "close_admin",
    ]


def test_recovery_aggregates_both_snapshot_failures_after_safe_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    stop_error = RuntimeError("fixed stop failed")
    runtime_error = RuntimeError("runtime snapshot failed")
    staged_error = RuntimeError("staged snapshot failed")
    events: list[str] = []

    class Admin:
        closed = False

        def query(self, _sql, *, maximum_rows):
            assert maximum_rows == 0
            events.append("disable_login")
            return SimpleNamespace(command_tag="DO", rows=(), columns=())

        def close(self):
            events.append("close_admin")
            self.closed = True

    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(None, (), (stop_error,)),
    )

    def capture(path, **_kwargs):
        if path == coordinator.DEFAULT_PLAN_PATH:
            events.append("snapshot_runtime")
            raise runtime_error
        events.append("snapshot_staged")
        raise staged_error

    monkeypatch.setattr(coordinator, "_capture_root_snapshot", capture)
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)

    with pytest.raises(BaseExceptionGroup) as captured:
        coordinator._perform_recovery_cleanup(
            coordinator_input=coordinator_input,
            original_run_lease={},
            causal_state={},
            admin_session=Admin(),
        )
    assert captured.value.exceptions == (
        stop_error,
        runtime_error,
        staged_error,
    )
    assert events == [
        "mechanical_stop",
        "snapshot_runtime",
        "snapshot_staged",
        "disable_login",
        "close_admin",
    ]


def test_prior_plan_tamper_is_loaded_only_after_fixed_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    plan_error = RuntimeError("prior plan tampered")
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(
            None,
            (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            ),
            (),
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda path, **_kwargs: events.append(f"snapshot:{path.name}")
        or (
            SimpleNamespace(raw=b"tampered")
            if path == coordinator.DEFAULT_PLAN_PATH
            else None
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "load_full_canary_plan",
        lambda: events.append("load_plan")
        or (_ for _ in ()).throw(plan_error),
    )

    with pytest.raises(RuntimeError, match="prior plan tampered"):
        coordinator._reconcile_prior_plan_before_credential_prepare(
            admin_session=SimpleNamespace(),
        )
    assert events[0] == "mechanical_stop"
    assert events.index("mechanical_stop") < events.index("load_plan")


def test_mechanical_stop_runs_after_observation_error_and_preserves_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_error = RuntimeError("preclaim observation failed")
    stop_error = RuntimeError("fixed stop failed")
    events: list[str] = []

    monkeypatch.setattr(
        coordinator,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: (_ for _ in ()).throw(observation_error),
    )
    monkeypatch.setattr(
        coordinator,
        "mechanically_stop_full_canary_services",
        lambda: events.append("mechanical_stop")
        or (_ for _ in ()).throw(stop_error),
    )

    attempt = coordinator._attempt_mechanical_reverse_stop()

    assert events == ["mechanical_stop"]
    assert attempt.prior_preclaim_generation is None
    assert attempt.stopped == ()
    assert attempt.errors == (stop_error, observation_error)
    assert isinstance(attempt.error, BaseExceptionGroup)
    assert attempt.error.exceptions == (stop_error, observation_error)


def test_recovery_stop_validator_resolves_full_bootstrap_receipt_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(revision="1" * 40, sha256="2" * 64)
    descriptor = {
        "schema": "muncho-full-canary-bootstrap-evidence-descriptor.v1",
        "path": "/var/lib/muncho-full-canary/envelope.json",
        "file_sha256": "3" * 64,
        "envelope_sha256": "4" * 64,
        "attempt_id": "5" * 64,
    }
    evidence = SimpleNamespace(
        approval=SimpleNamespace(value={"approved_at_unix": 10}),
        provisioning_receipt={"applied_at_unix": 11},
        reconciliation_receipt={
            "receipt_sha256": "6" * 64,
            "reconciled_at_unix": 12,
        },
        descriptor=SimpleNamespace(to_mapping=lambda: descriptor),
    )
    path = Path("/var/lib/muncho-full-canary/stopped.json")
    preclaim = {
        "receipt_sha256": "7" * 64,
        "result": {"outcome": "retired"},
    }
    unsigned = {
        "schema": coordinator.FULL_CANARY_RECEIPT_SCHEMA,
        "stage": "stopped",
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.sha256,
        "units_enabled": False,
        "reason": "operator_requested",
        "stop_order": [
            coordinator.GATEWAY_UNIT_NAME,
            coordinator.WRITER_UNIT_NAME,
            coordinator.EDGE_UNIT_NAME,
        ],
        "stopped_at_unix": 13,
        "receipt_path": str(path),
        "bootstrap_evidence_present": True,
        "bootstrap_evidence_descriptor": descriptor,
        "bootstrap_never_authorized_evidence": None,
        "owner_approval_receipt": {"scope": "exact"},
        "owner_approval_receipt_sha256": "8" * 64,
        "bootstrap_provisioning_receipt": {"receipt_sha256": "9" * 64},
        "bootstrap_reconciliation": evidence.reconciliation_receipt,
        "bootstrap_reconciliation_complete": True,
        "bootstrap_authority_may_require_owner_cleanup": False,
        "bootstrap_durable_evidence_recovery_required": False,
        "preclaim_reconciliation": preclaim,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": coordinator._sha256_json(unsigned),
    }
    monkeypatch.setattr(
        coordinator,
        "_stable_root_read",
        lambda *_args, **_kwargs: coordinator._canonical_bytes(receipt),
    )
    resolver_calls: list[Path] = []

    def reject_copied_truth(receipt_path, *, plan):
        resolver_calls.append(receipt_path)
        raise RuntimeError("copied approval/provisioning/flags drifted")

    monkeypatch.setattr(
        coordinator,
        "load_bootstrap_evidence_from_receipt",
        reject_copied_truth,
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_stop_receipt_bootstrap_truth_invalid",
    ):
        coordinator._validate_recovery_stop_receipt(
            receipt,
            plan=plan,
            expected_bootstrap_reconciliation=evidence,
        )
    assert resolver_calls == [path]


def test_manual_discord_retirement_stops_before_tampered_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    plan = SimpleNamespace(revision=coordinator_input.revision)
    coordinator_input.base_plan = plan
    gate = {}
    events: list[str] = []
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "load_coordinator_input",
        lambda: events.append("load_input") or coordinator_input,
    )
    monkeypatch.setattr(
        coordinator,
        "validate_dedicated_canary_host",
        lambda _plan: events.append("validate_host") or {},
    )
    monkeypatch.setattr(
        coordinator,
        "discord_retirement_gate",
        lambda: events.append("gate") or gate,
    )
    monkeypatch.setattr(
        coordinator,
        "_load_terminal_recovery_journal",
        lambda _input: events.append("journal") or {},
    )
    monkeypatch.setattr(
        coordinator,
        "_revalidate_discord_retirement_gate_source",
        lambda **_kwargs: events.append("revalidate_gate") or None,
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path == coordinator.DEFAULT_PLAN_PATH,
    )
    monkeypatch.setattr(coordinator, "load_full_canary_plan", lambda: plan)
    monkeypatch.setattr(
        coordinator,
        "_attempt_mechanical_reverse_stop",
        lambda: events.append("mechanical_stop")
        or coordinator._MechanicalStopAttempt(
            None,
            (
                coordinator.GATEWAY_UNIT_NAME,
                coordinator.WRITER_UNIT_NAME,
                coordinator.EDGE_UNIT_NAME,
            ),
            None,
        ),
    )

    def tampered(_plan):
        events.append("load_evidence")
        raise RuntimeError("evidence digest drifted")

    monkeypatch.setattr(coordinator, "load_bootstrap_evidence_envelope", tampered)

    with pytest.raises(
        coordinator.CoordinatorError,
        match="discord_token_recovery_bootstrap_evidence_required",
    ):
        coordinator.stop_and_retire_discord_token(
            gate_emitter=lambda _gate: events.append("emit_gate"),
            ack_reader=lambda **_kwargs: events.append("ack") or {},
        )
    assert events == [
        "mechanical_stop",
        "load_input",
        "validate_host",
        "gate",
        "emit_gate",
        "ack",
        "journal",
        "revalidate_gate",
        "load_evidence",
    ]


@pytest.mark.parametrize(
    ("state", "device", "inode", "size"),
    [
        ("install_intent", None, None, None),
        ("stage_allocated", 11, 12, 0),
        ("secret_staged", 11, 12, 32),
    ],
)
def test_discord_install_journal_parses_every_crash_state_without_secret_digest(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    device: int | None,
    inode: int | None,
    size: int | None,
) -> None:
    coordinator_input = _token_input()
    approval = _TokenApproval()
    journal = coordinator._discord_install_journal(
        coordinator_input=coordinator_input,
        approval=approval,
        state=state,
        prepared_at_unix=100,
        device=device,
        inode=inode,
        size=size,
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(coordinator, "_stable_root_read", lambda *_a, **_k: b"{}")
    monkeypatch.setattr(
        coordinator.DiscordTokenInstallApproval,
        "from_mapping",
        classmethod(lambda _cls, _value: approval),
    )

    parsed = coordinator._parse_discord_install_journal(
        journal,
        coordinator_input=coordinator_input,
    )

    assert parsed == journal
    assert parsed["content_or_digest_recorded"] is False
    assert "token_sha256" not in parsed


def _install_crash_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], SimpleNamespace]:
    events: list[str] = []
    coordinator_input = _token_input()
    approval = _TokenApproval()
    item = SimpleNamespace(
        st_dev=11,
        st_ino=12,
        st_mode=0o100400,
        st_nlink=1,
        st_uid=1001,
        st_gid=1002,
        st_size=32,
        st_mtime_ns=1,
        st_ctime_ns=1,
    )

    class Lock:
        def __enter__(self):
            events.append("lock_enter")

        def __exit__(self, *_args):
            events.append("lock_exit")

    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_consume_terminal_discord_retirement", lambda _i: None
    )
    monkeypatch.setattr(
        coordinator, "load_discord_token_install_approval", lambda _i: approval
    )
    monkeypatch.setattr(
        coordinator, "discord_token_install_gate", lambda **_k: {"ok": True}
    )
    monkeypatch.setattr(coordinator, "_lifecycle_lock", Lock)
    monkeypatch.setattr(
        coordinator,
        "_consume_terminal_recovery_journal",
        lambda _i: events.append("terminal_consumed"),
    )
    monkeypatch.setattr(
        coordinator, "_services_are_exactly_stopped_and_disabled", lambda: True
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        coordinator,
        "_create_empty_discord_token_stage",
        lambda _i: events.append("stage_allocated") or item,
    )
    monkeypatch.setattr(
        coordinator,
        "_write_discord_token_stage",
        lambda *_a: events.append("secret_staged") or item,
    )
    monkeypatch.setattr(
        coordinator,
        "_link_discord_token_stage",
        lambda *_a: events.append("token_linked") or item,
    )

    def publish(value, *, expected_previous_sha256):
        events.append(str(value.get("state") or "installed"))
        return SimpleNamespace(
            after=SimpleNamespace(sha256=value["receipt_sha256"]),
        )

    monkeypatch.setattr(coordinator, "_publish_discord_install_journal", publish)
    return events, item


def test_discord_install_crash_after_intent_preserves_recoverable_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _item = _install_crash_fixture(monkeypatch)
    monkeypatch.setattr(
        coordinator,
        "_create_empty_discord_token_stage",
        lambda _i: (_ for _ in ()).throw(RuntimeError("power loss after intent")),
    )

    with pytest.raises(RuntimeError, match="power loss after intent"):
        coordinator.install_discord_token(
            gate_emitter=lambda _gate: None,
            frame_reader=lambda: coordinator.OpaqueDiscordTokenFrame(
                bytearray(b"x" * 32)
            ),
        )

    assert "install_intent" in events
    assert "stage_allocated" not in events


def test_discord_install_crash_after_token_link_keeps_inode_bound_staged_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _item = _install_crash_fixture(monkeypatch)
    real_publish = coordinator._publish_discord_install_journal
    successful: list[str] = []

    def fail_terminal(value, *, expected_previous_sha256):
        state = str(value.get("state") or "installed")
        events.append(state)
        if state == "installed":
            raise RuntimeError("power loss before terminal receipt")
        successful.append(state)
        return SimpleNamespace(after=SimpleNamespace(sha256=value["receipt_sha256"]))

    monkeypatch.setattr(coordinator, "_publish_discord_install_journal", fail_terminal)

    with pytest.raises(RuntimeError, match="power loss before terminal receipt"):
        coordinator.install_discord_token(
            gate_emitter=lambda _gate: None,
            frame_reader=lambda: coordinator.OpaqueDiscordTokenFrame(
                bytearray(b"x" * 32)
            ),
        )

    assert successful == ["install_intent", "stage_allocated", "secret_staged"]
    assert events.index("token_linked") < events.index("installed")


@pytest.mark.parametrize(
    ("state_name", "device", "inode", "paths"),
    [
        ("install_intent", None, None, set()),
        ("stage_allocated", 11, 12, {coordinator.DISCORD_TOKEN_STAGE_PATH}),
        (
            "secret_staged",
            11,
            12,
            {coordinator.DISCORD_TOKEN_STAGE_PATH, coordinator.DISCORD_TOKEN_PATH},
        ),
    ],
)
@pytest.mark.parametrize("prepared_exists", [False, True])
def test_discord_install_journal_retirement_is_retryable_for_every_state(
    monkeypatch: pytest.MonkeyPatch,
    state_name: str,
    device: int | None,
    inode: int | None,
    paths: set[Path],
    prepared_exists: bool,
) -> None:
    coordinator_input = _token_input()
    journal = {
        "receipt_sha256": "d" * 64,
        "owner_subject_sha256": "b" * 64,
        "device": device,
        "inode": inode,
    }
    item = SimpleNamespace(st_dev=11, st_ino=12)
    state = coordinator._DiscordInstallState(
        value=journal,
        snapshot=SimpleNamespace(sha256="e" * 64),
        state=state_name,
        token_item=item if coordinator.DISCORD_TOKEN_PATH in paths else None,
        stage_item=item if coordinator.DISCORD_TOKEN_STAGE_PATH in paths else None,
    )
    live_paths = set(paths) | {coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH}
    publications: list[dict[str, object]] = []
    monkeypatch.setattr(
        coordinator, "_services_are_exactly_stopped_and_disabled", lambda: True
    )
    prepared = _prepared_retirement(device=device, inode=inode)
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (
            (prepared, SimpleNamespace(sha256=prepared["receipt_sha256"]))
            if prepared_exists
            else None
        ),
    )
    monkeypatch.setattr(coordinator, "_load_discord_install_state", lambda _i: state)
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda path: path in live_paths)

    def publish(_path, raw, *, expected_previous_sha256):
        value = coordinator._decode_mapping(raw, code="test")
        publications.append(dict(value))
        return SimpleNamespace(after=SimpleNamespace(sha256=value["receipt_sha256"]))

    def remove_artifact(path, **_kwargs):
        live_paths.discard(path)

    def remove_journal(_snapshot, *, state):
        live_paths.discard(coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH)

    monkeypatch.setattr(coordinator, "_publish_root_payload", publish)
    monkeypatch.setattr(
        coordinator, "_remove_discord_install_artifact", remove_artifact
    )
    monkeypatch.setattr(coordinator, "_remove_exact_root_snapshot", remove_journal)

    receipt = coordinator._retire_discord_install_journal(
        coordinator_input=coordinator_input,
        install_state=state,
    )

    assert receipt["state"] == "retired"
    assert receipt["token_device"] == device
    assert receipt["token_inode"] == inode
    assert not live_paths
    assert [value["state"] for value in publications] == (
        ["retired"] if prepared_exists else ["retirement_prepared", "retired"]
    )


@pytest.mark.parametrize(
    ("failure_point", "code"),
    [
        ("module", "tampered_module"),
        ("executable", "tampered_executable"),
        ("cmdline", "coordinator_process_cmdline_invalid"),
    ],
)
def test_cli_source_attestation_fails_before_later_identity_steps(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    code: str,
) -> None:
    coordinator_input = SimpleNamespace()
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    if failure_point == "module":
        monkeypatch.setattr(
            coordinator,
            "_sealed_coordinator_module_identity",
            lambda _i: (_ for _ in ()).throw(coordinator.CoordinatorError(code)),
        )
        monkeypatch.setattr(
            coordinator,
            "_current_executable_manifest_identity",
            lambda _i: pytest.fail(
                "executable inspection must not follow module failure"
            ),
        )
    else:
        monkeypatch.setattr(
            coordinator,
            "_sealed_coordinator_module_identity",
            lambda _i: ("/sealed/module.py", "1" * 64),
        )
        if failure_point == "executable":
            monkeypatch.setattr(
                coordinator,
                "_current_executable_manifest_identity",
                lambda _i: (_ for _ in ()).throw(coordinator.CoordinatorError(code)),
            )
        else:
            monkeypatch.setattr(
                coordinator,
                "_current_executable_manifest_identity",
                lambda _i: "2" * 64,
            )
            real_read = Path.read_bytes

            def wrong_cmdline(path: Path) -> bytes:
                if path == Path("/proc/self/cmdline"):
                    return b"wrong\0command\0"
                return real_read(path)

            monkeypatch.setattr(Path, "read_bytes", wrong_cmdline)
            monkeypatch.setattr(
                coordinator,
                "_expected_command_cmdline",
                lambda *_a, **_k: b"sealed\0command\0",
            )

    with pytest.raises(coordinator.CoordinatorError, match=code):
        coordinator._attest_current_cli_process(
            coordinator_input,
            command="install-discord-token",
        )


def test_real_retirement_gate_accepts_validated_terminal_recovery_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    install_state = coordinator._DiscordInstallState(
        value={
            "receipt_sha256": "d" * 64,
            "owner_subject_sha256": "b" * 64,
            "device": 11,
            "inode": 12,
        },
        snapshot=SimpleNamespace(),
        state="installed",
        token_item=SimpleNamespace(st_dev=11, st_ino=12),
        stage_item=None,
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator,
        "_load_terminal_recovery_journal",
        lambda _i: ({"schema": coordinator.RECOVERY_RECEIPT_SCHEMA}, SimpleNamespace()),
    )
    monkeypatch.setattr(
        coordinator, "_load_discord_install_state", lambda _i: install_state
    )
    monkeypatch.setattr(
        coordinator, "_services_are_exactly_stopped_and_disabled", lambda: True
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: (
            path
            in {
                coordinator.COORDINATOR_PROCESS_LEASE_PATH,
                coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
            }
        ),
    )

    gate = coordinator.discord_retirement_gate(now_unix=100)

    assert gate["process_lease_absent"] is True
    assert gate["discord_token_install_receipt_sha256"] == "d" * 64


def _prepared_retirement(
    *,
    device: int | None = 11,
    inode: int | None = 12,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema": coordinator.DISCORD_TOKEN_RETIREMENT_RECEIPT_SCHEMA,
        "ok": False,
        "state": "retirement_prepared",
        "release_sha": "1" * 40,
        "coordinator_input_sha256": "2" * 64,
        "discord_token_install_receipt_sha256": "d" * 64,
        "token_path": str(coordinator.DISCORD_TOKEN_PATH),
        "token_device": device,
        "token_inode": inode,
        "services_stopped_proven": True,
        "services_enabled": False,
        "token_removed": False,
        "install_receipt_removed": False,
        "prepared_at_unix": 100,
        "retired_at_unix": None,
    }
    return {**unsigned, "receipt_sha256": coordinator._sha256_json(unsigned)}


@pytest.mark.parametrize(
    ("artifact_subset", "state_present"),
    [
        ("token_and_install", True),
        ("install_only", True),
        ("neither", False),
    ],
)
def test_prepared_retirement_accepts_every_monotonic_terminal_install_subset(
    monkeypatch: pytest.MonkeyPatch,
    artifact_subset: str,
    state_present: bool,
) -> None:
    coordinator_input = _token_input()
    retirement = _prepared_retirement()
    token_item = (
        SimpleNamespace(st_dev=11, st_ino=12)
        if artifact_subset == "token_and_install"
        else None
    )
    install_state = coordinator._DiscordInstallState(
        value={
            "receipt_sha256": "d" * 64,
            "owner_subject_sha256": "b" * 64,
            "device": 11,
            "inode": 12,
        },
        snapshot=SimpleNamespace(),
        state="installed",
        token_item=token_item,
        stage_item=None,
    )
    live_paths = {coordinator.DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH}
    if state_present:
        live_paths.add(coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    if token_item is not None:
        live_paths.add(coordinator.DISCORD_TOKEN_PATH)
    observed_require_token: list[bool] = []
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (retirement, SimpleNamespace()),
    )

    def load_state(_input, *, require_terminal_token=True):
        observed_require_token.append(require_terminal_token)
        return install_state

    monkeypatch.setattr(coordinator, "_load_discord_install_state", load_state)
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path in live_paths,
    )

    loaded, _snapshot, state = coordinator._load_prepared_discord_retirement_source(
        coordinator_input,
        require_terminal_install=True,
    )

    assert loaded == retirement
    assert (state is not None) is state_present
    assert observed_require_token == ([False] if state_present else [])


def test_prepared_retirement_rejects_impossible_token_only_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (_prepared_retirement(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path == coordinator.DISCORD_TOKEN_PATH,
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="discord_token_retirement_impossible_artifact_subset",
    ):
        coordinator._load_prepared_discord_retirement_source(
            _token_input(),
            require_terminal_install=True,
        )


@pytest.mark.parametrize(
    "artifact_subset",
    ["token_and_install", "install_only", "neither"],
)
def test_prepared_terminal_install_retirement_reaches_terminal_from_every_boundary(
    monkeypatch: pytest.MonkeyPatch,
    artifact_subset: str,
) -> None:
    prepared = _prepared_retirement()
    retirement_snapshot = SimpleNamespace(
        sha256=prepared["receipt_sha256"],
    )
    install_receipt = {"receipt_sha256": "d" * 64}
    install_raw = coordinator._canonical_bytes(install_receipt)
    install_snapshot = SimpleNamespace(raw=install_raw, sha256="e" * 64)
    installed = SimpleNamespace(st_dev=11, st_ino=12)
    live_paths = set()
    if artifact_subset in {"token_and_install", "install_only"}:
        live_paths.add(coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH)
    if artifact_subset == "token_and_install":
        live_paths.add(coordinator.DISCORD_TOKEN_PATH)
    publications: list[dict[str, object]] = []
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (prepared, retirement_snapshot),
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path in live_paths,
    )

    def remove_secret(path, _item, *, state):
        live_paths.discard(path)

    def remove_snapshot(snapshot, *, state):
        assert snapshot is install_snapshot
        live_paths.discard(coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH)

    def publish(_path, raw, *, expected_previous_sha256):
        value = coordinator._decode_mapping(raw, code="test")
        publications.append(dict(value))
        return SimpleNamespace(after=SimpleNamespace(sha256=value["receipt_sha256"]))

    monkeypatch.setattr(coordinator, "_remove_exact_secret", remove_secret)
    monkeypatch.setattr(coordinator, "_remove_exact_root_snapshot", remove_snapshot)
    monkeypatch.setattr(coordinator, "_publish_root_payload", publish)

    receipt = coordinator._retire_discord_token_lease(
        coordinator_input=_token_input(),
        install_receipt=(
            install_receipt
            if artifact_subset in {"token_and_install", "install_only"}
            else None
        ),
        installed=(installed if artifact_subset == "token_and_install" else None),
        install_snapshot=(
            install_snapshot
            if artifact_subset in {"token_and_install", "install_only"}
            else None
        ),
    )

    assert receipt["state"] == "retired"
    assert not live_paths
    assert [item["state"] for item in publications] == ["retired"]


@pytest.mark.parametrize(
    ("journal_state", "device", "inode"),
    [
        ("install_intent", None, None),
        ("stage_allocated", 11, 12),
        ("secret_staged", 11, 12),
    ],
)
def test_prepared_retirement_accepts_bound_nonterminal_install_journal_subsets(
    monkeypatch: pytest.MonkeyPatch,
    journal_state: str,
    device: int | None,
    inode: int | None,
) -> None:
    retirement = _prepared_retirement(device=device, inode=inode)
    state = coordinator._DiscordInstallState(
        value={
            "receipt_sha256": "d" * 64,
            "owner_subject_sha256": "b" * 64,
            "device": device,
            "inode": inode,
        },
        snapshot=SimpleNamespace(),
        state=journal_state,
        token_item=None,
        stage_item=None,
    )
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (retirement, SimpleNamespace()),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_discord_install_state",
        lambda _i, **_kwargs: state,
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path == coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH,
    )

    _loaded, _snapshot, observed = coordinator._load_prepared_discord_retirement_source(
        _token_input(),
        require_terminal_install=False,
    )

    assert observed is state


def test_prepared_install_intent_stage_unlink_resumes_from_causal_inode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retirement = _prepared_retirement(device=11, inode=12)
    state = coordinator._DiscordInstallState(
        value={
            "receipt_sha256": "d" * 64,
            "owner_subject_sha256": "b" * 64,
            "device": None,
            "inode": None,
        },
        snapshot=SimpleNamespace(sha256="e" * 64),
        state="install_intent",
        token_item=None,
        stage_item=None,
    )
    live_paths = {coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH}
    publications: list[dict[str, object]] = []
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (
            retirement,
            SimpleNamespace(sha256=retirement["receipt_sha256"]),
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_discord_install_state",
        lambda _i, **_kwargs: state,
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path in live_paths,
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )

    def remove_journal(_snapshot, *, state):
        live_paths.discard(coordinator.DISCORD_TOKEN_INSTALL_RECEIPT_PATH)

    def publish(_path, raw, *, expected_previous_sha256):
        value = coordinator._decode_mapping(raw, code="test")
        publications.append(dict(value))
        return SimpleNamespace(after=SimpleNamespace(sha256=value["receipt_sha256"]))

    monkeypatch.setattr(coordinator, "_remove_exact_root_snapshot", remove_journal)
    monkeypatch.setattr(coordinator, "_publish_root_payload", publish)

    loaded, _snapshot, observed = coordinator._load_prepared_discord_retirement_source(
        _token_input(),
        require_terminal_install=False,
    )
    receipt = coordinator._retire_discord_install_journal(
        coordinator_input=_token_input(),
        install_state=observed,
    )

    assert loaded == retirement
    assert receipt["state"] == "retired"
    assert receipt["token_device"] == 11
    assert receipt["token_inode"] == 12
    assert not live_paths
    assert [item["state"] for item in publications] == ["retired"]


def test_active_recovery_gate_binds_prepared_retirement_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retirement = _prepared_retirement()
    gate, _coordinator_input = _recovery_gate_fixture(
        monkeypatch,
        discord_token_state="retirement_prepared",
        discord_token_install_receipt_sha256=retirement[
            "discord_token_install_receipt_sha256"
        ],
        token_device=retirement["token_device"],
        token_inode=retirement["token_inode"],
        discord_retirement_receipt_sha256=retirement["receipt_sha256"],
    )

    assert gate["discord_token_state"] == "retirement_prepared"
    assert gate["discord_retirement_receipt_sha256"] == retirement["receipt_sha256"]
    assert gate["discord_token_install_receipt_sha256"] == "d" * 64


@pytest.mark.parametrize("current_state", ["retirement_prepared", "retired"])
def test_active_recovery_resumes_prepared_retirement_after_artifact_removal(
    monkeypatch: pytest.MonkeyPatch,
    current_state: str,
) -> None:
    prepared = _prepared_retirement()
    if current_state == "retirement_prepared":
        current = prepared
    else:
        terminal_unsigned = {
            **{
                key: item
                for key, item in prepared.items()
                if key
                not in {
                    "receipt_sha256",
                    "ok",
                    "state",
                    "token_removed",
                    "install_receipt_removed",
                    "retired_at_unix",
                }
            },
            "ok": True,
            "state": "retired",
            "token_removed": True,
            "install_receipt_removed": True,
            "retired_at_unix": 101,
        }
        current = {
            **terminal_unsigned,
            "receipt_sha256": coordinator._sha256_json(terminal_unsigned),
        }
    gate = {
        "discord_token_state": "retirement_prepared",
        "discord_token_install_receipt_sha256": "d" * 64,
        "token_device": 11,
        "token_inode": 12,
        "discord_retirement_receipt_sha256": prepared["receipt_sha256"],
    }
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (current, SimpleNamespace()),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_prepared_discord_retirement_source",
        lambda *_args, **_kwargs: (current, SimpleNamespace(), None),
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)

    result = coordinator._recovery_discord_retirement_inputs(
        coordinator_input=_token_input(),
        gate=gate,
    )

    assert result == (None, None, None)


def test_prepared_retirement_keeps_unbound_failure_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_input = _token_input()
    monkeypatch.setattr(coordinator, "_stable_root_read", lambda *_a, **_k: b"{}")
    monkeypatch.setattr(
        coordinator.CoordinatorInput,
        "from_mapping",
        classmethod(lambda _cls, _value: coordinator_input),
    )
    monkeypatch.setattr(
        coordinator.CredentialPrepareApproval,
        "from_mapping",
        classmethod(lambda _cls, _value: (_ for _ in ()).throw(RuntimeError())),
    )
    monkeypatch.setattr(
        coordinator,
        "_load_discord_retirement",
        lambda _i: (_prepared_retirement(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path == coordinator.DISCORD_TOKEN_RETIREMENT_RECEIPT_PATH,
    )
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )

    failure = coordinator._unbound_failure(
        coordinator.CoordinatorError(
            "coordinator_process_recovery_not_required",
            phase="recovery_preflight",
        ),
        command="preflight-recovery",
    )

    assert failure["cleanup_status"] == "cleanup_blocked"
    assert failure["recovery_material_preserved"] is True
    assert failure["discord_token_removed"] is False


def test_main_attests_source_before_emitting_sensitive_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, object]] = []
    gate_calls: list[bool] = []
    monkeypatch.setattr(coordinator, "load_coordinator_input", lambda: _token_input())
    monkeypatch.setattr(
        coordinator,
        "_attest_current_cli_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            coordinator.CoordinatorError("tampered_source")
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "install_discord_token",
        lambda **_kwargs: gate_calls.append(True),
    )
    monkeypatch.setattr(
        coordinator,
        "_unbound_failure",
        lambda error, **_kwargs: {"ok": False, "error_code": str(error)},
    )
    monkeypatch.setattr(
        coordinator, "_emit_frame", lambda value: emitted.append(dict(value))
    )

    result = coordinator.main(["install-discord-token"])

    assert result == 2
    assert gate_calls == []
    assert emitted == [{"ok": False, "error_code": "tampered_source"}]


def test_coordinator_input_publication_retries_after_input_only_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = coordinator.CoordinatorInput(
        value={
            "revision": "1" * 40,
            "coordinator_input_sha256": "2" * 64,
        },
        writer_activation_plan=None,
        identities=None,
        artifacts={},
        base_plan=None,
    )
    stored: dict[Path, SimpleNamespace] = {}
    crash_once = True

    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(coordinator, "_lifecycle_lock", Lock)
    monkeypatch.setattr(
        coordinator,
        "_services_are_exactly_stopped_and_disabled",
        lambda: True,
    )
    monkeypatch.setattr(
        coordinator, "_validate_coordinator_input_live", lambda _v: None
    )
    monkeypatch.setattr(
        coordinator.os.path,
        "lexists",
        lambda path: path in stored,
    )
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda path, **_kwargs: stored.get(path),
    )

    def publish(path, raw, *, expected_previous_sha256):
        nonlocal crash_once
        if (
            path == coordinator.COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH
            and crash_once
        ):
            crash_once = False
            raise RuntimeError("power loss after input publication")
        snapshot = SimpleNamespace(
            raw=raw,
            sha256=coordinator._sha256_bytes(raw),
            item=SimpleNamespace(),
        )
        stored[path] = snapshot
        return SimpleNamespace(after=snapshot)

    monkeypatch.setattr(coordinator, "_publish_root_payload", publish)

    with pytest.raises(RuntimeError, match="power loss after input publication"):
        coordinator.publish_coordinator_input(value, now_unix=100)
    assert coordinator.COORDINATOR_INPUT_PATH in stored
    assert coordinator.COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH not in stored

    receipt = coordinator.publish_coordinator_input(value, now_unix=101)

    assert receipt["state"] == "published"
    assert coordinator.COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH in stored
    retry = coordinator.publish_coordinator_input(value, now_unix=999)
    assert retry == receipt


def _final_approval_request() -> coordinator.OwnerApprovalRequest:
    unsigned = {
        "schema": coordinator.OWNER_APPROVAL_REQUEST_SCHEMA,
        "ok": True,
        "state": "awaiting_final_owner_approval",
        "release_sha": "1" * 40,
        "coordinator_input_sha256": "2" * 64,
        "credential_prepare_approval_sha256": APPROVAL_SHA256,
        "owner_subject_sha256": "3" * 64,
        "approval_source_sha256": "4" * 64,
        "ephemeral_admin_username": ADMIN_USERNAME,
        "full_canary_plan_sha256": "5" * 64,
        "staged_plan_path": str(coordinator.DEFAULT_STAGED_PLAN_PATH),
        "staged_plan_file_sha256": "6" * 64,
        "approval_request_path": str(coordinator.OWNER_APPROVAL_REQUEST_PATH),
        "approval_path": str(coordinator.DEFAULT_APPROVAL_PATH),
        "hba_receipt_sha256": "7" * 64,
        "hba_expires_at_unix": 1_300,
        "fixture_expires_at_unix": 1_400,
        "credential_approval_expires_at_unix": 1_300,
        "requested_at_unix": 1_000,
        "approval_deadline_unix": 1_200,
        "owner_input_cutoff_unix": 1_170,
        "final_approval_transmit_margin_seconds": 30,
        "max_wait_seconds": 200,
        "prior_approval_file_sha256": None,
        "final_approval_frame_schema": coordinator.FINAL_APPROVAL_FRAME_SCHEMA,
    }
    return coordinator.OwnerApprovalRequest.from_mapping({
        **unsigned,
        "request_sha256": coordinator._sha256_json(unsigned),
    })


def _final_owner_approval_for_request(
    request: coordinator.OwnerApprovalRequest,
    *,
    approved_at_unix: int,
    expires_at_unix: int,
) -> coordinator.FullCanaryOwnerApproval:
    return coordinator.FullCanaryOwnerApproval.from_mapping({
        "schema": "muncho-full-canary-owner-approval.v1",
        "scope": "full_canary_runtime_start",
        "plan_sha256": request.value["full_canary_plan_sha256"],
        "authority_kind": ("trusted_root_bootstrap_out_of_band_owner"),
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": request.value["owner_subject_sha256"],
        "approval_source_sha256": request.value["approval_source_sha256"],
        "nonce_sha256": "8" * 64,
        "approved_at_unix": approved_at_unix,
        "expires_at_unix": expires_at_unix,
    })


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"M", "final_approval_frame_truncated"),
        (
            struct.pack("!4sI", coordinator.FINAL_APPROVAL_FRAME_MAGIC, 10) + b"{}",
            "final_approval_frame_truncated",
        ),
    ],
)
def test_partial_mfa1_is_a_hard_error_not_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    expected: str,
) -> None:
    read_fd = _pipe_frame(payload)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    try:
        with pytest.raises(coordinator.CoordinatorError, match=expected):
            coordinator._read_final_owner_approval_frame(fd=read_fd)
    finally:
        os.close(read_fd)


def test_zero_byte_mfa1_eof_is_the_only_no_secret_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd = _pipe_frame(b"")
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    try:
        with pytest.raises(
            coordinator._FinalApprovalNoSecretCancellation,
            match="final_approval_cancelled_no_secret",
        ):
            coordinator._read_final_owner_approval_frame(fd=read_fd)
    finally:
        os.close(read_fd)


def test_final_approval_zero_byte_cancel_is_exact_no_approval_mutation_and_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _final_approval_request()
    plan = SimpleNamespace(
        sha256=request.value["full_canary_plan_sha256"],
        to_mapping=lambda: {"plan": request.value["full_canary_plan_sha256"]},
    )
    coordinator_input = SimpleNamespace(
        revision=request.value["release_sha"],
        sha256=request.value["coordinator_input_sha256"],
    )
    credential = SimpleNamespace(
        sha256=request.value["credential_prepare_approval_sha256"],
        value={"owner_subject_sha256": request.value["owner_subject_sha256"]},
    )
    request_raw = coordinator._canonical_bytes(request.value)
    staged_raw = coordinator._canonical_bytes(plan.to_mapping())

    def artifact_item(*, inode: int, size: int) -> SimpleNamespace:
        return SimpleNamespace(
            st_dev=1,
            st_ino=inode,
            st_mode=0o100400,
            st_nlink=1,
            st_uid=0,
            st_gid=0,
            st_size=size,
            st_mtime_ns=1,
            st_ctime_ns=1,
        )

    request_snapshot = SimpleNamespace(
        raw=request_raw,
        sha256=coordinator._sha256_bytes(request_raw),
        item=artifact_item(inode=10, size=len(request_raw)),
    )
    staged_snapshot = SimpleNamespace(
        raw=staged_raw,
        sha256=request.value["staged_plan_file_sha256"],
        item=artifact_item(inode=11, size=len(staged_raw)),
    )

    def capture(path: Path, **_kwargs: object):
        if path == coordinator.OWNER_APPROVAL_REQUEST_PATH:
            return request_snapshot
        if path == coordinator.DEFAULT_STAGED_PLAN_PATH:
            return staged_snapshot
        if path == coordinator.DEFAULT_APPROVAL_PATH:
            return None
        raise AssertionError(f"unexpected snapshot path: {path}")

    emitted: list[dict[str, object]] = []
    read_fd = _pipe_frame(b"")

    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator,
        "_load_owner_approval_request_live",
        lambda: (request, plan, coordinator_input, credential),
    )
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        capture,
    )
    monkeypatch.setattr(coordinator, "_lifecycle_lock", Lock)
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_100)
    monkeypatch.setattr(
        coordinator,
        "_publish_root_payload",
        lambda *_args, **_kwargs: pytest.fail(
            "zero-byte cancellation must not publish an owner approval"
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_emit_frame",
        lambda value: emitted.append(dict(value)),
    )
    try:
        result = coordinator._execute_mutating_cli(
            command="install-final-approval",
            operation=lambda: coordinator.install_final_owner_approval(
                gate_emitter=coordinator._emit_frame,
                frame_reader=lambda: coordinator._read_final_owner_approval_frame(
                    fd=read_fd
                ),
            ),
        )
    finally:
        os.close(read_fd)

    assert result == 2
    assert emitted[0] == request.value
    receipt = emitted[1]
    assert set(receipt) == coordinator.FINAL_APPROVAL_CANCEL_RECEIPT_FIELDS
    assert receipt["schema"] == coordinator.FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA
    assert receipt["ok"] is False
    assert receipt["state"] == "cancelled_no_secret"
    assert receipt["frame_bytes_received"] == 0
    assert receipt["prior_approval_file_sha256"] is None
    assert receipt["observed_approval_file_sha256"] is None
    assert receipt["approval_path_matches_prior"] is True
    assert receipt["new_owner_approval_installed"] is False
    assert receipt["owner_approval_mutation_performed_by_this_helper"] is False
    assert (
        coordinator._parse_final_approval_cancel_receipt(
            receipt,
            request=request,
            plan=plan,
            coordinator_input=coordinator_input,
            credential_approval=credential,
            approval_request_before=request_snapshot,
            staged_plan_before=staged_snapshot,
        )
        == receipt
    )
    assert (
        owner_launcher.validate_final_approval_cancel_receipt(
            receipt,
            request=request.value,
            now_unix=1_100,
        )
        == receipt
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_input_cutoff_unix", 1_169),
        ("final_approval_transmit_margin_seconds", 29),
        ("approval_deadline_unix", 1_199),
        ("credential_approval_expires_at_unix", 1_229),
    ],
)
def test_owner_approval_request_rejects_inexact_cutoff_margin_or_deadline(
    field: str,
    value: int,
) -> None:
    request = dict(_final_approval_request().value)
    request[field] = value
    unsigned = {key: item for key, item in request.items() if key != "request_sha256"}
    request["request_sha256"] = coordinator._sha256_json(unsigned)

    with pytest.raises(
        coordinator.CoordinatorError,
        match="owner_approval_request_window_invalid",
    ):
        coordinator.OwnerApprovalRequest.from_mapping(request)


def test_final_approval_hard_deadline_is_bounded_by_credential_authority() -> None:
    assert (
        coordinator._final_approval_hard_deadline(
            hba_expires_at_unix=2_000,
            fixture_expires_at_unix=2_000,
            credential_approval_expires_at_unix=1_250,
        )
        == 1_220
    )


@pytest.mark.parametrize(
    ("consumer_name", "error_code"),
    [
        (
            "_require_install_final_owner_approval_binding",
            "final_owner_approval_not_bound",
        ),
        (
            "_require_runtime_final_owner_approval_binding",
            "owner_approval_ttl_or_binding_invalid",
        ),
    ],
)
@pytest.mark.parametrize(
    ("approved_at_unix", "expires_at_unix", "accepted"),
    [
        (1_170, 1_171, True),
        (1_171, 1_172, False),
        (1_100, 1_300, True),
        (1_100, 1_301, False),
    ],
)
def test_final_approval_consumers_enforce_cutoff_and_authority_boundaries(
    consumer_name: str,
    error_code: str,
    approved_at_unix: int,
    expires_at_unix: int,
    accepted: bool,
) -> None:
    request = _final_approval_request()
    approval = _final_owner_approval_for_request(
        request,
        approved_at_unix=approved_at_unix,
        expires_at_unix=expires_at_unix,
    )
    consumer = getattr(coordinator, consumer_name)

    if accepted:
        consumer(approval, request)
    else:
        with pytest.raises(coordinator.CoordinatorError, match=error_code):
            consumer(approval, request)


def test_worker_completion_is_read_only_truth_and_never_claims_safe_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    snapshot = _completion_snapshot(completion)
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_capture_root_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(
        coordinator,
        "recovery_gate",
        lambda: pytest.fail("completion preflight must not emit a takeover gate"),
    )

    observed = coordinator.preflight_recovery()

    assert observed == completion
    assert observed["recovery_worker_exit_proven"] is False
    assert observed["safe_to_delete_temporary_admin"] is False


def test_final_v2_rejects_self_digested_completion_projection_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_200)
    final = dict(
        coordinator._final_recovery_receipt_from_completion(
            coordinator_input=coordinator_input,
            completion=completion,
        )
    )
    final["cleanup_completed_at_unix"] = 1_101
    unsigned = {key: value for key, value in final.items() if key != "receipt_sha256"}
    final["receipt_sha256"] = coordinator._sha256_json(unsigned)

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_receipt_completion_projection_invalid",
    ):
        coordinator._parse_recovery_receipt_v2(
            final,
            coordinator_input=coordinator_input,
        )


def test_completion_rejects_self_digested_username_drift_from_original_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    completion["ephemeral_admin_username"] = "muncho_canary_admin_" + "b" * 16
    unsigned = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    completion["completion_sha256"] = coordinator._sha256_json(unsigned)

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_worker_completion_invalid",
    ):
        coordinator._parse_recovery_worker_completion(
            completion,
            coordinator_input=coordinator_input,
        )


@pytest.mark.parametrize(
    ("kind", "schema", "generation"),
    [
        ("run_process_lease", coordinator.RECOVERY_WORKER_LEASE_SCHEMA, 0),
        (
            "recovery_worker_lease",
            "muncho-full-canary-coordinator-process-lease.v1",
            1,
        ),
        ("run_process_lease", "muncho-full-canary-coordinator-process-lease.v1", 1),
        ("recovery_worker_lease", coordinator.RECOVERY_WORKER_LEASE_SCHEMA, 0),
    ],
)
def test_takeover_gate_rejects_crossed_predecessor_kind_schema_generation(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    schema: str,
    generation: int,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    gate.update(
        predecessor_kind=kind,
        predecessor_schema=schema,
        predecessor_generation=generation,
    )
    unsigned = {key: value for key, value in gate.items() if key != "gate_sha256"}
    gate["gate_sha256"] = coordinator._sha256_json(unsigned)

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_takeover_gate_invalid",
    ):
        coordinator._parse_recovery_takeover_gate(
            gate,
            coordinator_input=coordinator_input,
        )


def test_post_term_pid_reuse_skips_sigkill_and_proves_exact_target_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    closed: list[int] = []
    monkeypatch.setattr(
        coordinator, "_open_exact_recovery_target_pidfd", lambda **_kwargs: 88
    )
    monkeypatch.setattr(
        coordinator,
        "_pidfd_signal",
        lambda _pidfd, signum, **_kwargs: signals.append(signum) or True,
    )
    monkeypatch.setattr(
        coordinator, "_wait_for_pidfd_exit", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_target_is_exactly_alive",
        lambda *_args, **_kwargs: next(iter_alive),
    )
    monkeypatch.setattr(coordinator.os, "close", lambda fd: closed.append(fd))

    iter_alive = iter((True, False, False))
    attempted, proven = coordinator._terminate_exact_recovery_target(
        coordinator_input=SimpleNamespace(),
        target={"target_pid": 4321},
    )

    assert (attempted, proven) == (True, True)
    assert signals == [signal.SIGTERM]
    assert closed == [88]


@pytest.mark.parametrize(
    "failure_point",
    ["start", "owner_stat", "stat_parse", "cmdline", "executable"],
)
def test_proc_observation_uncertainty_is_never_treated_as_worker_absence(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    completion, coordinator_input = _recovery_completion()
    target = coordinator._worker_recovery_target(completion)
    monkeypatch.setattr(
        coordinator,
        "boot_identity",
        lambda: (target["target_boot_id_sha256"], target["target_boot_time_ns"]),
    )
    monkeypatch.setattr(
        coordinator,
        "process_start_time_ticks",
        (
            (lambda _pid: (_ for _ in ()).throw(PermissionError("denied")))
            if failure_point == "start"
            else lambda _pid: target["target_process_start_time_ticks"]
        ),
    )
    real_stat = Path.stat
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes

    def path_stat(path: Path):
        if path == Path(f"/proc/{target['target_pid']}"):
            if failure_point == "owner_stat":
                raise PermissionError("denied")
            return SimpleNamespace(st_uid=0, st_gid=0)
        return real_stat(path)

    def read_text(path: Path, *args, **kwargs):
        if path == Path(f"/proc/{target['target_pid']}/stat"):
            if failure_point == "stat_parse":
                return "malformed"
            return f"{target['target_pid']} (python) S 1 2 3"
        return real_read_text(path, *args, **kwargs)

    def read_bytes(path: Path):
        if path == Path(f"/proc/{target['target_pid']}/cmdline"):
            if failure_point == "cmdline":
                raise PermissionError("denied")
            return b"sealed-cmdline"
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "stat", path_stat)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        coordinator,
        "_expected_command_cmdline",
        lambda *_args, **_kwargs: b"sealed-cmdline",
    )
    monkeypatch.setattr(
        coordinator,
        "_process_executable_sha256",
        (
            (lambda _pid: (_ for _ in ()).throw(PermissionError("denied")))
            if failure_point == "executable"
            else lambda _pid: target["target_process_exe_sha256"]
        ),
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_process_identity_observation_failed",
    ):
        coordinator._recovery_target_is_exactly_alive(
            target,
            coordinator_input=coordinator_input,
        )


def test_finalizer_proc_permission_error_never_cas_or_safe_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    snapshot = _completion_snapshot(completion)
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_capture_root_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(
        coordinator,
        "boot_identity",
        lambda: (completion["recovery_worker_boot_id_sha256"], 0),
    )
    monkeypatch.setattr(
        coordinator,
        "process_start_time_ticks",
        lambda _pid: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(
        coordinator,
        "_open_recovery_process_lock",
        lambda **_kwargs: pytest.fail("uncertain identity cannot reach the lock"),
    )
    monkeypatch.setattr(
        coordinator,
        "_cas_recovery_journal",
        lambda **_kwargs: pytest.fail("uncertain identity cannot produce v2"),
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_process_identity_observation_failed",
    ):
        coordinator.finalize_recovery()


def test_legacy_v1_is_explicit_blocker_and_never_reaches_secret_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, coordinator_input = _legacy_recovery_receipt()
    raw = coordinator._canonical_bytes(receipt)
    snapshot = SimpleNamespace(
        path=coordinator.COORDINATOR_PROCESS_LEASE_PATH,
        raw=raw,
        sha256=coordinator._sha256_bytes(raw),
        item=_fake_stat(inode=91, size=len(raw)),
    )
    callbacks: list[str] = []
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_capture_root_snapshot", lambda *_args, **_kwargs: snapshot
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="legacy_recovery_receipt_reconciliation_required",
    ):
        coordinator.recover_full_canary(
            gate_emitter=lambda _value: callbacks.append("gate"),
            ack_reader=lambda **_kwargs: callbacks.append("ack"),
            admin_frame_reader=lambda **_kwargs: callbacks.append("secret"),
            admin_session_opener=lambda **_kwargs: callbacks.append("db"),
        )

    assert callbacks == []


def test_recovery_claim_persists_seq1_then_seq2_before_secret_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    coordinator_input.tls_server_name = "canary.europe-west3.sql.goog"
    coordinator_input.tls_peer_certificate_sha256 = "0" * 64
    predecessor = coordinator._load_recovery_predecessor(coordinator_input)
    ack = owner_launcher.build_recovery_ack(
        gate,
        now_unix=1_001,
        nonce=b"seq-one-two-recovery-nonce",
    )
    publications: list[tuple[object, dict[str, object]]] = []
    seq1_snapshot = SimpleNamespace(sha256="a" * 64)
    seq2_snapshot = SimpleNamespace(sha256="b" * 64)

    def cas(*, expected, value):
        publications.append((expected, dict(value)))
        after = seq1_snapshot if value["transition_seq"] == 1 else seq2_snapshot
        return SimpleNamespace(after=after)

    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_001)
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda *_args, **_kwargs: predecessor.snapshot,
    )
    monkeypatch.setattr(coordinator, "_snapshot_is_exact", lambda *_args: True)
    monkeypatch.setattr(
        coordinator,
        "_terminate_exact_recovery_target",
        lambda **_kwargs: (False, True),
    )
    monkeypatch.setattr(
        coordinator, "_open_recovery_process_lock", lambda **_kwargs: 55
    )
    monkeypatch.setattr(
        coordinator,
        "_current_recovery_worker_identity",
        lambda _input: _recovery_worker_identity(),
    )
    monkeypatch.setattr(
        coordinator,
        "_recovery_target_is_exactly_alive",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(coordinator, "_cas_recovery_journal", cas)

    claim = coordinator._claim_recovery_worker(
        coordinator_input=coordinator_input,
        predecessor=predecessor,
        gate=gate,
        ack=ack,
    )
    assert isinstance(claim, coordinator._RecoveryWorkerClaim)
    transitioned = coordinator._transition_recovery_worker_to_admin_authority(
        coordinator_input=coordinator_input,
        claim=claim,
    )
    secret_gate = coordinator._build_recovery_secret_gate(
        coordinator_input=coordinator_input,
        claim=transitioned,
    )

    assert [value["state"] for _expected, value in publications] == [
        "claimed_awaiting_admin",
        "admin_authority_may_be_in_use",
    ]
    assert publications[0][0] is predecessor.snapshot
    assert publications[1][0] is seq1_snapshot
    assert publications[0][1]["previous_transition_sha256"] == (
        predecessor.snapshot.sha256
    )
    assert publications[1][1]["previous_transition_sha256"] == (seq1_snapshot.sha256)
    assert secret_gate["recovery_worker_transition_seq"] == 2
    assert (
        secret_gate["recovery_worker_lease_sha256"]
        == (transitioned.lease["worker_lease_sha256"])
    )


def test_seq2_cas_failure_emits_no_secret_gate_and_releases_worker_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    predecessor = coordinator._load_recovery_predecessor(coordinator_input)
    claim = SimpleNamespace(lock_fd=55)
    emitted: list[Mapping[str, object]] = []
    closed: list[int] = []
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_load_recovery_predecessor", lambda _input: predecessor
    )
    monkeypatch.setattr(coordinator, "recovery_gate", lambda: gate)
    monkeypatch.setattr(
        coordinator,
        "_claim_recovery_worker",
        lambda **_kwargs: claim,
    )
    monkeypatch.setattr(
        coordinator,
        "_transition_recovery_worker_to_admin_authority",
        lambda **_kwargs: (_ for _ in ()).throw(
            coordinator.CoordinatorError("recovery_journal_cas_lost")
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_build_recovery_secret_gate",
        lambda **_kwargs: pytest.fail("secret gate must follow durable seq2"),
    )
    monkeypatch.setattr(
        coordinator, "_close_recovery_process_lock", lambda fd: closed.append(fd)
    )

    with pytest.raises(coordinator.CoordinatorError, match="recovery_journal_cas_lost"):
        coordinator.recover_full_canary(
            gate_emitter=lambda value: emitted.append(value),
            ack_reader=lambda **_kwargs: {"ack_sha256": "a" * 64},
            admin_frame_reader=lambda **_kwargs: pytest.fail(
                "admin frame must not be read"
            ),
            admin_session_opener=lambda **_kwargs: pytest.fail(
                "admin session must not open"
            ),
        )

    assert emitted == [gate]
    assert closed == [55]


def test_lock_contended_claim_is_zero_secret_causally_bound_loser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, coordinator_input = _recovery_gate_fixture(monkeypatch)
    predecessor = coordinator._load_recovery_predecessor(coordinator_input)
    ack = owner_launcher.build_recovery_ack(
        gate,
        now_unix=1_001,
        nonce=b"concurrent-recovery-loser-nonce",
    )
    successor_snapshot = SimpleNamespace(sha256="f" * 64)
    successor = {"schema": coordinator.RECOVERY_WORKER_LEASE_SCHEMA}
    successor_parsed = {
        "schema": coordinator.RECOVERY_WORKER_LEASE_SCHEMA,
        "state": "claimed_awaiting_admin",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": gate[
            "credential_prepare_approval_sha256"
        ],
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "original_run_process_lease_sha256": gate["original_run_process_lease_sha256"],
        "causal_recovery_state_sha256": gate["causal_recovery_state_sha256"],
        "predecessor_journal_sha256": predecessor.snapshot.sha256,
        "recovery_generation": 1,
        **_recovery_worker_identity(),
    }
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(coordinator, "_harden_secret_process", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_load_recovery_predecessor", lambda _input: predecessor
    )
    monkeypatch.setattr(coordinator, "recovery_gate", lambda: gate)
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_001)
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda *_args, **_kwargs: predecessor.snapshot,
    )
    monkeypatch.setattr(coordinator, "_snapshot_is_exact", lambda *_args: True)
    monkeypatch.setattr(
        coordinator,
        "_terminate_exact_recovery_target",
        lambda **_kwargs: (False, True),
    )
    monkeypatch.setattr(
        coordinator, "_open_recovery_process_lock", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        coordinator,
        "_observe_concurrent_recovery_successor",
        lambda **_kwargs: (successor, successor_snapshot),
    )
    monkeypatch.setattr(
        coordinator,
        "_successor_worker_identity",
        lambda *_args, **_kwargs: (successor_parsed, 1),
    )

    result = coordinator.recover_full_canary(
        gate_emitter=lambda value: emitted.append(dict(value)),
        ack_reader=lambda **_kwargs: ack,
        admin_frame_reader=lambda **_kwargs: pytest.fail(
            "claim loser must not read an admin frame"
        ),
        admin_session_opener=lambda **_kwargs: pytest.fail(
            "claim loser must not open an admin session"
        ),
    )

    assert emitted == [gate]
    assert result["state"] == "recovery_worker_claim_lost_no_secret"
    assert result["observed_successor_journal_sha256"] == successor_snapshot.sha256
    assert result["observed_successor_generation"] == 1
    assert result["secret_gate_emitted_by_loser"] is False
    assert result["admin_frame_bytes_received_by_loser"] == 0
    assert result["admin_session_opened_by_loser"] is False


def test_completion_worker_hang_is_killed_then_finalized_by_lock_and_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    snapshot = _completion_snapshot(completion)
    signals: list[int] = []
    closed: list[int] = []
    cas_values: list[dict[str, object]] = []
    alive = iter((True, True, False, False))
    waits = iter((False, True))
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_capture_root_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_200)
    monkeypatch.setattr(
        coordinator,
        "_recovery_target_is_exactly_alive",
        lambda *_args, **_kwargs: next(alive),
    )
    monkeypatch.setattr(
        coordinator, "_open_exact_recovery_target_pidfd", lambda **_kwargs: 88
    )
    monkeypatch.setattr(
        coordinator,
        "_pidfd_signal",
        lambda _pidfd, signum, **_kwargs: signals.append(signum) or True,
    )
    monkeypatch.setattr(
        coordinator,
        "_wait_for_pidfd_exit",
        lambda *_args, **_kwargs: next(waits),
    )
    monkeypatch.setattr(coordinator.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(
        coordinator, "_open_recovery_process_lock", lambda **_kwargs: 77
    )
    monkeypatch.setattr(
        coordinator, "_close_recovery_process_lock", lambda fd: closed.append(fd)
    )
    monkeypatch.setattr(
        coordinator,
        "_cas_recovery_journal",
        lambda **kwargs: cas_values.append(dict(kwargs["value"])),
    )
    monkeypatch.setattr(
        coordinator,
        "_read_recovery_admin_frame",
        lambda **_kwargs: pytest.fail("finalizer has no secret phase"),
    )

    receipt = coordinator.finalize_recovery()

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert closed == [88, 77]
    assert receipt == cas_values[0]
    assert receipt["schema"] == coordinator.RECOVERY_RECEIPT_SCHEMA
    assert (
        receipt["recovery_worker_completion_sha256"]
        == (completion["completion_sha256"])
    )
    assert receipt["recovery_worker_exit_proven"] is True
    assert receipt["safe_to_delete_temporary_admin"] is True


@pytest.mark.parametrize("pending_mode", ["termination", "lock", "alive_after_lock"])
def test_finalizer_pending_paths_never_claim_safe_delete_or_read_secrets(
    monkeypatch: pytest.MonkeyPatch,
    pending_mode: str,
) -> None:
    completion, coordinator_input = _recovery_completion()
    snapshot = _completion_snapshot(completion)
    closed: list[int] = []
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_capture_root_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_200)
    if pending_mode == "termination":
        monkeypatch.setattr(
            coordinator,
            "_terminate_exact_recovery_target",
            lambda **_kwargs: (_ for _ in ()).throw(
                coordinator.CoordinatorError("recovery_process_termination_unconfirmed")
            ),
        )
        monkeypatch.setattr(
            coordinator,
            "_open_recovery_process_lock",
            lambda **_kwargs: pytest.fail("lock follows termination proof"),
        )
    else:
        monkeypatch.setattr(
            coordinator,
            "_terminate_exact_recovery_target",
            lambda **_kwargs: (True, True),
        )
        monkeypatch.setattr(
            coordinator,
            "_open_recovery_process_lock",
            lambda **_kwargs: None if pending_mode == "lock" else 77,
        )
    monkeypatch.setattr(
        coordinator,
        "_recovery_target_is_exactly_alive",
        lambda *_args, **_kwargs: pending_mode in {"termination", "alive_after_lock"},
    )
    monkeypatch.setattr(
        coordinator, "_close_recovery_process_lock", lambda fd: closed.append(fd)
    )
    monkeypatch.setattr(
        coordinator,
        "_cas_recovery_journal",
        lambda **_kwargs: pytest.fail("pending finalization must not CAS success"),
    )
    monkeypatch.setattr(
        coordinator,
        "_read_recovery_admin_frame",
        lambda **_kwargs: pytest.fail("finalizer must not read a secret"),
    )

    pending = coordinator.finalize_recovery()

    assert pending["schema"] == coordinator.RECOVERY_FINALIZE_PENDING_RECEIPT_SCHEMA
    assert pending["state"] == "recovery_finalization_pending_no_secret"
    assert pending["completion_cas_succeeded_by_finalizer"] is False
    assert pending["secret_gate_emitted_by_finalizer"] is False
    assert pending["admin_frame_bytes_received_by_finalizer"] == 0
    assert pending["admin_session_opened_by_finalizer"] is False
    assert pending["retryable"] is True
    assert closed == ([77] if pending_mode == "alive_after_lock" else [])


def test_concurrent_finalizer_accepts_only_exact_persisted_v2_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    completion_snapshot = _completion_snapshot(completion)
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_200)
    final = coordinator._final_recovery_receipt_from_completion(
        coordinator_input=coordinator_input,
        completion=completion,
    )
    final_raw = coordinator._canonical_bytes(final)
    final_snapshot = SimpleNamespace(
        path=coordinator.COORDINATOR_PROCESS_LEASE_PATH,
        raw=final_raw,
        sha256=coordinator._sha256_bytes(final_raw),
        item=_fake_stat(inode=92, size=len(final_raw)),
    )
    snapshots = iter((completion_snapshot, final_snapshot))
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        coordinator,
        "_terminate_exact_recovery_target",
        lambda **_kwargs: (False, True),
    )
    monkeypatch.setattr(
        coordinator, "_open_recovery_process_lock", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        coordinator,
        "_cas_recovery_journal",
        lambda **_kwargs: pytest.fail("concurrent finalizer already committed v2"),
    )

    assert coordinator.finalize_recovery() == final


def test_concurrent_finalizer_rejects_self_consistent_different_completion_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    completion_snapshot = _completion_snapshot(completion)
    different = dict(completion)
    different["cleanup_completed_at_unix"] = 1_101
    different_unsigned = {
        key: value for key, value in different.items() if key != "completion_sha256"
    }
    different["completion_sha256"] = coordinator._sha256_json(different_unsigned)
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_200)
    forged_successor = coordinator._final_recovery_receipt_from_completion(
        coordinator_input=coordinator_input,
        completion=different,
    )
    forged_raw = coordinator._canonical_bytes(forged_successor)
    forged_snapshot = SimpleNamespace(
        path=coordinator.COORDINATOR_PROCESS_LEASE_PATH,
        raw=forged_raw,
        sha256=coordinator._sha256_bytes(forged_raw),
        item=_fake_stat(inode=94, size=len(forged_raw)),
    )
    snapshots = iter((completion_snapshot, forged_snapshot))
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator,
        "_capture_root_snapshot",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        coordinator,
        "_terminate_exact_recovery_target",
        lambda **_kwargs: (False, True),
    )
    monkeypatch.setattr(
        coordinator, "_open_recovery_process_lock", lambda **_kwargs: None
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="recovery_final_successor_state_conflict",
    ):
        coordinator.finalize_recovery()


def test_persisted_v2_finalization_is_idempotent_without_process_or_lock_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, coordinator_input = _recovery_completion()
    monkeypatch.setattr(
        coordinator, "_parse_process_lease", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_200)
    final = coordinator._final_recovery_receipt_from_completion(
        coordinator_input=coordinator_input,
        completion=completion,
    )
    raw = coordinator._canonical_bytes(final)
    snapshot = SimpleNamespace(
        path=coordinator.COORDINATOR_PROCESS_LEASE_PATH,
        raw=raw,
        sha256=coordinator._sha256_bytes(raw),
        item=_fake_stat(inode=93, size=len(raw)),
    )
    monkeypatch.setattr(coordinator, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        coordinator, "load_coordinator_input", lambda: coordinator_input
    )
    monkeypatch.setattr(
        coordinator, "_capture_root_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(
        coordinator,
        "_terminate_exact_recovery_target",
        lambda **_kwargs: pytest.fail("persisted v2 is already terminal"),
    )
    monkeypatch.setattr(
        coordinator,
        "_open_recovery_process_lock",
        lambda **_kwargs: pytest.fail("persisted v2 needs no lock"),
    )

    assert coordinator.finalize_recovery() == final


def test_partial_mrc2_after_stage2_gate_leaves_seq2_without_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, secret_gate, _input, claim, emitted, closed = _stage2_recovery_harness(
        monkeypatch
    )
    read_fd = _pipe_frame(coordinator.RECOVERY_ADMIN_FRAME_MAGIC + b"partial")
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(
        coordinator,
        "_publish_recovery_worker_completion",
        lambda **_kwargs: pytest.fail("partial MRC2 cannot publish completion"),
    )
    try:
        with pytest.raises(coordinator.CoordinatorError, match="admin_frame_truncated"):
            coordinator.recover_full_canary(
                gate_emitter=lambda value: emitted.append(dict(value)),
                ack_reader=lambda **_kwargs: {"ack_sha256": "a" * 64},
                admin_frame_reader=lambda **kwargs: (
                    coordinator._read_recovery_admin_frame(fd=read_fd, **kwargs)
                ),
                admin_session_opener=lambda **_kwargs: pytest.fail(
                    "partial MRC2 cannot open DB"
                ),
            )
    finally:
        os.close(read_fd)

    assert emitted == [gate, secret_gate]
    assert claim.lease["state"] == "admin_authority_may_be_in_use"
    assert claim.lease["transition_seq"] == 2
    assert closed == [55]


def test_full_mrc2_db_open_failure_zeroizes_frame_and_leaves_seq2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, secret_gate, _input, claim, emitted, closed = _stage2_recovery_harness(
        monkeypatch
    )
    payload = owner_launcher.build_recovery_admin_frame(
        secret_gate,
        ADMIN_USERNAME,
        bytearray(ADMIN_PASSWORD),
    )
    read_fd = _pipe_frame(bytes(payload))
    coordinator._zeroize(payload)
    frames: list[coordinator.OpaqueStdinAdminFrame] = []
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)

    def read(**kwargs):
        frame = coordinator._read_recovery_admin_frame(fd=read_fd, **kwargs)
        frames.append(frame)
        return frame

    try:
        with pytest.raises(RuntimeError, match="db open lost"):
            coordinator.recover_full_canary(
                gate_emitter=lambda value: emitted.append(dict(value)),
                ack_reader=lambda **_kwargs: {"ack_sha256": "a" * 64},
                admin_frame_reader=read,
                admin_session_opener=lambda **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("db open lost")
                ),
            )
    finally:
        os.close(read_fd)

    assert emitted == [gate, secret_gate]
    assert claim.lease["transition_seq"] == 2
    assert frames and frames[0].consumed is True
    assert closed == [55]


def test_db_open_cleanup_publishes_completion_but_not_safe_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, secret_gate, _input, claim, emitted, closed = _stage2_recovery_harness(
        monkeypatch
    )
    payload = owner_launcher.build_recovery_admin_frame(
        secret_gate,
        ADMIN_USERNAME,
        bytearray(ADMIN_PASSWORD),
    )
    read_fd = _pipe_frame(bytes(payload))
    coordinator._zeroize(payload)
    completion, _completion_input = _recovery_completion()

    class Session:
        closed = False

        def close(self):
            self.closed = True

    session = Session()

    def open_session(*, frame, **_kwargs):
        password = frame.consume_password()
        coordinator._zeroize(password)
        return session

    cleanup = {key: completion[key] for key in coordinator._RECOVERY_CLEANUP_FIELDS}

    def perform_cleanup(**_kwargs):
        session.close()
        return cleanup

    published: list[dict[str, object]] = []
    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(coordinator, "_perform_recovery_cleanup", perform_cleanup)
    monkeypatch.setattr(
        coordinator,
        "_publish_recovery_worker_completion",
        lambda **kwargs: published.append(dict(kwargs["cleanup"])) or completion,
    )
    try:
        result = coordinator.recover_full_canary(
            gate_emitter=lambda value: emitted.append(dict(value)),
            ack_reader=lambda **_kwargs: {"ack_sha256": "a" * 64},
            admin_frame_reader=lambda **kwargs: coordinator._read_recovery_admin_frame(
                fd=read_fd,
                **kwargs,
            ),
            admin_session_opener=open_session,
        )
    finally:
        os.close(read_fd)

    assert emitted == [gate, secret_gate]
    assert published == [cleanup]
    assert result == completion
    assert result["recovery_worker_exit_proven"] is False
    assert result["safe_to_delete_temporary_admin"] is False
    assert session.closed is True
    assert closed == [55]


def test_post_mrc2_cleanup_and_session_close_failure_receipt_is_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, secret_gate, _input, _claim, emitted, closed = _stage2_recovery_harness(
        monkeypatch
    )
    payload = owner_launcher.build_recovery_admin_frame(
        secret_gate,
        ADMIN_USERNAME,
        bytearray(ADMIN_PASSWORD),
    )
    read_fd = _pipe_frame(bytes(payload))
    coordinator._zeroize(payload)

    class Session:
        closed = False
        close_attempted = False

        def close(self):
            self.close_attempted = True
            raise RuntimeError("session close lost")

    session = Session()

    def open_session(*, frame, **_kwargs):
        password = frame.consume_password()
        coordinator._zeroize(password)
        return session

    monkeypatch.setattr(coordinator, "ADMIN_FRAME_FD", read_fd)
    monkeypatch.setattr(coordinator.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(
        coordinator,
        "_perform_recovery_cleanup",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cleanup lost")),
    )
    try:
        with pytest.raises(RuntimeError, match="cleanup lost") as caught:
            coordinator.recover_full_canary(
                gate_emitter=lambda value: emitted.append(dict(value)),
                ack_reader=lambda **_kwargs: {"ack_sha256": "a" * 64},
                admin_frame_reader=lambda **kwargs: (
                    coordinator._read_recovery_admin_frame(fd=read_fd, **kwargs)
                ),
                admin_session_opener=open_session,
            )
    finally:
        os.close(read_fd)

    monkeypatch.setattr(
        coordinator,
        "_stable_root_read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(coordinator.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        coordinator, "_services_are_exactly_stopped_and_disabled", lambda: True
    )
    failure = coordinator._unbound_failure(caught.value, command="recover")

    assert emitted == [gate, secret_gate]
    assert session.close_attempted is True
    assert session.closed is False
    assert failure["admin_session_closed"] is False
    assert failure["cleanup_status"] == "cleanup_blocked"
    assert failure["recovery_material_preserved"] is True
    assert closed == [55]


def _final_cancel_parser_fixture(
    *,
    request_state: str,
    staged_state: str,
    owner_state: str = "matching_prior",
    cancelled_at_unix: int,
) -> tuple[dict[str, object], tuple[object, ...]]:
    request = _final_approval_request()
    plan = SimpleNamespace(sha256=request.value["full_canary_plan_sha256"])
    coordinator_input = SimpleNamespace(
        revision=request.value["release_sha"],
        sha256=request.value["coordinator_input_sha256"],
    )
    credential = SimpleNamespace(
        sha256=request.value["credential_prepare_approval_sha256"],
    )
    request_raw = coordinator._canonical_bytes(request.value)
    request_before = SimpleNamespace(
        sha256=coordinator._sha256_bytes(request_raw),
    )
    staged_before = SimpleNamespace(
        sha256=request.value["staged_plan_file_sha256"],
    )
    observed_request = (
        None
        if request_state == "retired_absent"
        else ("a" * 64 if request_state == "superseded" else request_before.sha256)
    )
    observed_staged = (
        None
        if staged_state == "retired_absent"
        else ("a" * 64 if staged_state == "superseded" else staged_before.sha256)
    )
    observed_owner = None if owner_state == "matching_prior" else "d" * 64
    conflict = coordinator._final_approval_cancel_has_state_conflict(
        request_state=request_state,
        staged_state=staged_state,
        owner_state=owner_state,
    )
    unsigned: dict[str, object] = {
        "schema": coordinator.FINAL_APPROVAL_CANCEL_RECEIPT_SCHEMA,
        "ok": False,
        "state": (
            "cancelled_no_secret_state_conflict" if conflict else "cancelled_no_secret"
        ),
        "reason": "eof_before_mfa1",
        "release_sha": coordinator_input.revision,
        "coordinator_input_sha256": coordinator_input.sha256,
        "credential_prepare_approval_sha256": credential.sha256,
        "owner_subject_sha256": request.value["owner_subject_sha256"],
        "full_canary_plan_sha256": plan.sha256,
        "approval_request_sha256": request.sha256,
        "approval_request_path": str(coordinator.OWNER_APPROVAL_REQUEST_PATH),
        "expected_approval_request_file_sha256": request_before.sha256,
        "observed_approval_request_file_sha256": observed_request,
        "approval_request_artifact_state": request_state,
        "approval_request_present": request_state != "retired_absent",
        "approval_request_remains_active": request_state == "matching_active",
        "staged_plan_path": str(coordinator.DEFAULT_STAGED_PLAN_PATH),
        "expected_staged_plan_file_sha256": staged_before.sha256,
        "observed_staged_plan_file_sha256": observed_staged,
        "staged_plan_artifact_state": staged_state,
        "staged_plan_present": staged_state != "retired_absent",
        "approval_path": str(coordinator.DEFAULT_APPROVAL_PATH),
        "prior_approval_file_sha256": None,
        "observed_approval_file_sha256": observed_owner,
        "owner_approval_artifact_state": owner_state,
        "approval_path_matches_prior": owner_state == "matching_prior",
        "new_owner_approval_installed": (
            False if owner_state == "matching_prior" else None
        ),
        "frame_bytes_received": 0,
        "owner_approval_mutation_performed_by_this_helper": False,
        "cancelled_at_unix": cancelled_at_unix,
    }
    receipt = {**unsigned, "receipt_sha256": coordinator._sha256_json(unsigned)}
    context = (
        request,
        plan,
        coordinator_input,
        credential,
        request_before,
        staged_before,
    )
    return receipt, context


@pytest.mark.parametrize(
    (
        "request_state",
        "staged_state",
        "owner_state",
        "cancelled_at_unix",
        "expected_state",
    ),
    [
        (
            "matching_active",
            "matching_present",
            "matching_prior",
            1_100,
            "cancelled_no_secret",
        ),
        (
            "matching_expired",
            "matching_present",
            "matching_prior",
            1_201,
            "cancelled_no_secret",
        ),
        (
            "retired_absent",
            "retired_absent",
            "matching_prior",
            1_201,
            "cancelled_no_secret",
        ),
        (
            "matching_active",
            "retired_absent",
            "matching_prior",
            1_100,
            "cancelled_no_secret_state_conflict",
        ),
        (
            "retired_absent",
            "matching_present",
            "matching_prior",
            1_201,
            "cancelled_no_secret_state_conflict",
        ),
        (
            "drifted",
            "matching_present",
            "matching_prior",
            1_100,
            "cancelled_no_secret_state_conflict",
        ),
        (
            "matching_active",
            "matching_present",
            "drifted",
            1_100,
            "cancelled_no_secret_state_conflict",
        ),
    ],
)
def test_late_cancel_v2_exact_state_pair_matrix(
    request_state: str,
    staged_state: str,
    owner_state: str,
    cancelled_at_unix: int,
    expected_state: str,
) -> None:
    receipt, context = _final_cancel_parser_fixture(
        request_state=request_state,
        staged_state=staged_state,
        owner_state=owner_state,
        cancelled_at_unix=cancelled_at_unix,
    )
    request, plan, coordinator_input, credential, request_before, staged_before = (
        context
    )

    parsed = coordinator._parse_final_approval_cancel_receipt(
        receipt,
        request=request,
        plan=plan,
        coordinator_input=coordinator_input,
        credential_approval=credential,
        approval_request_before=request_before,
        staged_plan_before=staged_before,
    )

    assert parsed["state"] == expected_state


def test_late_cancel_rejects_forged_null_to_null_owner_drift() -> None:
    receipt, context = _final_cancel_parser_fixture(
        request_state="matching_active",
        staged_state="matching_present",
        owner_state="drifted",
        cancelled_at_unix=1_100,
    )
    receipt["observed_approval_file_sha256"] = None
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = coordinator._sha256_json(unsigned)
    request, plan, coordinator_input, credential, request_before, staged_before = (
        context
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="final_approval_cancel_receipt_invalid",
    ):
        coordinator._parse_final_approval_cancel_receipt(
            receipt,
            request=request,
            plan=plan,
            coordinator_input=coordinator_input,
            credential_approval=credential,
            approval_request_before=request_before,
            staged_plan_before=staged_before,
        )
