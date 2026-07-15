from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.canonical_writer_boundary as boundary
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.canonical_writer_readiness import (
    READINESS_RECEIPT_VERSION,
    WRITER_LIVENESS_RECEIPT_VERSION,
    attest_canonical_writer_liveness,
    attest_canonical_writer_startup_readiness,
    current_python_runtime_identity,
    module_file_identity,
    notify_systemd_writer_liveness,
    notify_systemd_writer_readiness,
    readiness_receipt_sha256,
    writer_liveness_status_text,
)


def _valid_ping(request_id: str) -> dict[str, str]:
    return {
        "request_id": request_id,
        "service": "canonical_writer",
        "protocol": "v1",
        "database_identity": CANONICAL_WRITER_MIGRATION_OWNER,
    }


def _runtime_identity() -> dict[str, object]:
    return {
        "effective_import_paths": ["/opt/releases/revision/site-packages"],
        "unexpected_import_paths": [],
        "loaded_module_origins": [
            "/opt/releases/revision/site-packages/gateway/"
            "canonical_writer_readiness.py"
        ],
        "unexpected_import_origins": [],
        "loaded_module_origins_complete": True,
        "effective_environment_variable_names": ["NOTIFY_SOCKET"],
        "effective_environment_variable_value_sha256": {
            "NOTIFY_SOCKET": hashlib.sha256(b"@notify").hexdigest()
        },
    }


def test_enabled_boundary_pings_before_writing_process_bound_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        boundary,
        "frozen_writer_boundary_config",
        lambda: SimpleNamespace(enabled=True),
    )
    request_id = str(uuid.uuid4())
    calls: list[tuple[object, object]] = []

    def writer_call(operation, payload):
        calls.append((operation, payload))
        return _valid_ping(request_id)

    receipt_path = tmp_path / "canonical-writer-readiness.json"
    receipt = attest_canonical_writer_startup_readiness(
        receipt_path=receipt_path,
        _writer_call=writer_call,
        _now_unix=lambda: 1_800_000_000,
        _pid=4242,
        _boot_identity_provider=lambda: ("b" * 64, 987654321),
        _process_start_time=lambda pid: 123456 if pid == 4242 else 0,
        _module_identity_provider=lambda: (
            "/opt/releases/revision/site-packages/gateway/"
            "canonical_writer_readiness.py",
            "a" * 64,
        ),
        _process_hardening_provider=lambda: (False, 0, 0),
        _python_runtime_provider=_runtime_identity,
    )

    assert calls == [(CanonicalWriterOperation.PING, {})]
    assert receipt is not None
    assert receipt["version"] == READINESS_RECEIPT_VERSION
    assert receipt["gateway_pid"] == 4242
    assert receipt["gateway_start_time_ticks"] == 123456
    assert receipt["boot_id_sha256"] == "b" * 64
    assert receipt["observed_at_boottime_ns"] == 987654321
    assert receipt["writer_request_id"] == request_id
    assert receipt["gateway_dumpable"] is False
    assert receipt["gateway_core_soft_limit"] == 0
    assert receipt["gateway_core_hard_limit"] == 0
    assert receipt["effective_environment_variable_value_sha256"] == {
        "NOTIFY_SOCKET": hashlib.sha256(b"@notify").hexdigest()
    }
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600


def test_disabled_boundary_is_exact_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        boundary,
        "frozen_writer_boundary_config",
        lambda: SimpleNamespace(enabled=False),
    )
    called = False

    def writer_call(*_args, **_kwargs):
        nonlocal called
        called = True

    result = attest_canonical_writer_startup_readiness(
        receipt_path=tmp_path / "receipt.json",
        _writer_call=writer_call,
    )

    assert result is None
    assert called is False
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(service="other"),
        lambda value: value.update(protocol="v2"),
        lambda value: value.update(database_identity="canonical_writer"),
        lambda value: value.update(request_id="not-a-uuid"),
        lambda value: value.update(extra="ambiguous"),
    ],
)
def test_invalid_ping_never_creates_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    monkeypatch.setattr(
        boundary,
        "frozen_writer_boundary_config",
        lambda: SimpleNamespace(enabled=True),
    )
    response = _valid_ping(str(uuid.uuid4()))
    mutate(response)
    receipt_path = tmp_path / "receipt.json"

    with pytest.raises(RuntimeError):
        attest_canonical_writer_startup_readiness(
            receipt_path=receipt_path,
            _writer_call=lambda *_args, **_kwargs: response,
            _pid=4242,
            _boot_identity_provider=lambda: ("b" * 64, 987654321),
            _process_start_time=lambda _pid: 123456,
            _module_identity_provider=lambda: ("/module.py", "a" * 64),
            _process_hardening_provider=lambda: (False, 0, 0),
            _python_runtime_provider=_runtime_identity,
        )

    assert not receipt_path.exists()


def test_systemd_notify_binds_status_to_receipt_digest(tmp_path: Path) -> None:
    del tmp_path
    notify_path = Path("/tmp") / f"cw-notify-{uuid.uuid4().hex}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    listener.bind(str(notify_path))
    listener.settimeout(2)
    receipt = {
        "version": READINESS_RECEIPT_VERSION,
        "gateway_pid": os.getpid(),
    }
    try:
        assert notify_systemd_writer_readiness(
            receipt,
            ready=True,
            _notify_socket=str(notify_path),
        )
        payload = listener.recv(4096).decode("utf-8")
    finally:
        listener.close()
        notify_path.unlink(missing_ok=True)

    assert payload == (
        "READY=1\n"
        f"STATUS={READINESS_RECEIPT_VERSION}:"
        f"{readiness_receipt_sha256(receipt)}\n"
    )


def test_missing_notify_socket_is_not_fabricated() -> None:
    assert (
        notify_systemd_writer_readiness(
            {"version": READINESS_RECEIPT_VERSION},
            ready=False,
            _notify_socket="",
        )
        is False
    )


def test_systemd_liveness_status_binds_startup_generation_and_receipt() -> None:
    notify_path = Path("/tmp") / f"cw-live-notify-{uuid.uuid4().hex}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    listener.bind(str(notify_path))
    listener.settimeout(2)
    startup_digest = "a" * 64
    receipt = {
        "version": WRITER_LIVENESS_RECEIPT_VERSION,
        "generation": 9,
        "proof": "digest-only-status",
    }
    liveness_digest = readiness_receipt_sha256(receipt)
    try:
        assert notify_systemd_writer_liveness(
            startup_digest,
            receipt,
            _notify_socket=str(notify_path),
        )
        payload = listener.recv(4096).decode("utf-8")
    finally:
        listener.close()
        notify_path.unlink(missing_ok=True)

    assert payload == (
        "STATUS="
        f"{WRITER_LIVENESS_RECEIPT_VERSION}:{startup_digest}:9:"
        f"{liveness_digest}\n"
    )
    assert writer_liveness_status_text(
        startup_digest,
        9,
        liveness_digest,
    ) in payload


@pytest.mark.parametrize(
    "startup,generation,liveness",
    [
        ("A" * 64, 1, "b" * 64),
        ("a" * 64, True, "b" * 64),
        ("a" * 64, 0, "b" * 64),
        ("a" * 64, 1, "not-a-digest"),
    ],
)
def test_liveness_status_shape_is_exact(startup, generation, liveness) -> None:
    with pytest.raises(ValueError):
        writer_liveness_status_text(startup, generation, liveness)


def test_module_identity_rejects_symlink_origin(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    link = tmp_path / "module-link.py"
    link.symlink_to(module)

    with pytest.raises(RuntimeError, match="module origin is invalid"):
        module_file_identity(link)


def test_runtime_identity_hashes_environment_values_without_recording_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "not-for-receipt"
    monkeypatch.setenv("MUNCHO_RUNTIME_TEST", secret_value)

    identity = current_python_runtime_identity()

    assert identity["effective_environment_variable_names"] == sorted(os.environ)
    assert identity["effective_environment_variable_value_sha256"][
        "MUNCHO_RUNTIME_TEST"
    ] == hashlib.sha256(secret_value.encode()).hexdigest()
    assert secret_value not in json.dumps(identity, sort_keys=True)


@pytest.mark.parametrize(
    "environment_names,environment_digests",
    [
        (["NOTIFY_SOCKET"], {}),
        (["NOTIFY_SOCKET"], {"OTHER": "a" * 64}),
        (["NOTIFY_SOCKET"], {"NOTIFY_SOCKET": "not-a-digest"}),
        (["A", "B"], {"B": "b" * 64, "A": "a" * 64}),
    ],
)
def test_readiness_rejects_non_exact_environment_digest_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_names: list[str],
    environment_digests: dict[str, str],
) -> None:
    monkeypatch.setattr(
        boundary,
        "frozen_writer_boundary_config",
        lambda: SimpleNamespace(enabled=True),
    )
    runtime = _runtime_identity()
    runtime["effective_environment_variable_names"] = environment_names
    runtime["effective_environment_variable_value_sha256"] = environment_digests

    with pytest.raises(RuntimeError, match="environment attestation"):
        attest_canonical_writer_startup_readiness(
            receipt_path=tmp_path / "readiness.json",
            _writer_call=lambda *_args: pytest.fail("PING must not run"),
            _process_hardening_provider=lambda: (False, 0, 0),
            _python_runtime_provider=lambda: runtime,
        )


def test_liveness_generations_replace_stale_receipt_atomically_after_ping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/tmp") / f"cw-live-{uuid.uuid4().hex}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    socket_path.chmod(0o660)
    receipt_path = tmp_path / "canonical-writer-liveness.json"
    receipt_path.write_text('{"stale":true}\n', encoding="utf-8")
    request_ids = iter((str(uuid.uuid4()), str(uuid.uuid4())))
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        boundary,
        "frozen_writer_boundary_config",
        lambda: SimpleNamespace(enabled=True, socket_path=socket_path),
    )

    def writer_call(operation, payload):
        assert not receipt_path.exists()
        calls.append((operation, payload))
        return _valid_ping(next(request_ids))

    try:
        first = attest_canonical_writer_liveness(
            1,
            receipt_path=receipt_path,
            _writer_call=writer_call,
            _now_unix=lambda: 1_800_000_001,
            _pid=4242,
            _boot_identity_provider=lambda: ("b" * 64, 1_000_000_001),
            _process_start_time=lambda pid: 123456 if pid == 4242 else 0,
        )
        second = attest_canonical_writer_liveness(
            2,
            receipt_path=receipt_path,
            _writer_call=writer_call,
            _now_unix=lambda: 1_800_000_002,
            _pid=4242,
            _boot_identity_provider=lambda: ("b" * 64, 2_000_000_002),
            _process_start_time=lambda pid: 123456 if pid == 4242 else 0,
        )
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)

    socket_identity = first
    assert first["version"] == WRITER_LIVENESS_RECEIPT_VERSION
    assert first["generation"] == 1
    assert second["generation"] == 2
    assert second["gateway_pid"] == 4242
    assert second["gateway_start_time_ticks"] == 123456
    assert second["boot_id_sha256"] == "b" * 64
    assert second["observed_at_boottime_ns"] == 2_000_000_002
    assert second["socket_path"] == str(socket_path)
    assert second["socket_device"] == socket_identity["socket_device"]
    assert second["socket_inode"] == socket_identity["socket_inode"]
    assert second["socket_owner_uid"] == socket_identity["socket_owner_uid"]
    assert second["socket_group_gid"] == socket_identity["socket_group_gid"]
    assert second["socket_mode"] == "0660"
    assert calls == [
        (CanonicalWriterOperation.PING, {}),
        (CanonicalWriterOperation.PING, {}),
    ]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == second
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert list(tmp_path.glob(f".{receipt_path.name}.*.tmp")) == []


def test_liveness_ping_failure_removes_stale_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "canonical-writer-liveness.json"
    receipt_path.write_text('{"generation":7}\n', encoding="utf-8")
    socket_identity = {
        "socket_path": "/run/muncho-canonical-writer/writer.sock",
        "socket_device": 7,
        "socket_inode": 99,
        "socket_owner_uid": 1002,
        "socket_group_gid": 2001,
        "socket_mode": "0660",
    }
    monkeypatch.setattr(
        boundary,
        "frozen_writer_boundary_config",
        lambda: SimpleNamespace(
            enabled=True,
            socket_path=Path(socket_identity["socket_path"]),
        ),
    )

    with pytest.raises(RuntimeError, match="writer unavailable"):
        attest_canonical_writer_liveness(
            8,
            receipt_path=receipt_path,
            _writer_call=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("writer unavailable")
            ),
            _socket_identity_provider=lambda _path: socket_identity,
        )

    assert not receipt_path.exists()


@pytest.mark.parametrize("generation", [0, -1, True, 2**63])
def test_liveness_generation_must_be_strictly_positive_and_bounded(
    generation: int,
) -> None:
    with pytest.raises(ValueError, match="generation is invalid"):
        attest_canonical_writer_liveness(generation)
