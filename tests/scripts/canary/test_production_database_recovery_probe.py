from __future__ import annotations

import base64
import io
import json
import os
import struct
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from gateway import canonical_writer_production_cutover as cutover
from scripts.canary import production_database_recovery_gate as owner
from scripts.canary import production_database_recovery_probe as remote


REVISION = "a" * 40
NOW = 1_800_000_000
PASSWORD = bytearray(b"fixed-S3cr:et\\value")
CA_PEM = "-----BEGIN CERTIFICATE-----\nYQ==\n-----END CERTIFICATE-----\n"


def _preflight() -> Mapping[str, Any]:
    unsigned = {
        "schema": remote.PREFLIGHT_SCHEMA,
        "ok": True,
        "release_revision": REVISION,
        "canary_instance_id": "9153645328899914617",
        "canary_host_identity_sha256": "1" * 64,
        "canary_network": remote.EXPECTED_NETWORK,
        "canary_subnetwork": remote.EXPECTED_SUBNETWORK,
        "canary_private_ip": remote.EXPECTED_PRIVATE_IP,
        "network_identity_sha256": "2" * 64,
        "release_manifest_file_sha256": "3" * 64,
        "stopped_units_sha256": "4" * 64,
        "psql_executable": str(remote.PSQL),
        "psql_executable_sha256": remote.EXPECTED_PSQL_SHA256,
        "database": cutover.DATABASE,
        "probe_contract_sha256": owner._sha_json(
            cutover.DATABASE_RECOVERY_PROBE_CONTRACT
        ),
        "accepts_caller_target": False,
        "accepts_caller_sql": False,
        "accepts_caller_command": False,
    }
    return {**unsigned, "preflight_sha256": owner._sha_json(unsigned)}


def _scratch() -> Mapping[str, Any]:
    return {
        "project": cutover.PROJECT,
        "instance": cutover.database_recovery_scratch_instance(REVISION),
        "region": cutover.PRODUCTION_SQL_REGION,
        "private_network": cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
        "database_version": "POSTGRES_18",
        "configuration_sha256": "6" * 64,
        "readback_sha256": "7" * 64,
        "create_operation_id": "create-op",
        "restore_operation_id": "restore-op",
        "restored_backup_id": "123",
        "private_ip": "10.23.45.67",
        "server_ca_pem": CA_PEM,
        "server_ca_sha256": owner._sha(CA_PEM.encode("ascii")),
        "ssl_mode": "ENCRYPTED_ONLY",
        "server_ca_mode": "GOOGLE_MANAGED_INTERNAL_CA",
        "connection_name": (
            f"{cutover.PROJECT}:{cutover.PRODUCTION_SQL_REGION}:"
            f"{cutover.database_recovery_scratch_instance(REVISION)}"
        ),
    }


def _psql_output(*, read_only: str = "on", rows: int = 14) -> bytes:
    required_columns = (
        "event_id",
        "schema_version",
        "event_type",
        "occurred_at",
        "case_id",
        "source",
        "actor",
        "subject",
        "evidence",
        "decision",
        "status",
        "next_action",
        "safety",
        "payload",
    )
    schema = {
        "columns": [
            {
                "schema_name": "public",
                "relation_name": "canonical_event_log",
                "relation_kind": "r",
                "ordinal_position": index,
                "column_name": name,
                "data_type": "text",
                "not_null": False,
            }
            for index, name in enumerate(required_columns, start=1)
        ],
        "column_count": len(required_columns),
    }
    sample_count = min(rows, 64)

    def sample(index: int) -> Mapping[str, str]:
        return {
            "event_id": str(uuid.UUID(int=index)),
            "envelope_md5": f"{index:032x}"[-32:],
        }

    content = {
        "row_count": rows,
        "minimum_event_id": str(uuid.UUID(int=1)) if rows else None,
        "maximum_event_id": str(uuid.UUID(int=rows)) if rows else None,
        "minimum_occurred_at": "2026-01-01 00:00:00+00" if rows else None,
        "maximum_occurred_at": "2026-01-02 00:00:00+00" if rows else None,
        "head": [sample(index) for index in range(1, sample_count + 1)],
        "tail": [
            sample(index)
            for index in range(rows - sample_count + 1, rows + 1)
        ],
    }
    return (
        read_only.encode("ascii")
        + b"\n"
        + json.dumps(schema, separators=(",", ":")).encode("ascii")
        + b"\n"
        + json.dumps(content, separators=(",", ":")).encode("ascii")
        + b"\n"
    )


def _test_anonymous_descriptor(_label: str) -> int:
    descriptor, path = tempfile.mkstemp(prefix="muncho-recovery-test-")
    os.unlink(path)
    os.fchmod(descriptor, 0o600)
    observed = os.fstat(descriptor)
    assert observed.st_nlink == 0
    return descriptor


def _psql_environment(
    password_fd: int = 10,
    ca_fd: int = 11,
) -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PGCONNECT_TIMEOUT": "15",
        "PGDATABASE": cutover.DATABASE,
        "PGHOSTADDR": "10.23.45.67",
        "PGOPTIONS": (
            "-c default_transaction_read_only=on "
            "-c statement_timeout=60000 -c lock_timeout=5000"
        ),
        "PGPASSFILE": f"/proc/self/fd/{password_fd}",
        "PGPORT": remote.DATABASE_PORT,
        "PGSSLMODE": remote.TLS_MODE,
        "PGSSLROOTCERT": f"/proc/self/fd/{ca_fd}",
        "PGUSER": remote.DATABASE_USER,
    }


def _frame(gate: Mapping[str, Any], password: bytearray | None = None) -> bytearray:
    return owner.FixedPrivateReadOnlyProbe._frame(
        revision=REVISION,
        gate=gate,
        scratch=_scratch(),
        secret_version="7",
        password=PASSWORD if password is None else password,
    )


def test_remote_probe_keeps_secret_out_of_argv_environment_output_and_disk(
    tmp_path: Path,
) -> None:
    preflight = _preflight()
    gate = remote.build_gate(preflight, now_unix=NOW)
    frame = _frame(gate)
    seen: dict[str, Any] = {}

    def execute(
        argv: Sequence[str], environment: Mapping[str, str], pass_fds: Sequence[int]
    ) -> bytes:
        seen["argv"] = tuple(argv)
        seen["environment"] = dict(environment)
        seen["fds"] = tuple(pass_fds)
        pgpass = os.read(pass_fds[0], 16 * 1024)
        os.lseek(pass_fds[0], 0, os.SEEK_SET)
        ca = os.read(pass_fds[1], 16 * 1024)
        assert pgpass.endswith(b"fixed-S3cr\\:et\\\\value\n")
        assert ca == CA_PEM.encode("ascii")
        public_process_material = "\0".join(argv) + "\0" + "\0".join(
            f"{key}={value}" for key, value in environment.items()
        )
        assert bytes(PASSWORD) not in public_process_material.encode("utf-8")
        assert environment["PGDATABASE"] == cutover.DATABASE
        assert environment["PGUSER"] == remote.DATABASE_USER
        assert environment["PGSSLMODE"] == "verify-ca"
        assert environment["PGPASSFILE"].startswith("/proc/self/fd/")
        assert environment["PGSSLROOTCERT"].startswith("/proc/self/fd/")
        return _psql_output()

    receipt = remote.run_probe_frame(
        io.BytesIO(bytes(frame)),
        revision=REVISION,
        preflight=preflight,
        gate=gate,
        clock=lambda: float(NOW + 1),
        psql_executor=execute,
        descriptor_factory=_test_anonymous_descriptor,
    )

    assert receipt["transaction_read_only"] is True
    assert receipt["tls_mode"] == "verify-ca"
    assert receipt["tls_ca_verified"] is True
    assert receipt["tls_hostname_verified"] is False
    assert receipt["scratch_private_ip"] == _scratch()["private_ip"]
    assert receipt["server_ca_sha256"] == _scratch()["server_ca_sha256"]
    assert bytes(PASSWORD) not in owner._canonical(receipt)
    for descriptor in seen["fds"]:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not any(
        bytes(PASSWORD) in path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    if Path("/proc/self/cmdline").is_file():
        assert bytes(PASSWORD) not in Path("/proc/self/cmdline").read_bytes()
        assert bytes(PASSWORD) not in Path("/proc/self/environ").read_bytes()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda frame: frame[:-1],
        lambda frame: frame + b"x",
        lambda frame: b"BAD!" + frame[4:],
        lambda frame: frame[:8] + struct.pack(">I", remote.MAX_PASSWORD_BYTES + 1) + frame[12:],
    ),
)
def test_partial_trailing_and_malformed_secret_frames_fail_closed(mutator: Any) -> None:
    preflight = _preflight()
    gate = remote.build_gate(preflight, now_unix=NOW)
    with pytest.raises(remote.RecoveryProbeError):
        remote.run_probe_frame(
            io.BytesIO(mutator(bytes(_frame(gate)))),
            revision=REVISION,
            preflight=preflight,
            gate=gate,
            clock=lambda: float(NOW + 1),
            psql_executor=lambda *_args: _psql_output(),
            descriptor_factory=_test_anonymous_descriptor,
        )


def test_writable_extra_and_oversized_psql_results_fail_closed() -> None:
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(_psql_output(read_only="off"))
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(_psql_output() + b"extra\n")
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(b"x" * (remote.MAX_PSQL_OUTPUT_BYTES + 1))


def test_psql_schema_and_sample_semantics_fail_closed() -> None:
    lines = _psql_output().decode("ascii").splitlines()
    schema = json.loads(lines[1])
    content = json.loads(lines[2])

    def render() -> bytes:
        return (
            b"on\n"
            + owner._canonical(schema)
            + b"\n"
            + owner._canonical(content)
            + b"\n"
        )

    removed = schema["columns"].pop()
    schema["column_count"] -= 1
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(render())
    schema["columns"].append(removed)
    schema["column_count"] += 1

    content["head"][1] = dict(content["head"][0])
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(render())
    content["head"][1] = {
        "event_id": str(uuid.UUID(int=2)),
        "envelope_md5": "2".zfill(32),
    }

    content["tail"].pop()
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(render())
    content["tail"].append(
        {
            "event_id": str(uuid.UUID(int=14)),
            "envelope_md5": "e".zfill(32),
        }
    )

    content["minimum_occurred_at"] = "2026-01-01 00:00:00"
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(render())

    empty_lines = _psql_output(rows=0).decode("ascii").splitlines()
    empty_content = json.loads(empty_lines[2])
    empty_content["minimum_event_id"] = str(uuid.UUID(int=1))
    malformed_empty = (
        b"on\n"
        + empty_lines[1].encode("ascii")
        + b"\n"
        + owner._canonical(empty_content)
        + b"\n"
    )
    with pytest.raises(remote.RecoveryProbeError, match="psql_output_invalid"):
        remote._parse_psql_output(malformed_empty)


def test_provider_ca_and_private_ip_drift_fail_before_secret_access() -> None:
    class NoSecret:
        calls = 0

        def access(self) -> tuple[str, bytearray]:
            self.calls += 1
            raise AssertionError("secret must not be fetched")

    class Transport:
        def require_available(self, release_revision: str) -> Mapping[str, Any]:
            assert release_revision == REVISION
            return _preflight()

        def open_probe(self, release_revision: str) -> Any:
            raise AssertionError("remote probe must not open")

    accessor = NoSecret()
    probe = owner.FixedPrivateReadOnlyProbe(
        REVISION,
        transport=Transport(),
        secret_accessor=accessor,
        clock=lambda: float(NOW),
    )
    probe.require_available()
    for name, replacement in (
        ("private_ip", "203.0.113.7"),
        ("server_ca_sha256", "f" * 64),
    ):
        scratch = dict(_scratch())
        scratch[name] = replacement
        with pytest.raises(owner.ProductionDatabaseRecoveryError):
            probe.probe(
                release_revision=REVISION,
                scratch=scratch,
                now_unix=NOW,
            )
    assert accessor.calls == 0


class _Accessor:
    def __init__(self) -> None:
        self.value = bytearray(PASSWORD)

    def access(self) -> tuple[str, bytearray]:
        return "7", self.value


class _Session:
    def __init__(self, gate: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
        self._gate = gate
        self._preflight = preflight
        self.termination_proven = False
        self.frame_reference: bytearray | None = None
        self.validated: Mapping[str, Any] | None = None

    def read_gate(self) -> Mapping[str, Any]:
        return self._gate

    def finish(self, frame: bytearray) -> Mapping[str, Any]:
        self.frame_reference = frame
        receipt = remote.run_probe_frame(
            io.BytesIO(bytes(frame)),
            revision=REVISION,
            preflight=self._preflight,
            gate=self._gate,
            clock=lambda: float(NOW + 1),
            psql_executor=lambda *_args: _psql_output(),
            descriptor_factory=_test_anonymous_descriptor,
        )
        return receipt

    def mark_validated(self, receipt: Mapping[str, Any]) -> None:
        self.validated = receipt
        self.termination_proven = True

    def abort_and_prove_terminated(self) -> None:
        self.termination_proven = True

    def close(self) -> None:
        if not self.termination_proven:
            raise RuntimeError("unproven termination")


class _Transport:
    def __init__(self) -> None:
        self.preflight = _preflight()
        self.session: _Session | None = None

    def require_available(self, release_revision: str) -> Mapping[str, Any]:
        assert release_revision == REVISION
        return self.preflight

    def open_probe(self, release_revision: str) -> _Session:
        assert release_revision == REVISION
        gate = remote.build_gate(self.preflight, now_unix=NOW)
        self.session = _Session(gate, self.preflight)
        return self.session


def test_injected_owner_to_remote_transport_is_release_bound_and_wipes_parent_buffers() -> None:
    transport = _Transport()
    accessor = _Accessor()
    probe = owner.FixedPrivateReadOnlyProbe(
        REVISION,
        transport=transport,
        secret_accessor=accessor,
        clock=lambda: float(NOW + 1),
    )
    probe.require_available()
    receipt = probe.probe(
        release_revision=REVISION,
        scratch=_scratch(),
        now_unix=NOW + 1,
    )
    assert receipt["ok"] is True
    assert transport.session is not None
    assert transport.session.validated == receipt
    assert transport.session.frame_reference is not None
    assert set(transport.session.frame_reference) == {0}
    assert set(accessor.value) == {0}


def test_secret_manager_access_is_fixed_crc_checked_and_caller_has_to_wipe() -> None:
    secret = bytearray(PASSWORD)
    body = owner._canonical({
        "name": (
            "projects/adventico-ai-platform/secrets/"
            "ai-platform-db-password/versions/7"
        ),
        "payload": {
            "data": base64.b64encode(secret).decode("ascii"),
            "dataCrc32c": str(owner._crc32c(secret)),
        },
    })
    calls: list[tuple[Any, ...]] = []

    def request(*args: Any) -> Any:
        calls.append(args)
        return SimpleNamespace(status=200, body=body)

    accessor = owner.FixedSecretManagerAccess(
        lambda: "owner-access-token",
        requester=request,
    )
    version, observed = accessor.access()
    assert version == "7"
    assert observed == secret
    assert calls[0][0] == "GET"
    assert calls[0][1] == owner.FixedSecretManagerAccess._URL
    assert calls[0][2] == {}
    assert bytes(secret) not in owner._canonical(calls[0][2])
    owner._zeroize(observed)
    assert set(observed) == {0}

    bad = json.loads(body)
    bad["payload"]["dataCrc32c"] = "0"
    bad_accessor = owner.FixedSecretManagerAccess(
        lambda: "owner-access-token",
        requester=lambda *_args: SimpleNamespace(
            status=200, body=owner._canonical(bad)
        ),
    )
    with pytest.raises(
        owner.ProductionDatabaseRecoveryError,
        match="production_database_recovery_secret_shape_invalid",
    ):
        bad_accessor.access()

    failed_headers: list[dict[str, str]] = []

    def failed_request(*args: Any) -> Any:
        failed_headers.append(args[2])
        raise RuntimeError("request failure containing sensitive headers")

    failed_accessor = owner.FixedSecretManagerAccess(
        lambda: "owner-access-token",
        requester=failed_request,
    )
    with pytest.raises(
        owner.ProductionDatabaseRecoveryError,
        match="^production_database_recovery_secret_access_failed$",
    ) as failure:
        failed_accessor.access()
    assert failure.value.__suppress_context__ is True
    assert failed_headers == [{}]


def test_remote_preflight_binds_exact_vm_network_release_and_psql() -> None:
    host = {
        "project_id": "adventico-ai-platform",
        "zone": "europe-west3-a",
        "instance_name": "muncho-canary-v2-01",
        "instance_id": "9153645328899914617",
        "host_identity_sha256": "1" * 64,
    }
    metadata = {
        path: value
        for path, value in zip(
            remote._NETWORK_METADATA_PATHS.values(),
            (
                "projects/39589465056/networks/muncho-canary-vpc",
                (
                    "projects/39589465056/regions/europe-west3/subnetworks/"
                    "muncho-canary-europe-west3"
                ),
                "10.90.0.2",
            ),
            strict=True,
        )
    }
    receipt = remote.collect_preflight(
        REVISION,
        host_observer=lambda: host,
        network_reader=lambda path: metadata[path],
        release_observer=lambda _revision: {
            "release_manifest_file_sha256": "3" * 64,
            "stopped_units_sha256": "4" * 64,
        },
        psql_hasher=lambda _path: remote.EXPECTED_PSQL_SHA256,
        descriptor_factory=_test_anonymous_descriptor,
    )
    assert owner._validate_remote_preflight(
        receipt, release_revision=REVISION
    ) == receipt
    drift = dict(metadata)
    drift[next(iter(remote._NETWORK_METADATA_PATHS.values()))] = (
        "projects/39589465056/networks/default"
    )
    with pytest.raises(remote.RecoveryProbeError, match="network_identity_invalid"):
        remote.collect_preflight(
            REVISION,
            host_observer=lambda: host,
            network_reader=lambda path: drift[path],
            release_observer=lambda _revision: {
                "release_manifest_file_sha256": "3" * 64,
                "stopped_units_sha256": "4" * 64,
            },
            psql_hasher=lambda _path: remote.EXPECTED_PSQL_SHA256,
            descriptor_factory=_test_anonymous_descriptor,
        )

    def unavailable_descriptor(_label: str) -> int:
        raise OSError("memfd unavailable")

    with pytest.raises(
        remote.RecoveryProbeError,
        match="anonymous_secret_unavailable",
    ):
        remote.collect_preflight(
            REVISION,
            host_observer=lambda: host,
            network_reader=lambda path: metadata[path],
            release_observer=lambda _revision: {
                "release_manifest_file_sha256": "3" * 64,
                "stopped_units_sha256": "4" * 64,
            },
            psql_hasher=lambda _path: remote.EXPECTED_PSQL_SHA256,
            descriptor_factory=unavailable_descriptor,
        )


@pytest.mark.parametrize(
    "wrapper",
    (
        Path("/usr/bin/psql"),
        Path("/usr/share/postgresql-common/pg_wrapper"),
    ),
)
def test_psql_identity_rejects_wrapper_and_alternative_paths(wrapper: Path) -> None:
    with pytest.raises(remote.RecoveryProbeError, match="psql_identity_invalid"):
        remote._root_executable_sha256(wrapper)
    assert remote.PSQL == Path("/usr/lib/postgresql/15/bin/psql")


def test_psql_contract_suppresses_stderr_and_never_places_secret_in_process_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    read_fd, write_fd = os.pipe()
    os.write(write_fd, _psql_output())
    os.close(write_fd)

    class Process:
        stdout = os.fdopen(read_fd, "rb", buffering=0)
        pid = os.getpid()

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(_timeout: float) -> int:
            return 0

    def popen(*args: Any, **kwargs: Any) -> Process:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(remote.subprocess, "Popen", popen)
    environment = _psql_environment()
    raw = remote._bounded_psql(remote._PSQL_ARGV, environment, (10, 11))
    assert raw == _psql_output()
    assert captured["kwargs"]["stderr"] == remote.subprocess.DEVNULL
    assert captured["kwargs"]["stdin"] == remote.subprocess.DEVNULL
    assert captured["kwargs"]["shell"] is False
    assert bytes(PASSWORD) not in repr(captured).encode("utf-8")

    for field, invalid in (
        ("PGDATABASE", "wrong_database"),
        ("PGUSER", "wrong_user"),
        ("PGHOSTADDR", "203.0.113.7"),
    ):
        drifted = _psql_environment()
        drifted[field] = invalid
        with pytest.raises(remote.RecoveryProbeError, match="psql_contract_invalid"):
            remote._bounded_psql(remote._PSQL_ARGV, drifted, (10, 11))


def test_bounded_psql_timeout_kills_the_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {"wait": []}

    class Output:
        @staticmethod
        def fileno() -> int:
            return 123

    class Process:
        stdout = Output()
        pid = 4567

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(timeout: float) -> int:
            observed["wait"].append(timeout)
            return -remote.signal.SIGKILL

    class Selector:
        def register(self, descriptor: int, event: int) -> None:
            observed["registered"] = (descriptor, event)

        @staticmethod
        def select(_timeout: float) -> list[Any]:
            raise AssertionError("expired deadline must not poll")

        def close(self) -> None:
            observed["selector_closed"] = True

    ticks = iter((100.0, 191.0))
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(remote.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(remote.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        remote.os,
        "killpg",
        lambda pid, signal_number: observed.update(
            {"killpg": (pid, signal_number)}
        ),
    )

    with pytest.raises(remote.RecoveryProbeError, match="psql_timeout"):
        remote._bounded_psql(remote._PSQL_ARGV, _psql_environment(), (10, 11))

    assert observed["registered"] == (123, remote.selectors.EVENT_READ)
    assert observed["killpg"] == (4567, remote.signal.SIGKILL)
    assert observed["wait"] == [5.0]
    assert observed["selector_closed"] is True
