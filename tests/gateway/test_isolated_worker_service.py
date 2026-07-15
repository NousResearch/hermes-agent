from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest

import gateway.isolated_worker_service as service_module
from gateway.isolated_worker import PROTOCOL, canonical_bytes
from gateway.isolated_worker_service import (
    CONFIG_SCHEMA,
    ServiceConfigError,
    activated_listener,
    load_service_config,
    parse_service_config,
)


def _config_value(tmp_path: Path, listener_path: Path) -> dict:
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700, exist_ok=True)
    os.chmod(lease_base, 0o700)
    lease_state = os.lstat(lease_base)
    executable = Path("/bin/bash")
    executable_state = os.lstat(executable)
    executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    unsigned = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "listener_path": str(listener_path),
        "expected_peer_uid": os.getuid(),
        "expected_peer_gid": os.getgid(),
        "socket_uid": os.getuid(),
        "socket_gid": os.getgid(),
        "lease_base": str(lease_base),
        # The official runner places pytest's temp tree below /private/tmp on
        # macOS. That parent is setgid there, so a newly-created directory can
        # legitimately inherit a group other than the process primary group.
        # Seal the fixture to the metadata that is actually on disk, exactly as
        # a deployment-generated worker config does.
        "lease_uid": lease_state.st_uid,
        "lease_gid": lease_state.st_gid,
        "network_isolated": True,
        "bwrap": {
            "path": str(executable),
            "sha256": executable_digest,
            "uid": executable_state.st_uid,
        },
        "shell": {
            "path": str(executable),
            "sha256": executable_digest,
            "uid": executable_state.st_uid,
        },
        "limits": {
            "maximum_timeout_seconds": 300,
            "maximum_output_bytes": 1024 * 1024,
            "maximum_active_leases": 8,
            "maximum_active_jobs_per_lease": 4,
            "lease_ttl_seconds": 900,
            "lease_quota_bytes": 4096,
            "lease_quota_entries": 100,
            "global_quota_bytes": 8192,
            "global_quota_entries": 200,
        },
        "read_only_binds": [],
    }
    return {
        **unsigned,
        "config_sha256": hashlib.sha256(canonical_bytes(unsigned)).hexdigest(),
    }


def test_service_config_is_exact_canonical_self_digested_and_sealed(
    tmp_path: Path,
) -> None:
    value = _config_value(tmp_path, tmp_path / "worker.sock")
    payload = canonical_bytes(value)
    config = parse_service_config(payload)
    assert config.listener_path == tmp_path / "worker.sock"
    assert config.policy.maximum_active_jobs_per_lease == 4
    assert config.policy.global_quota_bytes == 8192
    assert config.policy.global_quota_entries == 200

    bad_digest = {**value, "config_sha256": "0" * 64}
    with pytest.raises(ServiceConfigError, match="config_digest_mismatch"):
        parse_service_config(canonical_bytes(bad_digest))

    with pytest.raises(ServiceConfigError, match="config_not_canonical"):
        parse_service_config(json.dumps(value, sort_keys=True).encode("ascii"))

    with pytest.raises(ServiceConfigError, match="config_fields_not_exact"):
        parse_service_config(canonical_bytes({**value, "policy_override": True}))

    legacy_unsigned = dict(value)
    legacy_unsigned.pop("config_sha256")
    legacy_limits = dict(legacy_unsigned["limits"])
    legacy_limits.pop("global_quota_bytes")
    legacy_limits.pop("global_quota_entries")
    legacy_unsigned["limits"] = legacy_limits
    legacy = {
        **legacy_unsigned,
        "config_sha256": hashlib.sha256(
            canonical_bytes(legacy_unsigned)
        ).hexdigest(),
    }
    with pytest.raises(ServiceConfigError, match="limits_fields_not_exact"):
        parse_service_config(canonical_bytes(legacy))

    old_schema_unsigned = {**legacy_unsigned, "schema": "muncho.isolated-worker.config.v1"}
    old_schema_unsigned["limits"] = dict(value["limits"])
    old_schema = {
        **old_schema_unsigned,
        "config_sha256": hashlib.sha256(
            canonical_bytes(old_schema_unsigned)
        ).hexdigest(),
    }
    with pytest.raises(ServiceConfigError, match="config_identity_invalid"):
        parse_service_config(canonical_bytes(old_schema))

    path = tmp_path / "isolated-worker.json"
    path.write_bytes(payload)
    path.chmod(0o440)
    path_state = os.lstat(path)
    loaded = load_service_config(
        path,
        expected_owner_uid=path_state.st_uid,
        expected_owner_gid=path_state.st_gid,
    )
    assert loaded == config

    with pytest.raises(ServiceConfigError, match="config_file_not_sealed"):
        load_service_config(
            path,
            expected_owner_uid=path_state.st_uid,
            expected_owner_gid=path_state.st_gid + 1,
        )

    path.chmod(0o400)
    with pytest.raises(ServiceConfigError, match="config_file_not_sealed"):
        load_service_config(
            path,
            expected_owner_uid=path_state.st_uid,
            expected_owner_gid=path_state.st_gid,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="systemd socket activation is Linux-only")
def test_socket_activation_requires_exact_fd_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    socket_root = Path(tempfile.mkdtemp(prefix="iws-", dir="/tmp"))
    socket_path = socket_root / "worker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(4)
    descriptor = listener.detach()
    monkeypatch.setattr(service_module, "SOCKET_ACTIVATION_FD", descriptor)
    config = parse_service_config(canonical_bytes(_config_value(tmp_path, socket_path)))
    environment = {
        "LISTEN_PID": "1234",
        "LISTEN_FDS": "1",
        "LISTEN_FDNAMES": "isolated-worker",
    }

    activated = activated_listener(
        config,
        environment=environment,
        process_id=1234,
    )
    try:
        assert Path(activated.getsockname()) == socket_path
        assert activated.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        activated.close()

    with pytest.raises(ServiceConfigError, match="socket_activation_identity_invalid"):
        activated_listener(
            config,
            environment={**environment, "LISTEN_FDS": "2"},
            process_id=1234,
        )
    without_fd_name = dict(environment)
    without_fd_name.pop("LISTEN_FDNAMES")
    with pytest.raises(ServiceConfigError, match="socket_activation_identity_invalid"):
        activated_listener(
            config,
            environment=without_fd_name,
            process_id=1234,
        )
    shutil.rmtree(socket_root, ignore_errors=True)
