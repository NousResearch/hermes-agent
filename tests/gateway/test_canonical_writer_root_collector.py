from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
import socket
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_bootstrap as bootstrap
from gateway import canonical_writer_root_collector as collector
from gateway.canonical_writer_db import (
    ManagedCloudSQLAdminHBAReceipt,
)
from gateway.canonical_writer_deployment_preflight import (
    PreflightCheck,
    PreflightReport,
)


NOW = 2_000_000_000
REVISION = "1" * 40
ARTIFACT = "2" * 64
PLAN = "3" * 64
BOOT = "4" * 64
HOST = "5" * 64
SQL_PRIVATE_IP = "10.0.0.8"
SQL_TLS_SERVER_NAME = "db.internal"


def _hba_receipt(
    *,
    observed_at: int,
    certificate: str = "d" * 64,
    tls_server_name: str = SQL_TLS_SERVER_NAME,
    ttl_seconds: int = 30,
):
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=SQL_PRIVATE_IP,
        tls_server_name=tls_server_name,
        port=5432,
        server_certificate_sha256=certificate,
        database="cloudsqladmin",
        user="canonical_writer",
        observed_at_unix=observed_at,
        expires_at_unix=observed_at + ttl_seconds,
        sqlstate="28000",
        server_message=(
            'no pg_hba.conf entry for host "10.0.0.8", user '
            '"canonical_writer", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _snapshot():
    baseline = _hba_receipt(observed_at=NOW - 300)
    return {
        "deployment_mode": "writer_only",
        "gateway_uid": 1001,
        "gateway_gid": 2000,
        "writer_uid": 1002,
        "writer_gid": 2002,
        "projector_gid": 2003,
        "gateway_supplementary_gids": [2000, 2001],
        "writer_supplementary_gids": [2002, 2003],
        "socket": {
            "expected_group_gid": 2001,
            "device": 7,
            "inode": 99,
            "runtime_directory_device": 8,
            "runtime_directory_inode": 100,
        },
        "writer_kernel_network_enforcement": {
            "ip_address_deny": ["0.0.0.0/0", "::/0"],
            "ip_address_allow": ["10.0.0.8/32"],
            "cgroup_device": 9,
            "cgroup_inode": 101,
            "main_pid": 4343,
            "ingress_direct_program_ids": [11],
            "ingress_effective_program_ids": [11, 12],
            "egress_direct_program_ids": [13],
            "egress_effective_program_ids": [13, 14],
        },
        "gateway_process": {
            "systemd_main_pid": 4242,
            "systemd_main_pid_start_time_ticks": 123456,
        },
        "writer_deployment": {
            "policy": {
                "revision": REVISION,
                "artifact_digest_sha256": ARTIFACT,
                "artifact_root": f"/opt/releases/{REVISION}",
                "module_origin": (
                    f"/opt/releases/{REVISION}/venv/lib/python3.12/"
                    "site-packages/gateway/canonical_writer_bootstrap.py"
                ),
                "config_path": "/etc/muncho-canonical-writer/writer.json",
                "preapproved_external_native_executable_mappings": [
                    {"path": "/usr/lib/libc.so.6", "sha256": "c" * 64}
                ],
            },
            "attestation": {
                "process": {
                    "systemd_main_pid": 4343,
                    "systemd_main_pid_start_time_ticks": 654321,
                }
            },
        },
        "gateway_deployment": {
            "policy": {
                "revision": REVISION,
                "artifact_digest_sha256": ARTIFACT,
                "artifact_root": f"/opt/releases/{REVISION}",
                "module": "gateway.canonical_writer_gateway_bootstrap",
                "module_origin": (
                    f"/opt/releases/{REVISION}/venv/lib/python3.12/"
                    "site-packages/gateway/"
                    "canonical_writer_gateway_bootstrap.py"
                ),
                "read_write_paths": ["/run/hermes-cloud-gateway"],
                "preapproved_external_native_executable_mappings": [
                    {"path": "/usr/lib/libc.so.6", "sha256": "c" * 64}
                ],
            }
        },
        "writer_authority_surface": {
            "projection_exporter": {"policy": {"enabled": False}}
        },
        "database": {
            "expected_user": "canonical_writer",
            "connection": {
                "host": SQL_PRIVATE_IP,
                "tls_server_name": SQL_TLS_SERVER_NAME,
                "port": 5432,
                "database": "canonical",
                "user": "canonical_writer",
            },
            "policy": {
                "private_schema_identity_sha256": "e" * 64,
                "managed_cloudsqladmin_hba_rejection_receipt": (
                    baseline.as_dict()
                ),
                "managed_cloudsqladmin_hba_rejection_sha256": baseline.sha256,
            },
            "attestation": {
                "managed_cloudsqladmin_hba_rejection_sha256": baseline.sha256
            },
            "managed_cloudsqladmin_hba_rejection_evidence": {
                "complete": True,
                "collector_uid": 0,
                "source_owner_uid": 0,
                "source_mode": "0400",
                "source_symlink": False,
                "same_host": True,
                "same_tls_server_name": True,
                "same_port": True,
                "same_ca": True,
                "same_user": True,
                "same_credential": True,
                "receipt_sha256": baseline.sha256,
                "receipt": baseline.as_dict(),
            },
        },
        "discord_edge": {
            "gateway_enabled": False,
            "writer_authority_enabled": False,
            "unit_name": "muncho-discord-egress.service",
            "config_path": "/etc/muncho/discord-edge.json",
            "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
            "socket_path": "/run/muncho-discord-egress/edge.sock",
        },
    }


def _manifest_for(snapshot):
    return collector.TrustedDeploymentManifest.from_mapping(
        {
            "schema": collector.MANIFEST_SCHEMA,
            "mode": collector.WRITER_ONLY_MODE,
            "revision": REVISION,
            "artifact_sha256": ARTIFACT,
            "snapshot_policy_sha256": collector.snapshot_policy_sha256(
                snapshot
            ),
            "host_contract": {
                "gateway_unit_fragment_path": "/etc/systemd/system/hermes-cloud-gateway.service",
                "gateway_unit_fragment_sha256": "8" * 64,
                "writer_unit_fragment_path": "/etc/systemd/system/muncho-canonical-writer.service",
                "writer_unit_fragment_sha256": "9" * 64,
                "gateway_config_path": "/etc/hermes/config.yaml",
                "gateway_config_sha256": "a" * 64,
                "writer_config_path": "/etc/muncho-canonical-writer/writer.json",
                "writer_config_sha256": "b" * 64,
                "projection_export_path": "/var/lib/muncho-canonical-writer/projection/canonical-events.json",
                "external_iam_policy_sha256": "c" * 64,
                "external_iam_receipt_path": "/run/muncho-canonical-preflight/external-iam-receipt.json",
                "legacy_helper_path": str(
                    collector.LEGACY_CLOUD_SQL_HELPER_PATH
                ),
                "native_observation_plan_sha256": "f" * 64,
                "native_observation_receipt_path": (
                    f"/var/lib/muncho-writer-canary-evidence/{REVISION}/"
                    f"{'f' * 64}/native-observation.json"
                ),
                "native_observation_receipt_sha256": "e" * 64,
            },
            "snapshot_template": copy.deepcopy(snapshot),
        }
    )


def test_non_root_aborts_before_manifest_or_snapshot_access(monkeypatch):
    calls = []
    monkeypatch.setattr(collector, "_effective_uid", lambda: 1001)
    monkeypatch.setattr(
        collector,
        "_collect_live_snapshot",
        lambda *_args, **_kwargs: calls.append("snapshot"),
    )

    with pytest.raises(PermissionError, match="uid_0"):
        collector.collect_and_evaluate(
            "/must-not-be-read",
            "/must-not-be-written",
            activation_plan_sha256=PLAN,
        )

    assert calls == []


def test_systemd_collector_uses_only_absolute_bounded_argv(monkeypatch):
    observed = []
    stdout = "".join(
        f"{name}=\n" for name in collector._SYSTEMD_PROPERTIES
    )

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return collector.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(collector.subprocess, "run", run)

    result = collector._systemctl_show("hermes-cloud-gateway.service")

    assert set(result) == set(collector._SYSTEMD_PROPERTIES)
    assert observed[0][0][0] == "/usr/bin/systemctl"
    assert observed[0][0][-2:] == ["--", "hermes-cloud-gateway.service"]
    assert "shell" not in observed[0][1]
    assert observed[0][1]["timeout"] == 3
    assert observed[0][1]["env"] == {
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def test_systemd_collector_normalizes_systemd_252_empty_environment_files(
    monkeypatch,
):
    stdout = "".join(
        f"{name}=\n"
        for name in collector._SYSTEMD_PROPERTIES
        if name != "EnvironmentFiles"
    )
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, **_kwargs: collector.subprocess.CompletedProcess(
            command,
            0,
            stdout,
            "",
        ),
    )

    result = collector._systemctl_show("hermes-cloud-gateway.service")

    assert set(result) == set(collector._SYSTEMD_PROPERTIES)
    assert result["EnvironmentFiles"] == ""


@pytest.mark.parametrize(
    "missing",
    [
        {"MainPID"},
        {"EnvironmentFiles", "MainPID"},
    ],
)
def test_systemd_collector_rejects_every_other_missing_property(
    monkeypatch,
    missing,
):
    stdout = "".join(
        f"{name}=\n"
        for name in collector._SYSTEMD_PROPERTIES
        if name not in missing
    )
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, **_kwargs: collector.subprocess.CompletedProcess(
            command,
            0,
            stdout,
            "",
        ),
    )

    with pytest.raises(RuntimeError, match="systemd evidence is incomplete"):
        collector._systemctl_show("hermes-cloud-gateway.service")


@pytest.mark.parametrize("extra_line", ["UnknownProperty=\n", "MainPID=9999\n"])
def test_systemd_collector_rejects_unknown_or_duplicate_property(
    monkeypatch,
    extra_line,
):
    stdout = (
        "".join(f"{name}=\n" for name in collector._SYSTEMD_PROPERTIES)
        + extra_line
    )
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, **_kwargs: collector.subprocess.CompletedProcess(
            command,
            0,
            stdout,
            "",
        ),
    )

    with pytest.raises(RuntimeError, match="systemd evidence fields are invalid"):
        collector._systemctl_show("hermes-cloud-gateway.service")


def test_systemd_bind_read_only_parser_accepts_only_canonical_rbind_identity():
    release = f"/opt/releases/{REVISION}"

    assert collector._parse_bind_read_only_paths(
        f"{release}:{release}:rbind"
    ) == [release]


@pytest.mark.parametrize(
    "value",
    [
        f"/opt/releases/{REVISION}",
        f"/opt/releases/{REVISION}:/opt/releases/{REVISION}:ro",
        f"/opt/releases/{REVISION}:/opt/releases/other:rbind",
        "relative:relative:rbind",
        (
            f"/opt/releases/{REVISION}:/opt/releases/{REVISION}:rbind "
            f"/opt/releases/{REVISION}:/opt/releases/{REVISION}:rbind"
        ),
        f" /opt/releases/{REVISION}:/opt/releases/{REVISION}:rbind",
    ],
)
def test_systemd_bind_read_only_parser_rejects_malformed_or_drifted_values(value):
    with pytest.raises(RuntimeError, match="BindReadOnlyPaths"):
        collector._parse_bind_read_only_paths(value)


@pytest.mark.parametrize("value", ["any", "0.0.0.0/0 ::/0", "::/0 0.0.0.0/0"])
def test_systemd_ip_deny_parser_accepts_exact_universal_policy(value):
    assert collector._parse_universal_ip_deny(value) == (
        "0.0.0.0/0",
        "::/0",
    )


@pytest.mark.parametrize(
    "value",
    [
        "0.0.0.0/0",
        "::/0",
        "0.0.0.0/0 ::/1",
        "0.0.0.0/0 0.0.0.0/0 ::/0",
        " 0.0.0.0/0 ::/0",
        "0.0.0.1/0 ::/0",
    ],
)
def test_systemd_ip_deny_parser_rejects_partial_or_noncanonical_policy(value):
    with pytest.raises(RuntimeError):
        collector._parse_universal_ip_deny(value)


def test_systemd_ip_allow_parser_returns_one_canonical_database_network():
    assert collector._parse_systemd_ip_networks("10.0.0.8/32") == (
        "10.0.0.8/32",
    )
    with pytest.raises(RuntimeError):
        collector._parse_systemd_ip_networks("any")
    with pytest.raises(RuntimeError):
        collector._parse_systemd_ip_networks("10.0.0.8")


def _stat_value(
    mode,
    *,
    uid=1002,
    gid=2001,
    device=8,
    inode=100,
):
    return SimpleNamespace(
        st_mode=mode,
        st_uid=uid,
        st_gid=gid,
        st_dev=device,
        st_ino=inode,
    )


@pytest.mark.parametrize(
    "item",
    [
        _stat_value(stat.S_IFLNK | 0o777),
        _stat_value(stat.S_IFDIR | 0o2750, uid=1003),
        _stat_value(stat.S_IFDIR | 0o2750, gid=2002),
        _stat_value(stat.S_IFDIR | 0o0750),
    ],
)
def test_runtime_directory_rejects_symlink_owner_group_or_mode(item):
    with pytest.raises(RuntimeError, match="runtime directory identity"):
        collector._validate_runtime_directory_stat(
            item,
            writer_uid=1002,
            socket_gid=2001,
        )


def test_runtime_directory_rejects_posix_acl(monkeypatch):
    item = _stat_value(stat.S_IFDIR | 0o2750)
    monkeypatch.setattr(collector, "_validate_parent_chain", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collector.os, "lstat", lambda _path: item)
    monkeypatch.setattr(collector, "_has_posix_acl", lambda _path: True)

    with pytest.raises(RuntimeError, match="POSIX ACL"):
        collector._collect_writer_runtime_directory(
            writer_uid=1002,
            socket_gid=2001,
        )


@pytest.mark.parametrize(
    "item",
    [
        _stat_value(stat.S_IFLNK | 0o777, uid=0, gid=0),
        _stat_value(stat.S_IFDIR | 0o755, uid=1, gid=0),
        _stat_value(stat.S_IFDIR | 0o755, uid=0, gid=1),
        _stat_value(stat.S_IFDIR | 0o775, uid=0, gid=0),
    ],
)
def test_writer_cgroup_rejects_symlink_owner_group_or_writable_mode(item):
    with pytest.raises(RuntimeError, match="cgroup identity"):
        collector._validate_writer_cgroup_stat(item)


def _mock_verified_cgroup(monkeypatch):
    monkeypatch.setattr(
        collector,
        "_verified_writer_cgroup_fd",
        lambda: contextlib.nullcontext((42, 9, 101)),
    )
    monkeypatch.setattr(
        collector,
        "_read_cgroup_procs",
        lambda *_args, **_kwargs: (4343,),
    )


def _stable_bpf_query(_descriptor, *, attach_type, effective):
    values = {
        (collector._BPF_CGROUP_INET_INGRESS, False): (11,),
        (collector._BPF_CGROUP_INET_INGRESS, True): (11, 12),
        (collector._BPF_CGROUP_INET_EGRESS, False): (13,),
        (collector._BPF_CGROUP_INET_EGRESS, True): (13, 14),
    }
    return values[(attach_type, effective)]


def test_writer_cgroup_bpf_binding_requires_stable_direct_and_effective_programs(
    monkeypatch,
):
    _mock_verified_cgroup(monkeypatch)
    monkeypatch.setattr(collector, "_query_bpf_program_ids", _stable_bpf_query)

    binding = collector._collect_writer_cgroup_bpf_binding(writer_pid=4343)

    assert binding.as_dict() == {
        "cgroup_device": 9,
        "cgroup_inode": 101,
        "main_pid": 4343,
        "ingress_direct_program_ids": [11],
        "ingress_effective_program_ids": [11, 12],
        "egress_direct_program_ids": [13],
        "egress_effective_program_ids": [13, 14],
    }


def test_writer_cgroup_bpf_binding_rejects_no_program(monkeypatch):
    _mock_verified_cgroup(monkeypatch)
    monkeypatch.setattr(
        collector,
        "_query_bpf_program_ids",
        lambda *_args, **_kwargs: (),
    )

    with pytest.raises(RuntimeError, match="program IDs are invalid"):
        collector._collect_writer_cgroup_bpf_binding(writer_pid=4343)


def test_writer_cgroup_bpf_binding_rejects_only_one_direction(monkeypatch):
    _mock_verified_cgroup(monkeypatch)

    def query(descriptor, *, attach_type, effective):
        if attach_type == collector._BPF_CGROUP_INET_EGRESS:
            return ()
        return _stable_bpf_query(
            descriptor,
            attach_type=attach_type,
            effective=effective,
        )

    monkeypatch.setattr(collector, "_query_bpf_program_ids", query)

    with pytest.raises(RuntimeError, match="egress direct"):
        collector._collect_writer_cgroup_bpf_binding(writer_pid=4343)


def test_writer_cgroup_bpf_binding_rejects_program_rotation(monkeypatch):
    _mock_verified_cgroup(monkeypatch)
    calls = 0

    def query(descriptor, *, attach_type, effective):
        nonlocal calls
        calls += 1
        value = _stable_bpf_query(
            descriptor,
            attach_type=attach_type,
            effective=effective,
        )
        if calls > 4 and attach_type == collector._BPF_CGROUP_INET_INGRESS:
            return tuple(sorted({*value, 99}))
        return value

    monkeypatch.setattr(collector, "_query_bpf_program_ids", query)

    with pytest.raises(RuntimeError, match="rotated"):
        collector._collect_writer_cgroup_bpf_binding(writer_pid=4343)


@pytest.mark.parametrize(
    "members",
    [
        (),
        (4343, 4999),
        (4999,),
    ],
)
def test_writer_cgroup_bpf_binding_requires_exact_singleton_main_pid(
    monkeypatch,
    members,
):
    _mock_verified_cgroup(monkeypatch)
    monkeypatch.setattr(
        collector,
        "_read_cgroup_procs",
        lambda *_args, **_kwargs: members,
    )
    monkeypatch.setattr(collector, "_query_bpf_program_ids", _stable_bpf_query)

    with pytest.raises(RuntimeError, match="exact MainPID"):
        collector._collect_writer_cgroup_bpf_binding(writer_pid=4343)


def test_writer_cgroup_bpf_binding_rejects_membership_rotation(monkeypatch):
    _mock_verified_cgroup(monkeypatch)
    observations = iter(((4343,), (4999,)))
    monkeypatch.setattr(
        collector,
        "_read_cgroup_procs",
        lambda *_args, **_kwargs: next(observations),
    )
    monkeypatch.setattr(collector, "_query_bpf_program_ids", _stable_bpf_query)

    with pytest.raises(RuntimeError, match="membership rotated"):
        collector._collect_writer_cgroup_bpf_binding(writer_pid=4343)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"4343",
        b"0\n",
        b"1\n",
        b"04343\n",
        b"4343\n\n",
        b" 4343\n",
        b"not-a-pid\n",
        b"4343\n4343\n",
        b"9" * (collector._MAX_CGROUP_PROCS_BYTES + 1),
    ],
)
def test_cgroup_procs_parser_rejects_empty_or_malformed_payload(raw):
    with pytest.raises(RuntimeError, match="cgroup.procs"):
        collector._parse_cgroup_procs(raw)


def test_cgroup_procs_parser_returns_canonical_pid_set():
    assert collector._parse_cgroup_procs(b"4999\n4343\n") == (4343, 4999)


def test_bpf_query_fails_closed_on_unapproved_architecture(monkeypatch):
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(
        collector.os,
        "uname",
        lambda: SimpleNamespace(machine="riscv64"),
    )

    with pytest.raises(RuntimeError, match="architecture is not approved"):
        collector._bpf_syscall_number()


def test_bpf_query_fails_closed_on_kernel_permission_error(monkeypatch):
    class FakeSyscall:
        restype = None

        def __call__(self, *_args):
            collector.ctypes.set_errno(collector.errno.EPERM)
            return -1

    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(
        collector.os,
        "uname",
        lambda: SimpleNamespace(machine="x86_64"),
    )
    monkeypatch.setattr(
        collector.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(syscall=FakeSyscall()),
    )

    with pytest.raises(RuntimeError, match="EPERM"):
        collector._query_bpf_program_ids(
            42,
            attach_type=collector._BPF_CGROUP_INET_INGRESS,
            effective=False,
        )


class _FakeUnixClient:
    def __init__(self, credentials):
        self._credentials = credentials

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def settimeout(self, timeout):
        assert timeout == 1.0

    def connect(self, path):
        assert path == str(collector.DEFAULT_SOCKET_PATH)

    def getsockopt(self, level, option, size):
        assert level == socket.SOL_SOCKET
        assert option == 17
        assert size == struct.calcsize("=3i")
        return self._credentials


def test_writer_socket_peer_binds_path_listener_and_exact_so_peercred(monkeypatch):
    item = _stat_value(
        stat.S_IFSOCK | 0o660,
        device=7,
        inode=99,
    )
    monkeypatch.setattr(collector.os, "lstat", lambda _path: item)
    monkeypatch.setattr(collector, "_has_posix_acl", lambda _path: False)
    monkeypatch.setattr(
        collector,
        "_unix_listener_paths_for_pid",
        lambda _pid: [str(collector.DEFAULT_SOCKET_PATH)],
    )
    monkeypatch.setattr(collector.socket, "SO_PEERCRED", 17, raising=False)
    monkeypatch.setattr(
        collector.socket,
        "socket",
        lambda *_args: _FakeUnixClient(struct.pack("=3i", 4343, 1002, 2002)),
    )

    observed = collector._connect_and_validate_writer_socket_peer(
        writer_pid=4343,
        writer_uid=1002,
        writer_gid=2002,
        socket_gid=2001,
        expected_device=7,
        expected_inode=99,
    )

    assert (observed.st_dev, observed.st_ino) == (7, 99)


@pytest.mark.parametrize(
    "credentials",
    [
        (4999, 1002, 2002),
        (4343, 1003, 2002),
        (4343, 1002, 2003),
    ],
)
def test_writer_socket_peer_rejects_so_peercred_drift(monkeypatch, credentials):
    item = _stat_value(stat.S_IFSOCK | 0o660, device=7, inode=99)
    monkeypatch.setattr(collector.os, "lstat", lambda _path: item)
    monkeypatch.setattr(collector, "_has_posix_acl", lambda _path: False)
    monkeypatch.setattr(
        collector,
        "_unix_listener_paths_for_pid",
        lambda _pid: [str(collector.DEFAULT_SOCKET_PATH)],
    )
    monkeypatch.setattr(collector.socket, "SO_PEERCRED", 17, raising=False)
    monkeypatch.setattr(
        collector.socket,
        "socket",
        lambda *_args: _FakeUnixClient(struct.pack("=3i", *credentials)),
    )

    with pytest.raises(RuntimeError, match="SO_PEERCRED identity drifted"):
        collector._connect_and_validate_writer_socket_peer(
            writer_pid=4343,
            writer_uid=1002,
            writer_gid=2002,
            socket_gid=2001,
            expected_device=7,
            expected_inode=99,
        )


def test_writer_socket_peer_rejects_posix_acl_before_connect(monkeypatch):
    item = _stat_value(stat.S_IFSOCK | 0o660, device=7, inode=99)
    monkeypatch.setattr(collector.os, "lstat", lambda _path: item)
    monkeypatch.setattr(collector, "_has_posix_acl", lambda _path: True)
    monkeypatch.setattr(
        collector,
        "_unix_listener_paths_for_pid",
        lambda _pid: [str(collector.DEFAULT_SOCKET_PATH)],
    )

    with pytest.raises(RuntimeError, match="POSIX ACL"):
        collector._connect_and_validate_writer_socket_peer(
            writer_pid=4343,
            writer_uid=1002,
            writer_gid=2002,
            socket_gid=2001,
            expected_device=7,
            expected_inode=99,
        )


def test_writer_socket_peer_rejects_posix_acl_acquired_during_connect(monkeypatch):
    item = _stat_value(stat.S_IFSOCK | 0o660, device=7, inode=99)
    acl_samples = iter((False, True))
    monkeypatch.setattr(collector.os, "lstat", lambda _path: item)
    monkeypatch.setattr(collector, "_has_posix_acl", lambda _path: next(acl_samples))
    monkeypatch.setattr(
        collector,
        "_unix_listener_paths_for_pid",
        lambda _pid: [str(collector.DEFAULT_SOCKET_PATH)],
    )
    monkeypatch.setattr(collector.socket, "SO_PEERCRED", 17, raising=False)
    monkeypatch.setattr(
        collector.socket,
        "socket",
        lambda *_args: _FakeUnixClient(struct.pack("=3i", 4343, 1002, 2002)),
    )

    with pytest.raises(RuntimeError, match="acquired a POSIX ACL"):
        collector._connect_and_validate_writer_socket_peer(
            writer_pid=4343,
            writer_uid=1002,
            writer_gid=2002,
            socket_gid=2001,
            expected_device=7,
            expected_inode=99,
        )


@pytest.mark.parametrize("name", ["O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"])
def test_secure_directory_open_flags_are_mandatory(monkeypatch, name):
    monkeypatch.delattr(collector.os, name, raising=False)

    with pytest.raises(RuntimeError, match=name):
        collector._required_linux_open_flag(name)


def test_release_artifact_manifest_is_rehashed_against_live_tree(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / REVISION
    gateway = root / "gateway"
    bin_dir = root / "bin"
    gateway.mkdir(parents=True)
    bin_dir.mkdir()
    interpreter = bin_dir / "python"
    bootstrap = gateway / "canonical_writer_bootstrap.py"
    gateway_bootstrap = gateway / "canonical_writer_gateway_bootstrap.py"
    interpreter.write_bytes(b"ELF-test-interpreter")
    bootstrap.write_text("BOOTSTRAP = True\n", encoding="utf-8")
    gateway_bootstrap.write_text("GATEWAY = True\n", encoding="utf-8")
    interpreter.chmod(0o555)
    bootstrap.chmod(0o444)
    gateway_bootstrap.chmod(0o444)
    gateway.chmod(0o555)
    bin_dir.chmod(0o555)

    entries = []
    for path in (bin_dir, interpreter, gateway, bootstrap, gateway_bootstrap):
        item = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        value = {
            "path": relative,
            "kind": "directory" if stat.S_ISDIR(item.st_mode) else "file",
            "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        }
        if value["kind"] == "file":
            raw = path.read_bytes()
            value.update(
                {
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        entries.append(value)
    entries.sort(key=lambda value: value["path"])
    unsigned = {
        "schema": "muncho-writer-only-release.v1",
        "revision": REVISION,
        "artifact_root": str(root),
        "python_version": "3.12.10",
        "interpreter": str(interpreter),
        "writer_module": "gateway.canonical_writer_bootstrap",
        "writer_module_origin": str(bootstrap),
        "gateway_module": "gateway.canonical_writer_gateway_bootstrap",
        "gateway_module_origin": str(gateway_bootstrap),
        "entries": entries,
    }
    artifact = collector._sha256_json(unsigned)
    release_manifest = root / "release-manifest.json"
    release_manifest.write_text(
        json.dumps(
            {**unsigned, "artifact_sha256": artifact},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    release_manifest.chmod(0o400)
    root.chmod(0o555)
    root_stat = root.stat()
    snapshot = {
        "writer_deployment": {
            "policy": {
                "artifact_root": str(root),
                "interpreter": str(interpreter),
                "module_origin": str(bootstrap),
            }
        },
        "gateway_deployment": {
            "policy": {
                "artifact_root": str(root),
                "interpreter": str(interpreter),
                "module_origin": str(gateway_bootstrap),
            }
        },
    }
    trusted = collector.TrustedDeploymentManifest(
        revision=REVISION,
        artifact_sha256=artifact,
        snapshot_policy_sha256="5" * 64,
        host_contract={},
        snapshot_template={},
        manifest_sha256="6" * 64,
    )
    monkeypatch.setattr(
        collector,
        "_validate_parent_chain",
        lambda *_args, **_kwargs: None,
    )

    collector._verify_release_artifact(
        snapshot,
        trusted,
        _expected_uid=root_stat.st_uid,
        _expected_gid=root_stat.st_gid,
    )

    gateway_bootstrap.chmod(0o644)
    gateway_bootstrap.write_text("GATEWAY = False\n", encoding="utf-8")
    gateway_bootstrap.chmod(0o444)
    with pytest.raises(RuntimeError, match="digest changed"):
        collector._verify_release_artifact(
            snapshot,
            trusted,
            _expected_uid=root_stat.st_uid,
            _expected_gid=root_stat.st_gid,
        )


def test_collects_hba_last_evaluates_in_process_and_writes_bounded_receipt(
    monkeypatch,
):
    snapshot = _snapshot()
    manifest = _manifest_for(snapshot)
    active = _hba_receipt(observed_at=NOW - 1)
    calls = []
    written = []

    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector, "_effective_uid", lambda: 0)
    monkeypatch.setattr(collector, "load_trusted_manifest", lambda _path: manifest)
    monkeypatch.setattr(collector, "_boot_id_sha256", lambda: BOOT)
    monkeypatch.setattr(
        collector, "current_host_identity_sha256", lambda: HOST
    )
    monkeypatch.setattr(collector, "_current_unix", lambda: NOW)
    monkeypatch.setattr(
        collector,
        "_current_boottime_ns",
        lambda: 50_000_000_000,
    )

    def live_collector(_manifest, *, now_unix):
        calls.append("snapshot")
        assert now_unix == NOW
        live = copy.deepcopy(snapshot)
        live["authoritative_external_evidence"] = {
            "external_iam_receipt_sha256": "a" * 64,
            "native_observation_receipt_sha256": "b" * 64,
            "host_preparation_receipt_sha256": "3" * 64,
            "legacy_helper_evidence_sha256": "4" * 64,
        }
        return live

    def hba_probe(_manifest, live_snapshot):
        calls.append("hba")
        assert live_snapshot["database"]["policy"] == snapshot["database"][
            "policy"
        ]
        return active

    def evaluate(live_snapshot, _manifest, **_kwargs):
        calls.append("evaluate")
        evidence = live_snapshot["database"][
            "managed_cloudsqladmin_hba_rejection_evidence"
        ]
        assert evidence["receipt"] == active.as_dict()
        assert evidence["receipt_sha256"] == active.sha256
        return PreflightReport((PreflightCheck("focused.ok", True, "ok"),))

    monkeypatch.setattr(collector, "_authoritative_writer_only_report", evaluate)
    monkeypatch.setattr(
        collector,
        "evaluate_snapshot",
        lambda _snapshot: PreflightReport(
            (PreflightCheck("full.ok", True, "ok"),)
        ),
    )
    monkeypatch.setattr(collector, "_collect_live_snapshot", live_collector)
    monkeypatch.setattr(
        collector,
        "probe_active_hba_from_writer_config",
        hba_probe,
    )
    monkeypatch.setattr(
        collector,
        "_collect_runtime_liveness",
        lambda *_args, **_kwargs: collector.RuntimeLivenessBinding(
            sha256="1" * 64,
            generation=7,
        ),
    )
    monkeypatch.setattr(
        collector,
        "_validate_runtime_readiness",
        lambda *_args, **_kwargs: collector.RuntimeReadinessBinding(
            gateway_sha256="c" * 64,
            writer_sha256="d" * 64,
        ),
    )
    monkeypatch.setattr(
        collector,
        "_validate_runtime_code_closure",
        lambda _snapshot: collector.RuntimeCodeClosureBinding(
            gateway_sha256="e" * 64,
            writer_sha256="f" * 64,
        ),
    )
    monkeypatch.setattr(collector, "_fence_live_activation", lambda **_kwargs: None)
    monkeypatch.setattr(
        collector,
        "_collector_lock",
        lambda _path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(collector, "_invalidate_root_receipt", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "_write_root_receipt",
        lambda path, value: written.append((path, copy.deepcopy(value))),
    )
    archived = []
    monkeypatch.setattr(
        collector,
        "_write_append_only_root_evidence",
        lambda path, value: archived.append((path, copy.deepcopy(value))),
    )

    outcome = collector.collect_and_evaluate(
        "/etc/muncho/manifest.json",
        "/run/muncho-preflight/receipt.json",
        activation_plan_sha256=PLAN,
    )

    assert calls == ["snapshot", "hba", "evaluate"]
    assert outcome.report.ok
    assert written == [
        ("/run/muncho-preflight/receipt.json", dict(outcome.receipt))
    ]
    assert archived == [
        (
            Path(outcome.receipt["evidence_bundle_path"]),
            archived[0][1],
        )
    ]
    assert (
        collector._sha256_json(archived[0][1])
        == outcome.receipt["evidence_bundle_sha256"]
    )
    assert outcome.receipt["ok"] is True
    assert outcome.receipt["boot_id_sha256"] == BOOT
    assert outcome.receipt["hba_receipt_sha256"] == active.sha256
    assert outcome.receipt["gateway_main_pid"] == 4242
    assert outcome.receipt["writer_main_pid"] == 4343
    assert outcome.receipt["gateway_readiness_sha256"] == "c" * 64
    assert outcome.receipt["gateway_liveness_sha256"] == "1" * 64
    assert outcome.receipt["gateway_liveness_generation"] == 7
    assert outcome.receipt["writer_runtime_attestation_sha256"] == "d" * 64
    assert outcome.receipt["gateway_code_closure_sha256"] == "e" * 64
    assert outcome.receipt["writer_code_closure_sha256"] == "f" * 64
    assert outcome.receipt["expires_at_boottime_ns"] == 80_000_000_000
    encoded = json.dumps(outcome.receipt, sort_keys=True)
    assert "password-value-must-never-appear" not in encoded


def test_failed_in_process_report_never_writes_root_receipt(monkeypatch):
    snapshot = _snapshot()
    manifest = _manifest_for(snapshot)
    written = []
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector, "_effective_uid", lambda: 0)
    monkeypatch.setattr(collector, "load_trusted_manifest", lambda _path: manifest)
    monkeypatch.setattr(collector, "_boot_id_sha256", lambda: BOOT)
    monkeypatch.setattr(
        collector, "current_host_identity_sha256", lambda: HOST
    )
    monkeypatch.setattr(collector, "_current_unix", lambda: NOW)
    monkeypatch.setattr(
        collector,
        "_current_boottime_ns",
        lambda: 50_000_000_000,
    )
    monkeypatch.setattr(
        collector,
        "_collect_live_snapshot",
        lambda *_args, **_kwargs: copy.deepcopy(snapshot),
    )
    monkeypatch.setattr(
        collector,
        "_authoritative_writer_only_report",
        lambda *_args, **_kwargs: PreflightReport(
            (PreflightCheck("writer.failed", False, "blocked"),)
        ),
    )
    monkeypatch.setattr(
        collector,
        "evaluate_snapshot",
        lambda _snapshot: PreflightReport(
            (PreflightCheck("full.ok", True, "ok"),)
        ),
    )
    monkeypatch.setattr(
        collector,
        "probe_active_hba_from_writer_config",
        lambda *_args: _hba_receipt(observed_at=NOW - 1),
    )
    monkeypatch.setattr(
        collector,
        "_collect_runtime_liveness",
        lambda *_args, **_kwargs: collector.RuntimeLivenessBinding(
            sha256="1" * 64,
            generation=7,
        ),
    )
    monkeypatch.setattr(
        collector,
        "_validate_runtime_readiness",
        lambda *_args, **_kwargs: collector.RuntimeReadinessBinding(
            gateway_sha256="c" * 64,
            writer_sha256="d" * 64,
        ),
    )
    monkeypatch.setattr(
        collector,
        "_validate_runtime_code_closure",
        lambda _snapshot: collector.RuntimeCodeClosureBinding(
            gateway_sha256="e" * 64,
            writer_sha256="f" * 64,
        ),
    )
    monkeypatch.setattr(
        collector,
        "_collector_lock",
        lambda _path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(collector, "_invalidate_root_receipt", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "_write_root_receipt",
        lambda *_args, **_kwargs: written.append(True),
    )

    with pytest.raises(RuntimeError, match="writer.failed"):
        collector.collect_and_evaluate(
            "/etc/muncho/manifest.json",
            "/run/muncho-preflight/receipt.json",
            activation_plan_sha256=PLAN,
        )

    assert written == []


def test_joint_policy_and_attestation_change_cannot_bypass_manifest():
    snapshot = _snapshot()
    manifest = _manifest_for(snapshot)
    snapshot["writer_deployment"]["policy"]["revision"] = "9" * 40
    snapshot["writer_deployment"]["attestation"] = {
        "revision": "9" * 40
    }

    with pytest.raises(ValueError, match="approved manifest"):
        collector._bind_snapshot_to_manifest(snapshot, manifest)


@pytest.mark.parametrize(
    "observed_at,certificate,tls_server_name",
    [
        (NOW - 31, "d" * 64, SQL_TLS_SERVER_NAME),
        (NOW - 1, "e" * 64, SQL_TLS_SERVER_NAME),
        (NOW - 1, "d" * 64, "other.internal"),
    ],
)
def test_active_hba_must_be_at_most_thirty_seconds_old_and_peer_bound(
    observed_at,
    certificate,
    tls_server_name,
):
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="stale or not bound"):
        collector._install_active_hba_evidence(
            snapshot,
            _hba_receipt(
                observed_at=observed_at,
                certificate=certificate,
                tls_server_name=tls_server_name,
                ttl_seconds=100,
            ),
            collected_at_unix=NOW,
        )


def test_active_hba_probe_rejects_config_tls_name_drift(monkeypatch):
    snapshot = _snapshot()
    manifest = _manifest_for(snapshot)
    baseline = _hba_receipt(observed_at=NOW - 300)
    config = SimpleNamespace(
        discord_edge_authority=SimpleNamespace(enabled=False),
        privileges=SimpleNamespace(
            managed_cloudsqladmin_hba_rejection_receipt=baseline,
            managed_cloudsqladmin_hba_rejection_sha256=baseline.sha256,
        ),
        database=SimpleNamespace(
            host=SQL_PRIVATE_IP,
            tls_server_name="other.internal",
            port=5432,
            database="canonical",
            user="canonical_writer",
        ),
    )
    monkeypatch.setattr(bootstrap, "load_service_config", lambda _path: config)
    monkeypatch.setattr(
        collector,
        "collect_managed_cloudsqladmin_hba_receipt",
        lambda *_args, **_kwargs: pytest.fail("drifted config must not be probed"),
    )

    with pytest.raises(ValueError, match="does not match approved HBA policy"):
        collector.probe_active_hba_from_writer_config(manifest, snapshot)


def test_manifest_shape_and_digests_are_strict():
    snapshot = _snapshot()
    value = _manifest_for(snapshot).to_mapping()
    value["unknown"] = True
    with pytest.raises(ValueError, match="fields are not exact"):
        collector.TrustedDeploymentManifest.from_mapping(value)

    value = _manifest_for(snapshot).to_mapping()
    value["approved_plan_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="fields are not exact"):
        collector.TrustedDeploymentManifest.from_mapping(value)

    value = _manifest_for(snapshot).to_mapping()
    value["host_contract"]["gateway_config_path"] = "/var/lib/hermes/config.yaml"
    with pytest.raises(ValueError, match="managed config path is not pinned"):
        collector.TrustedDeploymentManifest.from_mapping(value)

    value = _manifest_for(snapshot).to_mapping()
    value["snapshot_template"]["gateway_deployment"]["policy"][
        "read_write_paths"
    ].append("/var/lib/hermes-gateway")
    value["snapshot_policy_sha256"] = collector.snapshot_policy_sha256(
        value["snapshot_template"]
    )
    with pytest.raises(ValueError, match="gateway policy is not minimal"):
        collector.TrustedDeploymentManifest.from_mapping(value)

    for mutation in ("missing", "extra"):
        value = _manifest_for(snapshot).to_mapping()
        connection = value["snapshot_template"]["database"]["connection"]
        if mutation == "missing":
            connection.pop("tls_server_name")
        else:
            connection["tls_server_name_alias"] = SQL_TLS_SERVER_NAME
        value["snapshot_policy_sha256"] = collector.snapshot_policy_sha256(
            value["snapshot_template"]
        )
        with pytest.raises(ValueError, match="connection fields are not exact"):
            collector.TrustedDeploymentManifest.from_mapping(value)

    value = _manifest_for(snapshot).to_mapping()
    value["snapshot_template"]["database"]["connection"]["host"] = "8.8.8.8"
    value["snapshot_policy_sha256"] = collector.snapshot_policy_sha256(
        value["snapshot_template"]
    )
    with pytest.raises(ValueError, match="exact private IPv4"):
        collector.TrustedDeploymentManifest.from_mapping(value)


def _systemd_evidence(unit, version, receipt, pid, start):
    digest = collector._sha256_json(receipt)
    return {
        "unit_name": unit,
        "unit_type": "notify",
        "notify_access": "main",
        "active_state": "active",
        "sub_state": "running",
        "systemd_main_pid": pid,
        "systemd_main_pid_start_time_ticks": start,
        "status_text": f"{version}:{digest}",
        "receipt_sha256": digest,
        "receipt": receipt,
    }


def test_runtime_readiness_is_bound_to_exact_systemd_main_pids_and_digests(
    monkeypatch,
):
    snapshot = _snapshot()
    request_id = "11111111-1111-4111-8111-111111111111"
    gateway_receipt = {
        "version": "canonical-writer-readiness-v1",
        "observed_at_unix": NOW,
        "observed_at_boottime_ns": 49_000_000_000,
        "boot_id_sha256": BOOT,
        "gateway_pid": 4242,
        "gateway_start_time_ticks": 123456,
        "writer_request_id": request_id,
        "writer_service": "canonical_writer",
        "writer_protocol": "v1",
        "database_identity": "canonical_brain_migration_owner",
        "gateway_module_origin": (
            f"/opt/releases/{REVISION}/venv/lib/python3.12/site-packages/"
            "gateway/canonical_writer_gateway_bootstrap.py"
        ),
        "gateway_module_sha256": "a" * 64,
        "gateway_dumpable": False,
        "gateway_core_soft_limit": 0,
        "gateway_core_hard_limit": 0,
        "effective_import_paths": [f"/opt/releases/{REVISION}/venv"],
        "unexpected_import_paths": [],
        "loaded_module_origins": [
            f"/opt/releases/{REVISION}/venv/lib/python3.12/site-packages/"
            "gateway/canonical_writer_gateway_bootstrap.py"
        ],
        "unexpected_import_origins": [],
        "loaded_module_origins_complete": True,
        "effective_environment_variable_names": ["NOTIFY_SOCKET"],
        "effective_environment_variable_value_sha256": {
            "NOTIFY_SOCKET": "d" * 64
        },
    }
    writer_receipt = {
        "version": "canonical-writer-runtime-attestation-v1",
        "observed_at_unix": NOW,
        "observed_at_boottime_ns": 49_000_000_000,
        "boot_id_sha256": BOOT,
        "writer_pid": 4343,
        "writer_start_time_ticks": 654321,
        "bootstrap_module_origin": snapshot["writer_deployment"]["policy"][
            "module_origin"
        ],
        "bootstrap_module_sha256": "b" * 64,
        "service_module_origin": (
            f"/opt/releases/{REVISION}/venv/lib/python3.12/site-packages/"
            "gateway/canonical_writer_service.py"
        ),
        "service_module_sha256": "c" * 64,
        "statement_catalog_sha256": collector.PRODUCTION_CATALOG_SHA256,
        "database_identity": "canonical_brain_migration_owner",
        "database_role": "canonical_writer",
        "private_schema_identity_sha256": "e" * 64,
        "managed_hba_baseline_sha256": snapshot["database"]["policy"][
            "managed_cloudsqladmin_hba_rejection_sha256"
        ],
        "discord_edge_authority_enabled": False,
        "socket_path": "/run/muncho-canonical-writer/writer.sock",
        "socket_inode": 99,
        "socket_device": 7,
        "socket_owner_uid": 1002,
        "socket_group_gid": 2001,
        "socket_mode": "0660",
        "writer_dumpable": False,
        "writer_core_soft_limit": 0,
        "writer_core_hard_limit": 0,
        "effective_import_paths": [f"/opt/releases/{REVISION}/venv"],
        "unexpected_import_paths": [],
        "loaded_module_origins": [
            snapshot["writer_deployment"]["policy"]["module_origin"]
        ],
        "unexpected_import_origins": [],
        "loaded_module_origins_complete": True,
        "effective_environment_variable_names": ["NOTIFY_SOCKET"],
        "effective_environment_variable_value_sha256": {
            "NOTIFY_SOCKET": "d" * 64
        },
    }
    snapshot["runtime_readiness"] = {
        "gateway": _systemd_evidence(
            "hermes-cloud-gateway.service",
            "canonical-writer-readiness-v1",
            gateway_receipt,
            4242,
            123456,
        ),
        "writer": _systemd_evidence(
            "muncho-canonical-writer.service",
            "canonical-writer-runtime-attestation-v1",
            writer_receipt,
            4343,
            654321,
        ),
    }
    module_digests = {
        gateway_receipt["gateway_module_origin"]: "a" * 64,
        writer_receipt["bootstrap_module_origin"]: "b" * 64,
        writer_receipt["service_module_origin"]: "c" * 64,
    }
    monkeypatch.setattr(
        collector,
        "_sha256_trusted_file",
        lambda path, **_kwargs: module_digests[path],
    )

    binding = collector._validate_runtime_readiness(
        snapshot,
        current_boot_id_sha256=BOOT,
        current_boottime_ns=50_000_000_000,
    )
    assert binding.gateway_sha256 == collector._sha256_json(gateway_receipt)
    assert binding.writer_sha256 == collector._sha256_json(writer_receipt)

    snapshot["runtime_readiness"]["writer"]["status_text"] = (
        "canonical-writer-runtime-attestation-v1:" + "f" * 64
    )
    with pytest.raises(RuntimeError, match="unauthenticated"):
        collector._validate_runtime_readiness(
            snapshot,
            current_boot_id_sha256=BOOT,
            current_boottime_ns=50_000_000_000,
        )


def _liveness_receipt(*, observed_at_boottime_ns: int, generation: int = 7):
    return {
        "version": "canonical-writer-liveness-v1",
        "generation": generation,
        "observed_at_unix": NOW,
        "observed_at_boottime_ns": observed_at_boottime_ns,
        "boot_id_sha256": BOOT,
        "gateway_pid": 4242,
        "gateway_start_time_ticks": 123456,
        "writer_request_id": "22222222-2222-4222-8222-222222222222",
        "writer_service": "canonical_writer",
        "writer_protocol": "v1",
        "database_identity": "canonical_brain_migration_owner",
        "socket_path": "/run/muncho-canonical-writer/writer.sock",
        "socket_device": 7,
        "socket_inode": 99,
        "socket_owner_uid": 1002,
        "socket_group_gid": 2001,
        "socket_mode": "0660",
    }


def _gateway_liveness_systemd(receipt, startup_digest="a" * 64):
    from gateway.canonical_writer_readiness import writer_liveness_status_text

    return {
        "Type": "notify",
        "NotifyAccess": "main",
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": "4242",
        "StatusText": writer_liveness_status_text(
            startup_digest,
            receipt["generation"],
            collector._sha256_json(receipt),
        ),
    }


def test_runtime_liveness_requires_fresh_ping_from_exact_gateway_pid(monkeypatch):
    snapshot = _snapshot()
    receipt = _liveness_receipt(observed_at_boottime_ns=49_000_000_000)
    monkeypatch.setattr(
        collector,
        "_read_runtime_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        collector,
        "_systemctl_show",
        lambda _unit: _gateway_liveness_systemd(receipt),
    )
    monkeypatch.setattr(
        collector,
        "_process_start_time_ticks",
        lambda pid: 123456 if pid == 4242 else 0,
    )

    binding = collector._collect_runtime_liveness(
        snapshot,
        gateway_readiness_sha256="a" * 64,
        current_boot_id_sha256=BOOT,
        current_boottime_ns=50_000_000_000,
    )

    assert binding.generation == 7
    assert binding.sha256 == collector._sha256_json(receipt)
    assert snapshot["runtime_liveness"] == receipt


def test_runtime_liveness_rejects_hung_same_pid_with_stale_ping(monkeypatch):
    snapshot = _snapshot()
    receipt = _liveness_receipt(observed_at_boottime_ns=40_000_000_000)
    monkeypatch.setattr(
        collector,
        "_read_runtime_receipt",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        collector,
        "_systemctl_show",
        lambda _unit: _gateway_liveness_systemd(receipt),
    )
    monkeypatch.setattr(
        collector,
        "_process_start_time_ticks",
        lambda pid: 123456 if pid == 4242 else 0,
    )

    with pytest.raises(RuntimeError, match="stale or unbound"):
        collector._collect_runtime_liveness(
            snapshot,
            gateway_readiness_sha256="a" * 64,
            current_boot_id_sha256=BOOT,
            current_boottime_ns=50_000_000_000,
        )


def test_same_uid_receipt_forgery_lacks_mainpid_status_authority(monkeypatch):
    snapshot = _snapshot()
    forged = _liveness_receipt(
        observed_at_boottime_ns=49_999_000_000,
        generation=8,
    )
    previous = _liveness_receipt(
        observed_at_boottime_ns=49_000_000_000,
        generation=7,
    )
    monkeypatch.setattr(
        collector,
        "_read_runtime_receipt",
        lambda *_args, **_kwargs: forged,
    )
    monkeypatch.setattr(
        collector,
        "_systemctl_show",
        lambda _unit: _gateway_liveness_systemd(previous),
    )

    with pytest.raises(RuntimeError, match="StatusText.*unauthenticated"):
        collector._collect_runtime_liveness(
            snapshot,
            gateway_readiness_sha256="a" * 64,
            current_boot_id_sha256=BOOT,
            current_boottime_ns=50_000_000_000,
        )


def test_liveness_transition_is_sampled_as_one_coherent_generation(monkeypatch):
    snapshot = _snapshot()
    previous = _liveness_receipt(
        observed_at_boottime_ns=48_000_000_000,
        generation=7,
    )
    current = _liveness_receipt(
        observed_at_boottime_ns=49_000_000_000,
        generation=8,
    )
    receipts = iter((previous, current, current, current))
    statuses = iter(
        (
            _gateway_liveness_systemd(previous),
            _gateway_liveness_systemd(current),
        )
    )
    monkeypatch.setattr(
        collector,
        "_read_runtime_receipt",
        lambda *_args, **_kwargs: next(receipts),
    )
    monkeypatch.setattr(
        collector,
        "_systemctl_show",
        lambda _unit: next(statuses),
    )
    monkeypatch.setattr(
        collector,
        "_process_start_time_ticks",
        lambda pid: 123456 if pid == 4242 else 0,
    )

    binding = collector._collect_runtime_liveness(
        snapshot,
        gateway_readiness_sha256="a" * 64,
        current_boot_id_sha256=BOOT,
        current_boottime_ns=50_000_000_000,
    )

    assert binding.generation == 8
    assert binding.sha256 == collector._sha256_json(current)


def test_heartbeat_cannot_be_forged_by_independent_gateway_uid_process(
    monkeypatch,
):
    observed = {1001: [4242, 4999], 1002: [4343]}
    monkeypatch.setattr(
        collector,
        "_pids_for_uid",
        lambda uid: observed[uid],
    )

    with pytest.raises(RuntimeError, match="UID surface is not exclusive"):
        collector._require_exclusive_service_uids(
            gateway_uid=1001,
            gateway_pid=4242,
            writer_uid=1002,
            writer_pid=4343,
        )


def _valid_receipt(manifest):
    return {
        "schema": collector.RECEIPT_SCHEMA,
        "ok": True,
        "mode": collector.WRITER_ONLY_MODE,
        "boot_id_sha256": BOOT,
        "host_identity_sha256": HOST,
        "collected_at_unix": NOW,
        "collected_at_boottime_ns": 100_000_000_000,
        "expires_at_boottime_ns": 130_000_000_000,
        "manifest_sha256": manifest.manifest_sha256,
        "activation_plan_sha256": PLAN,
        "revision": REVISION,
        "artifact_sha256": ARTIFACT,
        "snapshot_policy_sha256": manifest.snapshot_policy_sha256,
        "snapshot_sha256": "5" * 64,
        "report_sha256": "6" * 64,
        "full_report_sha256": "d" * 64,
        "additive_report_sha256": "e" * 64,
        "evidence_bundle_path": (
            f"{collector.DEFAULT_ROOT_EVIDENCE_ROOT}/{REVISION}/{PLAN}/"
            f"{'3' * 64}.json"
        ),
        "evidence_bundle_sha256": "3" * 64,
        "external_iam_receipt_sha256": "f" * 64,
        "native_observation_receipt_sha256": "1" * 64,
        "host_preparation_receipt_sha256": "4" * 64,
        "legacy_helper_evidence_sha256": "2" * 64,
        "hba_receipt_sha256": "7" * 64,
        "gateway_readiness_sha256": "8" * 64,
        "gateway_liveness_sha256": "c" * 64,
        "gateway_liveness_generation": 7,
        "writer_runtime_attestation_sha256": "9" * 64,
        "gateway_code_closure_sha256": "a" * 64,
        "writer_code_closure_sha256": "b" * 64,
        "gateway_main_pid": 4242,
        "gateway_start_time_ticks": 123456,
        "writer_main_pid": 4343,
        "writer_start_time_ticks": 654321,
        "writer_socket_device": 7,
        "writer_socket_inode": 99,
        "writer_runtime_directory_device": 8,
        "writer_runtime_directory_inode": 100,
        "writer_ip_address_allow_network": "10.0.0.8/32",
        "writer_cgroup_device": 9,
        "writer_cgroup_inode": 101,
        "writer_cgroup_main_pid": 4343,
        "writer_bpf_ingress_direct_program_ids": [11],
        "writer_bpf_ingress_effective_program_ids": [11, 12],
        "writer_bpf_egress_direct_program_ids": [13],
        "writer_bpf_egress_effective_program_ids": [13, 14],
        "failed_checks": [],
    }


def test_root_evidence_bundle_recomputes_reports_and_rejects_tampered_detail(
    monkeypatch,
):
    manifest = _manifest_for(_snapshot())
    full = PreflightReport(
        (PreflightCheck("full.check", True, "derived full detail"),)
    ).to_dict()
    additive = PreflightReport(
        (PreflightCheck("additive.check", True, "derived additive detail"),)
    ).to_dict()
    combined = {
        "ok": True,
        "checks": full["checks"] + additive["checks"],
    }
    receipt = _valid_receipt(manifest)
    snapshot = {
        "root_preflight_identity": {
            "boot_id_sha256": BOOT,
            "host_identity_sha256": HOST,
        },
        "authoritative_external_evidence": {
            name: receipt[name]
            for name in (
                "external_iam_receipt_sha256",
                "native_observation_receipt_sha256",
                "host_preparation_receipt_sha256",
                "legacy_helper_evidence_sha256",
            )
        }
    }
    bundle, path, digest = collector._build_root_evidence_bundle(
        manifest=manifest,
        activation_plan_sha256=PLAN,
        snapshot=snapshot,
        full_report=full,
        additive_report=additive,
        combined_report=combined,
    )
    components = bundle["component_sha256"]
    receipt.update(
        {
            "snapshot_sha256": components["snapshot_sha256"],
            "full_report_sha256": components["full_report_sha256"],
            "additive_report_sha256": components["additive_report_sha256"],
            "report_sha256": components["combined_report_sha256"],
            "evidence_bundle_path": str(path),
            "evidence_bundle_sha256": digest,
        }
    )
    monkeypatch.setattr(
        collector,
        "evaluate_snapshot",
        lambda _snapshot: PreflightReport(
            (PreflightCheck("full.check", True, "derived full detail"),)
        ),
    )
    monkeypatch.setattr(
        collector,
        "_authoritative_writer_only_report",
        lambda *_args, **_kwargs: PreflightReport(
            (
                PreflightCheck(
                    "additive.check", True, "derived additive detail"
                ),
            )
        ),
    )
    monkeypatch.setattr(
        collector,
        "_read_trusted_json",
        lambda path_value, **_kwargs: bundle
        if str(path_value) == str(path)
        else {},
    )
    assert collector._validate_root_evidence_bundle(
        receipt,
        manifest,
        expected_activation_plan_sha256=PLAN,
    ) == bundle

    tampered_full = copy.deepcopy(full)
    tampered_full["checks"][0]["detail"] = "self-consistent but invented"
    tampered_combined = {
        "ok": True,
        "checks": tampered_full["checks"] + additive["checks"],
    }
    tampered, tampered_path, tampered_digest = (
        collector._build_root_evidence_bundle(
            manifest=manifest,
            activation_plan_sha256=PLAN,
            snapshot=snapshot,
            full_report=tampered_full,
            additive_report=additive,
            combined_report=tampered_combined,
        )
    )
    tampered_components = tampered["component_sha256"]
    receipt.update(
        {
            "full_report_sha256": tampered_components["full_report_sha256"],
            "report_sha256": tampered_components["combined_report_sha256"],
            "evidence_bundle_path": str(tampered_path),
            "evidence_bundle_sha256": tampered_digest,
        }
    )
    monkeypatch.setattr(
        collector,
        "_read_trusted_json",
        lambda *_args, **_kwargs: tampered,
    )
    with pytest.raises(ValueError, match="derive from the snapshot"):
        collector._validate_root_evidence_bundle(
            receipt,
            manifest,
            expected_activation_plan_sha256=PLAN,
        )


def test_root_evidence_bundle_rejects_raw_secret_fields():
    manifest = _manifest_for(_snapshot())
    with pytest.raises(ValueError, match="forbidden secret field"):
        collector._build_root_evidence_bundle(
            manifest=manifest,
            activation_plan_sha256=PLAN,
            snapshot={"password": "not-archivable"},
            full_report={"ok": True, "checks": []},
            additive_report={"ok": True, "checks": []},
            combined_report={"ok": True, "checks": []},
        )


def test_root_evidence_archive_is_append_only_and_collision_exact(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root-preflight"
    monkeypatch.setattr(collector, "DEFAULT_ROOT_EVIDENCE_ROOT", root)
    monkeypatch.setattr(
        collector,
        "_ensure_root_evidence_directory",
        lambda path: path.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(collector.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(
        collector,
        "_read_exact_root_evidence_bytes",
        lambda path: path.read_bytes(),
    )
    monkeypatch.setattr(
        collector,
        "_read_trusted_json",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )
    value = {"schema": "test", "value": 1}
    digest = collector._sha256_json(value)
    path = collector._root_evidence_bundle_path(
        revision=REVISION,
        activation_plan_sha256=PLAN,
        bundle_sha256=digest,
    )
    collector._write_append_only_root_evidence(path, value)
    assert json.loads(path.read_text(encoding="utf-8")) == value
    collector._write_append_only_root_evidence(path, value)
    with pytest.raises(RuntimeError, match="collided"):
        collector._write_append_only_root_evidence(
            path,
            {"schema": "test", "value": 2},
        )


def test_exact_root_evidence_reader_accepts_unchanged_canonical_bytes(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "bundle.json"
    raw = collector._canonical_bytes({"schema": "test", "value": 1})
    path.write_bytes(raw)
    path.chmod(0o400)
    real_lstat = os.lstat
    real_fstat = os.fstat

    def projected(item):
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_nlink=item.st_nlink,
            st_uid=0,
            st_gid=0,
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
        )

    monkeypatch.setattr(collector, "_validate_parent_chain", lambda *_a, **_k: None)
    monkeypatch.setattr(collector, "_has_posix_acl", lambda _path: False)
    monkeypatch.setattr(collector.os, "lstat", lambda value: projected(real_lstat(value)))
    monkeypatch.setattr(collector.os, "fstat", lambda fd: projected(real_fstat(fd)))
    assert collector._read_exact_root_evidence_bytes(path) == raw


@pytest.mark.parametrize(
    "boot,now",
    [("8" * 64, 110_000_000_000), (BOOT, 99_999_999_999), (BOOT, 130_000_000_001)],
)
def test_receipt_rejects_other_boot_future_and_expired_evidence(boot, now):
    manifest = _manifest_for(_snapshot())
    with pytest.raises(ValueError):
        collector._validate_receipt_mapping(
            _valid_receipt(manifest),
            manifest,
            expected_activation_plan_sha256=PLAN,
            current_boot_id_sha256=boot,
            current_host_identity_sha256_value=HOST,
            current_boottime_ns=now,
        )


def test_receipt_rejects_cgroup_main_pid_different_from_writer_main_pid():
    manifest = _manifest_for(_snapshot())
    receipt = _valid_receipt(manifest)
    receipt["writer_cgroup_main_pid"] = 4999

    with pytest.raises(ValueError, match="cgroup MainPID"):
        collector._validate_receipt_mapping(
            receipt,
            manifest,
            expected_activation_plan_sha256=PLAN,
            current_boot_id_sha256=BOOT,
            current_host_identity_sha256_value=HOST,
            current_boottime_ns=110_000_000_000,
        )


def test_validate_fresh_receipt_refences_live_services_and_invalidates_on_drift(
    monkeypatch,
):
    manifest = _manifest_for(_snapshot())
    receipt = _valid_receipt(manifest)
    invalidated = []
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector, "_effective_uid", lambda: 0)
    monkeypatch.setattr(collector, "load_trusted_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        collector,
        "_collector_lock",
        lambda _path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        collector,
        "_read_trusted_json",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(collector, "_boot_id_sha256", lambda: BOOT)
    monkeypatch.setattr(
        collector, "current_host_identity_sha256", lambda: HOST
    )
    monkeypatch.setattr(
        collector,
        "_current_boottime_ns",
        lambda: 110_000_000_000,
    )
    monkeypatch.setattr(
        collector,
        "_fence_live_activation",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("PID drift")),
    )
    monkeypatch.setattr(
        collector,
        "_invalidate_root_receipt",
        lambda path: invalidated.append(path),
    )
    monkeypatch.setattr(
        collector,
        "_validate_root_evidence_bundle",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(RuntimeError, match="PID drift"):
        collector.validate_fresh_receipt(
            "/run/preflight/receipt.json",
            "/manifest",
            activation_plan_sha256=PLAN,
        )

    assert invalidated == ["/run/preflight/receipt.json"]


def test_validate_invalidates_old_receipt_when_manifest_load_fails(monkeypatch):
    invalidated = []
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector, "_effective_uid", lambda: 0)
    monkeypatch.setattr(
        collector,
        "_collector_lock",
        lambda _path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        collector,
        "load_trusted_manifest",
        lambda _path: (_ for _ in ()).throw(ValueError("manifest drift")),
    )
    monkeypatch.setattr(
        collector,
        "_invalidate_root_receipt",
        lambda path: invalidated.append(path),
    )

    with pytest.raises(ValueError, match="manifest drift"):
        collector.validate_fresh_receipt(
            "/run/preflight/receipt.json",
            "/manifest",
            activation_plan_sha256=PLAN,
        )

    assert invalidated == ["/run/preflight/receipt.json"]


def test_manifest_rejects_unpinned_discord_absence_path():
    snapshot = _snapshot()
    snapshot["discord_edge"]["token_path"] = "/tmp/absent-token"

    with pytest.raises(ValueError, match="absence paths are not pinned"):
        _manifest_for(snapshot)


def _runtime_code_closure_snapshot(native_path: str, approved_path: str):
    root = f"/opt/releases/{REVISION}"

    def deployment(name: str):
        policy = {
            "artifact_root": root,
            "import_paths": [
                {
                    "path": root,
                    "digest_sha256": ARTIFACT,
                    "object_type": "directory",
                }
            ],
            "preapproved_external_native_executable_mappings": [
                {"path": approved_path, "sha256": "c" * 64}
            ],
        }
        process = {
            "effective_import_paths": [root],
            "loaded_module_origins": [f"{root}/gateway/runtime.py"],
            "mapped_executable_paths": [native_path],
            "loaded_module_origins_complete": True,
            "mapped_executable_paths_complete": True,
            "unexpected_import_origins": [],
            "deleted_code_mappings": [],
            "writable_code_mappings": [],
            "environment_variable_names": ["NOTIFY_SOCKET"],
            "environment_variable_value_sha256": {
                "NOTIFY_SOCKET": "d" * 64
            },
            "environment_identity_sha256": "e" * 64,
        }
        if name == "gateway_deployment":
            policy.update(
                {
                    "dynamic_python_loading_mode": "disabled",
                    "dynamic_python_discovery_paths": [],
                }
            )
            process.update(
                {
                    "dynamic_python_loading_mode": "disabled",
                    "dynamic_python_discovery_paths": [],
                    "dynamic_python_loaded_origins": [],
                    "dynamic_python_writable_paths": [],
                }
            )
        return {
            "policy": policy,
            "attestation": {
                "process": process,
                "unit": {
                    "code_injection_environment_variable_names": [],
                    "environment_files": [],
                },
            },
        }

    return {
        "writer_deployment": deployment("writer_deployment"),
        "gateway_deployment": deployment("gateway_deployment"),
    }


def test_runtime_code_closure_rejects_unapproved_native_mapping(monkeypatch):
    snapshot = _runtime_code_closure_snapshot(
        "/usr/lib/evil.so",
        "/usr/lib/approved.so",
    )

    def fake_lstat(path):
        mode = stat.S_IFREG | 0o444 if str(path).endswith(".py") else stat.S_IFDIR | 0o555
        return SimpleNamespace(st_mode=mode, st_uid=0, st_gid=0)

    monkeypatch.setattr(collector.os, "lstat", fake_lstat)
    monkeypatch.setattr(collector, "_sha256_trusted_file", lambda _path: "c" * 64)

    with pytest.raises(RuntimeError, match="differ from approved policy"):
        collector._validate_runtime_code_closure(snapshot)


def test_runtime_code_closure_accepts_only_exact_preapproved_native_set(monkeypatch):
    path = "/usr/lib/approved.so"
    snapshot = _runtime_code_closure_snapshot(path, path)

    def fake_lstat(candidate):
        mode = (
            stat.S_IFREG | 0o444
            if str(candidate).endswith(".py")
            else stat.S_IFDIR | 0o555
        )
        return SimpleNamespace(st_mode=mode, st_uid=0, st_gid=0)

    monkeypatch.setattr(collector.os, "lstat", fake_lstat)
    monkeypatch.setattr(collector, "_sha256_trusted_file", lambda _path: "c" * 64)

    binding = collector._validate_runtime_code_closure(snapshot)

    assert len(binding.gateway_sha256) == 64
    assert len(binding.writer_sha256) == 64


def test_missing_discord_edge_account_is_stronger_absence_evidence(monkeypatch):
    monkeypatch.setattr(
        collector.pwd,
        "getpwnam",
        lambda _name: (_ for _ in ()).throw(KeyError("missing")),
    )
    monkeypatch.setattr(
        collector,
        "_pids_for_exact_python_module",
        lambda module: [] if module == "scripts.discord_edge_bootstrap" else [1],
    )

    assert collector._discord_edge_process_pids() == []


def _gateway_environment_evidence():
    process_values = {
        **collector._FIXED_GATEWAY_ENVIRONMENT,
        "NOTIFY_SOCKET": "/run/systemd/notify",
    }
    process_names = sorted(process_values)
    process_hashes = {
        name: collector._environment_value_sha256(process_values[name])
        for name in process_names
    }
    runtime_values = dict(process_values)
    runtime_names = sorted(runtime_values)
    runtime_hashes = {
        name: collector._environment_value_sha256(runtime_values[name])
        for name in runtime_names
    }
    return (
        {
            "environment_variable_names": process_names,
            "environment_variable_value_sha256": process_hashes,
        },
        {
            "effective_environment_variable_names": runtime_names,
            "effective_environment_variable_value_sha256": runtime_hashes,
        },
    )


def test_exact_environment_binds_fixed_values_without_exposing_them():
    process, receipt = _gateway_environment_evidence()

    digest = collector._validate_exact_runtime_environment(
        process,
        receipt,
        fixed_values=collector._FIXED_GATEWAY_ENVIRONMENT,
    )

    assert len(digest) == 64
    assert "/var/lib/hermes-gateway" not in json.dumps(receipt)


def test_exact_environment_rejects_allowed_name_with_changed_value():
    process, receipt = _gateway_environment_evidence()
    receipt["effective_environment_variable_value_sha256"]["HOME"] = "f" * 64

    with pytest.raises(RuntimeError, match="authority drifted|value drifted"):
        collector._validate_exact_runtime_environment(
            process,
            receipt,
            fixed_values=collector._FIXED_GATEWAY_ENVIRONMENT,
        )


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("getuid", "getgid", "fchown")),
    reason="POSIX ownership primitives are required",
)
def test_atomic_receipt_is_owner_only_canonical_json(tmp_path, monkeypatch):
    directory = tmp_path / "receipt-dir"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    target = directory / "receipt.json"
    monkeypatch.setattr(collector, "_validate_parent_chain", lambda *_args, **_kwargs: None)

    directory_stat = directory.stat()
    uid = directory_stat.st_uid
    gid = directory_stat.st_gid
    collector._atomic_write_json(
        target,
        {"z": 1, "a": True},
        owner_uid=uid,
        owner_gid=gid,
    )

    result = target.stat()
    assert result.st_nlink == 1
    assert result.st_mode & 0o777 == 0o400
    assert target.read_text(encoding="utf-8") == '{"a":true,"z":1}'
    assert list(directory.iterdir()) == [target]


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("getuid", "getgid", "link")),
    reason="POSIX ownership primitives are required",
)
def test_trusted_json_rejects_duplicate_keys_and_hardlinks(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_text('{"a":1,"a":2}', encoding="utf-8")
    source.chmod(0o400)
    source_stat = source.stat()
    uid = source_stat.st_uid
    gid = source_stat.st_gid
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        collector._read_trusted_json(
            source,
            expected_uid=uid,
            expected_gid=gid,
            require_trusted_parents=False,
        )

    source.chmod(0o400)
    linked = tmp_path / "manifest-hardlink.json"
    os.link(source, linked)
    with pytest.raises(ValueError, match="ownership or mode"):
        collector._read_trusted_json(
            source,
            expected_uid=uid,
            expected_gid=gid,
            require_trusted_parents=False,
        )
