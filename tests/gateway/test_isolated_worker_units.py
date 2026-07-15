from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

import gateway.isolated_worker_service as worker_service
from gateway.isolated_worker import PROTOCOL, canonical_bytes
from gateway.isolated_worker_service import CONFIG_SCHEMA
from gateway.isolated_worker_units import (
    BWRAP_PATH,
    GATEWAY_READY_PROBE_CONTRACT,
    ISOLATED_WORKER_CLIENT_GROUP,
    ISOLATED_WORKER_CONFIG,
    ISOLATED_WORKER_GROUP,
    ISOLATED_WORKER_LEASE_BASE,
    ISOLATED_WORKER_SERVICE_UNIT,
    ISOLATED_WORKER_SOCKET,
    ISOLATED_WORKER_SOCKET_UNIT,
    ISOLATED_WORKER_USER,
    LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION,
    LEASE_TMPFS_PREFLIGHT_CONTRACT,
    SHELL_PATH,
    UNIT_BUNDLE_SCHEMA,
    IsolatedWorkerUnitError,
    render_isolated_worker_units,
)


REVISION = "a" * 40
BWRAP_SHA256 = "b" * 64
SHELL_SHA256 = "c" * 64


def _arguments() -> dict:
    return {
        "revision": REVISION,
        "gateway_uid": 2001,
        "gateway_primary_gid": 2002,
        "socket_root_uid": 0,
        "socket_client_group": ISOLATED_WORKER_CLIENT_GROUP,
        "socket_client_gid": 2003,
        "worker_user": ISOLATED_WORKER_USER,
        "worker_group": ISOLATED_WORKER_GROUP,
        "worker_uid": 2004,
        "worker_gid": 2005,
        "bwrap_sha256": BWRAP_SHA256,
        "shell_sha256": SHELL_SHA256,
    }


def _bundle(**overrides):
    return render_isolated_worker_units(**{**_arguments(), **overrides})


def test_config_is_canonical_self_digested_and_matches_service_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    value = json.loads(bundle.config.decode("ascii"))
    assert canonical_bytes(value) == bundle.config
    assert value["schema"] == CONFIG_SCHEMA
    assert value["protocol"] == PROTOCOL
    assert value["listener_path"] == str(ISOLATED_WORKER_SOCKET)
    assert value["lease_base"] == str(ISOLATED_WORKER_LEASE_BASE)
    assert value["read_only_binds"] == []
    assert value["network_isolated"] is True
    assert value["limits"]["global_quota_bytes"] == 4_294_967_296
    assert value["limits"]["global_quota_entries"] == 200_000
    assert value["bwrap"] == {
        "path": str(BWRAP_PATH),
        "sha256": BWRAP_SHA256,
        "uid": 0,
    }
    assert value["shell"] == {
        "path": str(SHELL_PATH),
        "sha256": SHELL_SHA256,
        "uid": 0,
    }
    unsigned = dict(value)
    embedded_digest = unsigned.pop("config_sha256")
    assert embedded_digest == hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    assert embedded_digest == bundle.config_policy_sha256
    assert bundle.config_sha256 == hashlib.sha256(bundle.config).hexdigest()

    captured: dict = {}

    class CapturedPolicy:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(worker_service, "WorkerPolicy", CapturedPolicy)
    parsed = worker_service.parse_service_config(bundle.config)
    assert parsed.listener_path == ISOLATED_WORKER_SOCKET
    assert captured["expected_peer_uid"] == 2001
    assert captured["expected_peer_gid"] == 2002
    assert captured["socket_uid"] == 0
    assert captured["socket_gid"] == 2003
    assert captured["lease_base"] == ISOLATED_WORKER_LEASE_BASE
    assert captured["lease_uid"] == 2004
    assert captured["lease_gid"] == 2005
    assert captured["read_only_binds"] == ()
    assert captured["bwrap_path"] == BWRAP_PATH
    assert captured["bwrap_sha256"] == BWRAP_SHA256
    assert captured["shell"] == SHELL_PATH
    assert captured["shell_sha256"] == SHELL_SHA256


def test_socket_unit_pins_activation_identity_and_bounded_backlog() -> None:
    unit = _bundle().socket_unit.decode("ascii")
    assert f"ListenStream={ISOLATED_WORKER_SOCKET}\n" in unit
    assert f"Service={ISOLATED_WORKER_SERVICE_UNIT}\n" in unit
    assert "SocketUser=root\n" in unit
    assert f"SocketGroup={ISOLATED_WORKER_CLIENT_GROUP}\n" in unit
    assert "SocketMode=0660\n" in unit
    assert "DirectoryMode=0711\n" in unit
    assert "Accept=no\n" in unit
    assert "FileDescriptorName=isolated-worker\n" in unit
    assert "Backlog=128\n" in unit
    assert "RemoveOnStop=yes\n" in unit
    assert unit.count("ListenStream=") == 1
    assert "ListenDatagram=" not in unit
    assert "ListenFIFO=" not in unit


def test_service_is_release_local_unprivileged_offline_and_hardened() -> None:
    bundle = _bundle()
    unit = bundle.service_unit.decode("ascii")
    exact_exec = (
        f"ExecStart={bundle.interpreter} -B -P -s -m "
        f"gateway.isolated_worker_service --config {ISOLATED_WORKER_CONFIG}\n"
    )
    assert exact_exec in unit
    assert bundle.interpreter == (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{REVISION[:12]}"
        / ".venv/bin/python"
    )
    assert f"User={ISOLATED_WORKER_USER}\n" in unit
    assert f"Group={ISOLATED_WORKER_GROUP}\n" in unit
    assert "# PrincipalUID=2004\n" in unit
    assert "# PrincipalGID=2005\n" in unit
    assert f"AssertPathIsDirectory={ISOLATED_WORKER_LEASE_BASE}\n" in unit
    assert (
        f"TemporaryFileSystem={ISOLATED_WORKER_LEASE_BASE}:"
        "size=4294967296,nr_inodes=200001,"
        "mode=0700,uid=2004,gid=2005,nodev,nosuid,exec\n"
    ) in unit
    assert "StateDirectory=" not in unit
    assert "StateDirectoryQuota=" not in unit
    assert f"ReadWritePaths={ISOLATED_WORKER_LEASE_BASE}\n" not in unit
    assert "KillMode=control-group\n" in unit
    assert unit.count("KillMode=") == 1
    assert "KillMode=mixed\n" not in unit
    assert "PrivateNetwork=yes\n" in unit
    assert "RestrictAddressFamilies=AF_UNIX\n" in unit
    assert "IPAddressDeny=any\n" in unit
    assert "NoNewPrivileges=yes\n" in unit
    assert "CapabilityBoundingSet=\n" in unit
    assert "AmbientCapabilities=\n" in unit
    assert "ProtectSystem=strict\n" in unit
    assert "ProtectHome=yes\n" in unit
    assert "PrivateDevices=yes\n" in unit
    assert "PrivateTmp=yes\n" in unit
    assert "TasksMax=512\n" in unit
    assert "MemoryMax=2684354560\n" in unit
    assert "LimitFSIZE=4294967296\n" in unit
    assert "Environment=PATH=/run/muncho-no-path-fallback\n" in unit
    assert "Environment=HOME=/var/empty\n" in unit
    assert f"BindReadOnlyPaths={bundle.release_root}\n" in unit
    assert f"ReadOnlyPaths={ISOLATED_WORKER_CONFIG}\n" in unit
    assert "-/run/credentials" in unit
    assert "InaccessiblePaths=-/opt/adventico-ai-platform/hermes-home\n" in unit
    assert (
        "# GatewayReadyIntegration="
        f"required:{GATEWAY_READY_PROBE_CONTRACT}\n"
    ) in unit
    assert (
        "# LeaseFilesystemPreflight="
        f"required:{LEASE_TMPFS_PREFLIGHT_CONTRACT}\n"
    ) in unit

    lowered = unit.lower()
    assert "docker" not in lowered
    assert "network-online" not in lowered
    assert "loadcredential=" not in lowered
    assert "environmentfile=" not in lowered
    assert "passenvironment=" not in lowered
    assert "supplementarygroups=" not in lowered
    assert "privateNetwork=no".lower() not in lowered
    assert "ipaddressallow=" not in lowered
    assert "execstart=/usr/bin/env" not in lowered
    assert "environment=path=/usr/bin:/bin" not in lowered


def test_bundle_is_immutable_and_every_artifact_is_digest_bound() -> None:
    bundle = _bundle()
    assert bundle.config_owner_uid == 0
    assert bundle.config_owner_gid == 2005
    assert bundle.config_mode == 0o440
    assert bundle.lease_owner_uid == 2004
    assert bundle.lease_owner_gid == bundle.config_owner_gid == 2005
    assert bundle.lease_mode == 0o700
    artifacts = bundle.artifacts()
    digests = bundle.artifact_sha256()
    assert set(artifacts) == {
        str(ISOLATED_WORKER_CONFIG),
        ISOLATED_WORKER_SOCKET_UNIT,
        ISOLATED_WORKER_SERVICE_UNIT,
    }
    for name, payload in artifacts.items():
        assert digests[name] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(TypeError):
        artifacts["extra"] = b"forbidden"
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.revision = "d" * 40

    manifest = dict(bundle.manifest())
    assert manifest["schema"] == UNIT_BUNDLE_SCHEMA
    assert manifest.pop("bundle_sha256") == bundle.bundle_sha256
    expected = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert bundle.bundle_sha256 == hashlib.sha256(expected).hexdigest()
    assert manifest["secret_material_recorded"] is False
    assert manifest["secret_digest_recorded"] is False
    assert manifest["config_install"] == {
        "owner_uid": 0,
        "owner_gid": 2005,
        "mode": "0440",
    }
    assert manifest["lease_mountpoint_install"] == {
        "owner_uid": 0,
        "owner_gid": 0,
        "mode": "0700",
    }
    assert manifest["lease_runtime_filesystem"] == {
        "type": "tmpfs",
        "ephemeral_across_service_restart": True,
        "bytes": 4_294_967_296,
        "inodes": 200_001,
        "runtime_entry_limit": 200_000,
        "owner_uid": 2004,
        "owner_gid": 2005,
        "mode": "0700",
        "mount_flags": ["nodev", "nosuid", "exec"],
        "kernel_enforced": True,
        "host_preflight_required": True,
        "host_preflight_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
        "minimum_systemd_version": LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION,
    }
    assert manifest["gateway_ready_integration"] == {
        "required": True,
        "unit_ordering_sufficient": False,
        "probe_contract": GATEWAY_READY_PROBE_CONTRACT,
        "required_host_preflight_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"revision": "A" * 40}, "revision"),
        ({"revision": "a" * 39}, "revision"),
        ({"gateway_uid": True}, "gateway uid"),
        ({"gateway_primary_gid": 0}, "gateway primary gid"),
        ({"socket_root_uid": 1}, "root"),
        ({"socket_client_group": "worker-clients"}, "exact production identity"),
        ({"socket_client_gid": -1}, "socket client group gid"),
        ({"worker_user": "root"}, "exact production identity"),
        ({"worker_group": "docker"}, "exact production identity"),
        ({"worker_uid": 0}, "isolated worker uid"),
        ({"worker_gid": True}, "isolated worker gid"),
        ({"worker_uid": 2001}, "UIDs must be distinct"),
        ({"worker_gid": 2002}, "GIDs must be distinct"),
        ({"socket_client_gid": 2005}, "GIDs must be distinct"),
        ({"bwrap_sha256": "B" * 64}, "bwrap digest"),
        ({"shell_sha256": "0" * 63}, "shell digest"),
        ({"shell_sha256": BWRAP_SHA256}, "digests must be distinct"),
    ],
)
def test_renderer_rejects_adversarial_identity_and_digest_inputs(
    override: dict,
    message: str,
) -> None:
    with pytest.raises(IsolatedWorkerUnitError, match=message):
        _bundle(**override)
