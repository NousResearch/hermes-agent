from __future__ import annotations

import copy
import hashlib
import inspect
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_capability_canary_e2e as canary_e2e
from gateway import production_capability_prerequisites as runtime


REVISION = "a1234567890bcdef1234567890abcdef12345678"
NOW = 1_800_000_000


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _topology(revision: str = REVISION) -> dict:
    return {
        "schema": runtime.TOPOLOGY_SCHEMA,
        "prerequisite_receipt_path": str(runtime.PREREQUISITE_PATH),
        "collector_contract_sha256": runtime.packaged_prerequisite_contract_sha256(),
        "isolated_worker": {
            "socket_unit": runtime.ISOLATED_WORKER_SOCKET_UNIT,
            "socket_fragment_sha256": "1" * 64,
            "service_unit": runtime.ISOLATED_WORKER_SERVICE_UNIT,
            "service_fragment_sha256": "2" * 64,
            "config_path": str(runtime.ISOLATED_WORKER_CONFIG),
            "config_sha256": "3" * 64,
            "socket_path": str(runtime.ISOLATED_WORKER_SOCKET),
            "socket_uid": 0,
            "socket_gid": 992,
            "server_uid": 997,
            "server_gid": 996,
            "gateway_uid": 995,
            "gateway_gid": 991,
            "bwrap_path": str(runtime.BWRAP_PATH),
            "bwrap_sha256": "4" * 64,
            "shell_path": str(runtime.SHELL_PATH),
            "shell_sha256": "5" * 64,
        },
        "browser": {
            "unit": runtime.BROWSER_UNIT,
            "fragment_sha256": "6" * 64,
            "config_path": str(runtime.BROWSER_CONFIG_PATH),
            "config_sha256": "7" * 64,
            "socket_path": str(runtime.BROWSER_SOCKET_PATH),
            "service_uid": 998,
            "service_gid": 995,
            "node_path": str(runtime.production_browser_node(revision)),
            "node_sha256": "8" * 64,
            "wrapper_path": str(runtime.production_browser_wrapper(revision)),
            "wrapper_sha256": "9" * 64,
            "native_path": str(runtime.production_browser_native(revision)),
            "native_sha256": "a" * 64,
            "executable": str(runtime.production_browser_executable(revision)),
            "executable_sha256": "b" * 64,
            "agent_browser_config_path": str(
                runtime.production_agent_browser_config(revision)
            ),
            "agent_browser_config_sha256": "c" * 64,
        },
        "mac_ops": {
            "unit": runtime.MAC_OPS_UNIT,
            "fragment_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "config_path": str(runtime.MAC_OPS_CONFIG_PATH),
            "socket_path": str(runtime.MAC_OPS_SOCKET_PATH),
            "credential_path": str(runtime.MAC_OPS_CREDENTIAL_PATH),
            "journal_path": str(runtime.MAC_OPS_JOURNAL_PATH),
        },
        "routeback_edge": {
            "unit": runtime.ROUTEBACK_EDGE_UNIT,
            "fragment_sha256": "e" * 64,
            "config_sha256": "f" * 64,
            "config_path": str(runtime.ROUTEBACK_EDGE_CONFIG_PATH),
            "socket_path": str(runtime.ROUTEBACK_EDGE_SOCKET_PATH),
            "credential_path": str(runtime.ROUTEBACK_EDGE_CREDENTIAL_PATH),
            "readiness_path": str(runtime.ROUTEBACK_EDGE_READINESS_PATH),
        },
        "public_connector": {
            "unit": runtime.PUBLIC_CONNECTOR_UNIT,
            "fragment_sha256": "f" * 64,
            "config_path": str(runtime.PUBLIC_CONNECTOR_CONFIG_PATH),
            "socket_path": str(runtime.PUBLIC_CONNECTOR_SOCKET_PATH),
            "credential_path": str(runtime.PUBLIC_CONNECTOR_CREDENTIAL_PATH),
            "readiness_path": str(runtime.PUBLIC_CONNECTOR_READINESS_PATH),
        },
        "phase_b": {
            "unit": runtime.PHASE_B_UNIT,
            "fragment_sha256": "0" * 64,
            "readiness_path": str(runtime.PHASE_B_RECEIPT_PATH),
        },
        "codex_auth_file": str(runtime.CODEX_AUTH_PATH),
        "api_control_credential_file": str(runtime.API_SERVER_CREDENTIAL_PATH),
        "api_approval_credential_file": str(runtime.API_APPROVAL_CREDENTIAL_PATH),
        "gateway_identity": {"uid": 995, "gid": 991},
    }


def _service(
    topology: dict,
    name: str,
    unit: str,
    *,
    service_type: str,
    sub_state: str,
    main_pid: int,
) -> dict:
    identities = {
        "isolated_worker": ("muncho-worker", 997, "muncho-worker", 996),
        "browser": ("muncho-browser", 998, "muncho-browser", 995),
    }
    persistent = main_pid > 0
    if persistent:
        user, uid, group, gid = identities.get(
            name, ("muncho-test", 993, "muncho-test", 994)
        )
    else:
        user, uid, group, gid = ("root", 0, "root", 0)
    executable = "/usr/bin/true"
    cmdline_sha256 = hashlib.sha256(b"/usr/bin/true\x00").hexdigest()
    contract = {
        "effective_user": user,
        "effective_uid": uid,
        "effective_group": group,
        "effective_gid": gid,
        "effective_supplementary_groups": [gid],
        "unit_executable": executable,
        "unit_cmdline_sha256": cmdline_sha256,
    }
    fragment_sha256 = (
        topology["isolated_worker"]["service_fragment_sha256"]
        if name == "isolated_worker"
        else topology[name]["fragment_sha256"]
    )
    return {
        "unit": unit,
        "fragment_path": f"/etc/systemd/system/{unit}",
        "fragment_sha256": fragment_sha256,
        "unit_file_state": "static" if name == "isolated_worker" else "disabled",
        "active_state": "active",
        "sub_state": sub_state,
        "service_type": service_type,
        "main_pid": main_pid,
        "drop_in_paths": [],
        "need_daemon_reload": False,
        **contract,
        "unit_service_contract_sha256": hashlib.sha256(
            _canonical(contract)
        ).hexdigest(),
        "main_pid_executable": executable if persistent else None,
        "main_pid_uid": uid if persistent else None,
        "main_pid_gid": gid if persistent else None,
        "main_pid_groups": [gid] if persistent else None,
        "main_pid_cmdline_sha256": cmdline_sha256 if persistent else None,
        "main_pid_cgroup": f"/system.slice/{unit}" if persistent else None,
        "main_pid_mount_namespace_inode": 1001 if persistent else None,
        "main_pid_network_namespace_inode": 1002 if persistent else None,
        "process_identity_matches_unit": True,
        "readiness_receipt_sha256": (
            hashlib.sha256(f"{name}-ready".encode()).hexdigest()
            if name in {"phase_b", "routeback_edge", "public_connector"}
            else None
        ),
        "ready": True,
    }


def _services(
    topology: dict,
    *,
    lifecycle_phase: str = runtime.PREREQUISITE_LIFECYCLE_STAGED,
) -> dict:
    services = {
        "phase_b": _service(
            topology,
            "phase_b",
            runtime.PHASE_B_UNIT,
            service_type="oneshot",
            sub_state="exited",
            main_pid=0,
        ),
        "routeback_edge": _service(
            topology,
            "routeback_edge",
            runtime.ROUTEBACK_EDGE_UNIT,
            service_type="notify",
            sub_state="running",
            main_pid=101,
        ),
        "public_connector": _service(
            topology,
            "public_connector",
            runtime.PUBLIC_CONNECTOR_UNIT,
            service_type="notify",
            sub_state="running",
            main_pid=102,
        ),
        "mac_ops": _service(
            topology,
            "mac_ops",
            runtime.MAC_OPS_UNIT,
            service_type="simple",
            sub_state="running",
            main_pid=103,
        ),
        "isolated_worker": _service(
            topology,
            "isolated_worker",
            runtime.ISOLATED_WORKER_SERVICE_UNIT,
            service_type="simple",
            sub_state="running",
            main_pid=104,
        ),
        "browser": _service(
            topology,
            "browser",
            runtime.BROWSER_UNIT,
            service_type="notify",
            sub_state="running",
            main_pid=105,
        ),
    }
    boot_unit_file_state = (
        "disabled"
        if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
        else "enabled"
    )
    for name, service in services.items():
        if name != "isolated_worker":
            service["unit_file_state"] = boot_unit_file_state
    return services


def _socket(
    path: Path,
    *,
    device: int,
    inode: int,
    owner_uid: int,
    group_gid: int,
    main_pid: int,
) -> dict:
    return {
        "path": str(path),
        "device": device,
        "inode": inode,
        "owner_uid": owner_uid,
        "group_gid": group_gid,
        "mode": "0660",
        "main_pid": main_pid,
        "ready": True,
    }


def _sockets() -> dict:
    return {
        "routeback_edge": _socket(
            runtime.ROUTEBACK_EDGE_SOCKET_PATH,
            device=10,
            inode=11,
            owner_uid=993,
            group_gid=994,
            main_pid=101,
        ),
        "public_connector": _socket(
            runtime.PUBLIC_CONNECTOR_SOCKET_PATH,
            device=10,
            inode=12,
            owner_uid=993,
            group_gid=994,
            main_pid=102,
        ),
        "mac_ops": _socket(
            runtime.MAC_OPS_SOCKET_PATH,
            device=10,
            inode=13,
            owner_uid=993,
            group_gid=994,
            main_pid=103,
        ),
        "isolated_worker": _socket(
            runtime.ISOLATED_WORKER_SOCKET,
            device=10,
            inode=14,
            owner_uid=0,
            group_gid=992,
            main_pid=104,
        ),
        "browser": _socket(
            runtime.BROWSER_SOCKET_PATH,
            device=10,
            inode=15,
            owner_uid=998,
            group_gid=995,
            main_pid=105,
        ),
    }


def _lease(path: Path, *, owner_uid: int, owner_gid: int, refresh: bool) -> dict:
    return {
        "path": str(path),
        "owner_uid": owner_uid,
        "group_gid": owner_gid,
        "mode": "0400",
        "size": 128,
        "regular_one_link": True,
        "usable": True,
        "refresh_capable": refresh,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _runtime_dependencies(topology: dict, revision: str = REVISION) -> dict:
    browser = topology["browser"]
    return {
        "manifest_path": str(
            runtime.production_release_root(revision)
            / runtime.RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
        ),
        "manifest_sha256": "d" * 64,
        "agent_browser": {
            "version": runtime.AGENT_BROWSER_VERSION,
            "config_path": browser["agent_browser_config_path"],
            "config_sha256": browser["agent_browser_config_sha256"],
            "wrapper_path": browser["wrapper_path"],
            "wrapper_sha256": browser["wrapper_sha256"],
            "native_path": browser["native_path"],
            "native_sha256": browser["native_sha256"],
            "node_path": browser["node_path"],
            "node_version": runtime.NODE_VERSION,
            "node_sha256": browser["node_sha256"],
        },
        "chrome": {
            "version": runtime.CHROME_VERSION,
            "executable_path": browser["executable"],
            "executable_sha256": browser["executable_sha256"],
        },
        "ddgs": {
            "version": runtime.DDGS_VERSION,
            "files_sha256": "e" * 64,
            "gateway_uid_import_smoke": True,
        },
        "ready": True,
    }


def _gateway_state(topology: dict, revision: str = REVISION) -> dict:
    gateway = topology["gateway_identity"]
    unsigned = {
        "schema": "muncho-production-gateway-state-proof.v1",
        "gateway_uid": gateway["uid"],
        "gateway_gid": gateway["gid"],
        "hermes_home": str(runtime.PRODUCTION_HOME),
        "config": {
            "path": str(runtime.PRODUCTION_CONFIG_PATH),
            "memory_enabled": True,
            "user_profile_enabled": True,
            "readable": True,
        },
        "memory": {
            "home": str(runtime.PRODUCTION_HOME / "memories"),
            "memory": {
                "path": str(runtime.PRODUCTION_HOME / "memories/MEMORY.md"),
                "exists": True,
                "size": 128,
                "owner_uid": gateway["uid"],
                "group_gid": gateway["gid"],
                "mode": "0600",
                "readable": True,
            },
            "user": {
                "path": str(runtime.PRODUCTION_HOME / "memories/USER.md"),
                "exists": True,
                "size": 64,
                "owner_uid": gateway["uid"],
                "group_gid": gateway["gid"],
                "mode": "0600",
                "readable": True,
            },
            "built_in_load": True,
            "built_in_atomic_create_rewrite": True,
        },
        "skills": {
            "bundled_path": str(runtime.production_release_root(revision) / "skills"),
            "bundled_count": 12,
            "bundled_index_sha256": "f" * 64,
            "user_path": str(runtime.PRODUCTION_HOME / "skills"),
            "user_atomic_roundtrip": True,
        },
        "session_db": {
            "path": str(runtime.PRODUCTION_HOME / "state.db"),
            "journal_mode": "wal",
            "fts5_enabled": True,
            "real_fts_query": True,
            "owner_uid": gateway["uid"],
            "group_gid": gateway["gid"],
            "mode": "0600",
        },
        "state_directory": {
            "path": str(runtime.GATEWAY_STATE_DIRECTORY),
            "owner_uid": gateway["uid"],
            "group_gid": gateway["gid"],
            "mode": "0700",
            "atomic_roundtrip": True,
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "proof_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _execution_proofs(topology: dict) -> dict:
    worker = topology["isolated_worker"]
    browser = topology["browser"]
    return {
        "isolated_worker_exec": {
            "schema": runtime.WORKER_RECEIPT_SCHEMA,
            "lease_identity_sha256": "1" * 64,
            "socket_path": worker["socket_path"],
            "server_uid": worker["server_uid"],
            "server_gid": worker["server_gid"],
            "socket_uid": worker["socket_uid"],
            "socket_gid": worker["socket_gid"],
            "execution_round_trip": True,
            "output_sha256": hashlib.sha256(
                b"MUNCHO_ISOLATED_WORKER_READY\n"
            ).hexdigest(),
            "secret_material_recorded": False,
        },
        "browser_controller_command": {
            "schema": runtime.BROWSER_RECEIPT_SCHEMA,
            "session_identity_sha256": "2" * 64,
            "socket_path": browser["socket_path"],
            "server_uid": browser["service_uid"],
            "command_round_trip": True,
            "secret_material_recorded": False,
        },
    }


def _receipt_pair(
    *,
    revision: str = REVISION,
    now_unix: int = NOW,
    lifecycle_phase: str = runtime.PREREQUISITE_LIFECYCLE_STAGED,
) -> tuple[dict, dict, bytes]:
    topology = _topology(revision)
    services = _services(topology, lifecycle_phase=lifecycle_phase)
    sockets = _sockets()
    worker = topology["isolated_worker"]
    browser = topology["browser"]
    execution = _execution_proofs(topology)
    boot_id = b"boot-id-for-focused-test"
    unsigned = {
        "schema": runtime.PREREQUISITE_SCHEMA,
        "release_revision": revision,
        "lifecycle_phase": lifecycle_phase,
        "topology_identity_sha256": (
            runtime.production_capability_topology_identity_sha256(topology)
        ),
        "boot_id_sha256": hashlib.sha256(boot_id).hexdigest(),
        "observed_at_unix": now_unix,
        "services": services,
        "sockets": sockets,
        "isolated_worker": {
            "socket_unit": worker["socket_unit"],
            "socket_fragment_path": (
                f"/etc/systemd/system/{runtime.ISOLATED_WORKER_SOCKET_UNIT}"
            ),
            "socket_fragment_sha256": worker["socket_fragment_sha256"],
            "socket_unit_file_state": (
                "disabled"
                if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
                else "enabled"
            ),
            "socket_active_state": "active",
            "socket_sub_state": "listening",
            "socket_drop_in_paths": [],
            "socket_need_daemon_reload": False,
            "service_unit": worker["service_unit"],
            "service_main_pid": services["isolated_worker"]["main_pid"],
            "config_path": worker["config_path"],
            "config_sha256": worker["config_sha256"],
            "config_uid": 0,
            "config_gid": worker["server_gid"],
            "config_mode": "0440",
            "socket_path": worker["socket_path"],
            "socket_uid": worker["socket_uid"],
            "socket_gid": worker["socket_gid"],
            "socket_device": sockets["isolated_worker"]["device"],
            "socket_inode": sockets["isolated_worker"]["inode"],
            "socket_mode": sockets["isolated_worker"]["mode"],
            "bwrap_path": worker["bwrap_path"],
            "bwrap_sha256": worker["bwrap_sha256"],
            "shell_path": worker["shell_path"],
            "shell_sha256": worker["shell_sha256"],
            "ready": True,
        },
        "browser": {
            **browser,
            "config_uid": 0,
            "config_gid": browser["service_gid"],
            "config_mode": "0440",
            "socket_uid": browser["service_uid"],
            "socket_gid": browser["service_gid"],
            "socket_device": sockets["browser"]["device"],
            "socket_inode": sockets["browser"]["inode"],
            "socket_mode": sockets["browser"]["mode"],
            "service_main_pid": services["browser"]["main_pid"],
            "ready": True,
        },
        "runtime_dependencies": _runtime_dependencies(topology, revision),
        "gateway_state": _gateway_state(topology, revision),
        "capability_proofs": {
            "mac_ops_ping": {
                "main_pid": services["mac_ops"]["main_pid"],
                "service_identity_sha256": "3" * 64,
                "receipt_sha256": "4" * 64,
                "peer_main_pid_validated": True,
                "external_io": False,
                "ready": True,
            },
            **execution,
        },
        "credentials": {
            "api_control": _lease(
                runtime.API_SERVER_CREDENTIAL_PATH,
                owner_uid=0,
                owner_gid=0,
                refresh=False,
            ),
            "api_approval": _lease(
                runtime.API_APPROVAL_CREDENTIAL_PATH,
                owner_uid=0,
                owner_gid=0,
                refresh=False,
            ),
            "openai_codex": _lease(
                runtime.CODEX_AUTH_PATH,
                owner_uid=995,
                owner_gid=991,
                refresh=True,
            ),
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    return topology, receipt, boot_id


def _rehash(receipt: dict) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()


def test_api_credential_observation_requires_canonical_verifier_bytes(
    tmp_path: Path,
) -> None:
    from gateway.api_verifier_credentials import (
        build_api_approval_scrypt_verifier,
        build_api_bearer_verifier,
    )

    bearer = tmp_path / "bearer.json"
    approval = tmp_path / "approval.json"
    bearer.write_bytes(build_api_bearer_verifier("b" * 64))
    approval.write_bytes(build_api_approval_scrypt_verifier("p" * 64))
    assert (
        runtime._credential_observation(
            runtime.API_SERVER_CREDENTIAL_PATH,
            bearer,
            refresh_capable=False,
            verifier_kind="bearer",
        )["secret_digest_recorded"]
        is False
    )
    assert (
        runtime._credential_observation(
            runtime.API_APPROVAL_CREDENTIAL_PATH,
            approval,
            refresh_capable=False,
            verifier_kind="approval",
        )["secret_material_recorded"]
        is False
    )
    bearer.write_bytes(b'{"schema":"not-a-verifier"}')
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="production_live_api_verifier_invalid",
    ):
        runtime._credential_observation(
            runtime.API_SERVER_CREDENTIAL_PATH,
            bearer,
            refresh_capable=False,
            verifier_kind="bearer",
        )


def _systemd_values(unit: str) -> dict[str, str]:
    return {
        "ActiveState": "active",
        "SubState": "running",
        "Type": "simple",
        "MainPID": "123",
        "FragmentPath": f"/etc/systemd/system/{unit}",
        "UnitFileState": "disabled",
        "DropInPaths": "",
        "NeedDaemonReload": "no",
        "User": "muncho-test",
        "Group": "muncho-test",
        "SupplementaryGroups": "",
        "ControlGroup": f"/system.slice/{unit}",
    }


def _mock_service_observation_dependencies(
    monkeypatch: pytest.MonkeyPatch, *, process_matches: bool = True
) -> None:
    contract = {
        "effective_user": "muncho-test",
        "effective_uid": 995,
        "effective_group": "muncho-test",
        "effective_gid": 991,
        "effective_supplementary_groups": [991],
        "unit_executable": "/usr/bin/true",
        "unit_cmdline_sha256": hashlib.sha256(b"/usr/bin/true\x00").hexdigest(),
    }
    contract["unit_service_contract_sha256"] = hashlib.sha256(
        _canonical(contract)
    ).hexdigest()
    process = {
        "main_pid_executable": "/usr/bin/true",
        "main_pid_uid": 995,
        "main_pid_gid": 991,
        "main_pid_groups": [991],
        "main_pid_cmdline_sha256": contract["unit_cmdline_sha256"],
        "main_pid_cgroup": f"/system.slice/{runtime.BROWSER_UNIT}",
        "main_pid_mount_namespace_inode": 1001,
        "main_pid_network_namespace_inode": 1002,
        "process_identity_matches_unit": process_matches,
    }
    monkeypatch.setattr(
        runtime,
        "_read_bounded_regular",
        lambda _path, maximum: (
            b"[Service]\nUser=muncho-test\nGroup=muncho-test\n"
            b"ExecStart=/usr/bin/true\n",
            object(),
        ),
    )
    monkeypatch.setattr(
        runtime, "_service_identity_from_contract", lambda **_: contract
    )
    monkeypatch.setattr(runtime, "_service_process_identity", lambda **_: process)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DropInPaths", "/etc/systemd/system/browser.service.d/override.conf"),
        ("NeedDaemonReload", "yes"),
    ],
)
def test_live_service_observation_rejects_override_or_unloaded_change(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    _mock_service_observation_dependencies(monkeypatch)
    values = _systemd_values(runtime.BROWSER_UNIT)
    values[field] = value
    monkeypatch.setattr(runtime, "_systemd_show_service", lambda _unit: values)
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="effective_config_invalid",
    ):
        runtime._systemd_service_observation(
            "browser", runtime.BROWSER_UNIT, readiness_path=None
        )


def test_live_service_observation_rejects_process_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_service_observation_dependencies(monkeypatch, process_matches=False)
    values = _systemd_values(runtime.BROWSER_UNIT)
    monkeypatch.setattr(runtime, "_systemd_show_service", lambda _unit: values)
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="process_identity_invalid",
    ):
        runtime._systemd_service_observation(
            "browser", runtime.BROWSER_UNIT, readiness_path=None
        )


@pytest.mark.parametrize(
    ("lifecycle_phase", "unit_file_state"),
    [
        (runtime.PREREQUISITE_LIFECYCLE_STAGED, "disabled"),
        (runtime.PREREQUISITE_LIFECYCLE_COMMITTED, "enabled"),
    ],
)
def test_socket_unit_observation_binds_phase_and_rejects_override(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_phase: str,
    unit_file_state: str,
) -> None:
    values = {
        "ActiveState": "active",
        "SubState": "listening",
        "FragmentPath": f"/etc/systemd/system/{runtime.ISOLATED_WORKER_SOCKET_UNIT}",
        "UnitFileState": unit_file_state,
        "DropInPaths": "",
        "NeedDaemonReload": "no",
    }
    monkeypatch.setattr(runtime, "_systemd_show_socket_unit", lambda _unit: values)
    monkeypatch.setattr(
        runtime,
        "_read_bounded_regular",
        lambda _path, maximum: (b"exact socket unit\n", object()),
    )
    proof = runtime._systemd_socket_unit_observation(
        runtime.ISOLATED_WORKER_SOCKET_UNIT,
        lifecycle_phase=lifecycle_phase,
    )
    assert (
        proof["socket_fragment_sha256"]
        == hashlib.sha256(b"exact socket unit\n").hexdigest()
    )
    values["DropInPaths"] = "/etc/systemd/system/override.conf"
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="socket_unit_invalid",
    ):
        runtime._systemd_socket_unit_observation(
            runtime.ISOLATED_WORKER_SOCKET_UNIT,
            lifecycle_phase=lifecycle_phase,
        )


def _gateway_service_observation(expected_unit: bytes) -> dict:
    topology = _topology()
    observed = _service(
        topology,
        "browser",
        runtime.GATEWAY_UNIT,
        service_type="notify",
        sub_state="start",
        main_pid=4321,
    )
    observed.update({
        "fragment_path": f"/etc/systemd/system/{runtime.GATEWAY_UNIT}",
        "fragment_sha256": hashlib.sha256(expected_unit).hexdigest(),
        "unit_file_state": "enabled",
        "active_state": "activating",
        "effective_user": "muncho-test",
        "effective_uid": 995,
        "effective_group": "muncho-test",
        "effective_gid": 991,
        "effective_supplementary_groups": [991, 992],
        "main_pid_uid": 995,
        "main_pid_gid": 991,
        "main_pid_groups": [991, 992],
    })
    return observed


def test_gateway_service_identity_is_bound_to_pre_ready_main_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_unit = (
        b"[Service]\nUser=muncho-test\nGroup=muncho-test\n"
        b"SupplementaryGroups=muncho-client\n"
        b"ExecStart=/usr/bin/true --exact\n"
    )
    observed = _gateway_service_observation(expected_unit)
    monkeypatch.setattr(runtime.os, "getpid", lambda: 4321)
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 995)
    monkeypatch.setattr(runtime.os, "getegid", lambda: 991)
    monkeypatch.setattr(runtime.os, "getgroups", lambda: [992])
    monkeypatch.setattr(
        runtime, "_systemd_service_observation", lambda *_args, **_kwargs: observed
    )
    assert (
        runtime.attest_live_production_gateway_service_identity(
            expected_unit=expected_unit
        )
        == observed
    )


def test_exact_v3_topology_receipt_and_packaged_contract_are_accepted() -> None:
    topology, receipt, _boot_id = _receipt_pair()
    contract = runtime.packaged_prerequisite_contract()
    assert runtime.validate_production_capability_topology(topology) == topology
    assert (
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW,
        )
        == receipt
    )
    assert contract["isolated_worker_topology_fields"] == sorted(
        runtime._ISOLATED_WORKER_TOPOLOGY_FIELDS
    )
    assert contract["browser_topology_fields"] == sorted(
        runtime._BROWSER_TOPOLOGY_FIELDS
    )
    assert contract["isolated_worker_receipt_fields"] == sorted(
        runtime._ISOLATED_WORKER_RECEIPT_FIELDS
    )
    assert contract["browser_receipt_fields"] == sorted(runtime._BROWSER_RECEIPT_FIELDS)
    assert contract["lifecycle_phases"] == ["committed", "staged"]
    assert contract["isolated_canary_goal_terminal_schema"] == (
        canary_e2e.GOAL_CONTINUATION_TERMINAL_SCHEMA
    )
    assert "workspaces" not in contract["topology_fields"]
    assert "workspaces" not in contract["fields"]


@pytest.mark.parametrize(
    "lifecycle_phase",
    [
        runtime.PREREQUISITE_LIFECYCLE_STAGED,
        runtime.PREREQUISITE_LIFECYCLE_COMMITTED,
    ],
)
def test_lifecycle_phase_exactly_binds_every_boot_unit_state(
    lifecycle_phase: str,
) -> None:
    topology, receipt, _boot_id = _receipt_pair(lifecycle_phase=lifecycle_phase)
    expected = (
        "disabled"
        if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
        else "enabled"
    )
    assert {
        item["unit_file_state"]
        for name, item in receipt["services"].items()
        if name != "isolated_worker"
    } == {expected}
    assert receipt["services"]["isolated_worker"]["unit_file_state"] == "static"
    assert receipt["isolated_worker"]["socket_unit_file_state"] == expected
    assert (
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=lifecycle_phase,
            now_unix=NOW,
        )
        == receipt
    )
    opposite = (
        runtime.PREREQUISITE_LIFECYCLE_COMMITTED
        if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
        else runtime.PREREQUISITE_LIFECYCLE_STAGED
    )
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="lifecycle_phase_invalid",
    ):
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=opposite,
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    "lifecycle_phase",
    [
        runtime.PREREQUISITE_LIFECYCLE_STAGED,
        runtime.PREREQUISITE_LIFECYCLE_COMMITTED,
    ],
)
def test_lifecycle_phase_rejects_one_boot_unit_in_the_opposite_state(
    lifecycle_phase: str,
) -> None:
    topology, receipt, _boot_id = _receipt_pair(lifecycle_phase=lifecycle_phase)
    receipt["services"]["browser"]["unit_file_state"] = (
        "enabled"
        if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
        else "disabled"
    )
    _rehash(receipt)
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="production_prerequisite_service_invalid",
    ):
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=lifecycle_phase,
            now_unix=NOW,
        )

    topology, receipt, _boot_id = _receipt_pair(lifecycle_phase=lifecycle_phase)
    receipt["isolated_worker"]["socket_unit_file_state"] = (
        "enabled"
        if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
        else "disabled"
    )
    _rehash(receipt)
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="production_prerequisite_isolated_worker_invalid",
    ):
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=lifecycle_phase,
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda topology: topology.update({"docker": {}}),
        lambda topology: topology.update({"workspaces": {}}),
        lambda topology: topology["browser"].update({
            "cdp_url": "http://127.0.0.1:9222"
        }),
        lambda topology: topology["isolated_worker"].update({"gateway_uid": 994}),
        lambda topology: topology["isolated_worker"].update({"socket_uid": 1}),
        lambda topology: topology["isolated_worker"].update({"socket_gid": 991}),
        lambda topology: topology["browser"].update({"service_uid": 997}),
        lambda topology: topology["browser"].update({
            "wrapper_path": topology["browser"]["native_path"]
        }),
    ],
)
def test_topology_rejects_legacy_or_cross_boundary_fields(mutation) -> None:
    topology = _topology()
    mutation(topology)
    with pytest.raises(runtime.ProductionCapabilityPrerequisiteError):
        runtime.validate_production_capability_topology(topology)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update({"docker": {}}),
        lambda receipt: receipt.update({"workspaces": {}}),
        lambda receipt: receipt["browser"].update({"cdp_url": "http://127.0.0.1:9222"}),
        lambda receipt: receipt["capability_proofs"].update({
            "docker_gateway": {"ready": True}
        }),
    ],
)
def test_receipt_rejects_every_legacy_runtime_shape(mutation) -> None:
    topology, receipt, _boot_id = _receipt_pair()
    mutation(receipt)
    _rehash(receipt)
    with pytest.raises(runtime.ProductionCapabilityPrerequisiteError):
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda receipt: receipt["services"]["browser"].update({
                "active_state": "failed"
            }),
            "service_invalid",
        ),
        (
            lambda receipt: receipt["sockets"]["isolated_worker"].update({
                "owner_uid": 997
            }),
            "socket_invalid",
        ),
        (
            lambda receipt: receipt["isolated_worker"].update({
                "socket_fragment_sha256": "f" * 64
            }),
            "isolated_worker_invalid",
        ),
        (
            lambda receipt: receipt["browser"].update({"config_mode": "0640"}),
            "browser_invalid",
        ),
        (
            lambda receipt: receipt["capability_proofs"][
                "isolated_worker_exec"
            ].update({"execution_round_trip": False}),
            "isolated_worker_exec_proof_invalid",
        ),
        (
            lambda receipt: receipt["capability_proofs"][
                "browser_controller_command"
            ].update({"server_uid": 997}),
            "browser_controller_proof_invalid",
        ),
        (
            lambda receipt: receipt.update({"secret_material_recorded": True}),
            "identity_invalid",
        ),
    ],
)
def test_v3_boundary_and_secret_tampering_is_rejected(mutation, code: str) -> None:
    topology, receipt, _boot_id = _receipt_pair()
    mutation(receipt)
    _rehash(receipt)
    with pytest.raises(runtime.ProductionCapabilityPrerequisiteError, match=code):
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    "path",
    [
        ("services", "isolated_worker", "main_pid"),
        ("sockets", "browser", "inode"),
        ("isolated_worker", "socket_inode"),
        ("isolated_worker", "bwrap_sha256"),
        ("browser", "config_sha256"),
        ("runtime_dependencies", "agent_browser", "config_sha256"),
    ],
)
def test_post_collection_identity_race_is_rejected(path: tuple[str, ...]) -> None:
    topology, receipt, _boot_id = _receipt_pair()
    current = copy.deepcopy(receipt)
    parent = current
    for key in path[:-1]:
        parent = parent[key]
    value = parent[path[-1]]
    parent[path[-1]] = value + 1 if isinstance(value, int) else "f" * 64
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="production_live_.*_drifted",
    ):
        runtime.validate_live_production_capability_prerequisites(
            current,
            signed=receipt,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
        )


def test_execution_readiness_collector_runs_as_gateway_with_both_socket_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _topology()
    proofs = _execution_proofs(topology)
    captured: dict[str, object] = {}

    def run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout=_canonical(proofs) + b"\n")

    monkeypatch.setattr(runtime, "_run_gateway_process", run)
    assert (
        runtime._collect_execution_readiness_proofs(
            revision=REVISION, topology=topology
        )
        == proofs
    )
    assert captured["kwargs"]["extra_groups"] == (992, 995)
    assert captured["kwargs"]["timeout"] == 180
    command = " ".join(captured["arguments"])
    assert "attest_isolated_worker_execution" in command
    assert "attest_browser_controller_execution" in command
    assert str(runtime.BROWSER_ARTIFACT_PATH) in captured["arguments"]


def test_runtime_dependency_proof_uses_manifest_and_controller_owns_browser_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _topology()
    dependency = _runtime_dependencies(topology)
    manifest = {
        "agent_browser": copy.deepcopy(dependency["agent_browser"]),
        "chrome": copy.deepcopy(dependency["chrome"]),
        "python": {"distributions": {"ddgs": dependency["ddgs"]}},
    }
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"verified\n", stderr=b""
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_dependency_manifest",
        lambda **_kwargs: (manifest, b"manifest\n"),
    )

    def gateway_process(arguments, **_kwargs):
        calls.append(arguments)
        return SimpleNamespace(stdout=b'{"imported":true,"version":"9.14.4"}\n')

    monkeypatch.setattr(runtime, "_run_gateway_process", gateway_process)
    proof = runtime._collect_runtime_dependency_proof(
        revision=REVISION, topology=topology
    )
    assert (
        proof["agent_browser"]["config_sha256"]
        == topology["browser"]["agent_browser_config_sha256"]
    )
    assert proof["chrome"] == dependency["chrome"]
    assert len(calls) == 1
    assert "ddgs" in " ".join(calls[0])


def test_file_tamper_and_reboot_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, receipt, boot_id = _receipt_pair()
    payload = _canonical(receipt)
    current_boot = boot_id.decode("ascii")
    monkeypatch.setattr(runtime, "_read_boot_id", lambda: current_boot)
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime, "_read_stable_file", lambda _path: (payload, object()))
    monkeypatch.setattr(
        runtime,
        "collect_current_production_capability_prerequisite_receipt",
        lambda **_kwargs: receipt,
    )
    assert (
        runtime.load_production_capability_prerequisite_receipt(
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW,
        )
        == receipt
    )
    tampered = copy.deepcopy(receipt)
    tampered["browser"]["ready"] = False
    monkeypatch.setattr(
        runtime, "_read_stable_file", lambda _path: (_canonical(tampered), object())
    )
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError, match="identity_invalid"
    ):
        runtime.load_production_capability_prerequisite_receipt(
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW,
        )
    monkeypatch.setattr(runtime, "_read_stable_file", lambda _path: (payload, object()))
    current_boot = "different-boot"
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="boot_identity_drifted",
    ):
        runtime.load_production_capability_prerequisite_receipt(
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW,
        )


def test_stale_receipt_and_codex_owner_drift_are_rejected() -> None:
    topology, receipt, _boot_id = _receipt_pair()
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError, match="clock_invalid"
    ):
        runtime.validate_production_capability_prerequisite_receipt(
            receipt,
            revision=REVISION,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
            now_unix=NOW + runtime.MAX_PREREQUISITE_AGE_SECONDS + 1,
        )
    current = copy.deepcopy(receipt)
    current["credentials"]["openai_codex"]["owner_uid"] += 1
    with pytest.raises(
        runtime.ProductionCapabilityPrerequisiteError,
        match="codex_credential_owner_drifted",
    ):
        runtime.validate_live_production_capability_prerequisites(
            current,
            signed=receipt,
            topology=topology,
            lifecycle_phase=runtime.PREREQUISITE_LIFECYCLE_STAGED,
        )


def test_root_collector_atomically_installs_canonical_read_only_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _topology_value, receipt, _boot_id = _receipt_pair()
    target = tmp_path / "collector" / "prerequisite-receipt.json"
    target.parent.mkdir(mode=0o755)
    monkeypatch.setattr(runtime, "PREREQUISITE_PATH", target)
    monkeypatch.setattr(runtime, "sys_platform_is_linux_root", lambda: True)
    monkeypatch.setattr(
        runtime.os,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        ),
    )
    runtime._atomic_install_collected_receipt(receipt)
    assert target.read_bytes() == _canonical(receipt)
    assert target.stat().st_mode & 0o777 == 0o444


def test_root_collector_cli_requires_and_forwards_exact_lifecycle_phase(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def collect(**kwargs):
        captured.update(kwargs)
        return {
            "schema": runtime.PREREQUISITE_SCHEMA,
            "lifecycle_phase": kwargs["lifecycle_phase"],
        }

    monkeypatch.setattr(
        runtime,
        "collect_and_install_from_production_config",
        collect,
    )
    assert (
        runtime._main([
            "collect",
            "--revision",
            REVISION,
            "--config-sha256",
            "a" * 64,
            "--lifecycle-phase",
            runtime.PREREQUISITE_LIFECYCLE_COMMITTED,
        ])
        == 0
    )
    assert captured == {
        "revision": REVISION,
        "config_sha256": "a" * 64,
        "lifecycle_phase": runtime.PREREQUISITE_LIFECYCLE_COMMITTED,
    }
    assert json.loads(capsys.readouterr().out)["lifecycle_phase"] == "committed"
    with pytest.raises(SystemExit):
        runtime._main([
            "collect",
            "--revision",
            REVISION,
            "--config-sha256",
            "a" * 64,
        ])


@pytest.mark.parametrize(
    "lifecycle_phase",
    [
        runtime.PREREQUISITE_LIFECYCLE_STAGED,
        runtime.PREREQUISITE_LIFECYCLE_COMMITTED,
    ],
)
def test_live_collector_builds_v3_receipt_from_both_execution_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_phase: str,
) -> None:
    topology = _topology()
    payloads = {
        Path(f"/etc/systemd/system/{runtime.ISOLATED_WORKER_SERVICE_UNIT}"): (
            b"worker-service\n",
            SimpleNamespace(st_uid=0, st_gid=0, st_mode=stat.S_IFREG | 0o444),
        ),
        runtime.ISOLATED_WORKER_CONFIG: (
            b"worker-config",
            SimpleNamespace(st_uid=0, st_gid=996, st_mode=stat.S_IFREG | 0o440),
        ),
        runtime.BWRAP_PATH: (
            b"bwrap",
            SimpleNamespace(st_uid=0, st_gid=0, st_mode=stat.S_IFREG | 0o755),
        ),
        runtime.SHELL_PATH: (
            b"shell",
            SimpleNamespace(st_uid=0, st_gid=0, st_mode=stat.S_IFREG | 0o755),
        ),
        runtime.BROWSER_CONFIG_PATH: (
            b"browser-config",
            SimpleNamespace(st_uid=0, st_gid=995, st_mode=stat.S_IFREG | 0o440),
        ),
    }
    worker = topology["isolated_worker"]
    browser = topology["browser"]
    worker["service_fragment_sha256"] = hashlib.sha256(
        payloads[Path(f"/etc/systemd/system/{runtime.ISOLATED_WORKER_SERVICE_UNIT}")][0]
    ).hexdigest()
    worker["config_sha256"] = hashlib.sha256(
        payloads[runtime.ISOLATED_WORKER_CONFIG][0]
    ).hexdigest()
    worker["bwrap_sha256"] = hashlib.sha256(payloads[runtime.BWRAP_PATH][0]).hexdigest()
    worker["shell_sha256"] = hashlib.sha256(payloads[runtime.SHELL_PATH][0]).hexdigest()
    browser["config_sha256"] = hashlib.sha256(
        payloads[runtime.BROWSER_CONFIG_PATH][0]
    ).hexdigest()
    services = _services(topology, lifecycle_phase=lifecycle_phase)
    sockets = _sockets()

    monkeypatch.setattr(
        runtime,
        "_systemd_socket_unit_observation",
        lambda _unit, *, lifecycle_phase: {
            "socket_unit": runtime.ISOLATED_WORKER_SOCKET_UNIT,
            "socket_fragment_path": (
                f"/etc/systemd/system/{runtime.ISOLATED_WORKER_SOCKET_UNIT}"
            ),
            "socket_fragment_sha256": worker["socket_fragment_sha256"],
            "socket_unit_file_state": (
                "disabled"
                if lifecycle_phase == runtime.PREREQUISITE_LIFECYCLE_STAGED
                else "enabled"
            ),
            "socket_active_state": "active",
            "socket_sub_state": "listening",
            "socket_drop_in_paths": [],
            "socket_need_daemon_reload": False,
        },
    )
    monkeypatch.setattr(
        runtime,
        "_read_bounded_regular",
        lambda path, maximum: payloads[Path(path)],
    )
    monkeypatch.setattr(
        runtime,
        "_systemd_service_observation",
        lambda name, _unit, **_kwargs: copy.deepcopy(services[name]),
    )
    monkeypatch.setattr(
        runtime,
        "_socket_observation",
        lambda path, *, main_pid: {
            **copy.deepcopy(
                next(value for value in sockets.values() if value["path"] == str(path))
            ),
            "main_pid": main_pid,
        },
    )
    monkeypatch.setattr(
        runtime,
        "_collect_execution_readiness_proofs",
        lambda **_kwargs: _execution_proofs(topology),
    )
    monkeypatch.setattr(
        runtime,
        "_collect_runtime_dependency_proof",
        lambda **_kwargs: _runtime_dependencies(topology),
    )
    monkeypatch.setattr(
        runtime,
        "_collect_gateway_state_proof",
        lambda **_kwargs: _gateway_state(topology),
    )
    monkeypatch.setattr(
        runtime,
        "_collect_mac_ops_ping_proof",
        lambda **_kwargs: {
            "main_pid": services["mac_ops"]["main_pid"],
            "service_identity_sha256": "3" * 64,
            "receipt_sha256": "4" * 64,
            "peer_main_pid_validated": True,
            "external_io": False,
            "ready": True,
        },
    )
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime, "_read_boot_id", lambda: "focused-boot")

    def credential(public_path, _actual_path, *, refresh_capable, **_kwargs):
        is_codex = Path(public_path) == runtime.CODEX_AUTH_PATH
        return _lease(
            Path(public_path),
            owner_uid=995 if is_codex else 0,
            owner_gid=991 if is_codex else 0,
            refresh=refresh_capable,
        )

    monkeypatch.setattr(runtime, "_credential_observation", credential)
    receipt = runtime.collect_current_production_capability_prerequisite_receipt(
        revision=REVISION,
        topology=topology,
        lifecycle_phase=lifecycle_phase,
        mac_ops_edge_config={},
        now_unix=NOW,
    )
    assert receipt["schema"] == runtime.PREREQUISITE_SCHEMA
    assert receipt["lifecycle_phase"] == lifecycle_phase
    assert receipt["isolated_worker"]["service_main_pid"] == 104
    assert receipt["browser"]["service_main_pid"] == 105
    assert (
        receipt["capability_proofs"]["isolated_worker_exec"]["execution_round_trip"]
        is True
    )
    assert (
        receipt["capability_proofs"]["browser_controller_command"]["command_round_trip"]
        is True
    )


def test_prerequisite_implementation_contains_no_legacy_runtime_path() -> None:
    source = inspect.getsource(runtime).lower()
    assert "docker" not in source
    assert "cdp" not in source
