from __future__ import annotations

import copy
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_host_authority as authority


REVISION = "a" * 40
BOOT = "b" * 64
HOST = "c" * 64
PLAN_DIGEST = "d" * 64
NOW = 2_000_000_000


def _plan_mapping():
    root = f"/opt/muncho-canary-releases/{REVISION}"
    interpreter = f"{root}/venv/bin/python"
    return {
        "schema": authority.NATIVE_OBSERVATION_PLAN_SCHEMA,
        "boot_id_sha256": BOOT,
        "host_identity_sha256": HOST,
        "observation_id": str(uuid.UUID("11111111-1111-4111-8111-111111111111")),
        "revision": REVISION,
        "artifact_root": root,
        "artifact_sha256": "1" * 64,
        "release_manifest_file_sha256": "2" * 64,
        "config_collector_receipt_sha256": "9" * 64,
        "gateway_unit": {
            "name": "hermes-cloud-gateway.service",
            "path": "/etc/systemd/system/hermes-cloud-gateway.service",
            "sha256": "3" * 64,
        },
        "writer_unit": {
            "name": "muncho-canonical-writer.service",
            "path": "/etc/systemd/system/muncho-canonical-writer.service",
            "sha256": "4" * 64,
        },
        "gateway_argv": [
            interpreter,
            "-I",
            "-m",
            "gateway.canonical_writer_gateway_bootstrap",
        ],
        "writer_argv": [
            interpreter,
            "-I",
            "-m",
            "gateway.canonical_writer_bootstrap",
            "--config",
            "/etc/muncho-canonical-writer/writer.json",
        ],
        "gateway_config": {
            "path": "/etc/hermes/config.yaml",
            "sha256": "5" * 64,
        },
        "writer_config": {
            "path": "/etc/muncho-canonical-writer/writer.json",
            "sha256": "6" * 64,
        },
        "identities": {
            "gateway_uid": 993,
            "gateway_gid": 992,
            "gateway_supplementary_gids": [990, 992],
            "writer_uid": 999,
            "writer_gid": 994,
            "writer_supplementary_gids": [991, 994],
            "socket_group_gid": 990,
            "projector_uid": 992,
            "projector_gid": 991,
            "gateway_home": "/var/lib/hermes-gateway",
            "writer_home": "/nonexistent",
            "projector_home": "/nonexistent",
        },
        "database": {
            "ip_network": "10.91.0.3/32",
            "tls_server_name": "db.muncho.internal",
            "ca_path": "/etc/muncho/trust/cloudsql-server-ca.pem",
            "ca_sha256": "7" * 64,
        },
        "discord": {
            "unit_name": "muncho-discord-egress.service",
            "config_path": "/etc/muncho/discord-edge.json",
            "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
            "socket_path": "/run/muncho-discord-egress/edge.sock",
            "required_absent": True,
        },
        "native_discovery_policy": {
            "allowed_roots": ["/usr/lib"],
            "allowed_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
            "maximum_mappings": 256,
            "required_owner_uid": 0,
            "required_owner_gid": 0,
            "require_regular": True,
            "require_single_link": True,
            "forbid_symlink": True,
            "forbid_acl": True,
            "forbid_xattrs": True,
            "forbid_writable": True,
            "forbid_deleted": True,
            "exclude_artifact_root": True,
            "digest_algorithm": "sha256",
        },
        "legacy_helper_path": str(authority.LEGACY_CLOUD_SQL_HELPER_PATH),
        "external_iam_policy_sha256": "8" * 64,
    }


def _plan():
    return authority.NativeObservationPlan.from_mapping(_plan_mapping())


def _approval(plan=None, *, scope="native_observation", now=NOW):
    plan = plan or _plan()
    return authority.OwnerApprovalReceipt.from_mapping(
        {
            "schema": authority.OWNER_APPROVAL_RECEIPT_SCHEMA,
            "scope": scope,
            "plan_sha256": plan.sha256,
            "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
            "cryptographic_owner_proof": False,
            "owner_subject_sha256": "9" * 64,
            "approval_source_sha256": "a" * 64,
            "nonce_sha256": "b" * 64,
            "approved_at_unix": now - 1,
            "expires_at_unix": now + 300,
        }
    )


def _discord_absent():
    return {
        "unit_name": "muncho-discord-egress.service",
        "unit_exists": False,
        "unit_enabled": False,
        "unit_active": False,
        "main_pid": 0,
        "config_exists": False,
        "token_exists": False,
        "socket_exists": False,
        "process_pids": [],
    }


def _live_service(plan, label, pid, start):
    identity = plan.value["identities"]
    return {
        "unit_name": plan.value[f"{label}_unit"]["name"],
        "active_state": "active",
        "sub_state": "running",
        "unit_file_state": "disabled",
        "main_pid": pid,
        "start_time_ticks": start,
        "argv": list(plan.value[f"{label}_argv"]),
        "external_native_mappings": [
            {"path": "/usr/lib/x86_64-linux-gnu/libc.so.6", "sha256": "c" * 64}
        ],
        "kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
        "process_authority": {
            "pid": pid,
            "process_start_time_ticks": start,
            "effective_uid": identity[f"{label}_uid"],
            "effective_gid": identity[f"{label}_gid"],
            "supplementary_gids": identity[f"{label}_supplementary_gids"],
            "no_new_privileges": True,
            "effective_capabilities": [],
            "executable": plan.value[f"{label}_argv"][0],
        },
    }


def _receipt_mapping():
    plan = _plan()
    observed = 1_000_000_000
    observation = {
        "boot_id_sha256": BOOT,
        "host_identity_sha256": HOST,
        "observed_at_unix": NOW,
        "observed_at_boottime_ns": observed,
        "expires_at_boottime_ns": observed
        + authority.NATIVE_OBSERVATION_TTL_SECONDS * 1_000_000_000,
        "gateway_service": _live_service(plan, "gateway", 4242, 111),
        "writer_service": _live_service(plan, "writer", 4343, 222),
        "discord_absence": _discord_absent(),
        "legacy_helper_absence": {
            "path": str(authority.LEGACY_CLOUD_SQL_HELPER_PATH),
            "file_exists": False,
            "file_symlink": False,
            "parent_exists": False,
            "parent_symlink": False,
            "gateway_access": {"read": False, "write": False, "execute": False},
        },
    }
    stopped = lambda label: {
        "unit_name": plan.value[f"{label}_unit"]["name"],
        "load_state": "loaded",
        "active_state": "inactive",
        "sub_state": "dead",
        "unit_file_state": "disabled",
        "main_pid": 0,
    }
    final = {
        "boot_id_sha256": BOOT,
        "host_identity_sha256": HOST,
        "finalized_at_unix": NOW + 1,
        "finalized_at_boottime_ns": observed + 1_000_000_000,
        "gateway_service": stopped("gateway"),
        "writer_service": stopped("writer"),
        "discord_absence": _discord_absent(),
    }
    return {
        "schema": authority.NATIVE_OBSERVATION_RECEIPT_SCHEMA,
        "native_observation_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": _approval(plan).sha256,
        "host_preparation_receipt_sha256": "d" * 64,
        "external_iam_receipt_sha256": "e" * 64,
        "plan": plan.to_mapping(),
        "observation": observation,
        "final_state": final,
    }


def _checks(names):
    return [
        {"name": name, "passed": True, "detail": "exact"}
        for name in sorted(names)
    ]


def _iam_reports():
    foundation = {
        "schema": "muncho-isolated-canary-foundation-preflight.v2",
        "ok": True,
        "collected_at_unix": NOW - 1,
        "plan_sha256": "1" * 64,
        "satisfied_steps": list(authority._FOUNDATION_REQUIRED_STEPS),
        "spec": {
            "project": "adventico-ai-platform",
            "zone": "europe-west3-a",
            "service_account_name": "muncho-canary-v2-runtime",
        },
        "checks": _checks(authority._FOUNDATION_REQUIRED_CHECKS),
    }
    host = {
        "schema": "muncho-isolated-canary-host-preflight.v1",
        "ok": True,
        "collected_at_unix": NOW - 1,
        "plan_sha256": "2" * 64,
        "satisfied_steps": ["create_isolated_canary_vm"],
        "checks": _checks(authority._HOST_REQUIRED_CHECKS),
    }
    return foundation, host


def test_native_plan_is_discovery_only_and_exact():
    plan = _plan()
    assert "external_native_mappings" not in plan.to_mapping()
    assert plan.value["native_discovery_policy"][
        "allowed_kernel_executable_mappings"
    ] == ["[vdso]", "[vsyscall]"]

    injected = _plan_mapping()
    injected["external_native_mappings"] = []
    with pytest.raises(ValueError, match="fields are not exact"):
        authority.NativeObservationPlan.from_mapping(injected)

    missing = _plan_mapping()
    missing["native_discovery_policy"].pop("allowed_kernel_executable_mappings")
    with pytest.raises(ValueError, match="fields are not exact"):
        authority.NativeObservationPlan.from_mapping(missing)


def test_owner_approval_is_scope_plan_freshness_and_receipt_addressed():
    plan = _plan()
    receipt = _approval(plan)
    receipt.require(scope="native_observation", plan_sha256=plan.sha256, now_unix=NOW)
    assert authority.owner_approval_receipt_path(receipt) == (
        authority.DEFAULT_OWNER_APPROVAL_ROOT
        / "native_observation"
        / plan.sha256
        / f"{receipt.sha256}.json"
    )
    with pytest.raises(PermissionError):
        receipt.require(scope="activation", plan_sha256=plan.sha256, now_unix=NOW)
    with pytest.raises(PermissionError):
        receipt.require(
            scope="native_observation", plan_sha256="0" * 64, now_unix=NOW
        )
    with pytest.raises(PermissionError):
        receipt.require(
            scope="native_observation", plan_sha256=plan.sha256, now_unix=NOW + 301
        )
    overclaim = receipt.to_mapping()
    overclaim["cryptographic_owner_proof"] = True
    with pytest.raises(ValueError, match="trust semantics"):
        authority.OwnerApprovalReceipt.from_mapping(overclaim)


def test_external_iam_is_exact_fresh_projection_without_gcloud():
    foundation, host = _iam_reports()
    receipt = authority.build_external_iam_receipt(
        foundation,
        host,
        source_approval_sha256="3" * 64,
        now_unix=NOW,
    )
    assert receipt.evaluator_projection()["complete"] is True
    assert receipt.value["roles"] == [
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter",
    ]
    receipt.require_fresh(NOW + 300)
    receipt.require_fresh(NOW + 480, minimum_remaining_seconds=720)
    with pytest.raises(ValueError, match="stale"):
        receipt.require_fresh(NOW + 481, minimum_remaining_seconds=720)
    assert (
        receipt.value["expires_at_unix"] - receipt.value["collected_at_unix"]
        == authority.EXTERNAL_IAM_TTL_SECONDS
        == 1200
    )

    incomplete = copy.deepcopy(foundation)
    incomplete["satisfied_steps"].remove("grant_monitoring_writer")
    with pytest.raises(ValueError, match="complete resource set"):
        authority.build_external_iam_receipt(
            incomplete,
            host,
            source_approval_sha256="3" * 64,
            now_unix=NOW,
        )


def test_native_receipt_requires_stopped_state_and_exact_native_sets():
    valid = _receipt_mapping()
    receipt = authority.NativeObservationReceipt.from_mapping(valid)
    # Durable consumers may read a historically expired/same-host stopped receipt.
    assert receipt.sha256
    with pytest.raises(ValueError, match="validity window"):
        authority.NativeObservationReceipt.from_mapping(
            valid,
            current_boottime_ns=valid["observation"]["expires_at_boottime_ns"] + 1,
        )

    running = copy.deepcopy(valid)
    running["final_state"]["writer_service"]["main_pid"] = 4343
    with pytest.raises(ValueError, match="not finalized stopped"):
        authority.NativeObservationReceipt.from_mapping(running)

    injected = copy.deepcopy(valid)
    injected["observation"]["writer_service"]["external_native_mappings"][0][
        "path"
    ] = "/tmp/injected.so"
    with pytest.raises(ValueError, match="escapes discovery policy"):
        authority.NativeObservationReceipt.from_mapping(injected)

    other_boot = copy.deepcopy(valid)
    with pytest.raises(ValueError, match="replayed"):
        authority.NativeObservationReceipt.from_mapping(
            other_boot, current_boot_id_sha256="e" * 64
        )

    near_bound = copy.deepcopy(valid)
    near_bound["final_state"]["finalized_at_boottime_ns"] = near_bound[
        "observation"
    ]["expires_at_boottime_ns"]
    authority.NativeObservationReceipt.from_mapping(
        near_bound,
        current_boottime_ns=near_bound["observation"]["expires_at_boottime_ns"],
    )
    assert authority.NATIVE_OBSERVATION_TTL_SECONDS == 300


def test_authority_mapping_surfaces_sudo_polkit_caps_and_nnp():
    process = authority.ProcessAuthorityEvidence(
        pid=100,
        start_time_ticks=10,
        effective_uid=993,
        effective_gid=992,
        supplementary_gids=(990, 992),
        no_new_privileges=True,
        effective_capabilities=(),
        executable="/opt/release/venv/bin/python",
        argv=("/opt/release/venv/bin/python",),
    )
    denied = authority._authority_mapping(
        processes=(process,),
        sudo_commands=(),
        doas_commands=(),
        polkit_actions=(),
        writable_unit_paths=(),
        writable_cron_paths=(),
    )
    assert not any(value is True for value in denied.values())

    allowed = authority._authority_mapping(
        processes=(process,),
        sudo_commands=("(ALL) ALL",),
        doas_commands=(),
        polkit_actions=("org.freedesktop.systemd1.manage-units",),
        writable_unit_paths=("/etc/systemd/system/writer.service",),
        writable_cron_paths=("/etc/cron.d",),
    )
    assert allowed["can_manage_writer_units"] is True
    assert allowed["can_manage_cron"] is True
    assert allowed["can_switch_to_writer_uid"] is True


def test_pkcheck_uses_debian_12_noninteractive_default(monkeypatch):
    process = authority.ProcessAuthorityEvidence(
        pid=4242,
        start_time_ticks=123,
        effective_uid=993,
        effective_gid=992,
        supplementary_gids=(990, 992),
        no_new_privileges=True,
        effective_capabilities=(),
        executable="/opt/release/venv/bin/python",
        argv=("/opt/release/venv/bin/python",),
    )
    commands = []
    monkeypatch.setattr(authority, "_validate_privileged_binary", lambda *_a, **_k: None)
    monkeypatch.setattr(authority, "_process_start_time", lambda _path: 123)

    def run(argv, **_kwargs):
        commands.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 1, "", "")

    monkeypatch.setattr(authority, "_run_fixed", run)
    assert authority._polkit_actions(process) == ()
    assert all("--allow-user-interaction=no" not in command for command in commands)


def test_dpkg_absence_requires_exact_stderr(monkeypatch):
    monkeypatch.setattr(authority, "_validate_privileged_binary", lambda *_a, **_k: None)
    monkeypatch.setattr(
        authority,
        "_run_fixed",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 1, "", "dpkg-query: no packages found matching at\n"
        ),
    )
    assert authority._package_absent("at") is True
    monkeypatch.setattr(
        authority,
        "_run_fixed",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 1, "", "other\n"),
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        authority._package_absent("at")


def test_systemd_templates_are_not_sent_to_show(monkeypatch):
    outputs = iter(
        (
            "autovt@.service disabled\nwriter.service disabled\n",
            "user@.service loaded inactive dead x\nwriter.service loaded active running x\n"
            "data.mount loaded active mounted x\n",
        )
    )
    commands = []

    def run(argv, **_kwargs):
        commands.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, next(outputs), "")

    monkeypatch.setattr(
        authority,
        "_run_fixed",
        run,
    )
    assert authority._systemd_unit_names() == ("data.mount", "writer.service")
    assert commands[1][:3] == (authority._SYSTEMCTL, "list-units", "--all")
    assert not any(argument.startswith("--type=") for argument in commands[1])


def test_systemd_aliases_dedupe_only_after_provenance(monkeypatch):
    monkeypatch.setattr(authority, "_systemd_unit_names", lambda: ("ssh.service", "sshd.service"))
    block = (
        "Id=ssh.service\nLoadState=loaded\nFragmentPath=/usr/lib/systemd/system/ssh.service\n"
        "Transient=no\nUser=\nExecStart={}\nTriggers=\nTriggeredBy=\nWantedBy=\nRequiredBy=\n"
        "BoundBy=\nUpheldBy=\nRequisiteOf=\nOnSuccessOf=\nOnFailureOf=\n"
        "BindsTo=\nUpholds=\nRequisite=\nOnSuccess=\nOnFailure=\n"
    )
    monkeypatch.setattr(
        authority,
        "_run_fixed",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, block + "\n" + block, ""),
    )
    observed = []
    monkeypatch.setattr(
        authority,
        "_verify_systemd_alias",
        lambda alias, **kwargs: observed.append((alias, kwargs["canonical_name"])),
    )
    inventory = authority._systemd_inventory(999, 993)
    assert [item["Id"] for item in inventory] == ["ssh.service"]
    assert observed == [("sshd.service", "ssh.service")]


def test_reverse_activation_evidence_catches_direct_and_scanned_sources():
    def item(name, **relations):
        value = {
            "Id": name,
            "Transient": "no",
            "Triggers": "",
            "TriggeredBy": "",
            "WantedBy": "",
            "RequiredBy": "",
            "BoundBy": "",
            "UpheldBy": "",
            "RequisiteOf": "",
            "OnSuccessOf": "",
            "OnFailureOf": "",
            "BindsTo": "",
            "Upholds": "",
            "Requisite": "",
            "OnSuccess": "",
            "OnFailure": "",
        }
        value.update(relations)
        return value

    inventory = (
        item(
            "muncho-canonical-writer.service",
            TriggeredBy="writer.socket",
            WantedBy="data.mount writer.target",
        ),
        item("writer.socket", Triggers="muncho-canonical-writer.service"),
        item("writer.path", RequiredBy="muncho-canonical-writer.service"),
        item("writer.target"),
        item("data.mount"),
    )
    evidence = authority._reverse_activation_evidence(
        "muncho-canonical-writer.service",
        inventory,
    )
    assert evidence["triggered_by"] == ["writer.socket"]
    assert evidence["wanted_by"] == ["data.mount", "writer.target"]
    assert evidence["socket_units"] == ["writer.socket"]
    assert evidence["path_units"] == ["writer.path"]
    assert evidence["target_units"] == ["writer.target"]
    assert evidence["other_units"] == ["data.mount"]
    assert evidence["reverse_references"] == ["writer.path", "writer.socket"]

    approved = (
        item(
            "muncho-canonical-writer.service",
            BoundBy="hermes-cloud-gateway.service",
        ),
        item(
            "hermes-cloud-gateway.service",
            BindsTo="muncho-canonical-writer.service",
        ),
    )
    approved_evidence = authority._reverse_activation_evidence(
        "muncho-canonical-writer.service",
        approved,
    )
    assert approved_evidence["bound_by"] == ["hermes-cloud-gateway.service"]
    assert approved_evidence["service_units"] == ["hermes-cloud-gateway.service"]
    assert approved_evidence["reverse_references"] == [
        "hermes-cloud-gateway.service"
    ]


def test_user_systemd_validates_every_global_tree_entry(monkeypatch):
    roots = {"/etc/systemd/user", "/usr/lib/systemd/user"}
    malicious = Path("/etc/systemd/user/gateway-owned.socket")
    monkeypatch.setattr(
        authority.os.path,
        "lexists",
        lambda value: str(value) in roots,
    )
    monkeypatch.setattr(
        authority,
        "_native_systemd_state",
        lambda _unit: {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": 0,
        },
    )
    monkeypatch.setattr(
        authority.Path,
        "rglob",
        lambda path, _pattern: (malicious,)
        if str(path) == "/etc/systemd/user"
        else (),
    )
    monkeypatch.setattr(
        authority.os, "listxattr", lambda *_a, **_k: [], raising=False
    )

    def lstat(path):
        if str(path) == str(malicious):
            return SimpleNamespace(
                st_mode=stat.S_IFREG | 0o664,
                st_uid=993,
                st_gid=992,
            )
        return SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        )

    monkeypatch.setattr(authority.os, "lstat", lstat)
    with pytest.raises(RuntimeError, match="provenance"):
        authority._user_systemd_evidence(
            user_name="muncho-gateway",
            uid=993,
            home="/var/lib/hermes-gateway",
            service_read_write_paths=(),
        )


def test_user_systemd_safe_inventory_is_explicit(monkeypatch):
    roots = {
        "/etc/systemd/user",
        "/usr/lib/systemd/user",
        "/var/lib/hermes-gateway",
    }
    monkeypatch.setattr(
        authority.os.path,
        "lexists",
        lambda value: str(value) in roots,
    )
    monkeypatch.setattr(
        authority,
        "_native_systemd_state",
        lambda _unit: {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": 0,
        },
    )
    monkeypatch.setattr(authority.Path, "rglob", lambda *_a, **_k: ())
    monkeypatch.setattr(
        authority.os, "listxattr", lambda *_a, **_k: [], raising=False
    )
    monkeypatch.setattr(
        authority.os,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        ),
    )
    evidence = authority._user_systemd_evidence(
        user_name="muncho-gateway",
        uid=993,
        home="/var/lib/hermes-gateway",
        service_read_write_paths=(),
    )
    assert evidence["global_directories_protected"] is True
    assert evidence["activation_units"] == []
    assert evidence["runtime_activation_units"] == []
    assert evidence["global_activation_units"] == []


def test_cron_inventory_handles_system_macros_and_exact_absence(monkeypatch):
    crontab_gid = 123
    monkeypatch.setattr(
        authority,
        "_validate_privileged_binary",
        lambda *_a, **_k: SimpleNamespace(
            st_mode=stat.S_IFREG | stat.S_ISGID | 0o755,
            st_gid=crontab_gid,
        ),
    )
    monkeypatch.setattr(
        authority.grp,
        "getgrnam",
        lambda _name: SimpleNamespace(gr_gid=crontab_gid),
    )
    monkeypatch.setattr(
        authority,
        "_run_fixed",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            1,
            "",
            "no crontab for muncho-canonical-writer\n",
        ),
    )
    monkeypatch.setattr(
        authority.Path,
        "is_dir",
        lambda path: str(path) == "/etc/cron.d",
    )
    monkeypatch.setattr(
        authority.Path,
        "exists",
        lambda path: str(path) == "/etc/crontab",
    )
    monkeypatch.setattr(
        authority.Path,
        "iterdir",
        lambda _path: (Path("/etc/cron.d/muncho"),),
    )
    monkeypatch.setattr(
        authority.Path,
        "read_bytes",
        lambda path: (
            b"* * * * * muncho-canonical-writer /bin/true\n"
            if str(path) == "/etc/crontab"
            else b"@reboot muncho-canonical-writer /bin/true\n"
        ),
    )
    monkeypatch.setattr(
        authority.os,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFREG | 0o644),
    )
    entries = authority._cron_entries("muncho-canonical-writer")
    assert entries == (
        "/etc/cron.d/muncho:1:@reboot muncho-canonical-writer /bin/true",
        "/etc/crontab:1:* * * * * muncho-canonical-writer /bin/true",
    )


def test_at_absence_requires_package_binary_and_spool_all_absent(monkeypatch):
    monkeypatch.setattr(authority, "_package_absent", lambda _name: True)
    monkeypatch.setattr(authority.os.path, "lexists", lambda _path: False)
    assert authority._at_jobs("muncho-canonical-writer") == ()
    monkeypatch.setattr(
        authority.os.path,
        "lexists",
        lambda path: str(path) == "/var/spool/atjobs",
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        authority._at_jobs("muncho-canonical-writer")


def test_group_policy_reconciles_full_nss_membership_and_dormant_projector(
    monkeypatch,
):
    gateway_account = SimpleNamespace(
        pw_name="muncho-gateway",
        pw_uid=993,
        pw_gid=992,
        pw_dir="/var/lib/hermes-gateway",
        pw_shell="/usr/sbin/nologin",
    )
    writer_account = SimpleNamespace(
        pw_name="muncho-canonical-writer",
        pw_uid=999,
        pw_gid=994,
        pw_dir="/nonexistent",
        pw_shell="/usr/sbin/nologin",
    )
    projector_account = SimpleNamespace(
        pw_name="muncho-projector",
        pw_uid=992,
        pw_gid=991,
        pw_dir="/nonexistent",
        pw_shell="/usr/sbin/nologin",
    )
    groups = {
        992: SimpleNamespace(gr_name="muncho-gateway", gr_gid=992, gr_mem=[]),
        994: SimpleNamespace(
            gr_name="muncho-canonical-writer", gr_gid=994, gr_mem=[]
        ),
        990: SimpleNamespace(
            gr_name="muncho-writer-client", gr_gid=990, gr_mem=["muncho-gateway"]
        ),
        991: SimpleNamespace(
            gr_name="muncho-projector",
            gr_gid=991,
            gr_mem=["muncho-canonical-writer"],
        ),
    }
    monkeypatch.setattr(authority, "_identity_group_name", lambda gid: groups[gid].gr_name)
    monkeypatch.setattr(authority.grp, "getgrgid", lambda gid: groups[gid])
    monkeypatch.setattr(
        authority.pwd,
        "getpwall",
        lambda: [gateway_account, writer_account, projector_account],
    )
    monkeypatch.setattr(
        authority.pwd,
        "getpwnam",
        lambda name: {
            "muncho-gateway": gateway_account,
            "muncho-canonical-writer": writer_account,
            "muncho-projector": projector_account,
        }[name],
    )
    monkeypatch.setattr(
        authority.os,
        "getgrouplist",
        lambda name, _gid: {
            "muncho-gateway": [992, 990],
            "muncho-canonical-writer": [994, 991],
            "muncho-projector": [991],
        }[name],
    )
    monkeypatch.setattr(authority, "_pids_for_uid", lambda _uid: ())

    def process(pid, uid, gid, gids):
        return authority.ProcessAuthorityEvidence(
            pid=pid,
            start_time_ticks=pid,
            effective_uid=uid,
            effective_gid=gid,
            supplementary_gids=gids,
            no_new_privileges=True,
            effective_capabilities=(),
            executable="/opt/release/venv/bin/python",
            argv=("/opt/release/venv/bin/python",),
        )

    evidence = authority._group_policy(
        gateway=process(10, 993, 992, (990, 992)),
        children=(),
        writer=process(20, 999, 994, (991, 994)),
        allowed_gids=frozenset({990, 991, 992, 994}),
        gateway_user_name="muncho-gateway",
        writer_user_name="muncho-canonical-writer",
        gateway_gid=992,
        writer_gid=994,
        socket_group_gid=990,
        projector_gid=991,
    )
    assert evidence["complete"] is True
    assert evidence["gateway_account_gids"] == [990, 992]
    assert evidence["writer_account_gids"] == [991, 994]
    assert evidence["projector_identity"]["process_pids"] == []

    groups[991].gr_mem.append("root")
    with pytest.raises(RuntimeError, match="membership drifted"):
        authority._group_policy(
            gateway=process(10, 993, 992, (990, 992)),
            children=(),
            writer=process(20, 999, 994, (991, 994)),
            allowed_gids=frozenset({990, 991, 992, 994}),
            gateway_user_name="muncho-gateway",
            writer_user_name="muncho-canonical-writer",
            gateway_gid=992,
            writer_gid=994,
            socket_group_gid=990,
            projector_gid=991,
        )


def test_native_discovery_rejects_anonymous_and_detects_map_race(monkeypatch):
    plan = _plan()
    process = authority.ProcessAuthorityEvidence(
        pid=4242,
        start_time_ticks=10,
        effective_uid=993,
        effective_gid=992,
        supplementary_gids=(990, 992),
        no_new_privileges=True,
        effective_capabilities=(),
        executable=plan.value["gateway_argv"][0],
        argv=tuple(plan.value["gateway_argv"]),
    )
    anonymous = "1000-2000 r-xp 00000000 00:00 0\n"
    monkeypatch.setattr(authority, "_read_bounded", lambda *_a, **_k: anonymous)
    monkeypatch.setattr(authority, "_process_start_time", lambda _path: 10)
    with pytest.raises(RuntimeError, match="anonymous executable"):
        authority._discover_native_mappings(process, plan)

    before = (
        "1000-2000 r-xp 00000000 00:00 0 [vdso]\n"
        "2000-3000 r-xp 00000000 08:01 1 /usr/lib/liba.so\n"
    )
    after = before.replace("liba.so", "libb.so")
    reads = iter((before, after))
    monkeypatch.setattr(authority, "_read_bounded", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(
        authority,
        "_hash_native_mapping",
        lambda path, policy: {"path": str(path), "sha256": "f" * 64},
    )
    with pytest.raises(RuntimeError, match="changed during native mapping"):
        authority._discover_native_mappings(process, plan)


def test_native_receipt_wrong_path_fails_before_filesystem_write(monkeypatch):
    receipt = authority.NativeObservationReceipt.from_mapping(_receipt_mapping())
    monkeypatch.setattr(authority, "_require_root_linux", lambda: None)
    with pytest.raises(ValueError, match="plan-addressed"):
        authority.write_native_observation_receipt("/tmp/native.json", receipt)


@pytest.mark.parametrize("action", ["observe", "finalize", "build-iam"])
def test_packaged_host_authority_has_no_public_mutation_cli(action, tmp_path):
    repository = Path(__file__).resolve().parents[2]
    sentinel = tmp_path / "must-not-exist.json"
    program = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(repository)!r});"
        f"sys.argv=['gateway.canonical_writer_host_authority',{action!r},"
        f"'--receipt',{str(sentinel)!r}];"
        "runpy.run_module('gateway.canonical_writer_host_authority',run_name='__main__')"
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not sentinel.exists()
