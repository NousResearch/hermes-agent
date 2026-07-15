from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_continuity_package
from gateway import production_cron_cutover_runtime
from gateway import production_cron_migration
from gateway import production_owner_runtime
from gateway.operational_edge_catalog import (
    operation_catalog,
    required_cron_operations,
)
from gateway.operational_edge_readiness import (
    build_operational_edge_readiness,
)
from ops.muncho.runtime import mechanical_job_rail
from tests.gateway.test_production_capability_prerequisites import (
    _receipt_pair as _capability_receipt_pair,
)


NOW = 1_800_000_000
REVISION = "a" * 40


def _operational_receipt_key_ids() -> dict[str, str]:
    domains = (
        "adventico_email", "bitrix", "canonical", "github",
        "infrastructure", "skyvision_db", "skyvision_email",
        "skyvision_gitlab", "skyvision_panel",
    )
    key_digits = ("a", "b", "c", "d", "e", "f", "1", "2", "3")
    return dict(zip(domains, (digit * 64 for digit in key_digits), strict=True))


def _operational_key_foundation(
    writer_public_key_id: str = "8" * 64,
) -> dict:
    root = cutover.OPERATIONAL_EDGE_KEY_STAGING_ROOT
    keys = [
        {
            "domain": domain,
            "private_path": str(
                root / f"operational-edge-{domain}-receipt-private.pem"
            ),
            "private_uid": 0,
            "private_gid": 0,
            "private_mode": "0400",
            "public_path": str(root / f"{domain}-receipt-public.pem"),
            "public_uid": 0,
            "public_gid": 0,
            "public_mode": "0444",
            "public_key_id": key_id,
            "created": True,
        }
        for domain, key_id in _operational_receipt_key_ids().items()
    ]
    unsigned = {
        "schema": "muncho-operational-edge-key-foundation.v1",
        "writer_public_key_id": writer_public_key_id,
        "keys": keys,
        "key_count": len(keys),
        "keys_distinct": True,
        "retain_created_keys_on_rollback": True,
        "private_content_or_digest_recorded": False,
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": cutover._sha256_json(unsigned)}


def _hashed(value: dict, field: str) -> dict:
    return {**value, field: cutover._sha256_json(value)}


def _runtime_attestation(revision: str = REVISION) -> dict:
    unsigned = {
        "schema": production_owner_runtime.ATTESTATION_SCHEMA,
        "revision": revision,
        "manifest_sha256": "1" * 64,
        "tree_sha256": "2" * 64,
        "interpreter_sha256": "3" * 64,
        "pyvenv_cfg_sha256": "4" * 64,
        "sys_path_sha256": "5" * 64,
        "required_modules_sha256": "6" * 64,
        "module_origins_release_local": True,
        "ambient_python_environment_present": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(unsigned, "attestation_sha256")


def _service(
    name: str,
    *,
    active: bool,
    digest: str | None,
    observed_at: int = NOW,
    unit_file_state: str | None = None,
    drop_in_digest: str | None = None,
    sub_state: str | None = None,
    main_pid: int | None = None,
) -> cutover.ServiceObservation:
    absent = digest is None
    fragment = "" if absent else {
        cutover.GATEWAY_UNIT: cutover.GATEWAY_FRAGMENT,
        cutover.WRITER_UNIT: cutover.WRITER_FRAGMENT,
        cutover.CONNECTOR_UNIT: cutover.CONNECTOR_FRAGMENT,
        cutover.PHASE_B_UNIT: cutover.PHASE_B_FRAGMENT,
        cutover.ROUTEBACK_EDGE_UNIT: cutover.ROUTEBACK_EDGE_FRAGMENT,
        cutover.MAC_OPS_UNIT: cutover.MAC_OPS_FRAGMENT,
        cutover.BROWSER_UNIT: cutover.BROWSER_FRAGMENT,
        cutover.ISOLATED_WORKER_SOCKET_UNIT: (
            cutover.ISOLATED_WORKER_SOCKET_FRAGMENT
        ),
        cutover.ISOLATED_WORKER_SERVICE_UNIT: (
            cutover.ISOLATED_WORKER_SERVICE_FRAGMENT
        ),
        **{
            cutover.operational_edge_service_unit(domain): (
                f"/etc/systemd/system/"
                f"{cutover.operational_edge_service_unit(domain)}"
            )
            for domain in _operational_receipt_key_ids()
        },
    }[name]
    drop_in_paths = (
        [cutover.GATEWAY_CONNECTOR_DROP_IN]
        if drop_in_digest is not None
        else []
    )
    raw = {
        "schema": cutover.SERVICE_SCHEMA,
        "name": name,
        "fragment_path": fragment,
        "fragment_sha256": digest,
        "load_state": "not-found" if absent else "loaded",
        "active_state": "active" if active else "inactive",
        "sub_state": (
            "running" if active else "dead"
        ) if sub_state is None else sub_state,
        "unit_file_state": (
            "" if absent else "enabled"
        ) if unit_file_state is None else unit_file_state,
        "main_pid": (1234 if active else 0) if main_pid is None else main_pid,
        "drop_in_paths": drop_in_paths,
        "drop_in_sha256": (
            {cutover.GATEWAY_CONNECTOR_DROP_IN: drop_in_digest}
            if drop_in_digest is not None
            else {}
        ),
        "need_daemon_reload": False,
        "triggered_by": (
            [cutover.ISOLATED_WORKER_SOCKET_UNIT]
            if name == cutover.ISOLATED_WORKER_SERVICE_UNIT
            else []
        ),
        "triggers": (
            [cutover.ISOLATED_WORKER_SERVICE_UNIT]
            if name == cutover.ISOLATED_WORKER_SOCKET_UNIT
            else []
        ),
        "observed_at_unix": observed_at,
    }
    return cutover.ServiceObservation.from_mapping(_hashed(raw, "observation_sha256"))


def _operational_edge_runtime(plan, *, unit_file_state: str) -> dict:
    transition = plan.value["host_transition"]
    identity = transition["identity_foundation"]
    result = {}
    for index, domain in enumerate(sorted(_operational_receipt_key_ids())):
        service_role = f"operational_edge_{domain}"
        socket_role = f"operational_edge_{domain}_client"
        service_uid = identity["users"][service_role]["uid"]
        service_gid = identity["groups"][service_role]["gid"]
        socket_gid = identity["groups"][socket_role]["gid"]
        unit = cutover.operational_edge_service_unit(domain)
        service = _service(
            unit,
            active=True,
            digest=transition["files"][f"operational_edge_unit_{domain}"][
                "sha256"
            ],
            unit_file_state=unit_file_state,
            main_pid=2000 + index,
        ).to_mapping()
        result[domain] = {
            "service": service,
            "process_uid": service_uid,
            "process_gid": service_gid,
            "socket_path": f"/run/muncho-operational-edge/{domain}/edge.sock",
            "socket_uid": service_uid,
            "socket_gid": socket_gid,
            "socket_mode": "0660",
            "socket_device": 100 + index,
            "socket_inode": 1000 + index,
            "ready": True,
        }
    return result


def _operational_edge_readiness(plan, runtime: dict) -> dict:
    required = dict(required_cron_operations())
    catalog = operation_catalog()
    jobs = []
    for source_job_id, operation_id in required.items():
        operation = catalog[operation_id]
        live = runtime[operation.domain]
        jobs.append({
            "source_job_id": source_job_id,
            "operation_id": operation_id,
            "domain": operation.domain,
            "service_unit": live["service"]["name"],
            "service_uid": live["process_uid"],
            "service_gid": live["process_gid"],
            "socket_path": live["socket_path"],
            "socket_uid": live["socket_uid"],
            "socket_gid": live["socket_gid"],
            "socket_mode": live["socket_mode"],
            "main_pid": live["service"]["main_pid"],
            "peer_round_trip": True,
            "probe_operation_id": (
                operation.probe_operation_id or operation.operation_id
            ),
            "probe_return_code": 0,
            "probe_packet_schema": (
                "muncho-operational-edge-probe-packet.v1"
            ),
            "probe_packet_sha256": "7" * 64,
            "meaningful_packet": True,
            "error_only_packet": False,
        })
    return build_operational_edge_readiness(
        revision=plan.value["release_revision"],
        required_jobs=required,
        jobs=jobs,
        boot_id_sha256="9" * 64,
        observed_at_unix=NOW,
        collector_nonce="f1438b18-df67-46ea-ae46-5e4f3f863f09",
    )


def _snapshot(
    rows: int,
    *,
    marker: str = "1",
    observed_at: int = NOW,
    identity_override: dict | None = None,
) -> cutover.LegacySnapshot:
    raw = {
        "schema": cutover.SNAPSHOT_SCHEMA,
        "database": cutover.DATABASE,
        "shape": "legacy19",
        "source_owner": "legacy_event_owner",
        "relation_oid": "16421",
        "source_row_count": rows,
        "canonical14_sha256": marker * 64,
        "extended19_sha256": ("2" if marker != "2" else "3") * 64,
        "occurred_at_cutoff": "2026-07-14T09:00:00+00:00",
        "inserted_at_cutoff": "2026-07-14T09:00:01+00:00",
        "relation_identity_sha256": "4" * 64,
        "acl_identity_sha256": "5" * 64,
        "index_identity_sha256": "6" * 64,
        "observed_at_unix": observed_at,
    }
    raw.update(identity_override or {})
    return cutover.LegacySnapshot.from_mapping(_hashed(raw, "snapshot_sha256"))


class MemoryJournal:
    def __init__(self) -> None:
        self.values: dict[str, list[cutover.JournalEntry]] = {}

    def load(self, plan_sha256: str) -> list[cutover.JournalEntry]:
        return list(self.values.get(plan_sha256, ()))

    def append(self, plan_sha256: str, event: str, evidence, now_unix: int):
        entries = self.values.setdefault(plan_sha256, [])
        raw = {
            "schema": cutover.JOURNAL_SCHEMA,
            "plan_sha256": plan_sha256,
            "sequence": len(entries),
            "event": event,
            "previous_entry_sha256": None if not entries else entries[-1].sha256,
            "evidence": copy.deepcopy(dict(evidence)),
            "recorded_at_unix": now_unix,
        }
        entry = cutover.JournalEntry.from_mapping(
            _hashed(raw, "entry_sha256"), plan_sha256=plan_sha256
        )
        entries.append(entry)
        return entry


class Services:
    def __init__(self) -> None:
        self.legacy_gateway_digest = "7" * 64
        self.target_gateway_digest = "8" * 64
        self.target_writer_digest = "9" * 64
        self.target_connector_digest = "e" * 64
        self.target_drop_in_digest = "f" * 64
        self.gateway = _service(cutover.GATEWAY_UNIT, active=True, digest=self.legacy_gateway_digest)
        self.writer = _service(cutover.WRITER_UNIT, active=False, digest=None)
        self.connector = _service(cutover.CONNECTOR_UNIT, active=False, digest=None)
        self.target_installed = False
        self.boot_enabled = False
        self.calls: list[str] = []

    def observe_gateway(self):
        return self.gateway

    def observe_writer(self):
        return self.writer

    def observe_connector(self):
        return self.connector

    def stop_gateway(self):
        self.calls.append("stop_gateway")
        digest = self.target_gateway_digest if self.target_installed else self.legacy_gateway_digest
        self.gateway = _service(
            cutover.GATEWAY_UNIT,
            active=False,
            digest=digest,
            drop_in_digest=(
                self.target_drop_in_digest if self.target_installed else None
            ),
            unit_file_state=(
                "disabled" if self.target_installed and not self.boot_enabled
                else None
            ),
        )

    def stop_writer(self):
        self.calls.append("stop_writer")
        digest = self.target_writer_digest if self.target_installed else None
        self.writer = _service(
            cutover.WRITER_UNIT,
            active=False,
            digest=digest,
            unit_file_state=(
                "disabled" if self.target_installed and not self.boot_enabled
                else None
            ),
        )

    def stop_connector(self):
        self.calls.append("stop_connector")
        digest = self.target_connector_digest if self.target_installed else None
        self.connector = _service(
            cutover.CONNECTOR_UNIT,
            active=False,
            digest=digest,
            unit_file_state=(
                "disabled" if self.target_installed and not self.boot_enabled
                else None
            ),
        )

    def start_gateway(self):
        self.calls.append("start_gateway")
        digest = self.target_gateway_digest if self.target_installed else self.legacy_gateway_digest
        self.gateway = _service(
            cutover.GATEWAY_UNIT,
            active=True,
            digest=digest,
            drop_in_digest=(
                self.target_drop_in_digest if self.target_installed else None
            ),
            unit_file_state=(
                "disabled" if self.target_installed and not self.boot_enabled
                else None
            ),
        )

    def commit_boot(self):
        self.boot_enabled = True
        self.gateway = _service(
            cutover.GATEWAY_UNIT,
            active=True,
            digest=self.target_gateway_digest,
            drop_in_digest=self.target_drop_in_digest,
        )
        self.writer = _service(
            cutover.WRITER_UNIT,
            active=True,
            digest=self.target_writer_digest,
        )
        self.connector = _service(
            cutover.CONNECTOR_UNIT,
            active=True,
            digest=self.target_connector_digest,
        )


class Snapshots:
    def __init__(self, final: cutover.LegacySnapshot) -> None:
        self.final = final
        self.before = final

    def observe_final_tail(self, _plan):
        return self.final

    def observe_before_apply(self, _plan):
        return self.before


def _target() -> dict:
    return {
        "project": cutover.PROJECT,
        "zone": cutover.ZONE,
        "vm": cutover.VM_NAME,
        "database": cutover.DATABASE,
        "sql_instance": "production-pg18",
        "sql_host": "10.20.30.40",
        "tls_server_name": "production.example.internal",
        "port": 5432,
        "writer_login": "muncho_production_writer_login",
    }


def _artifact(name: str, digest: str) -> dict:
    return {
        "path": str(
            cutover.PRODUCTION_RELEASE_BASE
            / f"hermes-agent-{REVISION[:12]}"
            / "ops"
            / "muncho"
            / "cutover"
            / name
        ),
        "sha256": digest * 64,
    }


def _mechanical_host_facts() -> dict:
    unsigned = {
        "schema": mechanical_job_rail.HOST_FACTS_SCHEMA,
        "collected_at": "2027-01-15T08:00:00Z",
        "github_cli": {
            "path": str(mechanical_job_rail.GH_PATH),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
            "group_or_other_writable": False,
            "sha256": "8" * 64,
        },
        "git": {
            "path": str(mechanical_job_rail.GIT_PATH),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
            "group_or_other_writable": False,
            "sha256": "9" * 64,
        },
        "github_credential": {
            "path": str(mechanical_job_rail.CREDENTIAL_SOURCE),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "content_recorded": False,
            "size_recorded": False,
            "digest_recorded": False,
        },
        "provider_or_model_credential_observed": False,
        "discord_credential_observed": False,
    }
    return {
        **unsigned,
        "host_facts_sha256": mechanical_job_rail._sha256_bytes(
            mechanical_job_rail._canonical(unsigned)
        ),
    }


def _mechanical_package(host_facts: dict) -> dict:
    release = (
        mechanical_job_rail.RELEASES_ROOT / f"hermes-agent-{REVISION[:12]}"
    )
    unsigned = {
        "schema": mechanical_job_rail.MANIFEST_SCHEMA,
        "rail_schema": mechanical_job_rail.RAIL_SCHEMA,
        "release_revision": REVISION,
        "release_root": str(release),
        "job_allowlist": [{
            "job_id": mechanical_job_rail.JOB_ID,
            "argv": ["--execute"],
            "routine": str(release / mechanical_job_rail.ROUTINE_RELATIVE),
            "routine_sha256": "a" * 64,
            "hardening": str(release / mechanical_job_rail.HARDENING_RELATIVE),
            "hardening_sha256": "b" * 64,
            "fork_repository": "lomliev/hermes-agent",
            "upstream_repository_read_only": "NousResearch/hermes-agent",
            "auto_merge_or_deploy_enabled": False,
        }],
        "rail_sha256": "c" * 64,
        "host_facts_sha256": host_facts["host_facts_sha256"],
        "host_binaries": {
            str(mechanical_job_rail.GH_PATH): host_facts["github_cli"]["sha256"],
            str(mechanical_job_rail.GIT_PATH): host_facts["git"]["sha256"],
        },
        "units": {
            mechanical_job_rail.SERVICE_UNIT: "d" * 64,
            mechanical_job_rail.TIMER_UNIT: "e" * 64,
        },
        "credential_name": mechanical_job_rail.CREDENTIAL_NAME,
        "credential_value_recorded": False,
        "provider_or_model_dependency": False,
        "discord_dependency": False,
        "timer_started_by_package": False,
    }
    return {
        **unsigned,
        "manifest_sha256": mechanical_job_rail._sha256_bytes(
            mechanical_job_rail._canonical(unsigned)
        ),
    }


def _cron_authority() -> tuple[dict, dict, dict, dict]:
    inventory = production_cron_migration.inventory_jobs_bytes(b'{"jobs":[]}')
    host_facts = _mechanical_host_facts()
    package = _mechanical_package(host_facts)
    plan = production_cron_migration.build_owner_approved_plan(
        inventory,
        dispositions=[],
        approval_id="11111111-1111-4111-8111-111111111111",
        approved_by="owner",
    )
    assert plan["cutover_executable"] is True
    return inventory, plan, host_facts, package


def _freeze(private: Ed25519PrivateKey, services: Services, initial_rows: int = 14_073):
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    artifacts = {
        "observe": _artifact("observe.sql", "b"),
        "database_apply": _artifact("database-apply.sql", "1"),
        "database_rollback": _artifact("database-rollback.sql", "2"),
        "database_postflight": _artifact("database-postflight.sql", "3"),
        "host_activation": _artifact("activation.json", "4"),
        "host_rollback": _artifact("activation-rollback.json", "5"),
    }
    gateway_target = _service(
        cutover.GATEWAY_UNIT,
        active=False,
        digest=services.target_gateway_digest,
        drop_in_digest=services.target_drop_in_digest,
    )
    writer_target = _service(
        cutover.WRITER_UNIT,
        active=False,
        digest=services.target_writer_digest,
    )
    connector_target = _service(
        cutover.CONNECTOR_UNIT,
        active=False,
        digest=services.target_connector_digest,
    )
    capability_topology, _capability_receipt, _boot_id = (
        _capability_receipt_pair(revision=REVISION, now_unix=NOW)
    )
    initial_snapshot = _snapshot(initial_rows)
    legacy_truth_decision = cutover.build_legacy_truth_decision(
        mode="start_new_truth_epoch",
        decision_id=(
            "legacy-truth-decision:11111111-1111-4111-8111-111111111111"
        ),
        decision_event_id="22222222-2222-4222-8222-222222222222",
        owner_subject_sha256="a" * 64,
        reviewed_snapshot=initial_snapshot,
        truth_epoch_id="truth-epoch:33333333-3333-4333-8333-333333333333",
    )
    cron_inventory, cron_plan, host_facts, mechanical_package = _cron_authority()
    authority = cutover.build_cutover_authority(
        release_revision=REVISION,
        artifacts=artifacts,
        gateway_before=services.gateway,
        writer_before=services.writer,
        connector_before=services.connector,
        gateway_target_identity=gateway_target.stable_identity(),
        writer_target_identity=writer_target.stable_identity(),
        connector_target_identity=connector_target.stable_identity(),
        host_transition=_host_transition(
            services,
            gateway_target=gateway_target,
            writer_target=writer_target,
            connector_target=connector_target,
            capability_topology=capability_topology,
        ),
        capability_topology=capability_topology,
        cron_inventory=cron_inventory,
        cron_continuity_plan=cron_plan,
        mechanical_job_host_facts=host_facts,
        mechanical_job_package=mechanical_package,
        legacy_truth_decision=legacy_truth_decision,
        max_appended_rows=10_000,
        max_capture_delay_seconds=900,
    )
    return cutover.build_freeze_plan(
        release_revision=REVISION,
        target=_target(),
        owner_subject_sha256="a" * 64,
        owner_public_key_ed25519_hex=public,
        gateway_before=services.gateway,
        writer_before=services.writer,
        connector_before=services.connector,
        initial_snapshot=initial_snapshot,
        cutover_authority=authority,
        owner_runtime_attestation=_runtime_attestation(),
    )


def _approval(private: Ed25519PrivateKey, plan, *, sequence: int = 0, previous=None):
    authority_plan = (
        cutover.FreezePlan.from_mapping(plan.value["freeze_plan"])
        if isinstance(plan, cutover.CutoverPlan)
        else plan
    )
    kind = "freeze"
    raw = {
        "schema": cutover.APPROVAL_SCHEMA,
        "plan_kind": kind,
        "purpose": f"{kind}_{'apply' if sequence == 0 else 'resume'}",
        "sequence": sequence,
        "previous_approval_sha256": previous,
        "plan_sha256": authority_plan.sha256,
        "owner_subject_sha256": authority_plan.value["owner_subject_sha256"],
        "owner_public_key_ed25519_hex": authority_plan.value["owner_public_key_ed25519_hex"],
        "owner_key_id": authority_plan.value["owner_key_id"],
        "nonce_sha256": "c" * 64,
        "issued_at_unix": NOW - 1,
        "expires_at_unix": NOW + 600,
        "approved": True,
        "signature_ed25519_hex": "0" * 128,
        "approval_sha256": "0" * 64,
    }
    raw["signature_ed25519_hex"] = private.sign(
        cutover.approval_signature_payload(raw)
    ).hex()
    raw["approval_sha256"] = cutover._sha256_json(
        {key: value for key, value in raw.items() if key != "approval_sha256"}
    )
    return raw


def _capture(private, services, *, initial_rows=14_073, final_rows=14_081):
    plan = _freeze(private, services, initial_rows)
    journal = MemoryJournal()
    receipt = cutover.execute_final_tail_capture(
        plan,
        _approval(private, plan),
        cutover.FreezeDependencies(
            services=services,
            snapshots=Snapshots(_snapshot(final_rows, marker="2")),
            journal=journal,
            lock=nullcontext,
        ),
        now_unix=NOW,
    )
    return plan, receipt, journal


def _host_transition(
    services: Services,
    *,
    gateway_target: cutover.ServiceObservation,
    writer_target: cutover.ServiceObservation,
    connector_target: cutover.ServiceObservation,
    capability_topology: dict,
) -> dict:
    staged = cutover.EVIDENCE_ROOT / "staged" / "host"
    absent = {
        "state": "absent",
        "sha256": None,
        "uid": None,
        "gid": None,
        "mode": None,
    }
    file_specs = {
        "gateway_unit": (
            cutover.GATEWAY_FRAGMENT,
            gateway_target.value["fragment_sha256"],
            0,
            0,
            0o644,
            {
                "state": "present",
                "sha256": services.legacy_gateway_digest,
                "uid": 0,
                "gid": 0,
                "mode": 0o644,
            },
        ),
        "writer_unit": (
            cutover.WRITER_FRAGMENT,
            writer_target.value["fragment_sha256"],
            0,
            0,
            0o644,
            absent,
        ),
        "connector_unit": (
            cutover.CONNECTOR_FRAGMENT,
            connector_target.value["fragment_sha256"],
            0,
            0,
            0o644,
            absent,
        ),
        "phase_b_unit": (
            cutover.PHASE_B_FRAGMENT,
            capability_topology["phase_b"]["fragment_sha256"],
            0,
            0,
            0o644,
            absent,
        ),
        "routeback_unit": (
            cutover.ROUTEBACK_EDGE_FRAGMENT,
            capability_topology["routeback_edge"]["fragment_sha256"],
            0,
            0,
            0o644,
            absent,
        ),
        "mac_ops_unit": (
            cutover.MAC_OPS_FRAGMENT,
            capability_topology["mac_ops"]["fragment_sha256"],
            0,
            0,
            0o644,
            absent,
        ),
        "browser_unit": (
            cutover.BROWSER_FRAGMENT,
            capability_topology["browser"]["fragment_sha256"],
            0,
            0,
            0o644,
            absent,
        ),
        "browser_config": (
            cutover.BROWSER_CONFIG,
            capability_topology["browser"]["config_sha256"],
            0,
            capability_topology["browser"]["service_gid"],
            0o440,
            absent,
        ),
        "isolated_worker_socket_unit": (
            cutover.ISOLATED_WORKER_SOCKET_FRAGMENT,
            capability_topology["isolated_worker"][
                "socket_fragment_sha256"
            ],
            0,
            0,
            0o644,
            absent,
        ),
        "isolated_worker_service_unit": (
            cutover.ISOLATED_WORKER_SERVICE_FRAGMENT,
            capability_topology["isolated_worker"][
                "service_fragment_sha256"
            ],
            0,
            0,
            0o644,
            absent,
        ),
        "isolated_worker_config": (
            cutover.ISOLATED_WORKER_CONFIG_PATH,
            capability_topology["isolated_worker"]["config_sha256"],
            0,
            capability_topology["isolated_worker"]["server_gid"],
            0o440,
            absent,
        ),
        "gateway_connector_drop_in": (
            cutover.GATEWAY_CONNECTOR_DROP_IN,
            services.target_drop_in_digest,
            0,
            0,
            0o644,
            absent,
        ),
        "gateway_config": (
            cutover.GATEWAY_CONFIG_PATH,
            "a" * 64,
            capability_topology["gateway_identity"]["uid"],
            capability_topology["gateway_identity"]["gid"],
            0o640,
            {
                "state": "present",
                "sha256": "c" * 64,
                "uid": capability_topology["gateway_identity"]["uid"],
                "gid": capability_topology["gateway_identity"]["gid"],
                "mode": 0o640,
            },
        ),
        "writer_config": (
            cutover.WRITER_CONFIG_PATH,
            "e" * 64,
            0,
            2000,
            0o440,
            absent,
        ),
        "connector_config": (
            cutover.CONNECTOR_CONFIG_PATH,
            "b" * 64,
            0,
            2001,
            0o440,
            absent,
        ),
        "routeback_config": (
            str(cutover.ROUTEBACK_EDGE_CONFIG_PATH),
            capability_topology["routeback_edge"]["config_sha256"],
            0,
            2002,
            0o440,
            absent,
        ),
        "mac_ops_config": (
            str(cutover.MAC_OPS_CONFIG_PATH),
            capability_topology["mac_ops"]["config_sha256"],
            0,
            2003,
            0o440,
            absent,
        ),
        "api_bearer_verifier": (
            str(cutover.API_SERVER_CREDENTIAL_PATH),
            "8" * 64,
            0,
            0,
            0o400,
            absent,
        ),
        "api_approval_verifier": (
            str(cutover.API_APPROVAL_CREDENTIAL_PATH),
            "9" * 64,
            0,
            0,
            0o400,
            absent,
        ),
    }
    for domain in sorted(cutover.CREDENTIALS_BY_DOMAIN):
        file_specs[f"operational_edge_unit_{domain}"] = (
            f"/etc/systemd/system/{cutover.operational_edge_service_unit(domain)}",
            cutover._sha256_json({"unit": domain}),
            0,
            0,
            0o644,
            absent,
        )
        file_specs[f"operational_edge_config_{domain}"] = (
            str(cutover.operational_edge_config_path(domain)),
            cutover._sha256_json({"config": domain}),
            0,
            0,
            0o400,
            absent,
        )
    file_specs["operational_edge_client_config"] = (
        str(cutover.OPERATIONAL_EDGE_CLIENT_CONFIG_PATH),
        cutover._sha256_json({"client": "operational_edge"}),
        0,
        0,
        0o444,
        absent,
    )
    files = {
        name: {
            "staged_path": str(staged / Path(target).name),
            "target_path": target,
            "sha256": digest,
            "uid": uid,
            "gid": gid,
            "mode": mode,
            "pre": copy.deepcopy(pre),
        }
        for name, (target, digest, uid, gid, mode, pre) in file_specs.items()
    }
    operational_domains = sorted(_operational_receipt_key_ids())
    operational_identities = {
        domain: {
            "name": f"muncho-edge-{domain}",
            "uid": 2100 + index,
            "gid": 2100 + index,
            "socket_group": f"muncho-edge-{domain}-c",
            "socket_gid": 2200 + index,
        }
        for index, domain in enumerate(operational_domains)
    }
    group_specs = {
        "gateway": (
            "ai-platform-brain",
            capability_topology["gateway_identity"]["gid"],
            [],
        ),
        "writer": ("muncho-canonical-writer", 2000, []),
        "projector": (
            "muncho-projector",
            2004,
            ["muncho-canonical-writer"],
        ),
        "writer_client": (
            "muncho-writer-client",
            2005,
            ["ai-platform-brain"],
        ),
        "routeback": (
            "muncho-discord-egress",
            2002,
            ["ai-platform-brain"],
        ),
        "connector": (
            "muncho-discord-connector",
            2001,
            ["ai-platform-brain"],
        ),
        "mac_ops": (
            "muncho-mac-ops-edge", 2003, ["ai-platform-brain"]
        ),
        "browser": (
            "muncho-capability-browser",
            capability_topology["browser"]["service_gid"],
            ["ai-platform-brain"],
        ),
        "worker": (
            "muncho-worker",
            capability_topology["isolated_worker"]["server_gid"],
            [],
        ),
        "worker_client": (
            "muncho-worker-clients",
            capability_topology["isolated_worker"]["socket_gid"],
            ["ai-platform-brain"],
        ),
        **{
            f"operational_edge_{domain}": (
                item["name"], item["gid"], []
            )
            for domain, item in operational_identities.items()
        },
        **{
            f"operational_edge_{domain}_client": (
                item["socket_group"],
                item["socket_gid"],
                ["ai-platform-brain", item["name"]],
            )
            for domain, item in operational_identities.items()
        },
    }
    groups = {}
    for role, (name, gid, members) in group_specs.items():
        present = role == "gateway"
        groups[role] = {
            "name": name,
            "gid": gid,
            "members": sorted(members),
            "pre": {
                "state": "present" if present else "absent",
                "gid": gid if present else None,
                "members": (
                    [] if present else None
                ),
            },
        }
    user_specs = {
        "gateway": (
            "ai-platform-brain",
            capability_topology["gateway_identity"]["uid"],
            "/opt/adventico-ai-platform/canonical-brain",
            [
                "muncho-capability-browser",
                "muncho-discord-connector",
                "muncho-discord-egress",
                "muncho-mac-ops-edge",
                *[
                    operational_identities[domain]["socket_group"]
                    for domain in operational_domains
                ],
                "muncho-worker-clients",
                "muncho-writer-client",
            ],
            ["google-sudoers"],
        ),
        "writer": (
            "muncho-canonical-writer",
            2000,
            "/nonexistent",
            ["muncho-projector"],
            None,
        ),
        "projector": (
            "muncho-projector", 2004, "/nonexistent", [], None
        ),
        "routeback": (
            "muncho-discord-egress", 2002, "/nonexistent", [], None
        ),
        "connector": (
            "muncho-discord-connector", 2001, "/nonexistent", [], None
        ),
        "mac_ops": (
            "muncho-mac-ops-edge",
            2003,
            "/nonexistent",
            [],
            None,
        ),
        "browser": (
            "muncho-capability-browser",
            capability_topology["browser"]["service_uid"],
            "/nonexistent",
            [],
            None,
        ),
        "worker": (
            "muncho-worker",
            capability_topology["isolated_worker"]["server_uid"],
            "/nonexistent",
            [],
            None,
        ),
        **{
            f"operational_edge_{domain}": (
                item["name"],
                item["uid"],
                "/nonexistent",
                [item["socket_group"]],
                None,
            )
            for domain, item in operational_identities.items()
        },
    }
    users = {}
    for role, (name, uid, home, supplementary, before) in user_specs.items():
        present = role == "gateway"
        users[role] = {
            "name": name,
            "uid": uid,
            "primary_group": role,
            "home": home,
            "shell": "/usr/sbin/nologin",
            "supplementary_groups": sorted(supplementary),
            "pre": {
                "state": "present" if present else "absent",
                "uid": uid if present else None,
                "gid": group_specs[role][1] if present else None,
                "home": home if present else None,
                "shell": "/usr/sbin/nologin" if present else None,
                "supplementary_group_names": before,
            },
        }
    identity_unsigned = {
        "schema": cutover._IDENTITY_FOUNDATION_SCHEMA,
        "users": users,
        "groups": groups,
        "retain_created_dormant_on_rollback": True,
        "secret_material_recorded": False,
    }
    identity_foundation = {
        **identity_unsigned,
        "foundation_sha256": cutover._sha256_json(identity_unsigned),
    }
    key_unsigned = {
        "schema": cutover._DISCORD_KEY_FOUNDATION_SCHEMA,
        "writer": {
            "staged_private_path": str(
                cutover.EVIDENCE_ROOT
                / "staged/keys/writer-capability-private.pem"
            ),
            "private_path": str(cutover.WRITER_CAPABILITY_PRIVATE_KEY_PATH),
            "private_uid": 2000,
            "private_gid": 2000,
            "private_mode": 0o400,
            "public_path": str(cutover.WRITER_CAPABILITY_PUBLIC_KEY_PATH),
            "public_uid": 0,
            "public_gid": 2002,
            "public_mode": 0o440,
            "public_key_id": "8" * 64,
        },
        "edge": {
            "staged_private_path": str(
                cutover.EVIDENCE_ROOT
                / "staged/keys/discord-edge-receipt-private.pem"
            ),
            "private_path": str(cutover.EDGE_RECEIPT_PRIVATE_KEY_PATH),
            "private_uid": 0,
            "private_gid": 0,
            "private_mode": 0o400,
            "public_path": str(cutover.EDGE_RECEIPT_PUBLIC_KEY_PATH),
            "public_uid": 0,
            "public_gid": 2000,
            "public_mode": 0o440,
            "public_key_id": "9" * 64,
        },
        "pre_state": "absent",
        "keys_distinct": True,
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
    }
    key_foundation = {
        **key_unsigned,
        "foundation_sha256": cutover._sha256_json(key_unsigned),
    }
    operational_key_foundation = _operational_key_foundation()
    unsigned = {
        "schema": cutover._HOST_TRANSITION_SCHEMA,
        "files": files,
        "identity_foundation": identity_foundation,
        "discord_key_foundation": key_foundation,
        "operational_edge_key_foundation": operational_key_foundation,
        "operational_edge_key_foundation_sha256": (
            operational_key_foundation["receipt_sha256"]
        ),
        "operational_edge_receipt_public_key_ids": (
            _operational_receipt_key_ids()
        ),
        "release_owner_uid": capability_topology["gateway_identity"]["uid"],
        "release_owner_gid": capability_topology["gateway_identity"]["gid"],
        "isolated_worker_lease_mountpoint": {
            "target_path": str(cutover.ISOLATED_WORKER_LEASE_BASE),
            "uid": 0,
            "gid": 0,
            "mode": 0o700,
            "pre": {
                "state": "absent",
                "uid": None,
                "gid": None,
                "mode": None,
            },
        },
        "connector_token": {
            "path": cutover.CONNECTOR_TOKEN_PATH,
            "uid": 2001,
            "gid": 2001,
            "mode": 0o400,
            "regular_one_link": True,
            "content_or_digest_recorded": False,
            "gateway_readable": False,
            "source_path": (
                "/opt/adventico-ai-platform/hermes-home/"
                ".discord-session-token"
            ),
            "source_uid": capability_topology["gateway_identity"]["uid"],
            "source_gid": capability_topology["gateway_identity"]["gid"],
            "source_mode": 0o400,
        },
        "gateway_retired_token_paths": [
            "/opt/adventico-ai-platform/hermes-home/.discord-session-token"
        ],
        "routeback_token_paths": [
            "/etc/muncho/discord-edge-credentials/bot-token"
        ],
        "approval_passkey": {
            "path": str(cutover.API_APPROVAL_CREDENTIAL_PATH),
            "uid": 0,
            "gid": 0,
            "mode": 0o400,
            "regular_one_link": True,
            "content_or_digest_recorded": False,
            "gateway_readable": False,
            "source_path": str(cutover.STAGED_APPROVAL_PASSKEY_PATH),
            "source_uid": 0,
            "source_gid": 0,
            "source_mode": 0o400,
        },
        "retired_approval_passkey_paths": [
            str(cutover.STAGED_APPROVAL_PASSKEY_PATH)
        ],
        "gateway_direct_discord_enabled": False,
        "gateway_relay_platforms": ["discord"],
        "connector_operation_class": "ordinary_guild_acl_session_only",
        "routeback_operation_class": "canonical_guild_acl_routeback_rest_only",
        "discord_dm_allowed": False,
        "discord_policy_continuity": cutover.build_discord_policy_continuity(
            source_evidence_sha256="d" * 64,
            legacy_policy={
                "allowed_guild_ids": ["1282725267068157972"],
                "allowed_channel_ids": [
                    "1504852355588423801",
                    "1504852408227069993",
                    "1504852444407140402",
                    "1504852485083496561",
                    "1504852553031221391",
                    "1504852628373373028",
                    "1505499746939174993",
                    "1507239177350283274",
                    "1507239385010016308",
                    "1507239516409167942",
                    "1510888721614901358",
                ],
                "allowed_user_ids": [
                    "1279454038731264061",
                    "1282938967888498720",
                ],
                "allowed_role_ids": [
                    "1282725267068157972",
                    "1505077218374586468",
                ],
                "allow_all_users": False,
                "allow_bot_authors": False,
                "require_mention": True,
                "auto_thread": True,
                "thread_require_mention": False,
                "discord_dm_allowed": False,
                "free_response_channel_ids": [
                    "1504852355588423801",
                    "1505499746939174993",
                ],
                "public_only": False,
                "author_policy": "exact_ids_or_roles",
            },
            target_policy={
                "allowed_guild_ids": ["1282725267068157972"],
                "allowed_channel_ids": sorted(
                    cutover.SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS
                ),
                "allowed_user_ids": [],
                "allowed_role_ids": [],
                "allow_all_users": False,
                "allow_bot_authors": False,
                "require_mention": True,
                "auto_thread": True,
                "thread_require_mention": False,
                "discord_dm_allowed": False,
                "free_response_channel_ids": [
                    "1504852355588423801",
                    "1505499746939174993",
                ],
                "public_only": False,
                "author_policy": "guild_acl",
            },
        ),
        "secret_material_recorded": False,
    }
    return {**unsigned, "manifest_sha256": cutover._sha256_json(unsigned)}


def _cutover_plan(private, services):
    freeze, tail, _journal = _capture(private, services)
    return cutover.build_cutover_plan(
        freeze_plan=freeze,
        final_tail_receipt=tail,
        gateway_stopped=services.gateway,
        writer_stopped=services.writer,
        connector_stopped=services.connector,
    )


def _db_receipt(plan, schema: str, artifact: str, *, rollback=False, override=None):
    snapshot = plan.final_snapshot.value
    decision = plan.value["legacy_truth_decision"]
    legacy_state = rollback or "preflight" in schema
    raw = {
        "schema": schema,
        "plan_sha256": plan.sha256,
        "artifact_sha256": plan.value["artifacts"][artifact]["sha256"],
        "final_snapshot_sha256": plan.final_snapshot.sha256,
        "source_row_count": snapshot["source_row_count"],
        "archive_row_count": snapshot["source_row_count"],
        "canonical_row_count": (
            0 if legacy_state else snapshot["source_row_count"] + 1
        ),
        "archive_extended19_sha256": snapshot["extended19_sha256"],
        "canonical14_sha256": snapshot["canonical14_sha256"],
        "relation_identity_sha256": snapshot["relation_identity_sha256"],
        "acl_identity_sha256": snapshot["acl_identity_sha256"],
        "index_identity_sha256": snapshot["index_identity_sha256"],
        "roles_acl_sha256": "d" * 64,
        "zero_canonical_writer_writes": True,
        "legacy_truth_mode": decision["mode"],
        "legacy_truth_decision_sha256": decision["decision_sha256"],
        "legacy_truth_decision_event_id": decision["decision_event_id"],
        "accepted_event_set_sha256": cutover._sha256_json(
            decision["accepted_event_ids"]
        ),
        "trusted_legacy_event_count": (
            0 if legacy_state else len(decision["accepted_event_ids"])
        ),
        "truth_epoch_sha256": decision["truth_epoch_sha256"],
        "legacy_shape_restored": legacy_state,
        "ok": True,
        "secret_material_recorded": False,
    }
    raw.update(override or {})
    return _hashed(raw, "receipt_sha256")


def _host_receipt(plan, schema: str, artifact: str):
    raw = {
        "schema": schema,
        "plan_sha256": plan.sha256,
        "artifact_sha256": plan.value["artifacts"][artifact]["sha256"],
        "ok": True,
        "secret_material_recorded": False,
    }
    return _hashed(raw, "receipt_sha256")


def _host_apply_receipt(plan, services: Services):
    topology = plan.value["capability_topology"]
    operational_runtime = _operational_edge_runtime(
        plan,
        unit_file_state="disabled",
    )
    raw = {
        "schema": "muncho-production-writer-host-apply.v2",
        "plan_sha256": plan.sha256,
        "artifact_sha256": plan.value["artifacts"]["host_activation"]["sha256"],
        "gateway_stopped": services.gateway.to_mapping(),
        "writer_stopped": services.writer.to_mapping(),
        "connector_stopped": services.connector.to_mapping(),
        "phase_b_stopped": _service(
            cutover.PHASE_B_UNIT,
            active=False,
            digest=topology["phase_b"]["fragment_sha256"],
            unit_file_state="disabled",
        ).to_mapping(),
        "routeback_stopped": _service(
            cutover.ROUTEBACK_EDGE_UNIT,
            active=False,
            digest=topology["routeback_edge"]["fragment_sha256"],
            unit_file_state="disabled",
        ).to_mapping(),
        "mac_ops_stopped": _service(
            cutover.MAC_OPS_UNIT,
            active=False,
            digest=topology["mac_ops"]["fragment_sha256"],
            unit_file_state="disabled",
        ).to_mapping(),
        "browser_stopped": _service(
            cutover.BROWSER_UNIT,
            active=False,
            digest=topology["browser"]["fragment_sha256"],
            unit_file_state="disabled",
        ).to_mapping(),
        "isolated_worker_socket_stopped": _service(
            cutover.ISOLATED_WORKER_SOCKET_UNIT,
            active=False,
            digest=topology["isolated_worker"]["socket_fragment_sha256"],
            unit_file_state="disabled",
        ).to_mapping(),
        "isolated_worker_service_stopped": _service(
            cutover.ISOLATED_WORKER_SERVICE_UNIT,
            active=False,
            digest=topology["isolated_worker"]["service_fragment_sha256"],
            unit_file_state="static",
        ).to_mapping(),
        "isolated_worker_lease_mountpoint_prepared": True,
        "identity_foundation_sha256": plan.value["host_transition"][
            "identity_foundation"
        ]["foundation_sha256"],
        "identity_apply_receipt_sha256": "1" * 64,
        "discord_key_foundation_sha256": plan.value["host_transition"][
            "discord_key_foundation"
        ]["foundation_sha256"],
        "discord_foundation_receipt_sha256": "2" * 64,
        "writer_public_key_id": plan.value["host_transition"][
            "discord_key_foundation"
        ]["writer"]["public_key_id"],
        "edge_public_key_id": plan.value["host_transition"][
            "discord_key_foundation"
        ]["edge"]["public_key_id"],
        "operational_edge_key_foundation_sha256": plan.value[
            "host_transition"
        ]["operational_edge_key_foundation_sha256"],
        "operational_edge_key_provision_receipt_sha256": "6" * 64,
        "operational_edge_receipt_public_key_ids": plan.value[
            "host_transition"
        ]["operational_edge_receipt_public_key_ids"],
        "operational_edge_staged_key_copies_retained": True,
        "operational_edge_keys_ready": True,
        "operational_edge_runtime": operational_runtime,
        "operational_edge_asset_readback_receipt_sha256": "7" * 64,
        "operational_edge_readiness": _operational_edge_readiness(
            plan,
            operational_runtime,
        ),
        "only_operational_edge_services_started": True,
        "normal_services_remained_stopped": True,
        "discord_journals_clean_before_service_start": True,
        "normal_startup_journal_creation_disabled": True,
        "preexisting_memberships_converged_to_allowlist": True,
        "direct_discord_disabled": True,
        "discord_dm_allowed": False,
        "api_verifier_foundation_receipt_sha256": "5" * 64,
        "api_bearer_verifier_installed": True,
        "api_approval_verifier_installed": True,
        "api_source_secrets_root_only": True,
        "api_source_secrets_loaded_by_gateway": False,
        "ok": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(raw, "receipt_sha256")


def _host_boot_commit_receipt(plan, services: Services):
    topology = plan.value["capability_topology"]
    raw = {
        "schema": "muncho-production-host-boot-commit.v2",
        "plan_sha256": plan.sha256,
        "artifact_sha256": plan.value["artifacts"]["host_activation"]["sha256"],
        "gateway_observed": services.gateway.to_mapping(),
        "writer_active": services.writer.to_mapping(),
        "connector_active": services.connector.to_mapping(),
        "phase_b_active": _service(
            cutover.PHASE_B_UNIT,
            active=True,
            digest=topology["phase_b"]["fragment_sha256"],
            sub_state="exited",
            main_pid=0,
        ).to_mapping(),
        "routeback_active": _service(
            cutover.ROUTEBACK_EDGE_UNIT,
            active=True,
            digest=topology["routeback_edge"]["fragment_sha256"],
        ).to_mapping(),
        "mac_ops_active": _service(
            cutover.MAC_OPS_UNIT,
            active=True,
            digest=topology["mac_ops"]["fragment_sha256"],
        ).to_mapping(),
        "browser_active": _service(
            cutover.BROWSER_UNIT,
            active=True,
            digest=topology["browser"]["fragment_sha256"],
        ).to_mapping(),
        "isolated_worker_socket_active": _service(
            cutover.ISOLATED_WORKER_SOCKET_UNIT,
            active=True,
            digest=topology["isolated_worker"]["socket_fragment_sha256"],
            sub_state="listening",
            main_pid=0,
        ).to_mapping(),
        "isolated_worker_service_active": _service(
            cutover.ISOLATED_WORKER_SERVICE_UNIT,
            active=True,
            digest=topology["isolated_worker"]["service_fragment_sha256"],
            unit_file_state="static",
        ).to_mapping(),
        "operational_edge_runtime": _operational_edge_runtime(
            plan,
            unit_file_state="enabled",
        ),
        "boot_fence_released": True,
        "ok": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(raw, "receipt_sha256")


def _host_rollback_receipt(plan):
    raw = {
        "schema": "muncho-production-writer-host-rollback.v2",
        "plan_sha256": plan.sha256,
        "artifact_sha256": plan.value["artifacts"]["host_rollback"]["sha256"],
        "api_verifiers_removed": True,
        "api_source_secrets_preserved": True,
        "phase_b_removed": True,
        "routeback_removed": True,
        "mac_ops_removed": True,
        "browser_removed": True,
        "isolated_worker_socket_removed": True,
        "isolated_worker_service_removed": True,
        "isolated_worker_lease_mountpoint_restored": True,
        "identity_foundation_sha256": plan.value["host_transition"][
            "identity_foundation"
        ]["foundation_sha256"],
        "identity_rollback_receipt_sha256": "2" * 64,
        "discord_key_rollback_receipt_sha256": "3" * 64,
        "operational_edge_key_retention_receipt_sha256": "6" * 64,
        "operational_edge_keys_retained_dormant": True,
        "operational_edge_staged_key_copies_retained": True,
        "operational_edge_units_removed": [
            cutover.operational_edge_service_unit(domain)
            for domain in sorted(_operational_receipt_key_ids())
        ],
        "operational_edge_readiness_removed": True,
        "discord_journal_rollback_receipt_sha256": "4" * 64,
        "staged_private_keys_restored": True,
        "discord_journals_removed": True,
        "created_identities_retained_dormant": True,
        "preexisting_memberships_restored": True,
        "ok": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(raw, "receipt_sha256")


class Database:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.calls = []
        self.apply_override = None

    def preflight(self, plan):
        self.calls.append("preflight")
        return _db_receipt(plan, "muncho-production-legacy-cutover-preflight.v1", "database_postflight")

    def apply(self, plan):
        self.calls.append("apply")
        return _db_receipt(plan, "muncho-production-legacy-database-apply.v1", "database_apply", override=self.apply_override)

    def terminal(self, plan):
        self.calls.append("terminal")
        return _db_receipt(plan, "muncho-production-legacy-cutover-postflight.v1", "database_postflight")

    def rollback(self, plan, _receipt):
        self.calls.append("rollback")
        return _db_receipt(plan, "muncho-production-legacy-database-rollback.v1", "database_rollback", rollback=True)


class ApplyResponseLostDatabase(Database):
    def __init__(self, plan) -> None:
        super().__init__(plan)
        self.rollback_authority = "unset"

    def apply(self, plan):
        self.calls.append("apply")
        raise RuntimeError("injected response loss after database apply")

    def rollback(self, plan, receipt):
        self.rollback_authority = receipt
        return super().rollback(plan, receipt)


class Host:
    def __init__(self, plan, services: Services, *, fail_start=False) -> None:
        self.plan = plan
        self.services = services
        self.fail_start = fail_start
        self.calls = []

    def apply_stopped(self, plan):
        self.calls.append("apply")
        self.services.target_installed = True
        self.services.boot_enabled = False
        self.services.stop_gateway()
        self.services.stop_writer()
        self.services.stop_connector()
        return _host_apply_receipt(plan, self.services)

    def start_writer(self, plan):
        self.calls.append("start_writer")
        if self.fail_start:
            raise RuntimeError("injected writer start failure")
        self.services.writer = _service(
            cutover.WRITER_UNIT,
            active=True,
            digest=self.services.target_writer_digest,
            unit_file_state="disabled",
        )
        return _host_receipt(plan, "muncho-production-writer-start.v1", "host_activation")

    def start_prerequisites(self, plan):
        self.calls.append("start_prerequisites")
        self.services.connector = _service(
            cutover.CONNECTOR_UNIT,
            active=True,
            digest=self.services.target_connector_digest,
            unit_file_state="disabled",
        )
        return _host_receipt(
            plan,
            "muncho-production-prerequisite-services-start.v1",
            "host_activation",
        )

    def commit_boot(self, plan):
        self.calls.append("commit_boot")
        self.services.commit_boot()
        return _host_boot_commit_receipt(plan, self.services)

    def rollback(self, plan, _receipt):
        self.calls.append("rollback")
        self.services.target_installed = False
        self.services.boot_enabled = False
        self.services.stop_gateway()
        self.services.stop_writer()
        self.services.stop_connector()
        return _host_rollback_receipt(plan)


class Prerequisites:
    def __init__(self, *, drift: bool = False) -> None:
        self.drift = drift
        self.calls = 0

    def collect_and_validate(self, plan):
        self.calls += 1
        if self.drift:
            raise cutover.ProductionCutoverError(
                "production_capability_prerequisite_drifted"
            )
        unsigned = {
            "schema": "muncho-production-capability-prerequisite-acceptance.v1",
            "plan_sha256": plan.sha256,
            "prerequisite_receipt_sha256": "8" * 64,
            "prerequisite_file_sha256": "9" * 64,
            "topology_identity_sha256": (
                cutover.production_capability_topology_identity_sha256(
                    plan.value["capability_topology"]
                )
            ),
            "boot_id_sha256": "a" * 64,
            "ok": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        return _hashed(unsigned, "receipt_sha256")


def _packaged_coordinator_plan(plan: cutover.CutoverPlan) -> cutover.CutoverPlan:
    value = copy.deepcopy(plan.value)
    value["cron_continuity_plan"] = {
        "schema": production_cron_continuity_package.PLAN_SCHEMA,
        "plan_sha256": "c" * 64,
        "source_store_sha256": "1" * 64,
        "trusted_collector_package": {
            "release_root": str(
                cutover.PRODUCTION_RELEASE_BASE
                / f"hermes-agent-{REVISION[:12]}"
            )
        },
        "cutover_runtime_sha256": "d" * 64,
        "cutover_entrypoint_sha256": "e" * 64,
    }
    # The production entrypoint always reconstructs CutoverPlan from the exact
    # staged mapping.  This focused coordinator fixture changes only the
    # already-validated cron member so the state machine can be exercised
    # without reproducing the 28-record private package in this unit test.
    return cutover.CutoverPlan(value)


def _cron_receipt(
    action: str,
    plan: cutover.CutoverPlan,
    *,
    prior: str | None = None,
    activation_authority_sha256: str | None = None,
) -> dict:
    common = {
        "created_at": "2026-07-15T00:00:00Z",
        "cutover_plan_sha256": plan.sha256,
        "continuity_plan_sha256": "c" * 64,
        "artifact_index_sha256": "f" * 64,
        "source_store_sha256": "1" * 64,
        "provider_or_model_invoked": False,
        "discord_delivery_attempted": False,
        "secret_material_recorded": False,
    }
    if action == "preflight":
        unsigned = {
            "schema": production_cron_cutover_runtime.PREFLIGHT_SCHEMA,
            **common,
            "expected_target_store_sha256": "2" * 64,
            "collector_timer_count": 21,
            "gateway_writer_connector_stopped": True,
            "artifacts_valid": True,
            "source_store_unchanged": True,
            "prepared_recovery_sha256": "3" * 64,
            "jobs_archive_sha256": "1" * 64,
            "host_snapshot_sha256": "4" * 64,
            "spool_prestate_sha256": "5" * 64,
            "manifest_directory_prestate_sha256": "6" * 64,
            "legacy_auto_sync_disabled": True,
            "legacy_auto_sync_no_active_claim": True,
            "legacy_auto_sync_next_run_reconciled": True,
            "collector_execution_readiness_sha256": "7" * 64,
            "operational_edge_readiness_receipt_sha256": "8" * 64,
            "operational_edge_boot_id_sha256": "9" * 64,
            "operational_edge_observed_at_unix": 1_800_000_000,
            "operational_edge_maximum_age_seconds": 120,
            "operational_edge_collector_nonce": (
                "f1438b18-df67-46ea-ae46-5e4f3f863f09"
            ),
            "operational_edge_meaningful_packet_count": 14,
            "collector_execution_ready": True,
            "recovery_evidence_persisted": True,
            "runtime_target_mutation_performed": False,
        }
    elif action == "apply":
        unsigned = {
            "schema": production_cron_cutover_runtime.APPLY_SCHEMA,
            **common,
            "preflight_receipt_sha256": prior,
            "target_store_sha256": "2" * 64,
            "jobs_archive_path": "/root/jobs.json.archive",
            "jobs_archive_sha256": "1" * 64,
            "host_snapshot_path": "/root/host-snapshot.json",
            "host_snapshot_sha256": "4" * 64,
            "replacement_agent_record_count": 24,
            "collector_only_inert_record_count": 2,
            "preserved_inert_record_count": 1,
            "collector_unit_file_count": 42,
            "collector_timer_count": 21,
            "collector_manifest_installed": True,
            "collector_timers_disabled": True,
            "collector_timers_active": False,
            "collector_execution_readiness_sha256": "7" * 64,
            "operational_edge_readiness_receipt_sha256": "8" * 64,
            "operational_edge_boot_id_sha256": "9" * 64,
            "operational_edge_observed_at_unix": 1_800_000_000,
            "operational_edge_maximum_age_seconds": 120,
            "operational_edge_collector_nonce": (
                "f1438b18-df67-46ea-ae46-5e4f3f863f09"
            ),
            "operational_edge_meaningful_packet_count": 14,
            "collector_execution_ready": True,
            "service_identity_reused_from_owner_bound_foundation": True,
            "records_deleted": False,
            "jobs_executed": False,
        }
    elif action == "postflight":
        unsigned = {
            "schema": production_cron_cutover_runtime.POSTFLIGHT_SCHEMA,
            **common,
            "apply_receipt_sha256": prior,
            "target_store_sha256": "2" * 64,
            "jobs_store_matches_target": True,
            "collector_manifest_matches": True,
            "collector_units_match": True,
            "collector_timers_disabled": True,
            "collector_timers_active": False,
            "packet_root_gateway_readable": True,
            "production_mutation_performed": False,
        }
    elif action == "activation":
        unsigned = {
            "schema": production_cron_cutover_runtime.ACTIVATION_SCHEMA,
            **common,
            "postflight_receipt_sha256": prior,
            "activation_authority_sha256": activation_authority_sha256,
            "target_store_sha256": "2" * 64,
            "gateway_writer_connector_active": True,
            "collector_timer_count": 21,
            "collector_timers_enabled": True,
            "collector_timers_active": True,
            "jobs_executed_by_activation_action": False,
        }
    elif action == "rollback":
        unsigned = {
            "schema": production_cron_cutover_runtime.ROLLBACK_SCHEMA,
            **common,
            "apply_receipt_sha256": prior,
            "restored_store_sha256": "1" * 64,
            "collector_timers_disabled": True,
            "collector_timers_stopped": True,
            "host_file_prestate_restored": True,
            "jobs_store_byte_exact_restored": True,
            "collector_spool_prestate_restored": True,
            "collector_manifest_directory_prestate_restored": True,
            "owner_bound_service_identity_unchanged": True,
        }
    else:  # pragma: no cover - helper misuse
        raise AssertionError(action)
    return _hashed(unsigned, "receipt_sha256")


class CronBoundary:
    def __init__(self, services: Services, trace: list[str] | None = None) -> None:
        self.services = services
        self.trace = [] if trace is None else trace
        self.calls: list[str] = []
        self.preflight_receipt = None
        self.apply_receipt = None
        self.postflight_receipt = None
        self.activation_authority = None

    def _stopped(self) -> None:
        assert self.services.gateway.stopped
        assert self.services.writer.stopped
        assert self.services.connector.stopped

    def preflight(self, plan):
        self._stopped()
        self.calls.append("preflight")
        self.preflight_receipt = _cron_receipt("preflight", plan)
        return self.preflight_receipt

    def apply(self, plan, preflight_receipt):
        self._stopped()
        assert preflight_receipt == self.preflight_receipt
        self.calls.append("apply")
        self.apply_receipt = _cron_receipt(
            "apply", plan, prior=preflight_receipt["receipt_sha256"]
        )
        return self.apply_receipt

    def postflight(self, plan, apply_receipt):
        self._stopped()
        assert apply_receipt == self.apply_receipt
        self.calls.append("postflight")
        self.postflight_receipt = _cron_receipt(
            "postflight", plan, prior=apply_receipt["receipt_sha256"]
        )
        return self.postflight_receipt

    def activate(self, plan, postflight_receipt, activation_authority):
        assert self.services.gateway.value["active_state"] == "active"
        assert self.services.writer.value["active_state"] == "active"
        assert self.services.connector.value["active_state"] == "active"
        assert postflight_receipt == self.postflight_receipt
        self.calls.append("activate")
        self.activation_authority = activation_authority
        return _cron_receipt(
            "activation",
            plan,
            prior=postflight_receipt["receipt_sha256"],
            activation_authority_sha256=activation_authority[
                "authority_sha256"
            ],
        )

    def rollback(self, plan, apply_receipt):
        self._stopped()
        self.calls.append("rollback")
        self.trace.append("cron")
        return _cron_receipt(
            "rollback",
            plan,
            prior=(
                None
                if apply_receipt is None
                else apply_receipt["receipt_sha256"]
            ),
        )


def test_freeze_stops_exact_gateway_and_captures_final_append_only_tail() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan, receipt, journal = _capture(private, services)
    assert services.gateway.stopped and services.writer.stopped
    assert receipt.value["initial_row_count"] == 14_073
    assert receipt.value["final_row_count"] == 14_081
    assert [entry.value["event"] for entry in journal.load(plan.sha256)] == [
        "authority", "gateway_stopped", "final_tail_captured"
    ]
    assert cutover.execute_final_tail_capture(
        plan, _approval(private, plan),
        cutover.FreezeDependencies(services, Snapshots(receipt.snapshot), journal, nullcontext),
        now_unix=NOW,
    ).sha256 == receipt.sha256


def test_freeze_authority_rejects_nonexecutable_blanket_cron_disposition() -> None:
    private = Ed25519PrivateKey.generate()
    freeze = _freeze(private, Services())
    jobs = []
    for job_id, overrides in (
        ("script-job", {"script": "collect.py"}),
        ("workdir-job", {"workdir": "/srv/legacy"}),
    ):
        job = {
            "id": job_id,
            "name": "Reviewed model task",
            "enabled": True,
            "no_agent": False,
            "prompt": "Inspect the evidence and decide the next step.",
            "script": None,
            "workdir": None,
            "deliver": "local",
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "base_url": None,
            "enabled_toolsets": None,
        }
        job.update(overrides)
        jobs.append(job)
    inventory = production_cron_migration.inventory_jobs_bytes(
        json.dumps({"jobs": jobs}).encode()
    )
    plan = production_cron_migration.build_owner_approved_plan(
        inventory,
        dispositions=[
            {
                "index": row["index"],
                "record_sha256": row["record_sha256"],
                "disposition": "preserve_inert",
                "target": None,
            }
            for row in inventory["incompatible_enabled_records"]
        ],
        approval_id="11111111-1111-4111-8111-111111111111",
        approved_by="owner",
    )
    assert plan["cutover_executable"] is False
    raw = freeze.to_mapping()
    authority = copy.deepcopy(raw["cutover_authority"])
    authority["cron_inventory"] = inventory
    authority["cron_continuity_plan"] = plan
    authority["authority_sha256"] = cutover._sha256_json({
        name: item
        for name, item in authority.items()
        if name != "authority_sha256"
    })
    raw["cutover_authority"] = authority
    raw["plan_sha256"] = cutover._sha256_json({
        name: item for name, item in raw.items() if name != "plan_sha256"
    })

    with pytest.raises(
        ValueError,
        match="cron continuity authority is not executable",
    ):
        cutover.FreezePlan.from_mapping(raw)


def test_freeze_rejects_row_floor_regression_after_stopping_gateway() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _freeze(private, services, initial_rows=100)
    journal = MemoryJournal()
    with pytest.raises(cutover.ProductionCutoverError, match="row_floor"):
        cutover.execute_final_tail_capture(
            plan, _approval(private, plan),
            cutover.FreezeDependencies(
                services,
                Snapshots(_snapshot(99)),
                journal,
                nullcontext,
            ),
            now_unix=NOW,
        )
    assert not services.gateway.stopped
    assert services.calls[-1] == "start_gateway"
    entries = journal.load(plan.sha256)
    assert entries[-1].value["event"] == "freeze_aborted"
    assert entries[-1].value["evidence"]["database_mutated"] is False
    assert entries[-1].value["evidence"]["host_mutated"] is False


def test_explicit_pre_mutation_freeze_abort_recovers_after_process_crash() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _freeze(private, services, initial_rows=100)
    approval_value = _approval(private, plan)
    approval = cutover.CutoverApproval.from_mapping(
        approval_value,
        plan=plan,
        now_unix=NOW,
    )
    journal = MemoryJournal()
    cutover._append_authority(journal, plan.sha256, approval, NOW)
    services.stop_writer()
    services.stop_connector()
    services.stop_gateway()
    dependencies = cutover.FreezeDependencies(
        services,
        Snapshots(_snapshot(101)),
        journal,
        nullcontext,
    )

    receipt = cutover.abort_freeze(
        plan,
        approval_value,
        dependencies,
        cutover_plan_staged=False,
        now_unix=NOW + 3_600,
    )
    retried = cutover.abort_freeze(
        plan,
        approval_value,
        dependencies,
        cutover_plan_staged=False,
        now_unix=NOW + 3_601,
    )

    assert receipt == retried
    assert receipt["trigger"] == "owner_abort"
    assert receipt["gateway_legacy_restarted"] is True
    assert receipt["database_mutated"] is False
    assert receipt["host_mutated"] is False
    assert not services.gateway.stopped
    assert services.writer.stopped
    assert services.connector.stopped
    assert services.calls.count("start_gateway") == 1


def test_freeze_abort_rejects_any_staged_cutover_plan() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _freeze(private, services, initial_rows=100)
    approval_value = _approval(private, plan)
    journal = MemoryJournal()
    approval = cutover.CutoverApproval.from_mapping(
        approval_value,
        plan=plan,
        now_unix=NOW,
    )
    cutover._append_authority(journal, plan.sha256, approval, NOW)

    with pytest.raises(
        cutover.ProductionCutoverError,
        match="cutover_plan_present",
    ):
        cutover.abort_freeze(
            plan,
            approval_value,
            cutover.FreezeDependencies(
                services,
                Snapshots(_snapshot(101)),
                journal,
                nullcontext,
            ),
            cutover_plan_staged=True,
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relation_identity_sha256", "a" * 64),
        ("acl_identity_sha256", "b" * 64),
        ("index_identity_sha256", "c" * 64),
    ],
)
def test_freeze_rejects_exact_relation_acl_or_index_identity_drift(
    field: str,
    value: str,
) -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _freeze(private, services, initial_rows=100)
    with pytest.raises(cutover.ProductionCutoverError, match="storage_identity"):
        cutover.execute_final_tail_capture(
            plan,
            _approval(private, plan),
            cutover.FreezeDependencies(
                services,
                Snapshots(
                    _snapshot(101, identity_override={field: value})
                ),
                MemoryJournal(),
                nullcontext,
            ),
            now_unix=NOW,
        )


def test_cutover_plan_binds_separate_database_and_host_rollback_digests() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    contract = plan.value["rollback_contract"]
    assert contract["database_rollback_sha256"] == "2" * 64
    assert contract["host_rollback_sha256"] == "5" * 64
    assert contract["requires_connector_stopped"] is True
    assert contract["restart_legacy_gateway"] is True
    transition = plan.value["host_transition"]
    assert transition["gateway_direct_discord_enabled"] is False
    assert transition["discord_dm_allowed"] is False
    assert set(transition["files"]) == set(cutover._HOST_TRANSITION_FILE_NAMES)
    assert (
        transition["files"]["gateway_connector_drop_in"]["sha256"]
        == services.target_drop_in_digest
    )
    assert transition["connector_token"]["source_path"] in transition[
        "gateway_retired_token_paths"
    ]
    assert transition["approval_passkey"]["path"] == str(
        cutover.API_APPROVAL_CREDENTIAL_PATH
    )
    assert transition["approval_passkey"]["source_path"] in transition[
        "retired_approval_passkey_paths"
    ]
    tampered = plan.to_mapping()
    tampered["artifacts"]["database_rollback"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="self-digest"):
        cutover.CutoverPlan.from_mapping(tampered)


def test_owner_approval_is_exact_plan_bound_and_cryptographically_verified() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    approval = _approval(private, plan)
    assert cutover.CutoverApproval.from_mapping(approval, plan=plan, now_unix=NOW).sha256
    altered = copy.deepcopy(approval)
    altered["nonce_sha256"] = "e" * 64
    altered["approval_sha256"] = cutover._sha256_json({k: v for k, v in altered.items() if k != "approval_sha256"})
    with pytest.raises(PermissionError, match="signature"):
        cutover.CutoverApproval.from_mapping(altered, plan=plan, now_unix=NOW)


def test_successful_cutover_preserves_full_archive_and_canonical_digest_proofs() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    host = Host(plan, services)
    journal = MemoryJournal()
    terminal = cutover.execute_cutover(
        plan, _approval(private, plan),
        cutover.CutoverDependencies(
            services, Snapshots(plan.final_snapshot), database, host, journal,
            Prerequisites(), nullcontext
        ),
        now_unix=NOW,
    )
    assert terminal["schema"] == cutover.TERMINAL_SCHEMA
    assert terminal["rollback_used"] is False
    assert terminal["host_boot_commit_receipt_sha256"]
    assert services.gateway.value["active_state"] == "active"
    assert services.writer.value["active_state"] == "active"
    assert services.connector.value["active_state"] == "active"
    assert database.calls == ["preflight", "apply", "terminal"]
    assert cutover.execute_cutover(
        plan, _approval(private, plan),
        cutover.CutoverDependencies(
            services, Snapshots(plan.final_snapshot), database, host, journal,
            Prerequisites(), nullcontext
        ),
        now_unix=NOW,
    )["receipt_sha256"] == terminal["receipt_sha256"]
    assert database.calls == ["preflight", "apply", "terminal"]


def test_packaged_cron_cutover_is_ordered_and_journal_authority_is_exact() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _packaged_coordinator_plan(_cutover_plan(private, services))
    database = Database(plan)
    host = Host(plan, services)
    cron = CronBoundary(services)
    journal = MemoryJournal()

    terminal = cutover.execute_cutover(
        plan,
        _approval(private, plan),
        cutover.CutoverDependencies(
            services,
            Snapshots(plan.final_snapshot),
            database,
            host,
            journal,
            Prerequisites(),
            nullcontext,
            cron=cron,
        ),
        now_unix=NOW,
    )

    assert terminal["schema"] == cutover.TERMINAL_SCHEMA
    assert cron.calls == ["preflight", "apply", "postflight", "activate"]
    entries = journal.load(plan.sha256)
    events = [entry.value["event"] for entry in entries]
    assert events.index("host_applied") < events.index("cron_preflight_validated")
    assert events.index("cron_postflight_validated") < events.index(
        "prerequisites_started"
    )
    assert events.index("cron_postflight_validated") < events.index(
        "activation_commit_intent"
    )
    assert events.index("gateway_started") < events.index(
        "cron_activation_authority"
    )
    assert events.index("cron_activated") < events.index("terminal")
    expected_entries = {
        entry.value["event"]: entry.sha256
        for entry in entries
        if entry.value["event"] in {
            "database_terminal_validated",
            "activation_commit_intent",
            "boot_committed",
            "gateway_started",
        }
    }
    assert cron.activation_authority == (
        production_cron_cutover_runtime.build_activation_authority(
            cutover_plan_sha256=plan.sha256,
            cron_postflight_receipt_sha256=cron.postflight_receipt[
                "receipt_sha256"
            ],
            database_terminal_entry_sha256=expected_entries[
                "database_terminal_validated"
            ],
            activation_commit_intent_entry_sha256=expected_entries[
                "activation_commit_intent"
            ],
            boot_committed_entry_sha256=expected_entries["boot_committed"],
            gateway_started_entry_sha256=expected_entries["gateway_started"],
        )
    )
    assert cutover.execute_cutover(
        plan,
        _approval(private, plan),
        cutover.CutoverDependencies(
            services,
            Snapshots(plan.final_snapshot),
            database,
            host,
            journal,
            Prerequisites(),
            nullcontext,
            cron=cron,
        ),
        now_unix=NOW,
    )["receipt_sha256"] == terminal["receipt_sha256"]
    assert cron.calls == ["preflight", "apply", "postflight", "activate"]


def test_legacy_cron_plan_keeps_optional_boundary_as_strict_noop() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    cron = CronBoundary(services)
    terminal = cutover.execute_cutover(
        plan,
        _approval(private, plan),
        cutover.CutoverDependencies(
            services,
            Snapshots(plan.final_snapshot),
            Database(plan),
            Host(plan, services),
            MemoryJournal(),
            Prerequisites(),
            nullcontext,
            cron=cron,
        ),
        now_unix=NOW,
    )
    assert terminal["schema"] == cutover.TERMINAL_SCHEMA
    assert cron.calls == []


def test_packaged_cron_rollback_precedes_host_and_database_rollback() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _packaged_coordinator_plan(_cutover_plan(private, services))
    trace: list[str] = []

    class TracedDatabase(Database):
        def rollback(self, plan, receipt):
            trace.append("database")
            return super().rollback(plan, receipt)

    class TracedHost(Host):
        def rollback(self, plan, receipt):
            trace.append("host")
            return super().rollback(plan, receipt)

    database = TracedDatabase(plan)
    database.apply_override = {"archive_extended19_sha256": "f" * 64}
    cron = CronBoundary(services, trace)
    journal = MemoryJournal()
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan,
            _approval(private, plan),
            cutover.CutoverDependencies(
                services,
                Snapshots(plan.final_snapshot),
                database,
                TracedHost(plan, services),
                journal,
                Prerequisites(),
                nullcontext,
                cron=cron,
            ),
            now_unix=NOW,
        )
    assert trace == ["cron", "host", "database"]
    events = [entry.value["event"] for entry in journal.load(plan.sha256)]
    assert events.index("cron_rolled_back") < events.index("host_rolled_back")
    assert events.index("host_rolled_back") < events.index("database_rolled_back")
    assert cron.calls == ["preflight", "apply", "postflight", "rollback"]


def test_forged_cron_activation_receipt_is_not_appended_after_commit() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _packaged_coordinator_plan(_cutover_plan(private, services))

    class ForgedActivationCron(CronBoundary):
        def activate(self, plan, postflight_receipt, activation_authority):
            receipt = dict(
                super().activate(
                    plan,
                    postflight_receipt,
                    activation_authority,
                )
            )
            receipt["activation_authority_sha256"] = "0" * 64
            unsigned = {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
            return _hashed(unsigned, "receipt_sha256")

    journal = MemoryJournal()
    with pytest.raises(
        cutover.ProductionCutoverError,
        match="forward_recovery_required",
    ):
        cutover.execute_cutover(
            plan,
            _approval(private, plan),
            cutover.CutoverDependencies(
                services,
                Snapshots(plan.final_snapshot),
                Database(plan),
                Host(plan, services),
                journal,
                Prerequisites(),
                nullcontext,
                cron=ForgedActivationCron(services),
            ),
            now_unix=NOW,
        )
    assert "cron_activated" not in {
        entry.value["event"] for entry in journal.load(plan.sha256)
    }


def test_production_cron_boundary_invokes_only_digest_attested_entrypoint() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _packaged_coordinator_plan(_cutover_plan(private, services))
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        receipt = _cron_receipt("preflight", plan)
        return SimpleNamespace(
            returncode=0,
            stdout=cutover._canonical_bytes(receipt) + b"\n",
            stderr=b"",
        )

    receipt = cutover.ProductionCronCutoverBoundary(
        runner=runner
    ).preflight(plan)
    assert receipt["schema"] == production_cron_cutover_runtime.PREFLIGHT_SCHEMA
    assert len(calls) == 1
    argv, kwargs = calls[0]
    release = (
        cutover.PRODUCTION_RELEASE_BASE / f"hermes-agent-{REVISION[:12]}"
    )
    assert argv[:5] == (
        str(release / ".venv/bin/python"),
        "-B",
        "-I",
        str(
            release
            / production_cron_continuity_package.CUTOVER_ENTRYPOINT_RELATIVE_PATH
        ),
        "preflight",
    )
    assert argv[argv.index("--expected-entrypoint-sha256") + 1] == "e" * 64
    assert argv[argv.index("--expected-runtime-sha256") + 1] == "d" * 64
    assert kwargs["cwd"] == "/"
    assert kwargs["env"] == {
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
    }


def test_final_tail_drift_blocks_before_any_database_mutation() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    snapshots = Snapshots(plan.final_snapshot)
    snapshots.before = _snapshot(plan.final_snapshot.value["source_row_count"] + 1)
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan, _approval(private, plan),
            cutover.CutoverDependencies(
                services, snapshots, database, Host(plan, services), MemoryJournal(),
                Prerequisites(), nullcontext
            ),
            now_unix=NOW,
        )
    assert database.calls == []
    assert services.gateway.value["active_state"] == "active"
    assert services.gateway.value["fragment_sha256"] == services.legacy_gateway_digest


def test_final_tail_reobservation_ignores_only_collection_timestamp() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    snapshots = Snapshots(plan.final_snapshot)
    snapshots.before = cutover.LegacySnapshot.from_mapping(
        _hashed(
            {
                **{
                    key: copy.deepcopy(value)
                    for key, value in plan.final_snapshot.value.items()
                    if key != "snapshot_sha256"
                },
                "observed_at_unix": NOW + 30,
            },
            "snapshot_sha256",
        )
    )
    terminal = cutover.execute_cutover(
        plan,
        _approval(private, plan),
        cutover.CutoverDependencies(
            services,
            snapshots,
            database,
            Host(plan, services),
            MemoryJournal(),
            Prerequisites(),
            nullcontext,
        ),
        now_unix=NOW,
    )
    assert terminal["schema"] == cutover.TERMINAL_SCHEMA


def test_resume_after_host_applied_accepts_exact_target_unit_identities() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    host = Host(plan, services)
    journal = MemoryJournal()
    first_approval = _approval(private, plan)
    validated = cutover.CutoverApproval.from_mapping(
        first_approval, plan=plan, now_unix=NOW
    )
    journal.append(
        plan.sha256,
        "authority",
        {"approval_sha256": validated.sha256, "sequence": 0},
        NOW,
    )
    journal.append(
        plan.sha256,
        "database_apply_started",
        {"artifact_sha256": plan.value["artifacts"]["database_apply"]["sha256"]},
        NOW,
    )
    journal.append(
        plan.sha256,
        "database_applied",
        _db_receipt(
            plan,
            "muncho-production-legacy-database-apply.v1",
            "database_apply",
        ),
        NOW,
    )
    journal.append(
        plan.sha256,
        "host_apply_started",
        {"artifact_sha256": plan.value["artifacts"]["host_activation"]["sha256"]},
        NOW,
    )
    host_receipt = host.apply_stopped(plan)
    journal.append(plan.sha256, "host_applied", host_receipt, NOW)
    host.calls.clear()

    terminal = cutover.execute_cutover(
        plan,
        _approval(
            private,
            plan,
            sequence=1,
            previous=validated.sha256,
        ),
        cutover.CutoverDependencies(
            services,
            Snapshots(plan.final_snapshot),
            database,
            host,
            journal,
            Prerequisites(),
            nullcontext,
        ),
        now_unix=NOW,
    )
    assert terminal["schema"] == cutover.TERMINAL_SCHEMA
    assert host.calls == ["start_prerequisites", "start_writer", "commit_boot"]


def test_host_apply_receipt_rejects_target_unit_identity_drift() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    host = Host(plan, services)
    receipt = host.apply_stopped(plan)
    drifted = copy.deepcopy(receipt)
    drifted["gateway_stopped"] = _service(
        cutover.GATEWAY_UNIT,
        active=False,
        digest=services.target_gateway_digest,
        unit_file_state="enabled",
        drop_in_digest=services.target_drop_in_digest,
    ).to_mapping()
    unsigned = {
        key: value for key, value in drifted.items() if key != "receipt_sha256"
    }
    drifted["receipt_sha256"] = cutover._sha256_json(unsigned)
    with pytest.raises(cutover.ProductionCutoverError, match="host-apply"):
        cutover._require_host_apply_receipt(drifted, plan=plan)


def test_host_receipts_require_verifier_only_api_and_operational_key_proofs() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    apply_receipt = Host(plan, services).apply_stopped(plan)
    assert cutover._require_host_apply_receipt(
        apply_receipt, plan=plan
    )["api_source_secrets_loaded_by_gateway"] is False
    rollback_receipt = _host_rollback_receipt(plan)
    assert cutover._require_host_rollback_receipt(
        rollback_receipt, plan=plan
    )["api_source_secrets_preserved"] is True

    for receipt, field, validator in (
        (
            apply_receipt,
            "api_approval_verifier_installed",
            cutover._require_host_apply_receipt,
        ),
        (
            rollback_receipt,
            "api_verifiers_removed",
            cutover._require_host_rollback_receipt,
        ),
        (
            apply_receipt,
            "operational_edge_keys_ready",
            cutover._require_host_apply_receipt,
        ),
        (
            rollback_receipt,
            "operational_edge_keys_retained_dormant",
            cutover._require_host_rollback_receipt,
        ),
    ):
        drifted = copy.deepcopy(receipt)
        drifted[field] = False
        drifted["receipt_sha256"] = cutover._sha256_json(
            {
                key: value
                for key, value in drifted.items()
                if key != "receipt_sha256"
            }
        )
        with pytest.raises(cutover.ProductionCutoverError):
            validator(drifted, plan=plan)


def test_resume_target_drift_rolls_back_fail_closed_instead_of_leaving_database_applied() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    host = Host(plan, services)
    journal = MemoryJournal()
    first = _approval(private, plan)
    validated = cutover.CutoverApproval.from_mapping(
        first, plan=plan, now_unix=NOW
    )
    journal.append(
        plan.sha256,
        "authority",
        {"approval_sha256": validated.sha256, "sequence": 0},
        NOW,
    )
    journal.append(
        plan.sha256,
        "database_apply_started",
        {"artifact_sha256": plan.value["artifacts"]["database_apply"]["sha256"]},
        NOW,
    )
    journal.append(
        plan.sha256,
        "database_applied",
        _db_receipt(
            plan,
            "muncho-production-legacy-database-apply.v1",
            "database_apply",
        ),
        NOW,
    )
    journal.append(
        plan.sha256,
        "host_apply_started",
        {"artifact_sha256": plan.value["artifacts"]["host_activation"]["sha256"]},
        NOW,
    )
    journal.append(plan.sha256, "host_applied", host.apply_stopped(plan), NOW)
    services.gateway = _service(
        cutover.GATEWAY_UNIT,
        active=False,
        digest=services.target_gateway_digest,
        unit_file_state="enabled",
        drop_in_digest=services.target_drop_in_digest,
    )

    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan,
            _approval(
                private,
                plan,
                sequence=1,
                previous=validated.sha256,
            ),
            cutover.CutoverDependencies(
                services,
                Snapshots(plan.final_snapshot),
                database,
                host,
                journal,
                Prerequisites(),
                nullcontext,
            ),
            now_unix=NOW,
        )
    assert database.calls == ["rollback"]
    assert host.calls[-1] == "rollback"
    assert services.gateway.value["fragment_sha256"] == services.legacy_gateway_digest
    assert services.gateway.value["active_state"] == "active"


def test_host_failure_rolls_back_host_and_database_fail_closed() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    host = Host(plan, services, fail_start=True)
    journal = MemoryJournal()
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan, _approval(private, plan),
            cutover.CutoverDependencies(
                services, Snapshots(plan.final_snapshot), database, host, journal,
                Prerequisites(), nullcontext
            ),
            now_unix=NOW,
        )
    assert "rollback" in host.calls
    assert database.calls[-1] == "rollback"
    assert services.gateway.value["active_state"] == "active"
    assert services.gateway.value["fragment_sha256"] == services.legacy_gateway_digest
    assert journal.load(plan.sha256)[-1].value["event"] == "rollback_terminal"


def test_wrong_archive_digest_rolls_back_started_prerequisites_and_database() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    database.apply_override = {"archive_extended19_sha256": "f" * 64}
    host = Host(plan, services)
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan, _approval(private, plan),
            cutover.CutoverDependencies(
                services, Snapshots(plan.final_snapshot), database, host, MemoryJournal(),
                Prerequisites(), nullcontext
            ),
            now_unix=NOW,
        )
    assert host.calls == ["apply", "start_prerequisites", "rollback"]
    # A mutation may have committed before an invalid/lost response reached
    # the coordinator.  Reconciliation is mandatory without trusting it.
    assert database.calls[-1] == "rollback"


def test_database_apply_exception_reconciles_without_accepted_receipt() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = ApplyResponseLostDatabase(plan)
    journal = MemoryJournal()
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan,
            _approval(private, plan),
            cutover.CutoverDependencies(
                services,
                Snapshots(plan.final_snapshot),
                database,
                Host(plan, services),
                journal,
                Prerequisites(),
                nullcontext,
            ),
            now_unix=NOW,
        )
    assert database.calls == ["preflight", "apply", "rollback"]
    assert database.rollback_authority is None
    assert journal.load(plan.sha256)[-1].value["event"] == "rollback_terminal"


def test_unreceipted_database_apply_intent_rolls_back_instead_of_replaying() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = Database(plan)
    journal = MemoryJournal()
    first = _approval(private, plan)
    validated = cutover.CutoverApproval.from_mapping(
        first, plan=plan, now_unix=NOW
    )
    journal.append(
        plan.sha256,
        "authority",
        {"approval_sha256": validated.sha256, "sequence": 0},
        NOW,
    )
    journal.append(
        plan.sha256,
        "database_apply_started",
        {"artifact_sha256": plan.value["artifacts"]["database_apply"]["sha256"]},
        NOW,
    )
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan,
            _approval(
                private,
                plan,
                sequence=1,
                previous=validated.sha256,
            ),
            cutover.CutoverDependencies(
                services,
                Snapshots(plan.final_snapshot),
                database,
                Host(plan, services),
                journal,
                Prerequisites(),
                nullcontext,
            ),
            now_unix=NOW,
        )
    assert database.calls == ["rollback"]


def test_rollback_terminal_prevents_stale_forward_replay() -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    database = ApplyResponseLostDatabase(plan)
    journal = MemoryJournal()
    first = _approval(private, plan)
    validated = cutover.CutoverApproval.from_mapping(
        first, plan=plan, now_unix=NOW
    )
    dependencies = cutover.CutoverDependencies(
        services,
        Snapshots(plan.final_snapshot),
        database,
        Host(plan, services),
        journal,
        Prerequisites(),
        nullcontext,
    )
    with pytest.raises(cutover.ProductionCutoverError, match="rolled_back"):
        cutover.execute_cutover(
            plan, first, dependencies, now_unix=NOW
        )
    calls = list(database.calls)
    entry_count = len(journal.load(plan.sha256))
    with pytest.raises(
        cutover.ProductionCutoverError,
        match="already_rolled_back",
    ):
        cutover.execute_cutover(
            plan,
            _approval(
                private,
                plan,
                sequence=1,
                previous=validated.sha256,
            ),
            dependencies,
            now_unix=NOW,
        )
    assert database.calls == calls
    assert len(journal.load(plan.sha256)) == entry_count


def test_packaged_process_boundary_routes_missing_receipt_rollbacks_exactly(
    monkeypatch,
) -> None:
    private = Ed25519PrivateKey.generate()
    services = Services()
    plan = _cutover_plan(private, services)
    calls: list[tuple[tuple[str, ...], dict, dict]] = []

    def runner(argv, **kwargs):
        request = json.loads(kwargs["input"].decode("utf-8"))
        calls.append((argv, kwargs, request))
        response = {"edge": request["action"], "ok": True}
        return SimpleNamespace(
            returncode=0,
            stdout=cutover._canonical_bytes(response) + b"\n",
            stderr=b"",
        )

    process = cutover.ProductionArtifactProcessBoundary(runner=runner)
    monkeypatch.setattr(
        process,
        "_materialize",
        lambda **_kwargs: Path("/root/materialized-cutover-edge"),
    )
    database = cutover.ProductionDatabaseArtifactBoundary(process)
    host = cutover.ProductionHostArtifactBoundary(process)
    assert database.rollback(plan, None)["edge"] == "database_rollback"
    assert host.rollback(plan, None)["edge"] == "host_rollback"
    assert [item[0][1] for item in calls] == [
        "database_rollback",
        "host_rollback",
    ]
    for _argv, kwargs, request in calls:
        assert request["apply_receipt"] is None
        assert request["secret_material_recorded"] is False
        assert kwargs["env"] == {
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
        }
        unsigned = {
            key: value
            for key, value in request.items()
            if key != "request_sha256"
        }
        assert request["request_sha256"] == cutover._sha256_json(unsigned)


def test_artifact_binding_is_fixed_to_exact_production_release_root() -> None:
    with pytest.raises(ValueError, match="release-addressed"):
        cutover._artifact(
            {
                "path": f"/tmp/hermes-agent-{REVISION[:12]}/edge",
                "sha256": "a" * 64,
            },
            "test edge",
            REVISION,
        )


def test_systemd_mutation_uses_existing_fixed_command_boundary() -> None:
    commands = []

    def runner(command):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    boundary = cutover.ProductionSystemdServiceBoundary(runner=runner)
    boundary.stop_gateway()
    assert len(commands) == 1
    assert isinstance(commands[0], cutover.activation.Command)
    assert commands[0].argv == (
        cutover.SYSTEMCTL,
        "stop",
        "--",
        cutover.GATEWAY_UNIT,
    )
