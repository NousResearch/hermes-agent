from __future__ import annotations

import copy
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.operational_edge_catalog import (
    operation_catalog,
    required_cron_operations,
)
from gateway.operational_edge_readiness import (
    PROBE_PACKET_SCHEMA,
    build_operational_edge_readiness,
)
from ops.muncho.runtime import trusted_cron_collector_rail as rail


EDGE_BOOT_ID_SHA256 = "f" * 64
EDGE_OBSERVED_AT_UNIX = 1_800_000_000
EDGE_COLLECTOR_NONCE = "12345678-1234-4123-8123-123456789abc"


def _dependency_facts() -> dict[str, str]:
    return {
        path: f"{index + 1:064x}"
        for index, path in enumerate(
            sorted(
                {
                    path
                    for spec in rail.COLLECTOR_SPECS
                    for path in spec.dependency_paths
                }
            )
        )
    }


def _package() -> dict:
    return rail.build_package_manifest(
        revision="a" * 40,
        rail_sha256="b" * 64,
        dependency_facts=_dependency_facts(),
    )


def _operational_edge_receipt() -> dict:
    required = dict(required_cron_operations())
    catalog = operation_catalog()
    jobs = []
    for index, (job_id, operation_id) in enumerate(required.items(), start=1):
        operation = catalog[operation_id]
        jobs.append(
            {
                "source_job_id": job_id,
                "operation_id": operation_id,
                "domain": operation.domain,
                "service_unit": (
                    f"muncho-operational-edge-{operation.domain}.service"
                ),
                "service_uid": 3000 + index,
                "service_gid": 4000 + index,
                "socket_path": (
                    f"/run/muncho-operational-edge/{operation.domain}/edge.sock"
                ),
                "socket_uid": 3000 + index,
                "socket_gid": 4000 + index,
                "socket_mode": "0660",
                "main_pid": 5000 + index,
                "peer_round_trip": True,
                "probe_operation_id": (
                    operation.probe_operation_id or operation.operation_id
                ),
                "probe_return_code": 0,
                "probe_packet_schema": PROBE_PACKET_SCHEMA,
                "probe_packet_sha256": f"{index:064x}",
                "meaningful_packet": True,
                "error_only_packet": False,
            }
        )
    return build_operational_edge_readiness(
        revision="a" * 40,
        required_jobs=required,
        jobs=jobs,
        boot_id_sha256=EDGE_BOOT_ID_SHA256,
        observed_at_unix=EDGE_OBSERVED_AT_UNIX,
        collector_nonce=EDGE_COLLECTOR_NONCE,
    )


def test_package_is_exact_credential_isolated_and_never_activates() -> None:
    package = _package()
    units = rail.render_package_unit_files(package)
    contract = rail.catalog_public_contract()

    assert rail.validate_package_manifest(package) == package
    assert len(contract) == 21
    assert len(units) == 42
    assert package["service_user"] == "muncho-projector"
    assert package["service_group"] == "hermes-cloud-gateway"
    assert package["reader_group"] == "hermes-cloud-gateway"
    assert package["provider_or_model_dependency"] is False
    assert package["gateway_credential_dependency"] is False
    assert package["discord_credential_dependency"] is False
    assert package["timer_enabled_by_package"] is False
    assert package["timer_started_by_package"] is False
    assert all(row["semantic_judgment_allowed"] is False for row in contract)
    assert all(row["provider_or_model_allowed"] is False for row in contract)
    assert all(row["direct_discord_allowed"] is False for row in contract)
    assert all(
        row["historical_source_delivery_eligible"] is False
        for row in contract
    )
    by_job = {row["source_job_id"]: row for row in contract}
    assert by_job["06ef64d72891"]["historical_source_delivery"] == (
        "discord:1504852355588423801"
    )
    assert by_job["e62f55ca93ca"]["historical_source_delivery"] == (
        "discord:1504852355588423801:1524321461714681976"
    )
    assert any(
        row["execution_boundary"] == rail.EXECUTION_BOUNDARY_SCOPED
        for row in contract
    )

    services = b"\n".join(
        value for name, value in units.items() if name.endswith(".service")
    )
    timers = b"\n".join(
        value for name, value in units.items() if name.endswith(".timer")
    )
    for forbidden in (
        b"EnvironmentFile=",
        b"LoadCredential=",
        b"OPENAI_API_KEY",
        b"DISCORD_BOT_TOKEN",
        b"PassEnvironment=",
    ):
        assert forbidden not in services
    assert b"User=muncho-projector" in services
    assert b"Group=hermes-cloud-gateway" in services
    assert b"SupplementaryGroups=" in services
    assert b"Persistent=true" not in timers
    assert b"Persistent=false" in timers


def test_readiness_checks_real_service_identity_paths_and_scoped_edge() -> None:
    package = _package()

    def metadata(path: Path):
        text = str(path)
        mode = 0o755
        uid = 0
        gid = 0
        if text == "/opt/adventico-ai-platform/canonical-brain":
            mode = 0o750
            uid = 1000
            gid = 1000
        return SimpleNamespace(
            st_mode=stat.S_IFDIR | mode,
            st_uid=uid,
            st_gid=gid,
        )

    readiness = rail.collect_execution_readiness(
        package,
        account_lookup=lambda _name: SimpleNamespace(pw_uid=2003),
        group_lookup=lambda _name: SimpleNamespace(gr_gid=2004),
        stat_reader=metadata,
    )

    assert rail.validate_execution_readiness(
        readiness,
        manifest=package,
    ) == readiness
    assert readiness["direct_dependencies_ready"] is False
    assert readiness["blocked_path_count"] > 0
    assert readiness["scoped_execution_edge_required_count"] > 0
    assert readiness["scoped_execution_edge_packaged"] is False
    assert readiness["activation_ready"] is False
    assert readiness["permissions_widened"] is False
    assert readiness["credential_content_read"] is False


def test_readiness_receipt_rejects_a_claimed_ready_edge() -> None:
    package = _package()
    readiness = rail.collect_execution_readiness(
        package,
        account_lookup=lambda _name: SimpleNamespace(pw_uid=2003),
        group_lookup=lambda _name: SimpleNamespace(gr_gid=2004),
        stat_reader=lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        ),
    )
    readiness["scoped_execution_edge_packaged"] = True
    readiness["activation_ready"] = True
    readiness["readiness_sha256"] = rail._sha256(
        rail._canonical({
            name: item
            for name, item in readiness.items()
            if name != "readiness_sha256"
        })
    )

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="execution_readiness_invalid",
    ):
        rail.validate_execution_readiness(readiness, manifest=package)


def test_readiness_requires_exact_meaningful_operational_edge_receipt() -> None:
    package = _package()
    edge = _operational_edge_receipt()
    readiness = rail.collect_execution_readiness(
        package,
        operational_edge_receipt=edge,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
        account_lookup=lambda _name: SimpleNamespace(pw_uid=2003),
        group_lookup=lambda _name: SimpleNamespace(gr_gid=2004),
        stat_reader=lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=0,
        ),
    )

    assert rail.validate_execution_readiness(
        readiness,
        manifest=package,
        operational_edge_receipt=edge,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
    ) == readiness
    assert readiness["direct_dependencies_ready"] is True
    assert readiness["scoped_execution_edge_packaged"] is True
    assert readiness["scoped_execution_edge_receipt_sha256"] == edge[
        "receipt_sha256"
    ]
    assert readiness["scoped_execution_edge_meaningful_packet_count"] == len(
        required_cron_operations()
    )
    assert readiness["activation_ready"] is True

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="execution_readiness_invalid",
    ):
        rail.validate_execution_readiness(readiness, manifest=package)


def test_package_rejects_dependency_or_unit_drift() -> None:
    package = _package()
    drifted = copy.deepcopy(package)
    dependency = next(iter(drifted["dependency_sha256"]))
    drifted["dependency_sha256"][dependency] = "f" * 64

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="package_manifest_invalid",
    ):
        rail.validate_package_manifest(drifted)

    drifted = copy.deepcopy(package)
    first = rail.COLLECTOR_SPECS[0].source_job_id
    drifted["units"][first]["service_sha256"] = "f" * 64
    unsigned = {
        key: value
        for key, value in drifted.items()
        if key != "manifest_sha256"
    }
    drifted["manifest_sha256"] = rail._sha256(rail._canonical(unsigned))
    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="package_manifest_invalid",
    ):
        rail.validate_package_manifest(drifted)


def _voice_stage(tmp_path: Path) -> tuple[dict, Path]:
    voice_root = tmp_path / "voice"
    voice_root.mkdir()
    source = voice_root / "discord_g12345678901234567_vc12345678901234568_1.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "discord_voice_context.transcript",
                "timestamp": "2026-07-15T00:00:00Z",
                "guild_id": "12345678901234567",
                "voice_channel_id": "12345678901234568",
                "voice_channel_name": "Public voice",
                "text_channel_id": "12345678901234569",
                "user_id": "12345678901234570",
                "transcript": "Follow up on the deployment.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cursor = tmp_path / "state/cursor.json"
    stage = rail.stage_voice_packet(
        voice_root=voice_root,
        cursor_path=cursor,
        created_at="2026-07-15T00:01:00Z",
    )
    assert stage is not None
    return stage, cursor


def test_voice_cursor_moves_only_after_sent_readback_and_canonical_sent(
    tmp_path: Path,
) -> None:
    stage, cursor = _voice_stage(tmp_path)
    assert not cursor.exists()

    blocked = rail.build_voice_delivery_proof(
        stage=stage,
        connector_send_receipt_sha256="1" * 64,
        connector_readback_receipt_sha256=None,
        canonical_event_type="route_back.blocked",
        canonical_event_sha256="2" * 64,
    )
    result = rail.commit_voice_cursor(
        stage=stage,
        proof=blocked,
        cursor_path=cursor,
    )
    assert result["cursor_committed"] is False
    assert result["canonical_terminal_event"] == "route_back.blocked"
    assert not cursor.exists()

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="voice_delivery_proof_invalid",
    ):
        rail.build_voice_delivery_proof(
            stage=stage,
            connector_send_receipt_sha256="1" * 64,
            connector_readback_receipt_sha256=None,
            canonical_event_type="route_back.sent",
            canonical_event_sha256="2" * 64,
        )

    sent = rail.build_voice_delivery_proof(
        stage=stage,
        connector_send_receipt_sha256="1" * 64,
        connector_readback_receipt_sha256="3" * 64,
        canonical_event_type="route_back.sent",
        canonical_event_sha256="2" * 64,
    )
    result = rail.commit_voice_cursor(
        stage=stage,
        proof=sent,
        cursor_path=cursor,
    )
    persisted = json.loads(cursor.read_text(encoding="utf-8"))
    assert result["cursor_committed"] is True
    assert result["canonical_terminal_event"] == "route_back.sent"
    assert persisted["files"] == stage["proposed_offsets"]
    assert persisted["delivery_proof_sha256"] == sent["proof_sha256"]


def test_voice_cursor_drift_fails_closed(tmp_path: Path) -> None:
    stage, cursor = _voice_stage(tmp_path)
    cursor.parent.mkdir(parents=True)
    cursor.write_text(
        json.dumps(
            {
                "schema": rail.VOICE_CURSOR_SCHEMA,
                "files": {"unrelated.jsonl": 10},
            }
        ),
        encoding="utf-8",
    )
    proof = rail.build_voice_delivery_proof(
        stage=stage,
        connector_send_receipt_sha256="1" * 64,
        connector_readback_receipt_sha256="3" * 64,
        canonical_event_type="route_back.sent",
        canonical_event_sha256="2" * 64,
    )

    with pytest.raises(
        rail.TrustedCronCollectorError,
        match="voice_cursor_drifted",
    ):
        rail.commit_voice_cursor(
            stage=stage,
            proof=proof,
            cursor_path=cursor,
        )
