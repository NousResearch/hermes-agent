from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization

from gateway import canonical_capability_canary_e2e as contract
from gateway import canonical_capability_canary_live_driver as live
from gateway import canonical_capability_canary_producers as producers
from gateway.canonical_full_canary_live_driver import SSEConversation
from tests.gateway.test_canonical_capability_canary_e2e import (
    NOW,
    _common,
    _evidence,
    _fixture,
    _private_keys,
    _resign,
    _resign_exact_envelope,
    _signed,
    _sshsig_signature,
)


@pytest.fixture(autouse=True)
def _exact_bound_plan_publication(monkeypatch):
    monkeypatch.setattr(
        live,
        "load_bound_plan_publication_receipt",
        lambda plan: {
            "receipt_sha256": plan.plan_publication_receipt_sha256,
        },
    )


def _chmod_directory(path: Path, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(mode)


def _write_receipt(path: Path, value: dict) -> None:
    path.write_bytes(live._canonical_bytes(value))
    path.chmod(0o400)


def _retire_activated_fleet(
    activated: live.ActivatedProducerFleet,
) -> dict[str, object]:
    return {
        "schema": "muncho-production-capability-fleet-retirement.v1",
        "run_id": activated.readiness["run_id"],
        "readiness_sha256": activated.readiness["readiness_sha256"],
        "path": str(producers.DEFAULT_READINESS_PATH),
        "retired": True,
        "absence_verified": True,
    }


def test_collector_bundle_normalizes_target_and_retires_api_only_after_terminal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    class GoalCollector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def publish_api_observer_retirement(self):
            captured["retired"] = True
            return {"marker_sha256": "9" * 64}

        def controlled_gateway_restart(
            self, *, deadline, pre_restart_validator, runner
        ):
            captured["restart_deadline"] = deadline
            captured["restart_validator"] = pre_restart_validator
            captured["restart_runner"] = runner
            return {"restart": "observed"}

    class Primary:
        plan = SimpleNamespace(
            artifacts={
                "plugin_config": SimpleNamespace(sha256="8" * 64),
            }
        )
        frames = (
            SimpleNamespace(value={"event": "session_end"}),
        )

        def wait_session_end(self, *, timeout):
            captured["wait_timeout"] = timeout

    monkeypatch.setattr(live, "SegmentedGoalEvidenceCollector", GoalCollector)
    plan = SimpleNamespace(
        identities=SimpleNamespace(gateway_uid=2101, gateway_gid=2102)
    )
    bundle = live.CapabilityEvidenceCollectorBundle(Primary(), plan)
    fixture = live.PublishedFixture(
        value={
            "release_sha": "a" * 40,
            "release_artifact_sha256": "b" * 64,
            "run_id": "11111111-1111-4111-8111-111111111111",
            "valid_from_unix_ms": 1,
            "valid_until_unix_ms": 2,
            "public_discord_target": {
                "target_type": "public_channel",
                "guild_id": "123456789012345678",
                "channel_id": "223456789012345678",
            },
            "owner_id": live.PRODUCTION_OWNER_DISCORD_USER_ID,
        },
        sha256="c" * 64,
        path=tmp_path / "fixture.json",
        run_directory=tmp_path,
    )

    bundle.configure_goal(SimpleNamespace(), fixture)
    result = bundle.retire_historical_api_observer(
        deadline=time.monotonic() + 10,
    )

    assert captured["public_target"] == {
        "target_type": "public_guild_channel",
        "guild_id": "123456789012345678",
        "channel_id": "223456789012345678",
    }
    assert captured["api_observer_config_sha256"] == "8" * 64
    assert captured["retired"] is True
    assert result["marker_sha256"] == "9" * 64

    validator = lambda _frames: True
    runner = object()
    marker, restart = bundle.controlled_gateway_restart(
        deadline=time.monotonic() + 10,
        pre_restart_validator=validator,
        runner=runner,
    )
    assert marker == {"marker_sha256": "9" * 64}
    assert restart == {"restart": "observed"}
    assert captured["restart_validator"] is validator
    assert captured["restart_runner"] is runner


def _fixture_and_plans():
    keys = _private_keys()
    fixture = _fixture(keys)
    fixture["discord_bot_identities"] = {
        "production_bot_user_id": live.PRODUCTION_DISCORD_BOT_USER_ID,
        "connector_bot_user_id": (
            contract.PRODUCTION_DISCORD_CONNECTOR_BOT_USER_ID
        ),
        "routeback_bot_user_id": (
            contract.PRODUCTION_DISCORD_ROUTEBACK_BOT_USER_ID
        ),
    }
    uid = os.getuid()
    gid = os.getgid()
    plan = SimpleNamespace(
        revision=fixture["release_sha"],
        release_root=Path(fixture["release_root"]),
        release_artifact_sha256=fixture["release_artifact_sha256"],
        gateway_config_sha256=fixture["effective_config_sha256"],
        runtime_dependency_manifest_sha256=fixture[
            "installed_wheel_manifest_sha256"
        ],
        connector_allowed_guild_ids=(
            fixture["public_discord_target"]["guild_id"],
        ),
        connector_allowed_channel_ids=(
            fixture["public_discord_target"]["channel_id"],
        ),
        connector_allowed_user_ids=(live.PRODUCTION_OWNER_DISCORD_USER_ID,),
        connector_bot_user_id=fixture["discord_bot_identities"][
            "connector_bot_user_id"
        ],
        routeback_bot_user_id=fixture["discord_bot_identities"][
            "routeback_bot_user_id"
        ],
        mac_ops_service_identity_sha256=fixture[
            "business_edge_service_identity_sha256"
        ],
        bitrix_operational_edge_service_identity_sha256=fixture[
            "business_edge_service_identity_sha256"
        ],
        bitrix_operational_edge_revision=fixture["release_sha"],
        bitrix_operational_edge_service_unit=fixture[
            "bitrix_operational_edge_contract"
        ]["service_unit"],
        bitrix_operational_edge_asset_manifest_sha256=fixture[
            "bitrix_operational_edge_contract"
        ]["asset_manifest_sha256"],
        bitrix_operational_edge_asset_names=tuple(
            fixture["bitrix_operational_edge_contract"]["asset_names"]
        ),
        bitrix_operational_edge_asset_manifest_path=Path(
            fixture["bitrix_operational_edge_contract"][
                "asset_manifest_path"
            ]
        ),
        bitrix_operational_edge_rendered_unit_sha256=fixture[
            "bitrix_operational_edge_contract"
        ]["rendered_unit_sha256"],
        bitrix_operational_edge_rendered_unit_path=Path(
            fixture["bitrix_operational_edge_contract"][
                "rendered_unit_path"
            ]
        ),
        bitrix_operational_edge_rendered_config_sha256=fixture[
            "bitrix_operational_edge_contract"
        ]["rendered_config_sha256"],
        bitrix_operational_edge_rendered_config_path=Path(
            fixture["bitrix_operational_edge_contract"][
                "rendered_config_path"
            ]
        ),
        bitrix_operational_edge_rendered_trust_sha256=fixture[
            "bitrix_operational_edge_contract"
        ]["rendered_trust_sha256"],
        bitrix_operational_edge_rendered_trust_path=Path(
            fixture["bitrix_operational_edge_contract"][
                "rendered_trust_path"
            ]
        ),
        bitrix_operational_edge_service_user=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["service_user"],
        bitrix_operational_edge_service_group=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["service_group"],
        bitrix_operational_edge_service_uid=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["service_uid"],
        bitrix_operational_edge_service_gid=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["service_gid"],
        bitrix_operational_edge_socket_client_group=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["socket_client_group"],
        bitrix_operational_edge_socket_client_gid=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["socket_client_gid"],
        bitrix_operational_edge_identity_bootstrap_receipt_sha256=fixture[
            "bitrix_operational_edge_contract"
        ]["identity_bootstrap"]["receipt_sha256"],
        bitrix_operational_edge_receipt_public_key_id=fixture[
            "bitrix_operational_edge_contract"
        ]["receipt_key_contract"]["public_key_id"],
        bitrix_operational_edge_key_bootstrap_receipt_sha256=fixture[
            "bitrix_operational_edge_contract"
        ]["receipt_key_contract"]["key_bootstrap_receipt_sha256"],
        bitrix_operational_edge_credential_binding=fixture[
            "bitrix_operational_edge_contract"
        ]["credential_binding"],
        full_canary_terminal_receipt=fixture[
            "full_canary_terminal_receipt"
        ],
        full_canary_terminal_receipt_sha256=fixture[
            "full_canary_terminal_receipt_sha256"
        ],
        original_full_canary_owner_approval_sha256=fixture[
            "original_full_canary_owner_approval_sha256"
        ],
        plan_publication_receipt_sha256=fixture[
            "plan_publication_receipt_sha256"
        ],
        identities=SimpleNamespace(
            mac_ops_uid=uid,
            mac_ops_gid=gid,
            worker_uid=uid + 61_001,
            worker_gid=gid + 61_002,
            worker_client_gid=gid + 61_003,
            socket_client_gid=gid + 61_004,
            browser_uid=uid + 61_005,
        ),
        sha256="a" * 64,
    )
    plan.gateway_config_sha256 = hashlib.sha256(
        live.render_gateway_config(plan)
    ).hexdigest()
    fixture["effective_config_sha256"] = plan.gateway_config_sha256
    fixture_sha256 = contract._digest(fixture)
    evidence = _evidence(fixture, fixture_sha256, keys)
    full_plan = SimpleNamespace(
        revision=fixture["release_sha"],
        release={
            "artifact_sha256": fixture["release_artifact_sha256"],
            "artifact_root": fixture["release_root"],
        },
        identities=SimpleNamespace(
            writer_uid=uid,
            writer_gid=gid,
            edge_uid=uid,
            edge_gid=gid,
        ),
        sha256="b" * 64,
    )
    return keys, fixture, fixture_sha256, evidence, plan, full_plan


def _foundation_and_authority(
    *,
    tmp_path: Path,
    keys: dict,
    fixture: dict,
    plan,
    full_plan,
) -> tuple[dict, dict, str, str, Path]:
    uid = tmp_path.lstat().st_uid
    gid = tmp_path.lstat().st_gid
    receipt_root = (tmp_path / "producer-receipts").resolve()
    source = {
        "kind": "skyvision_mac_ops_ed25519",
        "path": "/Users/emillomliev/.ssh/skyvision_mac_ops_ed25519.pub",
        "comment": "skyvision-mac-ops",
        "fingerprint": "SHA256:test-owner-foundation",
        "file_sha256": hashlib.sha256(b"owner-public-source").hexdigest(),
        "uid": uid,
        "gid": gid,
        "mode": 0o600,
        "size": 96,
    }
    source_sha256 = producers._sha256_json(source)

    def public_hex(role: str) -> str:
        return keys[role].public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()

    authorities = {
        role: {
            "key_id": hashlib.sha256(bytes.fromhex(public_hex(role))).hexdigest(),
            "algorithm": producers.AUTHORITY_ALGORITHMS[role],
            "public_key_ed25519_hex": public_hex(role),
        }
        for role in producers.AUTHORITY_ROLES
    }
    endpoints = {}
    for index, role in enumerate(producers.ENDPOINT_ROLES, start=1):
        endpoints[role] = {
            "service_unit": f"muncho-capability-{role.replace('_', '-')}-producer.service",
            "service_identity_sha256": hashlib.sha256(
                f"service:{role}".encode()
            ).hexdigest(),
            # Endpoint identities are deliberately distinct from one another
            # and from the root-owned shared receipt directory.  The tests use
            # in-process endpoint doubles, so these need not exist in passwd.
            "uid": uid + 10_000 + index,
            "gid": gid + 20_000 + index,
            "socket_path": str((tmp_path / f"{role}.sock").resolve()),
            "private_key_path": str((tmp_path / f"{role}.key").resolve()),
            "public_key_path": str((tmp_path / f"{role}.pub").resolve()),
            "public_key_file_sha256": hashlib.sha256(
                f"public-file:{role}".encode()
            ).hexdigest(),
            "allowed_slots": [
                slot
                for slot in producers.RECEIPT_SLOTS
                if producers.SLOT_ROLE[slot] == role
            ],
            **authorities[role],
        }
    foundation_unsigned = {
        "schema": producers.PRODUCER_FOUNDATION_SCHEMA,
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "full_canary_terminal_receipt": copy.deepcopy(
            fixture["full_canary_terminal_receipt"]
        ),
        "full_canary_terminal_receipt_sha256": fixture[
            "full_canary_terminal_receipt_sha256"
        ],
        "original_full_canary_owner_approval_sha256": fixture[
            "original_full_canary_owner_approval_sha256"
        ],
        "service_identity_foundation_receipt_sha256": "8" * 64,
        "producer_identity_foundation_receipt_sha256": "7" * 64,
        "owner_id": fixture["owner_id"],
        "bitrix_operational_edge_contract": copy.deepcopy(
            fixture["bitrix_operational_edge_contract"]
        ),
        "discord_edge_evidence_contract": {
            "edge_service_unit": "muncho-discord-egress.service",
            "edge_socket_path": "/run/muncho-discord-egress/edge.sock",
            "edge_service_uid": uid + 30_001,
            "edge_service_gid": gid + 40_001,
            "receipt_public_key_path": (
                "/etc/muncho/keys/discord-edge-receipt-public.pem"
            ),
            "receipt_public_key_id": "a" * 64,
            "receipt_public_key_file_sha256": "b" * 64,
            "connector_service_unit": "muncho-discord-connector.service",
            "connector_socket_path": (
                "/run/muncho-discord-connector/connector.sock"
            ),
            "connector_service_uid": uid + 30_002,
            "connector_service_gid": gid + 40_002,
            "public_history_operation": "public.history.fetch",
            "direct_message_allowed": False,
            "token_or_token_digest_recorded": False,
        },
        "owner_authority": {
            "owner_id": fixture["owner_id"],
            **authorities["owner"],
            "public_key_source": source,
            "public_key_source_sha256": source_sha256,
        },
        "authority_keys": authorities,
        "endpoints": endpoints,
        "receipt_contract": {
            "base_root": str(receipt_root),
            "run_directory_uid": uid,
            "run_directory_gid": gid,
            "run_directory_mode": 0o730,
            "slot_filenames": dict(producers.SLOT_FILENAME),
            "slot_roles": dict(producers.SLOT_ROLE),
            "slot_native_binding_kinds": {
                slot: list(producers.SLOT_NATIVE_BINDING_KINDS[slot])
                for slot in producers.RECEIPT_SLOTS
            },
        },
        "producer_protocol": "role_local_native_evidence_v1",
        "root_can_sign_non_observer_roles": False,
        "token_or_token_digest_recorded": False,
        "signature_namespace": producers.PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
        "signature_algorithm": "sshsig-ed25519-sha512",
    }
    foundation = {
        **foundation_unsigned,
        "owner_signature": _sshsig_signature(
            keys["owner"],
            producers._canonical_bytes(foundation_unsigned),
            namespace_text=producers.PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
        ),
    }
    foundation_sha256 = producers.producer_foundation_sha256(foundation)
    authority_unsigned = {
        "schema": live.FIXTURE_PUBLICATION_AUTHORITY_SCHEMA,
        "run_id": fixture["run_id"],
        "owner_id": fixture["owner_id"],
        "full_canary_terminal_receipt": copy.deepcopy(
            fixture["full_canary_terminal_receipt"]
        ),
        "full_canary_terminal_receipt_sha256": fixture[
            "full_canary_terminal_receipt_sha256"
        ],
        "original_full_canary_owner_approval_sha256": fixture[
            "original_full_canary_owner_approval_sha256"
        ],
        "plan_publication_receipt_sha256": fixture[
            "plan_publication_receipt_sha256"
        ],
        "valid_from_unix_ms": fixture["valid_from_unix_ms"],
        "valid_until_unix_ms": fixture["valid_until_unix_ms"],
        "public_discord_target": fixture["public_discord_target"],
        "producer_foundation_sha256": foundation_sha256,
        "owner_key_id": authorities["owner"]["key_id"],
        "signature_algorithm": "sshsig-ed25519-sha512",
    }
    authority = {
        **authority_unsigned,
        "owner_signature": _sshsig_signature(
            keys["owner"],
            live._canonical_bytes(authority_unsigned),
        ),
    }
    return foundation, authority, public_hex("owner"), source_sha256, receipt_root


def _installed_contract(tmp_path: Path):
    keys, fixture, _digest, _evidence_value, plan, full_plan = (
        _fixture_and_plans()
    )
    foundation, authority, owner_public, source_sha, receipt_root = (
        _foundation_and_authority(
            tmp_path=tmp_path,
            keys=keys,
            fixture=fixture,
            plan=plan,
            full_plan=full_plan,
        )
    )
    reviewed_root = tmp_path / "reviewed"
    publication_root = tmp_path / "fixture-publications"
    live_root = tmp_path / "live"
    for path in (reviewed_root, publication_root, live_root):
        _chmod_directory(path, 0o700)
    source = (reviewed_root / "reviewed-live-fixture.json").resolve()
    uid = reviewed_root.lstat().st_uid
    gid = reviewed_root.lstat().st_gid
    live.install_reviewed_fixture(
        authority,
        producer_foundation=foundation,
        pinned_owner_public_key_ed25519_hex=owner_public,
        pinned_owner_public_key_source_sha256=source_sha,
        plan=plan,
        full_plan=full_plan,
        destination=source,
        receipt_root=receipt_root,
        publication_root=publication_root,
        uid=uid,
        gid=gid,
        host_identity_collector=lambda _plan: {
            "host_identity_sha256": fixture["host_identity_sha256"]
        },
        now_ms=lambda: NOW + 1,
    )
    reviewed = live._strict_json(
        source.read_bytes(), "reviewed_fixture_invalid"
    )
    reviewed_sha256 = contract._digest(reviewed)
    evidence = _evidence(reviewed, reviewed_sha256, keys)
    trusted = live.trust_producer_foundation(
        foundation,
        pinned_owner_public_key_ed25519_hex=owner_public,
        pinned_owner_public_key_source_sha256=source_sha,
    )
    return {
        "keys": keys,
        "fixture": reviewed,
        "fixture_sha256": reviewed_sha256,
        "evidence": evidence,
        "plan": plan,
        "full_plan": full_plan,
        "foundation": foundation,
        "trusted_foundation": trusted,
        "owner_public": owner_public,
        "source_sha": source_sha,
        "source": source,
        "publication_root": publication_root,
        "live_root": live_root,
        "uid": uid,
        "gid": gid,
    }


def _publish_installed_contract(value: dict) -> live.PublishedFixture:
    return live.publish_reviewed_fixture(
        plan=value["plan"],
        full_plan=value["full_plan"],
        producer_foundation=value["foundation"],
        pinned_owner_public_key_ed25519_hex=value["owner_public"],
        pinned_owner_public_key_source_sha256=value["source_sha"],
        source=value["source"],
        publication_root=value["publication_root"],
        live_root=value["live_root"],
        uid=value["uid"],
        gid=value["gid"],
    )


def _inbox(
    tmp_path: Path,
    published: live.PublishedFixture,
) -> live.FixedReceiptInbox:
    root = tmp_path / "producer-receipts"
    _chmod_directory(root, 0o711)
    uid = root.lstat().st_uid
    gid = root.lstat().st_gid
    return live.FixedReceiptInbox(
        fixture=published,
        root=root,
        role_identities={role: (uid, gid) for role in contract.AUTHORITY_ROLES},
        root_identity=(uid, gid),
        run_identity=(uid, gid, 0o730),
        poll_seconds=0.005,
    )


class _ActivationEndpoint:
    def __init__(
        self,
        *,
        role: str,
        foundation: dict,
        observed_at_unix_ms: int,
    ) -> None:
        endpoint = foundation["endpoints"][role]
        self.role = role
        self.socket_path = Path(endpoint["socket_path"])
        self.expected_peer = producers.PeerIdentity(
            os.getpid(), endpoint["uid"], endpoint["gid"]
        )
        self.endpoint = endpoint
        self.foundation = foundation
        self.observed_at_unix_ms = observed_at_unix_ms

    def call(self, value: dict) -> dict:
        if value == {"action": "readiness"}:
            unsigned = {
                "schema": producers.PRODUCER_ENDPOINT_READINESS_SCHEMA,
                "role": self.role,
                "foundation_sha256": producers.producer_foundation_sha256(
                    self.foundation
                ),
                "release_sha": self.foundation["release_sha"],
                "capability_plan_sha256": self.foundation[
                    "capability_plan_sha256"
                ],
                "full_canary_plan_sha256": self.foundation[
                    "full_canary_plan_sha256"
                ],
                "service_unit": self.endpoint["service_unit"],
                "service_identity_sha256": self.endpoint[
                    "service_identity_sha256"
                ],
                "main_pid": self.expected_peer.pid,
                "uid": self.expected_peer.uid,
                "gid": self.expected_peer.gid,
                "socket_path": str(self.socket_path),
                "allowed_slots": self.endpoint["allowed_slots"],
                "key_id": self.endpoint["key_id"],
                "algorithm": "ed25519",
                "public_key_ed25519_hex": self.endpoint[
                    "public_key_ed25519_hex"
                ],
                "public_key_file_sha256": self.endpoint[
                    "public_key_file_sha256"
                ],
                "private_key_or_digest_present": False,
                "observed_at_unix_ms": self.observed_at_unix_ms,
            }
            return {
                **unsigned,
                "readiness_sha256": producers._sha256_json(unsigned),
            }
        readiness = value["readiness"]
        unsigned = {
            "schema": producers.PRODUCER_ENDPOINT_ACTIVATION_SCHEMA,
            "role": self.role,
            "readiness_sha256": readiness["readiness_sha256"],
            "main_pid": self.expected_peer.pid,
            "activated_at_unix_ms": self.observed_at_unix_ms + 1,
        }
        return {
            **unsigned,
            "activation_sha256": producers._sha256_json(unsigned),
        }


def _routeback_identity(installed: dict, *, observed_at_unix_ms: int) -> dict:
    fixture = installed["fixture"]
    unsigned = {
        "schema": (
            "muncho-production-capability-routeback-bot-identity.v1"
        ),
        "plan_sha256": installed["plan"].sha256,
        "full_canary_plan_sha256": installed["full_plan"].sha256,
        "live_bot_user_id": fixture["discord_bot_identities"][
            "routeback_bot_user_id"
        ],
        "planned_routeback_bot_user_id": fixture[
            "discord_bot_identities"
        ]["routeback_bot_user_id"],
        "connector_bot_user_id": fixture["discord_bot_identities"][
            "connector_bot_user_id"
        ],
        "production_bot_user_id": fixture["discord_bot_identities"][
            "production_bot_user_id"
        ],
        "provenance": {
            "source": "discord_rest_api_v10_current_user",
            "http_method": "GET",
            "resource": "/users/@me",
            "credential_boundary": "sealed_routeback_credential_file",
        },
        "pairwise_distinct": True,
        "observed_at_unix": observed_at_unix_ms // 1000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "credential_file_metadata_sha256": "d" * 64,
    }
    return {**unsigned, "attestation_sha256": producers._sha256_json(unsigned)}


def _activate_installed_fleet(
    installed: dict,
    *,
    observed_at_unix_ms: int = NOW,
) -> live.ActivatedProducerFleet:
    clients = {
        role: _ActivationEndpoint(
            role=role,
            foundation=installed["foundation"],
            observed_at_unix_ms=observed_at_unix_ms,
        )
        for role in producers.ENDPOINT_ROLES
    }
    activation = producers.build_fleet_activation(
        foundation=installed["foundation"],
        fixture=installed["fixture"],
        fixture_sha256=installed["fixture_sha256"],
        pinned_owner_public_key_ed25519_hex=installed["owner_public"],
        pinned_owner_public_key_source_sha256=installed["source_sha"],
        owner_grant_sha256="e" * 64,
        owner_catalog_sha256="f" * 64,
        owner_authority_sha256="1" * 64,
        routeback_bot_identity=_routeback_identity(
            installed, observed_at_unix_ms=observed_at_unix_ms
        ),
        endpoint_clients=clients,
        now_ms=lambda: observed_at_unix_ms,
    )
    receipts = producers.activate_fleet_readiness(
        activation,
        endpoint_clients=clients,
        now_ms=observed_at_unix_ms + 1,
    )
    return live.ActivatedProducerFleet(
        readiness=activation,
        endpoint_activation_receipts=receipts,
    )


def _restart_checkpoint(
    *, fixture: dict, fixture_sha256: str, keys: dict
) -> dict:
    return _signed(
        "canonical_writer",
        {
            **_common(
                live.RESTART_CHECKPOINT_SCHEMA,
                fixture,
                fixture_sha256,
                25,
            ),
            "objective_id": "workspace_continuation",
            "worker_service_unit": live.DEFAULT_WORKER_SERVICE_UNIT_NAME,
            "next_unverified_step_id": "step:workspace:2",
            "checkpoint_event_id": "event:workspace:checkpoint",
            "checkpoint_event_sha256": "c" * 64,
            "restart_requested": True,
        },
        fixture=fixture,
        keys=keys,
        slot="worker_restart_checkpoint",
    )


def _slot_receipts(evidence: dict) -> dict[str, dict]:
    bundles = evidence["bundles"]
    return {
        "runtime": evidence["runtime_receipt"],
        "workspace_gateway": bundles["workspace_continuation"]["gateway_receipt"],
        "workspace_writer": bundles["workspace_continuation"]["writer_receipt"],
        "workspace_owner": bundles["workspace_continuation"][
            "owner_approval_receipt"
        ],
        "capability_denials": bundles["capability_denials"],
        "database_reconciliation": bundles["database_reconciliation"],
        "bitrix_edge": bundles["bitrix_boundary"]["edge_receipt"],
        "bitrix_writer": bundles["bitrix_boundary"]["writer_receipt"],
        "discord_edge": bundles["discord_routeback"]["edge_receipt"],
        "discord_writer": bundles["discord_routeback"]["writer_receipt"],
        "failure_gateway": bundles["failure_recovery"]["gateway_receipt"],
        "failure_writer": bundles["failure_recovery"]["writer_receipt"],
        "cleanup": evidence["cleanup_receipt"],
    }


def test_reviewed_objective_is_one_fixed_six_outcome_prompt_without_effort_hint():
    prompt = live.reviewed_objective_prompt()
    assert [item.objective_id for item in live.REVIEWED_OBJECTIVES] == [
        "workspace_continuation",
        "capability_denials",
        "database_reconciliation",
        "bitrix_boundary",
        "discord_routeback",
        "failure_recovery",
    ]
    assert all(prompt.count(f"[{item.objective_id}]") == 1 for item in live.REVIEWED_OBJECTIVES)
    assert "high" not in prompt.lower()
    assert "max" not in prompt.lower()
    assert "todo" not in prompt.lower()


def test_gateway_observer_payloads_join_model_core_to_signed_truth(tmp_path):
    installed = _installed_contract(tmp_path)
    published = _publish_installed_contract(installed)
    receipts = _slot_receipts(installed["evidence"])
    workspace_gateway = receipts["workspace_gateway"]["payload"]
    failure_gateway = receipts["failure_gateway"]["payload"]
    workspace_core = {
        name: copy.deepcopy(workspace_gateway[name])
        for name in live._WORKSPACE_MODEL_PROPOSAL_CORE_FIELDS
    }
    failure_core = {
        name: copy.deepcopy(failure_gateway[name])
        for name in live._FAILURE_MODEL_PROPOSAL_CORE_FIELDS
    }
    runtime_payload = receipts["runtime"]["payload"]
    goal_continuation_evidence = workspace_gateway[
        "goal_continuation_evidence"
    ]
    source = {
        "api_terminal_event_identity": {
            "transcript_sha256": workspace_gateway["transcript_sha256"]
        },
        "runtime_source_identity": {
            "gateway_process_identity_sha256": runtime_payload[
                "gateway_process_identity_sha256"
            ],
            "discord_connector_readiness_sha256": runtime_payload[
                "connector_readiness_receipt_sha256"
            ],
            "connector_bot_user_id": runtime_payload[
                "connector_bot_user_id"
            ],
            "connector_bot_user_id_provenance": runtime_payload[
                "connector_bot_user_id_provenance"
            ],
        },
        "model_proposal_core_identities": {
            "workspace_gateway": {
                "core_sha256": live._sha256_bytes(
                    live._canonical_bytes(workspace_core)
                )
            },
            "failure_gateway": {
                "core_sha256": live._sha256_bytes(
                    live._canonical_bytes(failure_core)
                )
            },
        },
        "frame_records": [
            {
                "observed_at_unix_ms": runtime_payload[
                    "observed_at_unix_ms"
                ]
            }
        ],
        "goal_continuation_identity": (
            live.build_goal_continuation_native_identity(
                goal_continuation_evidence
            )
        ),
        "observed_at_unix_ms": receipts["failure_writer"]["payload"][
            "observed_at_unix_ms"
        ]
        + 1,
    }
    payloads = live._build_gateway_observer_payloads(
        fixture=published,
        source_projection=source,
        model_proposal_cores={
            "workspace_gateway": workspace_core,
            "failure_gateway": failure_core,
        },
        goal_continuation_evidence=goal_continuation_evidence,
        non_observer_receipts={
            name: receipts[name]
            for name in (
                "workspace_owner",
                "workspace_writer",
                "failure_writer",
            )
        },
    )

    assert payloads["runtime"] == runtime_payload
    assert payloads["workspace_gateway"]["transcript_sha256"] == (
        source["api_terminal_event_identity"]["transcript_sha256"]
    )
    assert {
        name: payloads["workspace_gateway"][name]
        for name in live._WORKSPACE_MODEL_PROPOSAL_CORE_FIELDS
    } == workspace_core
    assert {
        name: payloads["failure_gateway"][name]
        for name in live._FAILURE_MODEL_PROPOSAL_CORE_FIELDS
    } == failure_core
    assert payloads["workspace_gateway"]["observed_at_unix_ms"] == (
        receipts["workspace_writer"]["payload"]["observed_at_unix_ms"]
    )

    tampered = copy.deepcopy(workspace_core)
    tampered["owner_grant_sha256"] = "0" * 64
    with pytest.raises(
        live.CapabilityLiveDriverError,
        match="gateway_observer_payload_sources_invalid",
    ):
        live._build_gateway_observer_payloads(
            fixture=published,
            source_projection=source,
            model_proposal_cores={
                "workspace_gateway": tampered,
                "failure_gateway": failure_core,
            },
            goal_continuation_evidence=goal_continuation_evidence,
            non_observer_receipts={
                name: receipts[name]
                for name in (
                    "workspace_owner",
                    "workspace_writer",
                    "failure_writer",
                )
            },
        )


def test_foundation_bound_fixture_publication_is_durable_and_crash_recoverable(
    tmp_path: Path,
):
    keys, fixture, _fixture_sha, _evidence_value, plan, full_plan = (
        _fixture_and_plans()
    )
    foundation, authority, owner_public, source_sha, receipt_root = (
        _foundation_and_authority(
            tmp_path=tmp_path,
            keys=keys,
            fixture=fixture,
            plan=plan,
            full_plan=full_plan,
        )
    )
    destination_parent = tmp_path / "reviewed"
    _chmod_directory(destination_parent, 0o700)
    destination = (destination_parent / "reviewed-live-fixture.json").resolve()
    publication_root = (tmp_path / "fixture-publications").resolve()
    _chmod_directory(publication_root, 0o700)
    uid = destination_parent.lstat().st_uid
    gid = destination_parent.lstat().st_gid
    crashes = 0

    def crash_after_fixture() -> None:
        nonlocal crashes
        crashes += 1
        if crashes == 1:
            raise RuntimeError("crash-after-fixture")

    arguments = {
        "producer_foundation": foundation,
        "pinned_owner_public_key_ed25519_hex": owner_public,
        "pinned_owner_public_key_source_sha256": source_sha,
        "plan": plan,
        "full_plan": full_plan,
        "destination": destination,
        "receipt_root": receipt_root,
        "publication_root": publication_root,
        "uid": uid,
        "gid": gid,
        "host_identity_collector": lambda _plan: {
            "host_identity_sha256": fixture["host_identity_sha256"]
        },
        "now_ms": lambda: NOW + 1,
    }
    with pytest.raises(RuntimeError, match="crash-after-fixture"):
        live.install_reviewed_fixture(
            authority,
            **arguments,
            after_fixture_publication=crash_after_fixture,
        )
    assert destination.exists()
    assert not list(publication_root.rglob("*.json"))

    receipt = live.install_reviewed_fixture(authority, **arguments)
    receipt_path = Path(receipt["receipt_path"])
    assert receipt_path.read_bytes() == live._canonical_bytes(receipt)
    assert receipt["fixture_file_identity"]["inode"] == destination.stat().st_ino
    assert receipt["producer_foundation_sha256"] == producers.producer_foundation_sha256(
        foundation
    )

    # An attacker cannot replace the externally pinned owner key with a
    # self-supplied foundation/fixture authority pair.
    attacker = copy.deepcopy(foundation)
    attacker["owner_authority"]["public_key_ed25519_hex"] = public_hex = (
        producers.Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )
    attacker["owner_authority"]["key_id"] = hashlib.sha256(
        bytes.fromhex(public_hex)
    ).hexdigest()
    with pytest.raises(live.CapabilityLiveDriverError) as raised:
        live.build_reviewed_fixture(
            authority,
            producer_foundation=attacker,
            pinned_owner_public_key_ed25519_hex=owner_public,
            pinned_owner_public_key_source_sha256=source_sha,
            plan=plan,
            full_plan=full_plan,
            receipt_root=receipt_root,
            host_identity_collector=lambda _plan: {
                "host_identity_sha256": fixture["host_identity_sha256"]
            },
            now_ms=lambda: NOW + 1,
        )
    assert raised.value.code == "producer_foundation_invalid"


def test_fixture_republication_is_crash_safe_and_divergence_fails(
    tmp_path: Path, monkeypatch
):
    installed = _installed_contract(tmp_path)
    source = installed["source"]
    fixture = installed["fixture"]

    original_retire = live._retire_exact_source
    calls = 0

    def crash_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("crash-after-live-publication")
        return original_retire(*args, **kwargs)

    monkeypatch.setattr(live, "_retire_exact_source", crash_once)
    with pytest.raises(RuntimeError, match="crash-after-live-publication"):
        _publish_installed_contract(installed)
    published = _publish_installed_contract(installed)
    assert published.path.read_bytes() == live._canonical_bytes(fixture)
    assert not source.exists()

    source.write_bytes(live._canonical_bytes(fixture))
    source.chmod(0o400)
    published.path.chmod(0o600)
    published.path.write_bytes(b"{}")
    published.path.chmod(0o400)
    with pytest.raises(live.CapabilityLiveDriverError) as raised:
        _publish_installed_contract(installed)
    # The durable publication receipt binds the original file identity.  Once
    # that installed byte is replaced, receipt validation is the earliest
    # authoritative failure and must fail closed before source retirement.
    assert raised.value.code == "fixture_publication_receipt_invalid"


def test_exclusive_publication_collision_preserves_concurrent_target(
    tmp_path: Path, monkeypatch
):
    parent = tmp_path / "publish"
    _chmod_directory(parent, 0o700)
    uid = parent.lstat().st_uid
    gid = parent.lstat().st_gid
    target = parent / "artifact.json"
    concurrent = b'{"concurrent":true}'
    original_link = os.link

    def collide(source, destination, *, follow_symlinks=True):
        Path(destination).write_bytes(concurrent)
        Path(destination).chmod(0o400)
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", collide)
    with pytest.raises(live.CapabilityLiveDriverError) as raised:
        live._publish_exclusive(target, b'{"ours":true}', uid=uid, gid=gid)
    monkeypatch.setattr(os, "link", original_link)
    assert raised.value.code == "publication_already_exists"
    assert target.read_bytes() == concurrent


def test_live_driver_routes_all_six_bundles_stops_and_runs_offline_verifier(
    tmp_path: Path,
    monkeypatch,
):
    installed = _installed_contract(tmp_path)
    keys = installed["keys"]
    fixture = installed["fixture"]
    fixture_sha256 = installed["fixture_sha256"]
    evidence = installed["evidence"]
    plan = installed["plan"]
    full_plan = installed["full_plan"]
    published = _publish_installed_contract(installed)
    holder: dict[str, live.FixedReceiptInbox] = {}
    activation_holder: dict[str, str] = {}
    slot_values = _slot_receipts(evidence)

    def inbox_factory(value):
        inbox = _inbox(tmp_path, value)
        holder["inbox"] = inbox
        return inbox

    def put(name: str, value: dict) -> dict:
        prepared = copy.deepcopy(value)
        if prepared.get("authority_role") != "owner":
            prepared["native_evidence"]["producer_readiness_sha256"] = (
                activation_holder["readiness_sha256"]
            )
            _resign_exact_envelope(prepared, keys=keys)
        _write_receipt(holder["inbox"].path(name), prepared)
        return prepared

    def activate(_foundation, _fixture):
        activated = _activate_installed_fleet(installed)
        activation_holder["readiness_sha256"] = activated.readiness[
            "readiness_sha256"
        ]
        put("runtime", slot_values["runtime"])
        return live.ActivatedProducerFleet(
            readiness=activated.readiness,
            endpoint_activation_receipts=(
                activated.endpoint_activation_receipts
            ),
            cleanup_producer=lambda _payload: put(
                "cleanup", slot_values["cleanup"]
            ),
        )

    class Lifecycle:
        stopped = False

        def start(self, _capability, *, defer_producers_until_api_admission=False):
            assert defer_producers_until_api_admission is True
            return {
                "ok": True,
                "receipt_sha256": "1" * 64,
                "gateway_runtime_readiness": {
                    "gateway_pid": 991,
                    "gateway_uid": 2103,
                    "gateway_gid": 2204,
                },
            }

        def prepare_api_admission_inputs(self, _approval, **kwargs):
            publication = kwargs["admission_publisher"]()
            return (
                {"stage": "api-inputs-prepared", "receipt_sha256": "5" * 64},
                publication,
            )

        def start_admitted_producers(self, _approval, **kwargs):
            publication = kwargs["admission_publisher"]()
            activated = kwargs["producer_fleet_activator"]()
            return (
                {
                    "stage": "runtime-live-pending-gateway-commit-ack",
                    "receipt_sha256": "6" * 64,
                },
                activated,
                publication,
            )

        def finalize_api_admission_gateway_commit(self, _approval, **_kwargs):
            model_release.set()
            return {
                "stage": "gateway-commit-acknowledged-pre-model",
                "gateway_commit_acknowledged": True,
                "model_release_allowed": True,
                "model_callback_released": False,
                "receipt_sha256": "7" * 64,
            }

        def stop(self, **kwargs):
            self.stopped = True
            cleanup = kwargs["cleanup_producer"]({})
            retirement = kwargs["producer_activation_retirer"]()
            return {
                "ok": True,
                "receipt_sha256": "2" * 64,
                "cleanup_receipt": cleanup,
                "producer_fleet_retirement": retirement,
                "cleanup_finalization": evidence["cleanup_finalization"],
            }

    class Collector:
        started = False
        closed = False

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    class Client:
        cleared = False

        def run(self, *, fixture: dict, session_id: str):
            assert model_release.wait(1), "model started before gateway commit ACK"
            assert fixture["task_policy"]["prompt"] == live.reviewed_objective_prompt()
            put(
                "worker_restart_checkpoint",
                _restart_checkpoint(
                    fixture=globals_fixture,
                    fixture_sha256=fixture_digest,
                    keys=keys,
                ),
            )
            for name, receipt in slot_values.items():
                if name not in {"runtime", "cleanup"}:
                    put(name, receipt)
            return SSEConversation(
                session_id=session_id,
                session_create_request_id="create-request",
                chat_stream_request_id="stream-request",
                api_run_id="api-run",
                api_message_id="api-message",
                events=(),
                assistant_completed={"content_sha256": "d" * 64},
                run_completed={"completed": True},
                observed_at_unix_ms=NOW + 2,
                completed_at_unix_ms=NOW + 80,
            )

        def clear_secrets(self):
            self.cleared = True

    globals_fixture = fixture
    fixture_digest = fixture_sha256
    lifecycle = Lifecycle()
    collector = Collector()
    client = Client()
    restarts: list[str] = []
    model_release = threading.Event()
    times = iter((NOW + 5, NOW + 15, *(NOW + 20 + index for index in range(100))))
    uid = published.run_directory.lstat().st_uid
    gid = published.run_directory.lstat().st_gid
    monkeypatch.setattr(
        live,
        "_observer_cleanup_payload",
        lambda _publication, **_kwargs: {"mechanical_cleanup": True},
    )
    catalog = {"catalog_sha256": "a" * 64}
    validated_authority = {
        "authority": {"schema": live.API_ADMISSION_OWNER_AUTHORITY_SCHEMA},
        "catalog": catalog,
        "validated_pregrant": {"grant_sha256": "b" * 64},
        "authority_sha256": "c" * 64,
    }
    publication = {
        "run_id": fixture["run_id"],
        "session_id": f"capability_{fixture['run_id']}",
        "capability_epoch_sha256": "d" * 64,
        "catalog_sha256": catalog["catalog_sha256"],
        "authority_sha256": validated_authority["authority_sha256"],
        "readback_verified": True,
        "receipt_sha256": "e" * 64,
    }
    goal_continuation_evidence = copy.deepcopy(
        slot_values["workspace_gateway"]["payload"][
            "goal_continuation_evidence"
        ]
    )
    goal_production_diff = {
        "schema": "muncho-production-capability-production-diff.v1",
        "run_id": published.value["run_id"],
        "fixture_sha256": published.sha256,
        "changed_surfaces": [],
        "production_mutation_observed": False,
        "diff_sha256": goal_continuation_evidence["terminal"][
            "production_diff_sha256"
        ],
    }
    monkeypatch.setattr(
        live,
        "build_live_probe_catalog",
        lambda **_kwargs: catalog,
    )
    monkeypatch.setattr(
        live,
        "validate_api_admission_owner_authority",
        lambda *_args, **_kwargs: validated_authority,
    )
    monkeypatch.setattr(
        live,
        "provision_api_admission_owner_authority",
        lambda *_args, **_kwargs: publication,
    )

    def admission_server(**kwargs):
        request = {
            "session_id": f"capability_{fixture['run_id']}",
            "capability_epoch_sha256": "d" * 64,
            "challenge_sha256": "f" * 64,
        }
        authorization = kwargs["authorizer"](request)
        ready_ack = {
            "schema": "hermes.api.run-admission-ready-ack.v1",
            "session_id": request["session_id"],
            "capability_epoch_sha256": request["capability_epoch_sha256"],
            "challenge_sha256": request["challenge_sha256"],
            "ready_receipt_sha256": "1" * 64,
            "acknowledged": True,
            "receipt_sha256": "2" * 64,
        }
        commitment = kwargs["committer"](request, authorization, ready_ack)
        commit_ack = {
            "schema": "hermes.api.run-admission-commit-ack.v1",
            "session_id": request["session_id"],
            "capability_epoch_sha256": request["capability_epoch_sha256"],
            "challenge_sha256": request["challenge_sha256"],
            "commit_receipt_sha256": "3" * 64,
            "acknowledged": True,
            "receipt_sha256": "4" * 64,
        }
        finalization = kwargs["finalizer"](request, commitment, commit_ack)
        return {"authorization": authorization, "finalization": finalization}
    driver = live.HonestCapabilityCanaryDriver(
        plan=plan,
        full_plan=full_plan,
        lifecycle=lifecycle,
        capability_approval=SimpleNamespace(sha256="3" * 64),
        producer_foundation_check=lambda: installed["trusted_foundation"],
        fixture_publisher=lambda _foundation: published,
        producer_fleet_activator=activate,
        admitted_producer_fleet_activator=(
            lambda foundation, fixed, _authority: activate(foundation, fixed)
        ),
        api_admission_authority_gate=(
            lambda _request, _catalog, _foundation, _fixed, _deadline: {
                "schema": live.API_ADMISSION_OWNER_AUTHORITY_SCHEMA
            }
        ),
        api_admission_server=admission_server,
        producer_fleet_retirer=_retire_activated_fleet,
        production_observation_gate=lambda phase, fixed, _deadline: (
            {
                "phase": "before",
                "run_id": fixed.value["run_id"],
                "fixture_sha256": fixed.sha256,
                "observation_sha256": "6" * 64,
            }
            if phase == "before"
            else {
                "schema": "muncho-production-capability-production-diff.v1",
                "run_id": fixed.value["run_id"],
                "fixture_sha256": fixed.sha256,
                "changed_surfaces": [],
                "production_mutation_observed": False,
                "diff_sha256": "7" * 64,
            }
        ),
        goal_continuation_gate=lambda *_args: (
            goal_continuation_evidence,
            goal_production_diff,
        ),
        gateway_observer_source_publisher=lambda *_args: {
            "mode": "0440",
            "uid": 0,
            "projection_sha256": "8" * 64,
        },
        inbox_factory=inbox_factory,
        collector=collector,
        client_factory=lambda _control, _session: client,
        control_key_reader=lambda: ("control-key", "e" * 64),
        restart=lambda: (
            restarts.append(live.DEFAULT_WORKER_SERVICE_UNIT_NAME)
            or {"receipt_sha256": "f" * 64}
        ),
        evidence_publisher=lambda fixed, value: live.publish_evidence(
            fixed, value, uid=uid, gid=gid
        ),
        verifier=lambda **_kwargs: {"ok": True},
        root_guard=lambda: None,
        receipt_timeout_seconds=2,
        now_ms=lambda: next(times),
        session_key_factory=lambda: "session-key",
    )
    result = driver.run()
    assert result["ok"] is True
    assert result["reviewed_objective_ids"] == [
        item.objective_id for item in live.REVIEWED_OBJECTIVES
    ]
    assert result["offline_verification_receipt"]["ok"] is True
    assert restarts == [live.DEFAULT_WORKER_SERVICE_UNIT_NAME]
    assert lifecycle.stopped is True
    assert collector.started is True
    assert collector.closed is True
    assert client.cleared is True


def test_receipt_bundle_order_is_enforced_even_when_resigned(tmp_path: Path):
    keys, fixture, fixture_sha256, evidence, plan, full_plan = _fixture_and_plans()
    published = live.PublishedFixture(
        value=fixture,
        sha256=fixture_sha256,
        path=tmp_path / "fixture.json",
        run_directory=tmp_path,
    )
    denial = evidence["bundles"]["capability_denials"]
    denial["payload"]["observed_at_unix_ms"] = NOW + 15
    _resign(denial, fixture=fixture, keys=keys)
    with pytest.raises(live.CapabilityLiveDriverError) as raised:
        live._validate_bundles_and_order(
            fixture=published,
            runtime=evidence["runtime_receipt"],
            bundles=evidence["bundles"],
            api_started_at_unix_ms=NOW + 15,
            api_completed_at_unix_ms=NOW + 80,
            checkpoint_at_unix_ms=NOW + 18,
        )
    assert raised.value.code == "signed_receipt_order_invalid"


def test_api_failure_cancels_watcher_before_stop_and_late_checkpoint_cannot_restart(
    tmp_path: Path,
    monkeypatch,
):
    installed = _installed_contract(tmp_path)
    keys = installed["keys"]
    fixture = installed["fixture"]
    fixture_sha256 = installed["fixture_sha256"]
    evidence = installed["evidence"]
    plan = installed["plan"]
    full_plan = installed["full_plan"]
    published = _publish_installed_contract(installed)
    holder: dict[str, live.FixedReceiptInbox] = {}
    activation_holder: dict[str, str] = {}
    events: list[str] = []
    late_thread: threading.Thread | None = None

    def inbox_factory(value):
        inbox = _inbox(tmp_path, value)
        holder["inbox"] = inbox
        return inbox

    def put(name: str, value: dict) -> dict:
        prepared = copy.deepcopy(value)
        if (
            prepared.get("authority_role") != "owner"
            and "readiness_sha256" in activation_holder
        ):
            prepared["native_evidence"]["producer_readiness_sha256"] = (
                activation_holder["readiness_sha256"]
            )
            _resign_exact_envelope(prepared, keys=keys)
        _write_receipt(holder["inbox"].path(name), prepared)
        return prepared

    def activate(_foundation, _fixture):
        activated = _activate_installed_fleet(installed)
        activation_holder["readiness_sha256"] = activated.readiness[
            "readiness_sha256"
        ]
        put("runtime", evidence["runtime_receipt"])
        return live.ActivatedProducerFleet(
            readiness=activated.readiness,
            endpoint_activation_receipts=(
                activated.endpoint_activation_receipts
            ),
            cleanup_producer=lambda _payload: put(
                "cleanup", evidence["cleanup_receipt"]
            ),
        )

    class Lifecycle:
        def start(self, _capability):
            return {"ok": True, "receipt_sha256": "1" * 64}

        def stop(self, **kwargs):
            events.append("stop")
            cleanup = kwargs["cleanup_producer"]({})
            return {
                "ok": True,
                "receipt_sha256": "2" * 64,
                "cleanup_receipt": cleanup,
                "producer_fleet_retirement": kwargs[
                    "producer_activation_retirer"
                ](),
                "cleanup_finalization": evidence["cleanup_finalization"],
            }

    class Collector:
        def start(self):
            events.append("collector-start")

        def close(self):
            events.append("collector-close")

    class Client:
        def run(self, **_kwargs):
            nonlocal late_thread

            def publish_late():
                time.sleep(0.1)
                _write_receipt(
                    holder["inbox"].path("worker_restart_checkpoint"),
                    _restart_checkpoint(
                        fixture=fixture,
                        fixture_sha256=fixture_sha256,
                        keys=keys,
                    ),
                )

            late_thread = threading.Thread(target=publish_late)
            late_thread.start()
            raise RuntimeError("api failed")

        def clear_secrets(self):
            events.append("client-clear")

    restarts: list[str] = []
    monkeypatch.setattr(
        live,
        "_observer_cleanup_payload",
        lambda _publication, **_kwargs: {"mechanical_cleanup": True},
    )
    driver = live.HonestCapabilityCanaryDriver(
        plan=plan,
        full_plan=full_plan,
        lifecycle=Lifecycle(),
        capability_approval=SimpleNamespace(sha256="3" * 64),
        producer_foundation_check=lambda: installed["trusted_foundation"],
        fixture_publisher=lambda _foundation: published,
        producer_fleet_activator=activate,
        producer_fleet_retirer=_retire_activated_fleet,
        production_observation_gate=lambda phase, fixed, _deadline: (
            {
                "phase": "before",
                "run_id": fixed.value["run_id"],
                "fixture_sha256": fixed.sha256,
                "observation_sha256": "6" * 64,
            }
            if phase == "before"
            else {
                "schema": "muncho-production-capability-production-diff.v1",
                "run_id": fixed.value["run_id"],
                "fixture_sha256": fixed.sha256,
                "changed_surfaces": [],
                "production_mutation_observed": False,
                "diff_sha256": "7" * 64,
            }
        ),
        goal_continuation_gate=lambda *_args: ({}, {}),
        gateway_observer_source_publisher=lambda *_args: {
            "mode": "0440",
            "uid": 0,
            "projection_sha256": "8" * 64,
        },
        inbox_factory=inbox_factory,
        collector=Collector(),
        client_factory=lambda _control, _session: Client(),
        control_key_reader=lambda: ("control-key", "e" * 64),
        restart=lambda: restarts.append("restart") or {},
        root_guard=lambda: None,
        receipt_timeout_seconds=1,
        now_ms=lambda: NOW + 1,
        session_key_factory=lambda: "session-key",
    )
    with pytest.raises(RuntimeError, match="capability live run failed closed"):
        driver.run()
    assert late_thread is not None
    late_thread.join()
    assert restarts == []
    assert "stop" in events
    assert events.index("stop") < events.index("collector-close")


def test_partial_collector_start_still_attempts_close_stop_and_aggregates(
    tmp_path: Path,
):
    installed = _installed_contract(tmp_path)
    plan = installed["plan"]
    full_plan = installed["full_plan"]
    published = _publish_installed_contract(installed)
    calls: list[str] = []

    class Lifecycle:
        def start(self, *_args):
            raise AssertionError("not reached")

        def stop(self):
            calls.append("stop")
            raise RuntimeError("stop failed")

    class Collector:
        def start(self):
            calls.append("collector-start")
            raise RuntimeError("partial collector start")

        def close(self):
            calls.append("collector-close")
            raise RuntimeError("collector close failed")

    driver = live.HonestCapabilityCanaryDriver(
        plan=plan,
        full_plan=full_plan,
        lifecycle=Lifecycle(),
        capability_approval=SimpleNamespace(sha256="3" * 64),
        producer_foundation_check=lambda: installed["trusted_foundation"],
        fixture_publisher=lambda _foundation: published,
        producer_fleet_activator=lambda _foundation, _fixture: (
            _activate_installed_fleet(installed)
        ),
        producer_fleet_retirer=_retire_activated_fleet,
        production_observation_gate=lambda phase, fixed, _deadline: (
            {
                "phase": "before",
                "run_id": fixed.value["run_id"],
                "fixture_sha256": fixed.sha256,
                "observation_sha256": "6" * 64,
            }
            if phase == "before"
            else {
                "schema": "muncho-production-capability-production-diff.v1",
                "run_id": fixed.value["run_id"],
                "fixture_sha256": fixed.sha256,
                "changed_surfaces": [],
                "production_mutation_observed": False,
                "diff_sha256": "7" * 64,
            }
        ),
        goal_continuation_gate=lambda *_args: ({}, {}),
        gateway_observer_source_publisher=lambda *_args: {
            "mode": "0440",
            "uid": 0,
            "projection_sha256": "8" * 64,
        },
        inbox_factory=lambda value: _inbox(tmp_path, value),
        collector=Collector(),
        client_factory=lambda *_args: None,
        root_guard=lambda: None,
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        driver.run()
    assert "cleanup failed closed" in str(raised.value)
    assert calls == ["collector-start", "stop", "collector-close"]


def test_missing_producer_foundation_blocks_before_fixture_consumption():
    called = False

    def fixture_publisher(_foundation):
        nonlocal called
        called = True
        raise AssertionError("must not be reached")

    with pytest.raises(live.CapabilityLiveDriverError) as raised:
        live.HonestCapabilityCanaryDriver(
            plan=SimpleNamespace(),
            full_plan=SimpleNamespace(),
            lifecycle=SimpleNamespace(),
            capability_approval=SimpleNamespace(sha256="3" * 64),
            producer_foundation_check=None,  # type: ignore[arg-type]
            fixture_publisher=fixture_publisher,
            producer_fleet_activator=lambda *_args: None,
            producer_fleet_retirer=lambda *_args: {},
            production_observation_gate=lambda *_args: {},
            goal_continuation_gate=lambda *_args: ({}, {}),
            gateway_observer_source_publisher=lambda *_args: {},
            inbox_factory=lambda _value: None,
            collector=SimpleNamespace(),
            client_factory=lambda *_args: None,
            root_guard=lambda: None,
        )
    assert raised.value.code == "producer_foundation_missing"
    assert called is False
