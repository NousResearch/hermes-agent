from __future__ import annotations

import copy
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_capability_canary_producer_units as units
from gateway import canonical_capability_canary_producers as producers
from gateway.discord_history_authority import (
    CANARY_HISTORY_READER_SERVICE_UNIT,
    CANARY_HISTORY_READER_SERVICE_USER,
)
from tests.gateway.test_canonical_capability_canary_producers import (
    _canonical,
    _command,
    _foundation,
    _sshsig,
)


REVISION = "a" * 40


def _probe_catalog() -> dict[str, Any]:
    return producers.build_probe_catalog(
        release_sha=REVISION,
        capability_plan_sha256="b" * 64,
        full_canary_plan_sha256="c" * 64,
        fixture_sha256="d" * 64,
        run_id="run-one",
        session_id="session-one",
        capability_epoch_sha256="e" * 64,
        case_ids={
            name: f"case:{name}"
            for name in (
                "workspace_continuation",
                "capability_denials",
                "database_reconciliation",
                "bitrix_boundary",
                "discord_routeback",
                "failure_recovery",
            )
        },
        workspace={
            "first_path_probe_id": "probe:first",
            "alternate_path_probe_id": "probe:alternate",
            "worker_restart_checkpoint_step_id": "step:restart",
        },
        commands={
            "allowed": [_command("allowed", b"printf allowed")],
            "denied": [
                {
                    "kind": kind,
                    "command": _command(
                        f"denied:{index}", f"deny-{index}".encode()
                    ),
                }
                for index, kind in enumerate(producers.DENIAL_KINDS)
            ],
        },
        database={
            "row_key": "row:one",
            "idempotency_key": "database:one",
            "read_probe_id": "database:read",
            "write_probe_id": "database:write",
            "lost_response_probe_id": "database:lost-response",
        },
        bitrix={
            "handoff_id": "handoff:bitrix",
            "selected_edge_id": "operational-edge:bitrix",
            "read_operation_id": "bitrix.crm.status_list",
            "read_arguments": {"entity_id": "STATUS"},
            "initial_read_probe_id": "bitrix:initial",
            "readback_probe_id": "bitrix:readback",
            "normalized_equality_excluded_fields": ["generated_at_utc"],
            "mutation_operation_id": "bitrix.crm.lead_add",
            "mutation_arguments": dict(
                producers.BITRIX_CANARY_MUTATION_ARGUMENTS
            ),
            "mutation_probe_id": "bitrix:denial",
        },
        discord={
            "public_target": {
                "target_type": "public_channel",
                "guild_id": "1282725267068157972",
                "channel_id": "1504852355588423801",
            },
            "public_idempotency_key": "discord:public",
            "private_target_kind": "dm",
            "private_probe_id": "discord:private",
        },
        failure={
            "probes": [
                {
                    "component": component,
                    "failure_id": f"failure:{component}",
                    "alternative_available": True,
                    "alternative_id": f"alternative:{component}",
                }
                for component in producers.FAILURE_COMPONENTS
            ]
        },
    )


def _identities() -> dict[str, dict[str, Any]]:
    values = {
        "business_edge": ("muncho-cap-business", "muncho-cap-business", 2201, 2301),
        "canonical_writer": ("muncho-cap-writer", "muncho-cap-writer", 2202, 2302),
        "discord_edge": ("muncho-cap-discord", "muncho-cap-discord", 2203, 2303),
        "gateway_observer": ("muncho-cap-observer", "muncho-cap-observer", 2204, 2304),
    }
    return {
        role: {
            "user": value[0],
            "group": value[1],
            "uid": value[2],
            "gid": value[3],
            "receipt_writer_gid": 2401,
            "bitrix_socket_gid": (
                2501
                if role in {"business_edge", "canonical_writer"}
                else None
            ),
        }
        for role, value in values.items()
    }


def _key_bootstrap(tmp_path: Path) -> units.ProducerKeyBootstrap:
    key_root = tmp_path / "keys"
    key_root.mkdir(mode=0o700)
    identity = key_root.lstat()
    return units.bootstrap_producer_keys(
        role_identities=_identities(),
        key_root=key_root,
        receipt_path=key_root / "bootstrap.json",
        root_uid=identity.st_uid,
        root_gid=identity.st_gid,
    )


def _production_foundation(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], units.ProducerKeyBootstrap]:
    prior, context = _foundation(tmp_path)
    identity = units.render_producer_unit_identity_contract(
        revision=REVISION,
        role_identities=_identities(),
    )
    keys = _key_bootstrap(tmp_path)
    endpoints = units.endpoint_contracts(
        identity_contract=identity,
        key_bootstrap=keys,
    )
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in prior.items()
        if key != "owner_signature"
    }
    unsigned["endpoints"] = endpoints
    unsigned["authority_keys"] = {
        **{
            role: {
                "key_id": keys.public_contracts[role]["key_id"],
                "algorithm": "ed25519",
                "public_key_ed25519_hex": keys.public_contracts[role][
                    "public_key_ed25519_hex"
                ],
            }
            for role in producers.ENDPOINT_ROLES
        },
        "owner": copy.deepcopy(prior["authority_keys"]["owner"]),
    }
    unsigned["bitrix_operational_edge_contract"]["identity_bootstrap"][
        "socket_client_gid"
    ] = _identities()["business_edge"]["bitrix_socket_gid"]
    unsigned["receipt_contract"] = {
        **copy.deepcopy(prior["receipt_contract"]),
        "base_root": str(producers.DEFAULT_RECEIPT_ROOT),
        "run_directory_uid": 0,
        "run_directory_gid": 2401,
        "run_directory_mode": 0o3770,
    }
    signature = _sshsig(
        context["owner_private"],
        producers.producer_foundation_signature_payload(unsigned),
        namespace=producers.PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
    )
    foundation = dict(
        producers.seal_producer_foundation(
            unsigned,
            owner_signature=signature,
            pinned_owner_public_key_ed25519_hex=context["owner_public"],
            pinned_owner_public_key_source_sha256=context["source_sha256"],
        )
    )
    return foundation, context, keys


def test_four_distinct_nonroot_units_bind_keys_groups_and_signed_inbox(
    tmp_path: Path,
) -> None:
    foundation, context, _keys = _production_foundation(tmp_path)
    bundle = units.render_producer_units(
        foundation=foundation,
        pinned_owner_public_key_ed25519_hex=context["owner_public"],
        pinned_owner_public_key_source_sha256=context["source_sha256"],
        role_identities=_identities(),
    )

    assert len(bundle.units) == 4
    assert len(bundle.configs) == 4
    assert bundle.manifest["config_install_contract"] == {
        str(units.producer_config_path(role)): {
            "uid": 0,
            "gid": identity["gid"],
            "mode": 0o440,
        }
        for role, identity in _identities().items()
    }
    assert bundle.manifest["native_root_contract"] == {
        str(units.DEFAULT_NATIVE_ROOT): {"uid": 0, "gid": 0, "mode": 0o755},
        **{
            str(units.DEFAULT_NATIVE_ROOT / role): {
                "uid": identity["uid"],
                "gid": identity["gid"],
                "mode": 0o700,
            }
            for role, identity in _identities().items()
        },
    }
    assert bundle.manifest["auxiliary_files"] == {
        str(units.DEFAULT_TMPFILES_PATH): producers._sha256_bytes(
            bundle.auxiliary_files[str(units.DEFAULT_TMPFILES_PATH)]
        )
    }
    tmpfiles = bundle.auxiliary_files[str(units.DEFAULT_TMPFILES_PATH)].decode(
        "ascii"
    )
    assert "d /run/muncho-capability-canary 0700 root root -\n" in tmpfiles
    for role, identity in _identities().items():
        assert (
            f"d {units.DEFAULT_NATIVE_ROOT / role} 0700 "
            f"{identity['user']} {identity['group']} -\n"
        ) in tmpfiles
    assert bundle.manifest["receipt_writer_group"] == {
        "group": units.PRODUCER_RECEIPT_WRITER_GROUP,
        "gid": 2401,
        "members": list(producers.ENDPOINT_ROLES),
        "run_directory_uid": 0,
        "run_directory_gid": 2401,
        "run_directory_mode": 0o3770,
        "cross_role_precreate_is_fail_closed_dos_only": True,
        "accepted_file_owner_must_match_role_uid_gid": True,
    }
    for role, identity in _identities().items():
        path = f"/etc/systemd/system/{producers.PRODUCER_SERVICE_UNITS[role]}"
        text = bundle.units[path].decode("ascii")
        assert f"User={identity['user']}\n" in text
        assert f"Group={identity['group']}\n" in text
        assert f"LoadCredential=producer-private-key:{producers.DEFAULT_KEY_ROOT}/{role}-private.pem\n" in text
        assert f"LoadCredential=producer-public-key:{producers.DEFAULT_KEY_ROOT}/{role}-public.pem\n" in text
        assert f"InaccessiblePaths={producers.DEFAULT_KEY_ROOT}\n" in text
        for secret_path in units.PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS:
            assert f"InaccessiblePaths=-{secret_path}\n" in text
        assert "Restart=no\n" in text
        assert "RuntimeMaxSec=900s\n" in text
        supplementary = next(
            row for row in text.splitlines() if row.startswith("SupplementaryGroups=")
        )
        assert units.PRODUCER_RECEIPT_WRITER_GROUP in supplementary
        assert (
            producers.BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP in supplementary
        ) is (role in {"business_edge", "canonical_writer"})
        config = json.loads(
            bundle.configs[str(units.producer_config_path(role))]
        )
        assert config["service_uid"] == identity["uid"]
        assert config["service_gid"] == identity["gid"]
        assert config["receipt_directory_gid"] == 2401
        assert config["receipt_directory_mode"] == 0o3770
        assert config["public_key_path"] == str(
            units.producer_public_key_path(role)
        )
        assert config["private_key_path"] == str(
            units.producer_private_key_projection_path(role)
        )
    assert CANARY_HISTORY_READER_SERVICE_UNIT == (
        producers.PRODUCER_SERVICE_UNITS["discord_edge"]
    )
    assert CANARY_HISTORY_READER_SERVICE_USER == (
        units.PRODUCER_ROLE_ACCOUNTS["discord_edge"][0]
    )
    assert bundle.manifest["private_key_content_or_digest_recorded"] is False
    assert bundle.manifest["authority_key_lifecycle"] == {
        "ownership": "owner_signed_foundation",
        "durable_across_canary_runs": True,
        "retired_per_run": False,
        "source_root_inaccessible_to_service": True,
        "private_keys_projected_by_systemd": True,
        "per_run_activation_readiness_retired_after_service_stop": True,
    }
    assert "PRIVATE KEY" not in json.dumps(bundle.manifest)


def test_unit_identity_rejects_any_uid_gid_or_group_alias() -> None:
    duplicate_uid = copy.deepcopy(_identities())
    duplicate_uid["discord_edge"]["uid"] = duplicate_uid["business_edge"]["uid"]
    with pytest.raises(
        producers.CapabilityProducerError,
        match="producer_role_identity_not_separated",
    ):
        units.render_producer_unit_identity_contract(
            revision=REVISION,
            role_identities=duplicate_uid,
        )

    observer_root = copy.deepcopy(_identities())
    observer_root["gateway_observer"]["uid"] = 0
    with pytest.raises(
        producers.CapabilityProducerError,
        match="producer_role_identity_invalid",
    ):
        units.render_producer_unit_identity_contract(
            revision=REVISION,
            role_identities=observer_root,
        )


def test_key_bootstrap_is_exactly_idempotent_and_public_source_is_root_only(
    tmp_path: Path,
) -> None:
    key_root = tmp_path / "keys"
    key_root.mkdir(mode=0o700)
    key_root_identity = key_root.lstat()
    kwargs = {
        "role_identities": _identities(),
        "key_root": key_root,
        "receipt_path": key_root / "bootstrap.json",
        "root_uid": key_root_identity.st_uid,
        "root_gid": key_root_identity.st_gid,
    }
    first = units.bootstrap_producer_keys(**kwargs)
    first_bytes = {
        path.name: path.read_bytes() for path in key_root.iterdir()
    }
    second = units.bootstrap_producer_keys(**kwargs)
    assert second.value == first.value
    assert {path.name: path.read_bytes() for path in key_root.iterdir()} == first_bytes
    for role in producers.ENDPOINT_ROLES:
        public = key_root / f"{role}-public.pem"
        private = key_root / f"{role}-private.pem"
        assert stat.S_IMODE(public.stat().st_mode) == 0o400
        assert stat.S_IMODE(private.stat().st_mode) == 0o400
        assert first.public_contracts[role]["public_key_source_path"] == str(public)
        assert first.public_contracts[role]["public_key_projection_path"] == str(
            units.producer_public_key_path(role)
        )

    public = key_root / "business_edge-public.pem"
    public.chmod(0o600)
    public.write_bytes(b"drifted-public")
    public.chmod(0o400)
    with pytest.raises(producers.CapabilityProducerError):
        units.bootstrap_producer_keys(**kwargs)


def test_role_owned_native_publication_binds_exact_payload_and_kinds(
    tmp_path: Path,
) -> None:
    role = "canonical_writer"
    run_id = "run-one"
    slot = "bitrix_writer"
    directory = tmp_path / role / run_id
    directory.mkdir(parents=True, mode=0o700)
    directory_identity = directory.lstat()
    payload = {
        "run_id": run_id,
        "release_sha": REVISION,
        "fixture_sha256": "f" * 64,
        "handoff_id": "handoff:bitrix",
    }
    binding = producers.NativeEvidenceBinding(
        kind="canonical_writer_handoff_events",
        source_identity_sha256="1" * 64,
        artifact_sha256="2" * 64,
        verification_receipt_sha256="3" * 64,
    ).to_mapping()
    unsigned = {
        "schema": units.NATIVE_PUBLICATION_SCHEMA,
        "role": role,
        "slot": slot,
        "run_id": run_id,
        "release_sha": REVISION,
        "fixture_sha256": "f" * 64,
        "payload": payload,
        "payload_sha256": producers._sha256_json(payload),
        "bindings": [binding],
    }
    value = {
        **unsigned,
        "publication_sha256": producers._sha256_json(unsigned),
    }
    path = directory / f"{slot}.json"
    path.write_bytes(_canonical(value))
    path.chmod(0o400)
    collector = units.RoleOwnedNativePublicationCollector(
        role=role,
        uid=directory_identity.st_uid,
        gid=directory_identity.st_gid,
        root=tmp_path,
        partial_kinds={slot: ("canonical_writer_handoff_events",)},
    )
    assert tuple(
        item.kind for item in collector.collect(slot=slot, payload=payload)
    ) == ("canonical_writer_handoff_events",)
    with pytest.raises(
        producers.CapabilityProducerError,
        match="native_publication_invalid",
    ):
        collector.collect(slot=slot, payload={**payload, "handoff_id": "changed"})


def test_canonical_writer_collector_queries_peer_authorized_projection() -> None:
    catalog = _probe_catalog()
    terminal = {
        "case_id": catalog["case_ids"]["workspace_continuation"],
        "terminal_event_id": "event:terminal",
        "terminal_event_sha256": "1" * 64,
    }
    payload = {
        "run_id": catalog["run_id"],
        "release_sha": catalog["release_sha"],
        "fixture_sha256": catalog["fixture_sha256"],
        "owner_grant_id": "grant:one",
        "owner_grant_sha256": "2" * 64,
        "terminal_ctw": terminal,
    }

    class Client:
        calls: list[tuple[Any, ...]] = []

        def call(self, operation, request, *, runtime):
            self.calls.append((operation, request, runtime))
            return SimpleNamespace(
                request_id="request:one",
                result={
                    "events": [
                        {
                            "event_id": terminal["terminal_event_id"],
                            "case_id": terminal["case_id"],
                            "content_sha256": terminal[
                                "terminal_event_sha256"
                            ],
                            "body": {
                                "owner_grant_id": payload["owner_grant_id"],
                                "owner_grant_sha256": payload[
                                    "owner_grant_sha256"
                                ],
                            },
                        }
                    ],
                    "has_more": False,
                },
            )

    client = Client()
    collector = units.CanonicalWriterProjectionNativeCollector(
        client=client,
        catalog=catalog,
        release_sha=REVISION,
        capability_plan_sha256="b" * 64,
        full_canary_plan_sha256="c" * 64,
        source_identity={
            "service_unit": "muncho-canonical-writer.service",
            "peer_authorization": "exact_current_systemd_main_pid_each_call",
        },
    )
    bindings = collector.collect(slot="workspace_writer", payload=payload)

    assert tuple(item.kind for item in bindings) == (
        "canonical_writer_resume_bundle",
        "canonical_writer_projection_events",
    )
    assert client.calls == [
        (
            "projection.read_events",
            {
                "case_id": terminal["case_id"],
                "after_event_id": "",
                "limit": 500,
            },
            {"platform": "capability-canary-producer"},
        )
    ]
    with pytest.raises(
        producers.CapabilityProducerError,
        match="canonical_writer_native_evidence_invalid",
    ):
        collector.collect(
            slot="workspace_writer",
            payload={**payload, "owner_grant_sha256": "3" * 64},
        )


def test_discord_edge_collector_verifies_signed_journal_and_public_history(
) -> None:
    from gateway.discord_edge_protocol import (
        DiscordEdgeAuthorityKind,
        DiscordEdgeIntent,
        DiscordEdgeOperation,
        DiscordEdgeReceiptOutcome,
        DiscordPublicTarget,
        make_request,
        sign_capability,
        sign_receipt,
        verify_request_capability,
    )

    catalog = _probe_catalog()
    target_value = catalog["discord"]["public_target"]
    observed_at_unix_ms = 2_000_000_000_000
    content = "Exact public canary response"
    message_id = "1504852355588423802"
    bot_user_id = "1504852355588423803"
    writer_key = Ed25519PrivateKey.generate()
    edge_key = Ed25519PrivateKey.generate()
    target = DiscordPublicTarget.from_mapping(
        {
            "target_type": "public_guild_channel",
            "guild_id": target_value["guild_id"],
            "channel_id": target_value["channel_id"],
        }
    )
    intent = DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        target=target,
        payload={"content": content},
        idempotency_key=catalog["discord"]["public_idempotency_key"],
    )
    capability_envelope = sign_capability(
        writer_key,
        intent,
        authority_kind=DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
        authority_ref="routeauth:canary:discord",
        issued_at_unix_ms=observed_at_unix_ms - 1_000,
        expires_at_unix_ms=observed_at_unix_ms + 60_000,
        capability_id="20000000-0000-4000-8000-000000000001",
    )
    request = make_request(
        intent,
        capability_envelope,
        request_id="30000000-0000-4000-8000-000000000001",
        now_unix_ms=observed_at_unix_ms - 500,
    )
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=observed_at_unix_ms,
    )
    receipt = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id=message_id,
        bot_user_id=bot_user_id,
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=intent.content_sha256,
        occurred_at_unix_ms=observed_at_unix_ms,
        receipt_id="40000000-0000-4000-8000-000000000001",
    )

    class EdgeClient:
        def __init__(self) -> None:
            self.queries: list[Any] = []

        def reconcile(self, query, *, require_preconnected):
            assert require_preconnected is False
            self.queries.append(query)
            return SimpleNamespace(
                request=request,
                state="verified",
                blocker=None,
                replayed=True,
                receipt=receipt,
            )

    class HistoryClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def read(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "messages": [
                    {
                        "message_id": message_id,
                        "author_id": bot_user_id,
                        "author_is_bot": True,
                        "content_truncated": False,
                        "content": content,
                    }
                ]
            }

    edge_client = EdgeClient()
    history_client = HistoryClient()
    contract = {
        "edge_service_unit": "muncho-discord-egress.service",
        "edge_socket_path": "/run/muncho-discord-egress/edge.sock",
        "edge_service_uid": 2110,
        "edge_service_gid": 2210,
        "receipt_public_key_path": (
            "/etc/muncho/keys/discord-edge-receipt-public.pem"
        ),
        "receipt_public_key_id": "a" * 64,
        "receipt_public_key_file_sha256": "b" * 64,
        "connector_service_unit": "muncho-discord-connector.service",
        "connector_socket_path": (
            "/run/muncho-discord-connector/connector.sock"
        ),
        "connector_service_uid": 2111,
        "connector_service_gid": 2211,
        "public_history_operation": "public.history.fetch",
        "direct_message_allowed": False,
        "token_or_token_digest_recorded": False,
    }
    collector = units.DiscordEdgeNativeCollector(
        edge_client=edge_client,
        history_client=history_client,
        edge_public_key=edge_key.public_key(),
        catalog=catalog,
        contract=contract,
    )
    private_denial = {
        "probe_id": catalog["discord"]["private_probe_id"],
        "target_kind": "dm",
        "blocker_code": "forbidden_target",
        "dispatch_attempted": False,
    }
    payload = {
        "run_id": catalog["run_id"],
        "release_sha": catalog["release_sha"],
        "fixture_sha256": catalog["fixture_sha256"],
        **target_value,
        "idempotency_key_sha256": producers._sha256_bytes(
            intent.idempotency_key.encode("utf-8")
        ),
        "request_sha256": intent.request_sha256,
        "content_sha256": intent.content_sha256,
        "platform_message_id": message_id,
        "routeback_bot_user_id": bot_user_id,
        "public_receipt_sha256": producers._sha256_json(
            receipt.to_message()
        ),
        "private_target_kind": "dm",
        "private_dispatch_attempted": False,
        "journal_unchanged_after_private_probe": True,
        "private_denial_receipt_sha256": producers._sha256_json(
            private_denial
        ),
        "observed_at_unix_ms": observed_at_unix_ms,
    }

    bindings = collector.collect(slot="discord_edge", payload=payload)

    assert tuple(item.kind for item in bindings) == (
        "discord_edge_signed_receipt",
        "discord_edge_journal_readback",
        "discord_public_readback",
        "discord_private_predispatch_denial",
        "routeback_bot_identity",
    )
    assert len(edge_client.queries) == 2
    assert edge_client.queries[0].to_message() == edge_client.queries[1].to_message()
    assert history_client.calls == [
        {"channel_id": target.channel_id, "limit": 25}
    ]
    with pytest.raises(
        producers.CapabilityProducerError,
        match="discord_native_evidence_invalid",
    ):
        collector.collect(
            slot="discord_edge",
            payload={**payload, "public_receipt_sha256": "f" * 64},
        )


def test_gateway_observer_cleanup_reloads_facts_and_rejects_root_outcomes(
    tmp_path: Path,
) -> None:
    foundation, context, _keys = _production_foundation(tmp_path)
    config = producers.ProducerConfig.from_mapping(
        json.loads(
            units._producer_config(
                role="gateway_observer",
                foundation=foundation,
            )
        )
    )
    stopped_state = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "disabled",
        "MainPID": 0,
        "FragmentPath": "/etc/systemd/system/stopped.service",
        "DropInPaths": "",
        "Type": "simple",
        "NotifyAccess": "none",
        "StatusText": "",
    }
    observer_state = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4242,
        "FragmentPath": f"/etc/systemd/system/{units._CLEANUP_OBSERVER_UNIT}",
        "DropInPaths": "",
        "Type": "simple",
        "NotifyAccess": "none",
        "StatusText": "",
    }
    states = {
        unit: dict(stopped_state)
        for unit in units._CLEANUP_NON_OBSERVER_SERVICE_UNITS
    }
    bundle = units.render_producer_units(
        foundation=foundation,
        pinned_owner_public_key_ed25519_hex=context["owner_public"],
        pinned_owner_public_key_source_sha256=context["source_sha256"],
        role_identities=_identities(),
    )
    inaccessibility_sha256 = producers._sha256_json(
        {
            "paths": list(units.PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS),
            "applies_to_roles": list(producers.ENDPOINT_ROLES),
            "unit_hash_bound": True,
            "cleanup_observer_has_no_credential_read_access": True,
        }
    )
    foundation_sha256 = producers.producer_foundation_sha256(foundation)
    observed_at_unix_ms = 2_000_000_000_000
    proof_unsigned = {
        "schema": (
            "muncho-production-capability-credential-consumer-stop-proof.v1"
        ),
        "plan_sha256": foundation["capability_plan_sha256"],
        "non_observer_stop_order": list(
            units._CLEANUP_NON_OBSERVER_SERVICE_UNITS
        ),
        "non_observer_services_state_sha256": producers._sha256_json(states),
        "all_credential_consumers_stopped": True,
        "observer_service_unit": units._CLEANUP_OBSERVER_UNIT,
        "observer_state_sha256": producers._sha256_json(observer_state),
        "observer_live_signing_only": True,
        "observer_credential_read_access": False,
        "producer_foundation_sha256": foundation_sha256,
        "unit_bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
        "credential_inaccessibility_contract_sha256": (
            inaccessibility_sha256
        ),
        "observed_at_unix": observed_at_unix_ms // 1000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    proof = {
        **proof_unsigned,
        "stop_proof_sha256": producers._sha256_json(proof_unsigned),
    }
    retirements: dict[str, Mapping[str, Any]] = {}
    absence: dict[str, Mapping[str, Any]] = {}
    for index, binding in enumerate(units._CLEANUP_CREDENTIAL_BINDINGS):
        target = f"/run/canary-secret-{index}"
        unsigned = {
            "credential_binding": binding,
            "target_path": target,
            "service_stop_proof_sha256": proof["stop_proof_sha256"],
        }
        retirements[binding] = {
            **unsigned,
            "receipt_sha256": producers._sha256_json(unsigned),
        }
        absence[binding] = {"path": target, "absent": True}
    key_unsigned = {
        "service_stop_proof_sha256": proof["stop_proof_sha256"],
        "both_pair_members_absent": True,
    }
    key_retirement = {
        **key_unsigned,
        "receipt_sha256": producers._sha256_json(key_unsigned),
    }
    key_absence = {
        "private_path": "/run/bitrix-private",
        "private_absent": True,
        "public_path": "/run/bitrix-public",
        "public_absent": True,
        "both_pair_members_absent": True,
    }
    observer_identity = {
        "role": "gateway_observer",
        "service_unit": units._CLEANUP_OBSERVER_UNIT,
        "live": True,
        "signing_only": True,
        "credential_read_access": False,
        "service_state_sha256": producers._sha256_json(observer_state),
        "producer_foundation_sha256": foundation_sha256,
        "unit_bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
        "credential_inaccessibility_contract_sha256": (
            inaccessibility_sha256
        ),
    }
    facts_unsigned = {
        "schema": "muncho-production-capability-cleanup-facts.v1",
        "revision": REVISION,
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "non_observer_stop_order": list(
            units._CLEANUP_NON_OBSERVER_SERVICE_UNITS
        ),
        "non_observer_service_states": states,
        "credential_consumer_stop_proof": proof,
        "observer_signer_identity": observer_identity,
        "retirements": retirements,
        "retirement_receipt_sha256s": {
            binding: value["receipt_sha256"]
            for binding, value in retirements.items()
        },
        "credential_absence": absence,
        "bitrix_receipt_key_retirement": key_retirement,
        "bitrix_receipt_key_absence": key_absence,
        "browser_session_retirement": {
            "path": "/run/browser",
            "empty": True,
            "retired": True,
            "secret_material_recorded": False,
        },
        "isolated_worker_lease_cleanup": {
            "path": "/run/worker",
            "empty": True,
            "retired": True,
            "secret_material_recorded": False,
        },
        "observed_at_unix_ms": observed_at_unix_ms,
    }
    facts = {
        **facts_unsigned,
        "facts_sha256": producers._sha256_json(facts_unsigned),
    }
    surface_sha256s = {
        name: producers._sha256_json({"surface": name})
        for name in units.PRODUCTION_DIFF_CATEGORIES
    }
    diff_unsigned = {
        "schema": "muncho-production-capability-production-diff.v1",
        "canary_revision": REVISION,
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "fixture_sha256": "d" * 64,
        "run_id": "run-one",
        "target": {"vm": "ai-platform-runtime-01"},
        "before_envelope_sha256": "1" * 64,
        "after_envelope_sha256": "2" * 64,
        "before_observation_sha256": "3" * 64,
        "after_observation_sha256": "4" * 64,
        "before_observed_at_unix_ms": observed_at_unix_ms - 2,
        "after_observed_at_unix_ms": observed_at_unix_ms - 1,
        "static_before_sha256": "5" * 64,
        "static_after_sha256": "5" * 64,
        "changed_surfaces": [],
        "surface_diffs": {
            name: {
                "before_sha256": digest,
                "after_sha256": digest,
                "changed": False,
            }
            for name, digest in surface_sha256s.items()
        },
        "expected_change_contract_sha256": "6" * 64,
        "unexpected_change_count": 0,
        "production_mutation_observed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_job_content_recorded": False,
    }
    diff = {
        **diff_unsigned,
        "diff_sha256": producers._sha256_json(diff_unsigned),
    }
    state_reader = lambda unit: (
        observer_state
        if unit == units._CLEANUP_OBSERVER_UNIT
        else states[unit]
    )
    collector = units.GatewayObserverCleanupNativeCollector(
        config=config,
        foundation=foundation,
        service_state_reader=state_reader,
        cleanup_facts_reader=lambda _root, _gid: (
            facts,
            producers._sha256_json(facts),
        ),
        production_diff_reader=lambda _root, _gid: diff,
    )
    payload = {
        "schema": "muncho-production-capability-cleanup.v1",
        "run_id": "run-one",
        "release_sha": REVISION,
        "fixture_sha256": "d" * 64,
        "observed_at_unix_ms": observed_at_unix_ms,
        "non_observer_service_units": list(
            units._CLEANUP_NON_OBSERVER_SERVICE_UNITS
        ),
        "non_observer_services_stopped": True,
        "non_observer_services_state_sha256": proof[
            "non_observer_services_state_sha256"
        ],
        "gateway_observer_signer_identity": observer_identity,
        "credential_consumer_stop_proof": proof,
        "credential_leases": list(units._CLEANUP_CREDENTIAL_BINDINGS),
        "credential_leases_retired": True,
        "retirements": retirements,
        "retirement_receipt_sha256s": facts[
            "retirement_receipt_sha256s"
        ],
        "credential_absence": absence,
        "credentials_absent": True,
        "bitrix_receipt_key_retirement": key_retirement,
        "bitrix_receipt_key_absence": key_absence,
        "discord_credential_topology": dict(
            units._CLEANUP_DISCORD_CREDENTIAL_TOPOLOGY
        ),
        "browser_session_retired": True,
        "isolated_worker_lease_cleanup_verified": True,
        "production_diff_sha256": diff["diff_sha256"],
    }

    bindings = collector.collect(slot="cleanup", payload=payload)
    assert tuple(item.kind for item in bindings) == producers.SLOT_NATIVE_BINDING_KINDS[
        "cleanup"
    ]
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_cleanup_evidence_invalid",
    ):
        collector.collect(
            slot="cleanup",
            payload={**payload, "credentials_absent": False},
        )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_cleanup_evidence_invalid",
    ):
        collector.collect(
            slot="cleanup",
            payload={**payload, "run_id": "another-run"},
        )

    tampered_facts = {**facts, "observed_at_unix_ms": observed_at_unix_ms + 1}
    tampered = units.GatewayObserverCleanupNativeCollector(
        config=config,
        foundation=foundation,
        service_state_reader=state_reader,
        cleanup_facts_reader=lambda _root, _gid: (
            tampered_facts,
            producers._sha256_json(tampered_facts),
        ),
        production_diff_reader=lambda _root, _gid: diff,
    )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_cleanup_evidence_invalid",
    ):
        tampered.collect(slot="cleanup", payload=payload)


def test_gateway_observer_source_projection_is_redacted_and_chain_bound(
    tmp_path: Path,
) -> None:
    foundation, _context, _keys = _production_foundation(tmp_path)
    source_fixture_sha256 = "7" * 64
    capability_fixture_sha256 = "8" * 64
    peer = SimpleNamespace(pid=4242, uid=2200, gid=2300, start_time_ticks=99)
    collector_service_sha256 = "9" * 64
    edge_service_sha256 = "a" * 64
    workspace_core = {
        "session_id": "session-one",
        "capability_epoch_sha256": "b" * 64,
        "task_workspace_evidence_sha256s": ["b" * 64, "c" * 64],
        "first_path_failure_receipt_sha256": "d" * 64,
        "alternate_read_receipt_sha256": "e" * 64,
        "model_requested_effort": "max",
        "later_request_effort": "max",
        "reasoning_tool_call_id": "call-one",
        "restart_count": 1,
        "used_command_sha256s": ["2" * 64, "3" * 64],
        "mutation_receipt_sha256s": ["8" * 64, "9" * 64],
        "approval_prompt_count": 0,
        "microapproval_prompt_count": 0,
        "replayed_mutation_count": 0,
        "owner_grant_id": "approval-one",
        "owner_grant_sha256": "1" * 64,
        "consumed_command_sha256s": ["2" * 64, "3" * 64],
        "terminal_plan_id": "plan-one",
        "terminal_plan_revision": 1,
    }
    failure_digests = [
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "8" * 64,
        "9" * 64,
    ]
    failure_core = {
        "failures": [
            {
                "component": component,
                "failure_observed": True,
                "failure_receipt_sha256": failure_digests[index * 2],
                "alternative_available": True,
                "alternative_attempted": True,
                "alternative_receipt_sha256": failure_digests[index * 2 + 1],
            }
            for index, component in enumerate(
                units._GATEWAY_OBSERVER_FAILURE_COMPONENTS
            )
        ],
        "model_retained_tool_control": True,
    }
    proposal_event = {
        "event_id": "event-gateway-proposal",
        "event_type": units.GATEWAY_OBSERVER_PROPOSAL_EVENT_TYPE,
        "case_id": "case:one",
        "occurred_at": "2026-07-15T00:00:00Z",
        "payload": {
            "evidence": [
                {
                    "schema": units.GATEWAY_OBSERVER_PROPOSAL_CORE_SCHEMA,
                    "slot": "workspace_gateway",
                    "core": workspace_core,
                },
                {
                    "schema": units.GATEWAY_OBSERVER_PROPOSAL_CORE_SCHEMA,
                    "slot": "failure_gateway",
                    "core": failure_core,
                },
            ]
        },
        "safety": {},
    }
    proposal_readback = {
        "case_id": "case:one",
        "events": [proposal_event],
        "truncated": False,
    }
    payloads = (
        ("plugin_ready", {"module_sha256": "b" * 64}),
        ("api_session_bound", {"success": True}),
        ("private_target_probe_ready", {"attempt_frame_sha256": "c" * 64}),
        ("private_target_probe_result", {"outcome": "blocked"}),
        (
            "pre_api_request",
            {
                "request_ordinal": 1,
                "reasoning_effort": "high",
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-5.6-sol",
                "api_request_sha256": "d" * 64,
            },
        ),
        (
            "post_api_request",
            {"request_ordinal": 1, "response_payload_sha256": "e" * 64},
        ),
        (
            "post_tool_call",
            {
                "tool_call_ordinal": 1,
                "tool_call_id": "call-one",
                "tool_name": "todo",
                "args_sha256": "f" * 64,
                "result_sha256": "1" * 64,
                "reasoning_directive": {"effort": "max"},
                "result_projection": {
                    "receipt_sha256": "2" * 64,
                    "task_prose": "must never leave the trusted collector",
                },
            },
        ),
        (
            "pre_api_request",
            {
                "request_ordinal": 2,
                "reasoning_effort": "max",
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-5.6-sol",
                "api_request_sha256": "8" * 64,
            },
        ),
        (
            "post_api_request",
            {"request_ordinal": 2, "response_payload_sha256": "9" * 64},
        ),
        (
            "post_tool_call",
            {
                "tool_call_ordinal": 2,
                "tool_call_id": "call-proposal",
                "tool_name": "canonical_event_append",
                "args_sha256": "a" * 64,
                "result_sha256": "0" * 64,
                "result_projection": {
                    "event_id": "event-gateway-proposal",
                },
            },
        ),
        (
            "canonical_case_readback",
            {
                "readback_sha256": producers._sha256_json(proposal_readback),
                "readback": proposal_readback,
            },
        ),
        ("session_end", {"completed": True, "outcome": "completed"}),
    )
    frames: list[Any] = []
    previous = units._OBSERVER_ZERO_CHAIN_SHA256
    for sequence, (event, payload) in enumerate(payloads, start=1):
        correlated = event != "plugin_ready"
        turn_correlated = event not in {
            "plugin_ready",
            "api_session_bound",
            "private_target_probe_ready",
            "private_target_probe_result",
        }
        frame = {
            "schema": "muncho-canary-evidence-frame.v1",
            "sequence": sequence,
            "event": event,
            "release_sha": REVISION,
            "release_sha256": "4" * 64,
            "canary_run_id": "source-run-one",
            "case_id": "case:one",
            "fixture_sha256": source_fixture_sha256,
            "collector_service_identity_sha256": collector_service_sha256,
            "discord_edge_service_identity_sha256": edge_service_sha256,
            "session_id": "session-one" if correlated else None,
            "turn_id": "turn-one" if turn_correlated else None,
            "observed_at_unix_ms": 2_000_000_000_000 + sequence,
            "payload": payload,
        }
        frame_sha256 = producers._sha256_json(frame)
        chain_head = producers._sha256_json(
            {
                "schema": units._OBSERVER_FRAME_CHAIN_SCHEMA,
                "previous_sha256": previous,
                "sequence": sequence,
                "frame_sha256": frame_sha256,
                "peer_pid": peer.pid,
                "peer_start_time_ticks": peer.start_time_ticks,
            }
        )
        frames.append(
            SimpleNamespace(
                value=frame,
                sha256=frame_sha256,
                chain_head_sha256=chain_head,
                peer=peer,
            )
        )
        previous = chain_head
    readiness_unsigned = {
        "schema": "collector-readiness.v1",
        "service_identity_sha256": collector_service_sha256,
        "edge_service_identity_sha256": edge_service_sha256,
        "collector_socket": {"path": "/run/collector.sock", "inode": 1},
    }
    collector_readiness = {
        **readiness_unsigned,
        "receipt_sha256": producers._sha256_json(readiness_unsigned),
    }
    runtime_source = {
        "gateway_process_identity_sha256": "5" * 64,
        "discord_connector_readiness_sha256": "6" * 64,
        "connector_bot_user_id": "1504852355588423803",
        "connector_bot_user_id_provenance": "discord_gateway_ready_user_id",
    }
    observer_endpoint = foundation["endpoints"]["gateway_observer"]
    observer_readiness_unsigned = {
        "schema": producers.PRODUCER_ENDPOINT_READINESS_SCHEMA,
        "role": "gateway_observer",
        "foundation_sha256": producers.producer_foundation_sha256(foundation),
        "release_sha": REVISION,
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "service_unit": observer_endpoint["service_unit"],
        "service_identity_sha256": observer_endpoint[
            "service_identity_sha256"
        ],
        "main_pid": os.getpid(),
        "uid": observer_endpoint["uid"],
        "gid": observer_endpoint["gid"],
    }
    observer_readiness = {
        **observer_readiness_unsigned,
        "readiness_sha256": producers._sha256_json(
            observer_readiness_unsigned
        ),
    }
    fleet_readiness_unsigned = {
        "schema": producers.PRODUCER_ACTIVATION_SCHEMA,
        "foundation_sha256": producers.producer_foundation_sha256(foundation),
        "release_sha": REVISION,
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "fixture_sha256": capability_fixture_sha256,
        "run_id": "run-one",
        "endpoint_readiness": {"gateway_observer": observer_readiness},
    }
    producer_readiness = {
        **fleet_readiness_unsigned,
        "readiness_sha256": producers._sha256_json(fleet_readiness_unsigned),
    }
    restart_unsigned = {
        "schema": "muncho-production-capability-worker-restart.v1",
        "service_unit": "muncho-isolated-worker.service",
        "command_sha256": "7" * 64,
        "completed_at_unix_ms": 2_000_000_000_020,
    }
    restart = {
        **restart_unsigned,
        "receipt_sha256": producers._sha256_json(restart_unsigned),
    }
    conversation = SimpleNamespace(
        session_id="session-one",
        session_create_request_id="request-session-one",
        chat_stream_request_id="request-chat-one",
        api_run_id="api-run-one",
        api_message_id="api-message-one",
        events=(("assistant.completed", {"content": "secret response"}),),
        assistant_completed={"content": "secret response"},
        run_completed={"status": "completed"},
        observed_at_unix_ms=2_000_000_000_010,
        completed_at_unix_ms=2_000_000_000_011,
    )
    terminal = units.build_api_terminal_event_identity(conversation)
    projection = units.build_gateway_observer_source_projection(
        foundation=foundation,
        fixture_sha256=capability_fixture_sha256,
        run_id="run-one",
        producer_readiness=producer_readiness,
        collector_readiness=collector_readiness,
        runtime_source_identity=runtime_source,
        frames=frames,
        worker_restart_receipt=restart,
        api_terminal_event_identity=terminal,
        observed_at_unix_ms=2_000_000_000_030,
    )

    encoded = json.dumps(projection, sort_keys=True)
    assert "must never leave the trusted collector" not in encoded
    assert "secret response" not in encoded
    assert "result_projection" not in encoded
    assert "failure_observed" not in encoded
    assert "model_retained_tool_control" not in encoded
    assert '"native_evidence":' not in encoded
    assert projection["native_evidence_bindings_recorded"] is False
    assert projection["semantic_task_prose_recorded"] is False
    assert projection["secret_material_recorded"] is False
    assert projection["secret_digest_recorded"] is False
    assert projection["frame_chain_head_sha256"] == previous
    assert projection["source_canary_run_id"] == "source-run-one"
    assert units.validate_gateway_observer_source_projection(
        projection,
        release_sha=REVISION,
        capability_plan_sha256=foundation["capability_plan_sha256"],
        full_canary_plan_sha256=foundation["full_canary_plan_sha256"],
        fixture_sha256=capability_fixture_sha256,
        run_id="run-one",
    ) == projection

    tampered = copy.deepcopy(projection)
    tampered["frame_records"][4]["chain_head_sha256"] = "0" * 64
    tampered_unsigned = {
        key: value for key, value in tampered.items() if key != "projection_sha256"
    }
    tampered["projection_sha256"] = producers._sha256_json(tampered_unsigned)
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_projection_invalid",
    ):
        units.validate_gateway_observer_source_projection(
            tampered,
            release_sha=REVISION,
            capability_plan_sha256=foundation["capability_plan_sha256"],
            full_canary_plan_sha256=foundation["full_canary_plan_sha256"],
            fixture_sha256=capability_fixture_sha256,
            run_id="run-one",
        )
    forbidden_source_values = {
        "success": True,
        "changed": False,
        "unexpected_change_count": 0,
        "bindings": [],
        "task_prose": "root-authored task claim",
        "job_prose": "root-authored job claim",
        "secret": "must-not-cross-source-boundary",
    }
    for forbidden_name, forbidden_value in forbidden_source_values.items():
        forbidden_projection = copy.deepcopy(projection)
        forbidden_projection["runtime_source_identity"][forbidden_name] = (
            forbidden_value
        )
        forbidden_unsigned = {
            key: value
            for key, value in forbidden_projection.items()
            if key != "projection_sha256"
        }
        forbidden_projection["projection_sha256"] = producers._sha256_json(
            forbidden_unsigned
        )
        with pytest.raises(
            producers.CapabilityProducerError,
            match="gateway_observer_source_projection_invalid",
        ):
            units.validate_gateway_observer_source_projection(
                forbidden_projection,
                release_sha=REVISION,
                capability_plan_sha256=foundation[
                    "capability_plan_sha256"
                ],
                full_canary_plan_sha256=foundation[
                    "full_canary_plan_sha256"
                ],
                fixture_sha256=capability_fixture_sha256,
                run_id="run-one",
            )

    config = producers.ProducerConfig.from_mapping(
        json.loads(
            units._producer_config(
                role="gateway_observer",
                foundation=foundation,
            )
        )
    )
    source_file_sha256 = producers._sha256_bytes(
        producers._canonical_bytes(projection)
    )
    collector = units.GatewayObserverSourceNativeCollector(
        config=config,
        foundation=foundation,
        source_reader=lambda _path, _gid: (projection, source_file_sha256),
    )
    common = {
        "run_id": "run-one",
        "release_sha": REVISION,
        "fixture_sha256": capability_fixture_sha256,
        "observed_at_unix_ms": 2_000_000_000_025,
    }
    runtime_payload = {
        "schema": "muncho-production-capability-runtime-receipt.v1",
        **common,
        "host_identity_sha256": "a" * 64,
        "release_artifact_sha256": "b" * 64,
        "installed_wheel_manifest_sha256": "c" * 64,
        "effective_config_sha256": "d" * 64,
        "tool_inventory_sha256": "e" * 64,
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "model": "gpt-5.6-sol",
        "initial_effort": "high",
        "adaptive_max_effort": "max",
        "max_turns": 90,
        "toolsets": list(units._GATEWAY_OBSERVER_REQUIRED_TOOLSETS),
        "kanban_auxiliary_planning_enabled": False,
        "kanban_auto_decompose": False,
        "kanban_dispatch_in_gateway": False,
        "prompt_cache_stable": True,
        "message_alternation_valid": True,
        "gateway_process_identity_sha256": runtime_source[
            "gateway_process_identity_sha256"
        ],
        "connector_bot_user_id": runtime_source["connector_bot_user_id"],
        "connector_bot_user_id_provenance": runtime_source[
            "connector_bot_user_id_provenance"
        ],
        "connector_readiness_receipt_sha256": runtime_source[
            "discord_connector_readiness_sha256"
        ],
    }
    runtime_bindings = collector.collect(
        slot="runtime", payload=runtime_payload
    )
    assert tuple(item.kind for item in runtime_bindings) == (
        "gateway_runtime_readiness",
        "discord_connector_readiness",
        "routeback_bot_identity",
    )
    assert projection["observer_activation_identity"]["observer_main_pid"] == (
        os.getpid()
    )
    replayed_projection = copy.deepcopy(projection)
    replayed_activation = replayed_projection["observer_activation_identity"]
    replayed_activation["observer_main_pid"] = os.getpid() + 100_000
    replayed_activation_unsigned = {
        key: value
        for key, value in replayed_activation.items()
        if key != "identity_sha256"
    }
    replayed_activation["identity_sha256"] = producers._sha256_json(
        replayed_activation_unsigned
    )
    replayed_projection["slot_membership"]["runtime"][
        "observer_activation_identity_sha256"
    ] = replayed_activation["identity_sha256"]
    replayed_unsigned = {
        key: value
        for key, value in replayed_projection.items()
        if key != "projection_sha256"
    }
    replayed_projection["projection_sha256"] = producers._sha256_json(
        replayed_unsigned
    )
    replayed_file_sha256 = producers._sha256_bytes(
        producers._canonical_bytes(replayed_projection)
    )
    replayed_collector = units.GatewayObserverSourceNativeCollector(
        config=config,
        foundation=foundation,
        source_reader=lambda _path, _gid: (
            replayed_projection,
            replayed_file_sha256,
        ),
    )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_evidence_invalid",
    ):
        replayed_collector.collect(slot="runtime", payload=runtime_payload)

    workspace_payload = {
        "schema": (
            "muncho-production-capability-canonical-task-workspace-gateway.v3"
        ),
        **common,
        "transcript_sha256": terminal["transcript_sha256"],
        **workspace_core,
    }
    workspace_bindings = collector.collect(
        slot="workspace_gateway", payload=workspace_payload
    )
    assert tuple(item.kind for item in workspace_bindings) == (
        "gateway_observer_frame_chain",
        "authenticated_api_terminal_event",
        "isolated_worker_restart_receipt",
    )

    failure_payload = {
        "schema": "muncho-production-capability-failure-gateway.v1",
        **common,
        "transcript_sha256": terminal["transcript_sha256"],
        **failure_core,
    }
    failure_bindings = collector.collect(
        slot="failure_gateway", payload=failure_payload
    )
    assert tuple(item.kind for item in failure_bindings) == (
        "gateway_observer_frame_chain",
        "authenticated_api_terminal_event",
        "failure_probe_receipts",
    )

    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_evidence_invalid",
    ):
        collector.collect(
            slot="workspace_gateway",
            payload={**workspace_payload, "owner_grant_id": "approval-two"},
        )
    swapped_failure_payload = copy.deepcopy(failure_payload)
    swapped_failure_payload["failures"][0][
        "alternative_receipt_sha256"
    ] = failure_digests[3]
    swapped_failure_payload["failures"][1][
        "alternative_receipt_sha256"
    ] = failure_digests[1]
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_evidence_invalid",
    ):
        collector.collect(
            slot="failure_gateway",
            payload=swapped_failure_payload,
        )

    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_evidence_invalid",
    ):
        collector.collect(
            slot="workspace_gateway",
            payload={
                **workspace_payload,
                "first_path_failure_receipt_sha256": "0" * 64,
            },
        )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_evidence_invalid",
    ):
        collector.collect(
            slot="runtime",
            payload={**runtime_payload, "outcome": "root-claimed"},
        )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_projection_invalid",
    ):
        collector.collect(
            slot="runtime",
            payload={**runtime_payload, "run_id": "stale-run"},
        )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_projection_invalid",
    ):
        units.validate_gateway_observer_source_projection(
            projection,
            release_sha=REVISION,
            capability_plan_sha256=foundation["capability_plan_sha256"],
            full_canary_plan_sha256=foundation["full_canary_plan_sha256"],
            fixture_sha256="0" * 64,
            run_id="another-run",
        )

    outcome_projection = copy.deepcopy(projection)
    outcome_projection["runtime_source_identity"]["outcome"] = "root-claimed"
    outcome_unsigned = {
        key: value
        for key, value in outcome_projection.items()
        if key != "projection_sha256"
    }
    outcome_projection["projection_sha256"] = producers._sha256_json(
        outcome_unsigned
    )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="gateway_observer_source_projection_invalid",
    ):
        units.validate_gateway_observer_source_projection(
            outcome_projection,
            release_sha=REVISION,
            capability_plan_sha256=foundation["capability_plan_sha256"],
            full_canary_plan_sha256=foundation["full_canary_plan_sha256"],
            fixture_sha256=capability_fixture_sha256,
            run_id="run-one",
        )


def test_fixed_native_publication_pump_accepts_only_canonical_slot_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProductionReceiptPump(producers.ProductionReceiptPump):
        def __init__(self) -> None:
            self.calls: list[tuple[str, Mapping[str, Any]]] = []

        def produce(
            self,
            *,
            slot: str,
            payload: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            self.calls.append((slot, payload))
            return {"slot": slot}

    receipt_pump = FakeProductionReceiptPump()
    native = units.FixedNativePublicationPump(
        pump=receipt_pump,
        root=tmp_path,
    )
    monkeypatch.setattr(
        native,
        "_wait_payload",
        lambda slot, *, deadline, cancel: {"slot": slot},
    )
    selected = tuple(
        slot
        for slot in producers.PRODUCTION_PRE_CLEANUP_PUMP_SLOTS
        if producers.SLOT_ROLE[slot] != "gateway_observer"
    )
    assert native.pump_slots(
        selected,
        deadline=1.0,
    ) == {slot: {"slot": slot} for slot in selected}
    assert [slot for slot, _payload in receipt_pump.calls] == list(selected)

    invalid_values: tuple[Any, ...] = (
        tuple(reversed(selected)),
        (selected[0], selected[0]),
        ("cleanup",),
        ({"not": "a slot"},),
        (),
    )
    for invalid in invalid_values:
        with pytest.raises(
            producers.CapabilityProducerError,
            match="native_publication_pump_slots_invalid",
        ):
            native.pump_slots(invalid, deadline=1.0)


def test_root_orchestrator_cannot_publish_for_a_nonroot_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(os, "getegid", lambda: 0)
    with pytest.raises(
        producers.CapabilityProducerError,
        match="native_publication_invalid",
    ):
        units.publish_role_native_publication(
            role="canonical_writer",
            slot="workspace_writer",
            payload={
                "run_id": "run-one",
                "release_sha": REVISION,
                "fixture_sha256": "f" * 64,
            },
            bindings=(),
            uid=2202,
            gid=2302,
            root=tmp_path,
        )


def test_cross_role_precreate_can_only_cause_fail_closed_collision(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o3770)
    parent = tmp_path.lstat()
    parent_mode = stat.S_IMODE(parent.st_mode)
    path = tmp_path / producers.SLOT_FILENAME["bitrix_writer"]
    malicious = _canonical(
        {
            "schema": producers.SIGNED_RECEIPT_SCHEMA,
            "authority_role": "canonical_writer",
            "payload": {"forged": True},
        }
    )
    producers._publish_no_replace(
        path,
        malicious,
        uid=parent.st_uid,
        gid=parent.st_gid,
        mode=0o400,
        parent_uid=parent.st_uid,
        parent_gid=parent.st_gid,
        parent_mode=parent_mode,
    )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="artifact_identity_invalid",
    ):
        producers._publish_no_replace(
            path,
            b'{"legitimate":true}',
            uid=parent.st_uid + 1,
            gid=parent.st_gid + 1,
            mode=0o400,
            parent_uid=parent.st_uid,
            parent_gid=parent.st_gid,
            parent_mode=parent_mode,
        )
    assert path.read_bytes() == malicious


@pytest.mark.skipif(
    os.geteuid() != 0 or sys.platform != "linux",
    reason="kernel principal separation proof requires root on Linux",
)
def test_distinct_linux_principals_cannot_replace_cross_role_receipt(
    tmp_path: Path,
) -> None:
    shared_gid = 24001
    first_uid, first_gid = 22001, 23001
    second_uid, second_gid = 22002, 23002
    os.chown(tmp_path, 0, shared_gid)
    tmp_path.chmod(0o3770)
    path = tmp_path / "bitrix-writer.json"
    first_payload = b'{"authority":"business-edge"}'

    def child_publish(uid: int, gid: int, payload: bytes) -> int:
        child = os.fork()
        if child == 0:  # pragma: no branch - child exits directly
            try:
                os.setgroups([shared_gid])
                os.setgid(gid)
                os.setuid(uid)
                producers._publish_no_replace(
                    path,
                    payload,
                    uid=uid,
                    gid=gid,
                    mode=0o400,
                    parent_uid=0,
                    parent_gid=shared_gid,
                    parent_mode=0o3770,
                )
            except BaseException:
                os._exit(23)
            os._exit(0)
        _pid, status = os.waitpid(child, 0)
        return os.waitstatus_to_exitcode(status)

    assert child_publish(first_uid, first_gid, first_payload) == 0
    assert path.lstat().st_uid == first_uid
    assert path.lstat().st_gid == first_gid
    assert child_publish(second_uid, second_gid, b'{"forged":true}') == 23
    assert path.read_bytes() == first_payload
    assert path.lstat().st_uid == first_uid
