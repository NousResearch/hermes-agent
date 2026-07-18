"""Fixed Caddy cutover transaction, recovery, and replay tests."""

from __future__ import annotations

import copy
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_production_cutover as cutover
from scripts.canary import owner_gate_caddy_cutover as caddy
from tests.gateway.test_canonical_writer_production_cutover import (
    Services,
    _cutover_plan,
)


NOW = 1_800_000_000
BRIDGE_REQUEST_ID = "a" * 64
LEGACY_REQUEST_ID = "L" * 32
ORIGINAL = b"""{
    admin localhost:2019
}

auth.lomliev.com {
    request_body {
        max_size 64KB
    }
    reverse_proxy 127.0.0.1:7341 {
        health_uri /readyz
    }
}

unrelated.example.com {
    respond \"ok\" 200
}
"""


def _authority() -> caddy._Authority:
    plan = _cutover_plan(Ed25519PrivateKey.generate(), Services())
    freeze = cutover.FreezePlan.from_mapping(plan.value["freeze_plan"])
    claim = {
        "schema": "muncho-production-cutover-passkey-claim.v1",
        "freeze_plan_sha256": freeze.sha256,
        "freeze_approval_sha256": plan.value["freeze_approval_sha256"],
        "freeze_publication_sha256": "1" * 64,
        "passkey_proof_sha256": "2" * 64,
        "authorization_receipt_sha256": "3" * 64,
        "action_envelope_sha256": "4" * 64,
        "action_payload_sha256": "8" * 64,
        "request_id": BRIDGE_REQUEST_ID,
        "consume_attempt_id": "9" * 64,
        "authority_release_sha": plan.value["release_revision"],
        "execution_window_expires_at_unix": NOW + 3600,
    }
    authority_unsigned = {
        "schema": caddy.AUTHORITY_SCHEMA,
        "release_revision": plan.value["release_revision"],
        "freeze_plan_sha256": freeze.sha256,
        "cutover_plan_sha256": plan.sha256,
        "freeze_approval_sha256": plan.value["freeze_approval_sha256"],
        "owner_subject_sha256": plan.value["owner_subject_sha256"],
        "owner_key_id": plan.value["owner_key_id"],
        "passkey_claim_entry_sha256": "5" * 64,
        "passkey_claim_recorded_at_unix": NOW - 1,
        "passkey_authorization_receipt_sha256": "3" * 64,
        "passkey_action_envelope_sha256": "4" * 64,
        "passkey_request_id": BRIDGE_REQUEST_ID,
        "passkey_consume_attempt_id": "9" * 64,
        "claim_before_any_caddy_write": True,
        "caller_selected_input_accepted": False,
    }
    authority_value = {
        **authority_unsigned,
        "authority_sha256": caddy._sha256(caddy._canonical(authority_unsigned)),
    }
    return caddy._Authority(
        freeze=freeze,
        plan=plan,
        approval_sha256=plan.value["freeze_approval_sha256"],
        claim_entry_sha256="5" * 64,
        claim_recorded_at_unix=NOW - 1,
        claim=claim,
        value=authority_value,
    )


def _store(tmp_path: Path) -> caddy.CaddyTransactionStore:
    return caddy.CaddyTransactionStore(
        tmp_path / "caddy-journal",
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )


class Boundary:
    def __init__(self, raw: bytes = ORIGINAL) -> None:
        self.original = ORIGINAL
        self.raw = raw
        self.generation = 0
        self.replace_calls: list[bytes] = []
        self.reload_modes: list[str] = []
        self.public_modes: list[str] = []
        self.fail_private_reload_once = False
        self.fail_private_public_once = False
        self.fail_bridge_once = False
        self.concurrent_payload_on_replace: bytes | None = None

    def _configs(self) -> caddy._DerivedConfigs:
        return caddy._derive_configs(
            self.original, bridge_request_id=BRIDGE_REQUEST_ID
        )

    def mode(self, raw: bytes | None = None) -> str:
        selected = self.raw if raw is None else raw
        configs = self._configs()
        if selected == configs.original:
            return "legacy"
        if selected == configs.approval_bridge:
            return "approval_bridge"
        if selected == configs.private_v2:
            return "private_v2"
        if selected == configs.maintenance:
            return "maintenance"
        return "unknown"

    def stable_read(self) -> caddy._StableConfig:
        return caddy._StableConfig(
            self.raw,
            (self.generation, len(self.raw), hash(self.raw)),
        )

    def validate_payload(self, payload: bytes, *, mode: str) -> Mapping[str, Any]:
        assert self.mode(payload) == mode
        projection = {
            "mode": mode,
            **(
                {"bridge_request_id": BRIDGE_REQUEST_ID}
                if mode == "approval_bridge"
                else {}
            ),
            "projection_sha256": caddy._sha256(
                caddy._canonical(
                    {
                        "mode": mode,
                        **(
                            {"bridge_request_id": BRIDGE_REQUEST_ID}
                            if mode == "approval_bridge"
                            else {}
                        ),
                    }
                )
            ),
        }
        return projection

    def replace(
        self,
        payload: bytes,
        *,
        expected: caddy._StableConfig,
    ) -> None:
        assert self.mode(payload) in {
            "legacy",
            "approval_bridge",
            "private_v2",
            "maintenance",
        }
        if self.concurrent_payload_on_replace is not None:
            self.raw = self.concurrent_payload_on_replace
            self.concurrent_payload_on_replace = None
            self.generation += 1
        if self.stable_read() != expected:
            raise caddy.OwnerGateCaddyCutoverError(
                "owner_gate_caddy_compare_and_swap_failed"
            )
        self.raw = payload
        self.generation += 1
        self.replace_calls.append(payload)

    def reload(self) -> None:
        mode = self.mode()
        self.reload_modes.append(mode)
        if mode == "private_v2" and self.fail_private_reload_once:
            self.fail_private_reload_once = False
            raise caddy.OwnerGateCaddyCutoverError(
                "owner_gate_caddy_command_failed"
            )

    def observe(self, *, mode: str) -> Mapping[str, Any]:
        assert self.mode() == mode
        unsigned = {
            "mode": mode,
            **(
                {"bridge_request_id": BRIDGE_REQUEST_ID}
                if mode == "approval_bridge"
                else {}
            ),
        }
        return {
            **unsigned,
            "projection_sha256": caddy._sha256(caddy._canonical(unsigned)),
        }

    def verify_public(self, *, expected_status: int) -> Mapping[str, Any]:
        mode = self.mode()
        self.public_modes.append(mode)
        expected_mode = "private_v2" if expected_status == 200 else "maintenance"
        assert mode == expected_mode
        if mode == "private_v2" and self.fail_private_public_once:
            self.fail_private_public_once = False
            raise caddy.OwnerGateCaddyCutoverError(
                "owner_gate_caddy_public_verify_failed"
            )
        return {"status": expected_status, "body_size": 0, "tls_verified": True}

    def verify_bridge(self, *, request_id: str) -> Mapping[str, Any]:
        assert request_id == BRIDGE_REQUEST_ID
        assert self.mode() == "approval_bridge"
        if self.fail_bridge_once:
            self.fail_bridge_once = False
            raise caddy.OwnerGateCaddyCutoverError(
                "owner_gate_caddy_bridge_verify_failed"
            )
        unsigned = {"request_id": request_id, "verified": True}
        return {
            **unsigned,
            "projection_sha256": caddy._sha256(caddy._canonical(unsigned)),
        }


def _bridge_document(authority: caddy._Authority) -> Mapping[str, Any]:
    unsigned = {
        "schema": caddy.BRIDGE_INPUT_SCHEMA,
        "release_revision": authority.plan.value["release_revision"],
        "freeze_plan_sha256": authority.freeze.sha256,
        "freeze_approval_sha256": authority.approval_sha256,
        "freeze_publication_sha256": authority.claim[
            "freeze_publication_sha256"
        ],
        "v2_request_id": BRIDGE_REQUEST_ID,
        "v2_transaction_id": "a" * 64,
        "v2_approval_url_sha256": caddy._sha256(
            f"https://{caddy.PUBLIC_HOST}/approve/{BRIDGE_REQUEST_ID}".encode()
        ),
        "v2_action_payload_sha256": authority.claim[
            "action_payload_sha256"
        ],
    }
    return {
        **unsigned,
        "document_sha256": caddy._sha256(caddy._canonical(unsigned)),
    }


def _seed_bridge(
    authority: caddy._Authority,
    boundary: Boundary,
    store: caddy.CaddyTransactionStore,
) -> None:
    if caddy._last(store.load(authority.freeze.sha256), "bridge_activated"):
        return
    foundation = caddy.validate_bridge_bootstrap_input(
        _bridge_document(authority)
    )
    configs, template_sha256, action = caddy._bridge_configs_and_action(
        foundation, ORIGINAL
    )
    for name, payload in (
        ("original.Caddyfile", configs.original),
        ("approval-bridge.Caddyfile", configs.approval_bridge),
        ("private-v2.Caddyfile", configs.private_v2),
        ("maintenance.Caddyfile", configs.maintenance),
    ):
        store.install_artifact(authority.freeze.sha256, name, payload)
    requested_unsigned = {
        "schema": caddy.BRIDGE_REQUEST_SCHEMA,
        "release_revision": foundation.release_revision,
        "freeze_plan_sha256": foundation.freeze_plan_sha256,
        "freeze_approval_sha256": foundation.freeze_approval_sha256,
        "freeze_publication_sha256": foundation.freeze_publication_sha256,
        "v2_request_id": foundation.v2_request_id,
        "v2_transaction_id": foundation.v2_transaction_id,
        "v2_approval_url_sha256": foundation.v2_approval_url_sha256,
        "v2_action_payload_sha256": foundation.v2_action_payload_sha256,
        "bootstrap_input_sha256": foundation.document_sha256,
        "legacy_passkey_request_id": LEGACY_REQUEST_ID,
        "legacy_passkey_request_sha256": "b" * 64,
        "legacy_approval_url": (
            f"https://{caddy.PUBLIC_HOST}/approve/{LEGACY_REQUEST_ID}"
        ),
        "bridge_action_sha256": caddy._sha256(caddy._canonical(action)),
        "route_contract_sha256": caddy._sha256(
            caddy._canonical(caddy._bridge_route_contract())
        ),
        "original_caddy_sha256": caddy._sha256(configs.original),
        "approval_bridge_template_sha256": template_sha256,
        "approval_bridge_caddy_sha256": caddy._sha256(
            configs.approval_bridge
        ),
        "default_local_v1_route_preserved": True,
        "production_mutation_performed": False,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "requested_at_unix": NOW - 3,
    }
    requested = {
        **requested_unsigned,
        "receipt_sha256": caddy._sha256(caddy._canonical(requested_unsigned)),
    }
    store.append(
        authority.freeze.sha256,
        "bridge_authorization_requested",
        requested,
        NOW - 3,
    )
    boundary.raw = configs.approval_bridge
    boundary.generation += 1
    projection = boundary.observe(mode="approval_bridge")
    bridge_unsigned = {
        "schema": caddy.BRIDGE_RECEIPT_SCHEMA,
        "release_revision": foundation.release_revision,
        "freeze_plan_sha256": foundation.freeze_plan_sha256,
        "freeze_approval_sha256": foundation.freeze_approval_sha256,
        "freeze_publication_sha256": foundation.freeze_publication_sha256,
        "v2_request_id": foundation.v2_request_id,
        "v2_transaction_id": foundation.v2_transaction_id,
        "v2_approval_url_sha256": foundation.v2_approval_url_sha256,
        "v2_action_payload_sha256": foundation.v2_action_payload_sha256,
        "bootstrap_input_sha256": foundation.document_sha256,
        "bridge_request_receipt_sha256": requested["receipt_sha256"],
        "legacy_passkey_request_id": LEGACY_REQUEST_ID,
        "legacy_passkey_request_sha256": "b" * 64,
        "legacy_passkey_grant_id": "grant_abcdefghijklmnop",
        "legacy_passkey_grant_sha256": "c" * 64,
        "legacy_passkey_consumed_grant_sha256": "d" * 64,
        "legacy_passkey_consume_entry_sha256": "e" * 64,
        "legacy_service_active_before_sha256": "f" * 64,
        "legacy_service_inactive_sha256": "0" * 64,
        "legacy_service_active_after_sha256": "1" * 64,
        "legacy_service_local_health_sha256": "2" * 64,
        "bridge_action_sha256": requested["bridge_action_sha256"],
        "route_contract_sha256": requested["route_contract_sha256"],
        "original_caddy_sha256": requested["original_caddy_sha256"],
        "approval_bridge_caddy_sha256": requested[
            "approval_bridge_caddy_sha256"
        ],
        "active_route_projection_sha256": projection["projection_sha256"],
        "default_local_v1_route_preserved": True,
        "exact_v2_approval_routes_only": True,
        "caddy_validated": True,
        "caddy_reloaded": True,
        "caddy_readback_verified": True,
        "rollback_mode": "pre_migration_exact_bytes",
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "activated_at_unix": NOW - 2,
    }
    bridge = {
        **bridge_unsigned,
        "receipt_sha256": caddy._sha256(caddy._canonical(bridge_unsigned)),
    }
    store.append(
        authority.freeze.sha256,
        "bridge_activated",
        bridge,
        NOW - 2,
    )


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(caddy._canonical(value))
    path.chmod(0o600)


def _legacy_request(
    action: Mapping[str, Any],
    *,
    request_id: str = LEGACY_REQUEST_ID,
) -> Mapping[str, Any]:
    created = NOW - 10
    expires = NOW + 600
    return {
        "schema": caddy.LEGACY_REQUEST_SCHEMA,
        "request_id": request_id,
        "requester_discord_user_id": caddy.OWNER_DISCORD_USER_ID,
        "approver_discord_user_id": caddy.OWNER_DISCORD_USER_ID,
        "approval_scope": caddy.BRIDGE_APPROVAL_SCOPE,
        "case_id": caddy.BRIDGE_CASE_ID,
        "target_system": caddy.BRIDGE_TARGET_SYSTEM,
        "action_summary": caddy.BRIDGE_ACTION_SUMMARY,
        "risk": caddy.BRIDGE_ACTION_RISK,
        "rollback": caddy.BRIDGE_ACTION_ROLLBACK,
        "action_hash": caddy._sha256(caddy._canonical(action)),
        "action_payload": copy.deepcopy(dict(action)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)),
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires)),
        "expires_at_ts": expires,
        "approved_methods": ["passkey"],
        "approver_label": "Emil",
    }


def _legacy_grant(
    request: Mapping[str, Any],
    *,
    method: str = "passkey",
) -> Mapping[str, Any]:
    granted = NOW - 1
    return {
        "schema": caddy.LEGACY_GRANT_SCHEMA,
        "grant_id": "grant_abcdefghijklmnop",
        "request_id": request["request_id"],
        "approved_by_discord_user_id": caddy.OWNER_DISCORD_USER_ID,
        "approval_scope": caddy.BRIDGE_APPROVAL_SCOPE,
        "case_id": caddy.BRIDGE_CASE_ID,
        "action_hash": request["action_hash"],
        "granted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(granted)),
        "expires_at": request["expires_at"],
        "expires_at_ts": request["expires_at_ts"],
        "method": method,
        "single_use": True,
        "used_at": None,
        "used_at_ts": None,
        "approver_label": request["approver_label"],
        "credential_id_hash": caddy.LEGACY_CREDENTIAL_ID_SHA256,
        "credential_sign_count": 7,
        "credential_backed_up": True,
    }


class RequestBoundary:
    def __init__(self, requests_root: Path, *, must_not_run: bool = False) -> None:
        self.requests_root = requests_root
        self.must_not_run = must_not_run
        self.calls = 0

    def create_bridge_request(
        self, *, action: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        assert not self.must_not_run
        self.calls += 1
        request = _legacy_request(action)
        _write_private_json(
            self.requests_root / f"{LEGACY_REQUEST_ID}.json", request
        )
        return {
            "request_id": LEGACY_REQUEST_ID,
            "action_hash": request["action_hash"],
            "approval_url": (
                f"https://{caddy.PUBLIC_HOST}/approve/{LEGACY_REQUEST_ID}"
            ),
        }


class ServiceBoundary:
    def __init__(self) -> None:
        self.active = True
        self.generation = 1
        self.events: list[str] = []

    @staticmethod
    def _value(state: str, generation: int) -> Mapping[str, Any]:
        unsigned = {"state": state, "generation": generation}
        return {
            **unsigned,
            "projection_sha256": caddy._sha256(caddy._canonical(unsigned)),
        }

    def observe_active(self) -> Mapping[str, Any]:
        assert self.active
        return self._value("active", self.generation)

    def stop_exact(self, expected: Mapping[str, Any]) -> Mapping[str, Any]:
        assert expected == self.observe_active()
        self.events.append("stop")
        self.active = False
        return self._value("inactive", self.generation)

    def start_exact(self, expected: Mapping[str, Any]) -> Mapping[str, Any]:
        assert not self.active
        assert expected["state"] == "active"
        self.events.append("start")
        self.generation += 1
        self.active = True
        return self.observe_active()

    def verify_local_v1(self) -> Mapping[str, Any]:
        assert self.active
        self.events.append("health")
        unsigned = {"state": "healthy", "generation": self.generation}
        return {
            **unsigned,
            "projection_sha256": caddy._sha256(caddy._canonical(unsigned)),
        }


def _bootstrap_roots(tmp_path: Path) -> tuple[Path, Path]:
    requests = tmp_path / "step_up_requests"
    grants = tmp_path / "step_up_verifications"
    requests.mkdir(mode=0o700)
    grants.mkdir(mode=0o700)
    return requests, grants


def _prepare(
    authority: caddy._Authority,
    boundary: Boundary,
    store: caddy.CaddyTransactionStore,
) -> Mapping[str, Any]:
    _seed_bridge(authority, boundary, store)
    return caddy.prepare_cutover(
        authority,
        boundary=boundary,
        store=store,
        now_unix=NOW,
    )


def _committed_lineage(
    _authority: caddy._Authority,
    *,
    journal: cutover.CutoverJournal,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    del journal
    return {"receipt_sha256": "6" * 64}, {"receipt_sha256": "7" * 64}


def _adapted_route(
    nested_routes: list[Mapping[str, Any]],
    *,
    errors: Mapping[str, Any] | None = None,
) -> bytes:
    server: dict[str, Any] = {
        "listen": [":443"],
        "routes": [
            {
                "match": [{"host": [caddy.PUBLIC_HOST]}],
                "handle": [
                    {"handler": "subroute", "routes": nested_routes}
                ],
            }
        ],
    }
    if errors is not None:
        server["errors"] = errors
    return caddy._canonical(
        {"apps": {"http": {"servers": {"secure": server}}}}
    )


def _proxy_handler(dial: str) -> Mapping[str, Any]:
    return {"handler": "reverse_proxy", "upstreams": [{"dial": dial}]}


def _maintenance_handler() -> Mapping[str, Any]:
    return {
        "handler": "static_response",
        "body": "Service temporarily unavailable",
        "status_code": 503,
    }


def test_route_derivation_changes_only_exact_upstream_and_builds_maintenance() -> None:
    configs = caddy._derive_configs(ORIGINAL)

    assert configs.approval_bridge.count(
        caddy.PRIVATE_V2_UPSTREAM.encode()
    ) == 2
    assert b"method GET" in configs.approval_bridge
    assert b"method POST" in configs.approval_bridge
    assert b"/approve/MUNCHO_V2_APPROVAL_REQUEST_ID/view" in (
        configs.approval_bridge
    )
    assert b"/approve/MUNCHO_V2_APPROVAL_REQUEST_ID/verify" in (
        configs.approval_bridge
    )
    assert b"/healthz" not in configs.approval_bridge
    assert b"/readyz" in configs.approval_bridge  # existing local health option only
    assert configs.approval_bridge.endswith(
        ORIGINAL[ORIGINAL.index(b"reverse_proxy 127.0.0.1:7341") :]
    )
    assert configs.private_v2.count(caddy.PRIVATE_V2_UPSTREAM.encode()) == 1
    assert b"127.0.0.1:7341" not in configs.private_v2
    assert configs.private_v2.replace(
        caddy.PRIVATE_V2_UPSTREAM.encode(), b"127.0.0.1:7341"
    ) == ORIGINAL
    assert b"reverse_proxy" not in configs.maintenance
    assert caddy.MAINTENANCE_RESPONSE in configs.maintenance
    assert b'unrelated.example.com {\n    respond "ok" 200' in configs.maintenance


@pytest.mark.parametrize(
    "raw",
    (
        ORIGINAL.replace(b"auth.lomliev.com", b"*.lomliev.com"),
        ORIGINAL.replace(
            b"reverse_proxy 127.0.0.1:7341",
            b"reverse_proxy 127.0.0.1:7341 127.0.0.1:7342",
        ),
        ORIGINAL.replace(b"127.0.0.1:7341", b"192.0.2.1:7341"),
        ORIGINAL.replace(
            b"unrelated.example.com {",
            b"auth.lomliev.com {",
        ),
    ),
)
def test_route_derivation_rejects_ambiguous_or_nonlocal_source(raw: bytes) -> None:
    with pytest.raises(caddy.OwnerGateCaddyCutoverError):
        caddy._derive_configs(raw)


@pytest.mark.parametrize(
    ("mode", "routes", "errors"),
    (
        (
            "private_v2",
            [
                {"handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)]},
                {"handle": [{"handler": "file_server"}]},
            ],
            None,
        ),
        (
            "private_v2",
            [
                {"handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)]},
                {"handle": [_maintenance_handler()]},
            ],
            None,
        ),
        (
            "maintenance",
            [
                {"handle": [_maintenance_handler()]},
                {"handle": [{"handler": "file_server"}]},
            ],
            None,
        ),
        (
            "maintenance",
            [
                {
                    "handle": [
                        {
                            **_maintenance_handler(),
                            "body": "different maintenance body",
                        }
                    ]
                }
            ],
            None,
        ),
        (
            "private_v2",
            [{"handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)]}],
            {"routes": [{"handle": [{"handler": "file_server"}]}]},
        ),
    ),
)
def test_adapted_route_rejects_alternate_terminal_or_error_route(
    mode: str,
    routes: list[Mapping[str, Any]],
    errors: Mapping[str, Any] | None,
) -> None:
    with pytest.raises(
        caddy.OwnerGateCaddyCutoverError,
        match="adapted_route_invalid",
    ):
        caddy._effective_route_projection(
            _adapted_route(routes, errors=errors),
            mode=mode,
        )


def test_adapted_route_accepts_only_one_mode_specific_terminal() -> None:
    private = caddy._effective_route_projection(
        _adapted_route(
            [
                {"handle": [{"handler": "request_body"}]},
                {"handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)]},
            ]
        ),
        mode="private_v2",
    )
    maintenance = caddy._effective_route_projection(
        _adapted_route([{"handle": [_maintenance_handler()]}]),
        mode="maintenance",
    )

    assert private["private_v2_upstream_active"] is True
    assert maintenance["maintenance_active"] is True


def test_adapted_bridge_accepts_only_exact_method_and_path_routes() -> None:
    bridge = caddy._effective_route_projection(
        _adapted_route(
            [
                {
                    "match": [
                        {
                            "method": ["GET"],
                            "path": list(caddy._bridge_get_paths(BRIDGE_REQUEST_ID)),
                        }
                    ],
                    "handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)],
                },
                {
                    "match": [
                        {
                            "method": ["POST"],
                            "path": list(caddy._bridge_post_paths(BRIDGE_REQUEST_ID)),
                        }
                    ],
                    "handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)],
                },
                {"handle": [_proxy_handler("127.0.0.1:7341")]},
            ]
        ),
        mode="approval_bridge",
    )

    assert bridge["still_on_current_host"] is True
    assert bridge["private_v2_upstream_active"] is False
    assert bridge["bridge_request_id"] == BRIDGE_REQUEST_ID


def test_adapted_bridge_rejects_prefix_or_wrong_method_matcher() -> None:
    raw = _adapted_route(
        [
            {
                "match": [
                    {
                        "method": ["GET", "POST"],
                        "path": ["/approve/*"],
                    }
                ],
                "handle": [_proxy_handler(caddy.PRIVATE_V2_UPSTREAM)],
            },
            {"handle": [_proxy_handler("127.0.0.1:7341")]},
        ]
    )

    with pytest.raises(caddy.OwnerGateCaddyCutoverError):
        caddy._effective_route_projection(raw, mode="approval_bridge")


def test_prepare_is_no_live_mutation_no_clobber_and_replay_stable(
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)

    first = _prepare(authority, boundary, store)
    second = _prepare(authority, boundary, store)

    assert second == first
    assert boundary.raw == boundary._configs().approval_bridge
    assert boundary.replace_calls == []
    assert first["live_config_mutated"] is False
    assert first["passkey_claim_entry_sha256"] == authority.claim_entry_sha256
    assert [entry.value["event"] for entry in store.load(authority.plan.sha256)] == [
        "prepare_intent",
        "prepared",
    ]


def test_prepare_rejects_artifact_clobber_on_replay(tmp_path: Path) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    private_path = (
        store._plan_root(authority.plan.sha256)
        / "private/private-v2.Caddyfile"
    )
    private_path.chmod(0o600)

    with pytest.raises(caddy.OwnerGateCaddyCutoverError):
        _prepare(authority, boundary, store)


def test_commit_requires_migration_and_restores_exact_preimage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    boundary.raw = caddy._derive_configs(ORIGINAL).private_v2
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", lambda *_a, **_k: None)

    with pytest.raises(
        caddy.OwnerGateCaddyCutoverError,
        match="legacy_migration_not_committed",
    ):
        caddy.commit_cutover(
            authority,
            boundary=boundary,
            store=store,
            legacy_journal=object(),
            now_unix=NOW + 1,
        )

    assert boundary.raw == ORIGINAL
    assert boundary.reload_modes == ["legacy"]


def test_activation_intent_without_terminal_never_restores_v1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    monkeypatch.setattr(
        caddy,
        "_legacy_commit_lineage",
        lambda *_a, **_k: ({"receipt_sha256": "6" * 64}, None),
    )

    receipt = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 1,
    )

    assert receipt["schema"] == caddy.MAINTENANCE_OBSERVATION_SCHEMA
    assert receipt["outcome"] == "maintenance_active_forward_recovery_required"
    assert receipt["legacy_terminal_receipt_sha256"] is None
    assert receipt["v1_route_restored"] is False
    assert boundary.mode() == "maintenance"
    assert ORIGINAL not in boundary.replace_calls


def test_commit_private_v2_and_replay_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", _committed_lineage)

    first = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 1,
    )
    second = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 2,
    )

    assert first == second
    assert first["outcome"] == "private_v2_active"
    assert first["v1_route_restored"] is False
    assert boundary.mode() == "private_v2"
    assert boundary.replace_calls.count(caddy._derive_configs(ORIGINAL).private_v2) == 1


def test_concurrent_foreign_edit_fails_cas_and_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    foreign = ORIGINAL.replace(b"max_size 64KB", b"max_size 63KB")
    boundary.concurrent_payload_on_replace = foreign
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", _committed_lineage)

    with pytest.raises(BaseExceptionGroup) as caught:
        caddy.commit_cutover(
            authority,
            boundary=boundary,
            store=store,
            legacy_journal=object(),
            now_unix=NOW + 1,
        )

    assert "compare_and_swap_failed" in repr(caught.value)
    assert "unowned_drift_detected" in repr(caught.value)
    assert boundary.raw == foreign
    assert boundary.replace_calls == []
    assert boundary.reload_modes == []


def test_crash_after_atomic_replace_resumes_without_restoring_v1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    configs = caddy._derive_configs(ORIGINAL)
    boundary.raw = configs.private_v2
    boundary.generation += 1
    store.append(
        authority.plan.sha256,
        "commit_started",
        {
            "authority_sha256": authority.sha256,
            "legacy_activation_commit_intent_receipt_sha256": "6" * 64,
            "legacy_terminal_receipt_sha256": "7" * 64,
            "rollback_mode": "post_migration_maintenance_only",
        },
        NOW + 1,
    )
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", _committed_lineage)

    receipt = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 2,
    )

    assert receipt["outcome"] == "private_v2_active"
    assert boundary.raw == configs.private_v2
    assert configs.original not in boundary.replace_calls


def test_reload_failure_converges_to_maintenance_never_v1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    boundary.fail_private_reload_once = True
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", _committed_lineage)

    receipt = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 1,
    )

    configs = caddy._derive_configs(ORIGINAL)
    assert receipt["outcome"] == "maintenance_active"
    assert receipt["public_status"] == 503
    assert receipt["rollback_mode"] == "post_migration_maintenance_only"
    assert boundary.raw == configs.maintenance
    assert configs.original not in boundary.replace_calls
    assert boundary.reload_modes == ["private_v2", "maintenance"]


def test_public_verify_failure_converges_to_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    boundary.fail_private_public_once = True
    store = _store(tmp_path)
    _prepare(authority, boundary, store)
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", _committed_lineage)

    receipt = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 1,
    )

    assert receipt["outcome"] == "maintenance_active"
    assert boundary.public_modes == ["private_v2", "maintenance"]


def test_terminal_tamper_cannot_change_passkey_or_rollback_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _authority()
    boundary = Boundary()
    store = _store(tmp_path)
    prepared = _prepare(authority, boundary, store)
    monkeypatch.setattr(caddy, "_legacy_commit_lineage", _committed_lineage)
    terminal = caddy.commit_cutover(
        authority,
        boundary=boundary,
        store=store,
        legacy_journal=object(),
        now_unix=NOW + 1,
    )
    changed = copy.deepcopy(terminal)
    changed["v1_route_restored"] = True
    unsigned = {key: item for key, item in changed.items() if key != "receipt_sha256"}
    changed["receipt_sha256"] = caddy._sha256(caddy._canonical(unsigned))

    with pytest.raises(caddy.OwnerGateCaddyCutoverError):
        caddy.validate_terminal_receipt(
            changed,
            plan=authority.plan,
            prepare_receipt=prepared,
        )


def test_bridge_input_requires_exact_lowercase_sha256_request_id() -> None:
    authority = _authority()
    valid = _bridge_document(authority)

    caddy.validate_bridge_bootstrap_input(valid)
    for invalid_id in ("A" * 64, "a" * 32, "_" * 64):
        changed = dict(valid)
        changed["v2_request_id"] = invalid_id
        unsigned = {
            key: item for key, item in changed.items()
            if key != "document_sha256"
        }
        changed["document_sha256"] = caddy._sha256(
            caddy._canonical(unsigned)
        )
        with pytest.raises(
            caddy.OwnerGateCaddyCutoverError,
            match="bridge_input_invalid",
        ):
            caddy.validate_bridge_bootstrap_input(changed)


def test_prepare_bridge_adopts_request_after_lost_helper_response(
    tmp_path: Path,
) -> None:
    authority = _authority()
    document = _bridge_document(authority)
    foundation = caddy.validate_bridge_bootstrap_input(document)
    _configs, _template, action = caddy._bridge_configs_and_action(
        foundation, ORIGINAL
    )
    requests, _grants = _bootstrap_roots(tmp_path)
    existing = _legacy_request(action)
    _write_private_json(
        requests / f"{LEGACY_REQUEST_ID}.json", existing
    )
    boundary = Boundary()
    store = _store(tmp_path)
    request_boundary = RequestBoundary(requests, must_not_run=True)

    first = caddy.prepare_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        request_boundary=request_boundary,
        now_unix=NOW,
        requests_root=requests,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    second = caddy.prepare_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        request_boundary=request_boundary,
        now_unix=NOW + 1,
        requests_root=requests,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    assert first == second
    assert first["legacy_passkey_request_id"] == LEGACY_REQUEST_ID
    assert request_boundary.calls == 0
    assert boundary.raw == ORIGINAL
    assert [
        entry.value["event"]
        for entry in store.load(authority.freeze.sha256)
    ] == ["bridge_request_intent", "bridge_authorization_requested"]


def test_activate_bridge_stops_v1_before_atomic_consume_and_replays(
    tmp_path: Path,
) -> None:
    authority = _authority()
    document = _bridge_document(authority)
    requests, grants = _bootstrap_roots(tmp_path)
    boundary = Boundary()
    store = _store(tmp_path)
    request_boundary = RequestBoundary(requests)
    requested = caddy.prepare_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        request_boundary=request_boundary,
        now_unix=NOW,
        requests_root=requests,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    request = json.loads(
        (requests / f"{LEGACY_REQUEST_ID}.json").read_text()
    )
    _write_private_json(
        grants / f"{LEGACY_REQUEST_ID}.json", _legacy_grant(request)
    )
    service = ServiceBoundary()

    first = caddy.activate_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        service=service,
        now_unix=NOW,
        requests_root=requests,
        grants_root=grants,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        legacy_lock=lambda: nullcontext(),
    )
    second = caddy.activate_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        service=service,
        now_unix=NOW + 1,
        requests_root=requests,
        grants_root=grants,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        legacy_lock=lambda: nullcontext(),
    )

    consumed = json.loads(
        (grants / f"{LEGACY_REQUEST_ID}.json").read_text()
    )
    events = [
        entry.value["event"]
        for entry in store.load(authority.freeze.sha256)
    ]
    assert first == second
    assert first["bridge_request_receipt_sha256"] == requested["receipt_sha256"]
    assert consumed["used_at_ts"] == NOW
    assert boundary.mode() == "approval_bridge"
    assert service.events == ["stop", "start", "health", "health"]
    assert events.index("bridge_intent") < events.index("legacy_service_stopped")
    assert events.index("legacy_service_stopped") < events.index(
        "legacy_grant_consumed"
    )
    assert events.index("legacy_grant_consumed") < events.index(
        "bridge_activated"
    )


def test_activate_bridge_recovers_a_crash_with_v1_already_stopped(
    tmp_path: Path,
) -> None:
    authority = _authority()
    document = _bridge_document(authority)
    requests, grants = _bootstrap_roots(tmp_path)
    boundary = Boundary()
    store = _store(tmp_path)
    requested = caddy.prepare_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        request_boundary=RequestBoundary(requests),
        now_unix=NOW,
        requests_root=requests,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    request = json.loads(
        (requests / f"{LEGACY_REQUEST_ID}.json").read_text()
    )
    _write_private_json(
        grants / f"{LEGACY_REQUEST_ID}.json", _legacy_grant(request)
    )
    service = ServiceBoundary()
    active_before = service.observe_active()
    store.append(
        authority.freeze.sha256,
        "bridge_intent",
        {
            "bootstrap_input_sha256": document["document_sha256"],
            "bridge_request_receipt_sha256": requested["receipt_sha256"],
            "bridge_action_sha256": requested["bridge_action_sha256"],
            "legacy_passkey_request_id": LEGACY_REQUEST_ID,
            "legacy_service_active_before_sha256": active_before[
                "projection_sha256"
            ],
            "legacy_service_active_before": active_before,
            "temporary_service_stop_required": True,
            "exact_preimage_restore_required_before_cutover_intent": True,
        },
        NOW,
    )
    # Model power loss after the durable intent and systemd stop but before
    # the stop receipt or grant consume reached the journal.
    service.active = False

    receipt = caddy.activate_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        service=service,
        now_unix=NOW + 1,
        requests_root=requests,
        grants_root=grants,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        legacy_lock=lambda: nullcontext(),
    )

    assert receipt["schema"] == caddy.BRIDGE_RECEIPT_SCHEMA
    assert boundary.mode() == "approval_bridge"
    assert service.active is True
    assert service.events == ["start", "health", "stop", "start", "health"]


def test_bridge_verify_failure_restores_exact_v1_and_restarts_service(
    tmp_path: Path,
) -> None:
    authority = _authority()
    document = _bridge_document(authority)
    requests, grants = _bootstrap_roots(tmp_path)
    boundary = Boundary()
    boundary.fail_bridge_once = True
    store = _store(tmp_path)
    caddy.prepare_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        request_boundary=RequestBoundary(requests),
        now_unix=NOW,
        requests_root=requests,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    request = json.loads(
        (requests / f"{LEGACY_REQUEST_ID}.json").read_text()
    )
    _write_private_json(
        grants / f"{LEGACY_REQUEST_ID}.json", _legacy_grant(request)
    )
    service = ServiceBoundary()

    with pytest.raises(
        caddy.OwnerGateCaddyCutoverError,
        match="bridge_activation_rolled_back",
    ):
        caddy.activate_bridge_bootstrap(
            document,
            boundary=boundary,
            store=store,
            service=service,
            now_unix=NOW,
            requests_root=requests,
            grants_root=grants,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            legacy_lock=lambda: nullcontext(),
        )

    assert boundary.raw == ORIGINAL
    assert service.active is True
    assert service.events == ["stop", "start", "health", "health"]
    assert caddy._last(
        store.load(authority.freeze.sha256), "bridge_exact_restore"
    ) is not None


def test_activate_bridge_rejects_non_passkey_grant_before_caddy_write(
    tmp_path: Path,
) -> None:
    authority = _authority()
    document = _bridge_document(authority)
    requests, grants = _bootstrap_roots(tmp_path)
    boundary = Boundary()
    store = _store(tmp_path)
    caddy.prepare_bridge_bootstrap(
        document,
        boundary=boundary,
        store=store,
        request_boundary=RequestBoundary(requests),
        now_unix=NOW,
        requests_root=requests,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    request = json.loads(
        (requests / f"{LEGACY_REQUEST_ID}.json").read_text()
    )
    _write_private_json(
        grants / f"{LEGACY_REQUEST_ID}.json",
        _legacy_grant(request, method="totp"),
    )
    service = ServiceBoundary()

    with pytest.raises(
        caddy.OwnerGateCaddyCutoverError,
        match="legacy_passkey_grant_invalid",
    ):
        caddy.activate_bridge_bootstrap(
            document,
            boundary=boundary,
            store=store,
            service=service,
            now_unix=NOW,
            requests_root=requests,
            grants_root=grants,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            legacy_lock=lambda: nullcontext(),
        )

    assert boundary.raw == ORIGINAL
    assert service.events == []


def test_fixed_staged_rejects_any_non_allowlisted_phase() -> None:
    with pytest.raises(caddy.OwnerGateCaddyCutoverError, match="phase_invalid"):
        caddy.execute_fixed_staged(
            "reload-arbitrary-path",
            now_unix=NOW,
            boundary_factory=Boundary,
            store_factory=lambda: None,
            legacy_journal_factory=lambda: None,
            lock=nullcontext,
            require_release_runtime=False,
        )


def test_parser_exposes_exact_four_fixed_phases() -> None:
    parser = caddy._parser()
    phase_action = next(
        action
        for action in parser._actions
        if action.dest == "phase"
    )
    phases = ("prepare-bridge", "activate-bridge", "prepare", "commit")

    assert tuple(phase_action.choices) == phases
    assert tuple(parser.parse_args([phase]).phase for phase in phases) == phases
