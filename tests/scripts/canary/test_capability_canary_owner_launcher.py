"""Owner-boundary tests for capability-canary secret transport."""

from __future__ import annotations

import base64
import io
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.canary.capability_canary_owner_launcher as launcher


def _jwt(expiry: int) -> str:
    encode = lambda value: (
        base64
        .urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    return f"{encode({'alg': 'none'})}.{encode({'exp': expiry})}.sig"


def _plan_inputs() -> dict[str, object]:
    terminal = _full_canary_terminal_receipt()
    unsigned = {
        "schema": launcher.CAPABILITY_PLAN_INPUTS_SCHEMA,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "identities": dict(launcher.CAPABILITY_PLANNED_IDENTITIES),
        "discord": {
            "connector_bot_user_id": "1600000000000000001",
            "routeback_bot_user_id": "1600000000000000002",
            "allowed_guild_ids": ["1282725267068157972"],
            "allowed_channel_ids": [launcher.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID],
            "allowed_user_ids": ["1279454038731264061"],
        },
        "artifacts": {
            "browser_node_sha256": "1" * 64,
            "browser_wrapper_sha256": "2" * 64,
            "browser_native_sha256": "3" * 64,
            "browser_executable_sha256": "4" * 64,
            "agent_browser_config_sha256": "5" * 64,
            "worker_bwrap_sha256": "6" * 64,
            "worker_shell_sha256": "7" * 64,
            "runtime_dependency_manifest_sha256": "8" * 64,
            "bitrix_operational_edge_asset_manifest_sha256": "9" * 64,
            "bitrix_operational_edge_rendered_unit_sha256": "a" * 64,
            "bitrix_operational_edge_rendered_config_sha256": "b" * 64,
            "bitrix_operational_edge_rendered_trust_sha256": "c" * 64,
            "bitrix_operational_edge_identity_bootstrap_receipt_sha256": "d" * 64,
            "bitrix_operational_edge_receipt_public_key_id": "e" * 64,
            "bitrix_operational_edge_key_bootstrap_receipt_sha256": "f" * 64,
        },
    }
    return {
        **unsigned,
        "inputs_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _foundation_inputs() -> dict[str, object]:
    terminal = _full_canary_terminal_receipt()
    unsigned = {
        "schema": launcher.CAPABILITY_BITRIX_FOUNDATION_INPUTS_SCHEMA,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_gid": 2211,
        "business_edge_uid": 2104,
        "release_artifact_sha256": "8" * 64,
        "asset_manifest_sha256": "9" * 64,
    }
    return {
        **unsigned,
        "inputs_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _full_canary_terminal_receipt() -> dict[str, object]:
    unsigned = {
        "schema": launcher.FULL_CANARY_TERMINAL_RECEIPT_SCHEMA,
        "ok": True,
        "state": "verified_stopped_and_credentials_retired",
        "release_sha": "a" * 40,
        "coordinator_input_sha256": "1" * 64,
        "full_canary_plan_sha256": "2" * 64,
        "owner_approval_sha256": "3" * 64,
        "phase_b_readiness_anchor_sha256": "4" * 64,
        "api_session_key_sha256": "5" * 64,
        "fixture_sha256": "6" * 64,
        "discord_token_install_receipt_sha256": "7" * 64,
        "coordinator_receipt_sha256": "8" * 64,
        "live_driver_receipt_sha256": "9" * 64,
        "services_stopped": True,
        "discord_token_retired": True,
        "temporary_admin_created": False,
        "bootstrap_credential_created": False,
        "completed_at_unix": 1_900_000_000,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _producer_bootstrap_receipt() -> dict[str, object]:
    terminal = _full_canary_terminal_receipt()
    unsigned = {
        "schema": launcher.CAPABILITY_PRODUCER_BOOTSTRAP_RECEIPT_SCHEMA,
        "revision": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "2" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "service_identity_foundation_receipt_sha256": "0" * 64,
        "producer_identity_foundation_receipt_sha256": "f" * 64,
        "preparation_sha256": "c" * 64,
        "foundation_sha256": "d" * 64,
        "install_receipt_sha256": "e" * 64,
        "unit_bundle_manifest_sha256": "f" * 64,
        "preflight_ready": True,
        "private_key_loaded_by_launcher": False,
        "secret_material_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _service_identity_foundation_receipt(
    terminal: dict[str, object],
) -> dict[str, object]:
    def observation(
        role: str,
        user: str,
        group: str,
        uid: int,
        gid: int,
    ) -> dict[str, object]:
        unsigned_observation = {
            "schema": launcher.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA,
            "plan_sha256": "b" * 64,
            "role": role,
            "state": "present_exact",
            "user": user,
            "group": group,
            "uid": uid,
            "gid": gid,
            "home": "/nonexistent",
            "shell": "/usr/sbin/nologin",
            "group_members": [],
            "supplementary_group_ids": [gid],
            "create_only_eligible": True,
            "secret_material_recorded": False,
        }
        return {
            **unsigned_observation,
            "receipt_sha256": __import__("hashlib")
            .sha256(launcher._canonical_bytes(unsigned_observation))
            .hexdigest(),
        }

    observations = {
        "mac_ops": observation(
            "mac_ops", "muncho-mac-ops", "muncho-mac-ops", 2104, 2205
        ),
        "connector": observation(
            "connector", "muncho-connector", "muncho-connector", 2105, 2206
        ),
    }
    unsigned = {
        "schema": "muncho-production-capability-service-identity-foundation.v1",
        "operation": "create_only_service_principals",
        "revision": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "plan_publication_receipt_sha256": "e" * 64,
        "receipt_path": "/var/lib/muncho-capability-canary/service-identities/foundation.json",
        "before": observations,
        "after": observations,
        "created": [],
        "create_only": True,
        "existing_identities_mutated": False,
        "retained_dormant_on_rollback": True,
        "mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _foundation_context(terminal: dict[str, object]) -> dict[str, object]:
    unsigned = {
        "schema": launcher.CAPABILITY_BITRIX_FOUNDATION_AUTHORING_CONTEXT_SCHEMA,
        "revision": "a" * 40,
        "staged_plan_path": launcher.FULL_CANARY_STAGED_PLAN_PATH,
        "staged_plan_file_sha256": "7" * 64,
        "staged_plan_identity": {
            "device": 1,
            "inode": 2,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "size": 4096,
            "mtime_ns": 3,
        },
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "release_artifact_sha256": "8" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "identities": {
            "service_uid": 2108,
            "service_gid": 2210,
            "socket_client_gid": 2211,
            "business_edge_uid": 2104,
        },
        "identity_observation": {
            "service_user": "muncho-edge-bitrix",
            "service_group": "muncho-edge-bitrix",
            "service_uid": 2108,
            "service_gid": 2210,
            "socket_client_group": "muncho-edge-bitrix-c",
            "socket_client_gid": 2211,
            "state": "absent_create_only_slot",
        },
        "asset_manifest_sha256": "9" * 64,
        "mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _foundation_authoring_receipt(
    tmp_path: Path,
    terminal: dict[str, object],
) -> tuple[Path, Path, dict[str, object]]:
    context = _foundation_context(terminal)
    inputs_unsigned = {
        "schema": launcher.CAPABILITY_BITRIX_FOUNDATION_INPUTS_SCHEMA,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_gid": 2211,
        "business_edge_uid": 2104,
        "release_artifact_sha256": "8" * 64,
        "asset_manifest_sha256": "9" * 64,
    }
    inputs = {
        **inputs_unsigned,
        "inputs_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(inputs_unsigned))
        .hexdigest(),
    }
    foundation_file = tmp_path / "bitrix-foundation-inputs.json"
    foundation_file.write_bytes(launcher._canonical_bytes(inputs))
    foundation_file.chmod(0o600)
    unsigned = {
        "schema": launcher.CAPABILITY_BITRIX_FOUNDATION_AUTHORING_RECEIPT_SCHEMA,
        "revision": "a" * 40,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "release_artifact_sha256": "8" * 64,
        "foundation_authoring_context": context,
        "foundation_authoring_context_receipt_sha256": context["receipt_sha256"],
        "foundation_inputs_sha256": inputs["inputs_sha256"],
        "output_file": str(foundation_file.resolve()),
        "output_file_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(inputs))
        .hexdigest(),
        "output_file_mode": "0600",
        "mutation_scope": "local_owner_file_create_only",
        "cloud_mutation_performed": False,
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }
    receipt_file = tmp_path / "foundation-authoring-receipt.json"
    receipt_file.write_bytes(launcher._canonical_bytes(receipt))
    receipt_file.chmod(0o600)
    return foundation_file.resolve(), receipt_file.resolve(), receipt


def _bitrix_foundation_receipt(
    terminal: dict[str, object], authoring: dict[str, object]
) -> dict[str, object]:
    unsigned = {
        "operation": "bootstrap_bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "release_artifact_sha256": "8" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "foundation_authoring_context_receipt_sha256": authoring[
            "foundation_authoring_context_receipt_sha256"
        ],
        "authority_sha256": "a" * 64,
        "owner_subject_sha256": "b" * 64,
        "expires_at_unix": 2_000_000_000,
        "expiry_watchdog": {},
        "identity_bootstrap_receipt_path": "/var/lib/identity.json",
        "identity_bootstrap_receipt_sha256": "c" * 64,
        "key_bootstrap_receipt_path": "/var/lib/key.json",
        "key_bootstrap_receipt_sha256": "d" * 64,
        "receipt_public_key_id": "e" * 64,
        "asset_manifest_sha256": "9" * 64,
        "asset_verification_sha256": "f" * 64,
        "asset_name_to_id": {},
        "rendered_unit_stage_path": "/var/lib/unit",
        "rendered_unit_sha256": "1" * 64,
        "rendered_config_stage_path": "/var/lib/config",
        "rendered_config_sha256": "2" * 64,
        "rendered_trust_path": "/etc/trust",
        "rendered_trust_sha256": "3" * 64,
        "read_peer_uids": [2104],
        "mutation_peer_uid": 2002,
        "private_content_or_digest_recorded": False,
        "schema": launcher.CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA,
        "receipt_path": "/var/lib/foundation.json",
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def _plan_publication_receipt(
    terminal: dict[str, object], *, plan_sha256: str = "b" * 64
) -> dict[str, object]:
    context = {"receipt_sha256": "4" * 64}
    unsigned = {
        "schema": launcher.CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA,
        "operation": "publish_capability_plan",
        "revision": "a" * 40,
        "plan_sha256": plan_sha256,
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "plan_authoring_context": context,
        "plan_authoring_context_receipt_sha256": context["receipt_sha256"],
        "plan_path": "/etc/muncho/capability-canary/runtime-plan.json",
        "plan_file_sha256": "5" * 64,
        "authority_sha256": "6" * 64,
        "owner_subject_sha256": "7" * 64,
        "connector_bot_user_id": "1600000000000000001",
        "routeback_bot_user_id": "1600000000000000002",
        "production_bot_user_id": "1501976597455044801",
        "stopped_service_state_sha256": "8" * 64,
        "prerequisite_evidence_sha256": "9" * 64,
        "receipt_path": "/var/lib/muncho/plan.json",
        "published_at_unix": 1_900_000_000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def test_owner_launcher_bootstraps_with_its_required_stdlib_only_flags():
    script = Path(launcher.__file__).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/var/empty/muncho-canary",
            str(script),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "storage-preflight" in completed.stdout


def test_owner_launcher_and_runtime_share_publication_and_identity_contracts():
    from gateway import canonical_capability_canary_producer_units as units
    from gateway import canonical_capability_canary_runtime as runtime

    assert (
        launcher.CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA
        == runtime.CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA
    )
    assert (
        launcher.CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA
        == runtime.CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA
    )
    assert (
        launcher.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA
        == runtime.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA
        == units.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA
    )
    terminal = _full_canary_terminal_receipt()
    receipt = _plan_publication_receipt(terminal)
    assert receipt["operation"] == "publish_capability_plan"
    assert (
        launcher.validate_plan_publication_receipt(
            receipt,
            revision="a" * 40,
            terminal_receipt=terminal,
        )
        == receipt
    )


@pytest.mark.parametrize(
    "kind,secret",
    (
        ("codex_access_token", None),
        ("mac_ops_gitlab_env", b"GITLAB_TOKEN=opaque\n"),
        ("discord_connector_token", b"opaque.discord-token"),
        ("api_server_control_key", b"opaque-api-control-key"),
        ("discord_routeback_token", b"opaque.routeback-token"),
        (
            "bitrix_operational_edge_webhook",
            b"https://example.bitrix24.eu/rest/1/opaque/",
        ),
    ),
)
def test_stdlib_owner_secret_frame_is_byte_exact_with_runtime(kind, secret):
    from gateway import canonical_capability_canary_runtime as runtime

    issued = 1_900_000_000
    payload = _jwt(issued + 3_600).encode() if kind == "codex_access_token" else secret
    arguments = {
        "kind": kind,
        "secret": payload,
        "plan_sha256": "a" * 64,
        "owner_subject_sha256": "b" * 64,
        "now_unix": issued,
        "ttl_seconds": 900,
        "lease_id": "c" * 32,
    }
    assert launcher.build_secret_lease_frame(**arguments) == (
        runtime.build_secret_lease_frame(**arguments)
    )


@pytest.mark.parametrize("kind", tuple(launcher._SECRET_LEASE_MAGIC_BY_KIND))
def test_all_six_owner_leases_require_reserve_immediately_before_transport(
    monkeypatch, kind
):
    issued = 1_000
    secret = (
        _jwt(issued + 1_000).encode("ascii")
        if kind == "codex_access_token"
        else b"opaque-secret"
    )
    frame = launcher.build_secret_lease_frame(
        kind=kind,
        secret=secret,
        plan_sha256="a" * 64,
        owner_subject_sha256="b" * 64,
        now_unix=issued,
        ttl_seconds=60,
        lease_id="c" * 32,
    )
    calls = []

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            calls.append((revision, action, frame))
            return {"ok": True}

    monkeypatch.setattr(launcher.time, "time", lambda: issued)
    assert launcher._invoke_secret_lease(
        Transport(), "d" * 40, "provision-test", frame
    ) == {"ok": True}
    monkeypatch.setattr(launcher.time, "time", lambda: issued + 1)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="capability_secret_lease_expiry_reserve_insufficient",
    ):
        launcher._invoke_secret_lease(Transport(), "d" * 40, "provision-test", frame)
    assert len(calls) == 1


def test_codex_reader_extracts_access_only_from_private_stable_file(tmp_path):
    path = tmp_path / "auth.json"
    access = _jwt(int(time.time()) + 3_600)
    path.write_text(
        json.dumps({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": access,
                "refresh_token": "must-never-be-leased",
            },
        })
    )
    path.chmod(0o600)

    result = launcher.read_codex_access_token(path)
    assert result == bytearray(access.encode())
    assert b"must-never-be-leased" not in result

    path.chmod(0o644)
    with pytest.raises(Exception, match="capability_owner_secret_source_invalid"):
        launcher.read_codex_access_token(path)


def test_mac_ops_reader_is_bounded_and_does_not_parse_or_echo_secret():
    raw = b"GITLAB_BASE_URL=https://gitlab.example\nGITLAB_TOKEN=opaque\n"
    result = launcher.read_mac_ops_env(io.BytesIO(raw))
    assert result == bytearray(raw)

    with pytest.raises(Exception, match="capability_mac_ops_secret_input_invalid"):
        launcher.read_mac_ops_env(io.BytesIO(b"x" * (64 * 1024 + 1)))

    class Tty(io.BytesIO):
        def isatty(self):
            return True

    with pytest.raises(Exception, match="capability_mac_ops_stdin_is_tty"):
        launcher.read_mac_ops_env(Tty(raw))


def test_discord_connector_reader_accepts_only_one_bounded_printable_token():
    token = b"discord.connector-token_123"
    assert launcher.read_discord_connector_token(io.BytesIO(token)) == bytearray(token)

    for invalid in (
        b"",
        b"token\n",
        b"token with-space",
        b"token\x00suffix",
        b"\xff",
        b"x" * 513,
    ):
        with pytest.raises(
            Exception,
            match="capability_discord_connector_secret_input_invalid",
        ):
            launcher.read_discord_connector_token(io.BytesIO(invalid))


def test_api_control_key_is_generated_in_memory_and_is_printable(monkeypatch):
    monkeypatch.setattr(
        launcher.secrets,
        "token_bytes",
        lambda size: bytes(range(size)),
    )
    key = launcher.generate_api_control_key()
    assert len(key) == 64
    assert key.decode("ascii").isalnum() or b"-" in key or b"_" in key
    assert b"=" not in key


@pytest.mark.parametrize(
    "reader,valid,error",
    (
        (
            launcher.read_discord_routeback_token,
            b"opaque.routeback-token",
            "capability_discord_routeback_secret_input_invalid",
        ),
        (
            launcher.read_bitrix_operational_edge_webhook,
            b"https://example.bitrix24.eu/rest/1/opaque/",
            "capability_bitrix_webhook_input_invalid",
        ),
    ),
)
def test_new_inherited_stdin_secret_readers_are_bounded_non_tty(
    reader,
    valid,
    error,
):
    assert reader(io.BytesIO(valid)) == bytearray(valid)
    with pytest.raises(Exception, match=error):
        reader(io.BytesIO(b"secret with whitespace"))

    class Tty(io.BytesIO):
        def isatty(self):
            return True

    with pytest.raises(Exception, match="stdin_is_tty"):
        reader(Tty(valid))


def test_bitrix_reader_strips_one_safe_terminal_newline_only():
    value = b"https://example.bitrix24.eu/rest/1/opaque/"
    assert launcher.read_bitrix_operational_edge_webhook(
        io.BytesIO(value + b"\n")
    ) == bytearray(value)
    assert launcher.read_bitrix_operational_edge_webhook(
        io.BytesIO(value + b"\r\n")
    ) == bytearray(value)
    with pytest.raises(Exception, match="capability_bitrix_webhook_input_invalid"):
        launcher.read_bitrix_operational_edge_webhook(io.BytesIO(value + b"\n\n"))


def test_transport_exposes_only_the_fixed_packaged_lifecycle_actions():
    revision = "a" * 40
    expected = {
        "contract",
        "storage-preflight",
        "collect-foundation-authoring-context",
        "collect-plan-authoring-context",
        "bootstrap-bitrix-foundation",
        "prepare-producer-foundation",
        "install-producer-foundation",
        "preflight-producer-foundation",
        "wait-api-admission-challenge",
        "stage-api-admission-owner-authority",
        "publish-plan",
        "preflight-stopped",
        "preflight-live",
        "provision-codex",
        "provision-mac-ops",
        "provision-discord-connector",
        "provision-api-control",
        "provision-discord-routeback",
        "provision-bitrix-operational-edge",
        "install-approval",
        "publish-live-fixture",
        "wait-production-observation-marker",
        "stage-production-observation",
        "wait-api-admission-challenge",
        "stage-api-admission-owner-authority",
        "start",
        "run-live",
        "stop",
        "retire-secrets",
    }
    transport = object.__new__(launcher.CapabilityCanaryTransport)

    assert transport._ACTIONS == expected
    for action in expected - {
        "storage-preflight",
        "publish-live-fixture",
        "run-live",
        "prepare-producer-foundation",
        "install-producer-foundation",
        "preflight-producer-foundation",
        "wait-api-admission-challenge",
        "stage-api-admission-owner-authority",
    }:
        command = transport._remote_command(revision, action)
        assert command[-6:] == (
            f"/opt/muncho-canary-releases/{revision}/venv/bin/python",
            "-B",
            "-I",
            "-m",
            launcher.RUNTIME_MODULE,
            action,
        )
        assert command[-1] == action
    assert transport._remote_command(
        revision, "wait-api-admission-challenge"
    )[-2:] == (launcher.PRODUCER_ADMISSION_MODULE, "wait-owner-challenge")
    assert transport._remote_command(
        revision, "stage-api-admission-owner-authority"
    )[-2:] == (launcher.PRODUCER_ADMISSION_MODULE, "stage-owner-authority")
    for action, remote_action in {
        "prepare-producer-foundation": "prepare-foundation",
        "install-producer-foundation": "install-foundation",
        "preflight-producer-foundation": "preflight",
    }.items():
        command = transport._remote_command(revision, action)
        assert command[-6:] == (
            f"/opt/muncho-canary-releases/{revision}/venv/bin/python",
            "-B",
            "-I",
            "-m",
            launcher.PRODUCER_FOUNDATION_MODULE,
            remote_action,
        )

    live = transport._remote_command(revision, "run-live")
    assert live[-6:] == (
        f"/opt/muncho-canary-releases/{revision}/venv/bin/python",
        "-B",
        "-I",
        "-m",
        "gateway.canonical_capability_canary_live_driver",
        "run",
    )

    publication = transport._remote_command(revision, "publish-live-fixture")
    assert publication[-6:] == (
        f"/opt/muncho-canary-releases/{revision}/venv/bin/python",
        "-B",
        "-I",
        "-m",
        launcher.LIVE_DRIVER_MODULE,
        "publish-fixture",
    )

    storage = transport._remote_command(revision, "storage-preflight")
    assert storage[-2:] == (launcher._REMOTE_STORAGE_PREFLIGHT, revision)
    assert storage[-6:-2] == ("/usr/bin/python3", "-B", "-I", "-c")
    assert "/opt/muncho-canary-releases" in storage[-2]
    assert "os.remove" not in storage[-2]
    assert "shutil.rmtree" not in storage[-2]

    for action in (
        "collect-foundation-authoring-context",
        "collect-plan-authoring-context",
    ):
        authoring = transport._remote_command(revision, action)
        assert authoring[-6:] == (
            f"/opt/muncho-canary-releases/{revision}/venv/bin/python",
            "-B",
            "-I",
            "-m",
            launcher.RUNTIME_MODULE,
            action,
        )

    with pytest.raises(Exception, match="capability_canary_command_invalid"):
        transport._remote_command(revision, "restart")
    with pytest.raises(Exception, match="capability_canary_command_invalid"):
        transport._remote_command("not-a-revision", "start")


def test_plan_publication_requires_private_exact_inputs_and_explicit_digest(
    tmp_path,
):
    path = tmp_path / "capability-plan-inputs.json"
    inputs = _plan_inputs()
    path.write_bytes(launcher._canonical_bytes(inputs))
    path.chmod(0o600)

    loaded = launcher.read_plan_publication_inputs(path.resolve())
    context_unsigned = {
        "schema": launcher.CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA,
        "revision": "a" * 40,
        "full_canary_terminal_receipt": loaded["full_canary_terminal_receipt"],
        "plan_inputs": loaded,
        "capability_plan_sha256": "c" * 64,
    }
    context = {
        **context_unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(context_unsigned))
        .hexdigest(),
    }
    authority = launcher.build_plan_publication_authority(
        revision="a" * 40,
        plan_sha256="c" * 64,
        owner_subject_sha256="d" * 64,
        inputs=loaded,
        plan_authoring_context=context,
    )
    assert authority["plan_sha256"] == "c" * 64
    assert authority["inputs"] == {
        key: inputs[key] for key in ("identities", "discord", "artifacts")
    }
    assert (
        authority["full_canary_terminal_receipt"]
        == (inputs["full_canary_terminal_receipt"])
    )
    assert authority["semantic_content_recorded"] is False
    assert "path" not in authority
    assert "opaque-secret-value" not in json.dumps(authority)
    assert authority["secret_material_recorded"] is False
    assert authority["secret_digest_recorded"] is False

    path.chmod(0o644)
    with pytest.raises(
        Exception,
        match="capability_plan_input_source_invalid",
    ):
        launcher.read_plan_publication_inputs(path.resolve())


def test_foundation_authoring_uses_packaged_collector_and_create_only_output(
    tmp_path,
):
    terminal = _full_canary_terminal_receipt()
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_bytes(launcher._canonical_bytes(terminal))
    terminal_path.chmod(0o600)
    output = tmp_path / "foundation-inputs.json"
    context = _foundation_context(terminal)
    captured = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "collect-foundation-authoring-context"
            request = json.loads(frame)
            assert request["full_canary_terminal_receipt"] == terminal
            captured["request"] = request
            return context

    receipt = launcher.author_bitrix_foundation_inputs(
        Transport(),
        revision="a" * 40,
        terminal_receipt_file=terminal_path.resolve(),
        output_file=output.resolve(),
    )
    inputs = json.loads(output.read_bytes())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert inputs["service_uid"] == 2108
    assert inputs["asset_manifest_sha256"] == "9" * 64
    assert receipt["foundation_authoring_context"] == context
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert (
        receipt["receipt_sha256"]
        == __import__("hashlib").sha256(launcher._canonical_bytes(unsigned)).hexdigest()
    )
    with pytest.raises(launcher.OwnerLauncherError, match="output_path_invalid"):
        launcher.author_bitrix_foundation_inputs(
            Transport(),
            revision="a" * 40,
            terminal_receipt_file=terminal_path.resolve(),
            output_file=output.resolve(),
        )


def test_foundation_authoring_rejects_stale_absent_identity_state():
    terminal = _full_canary_terminal_receipt()
    context = _foundation_context(terminal)
    context["identity_observation"]["state"] = "absent"
    unsigned = {key: value for key, value in context.items() if key != "receipt_sha256"}
    context["receipt_sha256"] = (
        __import__("hashlib").sha256(launcher._canonical_bytes(unsigned)).hexdigest()
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="capability_bitrix_foundation_authoring_context_invalid",
    ):
        launcher.validate_bitrix_foundation_authoring_context(
            context,
            revision="a" * 40,
            terminal_receipt=terminal,
        )


def test_plan_authoring_collects_fixed_staged_plan_and_publishes_owner_file(
    tmp_path,
):
    terminal = _full_canary_terminal_receipt()
    terminal_path = tmp_path / "full-terminal.json"
    terminal_path.write_bytes(launcher._canonical_bytes(terminal) + b"\n")
    terminal_path.chmod(0o600)
    output = tmp_path / "capability-plan-inputs.json"
    _foundation_file, foundation_receipt_file, foundation_authoring = (
        _foundation_authoring_receipt(tmp_path, terminal)
    )
    foundation_context = foundation_authoring["foundation_authoring_context"]
    bitrix = _bitrix_foundation_receipt(terminal, foundation_authoring)
    bitrix_file = tmp_path / "bitrix-bootstrap.json"
    bitrix_file.write_bytes(launcher._canonical_bytes(bitrix))
    bitrix_file.chmod(0o600)
    captured = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "collect-plan-authoring-context"
            request = json.loads(frame)
            captured["request"] = request
            inputs = _plan_inputs()
            inputs["full_canary_terminal_receipt"] = terminal
            inputs["full_canary_terminal_receipt_sha256"] = terminal["receipt_sha256"]
            inputs_unsigned = {
                key: value for key, value in inputs.items() if key != "inputs_sha256"
            }
            inputs["inputs_sha256"] = (
                __import__("hashlib")
                .sha256(launcher._canonical_bytes(inputs_unsigned))
                .hexdigest()
            )
            unsigned = {
                "schema": launcher.CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA,
                "revision": revision,
                "staged_plan_path": launcher.FULL_CANARY_STAGED_PLAN_PATH,
                "staged_plan_file_sha256": foundation_context[
                    "staged_plan_file_sha256"
                ],
                "staged_plan_identity": foundation_context["staged_plan_identity"],
                "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
                "full_canary_terminal_receipt": terminal,
                "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
                "original_full_canary_owner_approval_sha256": terminal[
                    "owner_approval_sha256"
                ],
                "foundation_authoring_context": foundation_authoring[
                    "foundation_authoring_context"
                ],
                "foundation_authoring_context_receipt_sha256": (
                    foundation_authoring["foundation_authoring_context_receipt_sha256"]
                ),
                "bitrix_foundation_receipt": bitrix,
                "bitrix_foundation_receipt_sha256": bitrix["receipt_sha256"],
                "host_identity_observations": {
                    "browser": {},
                    "execution": {},
                    "mac_ops": {},
                    "connector": {},
                    "producer": {
                        "schema": (
                            "muncho-production-capability-producer-host-identity.v2"
                        ),
                        "plan_sha256": "a" * 64,
                        "persistent_supplementary_memberships": False,
                    },
                },
                "plan_inputs": inputs,
                "capability_inputs_sha256": inputs["inputs_sha256"],
                "capability_plan_sha256": "a" * 64,
                "mutation_performed": False,
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
                "semantic_content_recorded": False,
            }
            return {
                **unsigned,
                "receipt_sha256": __import__("hashlib")
                .sha256(launcher._canonical_bytes(unsigned))
                .hexdigest(),
            }

    result = launcher.author_capability_plan_inputs(
        Transport(),
        revision="a" * 40,
        terminal_receipt_file=terminal_path.resolve(),
        foundation_authoring_receipt_file=foundation_receipt_file,
        bitrix_foundation_receipt_file=bitrix_file.resolve(),
        output_file=output.resolve(),
        connector_bot_user_id="1600000000000000001",
        routeback_bot_user_id="1600000000000000002",
    )

    assert result["capability_plan_sha256"] == "a" * 64
    assert result["cloud_mutation_performed"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    written = json.loads(output.read_bytes())
    assert written["discord"]["allowed_channel_ids"] == [
        launcher.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID
    ]
    assert (
        captured["request"]["foundation_authoring_context"]
        == (foundation_authoring["foundation_authoring_context"])
    )
    with pytest.raises(Exception, match="output_path_invalid"):
        launcher.author_capability_plan_inputs(
            Transport(),
            revision="a" * 40,
            terminal_receipt_file=terminal_path.resolve(),
            foundation_authoring_receipt_file=foundation_receipt_file,
            bitrix_foundation_receipt_file=bitrix_file.resolve(),
            output_file=output.resolve(),
            connector_bot_user_id="1600000000000000001",
            routeback_bot_user_id="1600000000000000002",
        )


def test_runtime_plan_collector_context_is_accepted_by_owner_launcher(
    monkeypatch,
):
    from gateway import canonical_capability_canary_runtime as runtime
    from tests.gateway.test_canonical_capability_canary_runtime import (
        _bitrix_foundation_receipt as runtime_bitrix_foundation_receipt,
        _foundation_authoring_context as runtime_foundation_context,
        _full_plan as runtime_full_plan,
        _plan as runtime_plan,
        _plan_authoring_context as runtime_plan_context,
    )

    manifest_raw = b"sealed-runtime-dependency-manifest"
    full_plan = runtime_full_plan()
    plan = runtime_plan(
        full_plan,
        runtime_dependency_manifest_sha256=__import__("hashlib")
        .sha256(manifest_raw)
        .hexdigest(),
    )
    terminal = dict(plan.full_canary_terminal_receipt)
    staged_raw = b"sealed-staged-plan"
    staged_identity = {
        "device": 1,
        "inode": 2,
        "uid": 0,
        "gid": 0,
        "mode": "0400",
        "size": len(staged_raw),
        "mtime_ns": 4,
    }
    foundation = runtime_foundation_context(plan)
    foundation["staged_plan_file_sha256"] = (
        __import__("hashlib").sha256(staged_raw).hexdigest()
    )
    foundation["staged_plan_identity"] = staged_identity
    foundation_unsigned = {
        key: value for key, value in foundation.items() if key != "receipt_sha256"
    }
    foundation["receipt_sha256"] = runtime._sha256_json(foundation_unsigned)
    bitrix = runtime_bitrix_foundation_receipt(plan, foundation)
    foundation_receipt = {
        "foundation_authoring_context": foundation,
        "foundation_authoring_context_receipt_sha256": foundation["receipt_sha256"],
    }
    request = launcher.build_plan_authoring_request(
        revision=full_plan.revision,
        terminal_receipt=terminal,
        foundation_authoring_receipt=foundation_receipt,
        bitrix_foundation_receipt=bitrix,
        connector_bot_user_id=plan.connector_bot_user_id,
        routeback_bot_user_id=plan.routeback_bot_user_id,
    )
    runtime_manifest = {
        "agent_browser": {
            "node_sha256": plan.browser_node_sha256,
            "wrapper_sha256": plan.browser_wrapper_sha256,
            "native_sha256": plan.browser_native_sha256,
            "config_sha256": plan.agent_browser_config_sha256,
        },
        "chrome": {"executable_sha256": plan.browser_executable_sha256},
    }
    expected_observations = runtime_plan_context(plan)["host_identity_observations"]
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_read_staged_full_canary_plan",
        lambda: (full_plan, staged_raw, staged_identity),
    )
    monkeypatch.setattr(
        runtime,
        "verify_release_runtime_dependency_manifest",
        lambda release_root, revision: (
            runtime_manifest
            if release_root == Path(full_plan.release["artifact_root"])
            and revision == full_plan.revision
            else pytest.fail("unexpected release")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_read_stable_file",
        lambda path, **_kwargs: (
            (manifest_raw, object())
            if path
            == Path(full_plan.release["artifact_root"])
            / runtime.RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
            else pytest.fail("unexpected mutable path")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_stable_executable_sha256",
        lambda path, _label: {
            runtime.BWRAP_PATH: plan.worker_bwrap_sha256,
            runtime.SHELL_PATH: plan.worker_shell_sha256,
        }[path],
    )
    monkeypatch.setattr(
        runtime,
        "browser_host_identity_receipt",
        lambda *_args, **_kwargs: expected_observations["browser"],
    )
    monkeypatch.setattr(
        runtime,
        "execution_host_identity_receipt",
        lambda *_args, **_kwargs: expected_observations["execution"],
    )
    monkeypatch.setattr(
        runtime,
        "service_host_identity_receipt",
        lambda *_args, role, **_kwargs: expected_observations[role],
    )

    context = runtime.collect_plan_authoring_context(request)
    assert "staged_plan_b64" not in context
    assert "staged_plan_bytes" not in context
    assert (
        launcher.validate_plan_authoring_context(
            context,
            revision=full_plan.revision,
            terminal_receipt=terminal,
            foundation_authoring_receipt=foundation_receipt,
            bitrix_foundation_receipt=bitrix,
        )
        == context
    )


def test_fixture_authoring_signs_fixed_public_target_and_is_create_only(
    tmp_path,
):
    terminal = _full_canary_terminal_receipt()
    producer = _producer_bootstrap_receipt()
    terminal_path = tmp_path / "full-terminal.json"
    producer_path = tmp_path / "producer.json"
    plan_publication_path = tmp_path / "plan-publication.json"
    output = tmp_path / "fixture-authority.json"
    terminal_path.write_bytes(launcher._canonical_bytes(terminal))
    producer_path.write_bytes(launcher._canonical_bytes(producer) + b"\n")
    plan_publication_path.write_bytes(
        launcher._canonical_bytes(_plan_publication_receipt(terminal))
    )
    terminal_path.chmod(0o600)
    producer_path.chmod(0o600)
    plan_publication_path.chmod(0o600)
    signed = {}

    class Authority:
        def to_mapping(self):
            return {"key_id": "1" * 64}

    authority = Authority()

    class Signer:
        def inspect(self):
            return authority

        def sign(self, payload, *, expected_authority):
            assert expected_authority is authority
            signed["payload"] = payload
            return "opaque-sshsig"

    result = launcher.author_live_fixture_authority(
        revision="a" * 40,
        terminal_receipt_file=terminal_path.resolve(),
        producer_receipt_file=producer_path.resolve(),
        plan_publication_receipt_file=plan_publication_path.resolve(),
        output_file=output.resolve(),
        run_id="capability-live-1",
        valid_for_seconds=900,
        now_unix_ms=1_900_000_000_000,
        signer=Signer(),
    )
    value = json.loads(output.read_bytes())

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert value["public_discord_target"] == {
        "target_type": "public_channel",
        "guild_id": launcher.PRODUCTION_CANARY_PUBLIC_GUILD_ID,
        "channel_id": launcher.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID,
    }
    assert value["owner_id"] == launcher.PRODUCTION_OWNER_USER_ID
    assert value["full_canary_terminal_receipt_sha256"] == terminal["receipt_sha256"]
    assert value["valid_until_unix_ms"] - value["valid_from_unix_ms"] == 900_000
    assert value["owner_signature"] == "opaque-sshsig"
    assert signed["payload"] == launcher._canonical_bytes({
        key: item for key, item in value.items() if key != "owner_signature"
    })
    assert result["cloud_mutation_performed"] is False
    assert result["producer_foundation_sha256"] == producer["foundation_sha256"]
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="capability_fixture_authoring_request_invalid",
    ):
        launcher.build_live_fixture_authority(
            producer_receipt=producer,
            plan_publication_receipt=_plan_publication_receipt(terminal),
            signer=Signer(),
            run_id="capability/live-1",
            valid_for_seconds=900,
            now_unix_ms=1_900_000_000_000,
        )


def test_full_canary_terminal_receipt_requires_terminal_stopped_truth():
    receipt = _full_canary_terminal_receipt()
    receipt["services_stopped"] = False
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = (
        __import__("hashlib").sha256(launcher._canonical_bytes(unsigned)).hexdigest()
    )
    with pytest.raises(Exception, match="full_canary_receipt_invalid"):
        launcher.validate_full_canary_terminal_receipt(
            receipt,
            revision="a" * 40,
        )


def test_bitrix_foundation_authority_precedes_and_never_binds_capability_plan(
    tmp_path,
):
    terminal = _full_canary_terminal_receipt()
    terminal_path = tmp_path / "full-terminal.json"
    terminal_path.write_bytes(launcher._canonical_bytes(terminal))
    terminal_path.chmod(0o600)
    path, authoring_path, authoring = _foundation_authoring_receipt(tmp_path, terminal)
    foundation_receipt = _bitrix_foundation_receipt(terminal, authoring)
    captured: dict[str, object] = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "bootstrap-bitrix-foundation"
            captured["frame"] = frame
            captured["during"] = bytes(frame)
            return foundation_receipt

    result = launcher.bootstrap_bitrix_foundation(
        Transport(),
        revision="a" * 40,
        owner_subject_sha256="d" * 64,
        foundation_file=path,
        terminal_receipt_file=terminal_path.resolve(),
        foundation_authoring_receipt_file=authoring_path,
    )
    authority = json.loads(captured["during"])
    assert result == foundation_receipt
    assert authority["full_canary_plan_sha256"] == "2" * 64
    assert authority["release_artifact_sha256"] == "8" * 64
    assert authority["full_canary_terminal_receipt"] == terminal
    assert authority["identities"] == {
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_gid": 2211,
        "business_edge_uid": 2104,
    }
    assert "plan_sha256" not in authority
    assert "capability_plan_sha256" not in authority
    assert captured["frame"] == bytearray(b"\x00" * len(captured["during"]))


def test_owner_bootstraps_signed_producer_foundation_end_to_end(
    monkeypatch,
    tmp_path,
):
    from gateway import canonical_capability_canary_producers as producers
    from tests.gateway.test_canonical_capability_canary_producer_units import (
        _identities,
        _producer_identity_foundation,
        _production_foundation,
    )
    from tests.gateway.test_canonical_capability_canary_producers import (
        _sshsig,
    )

    foundation, context, keys = _production_foundation(tmp_path)
    unsigned_foundation = {
        key: value for key, value in foundation.items() if key != "owner_signature"
    }
    owner = foundation["owner_authority"]
    source = owner["public_key_source"]
    authority_mapping = {
        "public_key_ed25519_hex": owner["public_key_ed25519_hex"],
        "key_id": owner["key_id"],
        "public_key_file_sha256": source["file_sha256"],
        "public_fingerprint": source["fingerprint"],
        "public_key_source": {
            "path": source["path"],
            "file_sha256": source["file_sha256"],
            "device": 7,
            "inode": 11,
            "uid": source["uid"],
            "gid": source["gid"],
            "mode": f"{source['mode']:04o}",
            "size": source["size"],
        },
    }

    class Authority:
        def to_mapping(self):
            return authority_mapping

    authority = Authority()

    class Signer:
        def inspect(self):
            return authority

        def sign(self, payload, *, expected_authority):
            assert expected_authority is authority
            return _sshsig(
                context["owner_private"],
                payload,
                namespace=producers.PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
            )

    terminal = _full_canary_terminal_receipt()
    terminal["full_canary_plan_sha256"] = "c" * 64
    terminal_unsigned = {
        key: value for key, value in terminal.items() if key != "receipt_sha256"
    }
    terminal["receipt_sha256"] = producers._sha256_json(terminal_unsigned)
    service_identity_foundation = _service_identity_foundation_receipt(terminal)
    identity_plan = SimpleNamespace(
        revision="a" * 40,
        sha256="b" * 64,
        full_canary_terminal_receipt_sha256=terminal["receipt_sha256"],
        original_full_canary_owner_approval_sha256=terminal["owner_approval_sha256"],
    )
    producer_identity_foundation = _producer_identity_foundation(
        plan=identity_plan,
        full_plan=SimpleNamespace(sha256="c" * 64),
    )
    unsigned_foundation["service_identity_foundation_receipt_sha256"] = (
        service_identity_foundation["receipt_sha256"]
    )
    unsigned_foundation["producer_identity_foundation_receipt_sha256"] = (
        producer_identity_foundation["receipt_sha256"]
    )
    preparation_unsigned = {
        "schema": launcher.FOUNDATION_PREPARATION_SCHEMA,
        "revision": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "service_identity_foundation": service_identity_foundation,
        "service_identity_foundation_receipt_sha256": (
            service_identity_foundation["receipt_sha256"]
        ),
        "producer_identity_foundation": producer_identity_foundation,
        "producer_identity_foundation_receipt_sha256": (
            producer_identity_foundation["receipt_sha256"]
        ),
        "role_identities": _identities(),
        "key_bootstrap_receipt_sha256": keys.value["receipt_sha256"],
        "owner_public_key_ed25519_hex": owner["public_key_ed25519_hex"],
        "owner_public_key_source_sha256": owner["public_key_source_sha256"],
        "unsigned_foundation": unsigned_foundation,
        "signature_payload_sha256": producers._sha256_bytes(
            producers.producer_foundation_signature_payload(unsigned_foundation)
        ),
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    preparation = {
        **preparation_unsigned,
        "preparation_sha256": producers._sha256_json(preparation_unsigned),
    }
    calls = []
    manifest_sha = "e" * 64
    installed_foundation = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            calls.append(action)
            assert revision == "a" * 40
            if action == "prepare-producer-foundation":
                request = json.loads(frame)
                assert request["owner_public_authority"] == authority_mapping
                return preparation
            if action == "install-producer-foundation":
                request = json.loads(frame)
                sealed = producers.seal_producer_foundation(
                    unsigned_foundation,
                    owner_signature=request["owner_signature"],
                    pinned_owner_public_key_ed25519_hex=context["owner_public"],
                    pinned_owner_public_key_source_sha256=context["source_sha256"],
                )
                installed_foundation["value"] = sealed
                install_unsigned = {
                    "schema": launcher.FOUNDATION_INSTALL_RECEIPT_SCHEMA,
                    "revision": "a" * 40,
                    "capability_plan_sha256": "b" * 64,
                    "full_canary_plan_sha256": "c" * 64,
                    "full_canary_terminal_receipt": terminal,
                    "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
                    "original_full_canary_owner_approval_sha256": terminal[
                        "owner_approval_sha256"
                    ],
                    "service_identity_foundation_receipt_sha256": (
                        service_identity_foundation["receipt_sha256"]
                    ),
                    "producer_identity_foundation_receipt_sha256": (
                        producer_identity_foundation["receipt_sha256"]
                    ),
                    "preparation_sha256": preparation["preparation_sha256"],
                    "foundation_sha256": producers.producer_foundation_sha256(sealed),
                    "unit_bundle_manifest_sha256": manifest_sha,
                    "installed_units": [],
                    "installed_configs": [],
                    "installed_auxiliary_files": [],
                    "native_root_contract": {},
                    "config_install_contract": {},
                    "volatile_runtime_contract": {},
                    "authority_key_lifecycle": {},
                    "daemon_reload_completed": True,
                    "volatile_runtime_materialized": True,
                    "services_started": False,
                    "secret_material_recorded": False,
                }
                return {
                    **install_unsigned,
                    "receipt_sha256": producers._sha256_json(install_unsigned),
                }
            assert action == "preflight-producer-foundation"
            return {
                "schema": ("muncho-capability-producer-installation-preflight.v2"),
                "revision": "a" * 40,
                "foundation_sha256": producers.producer_foundation_sha256(
                    installed_foundation["value"]
                ),
                "preparation_sha256": preparation["preparation_sha256"],
                "full_canary_terminal_receipt": terminal,
                "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
                "original_full_canary_owner_approval_sha256": terminal[
                    "owner_approval_sha256"
                ],
                "service_identity_foundation_receipt_sha256": (
                    service_identity_foundation["receipt_sha256"]
                ),
                "producer_identity_foundation_receipt_sha256": (
                    producer_identity_foundation["receipt_sha256"]
                ),
                "unit_bundle_manifest_sha256": manifest_sha,
                "ready": True,
                "mutation_performed": False,
            }

    monkeypatch.setattr(launcher, "harden_owner_secret_process", lambda: None)
    result = launcher.bootstrap_producer_foundation(
        Transport(),
        revision="a" * 40,
        plan_sha256="b" * 64,
        full_canary_plan_sha256="c" * 64,
        plan_publication_receipt_sha256="e" * 64,
        terminal_receipt=terminal,
        owner_signer=Signer(),
    )
    assert calls == [
        "prepare-producer-foundation",
        "install-producer-foundation",
        "preflight-producer-foundation",
    ]
    assert result["preflight_ready"] is True
    assert result["private_key_loaded_by_launcher"] is False
    assert "PRIVATE KEY" not in json.dumps(result)


def test_publish_plan_sends_only_canonical_authority_and_wipes_frame(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "capability-plan-inputs.json"
    path.write_bytes(launcher._canonical_bytes(_plan_inputs()))
    path.chmod(0o600)
    inputs = _plan_inputs()
    terminal = inputs["full_canary_terminal_receipt"]
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_bytes(launcher._canonical_bytes(terminal))
    terminal_path.chmod(0o600)
    context_unsigned = {
        "schema": launcher.CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA,
        "revision": "a" * 40,
        "full_canary_terminal_receipt": terminal,
        "plan_inputs": inputs,
        "capability_plan_sha256": "c" * 64,
    }
    context = {
        **context_unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(context_unsigned))
        .hexdigest(),
    }
    authoring_unsigned = {
        "schema": launcher.CAPABILITY_PLAN_AUTHORING_RECEIPT_SCHEMA,
        "revision": "a" * 40,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "foundation_authoring_context_receipt_sha256": "1" * 64,
        "bitrix_foundation_receipt_sha256": "2" * 64,
        "plan_authoring_context": context,
        "plan_authoring_context_receipt_sha256": context["receipt_sha256"],
        "staged_plan_file_sha256": "3" * 64,
        "capability_inputs_sha256": inputs["inputs_sha256"],
        "capability_plan_sha256": "c" * 64,
        "output_file": str(path.resolve()),
        "output_file_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(inputs))
        .hexdigest(),
        "output_file_mode": "0600",
        "mutation_scope": "local_owner_file_create_only",
        "cloud_mutation_performed": False,
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    authoring = {
        **authoring_unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(authoring_unsigned))
        .hexdigest(),
    }
    authoring_path = tmp_path / "plan-authoring.json"
    authoring_path.write_bytes(launcher._canonical_bytes(authoring))
    authoring_path.chmod(0o600)
    captured: dict[str, object] = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "publish-plan"
            captured["frame"] = frame
            captured["during"] = bytes(frame)
            receipt = _plan_publication_receipt(terminal, plan_sha256="c" * 64)
            receipt["plan_authoring_context"] = context
            receipt["plan_authoring_context_receipt_sha256"] = context["receipt_sha256"]
            unsigned = {
                key: value for key, value in receipt.items() if key != "receipt_sha256"
            }
            receipt["receipt_sha256"] = (
                __import__("hashlib")
                .sha256(launcher._canonical_bytes(unsigned))
                .hexdigest()
            )
            return receipt

    result = launcher.publish_capability_plan(
        Transport(),
        revision="a" * 40,
        owner_subject_sha256="d" * 64,
        plan_file=path.resolve(),
        terminal_receipt_file=terminal_path.resolve(),
        plan_authoring_receipt_file=authoring_path.resolve(),
    )

    authority = json.loads(captured["during"])
    assert result["schema"] == launcher.CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA
    assert authority["plan_sha256"] == "c" * 64
    assert authority["full_canary_plan_sha256"] == "2" * 64
    assert authority["owner_subject_sha256"] == "d" * 64
    assert captured["frame"] == bytearray(b"\x00" * len(captured["during"]))


def _fixture_publication_receipt() -> dict[str, object]:
    run_id = "capability-run-1"
    plan_sha = "b" * 64
    fixture_sha = "c" * 64
    terminal = _full_canary_terminal_receipt()
    unsigned = {
        "schema": launcher.FIXTURE_PUBLICATION_RECEIPT_SCHEMA,
        "run_id": run_id,
        "release_sha": "a" * 40,
        "capability_plan_sha256": plan_sha,
        "full_canary_plan_sha256": "d" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "plan_publication_receipt_sha256": "a" * 64,
        "producer_foundation_sha256": "e" * 64,
        "authority_sha256": "f" * 64,
        "fixture_path": launcher.REVIEWED_FIXTURE_PATH,
        "fixture_sha256": fixture_sha,
        "fixture_file_identity": {
            "device": 1,
            "inode": 2,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "size": 4096,
            "mtime_ns": 123_000_000,
        },
        "receipt_path": (
            f"{launcher.FIXTURE_PUBLICATION_ROOT}/{plan_sha}/{run_id}/"
            f"{fixture_sha}.json"
        ),
        "published_at_unix_ms": 123,
    }
    return {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }


def test_publish_live_fixture_is_bounded_canonical_stdin_and_wipes_frame():
    terminal = _full_canary_terminal_receipt()
    authority = {
        "schema": launcher.CAPABILITY_FIXTURE_AUTHORITY_SCHEMA,
        "run_id": "capability-run-1",
        "owner_id": launcher.PRODUCTION_OWNER_USER_ID,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "plan_publication_receipt_sha256": "a" * 64,
        "valid_from_unix_ms": 1_900_000_000_000,
        "valid_until_unix_ms": 1_900_000_900_000,
        "public_discord_target": {
            "target_type": "public_channel",
            "guild_id": launcher.PRODUCTION_CANARY_PUBLIC_GUILD_ID,
            "channel_id": launcher.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID,
        },
        "producer_foundation_sha256": "e" * 64,
        "owner_key_id": "f" * 64,
        "signature_algorithm": "sshsig-ed25519-sha512",
        "owner_signature": "opaque-sshsig-frame",
    }
    raw = launcher._canonical_bytes(authority)
    captured: dict[str, object] = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "publish-live-fixture"
            captured["frame"] = frame
            captured["during"] = bytes(frame)
            return _fixture_publication_receipt()

    result = launcher.publish_live_fixture(
        Transport(),
        revision="a" * 40,
        stream=io.BytesIO(raw),
    )
    assert result["schema"] == launcher.FIXTURE_PUBLICATION_RECEIPT_SCHEMA
    assert captured["during"] == raw
    assert captured["frame"] == bytearray(b"\x00" * len(raw))

    with pytest.raises(
        Exception,
        match="capability_fixture_authority_input_invalid",
    ):
        launcher.publish_live_fixture(
            Transport(),
            revision="a" * 40,
            stream=io.BytesIO(b'{"z":1, "a":2}'),
        )


def test_fixture_publication_receipt_rejects_path_or_digest_drift():
    receipt = _fixture_publication_receipt()
    authority = {
        "full_canary_terminal_receipt": receipt["full_canary_terminal_receipt"],
        "full_canary_terminal_receipt_sha256": receipt[
            "full_canary_terminal_receipt_sha256"
        ],
        "original_full_canary_owner_approval_sha256": receipt[
            "original_full_canary_owner_approval_sha256"
        ],
        "plan_publication_receipt_sha256": receipt["plan_publication_receipt_sha256"],
    }
    assert (
        launcher.validate_fixture_publication_receipt(
            receipt,
            revision="a" * 40,
            authority=authority,
        )["receipt_sha256"]
        == receipt["receipt_sha256"]
    )

    tampered = json.loads(json.dumps(receipt))
    tampered["receipt_path"] = "/tmp/attacker.json"
    with pytest.raises(
        Exception,
        match="capability_fixture_publication_receipt_invalid",
    ):
        launcher.validate_fixture_publication_receipt(
            tampered,
            revision="a" * 40,
            authority=authority,
        )


def test_production_observer_runs_pinned_source_from_bounded_stdin(monkeypatch):
    revision = "a" * 40
    source = "print('committed observer source')\n"
    source_sha256 = "b" * 64
    captured: dict[str, object] = {}

    class Identity:
        def account_for_read_only_preflight(self):
            return "owner@example.com"

        def require_stable(self):
            captured["identity_stable_checks"] = (
                int(captured.get("identity_stable_checks", 0)) + 1
            )

    class KnownHosts:
        def absolute_path(self):
            return "/trusted/google_compute_known_hosts"

    class Transport:
        _owner_identity = Identity()
        _known_hosts = KnownHosts()

        @staticmethod
        def _fixed_remote_environment(*, chdir):
            assert chdir == "/"
            return ("/usr/bin/env", "-i")

        @staticmethod
        def _authorization_snapshot(account):
            assert account == "owner@example.com"
            return ("1" * 64, "2" * 64, "3" * 64)

        @staticmethod
        def _run_remote_input(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return subprocess.CompletedProcess(command, 0, b'{"ok":true}\n', b"")

    monkeypatch.setattr(
        launcher,
        "_production_observer_source",
        lambda supplied_revision: (
            (
                source,
                source_sha256,
            )
            if supplied_revision == revision
            else pytest.fail("unexpected revision")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_stable_owner_file",
        lambda path, *, maximum: b"trusted-known-hosts\n",
    )

    observation, authority = launcher.CapabilityProductionObservationTransport(
        Transport()
    ).observe(
        phase="before",
        revision=revision,
        plan_sha256="c" * 64,
        full_canary_plan_sha256="d" * 64,
        fixture_sha256="e" * 64,
        run_id="capability-run-observed",
    )

    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[2:6] == (
        "/opt/adventico-ai-platform/hermes-agent/.venv/bin/python",
        "-B",
        "-I",
        "-",
    )
    assert "-c" not in command
    assert captured["input_bytes"] == source.encode("utf-8")
    assert captured["maximum_input_bytes"] == 512 * 1024
    assert captured["maximum_output_bytes"] == 2 * 1024 * 1024
    assert captured["timeout_seconds"] == 120
    assert observation == {"ok": True}
    assert authority["observer_source_sha256"] == source_sha256
    assert captured["identity_stable_checks"] == 2


def test_owner_observed_live_run_drives_exact_two_phase_handshake(monkeypatch):
    revision = "a" * 40
    plan_sha = "b" * 64
    full_sha = "c" * 64
    terminal = _full_canary_terminal_receipt()
    terminal["full_canary_plan_sha256"] = full_sha
    terminal_unsigned = {
        key: value for key, value in terminal.items() if key != "receipt_sha256"
    }
    terminal["receipt_sha256"] = (
        __import__("hashlib")
        .sha256(launcher._canonical_bytes(terminal_unsigned))
        .hexdigest()
    )
    fixture_sha = "d" * 64
    owner_sha = "e" * 64
    run_id = "capability-run-observed"
    before_observation_sha = "1" * 64
    after_observation_sha = "2" * 64
    diff_sha = "3" * 64
    evidence_sha = "4" * 64
    after_staged = launcher.threading.Event()
    frames: list[bytearray] = []
    calls: list[str] = []

    def signed_receipt(unsigned):
        return {
            **unsigned,
            "receipt_sha256": __import__("hashlib")
            .sha256(launcher._canonical_bytes(unsigned))
            .hexdigest(),
        }

    class Transport:
        def invoke(self, supplied_revision, action, *, frame=None):
            assert supplied_revision == revision
            calls.append(action)
            if action == "run-live":
                assert after_staged.wait(2)
                return {
                    "schema": ("muncho-production-capability-canary-live-driver.v2"),
                    "ok": True,
                    "release_sha": revision,
                    "capability_plan_sha256": plan_sha,
                    "full_canary_plan_sha256": full_sha,
                    "full_canary_terminal_receipt": terminal,
                    "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
                    "original_full_canary_owner_approval_sha256": terminal[
                        "owner_approval_sha256"
                    ],
                    "plan_publication_receipt_sha256": "7" * 64,
                    "capability_owner_approval_sha256": "8" * 64,
                    "lifecycle_start_receipt_sha256": "9" * 64,
                    "lifecycle_stop_receipt_sha256": "a" * 64,
                    "fixture_sha256": fixture_sha,
                    "run_id": run_id,
                    "production_before_observation_sha256": (before_observation_sha),
                    "production_diff_sha256": diff_sha,
                    "evidence_sha256": evidence_sha,
                }
            assert isinstance(frame, bytearray)
            frames.append(frame)
            value = json.loads(bytes(frame))
            if action == "wait-api-admission-challenge":
                assert value["schema"] == (
                    launcher.API_ADMISSION_OWNER_CHALLENGE_WAIT_SCHEMA
                )
                return {
                    "challenge_bundle_sha256": "b" * 64,
                    "run_id": run_id,
                    "fixture_sha256": fixture_sha,
                    "session_id": f"capability_{run_id}",
                    "capability_epoch_sha256": "c" * 64,
                    "request": {"challenge_sha256": "d" * 64},
                }
            if action == "stage-api-admission-owner-authority":
                authority = value["owner_authority"]
                staged_unsigned = {
                    "schema": launcher.API_ADMISSION_OWNER_AUTHORITY_STAGE_SCHEMA,
                    "run_id": run_id,
                    "fixture_sha256": fixture_sha,
                    "session_id": f"capability_{run_id}",
                    "capability_epoch_sha256": "c" * 64,
                    "challenge_sha256": "d" * 64,
                    "challenge_bundle_sha256": "b" * 64,
                    "owner_authority_sha256": __import__("hashlib")
                    .sha256(launcher._canonical_bytes(authority))
                    .hexdigest(),
                    "staged_path": (
                        "/var/lib/muncho-capability-canary-evidence/"
                        f"{run_id}/api-admission-owner-authority-staged.json"
                    ),
                    "staged_at_unix_ms": 1_900_000_000_000,
                    "secret_material_recorded": False,
                    "secret_digest_recorded": False,
                }
                return signed_receipt(staged_unsigned)
            phase = value["phase"]
            if action == "wait-production-observation-marker":
                assert value["schema"] == (
                    launcher.PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA
                )
                return signed_receipt({
                    "schema": (
                        "muncho-production-capability-production-"
                        "observation-marker-wait.v1"
                    ),
                    "phase": phase,
                    "run_id": run_id,
                    "fixture_sha256": fixture_sha,
                    "marker_sha256": ("5" if phase == "before" else "6") * 64,
                    "observer_live_verified": phase == "after",
                    "secret_material_recorded": False,
                    "secret_digest_recorded": False,
                })
            assert action == "stage-production-observation"
            observation_sha = value["observation_sha256"]
            staged_unsigned = {
                "schema": launcher.PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA,
                "phase": phase,
                "run_id": run_id,
                "fixture_sha256": fixture_sha,
                "staged_envelope_sha256": value["envelope_sha256"],
                "observation_sha256": observation_sha,
                "marker_sha256": ("5" if phase == "before" else "6") * 64,
                "production_diff_sha256": (None if phase == "before" else diff_sha),
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
            }
            if phase == "after":
                after_staged.set()
            return signed_receipt(staged_unsigned)

    def collect(_transport, *, phase, **kwargs):
        assert kwargs["revision"] == revision
        observation_sha = (
            before_observation_sha if phase == "before" else after_observation_sha
        )
        unsigned = {
            "schema": launcher.PRODUCTION_OBSERVATION_ENVELOPE_SCHEMA,
            "phase": phase,
            "fixture_sha256": fixture_sha,
            "run_id": run_id,
            "owner_subject_sha256": owner_sha,
            "observation_sha256": observation_sha,
        }
        signed = {**unsigned, "owner_signature": "opaque"}
        return {
            **signed,
            "envelope_sha256": __import__("hashlib")
            .sha256(launcher._canonical_bytes(signed))
            .hexdigest(),
        }

    monkeypatch.setattr(
        launcher,
        "collect_owner_signed_production_observation",
        collect,
    )
    monkeypatch.setattr(
        launcher,
        "build_api_admission_owner_authority",
        lambda *_args, **_kwargs: {"owner": "authority"},
    )
    result = launcher.run_live_with_owner_production_observations(
        Transport(),
        object(),
        revision=revision,
        plan_sha256=plan_sha,
        full_canary_plan_sha256=full_sha,
        terminal_receipt=terminal,
        fixture_sha256=fixture_sha,
        run_id=run_id,
        owner_subject_sha256=owner_sha,
        timeout_seconds=30,
    )

    assert result["production_diff_sha256"] == diff_sha
    assert result["live_evidence_sha256"] == evidence_sha
    assert result["full_canary_terminal_receipt_sha256"] == terminal["receipt_sha256"]
    assert result["plan_publication_receipt_sha256"] == "7" * 64
    assert result["capability_owner_approval_sha256"] == "8" * 64
    assert result["lifecycle_start_receipt_sha256"] == "9" * 64
    assert result["lifecycle_stop_receipt_sha256"] == "a" * 64
    assert calls.count("run-live") == 1
    assert [action for action in calls if action != "run-live"] == [
        "wait-production-observation-marker",
        "stage-production-observation",
        "wait-api-admission-challenge",
        "stage-api-admission-owner-authority",
        "wait-production-observation-marker",
        "stage-production-observation",
    ]
    assert all(frame == bytearray(len(frame)) for frame in frames)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("release_sha", "f" * 40),
        ("capability_plan_sha256", "f" * 64),
        ("full_canary_plan_sha256", "f" * 64),
        ("fixture_sha256", "f" * 64),
        ("run_id", "remote-substituted-run"),
    ),
)
def test_api_admission_owner_refuses_remote_binding_before_signing(
    monkeypatch,
    field,
    replacement,
):
    fixture = {
        "release_sha": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "run_id": "trusted-run",
        "valid_from_unix_ms": 1_900_000_000_000,
        "valid_until_unix_ms": 1_900_000_900_000,
    }
    fixture_sha = __import__("hashlib").sha256(
        launcher._canonical_bytes(fixture)
    ).hexdigest()
    expected_catalog = {"commands": {"allowed": [{"command_sha256": "1" * 64}]}}
    challenge = {
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha,
        "run_id": fixture["run_id"],
        "fixture": fixture,
        "request": {"session_id": "capability_trusted-run"},
        "catalog": expected_catalog,
    }
    challenge[field] = replacement
    signer_calls: list[str] = []

    class Signer:
        def inspect(self):
            signer_calls.append("inspect")
            raise AssertionError("signer must not be inspected")

        def sign(self, *_args, **_kwargs):
            signer_calls.append("sign")
            raise AssertionError("signer must not be called")

    monkeypatch.setattr(
        "gateway.canonical_capability_canary_e2e._validate_fixture",
        lambda *_args, **_kwargs: fixture,
    )
    monkeypatch.setattr(
        "gateway.canonical_capability_canary_live_driver.build_live_probe_catalog",
        lambda **_kwargs: expected_catalog,
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="api_admission_owner_challenge_invalid",
    ):
        launcher.build_api_admission_owner_authority(
            challenge,
            expected_revision="a" * 40,
            expected_plan_sha256="b" * 64,
            expected_full_canary_plan_sha256="c" * 64,
            expected_fixture_sha256=fixture_sha,
            expected_run_id="trusted-run",
            owner_signer=Signer(),
            now_unix_ms=1_900_000_001_000,
        )
    assert signer_calls == []


def test_api_admission_owner_refuses_cloud_substituted_command_catalog(
    monkeypatch,
):
    fixture = {
        "release_sha": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "run_id": "trusted-run",
        "valid_from_unix_ms": 1_900_000_000_000,
        "valid_until_unix_ms": 1_900_000_900_000,
    }
    fixture_sha = __import__("hashlib").sha256(
        launcher._canonical_bytes(fixture)
    ).hexdigest()
    expected_catalog = {
        "commands": {
            "allowed": [
                {
                    "command_id": "command:trusted",
                    "command_b64": "dHJ1c3RlZA==",
                    "command_sha256": "1" * 64,
                    "max_uses": 1,
                }
            ]
        }
    }
    substituted_catalog = json.loads(json.dumps(expected_catalog))
    substituted_catalog["commands"]["allowed"][0]["command_b64"] = "ZXZpbA=="
    challenge = {
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha,
        "run_id": fixture["run_id"],
        "fixture": fixture,
        "request": {"session_id": "capability_trusted-run"},
        "catalog": substituted_catalog,
    }
    signer_calls: list[str] = []

    class Signer:
        def inspect(self):
            signer_calls.append("inspect")
            raise AssertionError("signer must not be inspected")

        def sign(self, *_args, **_kwargs):
            signer_calls.append("sign")
            raise AssertionError("signer must not be called")

    monkeypatch.setattr(
        "gateway.canonical_capability_canary_e2e._validate_fixture",
        lambda *_args, **_kwargs: fixture,
    )
    monkeypatch.setattr(
        "gateway.canonical_capability_canary_live_driver.build_live_probe_catalog",
        lambda **_kwargs: expected_catalog,
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="api_admission_owner_challenge_invalid",
    ):
        launcher.build_api_admission_owner_authority(
            challenge,
            expected_revision="a" * 40,
            expected_plan_sha256="b" * 64,
            expected_full_canary_plan_sha256="c" * 64,
            expected_fixture_sha256=fixture_sha,
            expected_run_id="trusted-run",
            owner_signer=Signer(),
            now_unix_ms=1_900_000_001_000,
        )
    assert signer_calls == []


def test_plan_publication_inputs_reject_bot_reuse_and_tamper():
    same_bot = _plan_inputs()
    same_bot["discord"]["routeback_bot_user_id"] = same_bot["discord"][
        "connector_bot_user_id"
    ]
    unsigned = {key: value for key, value in same_bot.items() if key != "inputs_sha256"}
    same_bot["inputs_sha256"] = (
        __import__("hashlib").sha256(launcher._canonical_bytes(unsigned)).hexdigest()
    )
    with pytest.raises(ValueError, match="not distinct"):
        launcher.validate_plan_publication_inputs(same_bot)

    for role in ("connector_bot_user_id", "routeback_bot_user_id"):
        production_reuse = _plan_inputs()
        production_reuse["discord"][role] = launcher.PRODUCTION_DISCORD_BOT_USER_ID
        unsigned = {
            key: value
            for key, value in production_reuse.items()
            if key != "inputs_sha256"
        }
        production_reuse["inputs_sha256"] = (
            __import__("hashlib")
            .sha256(launcher._canonical_bytes(unsigned))
            .hexdigest()
        )
        with pytest.raises(ValueError, match="not distinct"):
            launcher.validate_plan_publication_inputs(production_reuse)

    tampered = _plan_inputs()
    tampered["identities"]["browser_uid"] += 1
    with pytest.raises(ValueError, match="identities are not fixed"):
        launcher.validate_plan_publication_inputs(tampered)

    for field, value in (
        ("browser_uid", 2999),
        ("browser_uid", launcher.CAPABILITY_PLANNED_IDENTITIES["worker_uid"]),
    ):
        rehashed = _plan_inputs()
        rehashed["identities"][field] = value
        unsigned = {
            key: item for key, item in rehashed.items() if key != "inputs_sha256"
        }
        rehashed["inputs_sha256"] = (
            __import__("hashlib")
            .sha256(launcher._canonical_bytes(unsigned))
            .hexdigest()
        )
        with pytest.raises(ValueError, match="identities are not fixed"):
            launcher.validate_plan_publication_inputs(rehashed)


def test_plan_authoring_request_rejects_production_bot_reuse():
    terminal = _full_canary_terminal_receipt()
    foundation = {
        "foundation_authoring_context": {},
        "foundation_authoring_context_receipt_sha256": "a" * 64,
    }
    bitrix = {"receipt_sha256": "b" * 64}
    for connector, routeback in (
        (launcher.PRODUCTION_DISCORD_BOT_USER_ID, "1600000000000000002"),
        ("1600000000000000001", launcher.PRODUCTION_DISCORD_BOT_USER_ID),
    ):
        with pytest.raises(ValueError, match="not distinct"):
            launcher.build_plan_authoring_request(
                revision="a" * 40,
                terminal_receipt=terminal,
                foundation_authoring_receipt=foundation,
                bitrix_foundation_receipt=bitrix,
                connector_bot_user_id=connector,
                routeback_bot_user_id=routeback,
            )


@pytest.mark.parametrize(
    "locked_channel_id", sorted(launcher.LOCKED_NONPUBLIC_CHANNEL_IDS)
)
def test_plan_publication_inputs_reject_each_locked_discord_channel(
    locked_channel_id,
):
    inputs = _plan_inputs()
    inputs["discord"]["allowed_channel_ids"] = [locked_channel_id]
    unsigned = {key: value for key, value in inputs.items() if key != "inputs_sha256"}
    inputs["inputs_sha256"] = (
        __import__("hashlib").sha256(launcher._canonical_bytes(unsigned)).hexdigest()
    )

    with pytest.raises(ValueError, match="public Discord target is invalid"):
        launcher.validate_plan_publication_inputs(inputs)


def test_storage_preflight_is_read_only_precise_and_identity_bound():
    revision = "a" * 40
    rollback = "b" * 40
    stale = "c" * 40
    available = 5 * 1024 * 1024 * 1024
    unsigned = {
        "schema": launcher.RELEASE_STORAGE_PREFLIGHT_SCHEMA,
        "ok": False,
        "revision": revision,
        "release_base": {
            "path": launcher.RELEASE_BASE,
            "state": "present_exact",
            "device": 1,
            "inode": 2,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
        },
        "capacity": {
            "available_bytes": available,
            "minimum_packaging_free_bytes": (launcher.MINIMUM_PACKAGING_FREE_BYTES),
            "shortfall_bytes": (launcher.MINIMUM_PACKAGING_FREE_BYTES - available),
        },
        "retention": {
            "maximum_managed_releases": launcher.MAXIMUM_MANAGED_RELEASES,
            "protect_target_release": True,
            "protect_newest_rollback_release": True,
        },
        "protected_release_paths": sorted([
            f"{launcher.RELEASE_BASE}/{revision}",
            f"{launcher.RELEASE_BASE}/{rollback}",
        ]),
        "cleanup_candidates": [
            {
                "revision": stale,
                "path": f"{launcher.RELEASE_BASE}/{stale}",
                "device": 1,
                "inode": 3,
                "uid": 0,
                "gid": 0,
                "mode": "0755",
                "mtime_ns": 10,
                "ctime_ns": 11,
            }
        ],
        "cleanup_mutation_performed": False,
        "cleanup_requires_fresh_owner_approval": True,
        "arbitrary_path_cleanup_allowed": False,
        "blocker": "minimum_packaging_free_bytes_not_met",
    }
    receipt = {
        **unsigned,
        "receipt_sha256": __import__("hashlib")
        .sha256(launcher._canonical_bytes(unsigned))
        .hexdigest(),
    }

    validated = launcher.validate_release_storage_preflight(
        receipt,
        revision=revision,
    )
    assert validated["ok"] is False
    assert validated["cleanup_mutation_performed"] is False
    assert validated["cleanup_candidates"][0]["path"].endswith(stale)

    tampered = json.loads(json.dumps(receipt))
    tampered["cleanup_candidates"][0]["uid"] = 1000
    with pytest.raises(
        Exception,
        match="capability_release_storage_preflight_invalid",
    ):
        launcher.validate_release_storage_preflight(
            tampered,
            revision=revision,
        )


def test_provision_codex_wipes_token_and_frame(monkeypatch, tmp_path):
    access = _jwt(int(time.time()) + 3_600)
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"tokens": {"access_token": access}}))
    path.chmod(0o600)
    captured: dict[str, bytes] = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            captured["during"] = bytes(frame)
            assert revision == "a" * 40
            assert action == "provision-codex"
            return {"ok": True}

    monkeypatch.setattr(launcher, "harden_owner_secret_process", lambda: None)
    result = launcher.provision_codex(
        Transport(),
        revision="a" * 40,
        plan_sha256="b" * 64,
        owner_subject_sha256="c" * 64,
        auth_path=path,
    )
    assert result == {"ok": True}
    assert access.encode() in captured["during"]
    assert b"refresh_token" not in captured["during"]


def test_provision_discord_connector_uses_its_kind_and_wipes_buffers(monkeypatch):
    captured: dict[str, object] = {}

    def build_frame(**kwargs):
        captured["kind"] = kwargs["kind"]
        captured["secret"] = kwargs["secret"]
        frame = bytearray(b"framed-connector-secret")
        captured["frame"] = frame
        return frame

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "provision-discord-connector"
            assert bytes(frame) == b"framed-connector-secret"
            return {"ok": True, "secret_material_recorded": False}

    monkeypatch.setattr(launcher, "harden_owner_secret_process", lambda: None)
    monkeypatch.setattr(launcher, "build_secret_lease_frame", build_frame)
    monkeypatch.setattr(
        launcher,
        "_secret_lease_frame_expiry",
        lambda _frame: int(time.time()) + 900,
    )
    result = launcher.provision_discord_connector(
        Transport(),
        revision="a" * 40,
        plan_sha256="b" * 64,
        owner_subject_sha256="c" * 64,
        stream=io.BytesIO(b"connector-token"),
    )

    assert result == {"ok": True, "secret_material_recorded": False}
    assert captured["kind"] == "discord_connector_token"
    assert captured["secret"] == bytearray(b"\x00" * len(b"connector-token"))
    assert captured["frame"] == bytearray(b"\x00" * len(b"framed-connector-secret"))


@pytest.mark.parametrize(
    "provisioner,action,kind,secret",
    (
        (
            launcher.provision_discord_routeback,
            "provision-discord-routeback",
            "discord_routeback_token",
            b"routeback-token",
        ),
        (
            launcher.provision_bitrix_operational_edge,
            "provision-bitrix-operational-edge",
            "bitrix_operational_edge_webhook",
            b"https://example.bitrix24.eu/rest/1/opaque/",
        ),
    ),
)
def test_new_stdin_provisioners_use_distinct_kind_and_wipe(
    monkeypatch,
    provisioner,
    action,
    kind,
    secret,
):
    captured = {}

    def build_frame(**kwargs):
        captured["kind"] = kwargs["kind"]
        captured["secret"] = kwargs["secret"]
        captured["frame"] = bytearray(b"opaque-framed-input")
        return captured["frame"]

    class Transport:
        def invoke(self, revision, remote_action, *, frame=None):
            assert revision == "a" * 40
            assert remote_action == action
            assert frame is captured["frame"]
            return {"ok": True}

    monkeypatch.setattr(launcher, "harden_owner_secret_process", lambda: None)
    monkeypatch.setattr(launcher, "build_secret_lease_frame", build_frame)
    monkeypatch.setattr(
        launcher,
        "_secret_lease_frame_expiry",
        lambda _frame: int(time.time()) + 900,
    )
    result = provisioner(
        Transport(),
        revision="a" * 40,
        plan_sha256="b" * 64,
        owner_subject_sha256="c" * 64,
        stream=io.BytesIO(secret),
    )
    assert result == {"ok": True}
    assert captured["kind"] == kind
    assert captured["secret"] == bytearray(len(secret))
    assert captured["frame"] == bytearray(len(b"opaque-framed-input"))


def test_api_control_provisioner_generates_secret_and_wipes(monkeypatch):
    generated = bytearray(b"generated-api-control-key")
    captured = {}

    def build_frame(**kwargs):
        captured["kind"] = kwargs["kind"]
        captured["secret"] = kwargs["secret"]
        captured["frame"] = bytearray(b"api-frame")
        return captured["frame"]

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "provision-api-control"
            return {"ok": True}

    monkeypatch.setattr(launcher, "harden_owner_secret_process", lambda: None)
    monkeypatch.setattr(launcher, "generate_api_control_key", lambda: generated)
    monkeypatch.setattr(launcher, "build_secret_lease_frame", build_frame)
    monkeypatch.setattr(
        launcher,
        "_secret_lease_frame_expiry",
        lambda _frame: int(time.time()) + 900,
    )
    assert launcher.provision_api_control(
        Transport(),
        revision="a" * 40,
        plan_sha256="b" * 64,
        owner_subject_sha256="c" * 64,
    ) == {"ok": True}
    assert captured["kind"] == "api_server_control_key"
    assert generated == bytearray(len(generated))
    assert captured["frame"] == bytearray(len(b"api-frame"))


@pytest.mark.parametrize(
    ("fixture_valid_until", "approval_not_after", "accepted"),
    ((2_000, 1_895, True), (1_035, 1_030, False)),
)
def test_install_approval_is_fresh_plan_bound_and_stdin_only(
    monkeypatch,
    tmp_path,
    fixture_valid_until,
    approval_not_after,
    accepted,
):
    revision = "a" * 40
    terminal = _full_canary_terminal_receipt()
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_bytes(launcher._canonical_bytes(terminal))
    terminal_path.chmod(0o600)
    calls = []
    bindings = {
        "api_control": "lease.api_control",
        "bitrix_operational_edge_webhook": "lease.bitrix_operational_edge",
        "discord_canonical_routeback_bot_token": "lease.discord_routeback",
        "discord_public_session_bot_token": "lease.discord_connector",
        "mac_ops_gitlab": "lease.mac_ops",
        "openai_codex": "lease.codex",
    }

    class Transport:
        def invoke(self, remote_revision, action, *, frame=None):
            calls.append((remote_revision, action, frame))
            if action == "preflight-stopped":
                return {
                    "schema": launcher.CAPABILITY_PREFLIGHT_SCHEMA,
                    "ok": True,
                    "phase": "stopped",
                    "revision": revision,
                    "plan_sha256": "b" * 64,
                    "full_canary_plan_sha256": "2" * 64,
                    "full_canary_terminal_receipt": terminal,
                    "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
                    "original_full_canary_owner_approval_sha256": terminal[
                        "owner_approval_sha256"
                    ],
                    "report_sha256": "d" * 64,
                    "state_sha256": "9" * 64,
                    "plan_publication_receipt_sha256": "a" * 64,
                    "service_identity_foundation_receipt_sha256": "3" * 64,
                    "producer_identity_foundation_receipt_sha256": "f" * 64,
                    "fixture_sha256": "6" * 64,
                    "fixture_publication_receipt_sha256": "5" * 64,
                    "fixture_valid_until_unix": fixture_valid_until,
                    "lease_expires_at_unix_by_binding": {
                        binding: 2_100 for binding in bindings
                    },
                    "minimum_lease_expires_at_unix": 2_100,
                    "bitrix_foundation_expires_at_unix": 2_200,
                    "bitrix_expiry_watchdog_authority_sha256": "4" * 64,
                    "approval_not_after_unix": approval_not_after,
                    "observed_at_unix": 1_000,
                    "checks": {
                        "browser.executable": True,
                        "browser.host_identity": True,
                        "browser.userns_sandbox": True,
                        "browser.principal_smoke": True,
                        "worker.executables": True,
                        "worker.systemd252_tmpfs_contract": True,
                        "execution.host_identity": True,
                        "service_identities.foundation": True,
                        "service_identity.mac_ops": True,
                        "service_identity.connector": True,
                        "producer.foundation": True,
                        "producer.host_identity": True,
                    },
                    "evidence": {
                        "service_identities.foundation": {
                            "receipt_sha256": "3" * 64,
                        },
                        "service_identity.mac_ops": {
                            "schema": (
                                "muncho-production-capability-service-host-identity.v2"
                            ),
                            "plan_sha256": "b" * 64,
                            "role": "mac_ops",
                            "state": "present_exact",
                            "receipt_sha256": "4" * 64,
                        },
                        "service_identity.connector": {
                            "schema": (
                                "muncho-production-capability-service-host-identity.v2"
                            ),
                            "plan_sha256": "b" * 64,
                            "role": "connector",
                            "state": "present_exact",
                            "receipt_sha256": "5" * 64,
                        },
                        "producer.foundation": {
                            "producer_identity_foundation_receipt_sha256": ("f" * 64),
                        },
                        "producer.host_identity": {
                            "schema": (
                                "muncho-production-capability-producer-host-identity.v2"
                            ),
                            "plan_sha256": "b" * 64,
                            "create_only_eligible": True,
                            "persistent_supplementary_memberships": False,
                        },
                        "browser.host_identity": {
                            "schema": launcher.CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA,
                            "plan_sha256": "b" * 64,
                            "create_only_eligible": True,
                            "receipt_sha256": "8" * 64,
                        },
                        "execution.host_identity": {
                            "schema": launcher.CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA,
                            "plan_sha256": "b" * 64,
                            "create_only_eligible": True,
                            "receipt_sha256": "7" * 64,
                        },
                        **{
                            evidence_name: {
                                "install_receipt_sha256": str(index) * 64,
                                "expires_at_unix": 2_100,
                            }
                            for index, evidence_name in enumerate(
                                bindings.values(), start=1
                            )
                        },
                    },
                }
            approval = json.loads(bytes(frame))
            from gateway.canonical_capability_canary_runtime import (
                CapabilityCanaryOwnerApproval,
            )

            CapabilityCanaryOwnerApproval.from_mapping(approval)
            assert approval["schema"] == launcher.CAPABILITY_APPROVAL_SCHEMA
            assert approval["plan_sha256"] == "b" * 64
            assert approval["full_canary_plan_sha256"] == "2" * 64
            assert approval["full_canary_terminal_receipt"] == terminal
            assert (
                approval["original_full_canary_owner_approval_sha256"]
                == (terminal["owner_approval_sha256"])
            )
            assert approval["stopped_preflight_state_sha256"] == "9" * 64
            assert approval["stopped_preflight_observed_at_unix"] == 1_000
            assert approval["plan_publication_receipt_sha256"] == "a" * 64
            assert approval["approval_not_after_unix"] == approval_not_after
            assert approval["owner_subject_sha256"] == "e" * 64
            assert approval["expires_at_unix"] - approval["approved_at_unix"] == (
                approval_not_after - 1_000
            )
            assert len(approval["nonce_sha256"]) == 64
            return {"ok": True, "approval_sha256": "f" * 64}

    monkeypatch.setattr(launcher.secrets, "token_bytes", lambda _size: b"n" * 32)
    if accepted:
        result = launcher.install_capability_approval(
            Transport(),
            revision=revision,
            owner_subject_sha256="e" * 64,
            terminal_receipt_file=terminal_path.resolve(),
            now_unix=1_000,
        )
        assert result == {"ok": True, "approval_sha256": "f" * 64}
        assert calls[0] == (revision, "preflight-stopped", None)
        assert calls[1][0:2] == (revision, "install-approval")
        assert calls[1][2] == bytearray(len(calls[1][2]))
    else:
        with pytest.raises(
            launcher.OwnerLauncherError,
            match="capability_stopped_preflight_invalid",
        ):
            launcher.install_capability_approval(
                Transport(),
                revision=revision,
                owner_subject_sha256="e" * 64,
                terminal_receipt_file=terminal_path.resolve(),
                now_unix=1_000,
            )
        assert calls == [(revision, "preflight-stopped", None)]


@pytest.mark.parametrize("action", ("start", "stop"))
def test_main_invokes_start_and_stop_without_a_secret_frame(
    monkeypatch,
    capfd,
    action,
):
    calls: list[tuple[str, str, object]] = []

    class Trusted:
        pass

    class Identity:
        owner_subject_sha256 = "b" * 64

        def __init__(self, *args, **kwargs):
            pass

        def account_for_read_only_preflight(self):
            return "owner@example.com"

    class Transport:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, revision, remote_action, *, frame=None):
            calls.append((revision, remote_action, frame))
            return {"ok": True, "action": remote_action}

    monkeypatch.setattr(launcher, "require_trusted_owner_runtime", lambda _r: Trusted())
    monkeypatch.setattr(
        launcher,
        "require_capability_launcher_provenance",
        lambda revision: calls.append((revision, "provenance", None)) or "f" * 64,
    )
    monkeypatch.setattr(launcher, "PinnedGcloudConfiguration", lambda: object())
    monkeypatch.setattr(launcher, "GcloudOwnerAccessToken", Identity)
    monkeypatch.setattr(launcher, "CapabilityCanaryTransport", Transport)

    assert launcher.main(("a" * 40, action)) == 0
    assert calls == [
        ("a" * 40, "provenance", None),
        ("a" * 40, action, None),
    ]
    assert json.loads(capfd.readouterr().out) == {"ok": True, "action": action}
