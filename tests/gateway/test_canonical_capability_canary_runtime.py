"""Focused tests for production-shaped capability-canary packaging."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import stat
import time
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import gateway.canonical_capability_canary_runtime as runtime
from gateway.canonical_full_canary_runtime import (
    FullCanaryIdentities,
    FullCanaryOwnerApproval,
    FullCanaryPlan,
)


def _full_plan() -> FullCanaryPlan:
    plan = object.__new__(FullCanaryPlan)
    identities = FullCanaryIdentities(
        writer_user="muncho_writer",
        writer_group="muncho_writer",
        writer_uid=2101,
        writer_gid=2201,
        gateway_user="hermes_gateway",
        gateway_group="hermes_gateway",
        gateway_uid=max(os.getuid(), 1),
        gateway_gid=max(os.getgid(), 1),
        socket_client_group="muncho_writer_clients",
        socket_client_gid=2203,
        edge_user="muncho-discord-egress",
        edge_group="muncho-discord-egress",
        edge_uid=2103,
        edge_gid=2204,
    )
    for name, value in {
        "revision": "a" * 40,
        "sha256": "b" * 64,
        "release": {
            "artifact_root": "/opt/muncho-canary-releases/" + "a" * 40,
            "artifact_sha256": "c" * 64,
            "interpreter": "/opt/muncho-canary-releases/" + "a" * 40 + "/venv/bin/python",
        },
        "identities": identities,
    }.items():
        object.__setattr__(plan, name, value)
    return plan


def _plan(
    full_plan: FullCanaryPlan | None = None,
) -> runtime.CapabilityCanaryPlan:
    return runtime.build_capability_plan(
        full_plan=_full_plan() if full_plan is None else full_plan,
        mac_ops_uid=2104,
        mac_ops_gid=2205,
        connector_uid=2105,
        connector_gid=2206,
        bitrix_operational_edge_uid=2108,
        bitrix_operational_edge_gid=2210,
        bitrix_operational_edge_client_gid=2211,
        browser_uid=2106,
        browser_gid=2207,
        worker_uid=2107,
        worker_gid=2208,
        worker_client_gid=2209,
        connector_bot_user_id="1600000000000000001",
        routeback_bot_user_id="1600000000000000002",
        connector_allowed_guild_ids=("1282725267068157972",),
        connector_allowed_channel_ids=(
            runtime.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID,
        ),
        connector_allowed_user_ids=("1279454038731264061",),
        browser_node_sha256="5" * 64,
        browser_wrapper_sha256="6" * 64,
        browser_native_sha256="7" * 64,
        browser_executable_sha256="3" * 64,
        agent_browser_config_sha256="8" * 64,
        worker_bwrap_sha256="9" * 64,
        worker_shell_sha256="a" * 64,
        runtime_dependency_manifest_sha256="4" * 64,
        bitrix_operational_edge_asset_manifest_sha256="b" * 64,
        bitrix_operational_edge_rendered_unit_sha256="c" * 64,
        bitrix_operational_edge_rendered_config_sha256="d" * 64,
        bitrix_operational_edge_rendered_trust_sha256="e" * 64,
        bitrix_operational_edge_identity_bootstrap_receipt_sha256="f" * 64,
        bitrix_operational_edge_receipt_public_key_id="1" * 64,
        bitrix_operational_edge_key_bootstrap_receipt_sha256="2" * 64,
    )


def _production_observation_wait_request(
    plan: runtime.CapabilityCanaryPlan,
    *,
    phase: str = "before",
) -> dict[str, object]:
    return {
        "schema": runtime.CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA,
        "phase": phase,
        "canary_revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": "c" * 64,
        "run_id": "capability-run-observed",
        "owner_subject_sha256": "d" * 64,
        "timeout_seconds": 30,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _stop_proof(
    plan: runtime.CapabilityCanaryPlan,
    *,
    observed_at_unix: int | None = None,
) -> dict[str, object]:
    stopped = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": 0,
        "UnitFileState": "disabled",
        "DropInPaths": "",
    }
    return dict(
        runtime.build_capability_stop_proof(
            plan,
            {
                unit: dict(stopped)
                for unit in runtime.CAPABILITY_STOP_ORDER
            },
            observed_at_unix=(
                int(time.time())
                if observed_at_unix is None
                else observed_at_unix
            ),
        )
    )


def _routeback_file_metadata(*, inode: int = 41) -> dict[str, int]:
    return {
        "device": 7,
        "inode": inode,
        "uid": _full_plan().identities.edge_uid,
        "gid": _full_plan().identities.edge_gid,
        "mode": 0o400,
        "size": 96,
        "mtime_ns": 1_700_000_000_000_000_001,
        "ctime_ns": 1_700_000_000_000_000_002,
    }


def _plan_publication_authority(
    plan: runtime.CapabilityCanaryPlan,
) -> dict[str, object]:
    inputs = {
        "identities": {
            "mac_ops_uid": plan.identities.mac_ops_uid,
            "mac_ops_gid": plan.identities.mac_ops_gid,
            "connector_uid": plan.identities.connector_uid,
            "connector_gid": plan.identities.connector_gid,
            "bitrix_operational_edge_uid": (
                plan.identities.bitrix_operational_edge_uid
            ),
            "bitrix_operational_edge_gid": (
                plan.identities.bitrix_operational_edge_gid
            ),
            "bitrix_operational_edge_client_gid": (
                plan.identities.bitrix_operational_edge_client_gid
            ),
            "browser_uid": plan.identities.browser_uid,
            "browser_gid": plan.identities.browser_gid,
            "worker_uid": plan.identities.worker_uid,
            "worker_gid": plan.identities.worker_gid,
            "worker_client_gid": plan.identities.worker_client_gid,
        },
        "discord": {
            "connector_bot_user_id": plan.connector_bot_user_id,
            "routeback_bot_user_id": plan.routeback_bot_user_id,
            "allowed_guild_ids": list(plan.connector_allowed_guild_ids),
            "allowed_channel_ids": list(plan.connector_allowed_channel_ids),
            "allowed_user_ids": list(plan.connector_allowed_user_ids),
        },
        "artifacts": {
            "browser_node_sha256": plan.browser_node_sha256,
            "browser_wrapper_sha256": plan.browser_wrapper_sha256,
            "browser_native_sha256": plan.browser_native_sha256,
            "browser_executable_sha256": plan.browser_executable_sha256,
            "agent_browser_config_sha256": plan.agent_browser_config_sha256,
            "worker_bwrap_sha256": plan.worker_bwrap_sha256,
            "worker_shell_sha256": plan.worker_shell_sha256,
            "runtime_dependency_manifest_sha256": (
                plan.runtime_dependency_manifest_sha256
            ),
            "bitrix_operational_edge_asset_manifest_sha256": (
                plan.bitrix_operational_edge_asset_manifest_sha256
            ),
            "bitrix_operational_edge_rendered_unit_sha256": (
                plan.bitrix_operational_edge_rendered_unit_sha256
            ),
            "bitrix_operational_edge_rendered_config_sha256": (
                plan.bitrix_operational_edge_rendered_config_sha256
            ),
            "bitrix_operational_edge_rendered_trust_sha256": (
                plan.bitrix_operational_edge_rendered_trust_sha256
            ),
            "bitrix_operational_edge_identity_bootstrap_receipt_sha256": (
                plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256
            ),
            "bitrix_operational_edge_receipt_public_key_id": (
                plan.bitrix_operational_edge_receipt_public_key_id
            ),
            "bitrix_operational_edge_key_bootstrap_receipt_sha256": (
                plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
            ),
        },
    }
    unsigned = {
        "schema": runtime.CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA,
        "scope": runtime.CAPABILITY_PLAN_PUBLICATION_SCOPE,
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "plan_sha256": plan.sha256,
        "owner_subject_sha256": "d" * 64,
        "authority_kind": "trusted_gcloud_owner_explicit_plan_digest",
        "cryptographic_owner_proof": False,
        "inputs": inputs,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "authority_sha256": runtime._sha256_json(unsigned),
    }


def _bitrix_foundation_authority(
    full: FullCanaryPlan,
    *,
    now_unix: int | None = None,
) -> dict[str, object]:
    issued = int(time.time()) if now_unix is None else now_unix
    unsigned = {
        "schema": runtime.CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA,
        "scope": runtime.CAPABILITY_BITRIX_FOUNDATION_SCOPE,
        "revision": full.revision,
        "full_canary_plan_sha256": full.sha256,
        "release_artifact_sha256": full.release["artifact_sha256"],
        "owner_subject_sha256": "a" * 64,
        "authority_kind": "trusted_gcloud_owner_explicit_foundation_digest",
        "cryptographic_owner_proof": False,
        "issued_at_unix": issued,
        "expires_at_unix": issued + 60,
        "identities": {
            "service_uid": 2108,
            "service_gid": 2210,
            "socket_client_gid": 2211,
            "business_edge_uid": 2104,
        },
        "asset_manifest_sha256": "b" * 64,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "authority_sha256": runtime._sha256_json(unsigned),
    }


def _jwt(expiry: int) -> bytearray:
    def segment(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return bytearray(f"{segment({'alg': 'none'})}.{segment({'exp': expiry})}.sig".encode())


def test_plan_renders_exact_model_owned_first_wave():
    plan = _plan()
    restored = runtime.CapabilityCanaryPlan.from_mapping(plan.to_mapping())
    assert restored == plan
    config = yaml.safe_load(runtime.render_gateway_config(plan))

    assert config["model"] == {
        "default": "gpt-5.6-sol",
        "provider": "openai-codex",
    }
    assert config["agent"]["adaptive_reasoning"] == {
        "enabled": True,
        "max_effort": "max",
    }
    assert config["platform_toolsets"]["api_server"] == list(
        runtime.FIRST_WAVE_TOOLSETS
    )
    assert config["platform_toolsets"]["relay"] == list(
        runtime.FIRST_WAVE_TOOLSETS
    )
    assert config["gateway"]["isolated_runtime"] is False
    assert set(config["platforms"]) == {"api_server", "relay"}
    assert config["platforms"]["relay"]["extra"]["relay_url"] == (
        "unix:///run/muncho-discord-connector/connector.sock"
    )
    assert "mac_ops" in config["platform_toolsets"]["api_server"]
    assert config["kanban"] == {
        "auxiliary_planning_enabled": False,
        "auto_decompose": False,
        "dispatch_in_gateway": False,
    }
    assert config["cron"] == {"enabled": False}
    assert config["plugins"] == {
        "enabled": [runtime.CAPABILITY_OBSERVER_PLUGIN]
    }
    assert config["hooks"] == {}
    assert config["memory"] == {
        "memory_enabled": True,
        "user_profile_enabled": True,
    }


def test_plan_binds_two_canary_bots_distinct_from_production():
    plan = _plan()
    connector = plan.to_mapping()["discord_connector"]

    assert connector["connector_bot_user_id"] == plan.connector_bot_user_id
    assert connector["routeback_bot_user_id"] == plan.routeback_bot_user_id
    assert connector["production_bot_user_id"] == (
        runtime.PRODUCTION_DISCORD_BOT_USER_ID
    )
    assert len(
        {
            connector["connector_bot_user_id"],
            connector["routeback_bot_user_id"],
            connector["production_bot_user_id"],
        }
    ) == 3

    tampered = plan.to_mapping()
    tampered["discord_connector"]["connector_bot_user_id"] = (
        runtime.PRODUCTION_DISCORD_BOT_USER_ID
    )
    with pytest.raises(ValueError, match="bot identities are not isolated"):
        runtime.CapabilityCanaryPlan.from_mapping(tampered)


def test_live_routeback_identity_is_plan_bound_and_secret_free(monkeypatch):
    plan = _plan()
    full = _full_plan()
    metadata = _routeback_file_metadata()
    calls = []

    class FakeApi:
        def current_user(self, *, timeout_seconds):
            calls.append(("current_user", timeout_seconds))
            return {
                "id": plan.routeback_bot_user_id,
                "bot": True,
                "username": "must-not-enter-the-attestation",
            }

    class FakeAdapter:
        def __init__(self):
            self._api = FakeApi()

        def close(self):
            calls.append(("close",))

    def factory(path, **kwargs):
        calls.append(("factory", path, kwargs))
        return FakeAdapter()

    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: dict(metadata),
    )
    receipt = runtime._attest_live_routeback_bot_identity(
        plan,
        full,
        adapter_factory=factory,
        now_unix=1_700_000_000,
    )

    assert receipt["schema"] == runtime.CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA
    assert receipt["plan_sha256"] == plan.sha256
    assert receipt["full_canary_plan_sha256"] == full.sha256
    assert receipt["live_bot_user_id"] == plan.routeback_bot_user_id
    assert receipt["planned_routeback_bot_user_id"] == plan.routeback_bot_user_id
    assert receipt["connector_bot_user_id"] == plan.connector_bot_user_id
    assert receipt["production_bot_user_id"] == (
        runtime.PRODUCTION_DISCORD_BOT_USER_ID
    )
    assert receipt["pairwise_distinct"] is True
    assert receipt["credential_file_metadata_sha256"] == runtime._sha256_json(
        metadata
    )
    assert receipt["provenance"] == {
        "source": "discord_rest_api_v10_current_user",
        "http_method": "GET",
        "resource": "/users/@me",
        "credential_boundary": "sealed_routeback_credential_file",
    }
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    assert receipt["attestation_sha256"] == runtime._sha256_json(
        {
            key: value
            for key, value in receipt.items()
            if key != "attestation_sha256"
        }
    )
    serialized = json.dumps(receipt)
    assert "must-not-enter-the-attestation" not in serialized
    assert "token" not in serialized
    assert calls == [
        (
            "factory",
            runtime.DEFAULT_EDGE_TOKEN_PATH,
            {
                "credentials_directory": runtime.DEFAULT_EDGE_TOKEN_DIRECTORY,
                "expected_owner_uid": full.identities.edge_uid,
                "timeout_seconds": 5.0,
            },
        ),
        ("current_user", 5.0),
        ("close",),
    ]
    assert runtime._require_routeback_credential_binding(
        plan, full, receipt
    ) == metadata


def test_routeback_credential_metadata_requires_root_sealed_parent(
    monkeypatch,
    tmp_path,
):
    full = _full_plan()
    directory = tmp_path / "discord-edge-credentials"
    credential_path = directory / "bot-token"
    directory.mkdir()
    credential_path.write_bytes(b"opaque-placeholder-not-read-by-this-test")
    directory_uid = [0]
    file_metadata = _routeback_file_metadata()
    real_lstat = os.lstat

    def fake_lstat(path):
        if Path(path) == directory:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o750,
                st_uid=directory_uid[0],
            )
        if Path(path) == credential_path:
            return SimpleNamespace(
                st_mode=stat.S_IFREG | file_metadata["mode"],
                st_nlink=1,
                st_uid=file_metadata["uid"],
                st_gid=file_metadata["gid"],
                st_size=file_metadata["size"],
                st_dev=file_metadata["device"],
                st_ino=file_metadata["inode"],
                st_mtime_ns=file_metadata["mtime_ns"],
                st_ctime_ns=file_metadata["ctime_ns"],
            )
        return real_lstat(path)

    monkeypatch.setattr(runtime, "DEFAULT_EDGE_TOKEN_DIRECTORY", directory)
    monkeypatch.setattr(runtime, "DEFAULT_EDGE_TOKEN_PATH", credential_path)
    monkeypatch.setattr(runtime.os, "lstat", fake_lstat)

    assert runtime._routeback_credential_file_metadata(full) == file_metadata
    directory_uid[0] = full.identities.edge_uid
    with pytest.raises(RuntimeError, match="directory is writable by the edge"):
        runtime._routeback_credential_file_metadata(full)


@pytest.mark.parametrize("failure_kind", ("mismatch", "unavailable"))
def test_routeback_identity_failure_prevents_edge_start(
    monkeypatch,
    failure_kind,
):
    plan = _plan()
    full = _full_plan()
    calls = []
    original_attestor = runtime._attest_live_routeback_bot_identity

    class FakeApi:
        def current_user(self, *, timeout_seconds):
            calls.append(("current_user", timeout_seconds))
            if failure_kind == "unavailable":
                raise RuntimeError("Discord API unavailable")
            return {
                "id": runtime.PRODUCTION_DISCORD_BOT_USER_ID,
                "bot": True,
            }

    class FakeAdapter:
        def __init__(self):
            self._api = FakeApi()

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: _routeback_file_metadata(),
    )
    monkeypatch.setattr(
        runtime,
        "_attest_live_routeback_bot_identity",
        lambda candidate, foundation: original_attestor(
            candidate,
            foundation,
            adapter_factory=lambda *_args, **_kwargs: FakeAdapter(),
        ),
    )
    lifecycle = runtime.CapabilityCanaryLifecycle(
        plan,
        full,
        runner=lambda command: calls.append(("systemd", command.argv)),
    )

    with pytest.raises(RuntimeError):
        lifecycle._start_routeback_edge()

    assert not any(call[0] == "systemd" for call in calls)
    assert calls[-1] == ("close",)


def test_routeback_credential_swap_after_start_stops_edge(monkeypatch):
    plan = _plan()
    full = _full_plan()
    original_metadata = _routeback_file_metadata(inode=41)
    swapped_metadata = _routeback_file_metadata(inode=42)

    class FakeApi:
        def current_user(self, *, timeout_seconds):
            assert timeout_seconds == 5.0
            return {"id": plan.routeback_bot_user_id, "bot": True}

    class FakeAdapter:
        _api = FakeApi()

        def close(self):
            return None

    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: dict(original_metadata),
    )
    identity = runtime._attest_live_routeback_bot_identity(
        plan,
        full,
        adapter_factory=lambda *_args, **_kwargs: FakeAdapter(),
    )
    observations = iter((original_metadata, swapped_metadata))
    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: dict(next(observations)),
    )
    monkeypatch.setattr(
        runtime,
        "_attest_live_routeback_bot_identity",
        lambda _plan, _full: identity,
    )
    commands = []

    def runner(command):
        commands.append(command.argv)
        return runtime.subprocess.CompletedProcess(
            command.argv,
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    lifecycle = runtime.CapabilityCanaryLifecycle(plan, full, runner=runner)
    with pytest.raises(RuntimeError, match="credential object changed"):
        lifecycle._start_routeback_edge()

    assert commands == [
        runtime.edge_start_command().argv,
        (runtime.SYSTEMCTL, "stop", runtime.EDGE_UNIT_NAME),
    ]


def test_plan_publication_authority_is_exact_and_digest_bound():
    plan = _plan()
    authority = _plan_publication_authority(plan)

    assert runtime.validate_plan_publication_authority(authority) == authority
    assert runtime.build_plan_from_publication_authority(
        authority, _full_plan()
    ) == plan

    tampered = json.loads(json.dumps(authority))
    tampered["inputs"]["identities"]["browser_uid"] += 1
    with pytest.raises(ValueError, match="self-digest drifted"):
        runtime.validate_plan_publication_authority(tampered)

    wrong_digest = json.loads(json.dumps(authority))
    wrong_digest["plan_sha256"] = "e" * 64
    unsigned = {
        key: value
        for key, value in wrong_digest.items()
        if key != "authority_sha256"
    }
    wrong_digest["authority_sha256"] = runtime._sha256_json(unsigned)
    with pytest.raises(PermissionError, match="approved capability plan digest"):
        runtime.build_plan_from_publication_authority(
            wrong_digest, _full_plan()
        )


@pytest.mark.parametrize("locked_channel_id", sorted(runtime.LOCKED_NONPUBLIC_CHANNEL_IDS))
def test_plan_publication_authority_rejects_each_locked_discord_channel(
    locked_channel_id,
):
    authority = _plan_publication_authority(_plan())
    authority["inputs"]["discord"]["allowed_channel_ids"] = [
        locked_channel_id
    ]
    unsigned = {
        key: value
        for key, value in authority.items()
        if key != "authority_sha256"
    }
    authority["authority_sha256"] = runtime._sha256_json(unsigned)

    with pytest.raises(ValueError, match="public Discord target is invalid"):
        runtime.validate_plan_publication_authority(authority)


def test_bitrix_preplan_bootstrap_retires_both_key_halves_and_reboots_cleanly(
    tmp_path,
):
    full = _full_plan()
    authority = _bitrix_foundation_authority(full)
    private_path = tmp_path / "keys/bitrix-private.pem"
    public_path = tmp_path / "trust/bitrix-public.pem"
    writer_path = tmp_path / "keys/writer-public.pem"
    writer_path.parent.mkdir(parents=True)
    writer_key = runtime.Ed25519PrivateKey.generate().public_key()
    writer_path.write_bytes(runtime._ed25519_public_pem(writer_key))
    os.chown(writer_path, os.geteuid(), os.getegid())
    writer_path.chmod(0o444)

    identity = {
        "service_user": "muncho-edge-bitrix",
        "service_group": "muncho-edge-bitrix",
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_group": "muncho-edge-bitrix-c",
        "socket_client_gid": 2211,
        "state": "present_exact",
    }

    def observe_identity(**_kwargs):
        return dict(identity)

    def verify_assets(**_kwargs):
        return {
            "manifest_sha256": "b" * 64,
            "verification_sha256": "c" * 64,
            "files": [
                {"asset_id": asset_id}
                for asset_id in runtime.BITRIX_OPERATIONAL_EDGE_ASSET_IDS.values()
            ],
        }

    arguments = {
        "full_plan": full,
        "runner": lambda command: runtime.subprocess.CompletedProcess(
            command.argv, 0, b"", b""
        ),
        "identity_observer": observe_identity,
        "asset_verifier": verify_assets,
        "private_key_path": private_path,
        "public_key_path": public_path,
        "writer_public_key_path": writer_path,
        "identity_receipt_path": tmp_path / "state/identity.json",
        "foundation_root": tmp_path / "state/foundations",
        "key_bootstrap_root": tmp_path / "state/key-bootstraps",
        "require_root": False,
    }
    first = runtime.bootstrap_bitrix_foundation(authority, **arguments)
    first_key_id = first["receipt_public_key_id"]
    assert private_path.exists() and public_path.exists()
    assert first["read_peer_uids"] == [
        full.identities.writer_uid,
        2104,
    ]
    assert first["mutation_peer_uid"] == full.identities.writer_uid
    assert '"plan_sha256":' not in json.dumps(first)
    key_receipt = runtime._load_lease_artifact(
        Path(first["key_bootstrap_receipt_path"]),
        schema=runtime.CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
    )
    assert key_receipt["retire_private_on_stop"] is True
    assert key_receipt["retire_public_on_stop"] is True
    assert "private_sha256" not in json.dumps(key_receipt)

    retirement = runtime.retire_bitrix_foundation_key_pair(
        key_receipt,
        reason="foundation_expired",
        now_unix=authority["expires_at_unix"],
        retirement_root=tmp_path / "state/key-retirements",
        require_root=False,
    )
    assert retirement["private_absent"] is True
    assert retirement["public_absent"] is True
    assert retirement["both_pair_members_absent"] is True
    assert not private_path.exists() and not public_path.exists()

    second = runtime.bootstrap_bitrix_foundation(authority, **arguments)
    assert private_path.exists() and public_path.exists()
    assert second["receipt_public_key_id"] != first_key_id
    assert "private_sha256" not in json.dumps(second)


def test_bitrix_foundation_authority_forbids_capability_plan_self_reference():
    authority = _bitrix_foundation_authority(_full_plan())
    authority["plan_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fields are not exact"):
        runtime.validate_bitrix_foundation_authority(authority)


def _watchdog_stopped_services():
    return {
        unit: {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": 0,
            "FragmentPath": "",
            "DropInPaths": "",
            "Type": "",
            "NotifyAccess": "",
            "StatusText": "",
        }
        for unit in runtime.CAPABILITY_STOP_ORDER
    }


def test_processless_socket_observation_normalizes_only_fixed_empty_properties():
    def result_for(unit, *, drop=()):
        values = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
            "Type": "",
            "NotifyAccess": "none",
            "StatusText": "",
        }
        for name in drop:
            values.pop(name)
        stdout = "".join(f"{name}={value}\n" for name, value in values.items())
        return runtime.subprocess.CompletedProcess([], 0, stdout.encode(), b"")

    socket = runtime.DEFAULT_WORKER_SOCKET_UNIT_NAME
    socket_missing = tuple(runtime._PROCESSLESS_UNIT_PROPERTY_DEFAULTS[socket])
    state = runtime.collect_capability_service_state(
        socket,
        runner=lambda _command: result_for(socket, drop=socket_missing),
    )
    assert state["MainPID"] == 0
    assert state["Type"] == ""
    assert state["NotifyAccess"] == ""
    assert state["StatusText"] == ""

    service = runtime.DEFAULT_WORKER_SERVICE_UNIT_NAME
    with pytest.raises(RuntimeError, match="fields are not exact"):
        runtime.collect_capability_service_state(
            service,
            runner=lambda _command: result_for(service, drop=("MainPID",)),
        )

    unexpected = result_for(socket)
    unexpected = runtime.subprocess.CompletedProcess(
        unexpected.args,
        unexpected.returncode,
        unexpected.stdout + b"Unexpected=x\n",
        unexpected.stderr,
    )
    with pytest.raises(RuntimeError, match="fields are not exact"):
        runtime.collect_capability_service_state(
            socket,
            runner=lambda _command: unexpected,
        )


def test_expiry_watchdog_is_absolute_persistent_and_idempotent(tmp_path):
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    arguments = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "authority_sha256": "d" * 64,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    first = runtime.arm_capability_expiry_watchdog(**arguments)
    second = runtime.arm_capability_expiry_watchdog(**arguments)
    assert second == first
    assert first["cleanup_at_unix"] == first["expires_at_unix"]
    paths = runtime._expiry_watchdog_paths(
        first["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    service = Path(paths["service_path"]).read_text("ascii")
    timer = Path(paths["timer_path"]).read_text("ascii")
    assert "Restart=on-failure\n" in service
    assert "expiry-cleanup --watchdog-id " + first["watchdog_id"] in service
    assert "OnCalendar=@1800000000\n" in timer
    assert "Persistent=true\n" in timer
    assert Path(paths["timer_wants_path"]).is_symlink()
    assert os.readlink(paths["timer_wants_path"]) == (
        "../" + str(paths["timer_name"])
    )
    assert all("enable" not in argv for argv in calls)
    assert calls.count((runtime.SYSTEMCTL, "start", paths["timer_name"])) == 2


def test_expiry_watchdog_retires_unjournaled_bitrix_pair_and_disarms(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", tmp_path / "no-plan.json")
    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    armed = runtime.arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        release_artifact_sha256="c" * 64,
        interpreter=Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        expires_at_unix=2_000,
        authority_sha256="d" * 64,
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=False,
        now_unix=1_000,
    )
    private_path = tmp_path / "keys/private.pem"
    public_path = tmp_path / "trust/public.pem"
    runtime._stage_bitrix_receipt_key_pair(
        private_path=private_path,
        public_path=public_path,
        require_root=False,
    )
    credential_paths = {
        binding: tmp_path / "credentials" / binding
        for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
    }
    completion = runtime.run_capability_expiry_cleanup(
        armed["watchdog_id"],
        runner=runner,
        now_unix=2_000,
        state_root=state_root,
        systemd_root=systemd_root,
        credential_paths=credential_paths,
        private_key_path=private_path,
        public_key_path=public_path,
        key_bootstrap_root=tmp_path / "key-bootstraps",
        key_retirement_root=tmp_path / "key-retirements",
        service_observer=lambda **_kwargs: _watchdog_stopped_services(),
        require_root=False,
    )
    assert completion["ok"] is True
    assert completion["all_six_credentials_absent_readback"] is True
    assert completion["bitrix_pair_absent"] is True
    assert completion["bitrix_key_pair_cleanup"][
        "private_content_or_digest_recorded"
    ] is False
    assert not private_path.exists() and not public_path.exists()
    assert not list(systemd_root.glob("muncho-capability-canary-expiry-*"))
    attempted = [
        argv[-1]
        for argv in calls
        if argv[:2] == (runtime.SYSTEMCTL, "stop")
        and argv[-1] in runtime.CAPABILITY_STOP_ORDER
    ]
    assert attempted == list(runtime.CAPABILITY_STOP_ORDER)


def test_expiry_watchdog_failure_is_retryable_and_attempts_every_stop(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", tmp_path / "no-plan.json")
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"
    armed = runtime.arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        release_artifact_sha256="c" * 64,
        interpreter=Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        expires_at_unix=2_000,
        authority_sha256="d" * 64,
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=False,
        now_unix=1_000,
    )
    states = _watchdog_stopped_services()
    states[runtime.GATEWAY_UNIT_NAME]["ActiveState"] = "active"
    credential_paths = {
        binding: tmp_path / "credentials" / binding
        for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
    }
    with pytest.raises(BaseExceptionGroup, match="will retry"):
        runtime.run_capability_expiry_cleanup(
            armed["watchdog_id"],
            runner=runner,
            now_unix=2_000,
            state_root=state_root,
            systemd_root=systemd_root,
            credential_paths=credential_paths,
            private_key_path=tmp_path / "keys/private.pem",
            public_key_path=tmp_path / "trust/public.pem",
            key_bootstrap_root=tmp_path / "key-bootstraps",
            key_retirement_root=tmp_path / "key-retirements",
            service_observer=lambda **_kwargs: states,
            require_root=False,
        )
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=state_root,
        systemd_root=systemd_root,
    )
    assert not os.path.lexists(paths["completion"])
    assert os.path.lexists(paths["timer_path"])
    attempted = [
        argv[-1]
        for argv in calls
        if argv[:2] == (runtime.SYSTEMCTL, "stop")
        and argv[-1] in runtime.CAPABILITY_STOP_ORDER
    ]
    assert attempted == list(runtime.CAPABILITY_STOP_ORDER)


def test_publish_plan_is_atomic_idempotent_and_tamper_evident(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    authority = _plan_publication_authority(plan)
    plan_path = tmp_path / "etc/muncho/capability-canary/runtime-plan.json"
    receipt_root = tmp_path / "state/plan-publications"

    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", plan_path)
    monkeypatch.setattr(
        runtime,
        "DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT",
        receipt_root,
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "load_full_canary_plan", _full_plan)
    monkeypatch.setattr(
        runtime,
        "build_plan_from_publication_authority",
        lambda value, full: (
            plan
            if value == authority and full.sha256 == plan.full_canary_plan_sha256
            else (_ for _ in ()).throw(AssertionError("unexpected plan authority"))
        ),
    )
    monkeypatch.setattr(
        runtime,
        "validate_dedicated_canary_host",
        lambda _plan: {"dedicated": True},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_release_manifest",
        lambda _plan: {"release": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "runtime_dependency_manifest_preflight",
        lambda _plan: {"runtime": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "browser_executable_preflight",
        lambda _plan: {"browser": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "worker_executables_preflight",
        lambda _plan: {"worker": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "validate_bitrix_foundation_for_plan",
        lambda _plan, _full: {"bitrix_foundation": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "collect_service_state",
        lambda unit: {"unit": unit, "state": "stopped"},
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_service_states",
        lambda _states, *, phase: {"all_stopped": phase == "stopped"},
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", contextlib.nullcontext)

    def publish(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read(path: Path, *, maximum: int) -> bytes:
        item = path.stat()
        assert stat.S_IMODE(item.st_mode) == 0o400
        raw = path.read_bytes()
        assert len(raw) <= maximum
        return raw

    monkeypatch.setattr(runtime, "_atomic_publish_root_file", publish)
    monkeypatch.setattr(runtime, "_read_published_plan_file", read)

    first = runtime.publish_capability_plan(authority)
    receipt_path = Path(first["receipt_path"])
    assert plan_path.read_bytes() == runtime._canonical_bytes(plan.to_mapping())
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert runtime.publish_capability_plan(authority) == first

    receipt = json.loads(receipt_path.read_text())
    receipt["owner_subject_sha256"] = "f" * 64
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(runtime._canonical_bytes(receipt))
    receipt_path.chmod(0o400)
    with pytest.raises(RuntimeError, match="receipt drifted"):
        runtime.publish_capability_plan(authority)


def test_publish_plan_blocks_incomplete_prior_publication(monkeypatch, tmp_path):
    plan = _plan()
    authority = _plan_publication_authority(plan)
    plan_path = tmp_path / "runtime-plan.json"
    plan_path.write_bytes(runtime._canonical_bytes(plan.to_mapping()))
    plan_path.chmod(0o400)

    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", plan_path)
    monkeypatch.setattr(
        runtime,
        "DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT",
        tmp_path / "receipts",
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "load_full_canary_plan", _full_plan)
    monkeypatch.setattr(
        runtime,
        "build_plan_from_publication_authority",
        lambda value, full: (
            plan
            if value == authority and full.sha256 == plan.full_canary_plan_sha256
            else (_ for _ in ()).throw(AssertionError("unexpected plan authority"))
        ),
    )
    monkeypatch.setattr(runtime, "validate_dedicated_canary_host", lambda _p: {})
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _p: {})
    monkeypatch.setattr(
        runtime, "runtime_dependency_manifest_preflight", lambda _p: {}
    )
    monkeypatch.setattr(runtime, "browser_executable_preflight", lambda _p: {})
    monkeypatch.setattr(runtime, "worker_executables_preflight", lambda _p: {})
    monkeypatch.setattr(
        runtime,
        "validate_bitrix_foundation_for_plan",
        lambda _plan, _full: {},
    )
    monkeypatch.setattr(
        runtime, "collect_service_state", lambda unit: {"unit": unit}
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_service_states",
        lambda _states, *, phase: {"stopped": phase == "stopped"},
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", contextlib.nullcontext)

    with pytest.raises(RuntimeError, match="publication is incomplete"):
        runtime.publish_capability_plan(authority)


def test_gateway_config_rejects_any_unapproved_semantic_extension():
    original = yaml.safe_load(runtime.render_gateway_config(_plan()))
    mutations = (
        lambda value: value.__setitem__("semantic_dispatcher", {"enabled": True}),
        lambda value: value["plugins"]["enabled"].append("rewrite_plugin"),
        lambda value: value["plugins"].__setitem__("autoload", True),
        lambda value: value["hooks"].__setitem__(
            "pre_tool_call", {"command": "/tmp/rewrite"}
        ),
        lambda value: value.__setitem__(
            "auxiliary", {"goal_judge": {"provider": "auto"}}
        ),
        lambda value: value.__setitem__("goals", {"max_turns": 20}),
    )
    for mutate in mutations:
        candidate = json.loads(json.dumps(original))
        mutate(candidate)
        with pytest.raises(ValueError):
            runtime.validate_capability_gateway_config(candidate)


def _extension_surface(plan: runtime.CapabilityCanaryPlan):
    plugin_instance = object()
    callbacks = {}
    for hook_name in runtime.CAPABILITY_OBSERVER_HOOKS:
        def callback(_self, _hook_name=hook_name):
            return _hook_name

        callback.__name__ = hook_name
        callback.__module__ = "hermes_plugins.muncho_canary_evidence"
        callbacks[hook_name] = [
            callback.__get__(plugin_instance, type(plugin_instance))
        ]
    module = SimpleNamespace(
        __name__="hermes_plugins.muncho_canary_evidence",
        __file__=str(
            plan.release_root
            / "plugins"
            / runtime.CAPABILITY_OBSERVER_PLUGIN
            / "__init__.py"
        ),
        _PLUGIN=plugin_instance,
    )
    manifest = SimpleNamespace(
        name=runtime.CAPABILITY_OBSERVER_PLUGIN,
        key=runtime.CAPABILITY_OBSERVER_PLUGIN,
        kind="standalone",
        source="bundled",
        path=str(
            plan.release_root / "plugins" / runtime.CAPABILITY_OBSERVER_PLUGIN
        ),
        provides_tools=[],
        provides_hooks=list(runtime.CAPABILITY_OBSERVER_HOOKS),
    )
    loaded = SimpleNamespace(
        manifest=manifest,
        module=module,
        tools_registered=[],
        hooks_registered=list(runtime.CAPABILITY_OBSERVER_HOOKS),
        middleware_registered=[],
        commands_registered=[],
        enabled=True,
        error=None,
        deferred=False,
    )
    manager = SimpleNamespace(
        _discovered=True,
        _isolated_allowlist=frozenset({runtime.CAPABILITY_OBSERVER_PLUGIN}),
        _isolated_discovery_failure=None,
        _plugins={runtime.CAPABILITY_OBSERVER_PLUGIN: loaded},
        _hooks=callbacks,
        _middleware={},
        _plugin_tool_names=set(),
        _plugin_platform_names=set(),
        _cli_commands={},
        _plugin_commands={},
        _context_engine=None,
        _plugin_skills={},
        _aux_tasks={},
        _slack_action_handlers=[],
        _cli_ref=None,
    )
    gateway_hooks = SimpleNamespace(_handlers={}, _loaded_hooks=[])
    return manager, gateway_hooks


def test_extension_surface_allows_only_the_sealed_observer():
    plan = _plan()
    manager, gateway_hooks = _extension_surface(plan)
    runtime.validate_capability_extension_surface(
        manager, gateway_hooks, plan=plan
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "general_discovery",
        "extra_plugin",
        "middleware",
        "tool",
        "command",
        "context_engine",
        "auxiliary_task",
        "callback_substitution",
        "gateway_hook",
        "module_substitution",
    ),
)
def test_extension_surface_fails_closed_on_behavior_changing_state(mutation):
    plan = _plan()
    manager, gateway_hooks = _extension_surface(plan)
    if mutation == "general_discovery":
        manager._isolated_allowlist = None
    elif mutation == "extra_plugin":
        manager._plugins["extra"] = manager._plugins[
            runtime.CAPABILITY_OBSERVER_PLUGIN
        ]
    elif mutation == "middleware":
        manager._middleware["llm_request"] = [lambda **_kwargs: {}]
    elif mutation == "tool":
        manager._plugin_tool_names.add("semantic_router")
    elif mutation == "command":
        manager._plugin_commands["dispatch"] = {"handler": lambda: None}
    elif mutation == "context_engine":
        manager._context_engine = object()
    elif mutation == "auxiliary_task":
        manager._aux_tasks["goal_judge"] = {"provider": "auto"}
    elif mutation == "callback_substitution":
        manager._hooks["pre_api_request"] = [lambda **_kwargs: None]
    elif mutation == "gateway_hook":
        gateway_hooks._handlers["agent:start"] = [lambda *_args: None]
    else:
        manager._plugins[
            runtime.CAPABILITY_OBSERVER_PLUGIN
        ].module.__file__ = str(plan.release_root / "substituted.py")
    with pytest.raises(RuntimeError):
        runtime.validate_capability_extension_surface(
            manager, gateway_hooks, plan=plan
        )


def test_gateway_and_mac_units_keep_secrets_outside_gateway():
    plan = _plan()
    gateway = runtime.render_gateway_unit(plan)
    mac = runtime.render_mac_ops_unit(plan)
    browser = runtime.render_browser_unit(plan)
    worker_socket = runtime.render_worker_socket_unit(plan)
    worker_service = runtime.render_worker_service_unit(plan)

    assert "-m gateway.run" in gateway
    assert "--require-capability-canary" in gateway
    assert "127.0.0.1" not in gateway  # API host remains sealed config, not argv.
    assert "EnvironmentFile=" not in gateway
    assert "DISCORD_BOT_TOKEN=" not in gateway
    assert "GITLAB_TOKEN=" not in gateway
    assert "UnsetEnvironment=" in gateway
    assert "DISCORD_TOKEN" in gateway
    assert "GITLAB_BASE_URL" in gateway
    assert "GITLAB_TOKEN" in gateway
    assert f"InaccessiblePaths={runtime.DEFAULT_MAC_OPS_CREDENTIAL_DIR}" in gateway
    assert f"BindReadOnlyPaths={runtime.DEFAULT_GATEWAY_AUTH_STORE}" in gateway
    assert f"InaccessiblePaths={runtime.DEFAULT_GATEWAY_AUTH_STORE}" not in gateway
    assert "GATEWAY_RELAY_PLATFORMS=discord" in gateway
    assert "TERMINAL_ENV=isolated_worker" in gateway
    assert f"TERMINAL_ISOLATED_WORKER_SOCKET={runtime.DEFAULT_WORKER_SOCKET}" in gateway
    assert runtime.DEFAULT_WORKER_CLIENT_GROUP in gateway
    assert runtime.DEFAULT_WORKER_SERVICE_UNIT_NAME in gateway
    assert runtime.DEFAULT_WORKER_SOCKET_UNIT_NAME in gateway
    assert "AGENT_BROWSER_EXECUTABLE_PATH=" not in gateway
    assert f"InaccessiblePaths={plan.browser_executable}" in gateway
    assert f"AssertPathExists={plan.browser_executable}" not in gateway
    assert f"ReadOnlyPaths={runtime.DEFAULT_BROWSER_CONFIG}" in gateway
    assert f"ReadOnlyPaths={runtime.DEFAULT_BROWSER_SOCKET.parent}" in gateway
    assert "/usr/bin/chromium" not in gateway
    assert "/srv/" not in gateway
    assert "docker" not in gateway.lower()
    assert "BROWSER_CDP_URL" not in gateway
    assert "remote-debugging" not in gateway
    assert "9222" not in gateway
    assert f"BindsTo={runtime.WRITER_UNIT_NAME}" in gateway
    assert "gateway.mac_ops_edge_service" in mac
    assert f"ReadOnlyPaths={runtime.DEFAULT_MAC_OPS_CREDENTIAL}" in mac
    assert f"User={plan.identities.mac_ops_user}" in mac
    assert "UnsetEnvironment=" in mac
    assert "OPENAI_API_KEY" in mac
    assert f"User={plan.identities.browser_user}" in browser
    assert f"Group={plan.identities.browser_group}" in browser
    assert f"# PrincipalUID={plan.identities.browser_uid}" in browser
    assert f"# PrincipalGID={plan.identities.browser_gid}" in browser
    assert plan.identities.gateway_user not in browser
    assert "--no-sandbox" not in browser
    assert "RestrictNamespaces" not in browser
    assert "Type=notify" in browser
    assert "NotifyAccess=main" in browser
    assert "gateway.browser_controller" in browser
    assert "remote-debugging" not in browser
    assert "9222" not in browser
    assert f"ListenStream={runtime.DEFAULT_WORKER_SOCKET}" in worker_socket
    assert "PrivateNetwork=yes" in worker_service
    assert "TemporaryFileSystem=" in worker_service
    assert "docker" not in worker_service.lower()


def test_connector_plan_unit_and_six_lease_bindings_are_exact():
    plan = _plan()
    config = json.loads(runtime.render_connector_config(plan))
    unit = runtime.render_connector_unit(plan)

    assert config["service"]["canary_history_reader"] == {
        "service_unit": runtime.CANARY_HISTORY_READER_SERVICE_UNIT,
        "service_user": runtime.CANARY_HISTORY_READER_SERVICE_USER,
        "requester_user_id": runtime.CANARY_REQUESTER_USER_ID,
    }
    assert config["discord"]["allowed_guild_ids"] == ["1282725267068157972"]
    assert config["discord"]["allowed_channel_ids"] == [
        runtime.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID
    ]
    assert config["discord"]["allowed_user_ids"] == ["1279454038731264061"]
    assert config["discord"]["reviewed_cron_history_targets"] == {}
    assert config["discord"]["allow_bot_authors"] is False
    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
    assert "Restart=no" in unit
    assert "RuntimeMaxSec=900s" in unit
    assert "DISCORD_BOT_TOKEN=" not in unit
    assert set(plan.to_mapping()["credential_bindings"]) == set(
        runtime.CAPABILITY_CREDENTIAL_BINDINGS
    )
    assert tuple(runtime.CAPABILITY_START_ORDER) == (
        runtime.PHASE_B_READINESS_UNIT_NAME,
        runtime.EDGE_UNIT_NAME,
        runtime.DEFAULT_DISCORD_CONNECTOR_UNIT,
        runtime.MAC_OPS_UNIT_NAME,
        runtime.DEFAULT_WORKER_SOCKET_UNIT_NAME,
        runtime.DEFAULT_WORKER_SERVICE_UNIT_NAME,
        runtime.DEFAULT_BROWSER_UNIT_NAME,
        runtime.WRITER_UNIT_NAME,
        runtime.BITRIX_OPERATIONAL_EDGE_UNIT,
        *(
            runtime.CAPABILITY_PRODUCER_SERVICE_UNITS[role]
            for role in runtime.CAPABILITY_PRODUCER_ROLES
        ),
        runtime.GATEWAY_UNIT_NAME,
    )
    assert tuple(runtime.CAPABILITY_STOP_ORDER) == (
        *runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER,
        runtime.CAPABILITY_OBSERVER_UNIT,
    )
    assert runtime.CAPABILITY_OBSERVER_UNIT not in (
        runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER
    )


@pytest.mark.parametrize("failure_index", range(len(runtime.CAPABILITY_STOP_ORDER)))
def test_cleanup_attempts_all_stops_after_each_injected_failure(
    failure_index,
):
    attempted = []

    def stop(unit):
        attempted.append(unit)
        if len(attempted) - 1 == failure_index:
            raise RuntimeError("injected stop failure")

    stopped, errors = runtime._attempt_capability_stop_order(stop)

    assert attempted == list(runtime.CAPABILITY_STOP_ORDER)
    assert len(errors) == 1
    assert len(stopped) == len(runtime.CAPABILITY_STOP_ORDER) - 1


def test_expired_capability_approval_causes_zero_mutations(monkeypatch):
    plan = _plan()
    approval = runtime.CapabilityCanaryOwnerApproval.from_mapping(
        {
            "schema": runtime.CAPABILITY_APPROVAL_SCHEMA,
            "scope": "production_capability_canary_runtime_start",
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
            "cryptographic_owner_proof": False,
            "owner_subject_sha256": "1" * 64,
            "approval_source_sha256": "2" * 64,
            "stopped_preflight_state_sha256": "4" * 64,
            "nonce_sha256": "3" * 64,
            "approved_at_unix": 1,
            "expires_at_unix": 61,
        }
    )
    mutations = []
    lifecycle = runtime.CapabilityCanaryLifecycle(
        plan,
        _full_plan(),
        runner=lambda command: mutations.append(command),
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)

    with pytest.raises(PermissionError, match="does not authorize"):
        lifecycle.start(approval, object.__new__(FullCanaryOwnerApproval))
    assert mutations == []


def test_approval_install_is_exclusive_nonce_journaled_and_retired(
    tmp_path, monkeypatch
):
    plan = _plan()
    full = _full_plan()
    now = int(time.time())
    approval = runtime.CapabilityCanaryOwnerApproval.from_mapping(
        {
            "schema": runtime.CAPABILITY_APPROVAL_SCHEMA,
            "scope": "production_capability_canary_runtime_start",
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
            "cryptographic_owner_proof": False,
            "owner_subject_sha256": "1" * 64,
            "approval_source_sha256": "2" * 64,
            "stopped_preflight_state_sha256": "4" * 64,
            "nonce_sha256": "3" * 64,
            "approved_at_unix": now - 1,
            "expires_at_unix": now + 299,
        }
    )
    approval_path = tmp_path / "etc/owner-approval.json"
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_PATH", approval_path)
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "validate_dedicated_canary_host", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _p: {})
    monkeypatch.setattr(
        runtime,
        "collect_capability_preflight",
        lambda *_a, **_k: {
            "report_sha256": "6" * 64,
            "state_sha256": "4" * 64,
            "ok": True,
        },
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        runtime,
        "_ensure_root_directory",
        lambda path: Path(path).mkdir(parents=True, exist_ok=True),
    )

    def write_exclusive(path, payload, *, mode):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, payload)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    def read_as_root(path, **_kwargs):
        path = Path(path)
        raw = path.read_bytes()
        item = path.stat()
        return raw, SimpleNamespace(
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
            st_ctime_ns=item.st_ctime_ns,
        )

    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write_exclusive)
    monkeypatch.setattr(runtime, "_read_stable_file", read_as_root)

    stale_value = dict(approval.value)
    stale_value["stopped_preflight_state_sha256"] = "5" * 64
    stale = runtime.CapabilityCanaryOwnerApproval.from_mapping(stale_value)
    with pytest.raises(PermissionError, match="current stopped preflight"):
        runtime.install_capability_approval(plan, full, stale)
    assert not approval_path.exists()

    receipt = runtime.install_capability_approval(plan, full, approval)
    assert approval_path.exists()
    assert receipt["approval_sha256"] == approval.sha256
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    retirement = runtime._remove_installed_capability_approval(plan, full)
    assert retirement["removed"] is True
    assert not approval_path.exists()
    with pytest.raises(PermissionError, match="nonce was already consumed"):
        runtime.install_capability_approval(plan, full, approval)


def test_connector_cleanup_requires_no_unacked_or_unresolved_work():
    safe = {
        "schema": "discord-public-connector-cleanup-snapshot.v1",
        "event_state_counts": {"acked": 2},
        "send_state_counts": {"verified": 1, "blocked": 1},
        "unresolved_dispatch_count": 0,
        "unacked_event_count": 0,
        "safe_to_retire": True,
    }
    assert runtime._connector_cleanup_snapshot_is_safe(safe)
    for event_state in ("pending", "delivering"):
        unsafe = dict(safe)
        unsafe["event_state_counts"] = {event_state: 1}
        unsafe["unacked_event_count"] = 1
        unsafe["safe_to_retire"] = False
        assert not runtime._connector_cleanup_snapshot_is_safe(unsafe)
    for send_state in ("prepared", "dispatching", "uncertain"):
        unsafe = dict(safe)
        unsafe["send_state_counts"] = {send_state: 1}
        unsafe["unresolved_dispatch_count"] = 1
        unsafe["safe_to_retire"] = False
        assert not runtime._connector_cleanup_snapshot_is_safe(unsafe)


def test_secret_frames_are_plan_bound_and_never_include_refresh_token():
    now = int(time.time())
    token = _jwt(now + 3_600)
    frame = runtime.build_secret_lease_frame(
        kind="codex_access_token",
        secret=token,
        plan_sha256="a" * 64,
        owner_subject_sha256="b" * 64,
        now_unix=now,
        lease_id="c" * 32,
    )
    metadata, decoded = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="codex_access_token",
        now_unix=now,
    )
    assert metadata["plan_sha256"] == "a" * 64
    assert decoded == token
    assert b"refresh_token" not in frame

    # The opaque secret intentionally has no digest (including in receipts).
    # Framing still rejects truncation/trailing-byte substitution.
    frame.append(0)
    with pytest.raises(ValueError):
        runtime.read_secret_lease_frame(
            io.BytesIO(frame),
            expected_kind="codex_access_token",
            now_unix=now,
        )


def test_codex_provision_and_retirement_receipts_are_secret_free(
    tmp_path, monkeypatch
):
    plan = _plan()
    now = int(time.time())
    token = _jwt(now + 3_600)
    frame = runtime.build_secret_lease_frame(
        kind="codex_access_token",
        secret=token,
        plan_sha256=plan.sha256,
        owner_subject_sha256="f" * 64,
        now_unix=now,
        lease_id="9" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame), expected_kind="codex_access_token", now_unix=now
    )
    auth_path = tmp_path / "profile/auth.json"
    journal = tmp_path / "control/codex.json"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        auth_path=auth_path,
        journal_path=journal,
    )
    store = json.loads(auth_path.read_text())
    entry = store["credential_pool"]["openai-codex"][0]
    assert entry["access_token"]
    assert "refresh_token" not in entry
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    assert bytes(token) not in json.dumps(receipt).encode()
    assert secret == bytearray(len(secret))

    monkeypatch.setenv("HERMES_HOME", str(auth_path.parent))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].access_token == store["credential_pool"]["openai-codex"][0][
        "access_token"
    ]
    assert entries[0].refresh_token is None

    retired = runtime.retire_secret_lease(
        kind="codex_access_token",
        target=auth_path,
        journal=journal,
        stop_proof=_stop_proof(plan),
    )
    assert retired["absent"] is True
    assert not auth_path.exists()
    assert retired["secret_digest_recorded"] is False


def test_discord_connector_lease_is_owned_and_secret_free(tmp_path):
    plan = _plan()
    plan = replace(
        plan,
        identities=replace(
            plan.identities,
            connector_uid=os.getuid(),
            connector_gid=os.getgid(),
        ),
    )
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="discord_connector_token",
        secret=bytearray(b"opaque.connector.token"),
        plan_sha256=plan.sha256,
        owner_subject_sha256="6" * 64,
        now_unix=now,
        lease_id="5" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="discord_connector_token",
        now_unix=now,
    )
    target = tmp_path / "connector/bot-token"
    journal = tmp_path / "control/connector.json"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        connector_path=target,
        journal_path=journal,
    )
    assert target.read_bytes() == b"opaque.connector.token"
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert receipt["kind"] == "discord_connector_token"
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    assert b"opaque.connector.token" not in json.dumps(receipt).encode()
    assert secret == bytearray(len(secret))
    retired = runtime.retire_secret_lease(
        kind="discord_connector_token",
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
    )
    assert retired["absent"] is True


@pytest.mark.parametrize(
    "kind,secret,path_argument,identity_fields",
    (
        (
            "api_server_control_key",
            b"generated-api-control-key",
            "api_control_path",
            (),
        ),
        (
            "bitrix_operational_edge_webhook",
            b"https://example.bitrix24.eu/rest/1/opaque/",
            "bitrix_path",
            (),
        ),
        (
            "mac_ops_gitlab_env",
            b"GITLAB_BASE_URL=https://gitlab.example\nGITLAB_TOKEN=opaque\n",
            "mac_path",
            ("mac_ops_uid", "mac_ops_gid"),
        ),
    ),
)
def test_additional_credential_leases_are_append_only_and_secret_free(
    tmp_path,
    kind,
    secret,
    path_argument,
    identity_fields,
):
    plan = _plan()
    if identity_fields:
        plan = replace(
            plan,
            identities=replace(
                plan.identities,
                **{
                    identity_fields[0]: os.getuid(),
                    identity_fields[1]: os.getgid(),
                },
            ),
        )
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind=kind,
        secret=secret,
        plan_sha256=plan.sha256,
        owner_subject_sha256="1" * 64,
        now_unix=now,
        lease_id="2" * 32,
    )
    metadata, decoded = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind=kind,
        now_unix=now,
    )
    target = tmp_path / "credentials" / "secret"
    journal = tmp_path / "control" / f"{kind}-leases"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        decoded,
        journal_path=journal,
        **{path_argument: target},
    )
    assert receipt["credential_binding"] == runtime._CREDENTIAL_BINDING_BY_KIND[kind]
    assert decoded == bytearray(len(decoded))
    assert target.exists()
    lease_root = journal / ("2" * 32)
    assert sorted(path.name for path in lease_root.iterdir()) == [
        "install-intent.json",
        "install-receipt.json",
    ]
    journal_bytes = b"".join(
        path.read_bytes() for path in lease_root.iterdir() if path.is_file()
    )
    assert secret not in journal_bytes
    assert b"secret_digest" in journal_bytes

    completion = runtime.retire_secret_lease(
        kind=kind,
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
        plan=plan,
    )
    assert completion["absent_after_stop"] is True
    assert completion["install_receipt_sha256"] == receipt["receipt_sha256"]
    assert not target.exists()
    assert sorted(path.name for path in lease_root.iterdir()) == [
        "install-intent.json",
        "install-receipt.json",
        "retirement-completion.json",
        "retirement-intent.json",
    ]
    assert runtime.retire_secret_lease(
        kind=kind,
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
        plan=plan,
    ) == completion


def test_partial_prestart_cleanup_attempts_all_six_and_removes_later_active_slots(
    tmp_path,
    monkeypatch,
):
    plan = replace(
        _plan(),
        identities=replace(
            _plan().identities,
            connector_uid=os.getuid(),
            connector_gid=os.getgid(),
        ),
    )
    full = _full_plan()
    monkeypatch.setattr(
        runtime,
        "DEFAULT_LIFECYCLE_RECEIPT_ROOT",
        tmp_path / "lifecycle",
    )
    monkeypatch.setattr(
        runtime,
        "_ensure_root_directory",
        lambda path: Path(path).mkdir(parents=True, exist_ok=True),
    )

    def write_exclusive(path, payload, *, mode):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, payload)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write_exclusive)
    by_kind = {}
    for index, kind in enumerate(
        (
            "api_server_control_key",
            "bitrix_operational_edge_webhook",
            "discord_routeback_token",
            "discord_connector_token",
            "mac_ops_gitlab_env",
            "codex_access_token",
        )
    ):
        mode = 0o600 if kind == "codex_access_token" else 0o400
        maximum = {
            "codex_access_token": 128 * 1024,
            "mac_ops_gitlab_env": runtime._MAX_SECRET_BYTES,
            "discord_connector_token": 512,
        }.get(kind, 8 * 1024)
        by_kind[kind] = runtime._SecretLeaseTarget(
            kind=kind,
            credential_binding=runtime._CREDENTIAL_BINDING_BY_KIND[kind],
            path=tmp_path / f"targets/{index}-{kind}",
            journal=tmp_path / f"journals/{index}-{kind}",
            uid=os.getuid(),
            gid=os.getgid(),
            mode=mode,
            parent_uid=os.getuid(),
            parent_gid=os.getgid(),
            parent_mode=0o700,
            maximum_bytes=maximum,
        )

    now = int(time.time())
    for offset, (kind, secret, path_argument) in enumerate(
        (
            (
                "bitrix_operational_edge_webhook",
                b"https://example.bitrix24.eu/rest/1/opaque/",
                "bitrix_path",
            ),
            (
                "discord_connector_token",
                b"opaque.connector.token",
                "connector_path",
            ),
        )
    ):
        spec = by_kind[kind]
        frame = runtime.build_secret_lease_frame(
            kind=kind,
            secret=secret,
            plan_sha256=plan.sha256,
            owner_subject_sha256="7" * 64,
            now_unix=now,
            lease_id=f"{offset + 1}" * 32,
        )
        metadata, decoded = runtime.read_secret_lease_frame(
            io.BytesIO(frame), expected_kind=kind, now_unix=now
        )
        runtime.provision_secret_lease(
            plan,
            metadata,
            decoded,
            journal_path=spec.journal,
            **{path_argument: spec.path},
        )
        assert spec.path.exists()

    ordered = tuple(
        by_kind[kind]
        for kind in (
            "api_server_control_key",
            "bitrix_operational_edge_webhook",
            "discord_routeback_token",
            "discord_connector_token",
            "mac_ops_gitlab_env",
            "codex_access_token",
        )
    )
    result = runtime.retire_secret_leases_best_effort(
        plan,
        full,
        targets=ordered,
        stop_proof=_stop_proof(plan),
    )

    assert result["ok"] is True
    assert result["all_six_credentials_absent_readback"] is True
    assert result["all_six_install_bound_retirement_completions"] is False
    assert result["slots"]["api_control"]["state"] == "never_installed_absent"
    assert result["slots"]["bitrix_operational_edge_webhook"][
        "state"
    ] == "install_bound_retired"
    assert result["slots"]["discord_public_session_bot_token"][
        "state"
    ] == "install_bound_retired"
    assert all(not spec.path.exists() for spec in ordered)


def test_routeback_lease_parent_is_root_controlled_and_not_edge_writable(
    tmp_path,
):
    full = _full_plan()
    production_spec = runtime._lease_target(
        _plan(full),
        kind="discord_routeback_token",
        full_plan=full,
        routeback_path=tmp_path / "unused/bot-token",
        journal_path=tmp_path / "unused-journal",
    )
    assert production_spec.parent_uid == os.geteuid()
    assert production_spec.parent_gid == full.identities.edge_gid
    assert production_spec.parent_mode == 0o750
    assert production_spec.parent_mode & stat.S_IWGRP == 0
    assert production_spec.uid == full.identities.edge_uid
    assert production_spec.mode == 0o400

    object.__setattr__(
        full,
        "identities",
        replace(
            full.identities,
            edge_uid=os.getuid(),
            edge_gid=os.getgid(),
        ),
    )
    plan = _plan(full)
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="discord_routeback_token",
        secret=b"opaque-routeback-token",
        plan_sha256=plan.sha256,
        owner_subject_sha256="3" * 64,
        now_unix=now,
        lease_id="4" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="discord_routeback_token",
        now_unix=now,
    )
    target = tmp_path / "routeback" / "bot-token"
    journal = tmp_path / "control" / "routeback-leases"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        full_plan=full,
        routeback_path=target,
        journal_path=journal,
    )
    assert receipt["credential_binding"] == (
        "discord_canonical_routeback_bot_token"
    )
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o750
    assert runtime.retire_secret_lease(
        kind="discord_routeback_token",
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
        plan=plan,
    )["absent"] is True


def test_exact_provision_retry_is_idempotent_but_different_secret_fails(tmp_path):
    plan = _plan()
    now = int(time.time())
    arguments = {
        "kind": "api_server_control_key",
        "secret": b"first-generated-control-key",
        "plan_sha256": plan.sha256,
        "owner_subject_sha256": "5" * 64,
        "now_unix": now,
        "lease_id": "6" * 32,
    }
    frame = runtime.build_secret_lease_frame(**arguments)
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    target = tmp_path / "api" / "key"
    journal = tmp_path / "control" / "api-leases"
    first = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    assert runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    ) == first
    different = runtime.build_secret_lease_frame(
        **{**arguments, "secret": b"other-generated-control-key"}
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(different),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    with pytest.raises(RuntimeError, match="different secret bytes"):
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            api_control_path=target,
            journal_path=journal,
        )
    assert secret == bytearray(len(secret))


def test_install_crash_after_intent_recovers_only_with_exact_retry(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="api_server_control_key",
        secret=b"crash-recovery-control-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="7" * 64,
        now_unix=now,
        lease_id="8" * 32,
    )
    target = tmp_path / "api" / "key"
    journal = tmp_path / "control" / "api-leases"
    original = runtime._atomic_no_replace_file
    interrupted = False

    def crash_once(path, payload, **kwargs):
        nonlocal interrupted
        if path == target and not interrupted:
            interrupted = True
            raise OSError("simulated install interruption")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(runtime, "_atomic_no_replace_file", crash_once)
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    with pytest.raises(OSError, match="simulated install interruption"):
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            api_control_path=target,
            journal_path=journal,
        )
    lease_root = journal / ("8" * 32)
    assert (lease_root / "install-intent.json").exists()
    assert not (lease_root / "install-receipt.json").exists()
    assert not target.exists()

    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    recovered = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    assert recovered["state"] == "provisioned"
    assert target.exists()


def test_retirement_crash_after_unlink_recovers_from_bound_intent(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="api_server_control_key",
        secret=b"retirement-recovery-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="9" * 64,
        now_unix=now,
        lease_id="a" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    target = tmp_path / "api" / "key"
    journal = tmp_path / "control" / "api-leases"
    runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    original = runtime._append_lease_artifact
    interrupted = False

    def crash_once(path, *, schema, value):
        nonlocal interrupted
        if schema == runtime.CAPABILITY_RETIREMENT_RECEIPT_SCHEMA and not interrupted:
            interrupted = True
            raise OSError("simulated retirement interruption")
        return original(path, schema=schema, value=value)

    monkeypatch.setattr(runtime, "_append_lease_artifact", crash_once)
    with pytest.raises(OSError, match="simulated retirement interruption"):
        runtime.retire_secret_lease(
            kind="api_server_control_key",
            target=target,
            journal=journal,
            stop_proof=_stop_proof(plan),
            plan=plan,
        )
    assert not target.exists()
    assert (
        journal / ("a" * 32) / "retirement-intent.json"
    ).exists()
    completion = runtime.retire_secret_lease(
        kind="api_server_control_key",
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
        plan=plan,
    )
    assert completion["absent_after_stop"] is True


def test_retirement_time_is_post_stop_and_cannot_reuse_install_ctime(tmp_path):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="api_server_control_key",
        secret=b"post-stop-ordering-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="4" * 64,
        now_unix=now,
        lease_id="d" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    target = tmp_path / "api/key"
    journal = tmp_path / "control/api-leases"
    installed = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    installed_at = installed["installed_at_unix"]
    stale = _stop_proof(plan, observed_at_unix=installed_at - 1)
    with pytest.raises(PermissionError, match="post-install stop proof"):
        runtime.retire_secret_lease(
            kind="api_server_control_key",
            target=target,
            journal=journal,
            stop_proof=stale,
            plan=plan,
            now_unix=installed_at + 10,
        )
    assert target.exists()

    stopped_at = installed_at + 5
    proof = _stop_proof(plan, observed_at_unix=stopped_at)
    completion = runtime.retire_secret_lease(
        kind="api_server_control_key",
        target=target,
        journal=journal,
        stop_proof=proof,
        plan=plan,
        now_unix=stopped_at + 1,
    )
    intent = json.loads(
        (journal / ("d" * 32) / "retirement-intent.json").read_text()
    )
    assert intent["requested_at_unix"] >= stopped_at
    assert intent["service_stop_proof_sha256"] == proof["stop_proof_sha256"]
    assert completion["service_stop_observed_at_unix"] == stopped_at
    assert completion["retired_at_unix"] >= stopped_at
    assert completion["retired_at_unix"] != installed_at
    assert completion["absent_after_stop"] is True


def _services_with_only_cleanup_observer_live():
    services = _watchdog_stopped_services()
    services[runtime.CAPABILITY_OBSERVER_UNIT] = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4242,
        "FragmentPath": str(
            Path("/etc/systemd/system") / runtime.CAPABILITY_OBSERVER_UNIT
        ),
        "DropInPaths": "",
        "Type": "simple",
        "NotifyAccess": "none",
        "StatusText": "",
    }
    return services


def test_credential_consumer_proof_keeps_only_blind_observer_live(monkeypatch):
    plan = _plan()
    services = _services_with_only_cleanup_observer_live()
    monkeypatch.setattr(
        runtime,
        "_producer_credential_inaccessibility_contract",
        lambda: {
            "paths": ["/etc/muncho/keys"],
            "applies_to_roles": list(runtime.CAPABILITY_PRODUCER_ROLES),
            "unit_hash_bound": True,
            "cleanup_observer_has_no_credential_read_access": True,
        },
    )
    proof = runtime.build_credential_consumer_stop_proof(
        plan,
        services,
        producer_foundation={
            "revision": plan.revision,
            "foundation_sha256": "1" * 64,
            "unit_bundle_manifest_sha256": "2" * 64,
            "ready": True,
            "mutation_performed": False,
        },
        observed_at_unix=100,
    )

    assert proof["schema"] == (
        runtime.CAPABILITY_CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA
    )
    assert proof["non_observer_stop_order"] == list(
        runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER
    )
    assert proof["observer_service_unit"] == runtime.CAPABILITY_OBSERVER_UNIT
    assert proof["observer_live_signing_only"] is True
    assert proof["observer_credential_read_access"] is False
    assert runtime._validate_capability_stop_proof(
        plan,
        proof,
        installed_at_unix=90,
        now_unix=110,
    ) == proof
    with pytest.raises(RuntimeError, match="live service"):
        runtime.build_capability_stop_proof(plan, services)

    tampered = json.loads(json.dumps(proof))
    tampered["observer_credential_read_access"] = True
    tampered["stop_proof_sha256"] = runtime._sha256_json(
        {
            key: value
            for key, value in tampered.items()
            if key != "stop_proof_sha256"
        }
    )
    with pytest.raises(PermissionError, match="consumer stop proof"):
        runtime._validate_capability_stop_proof(
            plan,
            tampered,
            installed_at_unix=90,
            now_unix=110,
        )


def test_cleanup_finalization_is_strictly_after_observer_and_activation_stop():
    plan = _plan()
    all_stopped = _watchdog_stopped_services()
    stop = runtime.build_capability_stop_proof(
        plan,
        all_stopped,
        observed_at_unix=200,
    )
    observer = runtime.build_capability_observer_stop_receipt(
        plan,
        all_stopped[runtime.CAPABILITY_OBSERVER_UNIT],
        stopped_at_unix_ms=200_100,
    )
    fleet_unsigned = {
        "schema": "muncho-production-capability-fleet-retirement.v1",
        "readiness_sha256": "1" * 64,
        "foundation_sha256": "2" * 64,
        "release_sha": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": "3" * 64,
        "run_id": "capability-run-1",
        "path": "/run/muncho-capability-canary/producer-activation.json",
        "retired": True,
        "absence_verified": True,
        "retired_at_unix_ms": 200_200,
    }
    fleet = {
        **fleet_unsigned,
        "receipt_sha256": runtime._sha256_json(fleet_unsigned),
    }
    cleanup = {
        "schema": "muncho-production-capability-canary-signed-receipt.v1",
        "authority_role": "gateway_observer",
        "key_id": "4" * 64,
        "signature_algorithm": "ed25519",
        "payload": {
            "run_id": "capability-run-1",
            "fixture_sha256": "3" * 64,
            "observed_at_unix_ms": 200_000,
        },
        "native_evidence": {},
        "signature": "5" * 128,
    }
    finalization = runtime.build_capability_cleanup_finalization(
        plan,
        cleanup_receipt=cleanup,
        observer_stop_receipt=observer,
        service_stop_proof=stop,
        producer_fleet_retirement=fleet,
        producer_activation_absent=True,
        credentials_absent=True,
        bitrix_receipt_key_pair_absent=True,
        full_canary_stopped_preflight_sha256="6" * 64,
        finalized_at_unix_ms=200_300,
    )
    assert finalization["cleanup_receipt_sha256"] == runtime._sha256_json(
        cleanup
    )
    assert finalization["observer_stop_receipt"] == observer
    assert finalization["service_stop_proof"] == stop
    assert finalization["producer_fleet_retirement"] == fleet

    with pytest.raises(ValueError, match="terminal truth"):
        runtime.build_capability_cleanup_finalization(
            plan,
            cleanup_receipt=cleanup,
            observer_stop_receipt=observer,
            service_stop_proof=stop,
            producer_fleet_retirement=fleet,
            producer_activation_absent=True,
            credentials_absent=True,
            bitrix_receipt_key_pair_absent=True,
            full_canary_stopped_preflight_sha256="6" * 64,
            finalized_at_unix_ms=200_199,
        )


def test_append_only_artifact_collision_fails_without_overwrite(tmp_path):
    path = tmp_path / "journal" / ("b" * 32) / "install-intent.json"
    first = runtime._append_lease_artifact(
        path,
        schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        value={
            "operation": "install_intent",
            "lease_id": "b" * 32,
            "marker": "first",
        },
    )
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="collided with different bytes"):
        runtime._append_lease_artifact(
            path,
            schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
            value={
                "operation": "install_intent",
                "lease_id": "b" * 32,
                "marker": "second",
            },
        )
    assert path.read_bytes() == before
    assert json.loads(before)["receipt_sha256"] == first["receipt_sha256"]


def test_secret_install_rejects_dangling_target_and_retirement_substitution(
    tmp_path,
):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="codex_access_token",
        secret=_jwt(now + 3_600),
        plan_sha256=plan.sha256,
        owner_subject_sha256="8" * 64,
        now_unix=now,
        lease_id="7" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame), expected_kind="codex_access_token", now_unix=now
    )
    auth_path = tmp_path / "profile/auth.json"
    auth_path.parent.mkdir(mode=0o700)
    os.chown(
        auth_path.parent,
        plan.identities.gateway_uid,
        plan.identities.gateway_gid,
    )
    auth_path.parent.chmod(0o700)
    auth_path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="already exists"):
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            auth_path=auth_path,
            journal_path=tmp_path / "control/codex.json",
        )

    auth_path.unlink()
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame), expected_kind="codex_access_token", now_unix=now
    )
    journal = tmp_path / "control/codex.json"
    runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        auth_path=auth_path,
        journal_path=journal,
    )
    original_mode = stat.S_IMODE(auth_path.stat().st_mode)
    auth_path.unlink()
    auth_path.write_text("substitute", encoding="utf-8")
    auth_path.chmod(original_mode)
    with pytest.raises(RuntimeError, match="identity is unsafe"):
        runtime.retire_secret_lease(
            kind="codex_access_token",
            target=auth_path,
            journal=journal,
            stop_proof=_stop_proof(plan),
        )


def test_overlay_cleanup_removes_only_exact_installed_bytes(tmp_path, monkeypatch):
    plan = _plan()
    target = tmp_path / "overlay.json"
    payload = b'{"sealed":true}\n'
    target.write_bytes(payload)
    target.chmod(0o600)
    identity = target.stat()
    binding = {
        "overlay": (
            target,
            payload,
            0o600,
            identity.st_uid,
            identity.st_gid,
            frozenset(),
        )
    }
    monkeypatch.setattr(runtime, "_capability_artifact_bindings", lambda *_: binding)

    removed = runtime._remove_exact_overlay_artifacts(plan, _full_plan())
    assert removed["overlay"]["removed"] is True
    assert not target.exists()

    target.write_bytes(b'{"substituted":true}\n')
    target.chmod(0o600)
    with pytest.raises(RuntimeError, match="substitution"):
        runtime._remove_exact_overlay_artifacts(plan, _full_plan())
    assert target.exists()


def test_browser_controller_config_is_exact_af_unix_and_rejects_tamper():
    plan = _plan()
    service = json.loads(runtime.render_browser_config(plan))
    client = runtime.capability_browser_controller_client_mapping(plan)

    assert service["socket_path"] == str(runtime.DEFAULT_BROWSER_SOCKET)
    assert service["allowed_client_uid"] == plan.identities.gateway_uid
    assert service["node_path"] == str(plan.browser_node)
    assert service["chrome_path"] == str(plan.browser_executable)
    assert client == {
        "schema": runtime.BROWSER_CONTROLLER_CLIENT_SCHEMA,
        "socket_path": str(runtime.DEFAULT_BROWSER_SOCKET),
        "server_uid": plan.identities.browser_uid,
        "artifact_root": str(runtime.DEFAULT_BROWSER_ARTIFACT_ROOT),
        "connect_timeout_seconds": 5,
        "request_timeout_seconds": runtime.BROWSER_COMMAND_TIMEOUT_SECONDS,
    }
    encoded = json.dumps({"service": service, "client": client})
    assert "cdp" not in encoded.lower()
    assert "remote-debugging" not in encoded
    assert "9222" not in encoded

    tampered = plan.to_mapping()
    tampered["browser"]["socket_path"] = "/tmp/controller.sock"
    tampered["capability_plan_sha256"] = runtime._sha256_json(
        {
            key: value
            for key, value in tampered.items()
            if key != "capability_plan_sha256"
        }
    )
    with pytest.raises(ValueError, match="AF_UNIX controller"):
        runtime.CapabilityCanaryPlan.from_mapping(tampered)


def _patch_browser_principals(monkeypatch, plan, *, browser_present=True):
    browser_user = SimpleNamespace(
        pw_name=plan.identities.browser_user,
        pw_uid=plan.identities.browser_uid,
        pw_gid=plan.identities.browser_gid,
        pw_dir=runtime.DEFAULT_BROWSER_HOME,
        pw_shell=runtime.DEFAULT_BROWSER_SHELL,
    )
    browser_group = SimpleNamespace(
        gr_name=plan.identities.browser_group,
        gr_gid=plan.identities.browser_gid,
        gr_mem=[],
    )
    projector_user = SimpleNamespace(
        pw_name=runtime.DEFAULT_PROJECTOR_USER,
        pw_uid=2198,
        pw_gid=2298,
    )
    projector_group = SimpleNamespace(
        gr_name=runtime.DEFAULT_PROJECTOR_GROUP,
        gr_gid=2298,
        gr_mem=[],
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_name",
        lambda name: (
            browser_user
            if name == plan.identities.browser_user and browser_present
            else projector_user if name == runtime.DEFAULT_PROJECTOR_USER else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_name",
        lambda name: (
            browser_group
            if name == plan.identities.browser_group and browser_present
            else projector_group if name == runtime.DEFAULT_PROJECTOR_GROUP else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_uid",
        lambda uid: (
            browser_user
            if uid == plan.identities.browser_uid and browser_present
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_gid",
        lambda gid: (
            browser_group
            if gid == plan.identities.browser_gid and browser_present
            else None
        ),
    )
    monkeypatch.setattr(
        runtime.os,
        "getgrouplist",
        lambda _user, primary_gid: [primary_gid],
    )


def test_browser_host_receipt_is_exact_and_allows_only_create_only_absence(
    monkeypatch,
):
    plan = _plan()
    _patch_browser_principals(monkeypatch, plan)
    present = runtime.browser_host_identity_receipt(
        plan, _full_plan(), allow_create_only_absence=False
    )
    assert present["browser"]["state"] == "present_exact"
    assert present["browser"]["supplementary_group_ids"] == [
        plan.identities.browser_gid
    ]
    assert present["projector"] == {
        "state": "present",
        "user": runtime.DEFAULT_PROJECTOR_USER,
        "group": runtime.DEFAULT_PROJECTOR_GROUP,
        "uid": 2198,
        "gid": 2298,
    }
    assert present["receipt_sha256"] == runtime._sha256_json(
        {key: value for key, value in present.items() if key != "receipt_sha256"}
    )

    _patch_browser_principals(monkeypatch, plan, browser_present=False)
    absent = runtime.browser_host_identity_receipt(
        plan, _full_plan(), allow_create_only_absence=True
    )
    assert absent["browser"]["state"] == "absent_create_only_slot"
    with pytest.raises(RuntimeError, match="principal is absent"):
        runtime.browser_host_identity_receipt(
            plan, _full_plan(), allow_create_only_absence=False
        )


def test_browser_identity_foundation_is_create_only_and_receipted(monkeypatch):
    plan = _plan()
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    before = {
        "receipt_sha256": "1" * 64,
        "browser": {"state": "absent_create_only_slot"},
    }
    after = {
        "receipt_sha256": "2" * 64,
        "browser": {"state": "present_exact"},
    }
    observations = iter((before, after))
    commands = []

    def observer(*_args, **_kwargs):
        return next(observations)

    def runner(command):
        commands.append(command.argv)
        return __import__("subprocess").CompletedProcess(
            command.argv, 0, stdout=b"", stderr=b""
        )

    receipt = runtime.ensure_browser_identity_create_only(
        plan,
        _full_plan(),
        runner=runner,
        observer=observer,
    )
    assert [command[0] for command in commands] == [
        runtime.GROUPADD,
        runtime.USERADD,
    ]
    assert receipt["created_group"] is True
    assert receipt["created_user"] is True
    assert receipt["retained_dormant_on_rollback"] is True
    assert receipt["host_identity"] == after
    assert receipt["receipt_sha256"] == runtime._sha256_json(
        {
            key: value
            for key, value in receipt.items()
            if key not in {"receipt_sha256", "host_identity"}
        }
    )


def test_browser_userns_and_principal_version_smokes_are_exact():
    plan = _plan()
    observed_paths = []

    def reader(path):
        observed_paths.append(path)
        return 1 if "unprivileged" in str(path) else 15452

    userns = runtime.browser_userns_preflight(reader=reader)
    assert userns["sandbox_required"] is True
    assert len(observed_paths) == 2
    with pytest.raises(RuntimeError, match="sandbox is disabled"):
        runtime.browser_userns_preflight(reader=lambda _path: 0)

    commands = []

    def runner(command):
        commands.append(command.argv)
        return __import__("subprocess").CompletedProcess(
            command.argv,
            0,
            stdout=(
                f"Google Chrome for Testing {runtime.RELEASE_CHROME_VERSION}\n"
            ).encode(),
            stderr=b"",
        )

    proof = runtime.browser_principal_version_smoke(plan, runner=runner)
    assert proof["uid"] == plan.identities.browser_uid
    assert commands[0][:6] == (
        runtime.RUNUSER,
        "--user",
        plan.identities.browser_user,
        "--group",
        plan.identities.browser_group,
        "--",
    )


def test_live_browser_preflight_binds_af_unix_socket_to_controller_mainpid():
    plan = _plan()
    state = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4242,
        "FragmentPath": str(runtime.DEFAULT_BROWSER_UNIT_PATH),
        "DropInPaths": "",
        "Type": "notify",
        "NotifyAccess": "main",
    }
    socket_state = SimpleNamespace(
        st_mode=stat.S_IFSOCK | 0o660,
        st_uid=plan.identities.browser_uid,
        st_gid=plan.identities.browser_gid,
        st_dev=11,
        st_ino=12,
    )

    proof = runtime.browser_service_runtime_preflight(
        plan,
        state,
        proc_stat=lambda _path: SimpleNamespace(
            st_uid=plan.identities.browser_uid,
            st_gid=plan.identities.browser_gid,
        ),
        socket_lstat=lambda _path: socket_state,
        listener_paths=lambda _pid: {str(runtime.DEFAULT_BROWSER_SOCKET)},
    )
    assert proof["main_pid"] == 4242
    assert proof["transport"] == "authenticated_af_unix"
    assert proof["socket_path"] == str(runtime.DEFAULT_BROWSER_SOCKET)
    with pytest.raises(RuntimeError, match="identity drifted"):
        runtime.browser_service_runtime_preflight(
            plan,
            state,
            proc_stat=lambda _path: SimpleNamespace(
                st_uid=plan.identities.gateway_uid,
                st_gid=plan.identities.gateway_gid,
            ),
            socket_lstat=lambda _path: socket_state,
            listener_paths=lambda _pid: {str(runtime.DEFAULT_BROWSER_SOCKET)},
        )


def test_canary_dependency_manifest_binds_exact_production_package(
    monkeypatch,
):
    plan = _plan()
    unsigned = {
        "schema": runtime.RUNTIME_DEPENDENCY_MANIFEST_SCHEMA,
        "release_revision": plan.revision,
        "agent_browser": {
            "version": runtime.RELEASE_AGENT_BROWSER_VERSION,
            "node_path": str(plan.browser_node),
            "node_sha256": plan.browser_node_sha256,
            "wrapper_path": str(plan.browser_wrapper),
            "wrapper_sha256": plan.browser_wrapper_sha256,
            "native_path": str(plan.browser_native),
            "native_sha256": plan.browser_native_sha256,
            "config_path": str(plan.agent_browser_config),
            "config_sha256": plan.agent_browser_config_sha256,
        },
        "chrome": {
            "version": runtime.RELEASE_CHROME_VERSION,
            "executable_path": str(plan.browser_executable),
            "executable_sha256": plan.browser_executable_sha256,
        },
        "python": {
            "distributions": {
                name: {"version": version}
                for name, version in runtime.RELEASE_DDGS_DISTRIBUTIONS.items()
            }
        },
        "secret_material_recorded": False,
    }
    manifest = {
        **unsigned,
        "manifest_sha256": runtime._sha256_json(unsigned),
    }
    raw = runtime._canonical_bytes(manifest) + b"\n"
    plan = replace(
        plan,
        runtime_dependency_manifest_sha256=runtime._sha256_bytes(raw),
    )
    monkeypatch.setattr(
        runtime,
        "_read_stable_file",
        lambda *_args, **_kwargs: (raw, SimpleNamespace()),
    )
    monkeypatch.setattr(
        runtime,
        "verify_release_runtime_dependency_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    proof = runtime.runtime_dependency_manifest_preflight(plan)

    assert proof["chrome_version"] == runtime.RELEASE_CHROME_VERSION
    assert proof["agent_browser_version"] == runtime.RELEASE_AGENT_BROWSER_VERSION
    assert proof["ddgs_version"] == "9.14.4"
    assert "/usr/bin/chromium" not in str(plan.browser_executable)


def test_contract_is_non_semantic_and_secret_free():
    contract = runtime.runtime_contract()
    assert contract["normal_gateway_loop"] is True
    assert contract["model_semantic_authority"] is True
    assert contract["codex_refresh_token_leased"] is False
    assert contract["discord_credential_in_gateway"] is False
    assert contract["mac_ops_credential_in_gateway"] is False
    assert contract["goal_judge_enabled"] is False
    assert contract["goal_continuations_enabled"] is False
    assert contract["mcp_auto_discovery_enabled"] is False
    assert contract["gateway_event_hooks_enabled"] is False
    assert contract["shell_hooks_enabled"] is False
    assert contract["plugin_middleware_enabled"] is False
    assert contract["plugin_allowlist"] == [runtime.CAPABILITY_OBSERVER_PLUGIN]
    assert "mac_ops" in contract["toolsets"]
    assert contract["credential_bindings"] == list(
        runtime.CAPABILITY_CREDENTIAL_BINDINGS
    )
    assert contract["browser_identity"] == "dedicated_create_only_principal"
    assert contract["browser_gateway_access"] == (
        "authenticated_af_unix_controller_only"
    )
    assert contract["browser_sandbox"] == "unprivileged_user_namespace_required"
    assert contract["terminal_gateway_access"] == (
        "authenticated_af_unix_isolated_worker_only"
    )
    assert contract["terminal_network_access"] is False
    assert contract["workspace_policy"] == (
        "ephemeral_isolated_worker_lease_no_host_projection"
    )


def test_capability_effective_environment_is_exact_and_rejects_unknown():
    config = yaml.safe_load(runtime.render_gateway_config(_plan()))
    terminal = config["terminal"]
    env = {
        "HOME": str(runtime.DEFAULT_GATEWAY_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "hermes_gateway",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SHELL": "/usr/sbin/nologin",
        "TZ": "UTC",
        "USER": "hermes_gateway",
        "HERMES_CONFIG": str(runtime.DEFAULT_GATEWAY_CONFIG),
        "HERMES_HOME": str(runtime.DEFAULT_GATEWAY_PROFILE_HOME),
        "HERMES_EXEC_ASK": "1",
        "HERMES_MANAGED_DIR": str(runtime.DEFAULT_DISABLED_MANAGED_SCOPE),
        "HERMES_MAX_ITERATIONS": "90",
        "HERMES_QUIET": "1",
        "SSL_CERT_FILE": str(runtime.DEFAULT_GATEWAY_CA_BUNDLE),
        "GATEWAY_RELAY_URL": "unix:///run/muncho-discord-connector/connector.sock",
        "GATEWAY_RELAY_PLATFORMS": "discord",
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_CWD": "/workspace",
        "TERMINAL_TIMEOUT": "180",
        "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": terminal[
            "isolated_worker_socket"
        ],
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(
            terminal["isolated_worker_server_uid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(
            terminal["isolated_worker_server_gid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": str(
            terminal["isolated_worker_socket_uid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(
            terminal["isolated_worker_socket_gid"]
        ),
        "CREDENTIALS_DIRECTORY": "/run/credentials/hermes-cloud-gateway.service",
        "RUNTIME_DIRECTORY": str(runtime.DEFAULT_GATEWAY_RUNTIME),
        "STATE_DIRECTORY": str(runtime.DEFAULT_GATEWAY_HOME),
        "NOTIFY_SOCKET": "/run/systemd/notify",
        "SYSTEMD_EXEC_PID": str(os.getpid()),
        "_HERMES_GATEWAY": "1",
    }
    assert runtime.capability_gateway_effective_environment_is_sealed(env, config)
    env["DISCORD_BOT_TOKEN"] = "forbidden"
    assert not runtime.capability_gateway_effective_environment_is_sealed(
        env, config
    )


def test_plan_has_no_host_workspace_projection_or_legacy_execution_transport():
    plan = _plan()
    value = plan.to_mapping()
    assert "workspaces" not in value
    assert "writable_root" not in value
    assert value["execution_workspace"] == {
        "path": "/workspace",
        "host_projection_enabled": False,
        "read_only_binds": [],
        "ephemeral_across_worker_restart": True,
        "lease_quota_bytes": runtime.SERVICE_GLOBAL_QUOTA_BYTES,
        "lease_quota_entries": runtime.LEASE_QUOTA_ENTRIES,
    }
    serialized = json.dumps(value).lower()
    assert "/srv/" not in serialized
    assert "docker" not in serialized
    assert "cdp" not in serialized
    assert "9222" not in serialized


def test_execution_identity_foundation_is_distinct_and_create_only(monkeypatch):
    plan = _plan()
    worker_user = SimpleNamespace(
        pw_name=plan.identities.worker_user,
        pw_uid=plan.identities.worker_uid,
        pw_gid=plan.identities.worker_gid,
        pw_dir=runtime.DEFAULT_WORKER_HOME,
        pw_shell=runtime.DEFAULT_WORKER_SHELL,
    )
    worker_group = SimpleNamespace(
        gr_name=plan.identities.worker_group,
        gr_gid=plan.identities.worker_gid,
        gr_mem=[],
    )
    client_group = SimpleNamespace(
        gr_name=plan.identities.worker_client_group,
        gr_gid=plan.identities.worker_client_gid,
        gr_mem=[],
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_name",
        lambda name: worker_user if name == plan.identities.worker_user else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_uid",
        lambda uid: worker_user if uid == plan.identities.worker_uid else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_name",
        lambda name: (
            worker_group
            if name == plan.identities.worker_group
            else client_group
            if name == plan.identities.worker_client_group
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_gid",
        lambda gid: (
            worker_group
            if gid == plan.identities.worker_gid
            else client_group
            if gid == plan.identities.worker_client_gid
            else None
        ),
    )
    monkeypatch.setattr(runtime.os, "getgrouplist", lambda *_args: [plan.identities.worker_gid])
    receipt = runtime.execution_host_identity_receipt(
        plan, _full_plan(), allow_create_only_absence=False
    )
    assert receipt["worker"]["state"] == "present_exact"
    assert receipt["socket_client_group"]["state"] == "present_exact"
    assert receipt["worker"]["supplementary_group_ids"] == [
        plan.identities.worker_gid
    ]

    before = {
        "receipt_sha256": "1" * 64,
        "worker": {"state": "absent_create_only_slot"},
        "socket_client_group": {"state": "absent_create_only_slot"},
    }
    after = {
        "receipt_sha256": "2" * 64,
        "worker": {"state": "present_exact"},
        "socket_client_group": {"state": "present_exact"},
    }
    observations = iter((before, after))
    commands = []
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    foundation = runtime.ensure_execution_identities_create_only(
        plan,
        _full_plan(),
        observer=lambda *_args, **_kwargs: next(observations),
        runner=lambda command: (
            commands.append(command.argv)
            or __import__("subprocess").CompletedProcess(
                command.argv, 0, stdout=b"", stderr=b""
            )
        ),
    )
    assert foundation["created"] == [
        "worker_group",
        "worker_client_group",
        "worker_user",
    ]
    assert [command[0] for command in commands] == [
        runtime.GROUPADD,
        runtime.GROUPADD,
        runtime.USERADD,
    ]


def test_systemd252_and_live_worker_tmpfs_contract_are_exact(monkeypatch):
    plan = _plan()
    mountpoint = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700,
        st_uid=0,
        st_gid=0,
    )
    monkeypatch.setattr(runtime.os, "lstat", lambda _path: mountpoint)
    proof = runtime.worker_systemd252_preflight(
        plan,
        runner=lambda command: __import__("subprocess").CompletedProcess(
            command.argv,
            0,
            stdout=b"systemd 252 (252.39-1~deb12u2)\n+PAM\n",
            stderr=b"",
        ),
    )
    assert proof["systemd_major"] == 252
    assert proof["contract"] == runtime.LEASE_TMPFS_PREFLIGHT_CONTRACT

    state = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4243,
        "FragmentPath": str(runtime.DEFAULT_WORKER_SERVICE_UNIT_PATH),
        "DropInPaths": "",
        "Type": "simple",
        "NotifyAccess": "none",
    }
    mountinfo = (
        b"101 99 0:55 / /var/lib/muncho-isolated-worker "
        b"rw,nosuid,nodev,relatime - tmpfs tmpfs "
        b"rw,size=4194304k,nr_inodes=200001,mode=700,uid=2107,gid=2208\n"
    )
    live = runtime.worker_tmpfs_runtime_preflight(
        plan,
        state,
        mountinfo_reader=lambda _path: mountinfo,
        path_lstat=lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=plan.identities.worker_uid,
            st_gid=plan.identities.worker_gid,
        ),
        path_statvfs=lambda _path: SimpleNamespace(
            f_blocks=runtime.SERVICE_GLOBAL_QUOTA_BYTES // 4096,
            f_frsize=4096,
            f_files=runtime.SERVICE_TMPFS_INODE_LIMIT,
        ),
    )
    assert live["filesystem"] == "tmpfs"
    assert live["mount_flags"] == ["nodev", "nosuid", "exec"]


def test_real_execution_readiness_helpers_are_both_required(monkeypatch):
    plan = _plan()
    observed = {}
    monkeypatch.setattr(runtime.os, "geteuid", lambda: plan.identities.gateway_uid)
    monkeypatch.setattr(runtime.os, "getegid", lambda: plan.identities.gateway_gid)

    def worker(**kwargs):
        observed["worker"] = kwargs
        return {
            "schema": runtime.WORKER_RECEIPT_SCHEMA,
            "lease_identity_sha256": hashlib.sha256(
                b"muncho-worker-readiness-v1\x00"
                + plan.revision.encode()
                + b"\x00"
                + plan.worker_config_sha256.encode()
            ).hexdigest(),
            "socket_path": str(runtime.DEFAULT_WORKER_SOCKET),
            "server_uid": plan.identities.worker_uid,
            "server_gid": plan.identities.worker_gid,
            "socket_uid": 0,
            "socket_gid": plan.identities.worker_client_gid,
            "execution_round_trip": True,
            "output_sha256": hashlib.sha256(
                b"MUNCHO_ISOLATED_WORKER_READY\n"
            ).hexdigest(),
            "secret_material_recorded": False,
        }

    def browser(**kwargs):
        observed["browser"] = kwargs
        return {
            "schema": runtime.BROWSER_RECEIPT_SCHEMA,
            "session_identity_sha256": hashlib.sha256(
                b"muncho-browser-readiness-v1\x00"
                + plan.revision.encode()
                + b"\x00"
                + plan.browser_config_sha256.encode()
            ).hexdigest(),
            "socket_path": str(runtime.DEFAULT_BROWSER_SOCKET),
            "server_uid": plan.identities.browser_uid,
            "command_round_trip": True,
            "secret_material_recorded": False,
        }

    monkeypatch.setattr(runtime, "attest_isolated_worker_execution", worker)
    monkeypatch.setattr(runtime, "attest_browser_controller_execution", browser)
    receipt = runtime.attest_capability_execution_readiness(plan)
    assert observed["worker"]["socket_path"] == runtime.DEFAULT_WORKER_SOCKET
    assert observed["worker"]["config_sha256"] == plan.worker_config_sha256
    assert observed["browser"]["config_sha256"] == plan.browser_config_sha256
    assert receipt["schema"] == runtime.CAPABILITY_EXECUTION_READINESS_SCHEMA
    assert receipt["isolated_worker"]["execution_round_trip"] is True
    assert receipt["browser_controller"]["command_round_trip"] is True

    monkeypatch.setattr(
        runtime,
        "attest_isolated_worker_execution",
        lambda **_kwargs: {**worker(**_kwargs), "unexpected": True},
    )
    with pytest.raises(RuntimeError, match="readiness receipt is invalid"):
        runtime.attest_capability_execution_readiness(plan)


def test_production_observation_marker_wait_is_exact_and_bounded(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    request = _production_observation_wait_request(plan, phase="after")
    raw = runtime._canonical_bytes(request)
    assert runtime.read_production_observation_wait_request(
        io.BytesIO(raw), plan=plan
    ) == request

    marker_path = tmp_path / "awaiting-production-after.json"
    marker_path.write_bytes(b"marker-present")
    observed = {}
    monkeypatch.setattr(
        runtime,
        "_production_observation_marker_path",
        lambda **_kwargs: marker_path,
    )

    def load_marker(_plan, **kwargs):
        observed.update(kwargs)
        return {"marker_sha256": "f" * 64}

    monkeypatch.setattr(
        runtime,
        "load_capability_production_observation_marker",
        load_marker,
    )
    receipt = runtime.wait_for_capability_production_observation_marker(
        plan,
        request,
        observer_gid=2200,
    )
    assert receipt["phase"] == "after"
    assert receipt["observer_live_verified"] is True
    assert observed["require_current_observer"] is True

    tampered = {**request, "timeout_seconds": 301}
    with pytest.raises(PermissionError, match="wait request is invalid"):
        runtime.read_production_observation_wait_request(
            io.BytesIO(runtime._canonical_bytes(tampered)), plan=plan
        )


def test_after_owner_observation_stages_then_publishes_exact_diff(monkeypatch):
    plan = _plan()
    envelope = {
        "phase": "after",
        "fixture_sha256": "c" * 64,
        "run_id": "capability-run-observed",
        "owner_subject_sha256": "d" * 64,
        "observation_sha256": "e" * 64,
    }
    calls = []
    before_envelope = {
        "phase": "before",
        "signed_at_unix_ms": 10,
    }
    monkeypatch.setattr(
        runtime,
        "_read_exact_file",
        lambda *_args, **_kwargs: (
            runtime._canonical_bytes(before_envelope),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "stage_owner_signed_production_observation",
        lambda value, **_kwargs: {
            "envelope_sha256": "1" * 64,
            "marker_sha256": "2" * 64,
        },
    )

    def load_observation(**kwargs):
        calls.append(
            ("load", kwargs["phase"], kwargs.get("now_unix_ms"))
        )
        return {"phase": kwargs["phase"], "signed_at_unix_ms": 10}

    monkeypatch.setattr(
        runtime,
        "load_staged_owner_signed_production_observation",
        load_observation,
    )
    monkeypatch.setattr(
        runtime,
        "build_capability_production_diff",
        lambda before, after, **_kwargs: {
            "schema": runtime.CAPABILITY_PRODUCTION_DIFF_SCHEMA,
            "diff_sha256": "3" * 64,
        },
    )

    def publish_diff(value, **kwargs):
        calls.append(("publish", value["diff_sha256"]))
        assert kwargs["run_id"] == envelope["run_id"]
        assert kwargs["observer_gid"] == 2200
        return {"diff_sha256": value["diff_sha256"]}

    monkeypatch.setattr(
        runtime,
        "publish_capability_production_diff",
        publish_diff,
    )
    receipt = runtime.stage_and_publish_owner_signed_production_observation(
        envelope,
        plan=plan,
        observer_gid=2200,
    )
    assert calls == [
        ("load", "before", 10),
        ("load", "after", None),
        ("publish", "3" * 64),
    ]
    assert receipt["observation_sha256"] == "e" * 64
    assert receipt["production_diff_sha256"] == "3" * 64
    assert receipt["schema"] == (
        runtime.CAPABILITY_PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA
    )


def test_public_api_exports_only_live_v2_symbols():
    assert runtime.__all__
    assert all(hasattr(runtime, name) for name in runtime.__all__)
    assert "WorkspaceBinding" not in runtime.__all__
    assert "browser_runtime_preflight" not in runtime.__all__
    namespace = {}
    exec(
        "from gateway.canonical_capability_canary_runtime import *",
        namespace,
    )
    assert "attest_capability_execution_readiness" in namespace
    assert "render_worker_service_unit" in namespace
