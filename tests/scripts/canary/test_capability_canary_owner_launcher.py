"""Owner-boundary tests for capability-canary secret transport."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import scripts.canary.capability_canary_owner_launcher as launcher


def _jwt(expiry: int) -> str:
    encode = lambda value: base64.urlsafe_b64encode(
        json.dumps(value, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return f"{encode({'alg': 'none'})}.{encode({'exp': expiry})}.sig"


def _plan_inputs() -> dict[str, object]:
    unsigned = {
        "schema": launcher.CAPABILITY_PLAN_INPUTS_SCHEMA,
        "identities": {
            "mac_ops_uid": 2104,
            "mac_ops_gid": 2205,
            "connector_uid": 2105,
            "connector_gid": 2206,
            "bitrix_operational_edge_uid": 2108,
            "bitrix_operational_edge_gid": 2210,
            "bitrix_operational_edge_client_gid": 2211,
            "browser_uid": 2106,
            "browser_gid": 2207,
            "worker_uid": 2107,
            "worker_gid": 2208,
            "worker_client_gid": 2209,
        },
        "discord": {
            "connector_bot_user_id": "1600000000000000001",
            "routeback_bot_user_id": "1600000000000000002",
            "allowed_guild_ids": ["1282725267068157972"],
            "allowed_channel_ids": [
                launcher.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID
            ],
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
        "inputs_sha256": __import__("hashlib").sha256(
            launcher._canonical_bytes(unsigned)
        ).hexdigest(),
    }


def _foundation_inputs() -> dict[str, object]:
    unsigned = {
        "schema": launcher.CAPABILITY_BITRIX_FOUNDATION_INPUTS_SCHEMA,
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_gid": 2211,
        "business_edge_uid": 2104,
        "asset_manifest_sha256": "9" * 64,
    }
    return {
        **unsigned,
        "inputs_sha256": __import__("hashlib").sha256(
            launcher._canonical_bytes(unsigned)
        ).hexdigest(),
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
    payload = (
        _jwt(issued + 3_600).encode()
        if kind == "codex_access_token"
        else secret
    )
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


def test_codex_reader_extracts_access_only_from_private_stable_file(tmp_path):
    path = tmp_path / "auth.json"
    access = _jwt(int(time.time()) + 3_600)
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access,
                    "refresh_token": "must-never-be-leased",
                },
            }
        )
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


def test_discord_connector_reader_accepts_only_one_bounded_printable_token():
    token = b"discord.connector-token_123"
    assert launcher.read_discord_connector_token(io.BytesIO(token)) == bytearray(
        token
    )

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
        launcher.read_bitrix_operational_edge_webhook(
            io.BytesIO(value + b"\n\n")
        )


def test_transport_exposes_only_the_fixed_packaged_lifecycle_actions():
    revision = "a" * 40
    expected = {
        "contract",
        "storage-preflight",
        "bootstrap-bitrix-foundation",
        "prepare-producer-foundation",
        "install-producer-foundation",
        "preflight-producer-foundation",
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
    authority = launcher.build_plan_publication_authority(
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        plan_sha256="c" * 64,
        owner_subject_sha256="d" * 64,
        inputs=loaded,
    )
    assert authority["plan_sha256"] == "c" * 64
    assert authority["inputs"] == {
        key: inputs[key] for key in ("identities", "discord", "artifacts")
    }
    assert authority["semantic_content_recorded"] is False
    assert "path" not in authority
    assert "token" not in json.dumps(authority).lower()

    path.chmod(0o644)
    with pytest.raises(
        Exception,
        match="capability_plan_input_source_invalid",
    ):
        launcher.read_plan_publication_inputs(path.resolve())


def test_bitrix_foundation_authority_precedes_and_never_binds_capability_plan(
    tmp_path,
):
    path = tmp_path / "bitrix-foundation-inputs.json"
    path.write_bytes(launcher._canonical_bytes(_foundation_inputs()))
    path.chmod(0o600)
    captured: dict[str, object] = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "bootstrap-bitrix-foundation"
            captured["frame"] = frame
            captured["during"] = bytes(frame)
            return {"schema": "foundation", "ok": True}

    result = launcher.bootstrap_bitrix_foundation(
        Transport(),
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        release_artifact_sha256="c" * 64,
        owner_subject_sha256="d" * 64,
        foundation_file=path.resolve(),
    )
    authority = json.loads(captured["during"])
    assert result == {"schema": "foundation", "ok": True}
    assert authority["full_canary_plan_sha256"] == "b" * 64
    assert authority["release_artifact_sha256"] == "c" * 64
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

    preparation_unsigned = {
        "schema": launcher.FOUNDATION_PREPARATION_SCHEMA,
        "revision": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "role_identities": _identities(),
        "key_bootstrap_receipt_sha256": keys.value["receipt_sha256"],
        "owner_public_key_ed25519_hex": owner["public_key_ed25519_hex"],
        "owner_public_key_source_sha256": owner[
            "public_key_source_sha256"
        ],
        "unsigned_foundation": unsigned_foundation,
        "signature_payload_sha256": producers._sha256_bytes(
            producers.producer_foundation_signature_payload(
                unsigned_foundation
            )
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
                    pinned_owner_public_key_ed25519_hex=context[
                        "owner_public"
                    ],
                    pinned_owner_public_key_source_sha256=context[
                        "source_sha256"
                    ],
                )
                install_unsigned = {
                    "schema": launcher.FOUNDATION_INSTALL_RECEIPT_SCHEMA,
                    "revision": "a" * 40,
                    "capability_plan_sha256": "b" * 64,
                    "full_canary_plan_sha256": "c" * 64,
                    "preparation_sha256": preparation[
                        "preparation_sha256"
                    ],
                    "foundation_sha256": producers.producer_foundation_sha256(
                        sealed
                    ),
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
                    "receipt_sha256": producers._sha256_json(
                        install_unsigned
                    ),
                }
            assert action == "preflight-producer-foundation"
            return {
                "schema": (
                    "muncho-capability-producer-installation-preflight.v1"
                ),
                "revision": "a" * 40,
                "foundation_sha256": producers.producer_foundation_sha256(
                    foundation
                ),
                "preparation_sha256": preparation["preparation_sha256"],
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
    captured: dict[str, object] = {}

    class Transport:
        def invoke(self, revision, action, *, frame=None):
            assert revision == "a" * 40
            assert action == "publish-plan"
            captured["frame"] = frame
            captured["during"] = bytes(frame)
            return {"schema": "receipt", "ok": True}

    result = launcher.publish_capability_plan(
        Transport(),
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        plan_sha256="c" * 64,
        owner_subject_sha256="d" * 64,
        plan_file=path.resolve(),
    )

    authority = json.loads(captured["during"])
    assert result == {"schema": "receipt", "ok": True}
    assert authority["plan_sha256"] == "c" * 64
    assert authority["full_canary_plan_sha256"] == "b" * 64
    assert authority["owner_subject_sha256"] == "d" * 64
    assert captured["frame"] == bytearray(b"\x00" * len(captured["during"]))


def _fixture_publication_receipt() -> dict[str, object]:
    run_id = "capability-run-1"
    plan_sha = "b" * 64
    fixture_sha = "c" * 64
    unsigned = {
        "schema": launcher.FIXTURE_PUBLICATION_RECEIPT_SCHEMA,
        "run_id": run_id,
        "release_sha": "a" * 40,
        "capability_plan_sha256": plan_sha,
        "full_canary_plan_sha256": "d" * 64,
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
        "receipt_sha256": __import__("hashlib").sha256(
            launcher._canonical_bytes(unsigned)
        ).hexdigest(),
    }


def test_publish_live_fixture_is_bounded_canonical_stdin_and_wipes_frame():
    authority = {
        "schema": "muncho-production-capability-canary-fixture-authority.v1",
        "run_id": "capability-run-1",
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
    assert launcher.validate_fixture_publication_receipt(
        receipt,
        revision="a" * 40,
    )["receipt_sha256"] == receipt["receipt_sha256"]

    tampered = json.loads(json.dumps(receipt))
    tampered["receipt_path"] = "/tmp/attacker.json"
    with pytest.raises(
        Exception,
        match="capability_fixture_publication_receipt_invalid",
    ):
        launcher.validate_fixture_publication_receipt(
            tampered,
            revision="a" * 40,
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
            source,
            source_sha256,
        )
        if supplied_revision == revision
        else pytest.fail("unexpected revision"),
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
            "receipt_sha256": __import__("hashlib").sha256(
                launcher._canonical_bytes(unsigned)
            ).hexdigest(),
        }

    class Transport:
        def invoke(self, supplied_revision, action, *, frame=None):
            assert supplied_revision == revision
            calls.append(action)
            if action == "run-live":
                assert after_staged.wait(2)
                return {
                    "schema": (
                        "muncho-production-capability-canary-live-driver.v1"
                    ),
                    "ok": True,
                    "release_sha": revision,
                    "capability_plan_sha256": plan_sha,
                    "full_canary_plan_sha256": full_sha,
                    "fixture_sha256": fixture_sha,
                    "run_id": run_id,
                    "production_before_observation_sha256": (
                        before_observation_sha
                    ),
                    "production_diff_sha256": diff_sha,
                    "evidence_sha256": evidence_sha,
                }
            assert isinstance(frame, bytearray)
            frames.append(frame)
            value = json.loads(bytes(frame))
            phase = value["phase"]
            if action == "wait-production-observation-marker":
                assert value["schema"] == (
                    launcher.PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA
                )
                return signed_receipt(
                    {
                        "schema": (
                            "muncho-production-capability-production-"
                            "observation-marker-wait.v1"
                        ),
                        "phase": phase,
                        "run_id": run_id,
                        "fixture_sha256": fixture_sha,
                        "marker_sha256": ("5" if phase == "before" else "6")
                        * 64,
                        "observer_live_verified": phase == "after",
                        "secret_material_recorded": False,
                        "secret_digest_recorded": False,
                    }
                )
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
                "production_diff_sha256": (
                    None if phase == "before" else diff_sha
                ),
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
            "envelope_sha256": __import__("hashlib").sha256(
                launcher._canonical_bytes(signed)
            ).hexdigest(),
        }

    monkeypatch.setattr(
        launcher,
        "collect_owner_signed_production_observation",
        collect,
    )
    result = launcher.run_live_with_owner_production_observations(
        Transport(),
        object(),
        revision=revision,
        plan_sha256=plan_sha,
        full_canary_plan_sha256=full_sha,
        fixture_sha256=fixture_sha,
        run_id=run_id,
        owner_subject_sha256=owner_sha,
        timeout_seconds=30,
    )

    assert result["production_diff_sha256"] == diff_sha
    assert result["live_evidence_sha256"] == evidence_sha
    assert calls.count("run-live") == 1
    assert [action for action in calls if action != "run-live"] == [
        "wait-production-observation-marker",
        "stage-production-observation",
        "wait-production-observation-marker",
        "stage-production-observation",
    ]
    assert all(frame == bytearray(len(frame)) for frame in frames)


def test_plan_publication_inputs_reject_bot_reuse_and_tamper():
    same_bot = _plan_inputs()
    same_bot["discord"]["routeback_bot_user_id"] = same_bot["discord"][
        "connector_bot_user_id"
    ]
    unsigned = {
        key: value for key, value in same_bot.items() if key != "inputs_sha256"
    }
    same_bot["inputs_sha256"] = __import__("hashlib").sha256(
        launcher._canonical_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(ValueError, match="not distinct"):
        launcher.validate_plan_publication_inputs(same_bot)

    tampered = _plan_inputs()
    tampered["identities"]["browser_uid"] += 1
    with pytest.raises(ValueError, match="digest drifted"):
        launcher.validate_plan_publication_inputs(tampered)


@pytest.mark.parametrize("locked_channel_id", sorted(launcher.LOCKED_NONPUBLIC_CHANNEL_IDS))
def test_plan_publication_inputs_reject_each_locked_discord_channel(
    locked_channel_id,
):
    inputs = _plan_inputs()
    inputs["discord"]["allowed_channel_ids"] = [locked_channel_id]
    unsigned = {
        key: value for key, value in inputs.items() if key != "inputs_sha256"
    }
    inputs["inputs_sha256"] = __import__("hashlib").sha256(
        launcher._canonical_bytes(unsigned)
    ).hexdigest()

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
            "minimum_packaging_free_bytes": (
                launcher.MINIMUM_PACKAGING_FREE_BYTES
            ),
            "shortfall_bytes": (
                launcher.MINIMUM_PACKAGING_FREE_BYTES - available
            ),
        },
        "retention": {
            "maximum_managed_releases": launcher.MAXIMUM_MANAGED_RELEASES,
            "protect_target_release": True,
            "protect_newest_rollback_release": True,
        },
        "protected_release_paths": sorted(
            [
                f"{launcher.RELEASE_BASE}/{revision}",
                f"{launcher.RELEASE_BASE}/{rollback}",
            ]
        ),
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
        "receipt_sha256": __import__("hashlib").sha256(
            launcher._canonical_bytes(unsigned)
        ).hexdigest(),
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
    assert captured["frame"] == bytearray(
        b"\x00" * len(b"framed-connector-secret")
    )


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
    assert launcher.provision_api_control(
        Transport(),
        revision="a" * 40,
        plan_sha256="b" * 64,
        owner_subject_sha256="c" * 64,
    ) == {"ok": True}
    assert captured["kind"] == "api_server_control_key"
    assert generated == bytearray(len(generated))
    assert captured["frame"] == bytearray(len(b"api-frame"))


def test_install_approval_is_fresh_plan_bound_and_stdin_only(monkeypatch):
    revision = "a" * 40
    calls = []

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
                    "full_canary_plan_sha256": "c" * 64,
                    "report_sha256": "d" * 64,
                    "state_sha256": "9" * 64,
                    "checks": {
                        "browser.executable": True,
                        "browser.host_identity": True,
                        "browser.userns_sandbox": True,
                        "browser.principal_smoke": True,
                        "worker.executables": True,
                        "worker.systemd252_tmpfs_contract": True,
                        "execution.host_identity": True,
                    },
                    "evidence": {
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
                    },
                }
            approval = json.loads(bytes(frame))
            assert approval["schema"] == launcher.CAPABILITY_APPROVAL_SCHEMA
            assert approval["plan_sha256"] == "b" * 64
            assert approval["full_canary_plan_sha256"] == "c" * 64
            assert approval["stopped_preflight_state_sha256"] == "9" * 64
            assert approval["owner_subject_sha256"] == "e" * 64
            assert approval["expires_at_unix"] - approval["approved_at_unix"] == 300
            assert len(approval["nonce_sha256"]) == 64
            return {"ok": True, "approval_sha256": "f" * 64}

    monkeypatch.setattr(launcher.secrets, "token_bytes", lambda _size: b"n" * 32)
    result = launcher.install_capability_approval(
        Transport(),
        revision=revision,
        owner_subject_sha256="e" * 64,
        now_unix=1_000,
    )

    assert result == {"ok": True, "approval_sha256": "f" * 64}
    assert calls[0] == (revision, "preflight-stopped", None)
    assert calls[1][0:2] == (revision, "install-approval")
    assert calls[1][2] == bytearray(len(calls[1][2]))


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
