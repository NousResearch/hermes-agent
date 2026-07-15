from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_full_canary_fixture_publisher as publisher


REVISION = "a" * 40
IAM_SHA256 = "5" * 64
RUN_ID = "12345678-1234-4234-8234-123456789abc"


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _public_key(path: Path) -> tuple[str, str]:
    key = Ed25519PrivateKey.generate().public_key()
    pem = key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    path.write_bytes(pem)
    path.chmod(0o440)
    raw = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return raw.hex(), hashlib.sha256(raw).hexdigest()


def _prepare(tmp_path, monkeypatch):
    uid = os.geteuid()
    # BSD/macOS inherits the parent directory's group (not necessarily the
    # process egid) for files below pytest's /private/tmp root.  Bind the
    # injected NSS identities to the actual isolated fixture hierarchy.
    gid = tmp_path.stat().st_gid
    etc = tmp_path / "etc/muncho/full-canary"
    keys = tmp_path / "etc/muncho/keys"
    staged = tmp_path / "etc/muncho/staged"
    var_lib = tmp_path / "var/lib"
    for path, mode in ((etc, 0o755), (keys, 0o755), (staged, 0o755), (var_lib, 0o755)):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(mode)
    writer_public = keys / "writer-capability-public.pem"
    edge_public = keys / "discord-edge-receipt-public.pem"
    writer_hex, writer_id = _public_key(writer_public)
    edge_hex, edge_id = _public_key(edge_public)
    writer_config = staged / "writer.json"
    gateway_config = staged / "gateway.yaml"
    edge_config = staged / "discord-edge.json"
    writer_config.write_bytes(_canonical({
        "discord_edge_authority": {
            "edge_receipt_public_key_file": str(edge_public),
            "edge_receipt_public_key_id": edge_id,
        }
    }))
    gateway_config.write_bytes(_canonical({"gateway": "sealed"}))
    edge_config.write_bytes(_canonical({
        "keys": {
            "writer_capability_public_key_file": str(writer_public),
            "writer_capability_public_key_id": writer_id,
            "edge_receipt_public_key_id": edge_id,
        }
    }))
    for path in (writer_config, gateway_config, edge_config):
        path.chmod(0o440)
    fixture = etc / "fixture.json"
    publication_root = var_lib / "muncho-full-canary/fixture-publications"
    monkeypatch.setattr(publisher, "ROOT_UID", uid)
    monkeypatch.setattr(publisher, "ROOT_GID", gid)
    monkeypatch.setattr(publisher.sys, "platform", "linux")
    monkeypatch.setattr(publisher, "DEFAULT_E2E_FIXTURE", fixture)
    monkeypatch.setattr(
        publisher,
        "DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY",
        writer_public,
    )
    monkeypatch.setattr(publisher, "EDGE_RECEIPT_PUBLIC_KEY", edge_public)
    monkeypatch.setattr(publisher, "DEFAULT_WRITER_CONFIG_SOURCE", writer_config)
    monkeypatch.setattr(publisher, "DEFAULT_GATEWAY_CONFIG_SOURCE", gateway_config)
    monkeypatch.setattr(publisher, "DEFAULT_EDGE_CONFIG_SOURCE", edge_config)
    monkeypatch.setattr(publisher, "PUBLICATION_ROOT", publication_root)
    monkeypatch.setattr(
        publisher,
        "_exact_nss",
        lambda _user, _group: (uid, gid),
    )
    identities = SimpleNamespace(
        writer_uid=uid,
        writer_gid=gid,
        gateway_uid=uid,
        gateway_gid=gid,
    )
    activation = SimpleNamespace(
        revision=REVISION,
        identities=identities,
        digests=SimpleNamespace(external_iam_policy_sha256=IAM_SHA256),
        deployment_manifest={
            "snapshot_template": {
                "writer_deployment": {
                    "policy": {
                        "artifact_root": f"/opt/muncho-canary-releases/{REVISION}",
                        "artifact_digest_sha256": "6" * 64,
                    }
                }
            }
        },
        sha256="7" * 64,
    )
    monkeypatch.setattr(publisher, "load_activation_plan", lambda _path: activation)
    stopped = {
        publisher.EDGE_UNIT_NAME: {"state": "absent"},
        publisher.WRITER_UNIT_NAME: {"state": "absent"},
        publisher.GATEWAY_UNIT_NAME: {"state": "absent"},
    }
    monkeypatch.setattr(publisher, "_stopped_service_states", lambda: stopped)
    return {
        "fixture": fixture,
        "publication_root": publication_root,
        "writer_hex": writer_hex,
        "edge_hex": edge_hex,
        "writer_id": writer_id,
        "edge_id": edge_id,
        "stopped": stopped,
    }


def _kwargs(**changes):
    value = {
        "external_iam_policy_sha256": IAM_SHA256,
        "canary_run_id": RUN_ID,
        "valid_from_unix_ms": 999_000,
        "valid_until_unix_ms": 4_000_000,
        "owner_discord_user_id": publisher.OWNER_DISCORD_USER_ID,
        "guild_id": publisher.PUBLIC_GUILD_ID,
        "channel_id": publisher.PUBLIC_CHANNEL_ID,
        "expected_prompt_sha256": publisher.COMPLEX_CANARY_PROMPT_SHA256,
    }
    value.update(changes)
    return value


def test_fixture_plan_binds_public_target_release_keys_and_model_owned_task(
    tmp_path,
    monkeypatch,
):
    prepared = _prepare(tmp_path, monkeypatch)

    plan = publisher.build_fixture_publication_plan(REVISION, **_kwargs())
    fixture = plan["fixture"]

    assert fixture["release_sha"] == REVISION
    assert fixture["release_artifact_sha256"] == "6" * 64
    assert "api_session_key_sha256" not in fixture
    assert plan["api_session_key_present"] is False
    assert fixture["model_route"] == publisher._MODEL_ROUTE
    assert fixture["task_policy"] == {
        "minimum_completed_steps": 5,
        "prompt": publisher.COMPLEX_CANARY_PROMPT,
        "prompt_sha256": publisher.COMPLEX_CANARY_PROMPT_SHA256,
    }
    prompt_lower = publisher.COMPLEX_CANARY_PROMPT.lower()
    assert all(
        forbidden not in prompt_lower
        for forbidden in ("reasoning", "effort", "todo", "max")
    )
    assert fixture["public_routeback"]["target"] == {
        "target_type": "public_guild_channel",
        "guild_id": publisher.PUBLIC_GUILD_ID,
        "channel_id": publisher.PUBLIC_CHANNEL_ID,
    }
    assert fixture["discord_public_keys"] == {
        "writer_capability_ed25519_hex": prepared["writer_hex"],
        "edge_receipt_ed25519_hex": prepared["edge_hex"],
    }
    assert plan["external_iam_policy_sha256"] == IAM_SHA256
    assert plan["invariants"]["private_keys_read"] is False
    assert plan["invariants"]["api_session_secret_present"] is False
    assert all("private" not in name for name in plan["writer_public_key"])
    assert all("private" not in name for name in plan["edge_public_key"])
    unsigned = {name: item for name, item in plan.items() if name != "plan_sha256"}
    assert plan["plan_sha256"] == publisher._sha256_json(unsigned)


@pytest.mark.parametrize(
    "channel_id",
    [
        "1501999999999999999",
        *sorted(publisher.LOCKED_PRIVATE_CHANNEL_IDS),
    ],
)
def test_fixture_rejects_dm_or_unapproved_channel_before_publication(
    tmp_path,
    monkeypatch,
    channel_id,
):
    prepared = _prepare(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="owner intent"):
        publisher.build_fixture_publication_plan(
            REVISION,
            **_kwargs(channel_id=channel_id),
        )
    assert not prepared["fixture"].exists()
    assert not prepared["publication_root"].exists()


def test_fixture_publication_is_atomic_readback_verified_and_idempotent(
    tmp_path,
    monkeypatch,
):
    prepared = _prepare(tmp_path, monkeypatch)
    plan = publisher.build_fixture_publication_plan(REVISION, **_kwargs())

    receipt = publisher.apply_fixture_publication(
        REVISION,
        plan["plan_sha256"],
        **_kwargs(),
        clock=lambda: 1_000.0,
    )

    fixture_raw = publisher._canonical_bytes(plan["fixture"])
    assert prepared["fixture"].read_bytes() == fixture_raw
    fixture_item = prepared["fixture"].stat()
    assert stat.S_IMODE(fixture_item.st_mode) == 0o440
    assert receipt["fixture_sha256"] == hashlib.sha256(fixture_raw).hexdigest()
    assert receipt["state"] == "base_fixture_published_services_stopped"
    assert receipt["services_started"] is False
    assert receipt["discord_dispatched"] is False
    assert receipt["api_session_secret_present"] is False
    assert receipt["api_session_key_present"] is False
    receipt_path = Path(receipt["publication_receipt_path"])
    assert receipt_path.read_bytes() == publisher._canonical_bytes(receipt)

    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (prepared["fixture"], receipt_path)
    }
    retry = publisher.apply_fixture_publication(
        REVISION,
        plan["plan_sha256"],
        **_kwargs(),
        clock=lambda: 1_001.0,
    )
    after = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in before
    }
    assert retry == receipt
    assert after == before

    with pytest.raises(RuntimeError, match="no-replace collision"):
        publisher.apply_fixture_publication(
            REVISION,
            publisher.build_fixture_publication_plan(
                REVISION,
                **_kwargs(canary_run_id="22345678-1234-4234-8234-123456789abc"),
            )["plan_sha256"],
            **_kwargs(canary_run_id="22345678-1234-4234-8234-123456789abc"),
            clock=lambda: 1_002.0,
        )


def test_fixture_plan_rejects_key_config_drift(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    edge_config = publisher.DEFAULT_EDGE_CONFIG_SOURCE
    edge_config.chmod(0o600)
    edge_config.write_bytes(_canonical({
        "keys": {
            "writer_capability_public_key_file": str(
                publisher.DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY
            ),
            "writer_capability_public_key_id": "9" * 64,
            "edge_receipt_public_key_id": "8" * 64,
        }
    }))
    edge_config.chmod(0o440)

    with pytest.raises(RuntimeError, match="differ from staged"):
        publisher.build_fixture_publication_plan(REVISION, **_kwargs())
