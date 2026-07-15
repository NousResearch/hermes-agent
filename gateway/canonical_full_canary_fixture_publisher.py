#!/usr/bin/env python3
"""Publish the owner-bound, secret-free base fixture for one live canary.

This is a mechanical source-to-runtime boundary.  It accepts only exact owner
bindings (release, run/window, public Discord target, and IAM-policy digest),
reads only root-controlled public keys/configuration, and publishes a
no-replace fixture plus receipt while all canary services are stopped.  The
raw one-shot API session key is deliberately absent: the live-driver staging
boundary later places only its SHA-256 into a copy of this base fixture.

No model, task classifier, route selector, Discord credential, private key,
database credential, service mutation, or IAM mutation is present here.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import pwd
import re
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway.canonical_full_canary_e2e import (
    FIXTURE_SCHEMA,
    _MODEL_ROUTE,
    _validate_fixture,
)
from gateway.canonical_full_canary_runtime import (
    DEFAULT_EDGE_CONFIG_SOURCE,
    DEFAULT_E2E_FIXTURE,
    DEFAULT_GATEWAY_CONFIG_SOURCE,
    DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY,
    DEFAULT_WRITER_CONFIG_SOURCE,
    EDGE_UNIT_NAME,
    GATEWAY_UNIT_NAME,
    WRITER_UNIT_NAME,
    _read_stable_file,
    collect_service_state,
    evaluate_service_states,
)
from gateway.canonical_writer_activation import (
    DEFAULT_PLAN_PATH as WRITER_ACTIVATION_PLAN_PATH,
    load_activation_plan,
)
from gateway.support_ops_team_registry import (
    SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS,
)


FIXTURE_PUBLICATION_PLAN_SCHEMA = "muncho-full-canary-fixture-publication-plan.v1"
FIXTURE_PUBLICATION_RECEIPT_SCHEMA = (
    "muncho-full-canary-fixture-publication-receipt.v1"
)
FIXTURE_PUBLICATION_FAILURE_SCHEMA = (
    "muncho-full-canary-fixture-publication-failure.v1"
)

OWNER_DISCORD_USER_ID = "1279454038731264061"
PUBLIC_GUILD_ID = "1282725267068157972"
PUBLIC_CHANNEL_ID = "1526858760100909066"
LOCKED_PRIVATE_CHANNEL_IDS = SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS
EDGE_RECEIPT_PUBLIC_KEY = Path(
    "/etc/muncho/keys/discord-edge-receipt-public.pem"
)
PUBLICATION_ROOT = Path("/var/lib/muncho-full-canary/fixture-publications")
EDGE_USER = "muncho-discord-egress"
EDGE_GROUP = "muncho-discord-egress"
WRITER_USER = "muncho-canonical-writer"
WRITER_GROUP = "muncho-canonical-writer"
GATEWAY_USER = "hermes-cloud-gateway"
GATEWAY_GROUP = "hermes-cloud-gateway"
_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,25}$")
_MAX_ARTIFACT_BYTES = 512 * 1024
_PUBLICATION_DIRECTORY_MODE = 0o700
_PUBLICATION_RECEIPT_MODE = 0o400
_FIXTURE_MODE = 0o440
_MINIMUM_REMAINING_WINDOW_MS = 10 * 60 * 1000
ROOT_UID = 0
ROOT_GID = 0

# This is task content, not routing logic.  The model owns the actual plan,
# decisions, effort choice, tool use, and completion sequence.  In particular,
# the fixture must not tell the model which effort level or control tool to use:
# the canary verifies that choice independently from the resulting receipts.
COMPLEX_CANARY_PROMPT = (
    "Complete a release-readiness investigation for this isolated Muncho "
    "canary. Do not assume success from this request. Use the available live "
    "runtime evidence and Canonical Task Workspace to create and maintain "
    "your own dependency-aware plan with explicit success criteria; inspect "
    "the writer and gateway evidence; reconcile the findings into canonical "
    "case truth; verify every criterion with fresh readback; and only then "
    "route a concise final outcome to the configured public Discord target, "
    "confirming the delivery receipt. Discord DMs and private targets are "
    "forbidden. If one approach is blocked, pursue safe read-only or "
    "idempotent alternatives, record the exact blocker, and continue all "
    "independent work. Do not stop at the first blocker or leave feasible "
    "steps partial."
)
COMPLEX_CANARY_PROMPT_SHA256 = hashlib.sha256(
    COMPLEX_CANARY_PROMPT.encode("utf-8")
).hexdigest()

Clock = Callable[[], float]
Runner = Callable[..., Any]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError) as exc:
        raise ValueError("full-canary fixture value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not an exact SHA-256")
    return value


def _exact_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("canary run id is invalid") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise ValueError("canary run id is not canonical")
    return value


def _exact_nss(user_name: str, group_name: str) -> tuple[int, int]:
    try:
        user = pwd.getpwnam(user_name)
        by_uid = pwd.getpwuid(user.pw_uid)
        group = grp.getgrnam(group_name)
        by_gid = grp.getgrgid(group.gr_gid)
    except (KeyError, OSError) as exc:
        raise RuntimeError("full-canary fixture service identity is unavailable") from exc
    if (
        user.pw_name != user_name
        or by_uid.pw_name != user_name
        or group.gr_name != group_name
        or by_gid.gr_name != group_name
        or user.pw_gid != group.gr_gid
        or user.pw_uid <= 0
        or group.gr_gid <= 0
    ):
        raise RuntimeError("full-canary fixture service identity drifted")
    return user.pw_uid, group.gr_gid


def _decode_mapping(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid JSON") from exc
    canonical = _canonical_bytes(value)
    if not isinstance(value, dict) or raw not in {canonical, canonical + b"\n"}:
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _public_key_provenance(
    path: Path,
    *,
    expected_gid: int,
) -> dict[str, Any]:
    raw, item = _read_stable_file(
        path,
        maximum=16 * 1024,
        expected_uid=ROOT_UID,
        expected_gid=expected_gid,
        allowed_modes=frozenset({_FIXTURE_MODE}),
    )
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("full-canary fixture public key is invalid PEM") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise RuntimeError("full-canary fixture public key is not Ed25519")
    public_raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "path": str(path),
        "file_sha256": _sha256_bytes(raw),
        "public_key_ed25519_hex": public_raw.hex(),
        "key_id": _sha256_bytes(public_raw),
        "device": item.st_dev,
        "inode": item.st_ino,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "size": item.st_size,
    }


def _config_sources(
    *,
    writer_gid: int,
    gateway_gid: int,
    edge_gid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    writer_raw, _writer_item = _read_stable_file(
        DEFAULT_WRITER_CONFIG_SOURCE,
        maximum=_MAX_ARTIFACT_BYTES,
        expected_uid=ROOT_UID,
        expected_gid=writer_gid,
        allowed_modes=frozenset({_FIXTURE_MODE}),
    )
    # Gateway config is not semantically interpreted here.  Reading it proves
    # the exact staged input namespace is complete before fixture publication.
    gateway_raw, _gateway_item = _read_stable_file(
        DEFAULT_GATEWAY_CONFIG_SOURCE,
        maximum=_MAX_ARTIFACT_BYTES,
        expected_uid=ROOT_UID,
        expected_gid=gateway_gid,
        allowed_modes=frozenset({_FIXTURE_MODE}),
    )
    edge_raw, _edge_item = _read_stable_file(
        DEFAULT_EDGE_CONFIG_SOURCE,
        maximum=_MAX_ARTIFACT_BYTES,
        expected_uid=ROOT_UID,
        expected_gid=edge_gid,
        allowed_modes=frozenset({_FIXTURE_MODE}),
    )
    writer = _decode_mapping(writer_raw, "staged writer config")
    edge = _decode_mapping(edge_raw, "staged Discord edge config")
    writer_authority = writer.get("discord_edge_authority")
    edge_keys = edge.get("keys")
    if not isinstance(writer_authority, Mapping) or not isinstance(edge_keys, Mapping):
        raise RuntimeError("full-canary fixture key configuration is missing")
    return (
        {
            "writer_config_file_sha256": _sha256_bytes(writer_raw),
            "gateway_config_file_sha256": _sha256_bytes(gateway_raw),
            "edge_config_file_sha256": _sha256_bytes(edge_raw),
            "writer_edge_receipt_public_key_file": writer_authority.get(
                "edge_receipt_public_key_file"
            ),
            "writer_edge_receipt_public_key_id": writer_authority.get(
                "edge_receipt_public_key_id"
            ),
            "edge_writer_capability_public_key_file": edge_keys.get(
                "writer_capability_public_key_file"
            ),
            "edge_writer_capability_public_key_id": edge_keys.get(
                "writer_capability_public_key_id"
            ),
            "edge_receipt_public_key_id": edge_keys.get(
                "edge_receipt_public_key_id"
            ),
        },
        writer,
    )


def _stopped_service_states() -> dict[str, Mapping[str, Any]]:
    states = {
        unit: collect_service_state(unit)
        for unit in (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    }
    checks = evaluate_service_states(states, phase="stopped")
    if not checks or not all(checks.values()):
        raise RuntimeError("full-canary fixture publication requires stopped services")
    return states


def _activation_release_artifact(plan: Any) -> str:
    try:
        policy = plan.deployment_manifest["snapshot_template"][
            "writer_deployment"
        ]["policy"]
        artifact_root = policy["artifact_root"]
        artifact_sha256 = policy["artifact_digest_sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("activation release binding is incomplete") from exc
    if artifact_root != f"/opt/muncho-canary-releases/{plan.revision}":
        raise RuntimeError("activation release root is not revision-addressed")
    return _digest(artifact_sha256, "release artifact digest")


def _fixture(
    *,
    release_sha: str,
    release_artifact_sha256: str,
    canary_run_id: str,
    valid_from_unix_ms: int,
    valid_until_unix_ms: int,
    owner_discord_user_id: str,
    guild_id: str,
    channel_id: str,
    writer_public_hex: str,
    edge_public_hex: str,
) -> dict[str, Any]:
    case_id = f"case:full-canary:{canary_run_id}"
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "canary_run_id": canary_run_id,
        "release_sha": release_sha,
        "release_artifact_sha256": release_artifact_sha256,
        "valid_from_unix_ms": valid_from_unix_ms,
        "valid_until_unix_ms": valid_until_unix_ms,
        "case_id": case_id,
        "owner_discord_user_id": owner_discord_user_id,
        "source": {
            "platform": "api_server",
            "control_protocol": "authenticated_loopback_api_server.v1",
            "host": "127.0.0.1",
            "port": 8642,
            "session_create_endpoint": "/api/sessions",
            "chat_stream_endpoint_template": (
                "/api/sessions/{session_id}/chat/stream"
            ),
        },
        "model_route": dict(_MODEL_ROUTE),
        "task_policy": {
            "minimum_completed_steps": 5,
            "prompt": COMPLEX_CANARY_PROMPT,
            "prompt_sha256": COMPLEX_CANARY_PROMPT_SHA256,
        },
        "public_routeback": {
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": guild_id,
                "channel_id": channel_id,
            },
            "canonical_idempotency_key": (
                f"full-canary-routeback:{canary_run_id}"
            ),
        },
        "discord_public_keys": {
            "writer_capability_ed25519_hex": writer_public_hex,
            "edge_receipt_ed25519_hex": edge_public_hex,
        },
    }
    # The offline/live verifier validates the session-bound copy.  Reuse that
    # strict validator with a non-authorizing sentinel, then remove the field
    # so the published base can only be bound by prepare_session_bound_plan().
    validated = _validate_fixture({
        **fixture,
        "api_session_key_sha256": _ZERO_SHA256,
    })
    del validated["api_session_key_sha256"]
    return validated


def build_fixture_publication_plan(
    release_sha: str,
    *,
    external_iam_policy_sha256: str,
    canary_run_id: str,
    valid_from_unix_ms: int,
    valid_until_unix_ms: int,
    owner_discord_user_id: str,
    guild_id: str,
    channel_id: str,
    expected_prompt_sha256: str,
) -> dict[str, Any]:
    if not isinstance(release_sha, str) or _REVISION_RE.fullmatch(release_sha) is None:
        raise ValueError("fixture release SHA is invalid")
    external_iam_policy_sha256 = _digest(
        external_iam_policy_sha256,
        "external IAM policy digest",
    )
    canary_run_id = _exact_uuid(canary_run_id)
    if (
        type(valid_from_unix_ms) is not int
        or type(valid_until_unix_ms) is not int
        or valid_from_unix_ms < 1
        or valid_until_unix_ms <= valid_from_unix_ms
        or valid_until_unix_ms - valid_from_unix_ms > 3_600_000
        or owner_discord_user_id != OWNER_DISCORD_USER_ID
        or guild_id != PUBLIC_GUILD_ID
        or channel_id != PUBLIC_CHANNEL_ID
        or channel_id in LOCKED_PRIVATE_CHANNEL_IDS
        or _SNOWFLAKE_RE.fullmatch(owner_discord_user_id) is None
        or _SNOWFLAKE_RE.fullmatch(guild_id) is None
        or _SNOWFLAKE_RE.fullmatch(channel_id) is None
        or _digest(expected_prompt_sha256, "fixture prompt digest")
        != COMPLEX_CANARY_PROMPT_SHA256
    ):
        raise ValueError("fixture owner intent is invalid")

    writer_uid, writer_gid = _exact_nss(WRITER_USER, WRITER_GROUP)
    gateway_uid, gateway_gid = _exact_nss(GATEWAY_USER, GATEWAY_GROUP)
    edge_uid, edge_gid = _exact_nss(EDGE_USER, EDGE_GROUP)
    activation = load_activation_plan(WRITER_ACTIVATION_PLAN_PATH)
    if (
        activation.revision != release_sha
        or activation.identities.writer_uid != writer_uid
        or activation.identities.writer_gid != writer_gid
        or activation.identities.gateway_uid != gateway_uid
        or activation.identities.gateway_gid != gateway_gid
        or activation.digests.external_iam_policy_sha256
        != external_iam_policy_sha256
    ):
        raise RuntimeError("fixture activation/IAM binding diverged")
    release_artifact_sha256 = _activation_release_artifact(activation)
    writer_key = _public_key_provenance(
        DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY,
        expected_gid=edge_gid,
    )
    edge_key = _public_key_provenance(
        EDGE_RECEIPT_PUBLIC_KEY,
        expected_gid=writer_gid,
    )
    if writer_key["key_id"] == edge_key["key_id"]:
        raise RuntimeError("fixture Discord public keys must be distinct")
    config_binding, _writer_config = _config_sources(
        writer_gid=writer_gid,
        gateway_gid=gateway_gid,
        edge_gid=edge_gid,
    )
    if config_binding != {
        **{
            name: config_binding[name]
            for name in (
                "writer_config_file_sha256",
                "gateway_config_file_sha256",
                "edge_config_file_sha256",
            )
        },
        "writer_edge_receipt_public_key_file": str(EDGE_RECEIPT_PUBLIC_KEY),
        "writer_edge_receipt_public_key_id": edge_key["key_id"],
        "edge_writer_capability_public_key_file": str(
            DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY
        ),
        "edge_writer_capability_public_key_id": writer_key["key_id"],
        "edge_receipt_public_key_id": edge_key["key_id"],
    }:
        raise RuntimeError("fixture public keys differ from staged service configs")
    fixture = _fixture(
        release_sha=release_sha,
        release_artifact_sha256=release_artifact_sha256,
        canary_run_id=canary_run_id,
        valid_from_unix_ms=valid_from_unix_ms,
        valid_until_unix_ms=valid_until_unix_ms,
        owner_discord_user_id=owner_discord_user_id,
        guild_id=guild_id,
        channel_id=channel_id,
        writer_public_hex=writer_key["public_key_ed25519_hex"],
        edge_public_hex=edge_key["public_key_ed25519_hex"],
    )
    fixture_raw = _canonical_bytes(fixture)
    fixture_sha256 = _sha256_bytes(fixture_raw)
    service_states = _stopped_service_states()
    receipt_path = PUBLICATION_ROOT / f"{fixture_sha256}.json"
    unsigned = {
        "schema": FIXTURE_PUBLICATION_PLAN_SCHEMA,
        "release_sha": release_sha,
        "release_artifact_sha256": release_artifact_sha256,
        "activation_plan_sha256": activation.sha256,
        "external_iam_policy_sha256": external_iam_policy_sha256,
        "canary_run_id": canary_run_id,
        "valid_from_unix_ms": valid_from_unix_ms,
        "valid_until_unix_ms": valid_until_unix_ms,
        "owner_discord_user_id": owner_discord_user_id,
        "public_target": fixture["public_routeback"]["target"],
        "prompt_sha256": COMPLEX_CANARY_PROMPT_SHA256,
        "api_session_key_present": False,
        "fixture": fixture,
        "fixture_sha256": fixture_sha256,
        "fixture_path": str(DEFAULT_E2E_FIXTURE),
        "fixture_gid": gateway_gid,
        "writer_public_key": writer_key,
        "edge_public_key": edge_key,
        "config_binding": config_binding,
        "service_states": service_states,
        "publication_receipt_path": str(receipt_path),
        "invariants": {
            "services_started": False,
            "service_units_changed": False,
            "discord_dispatched": False,
            "iam_mutated": False,
            "private_keys_read": False,
            "api_session_secret_present": False,
        },
    }
    return {**unsigned, "plan_sha256": _sha256_json(unsigned)}


def _require_directory(path: Path, *, mode: int) -> None:
    item = path.lstat()
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != ROOT_UID
        or item.st_gid != ROOT_GID
        or stat.S_IMODE(item.st_mode) != mode
        or item.st_mode & 0o022
    ):
        raise RuntimeError("fixture publication directory is not root-controlled")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_publication_root() -> None:
    parent = PUBLICATION_ROOT.parent
    grandparent = parent.parent
    if not os.path.lexists(grandparent):
        raise RuntimeError("fixture publication parent hierarchy is absent")
    if not os.path.lexists(parent):
        os.mkdir(parent, _PUBLICATION_DIRECTORY_MODE)
        os.chown(parent, ROOT_UID, ROOT_GID, follow_symlinks=False)
        os.chmod(parent, _PUBLICATION_DIRECTORY_MODE, follow_symlinks=False)
        _fsync_directory(grandparent)
    _require_directory(parent, mode=_PUBLICATION_DIRECTORY_MODE)
    if not os.path.lexists(PUBLICATION_ROOT):
        os.mkdir(PUBLICATION_ROOT, _PUBLICATION_DIRECTORY_MODE)
        os.chown(
            PUBLICATION_ROOT,
            ROOT_UID,
            ROOT_GID,
            follow_symlinks=False,
        )
        os.chmod(PUBLICATION_ROOT, _PUBLICATION_DIRECTORY_MODE, follow_symlinks=False)
        _fsync_directory(parent)
    _require_directory(PUBLICATION_ROOT, mode=_PUBLICATION_DIRECTORY_MODE)


def _write_no_replace(path: Path, raw: bytes, *, uid: int, gid: int, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        os.fchown(descriptor, uid, gid)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("fixture publication write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _publish_or_validate(
    path: Path,
    raw: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    if os.path.lexists(path):
        existing, _item = _read_stable_file(
            path,
            maximum=_MAX_ARTIFACT_BYTES,
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
        )
        if existing != raw:
            raise RuntimeError("fixture publication no-replace collision diverged")
        return
    _write_no_replace(path, raw, uid=uid, gid=gid, mode=mode)
    existing, _item = _read_stable_file(
        path,
        maximum=_MAX_ARTIFACT_BYTES,
        expected_uid=uid,
        expected_gid=gid,
        allowed_modes=frozenset({mode}),
    )
    if existing != raw:
        raise RuntimeError("fixture publication readback diverged")


def _publication_receipt(
    plan: Mapping[str, Any],
    *,
    published_at_unix_ms: int,
    service_states_after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fixture_raw = _canonical_bytes(plan["fixture"])
    unsigned = {
        "schema": FIXTURE_PUBLICATION_RECEIPT_SCHEMA,
        "ok": True,
        "state": "base_fixture_published_services_stopped",
        "release_sha": plan["release_sha"],
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "activation_plan_sha256": plan["activation_plan_sha256"],
        "external_iam_policy_sha256": plan["external_iam_policy_sha256"],
        "canary_run_id": plan["canary_run_id"],
        "owner_discord_user_id": plan["owner_discord_user_id"],
        "public_target": plan["public_target"],
        "prompt_sha256": plan["prompt_sha256"],
        "api_session_key_present": False,
        "fixture_path": str(DEFAULT_E2E_FIXTURE),
        "fixture_gid": plan["fixture_gid"],
        "fixture_sha256": plan["fixture_sha256"],
        "fixture_file_sha256": _sha256_bytes(fixture_raw),
        "writer_public_key_id": plan["writer_public_key"]["key_id"],
        "edge_public_key_id": plan["edge_public_key"]["key_id"],
        "service_states_before": plan["service_states"],
        "service_states_after": dict(service_states_after),
        "services_started": False,
        "discord_dispatched": False,
        "iam_mutated": False,
        "private_keys_read": False,
        "api_session_secret_present": False,
        "approved_plan_sha256": plan["plan_sha256"],
        "publication_receipt_path": plan["publication_receipt_path"],
        "published_at_unix_ms": published_at_unix_ms,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def apply_fixture_publication(
    release_sha: str,
    approved_plan_sha256: str,
    *,
    external_iam_policy_sha256: str,
    canary_run_id: str,
    valid_from_unix_ms: int,
    valid_until_unix_ms: int,
    owner_discord_user_id: str,
    guild_id: str,
    channel_id: str,
    expected_prompt_sha256: str,
    clock: Clock = time.time,
) -> dict[str, Any]:
    geteuid = getattr(os, "geteuid", None)
    if sys.platform != "linux" or geteuid is None or geteuid() != ROOT_UID:
        raise PermissionError("fixture publication requires root Linux")
    approved_plan_sha256 = _digest(
        approved_plan_sha256,
        "approved fixture publication plan",
    )
    plan = build_fixture_publication_plan(
        release_sha,
        external_iam_policy_sha256=external_iam_policy_sha256,
        canary_run_id=canary_run_id,
        valid_from_unix_ms=valid_from_unix_ms,
        valid_until_unix_ms=valid_until_unix_ms,
        owner_discord_user_id=owner_discord_user_id,
        guild_id=guild_id,
        channel_id=channel_id,
        expected_prompt_sha256=expected_prompt_sha256,
    )
    if plan["plan_sha256"] != approved_plan_sha256:
        raise PermissionError("fixture publication approval differs from plan")
    receipt_path = Path(plan["publication_receipt_path"])
    fixture_raw = _canonical_bytes(plan["fixture"])
    if os.path.lexists(receipt_path):
        existing_fixture, _fixture_item = _read_stable_file(
            DEFAULT_E2E_FIXTURE,
            maximum=_MAX_ARTIFACT_BYTES,
            expected_uid=ROOT_UID,
            expected_gid=int(plan["fixture_gid"]),
            allowed_modes=frozenset({_FIXTURE_MODE}),
        )
        receipt_raw, _receipt_item = _read_stable_file(
            receipt_path,
            maximum=_MAX_ARTIFACT_BYTES,
            expected_uid=ROOT_UID,
            expected_gid=ROOT_GID,
            allowed_modes=frozenset({_PUBLICATION_RECEIPT_MODE}),
        )
        existing = _decode_mapping(
            receipt_raw,
            "fixture publication receipt",
        )
        published_at = existing.get("published_at_unix_ms")
        if (
            existing_fixture != fixture_raw
            or type(published_at) is not int
            or published_at < plan["valid_from_unix_ms"]
            or published_at >= plan["valid_until_unix_ms"]
            or existing
            != _publication_receipt(
                plan,
                published_at_unix_ms=published_at,
                service_states_after=plan["service_states"],
            )
        ):
            raise RuntimeError("fixture publication receipt diverged")
        return existing
    now_ms = int(clock() * 1000)
    if (
        not plan["valid_from_unix_ms"] <= now_ms < plan["valid_until_unix_ms"]
        or plan["valid_until_unix_ms"] - now_ms < _MINIMUM_REMAINING_WINDOW_MS
    ):
        raise RuntimeError("fixture publication window is not fresh enough")
    target_parent = DEFAULT_E2E_FIXTURE.parent
    _require_directory(target_parent, mode=0o755)
    _publish_or_validate(
        DEFAULT_E2E_FIXTURE,
        fixture_raw,
        uid=ROOT_UID,
        gid=int(plan["fixture_gid"]),
        mode=_FIXTURE_MODE,
    )
    states_after = _stopped_service_states()
    if states_after != plan["service_states"]:
        raise RuntimeError("service state drifted during fixture publication")
    _ensure_publication_root()
    receipt = _publication_receipt(
        plan,
        published_at_unix_ms=now_ms,
        service_states_after=states_after,
    )
    receipt_raw = _canonical_bytes(receipt)
    _publish_or_validate(
        receipt_path,
        receipt_raw,
        uid=ROOT_UID,
        gid=ROOT_GID,
        mode=_PUBLICATION_RECEIPT_MODE,
    )
    return receipt


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ValueError("invalid fixture publication CLI arguments")


class _StoreOnce(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        del parser, option_string
        if getattr(namespace, self.dest, None) is not None:
            raise ValueError("fixture publication CLI option was repeated")
        setattr(namespace, self.dest, values)


def _sha(value: str) -> str:
    return _digest(value, "CLI digest")


def _revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid revision")
    return value


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid positive integer") from exc
    if parsed < 1 or str(parsed) != value:
        raise argparse.ArgumentTypeError("invalid positive integer")
    return parsed


def _add_intent(parser: argparse.ArgumentParser) -> None:
    for name, kind in (
        ("release-sha", _revision),
        ("external-iam-policy-sha256", _sha),
        ("canary-run-id", _exact_uuid),
        ("valid-from-unix-ms", _positive),
        ("valid-until-unix-ms", _positive),
        ("owner-discord-user-id", str),
        ("guild-id", str),
        ("channel-id", str),
        ("expected-prompt-sha256", _sha),
    ):
        parser.add_argument(
            f"--{name}",
            required=True,
            default=None,
            type=kind,
            action=_StoreOnce,
        )


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(allow_abbrev=False)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_Parser,
    )
    plan = subparsers.add_parser("plan", allow_abbrev=False)
    _add_intent(plan)
    apply = subparsers.add_parser("apply", allow_abbrev=False)
    _add_intent(apply)
    apply.add_argument(
        "--approved-plan-sha256",
        required=True,
        default=None,
        type=_sha,
        action=_StoreOnce,
    )
    return parser


def _kwargs(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "external_iam_policy_sha256": arguments.external_iam_policy_sha256,
        "canary_run_id": arguments.canary_run_id,
        "valid_from_unix_ms": arguments.valid_from_unix_ms,
        "valid_until_unix_ms": arguments.valid_until_unix_ms,
        "owner_discord_user_id": arguments.owner_discord_user_id,
        "guild_id": arguments.guild_id,
        "channel_id": arguments.channel_id,
        "expected_prompt_sha256": arguments.expected_prompt_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "plan":
            result = build_fixture_publication_plan(
                arguments.release_sha,
                **_kwargs(arguments),
            )
        else:
            result = apply_fixture_publication(
                arguments.release_sha,
                arguments.approved_plan_sha256,
                **_kwargs(arguments),
            )
        sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
        sys.stdout.buffer.flush()
        return 0
    except BaseException:
        unsigned = {
            "schema": FIXTURE_PUBLICATION_FAILURE_SCHEMA,
            "ok": False,
            "error_code": "fixture_publication_failed",
        }
        failure = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        sys.stdout.buffer.write(_canonical_bytes(failure) + b"\n")
        sys.stdout.buffer.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLEX_CANARY_PROMPT",
    "COMPLEX_CANARY_PROMPT_SHA256",
    "EDGE_RECEIPT_PUBLIC_KEY",
    "FIXTURE_PUBLICATION_FAILURE_SCHEMA",
    "FIXTURE_PUBLICATION_PLAN_SCHEMA",
    "FIXTURE_PUBLICATION_RECEIPT_SCHEMA",
    "OWNER_DISCORD_USER_ID",
    "PUBLIC_CHANNEL_ID",
    "PUBLIC_GUILD_ID",
    "LOCKED_PRIVATE_CHANNEL_IDS",
    "PUBLICATION_ROOT",
    "apply_fixture_publication",
    "build_fixture_publication_plan",
    "main",
]
