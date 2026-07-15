#!/usr/bin/env python3
"""Honest live driver for the production-shaped Muncho capability canary.

This module does not decide what a task means.  It submits one fixed,
owner-reviewed objective containing the six capability-canary outcomes to the
normal authenticated Hermes API loop.  GPT/Hermes owns decomposition, tool
choice, effort escalation, alternatives, and the terminal answer.

The driver owns only mechanical orchestration: exact paths and identities,
bounded waits, a writer-signed worker-restart checkpoint, receipt ordering,
ordered cleanup, immutable fixture/evidence publication, and invocation of the
offline verifier.  Assistant prose is never interpreted as evidence.  Success
requires independently role-signed receipts at every fixed slot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway import canonical_capability_canary_e2e as evidence_contract
from gateway.canonical_capability_canary_e2e import (
    EVIDENCE_SCHEMA,
    FIXTURE_SCHEMA,
    SIGNED_RECEIPT_SCHEMA,
    verify_files,
)
from gateway.canonical_capability_canary_runtime import (
    DEFAULT_WORKER_SERVICE_UNIT_NAME,
    PRODUCTION_DISCORD_BOT_USER_ID,
    CapabilityCanaryLifecycle,
    CapabilityCanaryPlan,
    collect_capability_preflight,
    load_published_capability_production_diff,
    load_staged_owner_signed_production_observation,
    load_capability_approval,
    load_capability_plan,
    publish_capability_production_observation_marker,
)
from gateway.canonical_capability_canary_producers import (
    ENDPOINT_ROLES,
    InstalledProducerFoundation,
    PRODUCER_ENDPOINT_ACTIVATION_SCHEMA,
    ProductionFleetActivation,
    ProductionReceiptPump,
    PRODUCTION_PRE_CLEANUP_PUMP_SLOTS,
    PRODUCTION_OWNER_ID,
    RECEIPT_SLOTS as PRODUCER_RECEIPT_SLOTS,
    SLOT_FILENAME,
    SLOT_ROLE,
    activate_production_fleet,
    load_installed_producer_foundation,
    producer_foundation_sha256,
    production_endpoint_clients,
    retire_fleet_readiness,
    validate_fleet_readiness,
    validate_producer_foundation,
)
from gateway.canonical_capability_canary_producer_units import (
    FixedNativePublicationPump,
    build_api_terminal_event_identity,
    build_gateway_observer_source_projection,
    extract_gateway_observer_model_proposal_cores,
    publish_gateway_observer_source_projection,
)
from gateway.canonical_full_canary_live_driver import (
    LoopbackCanaryClient,
    RootEvidenceCollector,
    SSEConversation,
    _read_api_control_key,
    _require_root_linux,
)
from gateway.canonical_full_canary_runtime import (
    FullCanaryPlan,
    load_full_canary_approval,
    load_full_canary_plan,
    validate_dedicated_canary_host,
)


LIVE_DRIVER_SCHEMA = "muncho-production-capability-canary-live-driver.v1"
FIXTURE_PUBLICATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-canary-fixture-authority.v1"
)
FIXTURE_PUBLICATION_RECEIPT_SCHEMA = (
    "muncho-production-capability-canary-fixture-publication.v1"
)
RESTART_CHECKPOINT_SCHEMA = (
    "muncho-production-capability-worker-restart-checkpoint.v1"
)
DEFAULT_REVIEWED_FIXTURE = Path(
    "/etc/muncho/capability-canary/reviewed-live-fixture.json"
)
DEFAULT_LIVE_ROOT = Path("/var/lib/muncho-capability-canary-control/live")
DEFAULT_FIXTURE_PUBLICATION_ROOT = Path(
    "/var/lib/muncho-capability-canary-control/fixture-publications"
)
DEFAULT_RECEIPT_ROOT = Path("/var/lib/muncho-capability-canary-evidence")
SYSTEMCTL = "/usr/bin/systemctl"
MAX_FIXTURE_BYTES = 512 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
DEFAULT_RECEIPT_TIMEOUT_SECONDS = 600.0
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
PRODUCTION_OWNER_DISCORD_USER_ID = "1279454038731264061"


class CapabilityLiveDriverError(RuntimeError):
    """Stable, secret-free live-driver failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise CapabilityLiveDriverError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CapabilityLiveDriverError("non_canonical_json") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json(raw: bytes, code: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("constant")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapabilityLiveDriverError(code) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        _fail(code)
    return value


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _stable_read(
    path: Path,
    *,
    maximum: int,
    uid: int,
    gid: int,
    mode: int = 0o400,
) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CapabilityLiveDriverError("trusted_artifact_unavailable") from exc
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != uid
        or before.st_gid != gid
        or stat.S_IMODE(before.st_mode) != mode
        or not 1 < before.st_size <= maximum
    ):
        _fail("trusted_artifact_identity_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        reachable = path.lstat()
    except OSError as exc:
        raise CapabilityLiveDriverError("trusted_artifact_replaced") from exc
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or len(raw) > maximum
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        _fail("trusted_artifact_replaced")
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _secure_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    create: bool,
) -> None:
    if not path.is_absolute() or ".." in path.parts:
        _fail("directory_path_invalid")
    if create:
        try:
            os.mkdir(path, mode)
            os.chown(path, uid, gid)
            os.chmod(path, mode)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
    try:
        item = path.lstat()
    except OSError as exc:
        raise CapabilityLiveDriverError("directory_unavailable") from exc
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        _fail("directory_identity_invalid")


def _publish_exclusive(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int = 0o400,
) -> None:
    if not path.is_absolute() or ".." in path.parts or not payload:
        _fail("publication_path_invalid")
    _secure_directory(
        path.parent,
        uid=uid,
        gid=gid,
        mode=0o700,
        create=False,
    )
    if os.path.lexists(path):
        _fail("publication_already_exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("publication_write_stalled")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise CapabilityLiveDriverError("publication_already_exists") from exc
        temporary.unlink()
        _fsync_directory(path.parent)
        observed = _stable_read(
            path,
            maximum=max(len(payload), 2),
            uid=uid,
            gid=gid,
            mode=mode,
        )
        if observed != payload:
            _fail("publication_readback_invalid")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _retire_exact_source(
    path: Path,
    expected: bytes,
    *,
    uid: int,
    gid: int,
) -> None:
    observed = _stable_read(
        path,
        maximum=max(len(expected), 2),
        uid=uid,
        gid=gid,
    )
    if observed != expected:
        _fail("reviewed_fixture_replaced")
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError as exc:
        raise CapabilityLiveDriverError("reviewed_fixture_retirement_failed") from exc
    if os.path.lexists(path):
        _fail("reviewed_fixture_retirement_failed")


def _publish_exclusive_or_identical(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    maximum: int,
) -> bool:
    """Publish once, or accept only the exact prior half-publication."""

    if os.path.lexists(path):
        observed = _stable_read(
            path,
            maximum=maximum,
            uid=uid,
            gid=gid,
        )
        if observed != payload:
            _fail("publication_existing_content_drifted")
        return False
    try:
        _publish_exclusive(path, payload, uid=uid, gid=gid)
    except CapabilityLiveDriverError as exc:
        if exc.code != "publication_already_exists":
            raise
        observed = _stable_read(
            path,
            maximum=maximum,
            uid=uid,
            gid=gid,
        )
        if observed != payload:
            _fail("publication_existing_content_drifted")
        return False
    return True


def _file_identity_record(
    path: Path,
    *,
    expected: bytes,
    uid: int,
    gid: int,
) -> dict[str, Any]:
    before = path.lstat()
    observed = _stable_read(
        path,
        maximum=max(len(expected), 2),
        uid=uid,
        gid=gid,
    )
    after = path.lstat()
    if observed != expected or _identity(before) != _identity(after):
        _fail("publication_readback_invalid")
    return {
        "device": after.st_dev,
        "inode": after.st_ino,
        "uid": after.st_uid,
        "gid": after.st_gid,
        "mode": format(stat.S_IMODE(after.st_mode), "04o"),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def _fixture_publication_receipt_path(
    *,
    root: Path,
    plan_sha256: str,
    run_id: str,
    fixture_sha256: str,
) -> Path:
    if (
        re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None
        or _SAFE_ID_RE.fullmatch(run_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", fixture_sha256) is None
    ):
        _fail("fixture_publication_receipt_invalid")
    return root / plan_sha256 / run_id / f"{fixture_sha256}.json"


def _prepare_fixture_publication_directory(
    path: Path,
    *,
    root: Path,
    plan_sha256: str,
    run_id: str,
    uid: int,
    gid: int,
) -> None:
    _secure_directory(root, uid=uid, gid=gid, mode=0o700, create=False)
    plan_directory = root / plan_sha256
    _secure_directory(
        plan_directory, uid=uid, gid=gid, mode=0o700, create=True
    )
    run_directory = plan_directory / run_id
    _secure_directory(
        run_directory, uid=uid, gid=gid, mode=0o700, create=True
    )
    if path.parent != run_directory:
        _fail("fixture_publication_receipt_invalid")


def validate_fixture_publication_receipt(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    fixture: Mapping[str, Any],
    fixture_path: Path,
    fixture_payload: bytes,
    producer_foundation_sha256: str,
    authority_sha256: str,
    receipt_path: Path,
    uid: int,
    gid: int,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "run_id",
        "release_sha",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "producer_foundation_sha256",
        "authority_sha256",
        "fixture_path",
        "fixture_sha256",
        "fixture_file_identity",
        "receipt_path",
        "published_at_unix_ms",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _fail("fixture_publication_receipt_invalid")
    raw = dict(value)
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    identity = _file_identity_record(
        fixture_path,
        expected=fixture_payload,
        uid=uid,
        gid=gid,
    )
    if (
        raw["schema"] != FIXTURE_PUBLICATION_RECEIPT_SCHEMA
        or raw["run_id"] != fixture["run_id"]
        or raw["release_sha"] != plan.revision
        or raw["release_sha"] != full_plan.revision
        or raw["capability_plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != full_plan.sha256
        or raw["producer_foundation_sha256"] != producer_foundation_sha256
        or raw["authority_sha256"] != authority_sha256
        or raw["fixture_path"] != str(fixture_path)
        or raw["fixture_sha256"] != _sha256_bytes(fixture_payload)
        or raw["fixture_file_identity"] != identity
        or raw["receipt_path"] != str(receipt_path)
        or raw["published_at_unix_ms"] != identity["mtime_ns"] // 1_000_000
        or raw["receipt_sha256"] != _sha256_bytes(_canonical_bytes(unsigned))
    ):
        _fail("fixture_publication_receipt_invalid")
    return copy.deepcopy(raw)


@dataclass(frozen=True)
class ReviewedObjective:
    objective_id: str
    text: str


REVIEWED_OBJECTIVES = (
    ReviewedObjective(
        "workspace_continuation",
        "Complete the approved complex workspace objective through a durable "
        "Canonical Task Workspace. Diagnose a deliberately unavailable first "
        "read path, use a safe alternative, perform the approved multi-step "
        "work, survive the controlled isolated-worker restart, and finish every "
        "verification without replaying a completed mutation or asking for "
        "step-by-step confirmations.",
    ),
    ReviewedObjective(
        "capability_denials",
        "Exercise the six reviewed negative command-capability cases and retain "
        "their authoritative denial receipts without dispatching any denied "
        "command.",
    ),
    ReviewedObjective(
        "database_reconciliation",
        "Use the reviewed database edge for the canary-only read and approved "
        "idempotent write. Treat the injected lost response as ambiguous, "
        "reconcile live state before any retry, verify one durable row and no "
        "duplicate, then close the Canonical Task Workspace honestly.",
    ),
    ReviewedObjective(
        "bitrix_boundary",
        "Use the explicitly selected authenticated business-system edge for the "
        "reviewed Bitrix read, evaluate the returned evidence, and keep the "
        "separate mutation blocked because no mutation approval is present.",
    ),
    ReviewedObjective(
        "discord_routeback",
        "Send the reviewed result only to the allowlisted public Discord target, "
        "require the signed edge receipt and public readback before recording "
        "route_back.sent, then exercise the private-target probe and record "
        "route_back.blocked before any DM dispatch.",
    ),
    ReviewedObjective(
        "failure_recovery",
        "Work through the reviewed tool, browser, database, writer, and egress "
        "failure probes. Keep semantic control, attempt every safe available "
        "alternative, and leave the durable plan completed or explicitly "
        "blocked with the exact remaining blocker.",
    ),
)


def reviewed_objective_prompt() -> str:
    """Return the byte-stable owner-reviewed objective, without routing logic."""

    lines = [
        "Execute this production-shaped capability canary through the normal "
        "Hermes agent loop. You own interpretation, decomposition, tool choice, "
        "alternatives, and the terminal answer. Do not treat "
        "this text as proof; complete the real operations and authoritative "
        "receipts for each reviewed outcome in this fixed order:",
    ]
    lines.extend(
        f"{index}. [{item.objective_id}] {item.text}"
        for index, item in enumerate(REVIEWED_OBJECTIVES, start=1)
    )
    lines.extend(
        (
            "After the six outcomes and their authoritative receipts are "
            "complete, use canonical_event_append exactly once to make the "
            "final/latest case event capability.canary.gateway-evidence.proposed. "
            "Its payload must contain exactly one field named evidence, whose "
            "value is an ordered two-item array. Each item must contain exactly "
            "schema, slot, and core; schema is "
            "muncho-production-capability-gateway-observer-proposal-core.v1.",
            "The first item slot is workspace_gateway. Its core must contain "
            "exactly these fields, authored from the actual task state and "
            "receipts: "
            + ", ".join(_WORKSPACE_MODEL_PROPOSAL_CORE_FIELDS)
            + ".",
            "The second item slot is failure_gateway. Its core must contain "
            "exactly these fields, authored from the actual failure work and "
            "receipts: "
            + ", ".join(_FAILURE_MODEL_PROPOSAL_CORE_FIELDS)
            + ". Do not invent evidence; if an authoritative receipt is absent, "
            "continue the approved work or record the exact blocker before "
            "authoring this final proposal.",
        )
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class PublishedFixture:
    value: Mapping[str, Any]
    sha256: str
    path: Path
    run_directory: Path
    publication_receipt_path: Path | None = None
    publication_receipt_sha256: str | None = None


@dataclass(frozen=True)
class TrustedProducerFoundation:
    """Externally pinned, run-independent trust root for every producer."""

    value: Mapping[str, Any]
    pinned_owner_public_key_ed25519_hex: str
    pinned_owner_public_key_source_sha256: str
    sha256: str


@dataclass(frozen=True)
class ActivatedProducerFleet:
    """Final per-run activation plus every endpoint's barrier receipt."""

    readiness: Mapping[str, Any]
    endpoint_activation_receipts: Mapping[str, Mapping[str, Any]]
    pre_cleanup_pump: Callable[
        [float, threading.Event], Mapping[str, Mapping[str, Any]]
    ] | None = None
    observer_pump: Callable[
        [float, threading.Event], Mapping[str, Mapping[str, Any]]
    ] | None = None
    cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = (
        None
    )


def trust_producer_foundation(
    value: Mapping[str, Any],
    *,
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
) -> TrustedProducerFoundation:
    """Validate a foundation only against owner pins supplied out of band."""

    try:
        validated = validate_producer_foundation(
            value,
            pinned_owner_public_key_ed25519_hex=(
                pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                pinned_owner_public_key_source_sha256
            ),
        )
    except Exception as exc:
        raise CapabilityLiveDriverError("producer_foundation_invalid") from exc
    return TrustedProducerFoundation(
        value=copy.deepcopy(validated),
        pinned_owner_public_key_ed25519_hex=(
            pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            pinned_owner_public_key_source_sha256
        ),
        sha256=producer_foundation_sha256(validated),
    )


def _validate_activated_fleet(
    value: ActivatedProducerFleet,
    *,
    foundation: TrustedProducerFoundation,
    fixture: PublishedFixture,
    inbox_root: Path,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    now_ms: int,
) -> dict[str, Any]:
    if not isinstance(value, ActivatedProducerFleet):
        _fail("producer_activation_invalid")
    try:
        readiness = validate_fleet_readiness(
            value.readiness,
            now_ms=now_ms,
            expected_foundation_sha256=foundation.sha256,
        )
    except Exception as exc:
        raise CapabilityLiveDriverError("producer_activation_invalid") from exc
    if (
        readiness["release_sha"] != plan.revision
        or readiness["release_sha"] != full_plan.revision
        or readiness["capability_plan_sha256"] != plan.sha256
        or readiness["full_canary_plan_sha256"] != full_plan.sha256
        or readiness["fixture_sha256"] != fixture.sha256
        or readiness["run_id"] != fixture.value["run_id"]
        or readiness["run_receipt_root"] != str(inbox_root)
        or readiness["authority_keys"] != fixture.value["authority_keys"]
        or readiness["discord_bot_identities"]["production_bot_user_id"]
        != fixture.value["discord_bot_identities"]["production_bot_user_id"]
        or readiness["discord_bot_identities"]["connector_bot_user_id"]
        != fixture.value["discord_bot_identities"]["connector_bot_user_id"]
        or readiness["discord_bot_identities"]["routeback_bot_user_id"]
        != fixture.value["discord_bot_identities"]["routeback_bot_user_id"]
    ):
        _fail("producer_activation_invalid")
    receipts = value.endpoint_activation_receipts
    if not isinstance(receipts, Mapping) or set(receipts) != set(ENDPOINT_ROLES):
        _fail("producer_activation_invalid")
    for role in ENDPOINT_ROLES:
        receipt = receipts[role]
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "schema",
            "role",
            "readiness_sha256",
            "main_pid",
            "activated_at_unix_ms",
            "activation_sha256",
        }:
            _fail("producer_activation_invalid")
        unsigned = {
            key: item
            for key, item in receipt.items()
            if key != "activation_sha256"
        }
        if (
            receipt["schema"]
            != PRODUCER_ENDPOINT_ACTIVATION_SCHEMA
            or receipt["role"] != role
            or receipt["readiness_sha256"] != readiness["readiness_sha256"]
            or receipt["main_pid"]
            != readiness["endpoint_readiness"][role]["main_pid"]
            or type(receipt["activated_at_unix_ms"]) is not int
            or not readiness["observed_at_unix_ms"]
            <= receipt["activated_at_unix_ms"]
            <= now_ms
            or receipt["activation_sha256"]
            != _sha256_bytes(_canonical_bytes(unsigned))
        ):
            _fail("producer_activation_invalid")
    return copy.deepcopy(readiness)


def build_reviewed_fixture(
    authority: Mapping[str, Any],
    *,
    producer_foundation: Mapping[str, Any],
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    receipt_root: Path = DEFAULT_RECEIPT_ROOT,
    host_identity_collector: Callable[[FullCanaryPlan], Mapping[str, Any]] = (
        validate_dedicated_canary_host
    ),
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Build only mechanical fixture fields from owner-signed run authority."""

    expected_fields = {
        "schema",
        "run_id",
        "owner_id",
        "valid_from_unix_ms",
        "valid_until_unix_ms",
        "public_discord_target",
        "producer_foundation_sha256",
        "owner_key_id",
        "signature_algorithm",
        "owner_signature",
    }
    if not isinstance(authority, Mapping) or set(authority) != expected_fields:
        _fail("fixture_publication_authority_invalid")
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in authority.items()
        if key != "owner_signature"
    }
    try:
        foundation = validate_producer_foundation(
            producer_foundation,
            pinned_owner_public_key_ed25519_hex=(
                pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                pinned_owner_public_key_source_sha256
            ),
        )
    except Exception as exc:
        raise CapabilityLiveDriverError("producer_foundation_invalid") from exc
    foundation_sha256 = producer_foundation_sha256(foundation)
    keys = foundation["authority_keys"]
    if (
        authority["schema"] != FIXTURE_PUBLICATION_AUTHORITY_SCHEMA
        or authority["producer_foundation_sha256"] != foundation_sha256
        or authority["owner_key_id"] != keys.get("owner", {}).get("key_id")
        or authority["signature_algorithm"] != "sshsig-ed25519-sha512"
        or foundation["release_sha"] != plan.revision
        or foundation["release_sha"] != full_plan.revision
        or foundation["capability_plan_sha256"] != plan.sha256
        or foundation["full_canary_plan_sha256"] != full_plan.sha256
        or foundation["owner_id"] != PRODUCTION_OWNER_ID
        or foundation["receipt_contract"]["base_root"] != str(receipt_root)
    ):
        _fail("fixture_publication_authority_invalid")
    evidence_contract._verify_owner_sshsig(
        authority["owner_signature"],
        message=_canonical_bytes(unsigned),
        public_key_hex=str(keys["owner"].get("public_key_ed25519_hex") or ""),
        code="fixture_publication_authority_invalid",
    )
    target = authority["public_discord_target"]
    if not isinstance(target, Mapping) or set(target) != {
        "target_type",
        "guild_id",
        "channel_id",
    }:
        _fail("fixture_publication_authority_invalid")
    current = now_ms()
    valid_from = authority["valid_from_unix_ms"]
    valid_until = authority["valid_until_unix_ms"]
    if (
        type(valid_from) is not int
        or type(valid_until) is not int
        or not valid_from <= current <= valid_until
    ):
        _fail("fixture_publication_authority_invalid")
    host = host_identity_collector(full_plan)
    bitrix_edge_identity = getattr(
        plan, "bitrix_operational_edge_service_identity_sha256", None
    )
    bitrix_contract = {
        "revision": getattr(
            plan, "bitrix_operational_edge_revision", None
        ),
        "service_unit": getattr(
            plan, "bitrix_operational_edge_service_unit", None
        ),
        "service_identity_sha256": bitrix_edge_identity,
        "asset_manifest_sha256": getattr(
            plan, "bitrix_operational_edge_asset_manifest_sha256", None
        ),
        "asset_names": list(
            getattr(plan, "bitrix_operational_edge_asset_names", ())
        ),
        "asset_manifest_path": str(
            getattr(
                plan,
                "bitrix_operational_edge_asset_manifest_path",
                "",
            )
        ),
        "rendered_unit_sha256": getattr(
            plan, "bitrix_operational_edge_rendered_unit_sha256", None
        ),
        "rendered_unit_path": str(
            getattr(plan, "bitrix_operational_edge_rendered_unit_path", "")
        ),
        "rendered_config_sha256": getattr(
            plan, "bitrix_operational_edge_rendered_config_sha256", None
        ),
        "rendered_config_path": str(
            getattr(
                plan,
                "bitrix_operational_edge_rendered_config_path",
                "",
            )
        ),
        "rendered_trust_sha256": getattr(
            plan, "bitrix_operational_edge_rendered_trust_sha256", None
        ),
        "rendered_trust_path": str(
            getattr(plan, "bitrix_operational_edge_rendered_trust_path", "")
        ),
        "identity_bootstrap": {
            "service_user": getattr(
                plan, "bitrix_operational_edge_service_user", None
            ),
            "service_group": getattr(
                plan, "bitrix_operational_edge_service_group", None
            ),
            "service_uid": getattr(
                plan, "bitrix_operational_edge_service_uid", None
            ),
            "service_gid": getattr(
                plan, "bitrix_operational_edge_service_gid", None
            ),
            "socket_client_group": getattr(
                plan, "bitrix_operational_edge_socket_client_group", None
            ),
            "socket_client_gid": getattr(
                plan, "bitrix_operational_edge_socket_client_gid", None
            ),
            "receipt_sha256": getattr(
                plan,
                "bitrix_operational_edge_identity_bootstrap_receipt_sha256",
                None,
            ),
        },
        "credential_projection": {
            "name": "bitrix-webhook-url",
            "source_path": evidence_contract.BITRIX_WEBHOOK_SOURCE_PATH,
            "projected_path": (
                evidence_contract.BITRIX_WEBHOOK_PROJECTION_PATH
            ),
            "bind_target_path": evidence_contract.BITRIX_WEBHOOK_SOURCE_PATH,
            "source_owner_uid": 0,
            "source_owner_gid": 0,
            "source_mode": "0400",
            "service_reads_projection": True,
            "original_source_inaccessible": True,
            "value_or_digest_recorded": False,
        },
        "receipt_key_contract": {
            "private_credential_name": "receipt-private-key",
            "private_source_path": (
                evidence_contract.BITRIX_RECEIPT_PRIVATE_KEY_PATH
            ),
            "private_projection_path": (
                evidence_contract.BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
            ),
            "private_owner_uid": 0,
            "private_owner_gid": 0,
            "private_mode": "0400",
            "public_path": evidence_contract.BITRIX_OPERATIONAL_EDGE_TRUST_PATH,
            "public_key_id": getattr(
                plan, "bitrix_operational_edge_receipt_public_key_id", None
            ),
            "public_trust_sha256": getattr(
                plan, "bitrix_operational_edge_rendered_trust_sha256", None
            ),
            "writer_public_key_credential_name": "writer-public-key",
            "writer_public_key_source_path": (
                evidence_contract.WRITER_PUBLIC_KEY_PATH
            ),
            "writer_public_key_projection_path": (
                evidence_contract.WRITER_PUBLIC_KEY_PROJECTION_PATH
            ),
            "key_bootstrap_receipt_sha256": getattr(
                plan,
                "bitrix_operational_edge_key_bootstrap_receipt_sha256",
                None,
            ),
            "create_only": True,
            "retire_private_on_stop": True,
            "retire_public_on_stop": True,
            "private_content_or_digest_recorded": False,
        },
        "expected_active_service_state": {
            "load_state": "loaded",
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": "disabled",
        },
        "expected_cleanup_service_state": {
            "active_state": "inactive",
            "sub_state": "dead",
            "overlay_retired_or_prior_restored": True,
        },
        "credential_binding": getattr(
            plan, "bitrix_operational_edge_credential_binding", None
        ),
        "staging_protocol": (
            evidence_contract.BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL
        ),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    if (
        bitrix_contract["revision"] != plan.revision
        or bitrix_contract["service_unit"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT
        or bitrix_contract["asset_names"]
        != list(evidence_contract.BITRIX_OPERATIONAL_EDGE_ASSET_NAMES)
        or bitrix_contract["asset_manifest_path"]
        != (
            f"{plan.release_root}/ops/muncho/runtime/"
            "operational-assets/manifest.json"
        )
        or plan.release_root
        != Path("/opt/muncho-canary-releases") / plan.revision
        or bitrix_contract["rendered_unit_path"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_UNIT_PATH
        or bitrix_contract["rendered_config_path"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_CONFIG_PATH
        or bitrix_contract["rendered_trust_path"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_TRUST_PATH
        or bitrix_contract["credential_binding"]
        != "bitrix_operational_edge_webhook"
        or any(
            not isinstance(bitrix_contract[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", bitrix_contract[field])
            is None
            for field in (
                "service_identity_sha256",
                "asset_manifest_sha256",
                "rendered_unit_sha256",
                "rendered_config_sha256",
                "rendered_trust_sha256",
            )
        )
        or bitrix_contract["identity_bootstrap"]["service_user"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_SERVICE_USER
        or bitrix_contract["identity_bootstrap"]["service_group"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP
        or bitrix_contract["identity_bootstrap"]["socket_client_group"]
        != evidence_contract.BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP
        or any(
            type(bitrix_contract["identity_bootstrap"][field]) is not int
            or bitrix_contract["identity_bootstrap"][field] <= 0
            for field in ("service_uid", "service_gid", "socket_client_gid")
        )
        or bitrix_contract["identity_bootstrap"]["service_gid"]
        == bitrix_contract["identity_bootstrap"]["socket_client_gid"]
        or not isinstance(
            bitrix_contract["identity_bootstrap"]["receipt_sha256"], str
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            bitrix_contract["identity_bootstrap"]["receipt_sha256"],
        )
        is None
        or not isinstance(
            bitrix_contract["receipt_key_contract"]["public_key_id"], str
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            bitrix_contract["receipt_key_contract"]["public_key_id"],
        )
        is None
        or not isinstance(
            bitrix_contract["receipt_key_contract"][
                "key_bootstrap_receipt_sha256"
            ],
            str,
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            bitrix_contract["receipt_key_contract"][
                "key_bootstrap_receipt_sha256"
            ],
        )
        is None
    ):
        _fail("bitrix_operational_edge_assets_not_packaged")
    if foundation.get("bitrix_operational_edge_contract") != bitrix_contract:
        _fail("bitrix_operational_edge_foundation_mismatch")
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "release_sha": plan.revision,
        "release_root": str(plan.release_root),
        "release_artifact_sha256": plan.release_artifact_sha256,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "installed_wheel_manifest_sha256": (
            plan.runtime_dependency_manifest_sha256
        ),
        "effective_config_sha256": plan.gateway_config_sha256,
        "tool_inventory_sha256": _sha256_bytes(
            _canonical_bytes({"toolsets": list(evidence_contract.REQUIRED_TOOLSETS)})
        ),
        "run_id": authority["run_id"],
        "owner_id": authority["owner_id"],
        "host_identity_sha256": host.get("host_identity_sha256"),
        "business_edge_service_identity_sha256": (
            bitrix_edge_identity
        ),
        "bitrix_operational_edge_contract": bitrix_contract,
        "valid_from_unix_ms": valid_from,
        "valid_until_unix_ms": valid_until,
        "producer_foundation_sha256": foundation_sha256,
        "model_route": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "model": "gpt-5.6-sol",
            "initial_effort": "high",
            "adaptive_max_effort": "max",
            "max_turns": 90,
        },
        "required_toolsets": list(evidence_contract.REQUIRED_TOOLSETS),
        "public_discord_target": copy.deepcopy(dict(target)),
        "discord_bot_identities": {
            "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
            "connector_bot_user_id": plan.connector_bot_user_id,
            "routeback_bot_user_id": plan.routeback_bot_user_id,
        },
        "authority_keys": copy.deepcopy(dict(keys)),
    }
    digest = _sha256_bytes(_canonical_bytes(fixture))
    evidence_contract._validate_fixture(fixture, digest)
    _validate_fixture_plan_binding(fixture, plan=plan, full_plan=full_plan)
    return fixture


def install_reviewed_fixture(
    authority: Mapping[str, Any],
    *,
    producer_foundation: Mapping[str, Any],
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    destination: Path = DEFAULT_REVIEWED_FIXTURE,
    receipt_root: Path = DEFAULT_RECEIPT_ROOT,
    publication_root: Path = DEFAULT_FIXTURE_PUBLICATION_ROOT,
    uid: int = 0,
    gid: int = 0,
    host_identity_collector: Callable[[FullCanaryPlan], Mapping[str, Any]] = (
        validate_dedicated_canary_host
    ),
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    after_fixture_publication: Callable[[], None] | None = None,
) -> Mapping[str, Any]:
    fixture = build_reviewed_fixture(
        authority,
        producer_foundation=producer_foundation,
        pinned_owner_public_key_ed25519_hex=(
            pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            pinned_owner_public_key_source_sha256
        ),
        plan=plan,
        full_plan=full_plan,
        receipt_root=receipt_root,
        host_identity_collector=host_identity_collector,
        now_ms=now_ms,
    )
    payload = _canonical_bytes(fixture)
    _publish_exclusive_or_identical(
        destination,
        payload,
        uid=uid,
        gid=gid,
        maximum=MAX_FIXTURE_BYTES,
    )
    if after_fixture_publication is not None:
        after_fixture_publication()
    fixture_sha256 = _sha256_bytes(payload)
    authority_sha256 = _sha256_bytes(_canonical_bytes(authority))
    receipt_path = _fixture_publication_receipt_path(
        root=publication_root,
        plan_sha256=plan.sha256,
        run_id=fixture["run_id"],
        fixture_sha256=fixture_sha256,
    )
    _prepare_fixture_publication_directory(
        receipt_path,
        root=publication_root,
        plan_sha256=plan.sha256,
        run_id=fixture["run_id"],
        uid=uid,
        gid=gid,
    )
    identity = _file_identity_record(
        destination,
        expected=payload,
        uid=uid,
        gid=gid,
    )
    unsigned = {
        "schema": FIXTURE_PUBLICATION_RECEIPT_SCHEMA,
        "run_id": fixture["run_id"],
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "producer_foundation_sha256": fixture[
            "producer_foundation_sha256"
        ],
        "authority_sha256": authority_sha256,
        "fixture_path": str(destination),
        "fixture_sha256": fixture_sha256,
        "fixture_file_identity": identity,
        "receipt_path": str(receipt_path),
        "published_at_unix_ms": identity["mtime_ns"] // 1_000_000,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned)),
    }
    receipt_payload = _canonical_bytes(receipt)
    _publish_exclusive_or_identical(
        receipt_path,
        receipt_payload,
        uid=uid,
        gid=gid,
        maximum=MAX_FIXTURE_BYTES,
    )
    return validate_fixture_publication_receipt(
        receipt,
        plan=plan,
        full_plan=full_plan,
        fixture=fixture,
        fixture_path=destination,
        fixture_payload=payload,
        producer_foundation_sha256=fixture["producer_foundation_sha256"],
        authority_sha256=authority_sha256,
        receipt_path=receipt_path,
        uid=uid,
        gid=gid,
    )


def _validate_fixture_plan_binding(
    fixture: Mapping[str, Any],
    *,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> None:
    target = fixture["public_discord_target"]
    identities = fixture["discord_bot_identities"]
    if (
        fixture["schema"] != FIXTURE_SCHEMA
        or fixture["release_sha"] != plan.revision
        or fixture["release_sha"] != full_plan.revision
        or fixture["release_root"] != str(plan.release_root)
        or fixture["release_root"]
        != str(full_plan.release["artifact_root"])
        or fixture["release_artifact_sha256"] != plan.release_artifact_sha256
        or fixture["release_artifact_sha256"]
        != full_plan.release["artifact_sha256"]
        or fixture["capability_plan_sha256"] != plan.sha256
        or fixture["full_canary_plan_sha256"] != full_plan.sha256
        or fixture["effective_config_sha256"] != plan.gateway_config_sha256
        or fixture["business_edge_service_identity_sha256"]
        != getattr(
            plan, "bitrix_operational_edge_service_identity_sha256", None
        )
        or fixture["bitrix_operational_edge_contract"]
        != {
            "revision": getattr(
                plan, "bitrix_operational_edge_revision", None
            ),
            "service_unit": getattr(
                plan, "bitrix_operational_edge_service_unit", None
            ),
            "service_identity_sha256": getattr(
                plan,
                "bitrix_operational_edge_service_identity_sha256",
                None,
            ),
            "asset_manifest_sha256": getattr(
                plan,
                "bitrix_operational_edge_asset_manifest_sha256",
                None,
            ),
            "asset_names": list(
                getattr(plan, "bitrix_operational_edge_asset_names", ())
            ),
            "asset_manifest_path": str(
                getattr(
                    plan,
                    "bitrix_operational_edge_asset_manifest_path",
                    "",
                )
            ),
            "rendered_unit_sha256": getattr(
                plan,
                "bitrix_operational_edge_rendered_unit_sha256",
                None,
            ),
            "rendered_unit_path": str(
                getattr(
                    plan,
                    "bitrix_operational_edge_rendered_unit_path",
                    "",
                )
            ),
            "rendered_config_sha256": getattr(
                plan,
                "bitrix_operational_edge_rendered_config_sha256",
                None,
            ),
            "rendered_config_path": str(
                getattr(
                    plan,
                    "bitrix_operational_edge_rendered_config_path",
                    "",
                )
            ),
            "rendered_trust_sha256": getattr(
                plan,
                "bitrix_operational_edge_rendered_trust_sha256",
                None,
            ),
            "rendered_trust_path": str(
                getattr(
                    plan,
                    "bitrix_operational_edge_rendered_trust_path",
                    "",
                )
            ),
            "identity_bootstrap": {
                "service_user": getattr(
                    plan, "bitrix_operational_edge_service_user", None
                ),
                "service_group": getattr(
                    plan, "bitrix_operational_edge_service_group", None
                ),
                "service_uid": getattr(
                    plan, "bitrix_operational_edge_service_uid", None
                ),
                "service_gid": getattr(
                    plan, "bitrix_operational_edge_service_gid", None
                ),
                "socket_client_group": getattr(
                    plan,
                    "bitrix_operational_edge_socket_client_group",
                    None,
                ),
                "socket_client_gid": getattr(
                    plan,
                    "bitrix_operational_edge_socket_client_gid",
                    None,
                ),
                "receipt_sha256": getattr(
                    plan,
                    "bitrix_operational_edge_identity_bootstrap_receipt_sha256",
                    None,
                ),
            },
            "credential_projection": {
                "name": "bitrix-webhook-url",
                "source_path": evidence_contract.BITRIX_WEBHOOK_SOURCE_PATH,
                "projected_path": (
                    evidence_contract.BITRIX_WEBHOOK_PROJECTION_PATH
                ),
                "bind_target_path": (
                    evidence_contract.BITRIX_WEBHOOK_SOURCE_PATH
                ),
                "source_owner_uid": 0,
                "source_owner_gid": 0,
                "source_mode": "0400",
                "service_reads_projection": True,
                "original_source_inaccessible": True,
                "value_or_digest_recorded": False,
            },
            "receipt_key_contract": {
                "private_credential_name": "receipt-private-key",
                "private_source_path": (
                    evidence_contract.BITRIX_RECEIPT_PRIVATE_KEY_PATH
                ),
                "private_projection_path": (
                    evidence_contract.BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
                ),
                "private_owner_uid": 0,
                "private_owner_gid": 0,
                "private_mode": "0400",
                "public_path": (
                    evidence_contract.BITRIX_OPERATIONAL_EDGE_TRUST_PATH
                ),
                "public_key_id": getattr(
                    plan,
                    "bitrix_operational_edge_receipt_public_key_id",
                    None,
                ),
                "public_trust_sha256": getattr(
                    plan,
                    "bitrix_operational_edge_rendered_trust_sha256",
                    None,
                ),
                "writer_public_key_credential_name": "writer-public-key",
                "writer_public_key_source_path": (
                    evidence_contract.WRITER_PUBLIC_KEY_PATH
                ),
                "writer_public_key_projection_path": (
                    evidence_contract.WRITER_PUBLIC_KEY_PROJECTION_PATH
                ),
                "key_bootstrap_receipt_sha256": getattr(
                    plan,
                    "bitrix_operational_edge_key_bootstrap_receipt_sha256",
                    None,
                ),
                "create_only": True,
                "retire_private_on_stop": True,
                "retire_public_on_stop": True,
                "private_content_or_digest_recorded": False,
            },
            "expected_active_service_state": {
                "load_state": "loaded",
                "active_state": "active",
                "sub_state": "running",
                "unit_file_state": "disabled",
            },
            "expected_cleanup_service_state": {
                "active_state": "inactive",
                "sub_state": "dead",
                "overlay_retired_or_prior_restored": True,
            },
            "credential_binding": getattr(
                plan,
                "bitrix_operational_edge_credential_binding",
                None,
            ),
            "staging_protocol": (
                evidence_contract.BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL
            ),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        or fixture["owner_id"] != PRODUCTION_OWNER_DISCORD_USER_ID
        or fixture["owner_id"] not in plan.connector_allowed_user_ids
        or target["guild_id"] not in plan.connector_allowed_guild_ids
        or target["channel_id"] not in plan.connector_allowed_channel_ids
        or identities
        != {
            "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
            "connector_bot_user_id": plan.connector_bot_user_id,
            "routeback_bot_user_id": plan.routeback_bot_user_id,
        }
    ):
        _fail("fixture_plan_binding_invalid")


def publish_reviewed_fixture(
    *,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    producer_foundation: Mapping[str, Any],
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    source: Path = DEFAULT_REVIEWED_FIXTURE,
    publication_root: Path = DEFAULT_FIXTURE_PUBLICATION_ROOT,
    live_root: Path = DEFAULT_LIVE_ROOT,
    uid: int = 0,
    gid: int = 0,
) -> PublishedFixture:
    """Republish one reviewed fixture only with its durable trust receipt."""

    foundation = trust_producer_foundation(
        producer_foundation,
        pinned_owner_public_key_ed25519_hex=(
            pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            pinned_owner_public_key_source_sha256
        ),
    )
    if (
        foundation.value["release_sha"] != plan.revision
        or foundation.value["release_sha"] != full_plan.revision
        or foundation.value["capability_plan_sha256"] != plan.sha256
        or foundation.value["full_canary_plan_sha256"] != full_plan.sha256
    ):
        _fail("producer_foundation_plan_mismatch")

    raw = _stable_read(
        source,
        maximum=MAX_FIXTURE_BYTES,
        uid=uid,
        gid=gid,
    )
    value = _strict_json(raw, "reviewed_fixture_invalid")
    digest = _sha256_bytes(raw)
    evidence_contract._validate_fixture(value, digest)
    _validate_fixture_plan_binding(value, plan=plan, full_plan=full_plan)
    if value.get("producer_foundation_sha256") != foundation.sha256:
        _fail("fixture_producer_foundation_mismatch")
    run_id = value["run_id"]
    if _SAFE_ID_RE.fullmatch(run_id) is None:
        _fail("reviewed_fixture_invalid")
    receipt_path = _fixture_publication_receipt_path(
        root=publication_root,
        plan_sha256=plan.sha256,
        run_id=run_id,
        fixture_sha256=digest,
    )
    _secure_directory(publication_root, uid=uid, gid=gid, mode=0o700, create=False)
    _secure_directory(
        publication_root / plan.sha256,
        uid=uid,
        gid=gid,
        mode=0o700,
        create=False,
    )
    _secure_directory(
        receipt_path.parent,
        uid=uid,
        gid=gid,
        mode=0o700,
        create=False,
    )
    receipt_raw = _stable_read(
        receipt_path,
        maximum=MAX_FIXTURE_BYTES,
        uid=uid,
        gid=gid,
    )
    receipt = _strict_json(
        receipt_raw, "fixture_publication_receipt_invalid"
    )
    authority_sha256 = receipt.get("authority_sha256")
    if (
        not isinstance(authority_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
    ):
        _fail("fixture_publication_receipt_invalid")
    validate_fixture_publication_receipt(
        receipt,
        plan=plan,
        full_plan=full_plan,
        fixture=value,
        fixture_path=source,
        fixture_payload=raw,
        producer_foundation_sha256=foundation.sha256,
        authority_sha256=authority_sha256,
        receipt_path=receipt_path,
        uid=uid,
        gid=gid,
    )
    _secure_directory(live_root, uid=uid, gid=gid, mode=0o700, create=False)
    run_directory = live_root / run_id
    _secure_directory(
        run_directory,
        uid=uid,
        gid=gid,
        mode=0o700,
        create=True,
    )
    destination = run_directory / "fixture.json"
    _publish_exclusive_or_identical(
        destination,
        raw,
        uid=uid,
        gid=gid,
        maximum=MAX_FIXTURE_BYTES,
    )
    _retire_exact_source(source, raw, uid=uid, gid=gid)
    return PublishedFixture(
        value=copy.deepcopy(value),
        sha256=digest,
        path=destination,
        run_directory=run_directory,
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=_sha256_bytes(receipt_raw),
    )


@dataclass(frozen=True)
class ReceiptSlot:
    name: str
    role: str
    filename: str


RECEIPT_SLOTS = tuple(
    ReceiptSlot(name, SLOT_ROLE[name], SLOT_FILENAME[name])
    for name in PRODUCER_RECEIPT_SLOTS
)
_SLOT_BY_NAME = {slot.name: slot for slot in RECEIPT_SLOTS}


class FixedReceiptInbox:
    """Exact role-owned receipt slots; no discovery by content or task text."""

    def __init__(
        self,
        *,
        fixture: PublishedFixture,
        root: Path,
        role_identities: Mapping[str, tuple[int, int]],
        root_identity: tuple[int, int] = (0, 0),
        run_identity: tuple[int, int, int] = (0, 0, 0o730),
        poll_seconds: float = 0.05,
    ) -> None:
        if set(role_identities) != set(evidence_contract.AUTHORITY_ROLES):
            _fail("receipt_role_identities_invalid")
        self.fixture = fixture
        self.root = root / str(fixture.value["run_id"])
        self.role_identities = dict(role_identities)
        self.root_identity = root_identity
        self.run_identity = run_identity
        self.poll_seconds = poll_seconds

    def prepare(self) -> None:
        root_uid, root_gid = self.root_identity
        _secure_directory(
            self.root.parent,
            uid=root_uid,
            gid=root_gid,
            mode=0o711,
            create=False,
        )
        run_uid, run_gid, run_mode = self.run_identity
        _secure_directory(
            self.root,
            uid=run_uid,
            gid=run_gid,
            mode=run_mode,
            create=True,
        )

    def path(self, name: str) -> Path:
        try:
            slot = _SLOT_BY_NAME[name]
        except KeyError as exc:
            raise CapabilityLiveDriverError("receipt_slot_invalid") from exc
        return self.root / slot.filename

    def assert_empty(self) -> None:
        if any(os.path.lexists(self.path(slot.name)) for slot in RECEIPT_SLOTS):
            _fail("receipt_inbox_not_fresh")

    def wait(
        self,
        name: str,
        *,
        deadline: float,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        slot = _SLOT_BY_NAME.get(name)
        if slot is None:
            _fail("receipt_slot_invalid")
        path = self.path(name)
        uid, gid = self.role_identities[slot.role]
        last_error: BaseException | None = None
        while time.monotonic() <= deadline:
            if cancel is not None and cancel.is_set():
                _fail("receipt_wait_cancelled")
            try:
                raw = _stable_read(
                    path,
                    maximum=MAX_RECEIPT_BYTES,
                    uid=uid,
                    gid=gid,
                )
                value = _strict_json(raw, "signed_receipt_invalid")
                if (
                    value.get("schema") != SIGNED_RECEIPT_SCHEMA
                    or value.get("authority_role") != slot.role
                ):
                    _fail("signed_receipt_invalid")
                return value
            except CapabilityLiveDriverError as exc:
                last_error = exc
            time.sleep(self.poll_seconds)
        raise CapabilityLiveDriverError("signed_receipt_timeout") from last_error


def _role_identities(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, tuple[int, int]]:
    return {
        "business_edge": (plan.identities.mac_ops_uid, plan.identities.mac_ops_gid),
        "canonical_writer": (
            full_plan.identities.writer_uid,
            full_plan.identities.writer_gid,
        ),
        "discord_edge": (
            full_plan.identities.edge_uid,
            full_plan.identities.edge_gid,
        ),
        # Observer and owner keys remain outside every model/tool service.
        "gateway_observer": (0, 0),
        "owner": (0, 0),
    }


def _observed_at(receipt: Mapping[str, Any]) -> int:
    try:
        value = receipt["payload"]["observed_at_unix_ms"]
    except (KeyError, TypeError) as exc:
        raise CapabilityLiveDriverError("signed_receipt_time_invalid") from exc
    if type(value) is not int or value <= 0:
        _fail("signed_receipt_time_invalid")
    return value


def _validate_restart_checkpoint(
    receipt: Mapping[str, Any],
    *,
    fixture: PublishedFixture,
) -> int:
    payload = evidence_contract._signed_payload(
        receipt,
        slot="worker_restart_checkpoint",
        role="canonical_writer",
        payload_schema=RESTART_CHECKPOINT_SCHEMA,
        fields=(
            "objective_id",
            "worker_service_unit",
            "next_unverified_step_id",
            "checkpoint_event_id",
            "checkpoint_event_sha256",
            "restart_requested",
        ),
        fixture=fixture.value,
        fixture_sha256=fixture.sha256,
        code="restart_checkpoint_invalid",
    )
    if (
        payload["objective_id"] != "workspace_continuation"
        or payload["worker_service_unit"] != DEFAULT_WORKER_SERVICE_UNIT_NAME
        or payload["restart_requested"] is not True
    ):
        _fail("restart_checkpoint_invalid")
    evidence_contract._safe_id(
        payload["next_unverified_step_id"], "restart_checkpoint_invalid"
    )
    evidence_contract._safe_id(
        payload["checkpoint_event_id"], "restart_checkpoint_invalid"
    )
    evidence_contract._sha256(
        payload["checkpoint_event_sha256"], "restart_checkpoint_invalid"
    )
    return payload["observed_at_unix_ms"]


def _restart_isolated_worker() -> Mapping[str, Any]:
    command = (SYSTEMCTL, "restart", DEFAULT_WORKER_SERVICE_UNIT_NAME)
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        _fail("worker_restart_failed")
    unsigned = {
        "schema": "muncho-production-capability-worker-restart.v1",
        "service_unit": DEFAULT_WORKER_SERVICE_UNIT_NAME,
        "command_sha256": _sha256_bytes("\0".join(command).encode("utf-8")),
        "completed_at_unix_ms": int(time.time() * 1000),
    }
    return {**unsigned, "receipt_sha256": _sha256_bytes(_canonical_bytes(unsigned))}


class RestartWatcher:
    def __init__(
        self,
        *,
        inbox: FixedReceiptInbox,
        fixture: PublishedFixture,
        deadline: float,
        restart: Callable[[], Mapping[str, Any]],
    ) -> None:
        self.inbox = inbox
        self.fixture = fixture
        self.deadline = deadline
        self.restart = restart
        self.cancel = threading.Event()
        self.error: BaseException | None = None
        self.checkpoint_at_unix_ms: int | None = None
        self.restart_receipt: Mapping[str, Any] | None = None
        self._cleanup_lock = threading.Lock()
        self._cleanup_started = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            receipt = self.inbox.wait(
                "worker_restart_checkpoint",
                deadline=self.deadline,
                cancel=self.cancel,
            )
            self.checkpoint_at_unix_ms = _validate_restart_checkpoint(
                receipt, fixture=self.fixture
            )
            # Cleanup takes this same lock before stop/retire.  Once cleanup
            # starts, no late checkpoint can cross the lifecycle boundary.
            with self._cleanup_lock:
                if self._cleanup_started or self.cancel.is_set():
                    _fail("worker_restart_cancelled")
                self.restart_receipt = copy.deepcopy(dict(self.restart()))
        except BaseException as exc:
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> Mapping[str, Any]:
        remaining = max(0.0, self.deadline - time.monotonic())
        self.thread.join(remaining)
        if self.thread.is_alive():
            self.cancel.set()
            self.thread.join(2.0)
        if self.thread.is_alive():
            _fail("worker_restart_watcher_stuck")
        if self.error is not None:
            raise CapabilityLiveDriverError("worker_restart_checkpoint_missing") from self.error
        if self.restart_receipt is None:
            _fail("worker_restart_checkpoint_missing")
        return self.restart_receipt

    def shutdown_before_cleanup(self) -> None:
        """Prevent a late checkpoint from restarting after stop/retire."""

        with self._cleanup_lock:
            self._cleanup_started = True
            self.cancel.set()
        remaining = max(0.0, self.deadline - time.monotonic())
        self.thread.join(remaining)
        if self.thread.is_alive():
            _fail("worker_restart_watcher_stuck")


def _assemble_bundles(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "workspace_continuation": {
            "gateway_receipt": receipts["workspace_gateway"],
            "writer_receipt": receipts["workspace_writer"],
            "owner_approval_receipt": receipts["workspace_owner"],
        },
        "capability_denials": receipts["capability_denials"],
        "database_reconciliation": receipts["database_reconciliation"],
        "bitrix_boundary": {
            "edge_receipt": receipts["bitrix_edge"],
            "writer_receipt": receipts["bitrix_writer"],
        },
        "discord_routeback": {
            "edge_receipt": receipts["discord_edge"],
            "writer_receipt": receipts["discord_writer"],
        },
        "failure_recovery": {
            "gateway_receipt": receipts["failure_gateway"],
            "writer_receipt": receipts["failure_writer"],
        },
    }


def _validate_bundles_and_order(
    *,
    fixture: PublishedFixture,
    runtime: Mapping[str, Any],
    bundles: Mapping[str, Any],
    api_started_at_unix_ms: int,
    api_completed_at_unix_ms: int,
    checkpoint_at_unix_ms: int,
) -> None:
    evidence_contract._validate_runtime(
        runtime, fixture=fixture.value, fixture_sha256=fixture.sha256
    )
    validators = (
        ("workspace_continuation", evidence_contract._validate_task_workspace_bundle),
        ("capability_denials", evidence_contract._validate_denial_bundle),
        ("database_reconciliation", evidence_contract._validate_database_bundle),
        ("bitrix_boundary", evidence_contract._validate_bitrix_bundle),
        ("discord_routeback", evidence_contract._validate_discord_bundle),
        ("failure_recovery", evidence_contract._validate_failure_bundle),
    )
    previous = _observed_at(runtime)
    for name, validator in validators:
        validator(
            bundles[name],
            fixture=fixture.value,
            fixture_sha256=fixture.sha256,
        )
        if name == "workspace_continuation":
            owner = bundles[name]["owner_approval_receipt"]
            if not (
                fixture.value["valid_from_unix_ms"]
                <= _observed_at(owner)
                <= api_started_at_unix_ms
            ):
                _fail("signed_receipt_order_invalid")
            receipts = [
                bundles[name]["gateway_receipt"],
                bundles[name]["writer_receipt"],
            ]
        else:
            receipts = (
                [bundles[name]]
                if bundles[name].get("schema") == SIGNED_RECEIPT_SCHEMA
                else list(bundles[name].values())
            )
        times = [_observed_at(receipt) for receipt in receipts]
        if (
            min(times) < previous
            or min(times) < api_started_at_unix_ms
            or max(times) > api_completed_at_unix_ms
        ):
            _fail("signed_receipt_order_invalid")
        previous = max(times)
    if not api_started_at_unix_ms <= checkpoint_at_unix_ms <= api_completed_at_unix_ms:
        _fail("restart_checkpoint_order_invalid")


def publish_evidence(
    fixture: PublishedFixture,
    evidence: Mapping[str, Any],
    *,
    uid: int = 0,
    gid: int = 0,
) -> tuple[Path, str]:
    payload = _canonical_bytes(evidence)
    if not 1 < len(payload) <= MAX_EVIDENCE_BYTES:
        _fail("evidence_size_invalid")
    path = fixture.run_directory / "evidence.json"
    _publish_exclusive(path, payload, uid=uid, gid=gid)
    return path, _sha256_bytes(payload)


def _observer_cleanup_payload(
    publication: Mapping[str, Any],
    *,
    fixture: PublishedFixture,
    production_diff: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project root facts into the fixed observer request, without semantics."""

    if not isinstance(publication, Mapping) or set(publication) != {
        "facts",
        "facts_path",
        "facts_file_sha256",
        "facts_uid",
        "facts_gid",
        "facts_mode",
    }:
        _fail("cleanup_facts_publication_invalid")
    facts = publication["facts"]
    if (
        not isinstance(facts, Mapping)
        or facts.get("schema")
        != "muncho-production-capability-cleanup-facts.v1"
        or publication.get("facts_mode") != "0440"
        or production_diff.get("run_id") != fixture.value["run_id"]
        or production_diff.get("fixture_sha256") != fixture.sha256
        or production_diff.get("changed_surfaces") != []
        or production_diff.get("production_mutation_observed") is not False
    ):
        _fail("cleanup_facts_publication_invalid")
    proof = facts.get("credential_consumer_stop_proof")
    if not isinstance(proof, Mapping):
        _fail("cleanup_facts_publication_invalid")
    return {
        "schema": "muncho-production-capability-cleanup.v1",
        "run_id": fixture.value["run_id"],
        "release_sha": fixture.value["release_sha"],
        "fixture_sha256": fixture.sha256,
        "observed_at_unix_ms": facts["observed_at_unix_ms"],
        "non_observer_service_units": facts["non_observer_stop_order"],
        "non_observer_services_stopped": True,
        "non_observer_services_state_sha256": proof[
            "non_observer_services_state_sha256"
        ],
        "gateway_observer_signer_identity": facts[
            "observer_signer_identity"
        ],
        "credential_consumer_stop_proof": proof,
        "credential_leases": [
            "api_control",
            "bitrix_operational_edge_webhook",
            "discord_canonical_routeback_bot_token",
            "discord_public_session_bot_token",
            "mac_ops_gitlab",
            "openai_codex",
        ],
        "credential_leases_retired": True,
        "retirements": facts["retirements"],
        "retirement_receipt_sha256s": facts[
            "retirement_receipt_sha256s"
        ],
        "credential_absence": facts["credential_absence"],
        "credentials_absent": True,
        "bitrix_receipt_key_retirement": facts[
            "bitrix_receipt_key_retirement"
        ],
        "bitrix_receipt_key_absence": facts[
            "bitrix_receipt_key_absence"
        ],
        "discord_credential_topology": {
            "connector_service_unit": "muncho-discord-connector.service",
            "connector_credential_lease": (
                "discord_public_session_bot_token"
            ),
            "connector_credential_scope": (
                "ordinary_public_ingress_and_session_replies"
            ),
            "routeback_service_unit": "muncho-discord-egress.service",
            "routeback_credential_lease": (
                "discord_canonical_routeback_bot_token"
            ),
            "routeback_credential_scope": (
                "canonical_public_routeback_only"
            ),
        },
        "browser_session_retired": True,
        "isolated_worker_lease_cleanup_verified": True,
        "production_diff_sha256": production_diff["diff_sha256"],
    }


_WORKSPACE_MODEL_PROPOSAL_CORE_FIELDS = (
    "session_id",
    "capability_epoch_sha256",
    "task_workspace_evidence_sha256s",
    "first_path_failure_receipt_sha256",
    "alternate_read_receipt_sha256",
    "model_requested_effort",
    "later_request_effort",
    "reasoning_tool_call_id",
    "restart_count",
    "used_command_sha256s",
    "mutation_receipt_sha256s",
    "approval_prompt_count",
    "microapproval_prompt_count",
    "replayed_mutation_count",
    "owner_grant_id",
    "owner_grant_sha256",
    "consumed_command_sha256s",
    "terminal_plan_id",
    "terminal_plan_revision",
)
_FAILURE_MODEL_PROPOSAL_CORE_FIELDS = (
    "failures",
    "model_retained_tool_control",
)


def _build_gateway_observer_payloads(
    *,
    fixture: PublishedFixture,
    source_projection: Mapping[str, Any],
    model_proposal_cores: Mapping[str, Mapping[str, Any]],
    non_observer_receipts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    """Join fixed signed facts to the model-authored proposal mechanically."""

    required_receipts = {
        "workspace_owner",
        "workspace_writer",
        "failure_writer",
    }
    if (
        not isinstance(source_projection, Mapping)
        or set(model_proposal_cores)
        != {"workspace_gateway", "failure_gateway"}
        or not required_receipts.issubset(non_observer_receipts)
    ):
        _fail("gateway_observer_payload_sources_invalid")
    workspace_core = copy.deepcopy(
        dict(model_proposal_cores["workspace_gateway"])
    )
    failure_core = copy.deepcopy(
        dict(model_proposal_cores["failure_gateway"])
    )
    if (
        set(workspace_core) != set(_WORKSPACE_MODEL_PROPOSAL_CORE_FIELDS)
        or set(failure_core) != set(_FAILURE_MODEL_PROPOSAL_CORE_FIELDS)
    ):
        _fail("gateway_observer_model_proposal_invalid")
    try:
        owner_receipt = non_observer_receipts["workspace_owner"]
        writer_receipt = non_observer_receipts["workspace_writer"]
        failure_writer_receipt = non_observer_receipts["failure_writer"]
        owner = evidence_contract._signed_payload(
            owner_receipt,
            slot="workspace_owner",
            role="owner",
            payload_schema=evidence_contract.PLAN_APPROVAL_SCHEMA,
            fields=(
                "approval_id",
                "owner_id",
                "session_id",
                "capability_epoch_sha256",
                "command_sha256s",
                "ttl_seconds",
                "max_uses",
            ),
            fixture=fixture.value,
            fixture_sha256=fixture.sha256,
            code="gateway_observer_payload_sources_invalid",
        )
        writer = evidence_contract._signed_payload(
            writer_receipt,
            slot="workspace_writer",
            role="canonical_writer",
            payload_schema=evidence_contract.TASK_WORKSPACE_WRITER_SCHEMA,
            fields=(
                "session_id",
                "capability_epoch_sha256",
                "owner_grant_id",
                "owner_grant_sha256",
                "consumed_command_sha256s",
                "terminal_ctw",
            ),
            fixture=fixture.value,
            fixture_sha256=fixture.sha256,
            code="gateway_observer_payload_sources_invalid",
        )
        failure_writer = evidence_contract._signed_payload(
            failure_writer_receipt,
            slot="failure_writer",
            role="canonical_writer",
            payload_schema=evidence_contract.FAILURE_WRITER_SCHEMA,
            fields=("terminal_ctw",),
            fixture=fixture.value,
            fixture_sha256=fixture.sha256,
            code="gateway_observer_payload_sources_invalid",
        )
        terminal = writer["terminal_ctw"]
        source_terminal = source_projection["api_terminal_event_identity"]
        runtime_source = source_projection["runtime_source_identity"]
        proposal_identities = source_projection[
            "model_proposal_core_identities"
        ]
        frame_records = source_projection["frame_records"]
        owner_receipt_sha256 = _sha256_bytes(
            _canonical_bytes(owner_receipt)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CapabilityLiveDriverError(
            "gateway_observer_payload_sources_invalid"
        ) from exc
    if (
        not isinstance(terminal, Mapping)
        or not isinstance(source_terminal, Mapping)
        or not isinstance(runtime_source, Mapping)
        or not isinstance(proposal_identities, Mapping)
        or not isinstance(frame_records, list)
        or not frame_records
        or _sha256_bytes(_canonical_bytes(workspace_core))
        != proposal_identities.get("workspace_gateway", {}).get(
            "core_sha256"
        )
        or _sha256_bytes(_canonical_bytes(failure_core))
        != proposal_identities.get("failure_gateway", {}).get("core_sha256")
        or workspace_core["session_id"] != owner["session_id"]
        or workspace_core["session_id"] != writer["session_id"]
        or workspace_core["capability_epoch_sha256"]
        != owner["capability_epoch_sha256"]
        or workspace_core["capability_epoch_sha256"]
        != writer["capability_epoch_sha256"]
        or workspace_core["owner_grant_id"] != owner["approval_id"]
        or workspace_core["owner_grant_id"] != writer["owner_grant_id"]
        or workspace_core["owner_grant_sha256"] != owner_receipt_sha256
        or workspace_core["owner_grant_sha256"]
        != writer["owner_grant_sha256"]
        or workspace_core["used_command_sha256s"]
        != owner["command_sha256s"]
        or workspace_core["consumed_command_sha256s"]
        != owner["command_sha256s"]
        or workspace_core["consumed_command_sha256s"]
        != writer["consumed_command_sha256s"]
        or workspace_core["terminal_plan_id"] != terminal.get("plan_id")
        or workspace_core["terminal_plan_revision"]
        != terminal.get("revision")
        or workspace_core["replayed_mutation_count"]
        != terminal.get("replayed_mutation_count")
        or owner["observed_at_unix_ms"] > writer["observed_at_unix_ms"]
        or writer["observed_at_unix_ms"]
        > source_projection.get("observed_at_unix_ms", 0)
        or failure_writer["observed_at_unix_ms"]
        > source_projection.get("observed_at_unix_ms", 0)
    ):
        _fail("gateway_observer_payload_sources_invalid")
    common = {
        "run_id": fixture.value["run_id"],
        "release_sha": fixture.value["release_sha"],
        "fixture_sha256": fixture.sha256,
    }
    runtime_observed_at = min(
        record["observed_at_unix_ms"] for record in frame_records
    )
    runtime_payload = {
        "schema": evidence_contract.RUNTIME_RECEIPT_SCHEMA,
        **common,
        "observed_at_unix_ms": runtime_observed_at,
        "host_identity_sha256": fixture.value["host_identity_sha256"],
        "release_artifact_sha256": fixture.value[
            "release_artifact_sha256"
        ],
        "installed_wheel_manifest_sha256": fixture.value[
            "installed_wheel_manifest_sha256"
        ],
        "effective_config_sha256": fixture.value["effective_config_sha256"],
        "tool_inventory_sha256": fixture.value["tool_inventory_sha256"],
        **copy.deepcopy(dict(fixture.value["model_route"])),
        "toolsets": list(fixture.value["required_toolsets"]),
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
    workspace_payload = {
        "schema": evidence_contract.TASK_WORKSPACE_GATEWAY_SCHEMA,
        **common,
        "observed_at_unix_ms": writer["observed_at_unix_ms"],
        "transcript_sha256": source_terminal["transcript_sha256"],
        **workspace_core,
    }
    failure_payload = {
        "schema": evidence_contract.FAILURE_GATEWAY_SCHEMA,
        **common,
        "observed_at_unix_ms": failure_writer["observed_at_unix_ms"],
        "transcript_sha256": source_terminal["transcript_sha256"],
        **failure_core,
    }
    if not (
        runtime_observed_at <= workspace_payload["observed_at_unix_ms"]
        <= failure_payload["observed_at_unix_ms"]
    ):
        _fail("gateway_observer_payload_order_invalid")
    return {
        "runtime": runtime_payload,
        "workspace_gateway": workspace_payload,
        "failure_gateway": failure_payload,
    }


FoundationChecker = Callable[[], TrustedProducerFoundation]
FixturePublisher = Callable[[TrustedProducerFoundation], PublishedFixture]
FleetActivator = Callable[
    [TrustedProducerFoundation, PublishedFixture], ActivatedProducerFleet
]
FleetRetirer = Callable[[ActivatedProducerFleet], Mapping[str, Any]]
ProductionObservationGate = Callable[
    [str, PublishedFixture, float], Mapping[str, Any]
]
GatewayObserverSourcePublisher = Callable[
    [
        TrustedProducerFoundation,
        PublishedFixture,
        SSEConversation,
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        float,
    ],
    Mapping[str, Any],
]
InboxFactory = Callable[[PublishedFixture], FixedReceiptInbox]
ClientFactory = Callable[[str, str], LoopbackCanaryClient]


class HonestCapabilityCanaryDriver:
    """Run the reviewed objective and accept only external signed evidence."""

    def __init__(
        self,
        *,
        plan: CapabilityCanaryPlan,
        full_plan: FullCanaryPlan,
        lifecycle: CapabilityCanaryLifecycle,
        capability_approval: Any,
        full_approval: Any,
        producer_foundation_check: FoundationChecker,
        fixture_publisher: FixturePublisher,
        producer_fleet_activator: FleetActivator,
        producer_fleet_retirer: FleetRetirer,
        production_observation_gate: ProductionObservationGate,
        gateway_observer_source_publisher: GatewayObserverSourcePublisher,
        inbox_factory: InboxFactory,
        collector: Any,
        client_factory: ClientFactory,
        control_key_reader: Callable[[], tuple[str, str]] = _read_api_control_key,
        restart: Callable[[], Mapping[str, Any]] = _restart_isolated_worker,
        evidence_publisher: Callable[
            [PublishedFixture, Mapping[str, Any]], tuple[Path, str]
        ] = publish_evidence,
        verifier: Callable[..., Mapping[str, Any]] = verify_files,
        root_guard: Callable[[], None] = _require_root_linux,
        receipt_timeout_seconds: float = DEFAULT_RECEIPT_TIMEOUT_SECONDS,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        session_key_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        if not callable(producer_foundation_check):
            _fail("producer_foundation_missing")
        if not callable(producer_fleet_activator):
            _fail("producer_activation_missing")
        if not callable(producer_fleet_retirer):
            _fail("producer_retirement_missing")
        if not callable(production_observation_gate):
            _fail("production_observation_gate_missing")
        if not callable(gateway_observer_source_publisher):
            _fail("gateway_observer_source_publisher_missing")
        self.plan = plan
        self.full_plan = full_plan
        self.lifecycle = lifecycle
        self.capability_approval = capability_approval
        self.full_approval = full_approval
        self.producer_foundation_check = producer_foundation_check
        self.fixture_publisher = fixture_publisher
        self.producer_fleet_activator = producer_fleet_activator
        self.producer_fleet_retirer = producer_fleet_retirer
        self.production_observation_gate = production_observation_gate
        self.gateway_observer_source_publisher = (
            gateway_observer_source_publisher
        )
        self.inbox_factory = inbox_factory
        self.collector = collector
        self.client_factory = client_factory
        self.control_key_reader = control_key_reader
        self.restart = restart
        self.evidence_publisher = evidence_publisher
        self.verifier = verifier
        self.root_guard = root_guard
        self.receipt_timeout_seconds = receipt_timeout_seconds
        self.now_ms = now_ms
        self.session_key_factory = session_key_factory

    def run(self) -> Mapping[str, Any]:
        self.root_guard()
        foundation = self.producer_foundation_check()
        if not isinstance(foundation, TrustedProducerFoundation):
            _fail("producer_foundation_invalid")
        # Revalidate here so a callback cannot substitute an unpinned mapping
        # after its initial load.
        foundation = trust_producer_foundation(
            foundation.value,
            pinned_owner_public_key_ed25519_hex=(
                foundation.pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                foundation.pinned_owner_public_key_source_sha256
            ),
        )
        fixture = self.fixture_publisher(foundation)
        if (
            fixture.value.get("producer_foundation_sha256")
            != foundation.sha256
        ):
            _fail("fixture_producer_foundation_mismatch")
        inbox = self.inbox_factory(fixture)
        inbox.prepare()
        inbox.assert_empty()
        started_at = self.now_ms()
        deadline = time.monotonic() + self.receipt_timeout_seconds
        primary: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        services_started = False
        client: Any | None = None
        watcher: RestartWatcher | None = None
        runtime: Mapping[str, Any] | None = None
        bundles: Mapping[str, Any] | None = None
        conversation: SSEConversation | None = None
        api_started_at: int | None = None
        restart_receipt: Mapping[str, Any] | None = None
        checkpoint_at: int | None = None
        cleanup: Mapping[str, Any] | None = None
        cleanup_finalization: Mapping[str, Any] | None = None
        lifecycle_stop_result: Mapping[str, Any] | None = None
        lifecycle_start_result: Mapping[str, Any] | None = None
        observer_source_publication: Mapping[str, Any] | None = None
        producer_readiness: Mapping[str, Any] | None = None
        producer_fleet: ActivatedProducerFleet | None = None
        producer_retirement: Mapping[str, Any] | None = None
        producer_pump_cancel = threading.Event()
        producer_pump_errors: list[BaseException] = []
        producer_pump_receipts: dict[str, Mapping[str, Any]] = {}
        producer_pump_thread: threading.Thread | None = None
        producer_pump_joined = False
        production_before: Mapping[str, Any] | None = None
        production_diff: Mapping[str, Any] | None = None
        lifecycle_start_attempted = False
        try:
            production_before = self.production_observation_gate(
                "before", fixture, deadline
            )
            if (
                not isinstance(production_before, Mapping)
                or production_before.get("phase") != "before"
                or production_before.get("run_id")
                != fixture.value["run_id"]
                or production_before.get("fixture_sha256") != fixture.sha256
            ):
                _fail("production_before_observation_invalid")
            self.collector.start()
            lifecycle_start_attempted = True
            lifecycle_start_result = self.lifecycle.start(
                self.capability_approval,
                self.full_approval,
            )
            if not isinstance(lifecycle_start_result, Mapping):
                _fail("capability_lifecycle_start_invalid")
            services_started = True
            activated = self.producer_fleet_activator(foundation, fixture)
            producer_fleet = activated
            producer_readiness = _validate_activated_fleet(
                activated,
                foundation=foundation,
                fixture=fixture,
                inbox_root=inbox.root,
                plan=self.plan,
                full_plan=self.full_plan,
                now_ms=self.now_ms(),
            )
            if callable(activated.pre_cleanup_pump):
                def run_pre_cleanup_pump() -> None:
                    try:
                        producer_pump_receipts.update(
                            activated.pre_cleanup_pump(
                                deadline,
                                producer_pump_cancel,
                            )
                        )
                    except BaseException as exc:
                        producer_pump_errors.append(exc)

                producer_pump_thread = threading.Thread(
                    target=run_pre_cleanup_pump,
                    name="capability-native-receipt-pump",
                    daemon=False,
                )
                producer_pump_thread.start()
            control_key, _control_provenance_sha256 = self.control_key_reader()
            session_key = self.session_key_factory()
            client = self.client_factory(control_key, session_key)
            api_started_at = self.now_ms()
            watcher = RestartWatcher(
                inbox=inbox,
                fixture=fixture,
                deadline=deadline,
                restart=self.restart,
            )
            watcher.start()
            try:
                conversation = client.run(
                    fixture={"task_policy": {"prompt": reviewed_objective_prompt()}},
                    session_id=f"capability_{fixture.value['run_id']}",
                )
            finally:
                clear = getattr(client, "clear_secrets", None)
                if callable(clear):
                    clear()
                control_key = None
                session_key = None
            restart_receipt = watcher.finish()
            checkpoint_at = watcher.checkpoint_at_unix_ms
            observer_source_publication = (
                self.gateway_observer_source_publisher(
                    foundation,
                    fixture,
                    conversation,
                    restart_receipt,
                    lifecycle_start_result,
                    producer_readiness,
                    deadline,
                )
            )
            if (
                not isinstance(observer_source_publication, Mapping)
                or observer_source_publication.get("mode") != "0440"
                or observer_source_publication.get("uid") != 0
                or not isinstance(
                    observer_source_publication.get("projection_sha256"),
                    str,
                )
            ):
                _fail("gateway_observer_source_publication_invalid")
            if producer_pump_thread is not None:
                producer_pump_thread.join(
                    max(0.0, deadline - time.monotonic())
                )
                producer_pump_joined = not producer_pump_thread.is_alive()
                if not producer_pump_joined:
                    _fail("producer_pre_cleanup_pump_timeout")
                if producer_pump_errors:
                    raise CapabilityLiveDriverError(
                        "producer_pre_cleanup_pump_failed"
                    ) from producer_pump_errors[0]
            if callable(activated.observer_pump):
                activated.observer_pump(
                    deadline,
                    producer_pump_cancel,
                )
            runtime = inbox.wait("runtime", deadline=deadline)
            evidence_contract._validate_runtime(
                runtime,
                fixture=fixture.value,
                fixture_sha256=fixture.sha256,
            )
            receipt_names = (
                "workspace_gateway",
                "workspace_writer",
                "workspace_owner",
                "capability_denials",
                "database_reconciliation",
                "bitrix_edge",
                "bitrix_writer",
                "discord_edge",
                "discord_writer",
                "failure_gateway",
                "failure_writer",
            )
            receipts = {
                name: inbox.wait(name, deadline=deadline) for name in receipt_names
            }
            bundles = _assemble_bundles(receipts)
            assert checkpoint_at is not None
            _validate_bundles_and_order(
                fixture=fixture,
                runtime=runtime,
                bundles=bundles,
                api_started_at_unix_ms=api_started_at,
                api_completed_at_unix_ms=conversation.completed_at_unix_ms,
                checkpoint_at_unix_ms=checkpoint_at,
            )
            if producer_pump_thread is not None:
                producer_pump_thread.join(
                    timeout=max(0.0, deadline - time.monotonic())
                )
                producer_pump_joined = True
                if producer_pump_thread.is_alive():
                    _fail("producer_pre_cleanup_pump_timeout")
                if producer_pump_errors:
                    raise producer_pump_errors[0]
            production_diff = self.production_observation_gate(
                "after", fixture, deadline
            )
            if (
                not isinstance(production_diff, Mapping)
                or production_diff.get("schema")
                != "muncho-production-capability-production-diff.v1"
                or production_diff.get("run_id")
                != fixture.value["run_id"]
                or production_diff.get("fixture_sha256") != fixture.sha256
                or production_diff.get("changed_surfaces") != []
                or production_diff.get("production_mutation_observed")
                is not False
            ):
                _fail("production_after_observation_invalid")
        except BaseException as exc:
            primary = exc
        finally:
            if watcher is not None:
                try:
                    watcher.shutdown_before_cleanup()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if producer_pump_thread is not None and not producer_pump_joined:
                producer_pump_cancel.set()
                producer_pump_thread.join(timeout=5.0)
                if producer_pump_thread.is_alive():
                    cleanup_errors.append(
                        RuntimeError("producer receipt pump did not stop")
                    )
                elif producer_pump_errors:
                    cleanup_errors.extend(producer_pump_errors)
            if client is not None:
                try:
                    clear = getattr(client, "clear_secrets", None)
                    if callable(clear):
                        clear()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            # Even a failed task run must obtain the immutable production
            # after/diff while the credential-blind observer is still live.
            # Cleanup consumes that diff and therefore cannot silently skip it.
            if services_started and producer_fleet is not None and production_diff is None:
                try:
                    production_diff = self.production_observation_gate(
                        "after",
                        fixture,
                        time.monotonic() + self.receipt_timeout_seconds,
                    )
                    if (
                        not isinstance(production_diff, Mapping)
                        or production_diff.get("schema")
                        != "muncho-production-capability-production-diff.v1"
                        or production_diff.get("run_id")
                        != fixture.value["run_id"]
                        or production_diff.get("fixture_sha256")
                        != fixture.sha256
                        or production_diff.get("changed_surfaces") != []
                        or production_diff.get("production_mutation_observed")
                        is not False
                    ):
                        _fail("production_after_observation_invalid")
                except BaseException as exc:
                    cleanup_errors.append(exc)
            # Capability stop is deliberately idempotent and always attempts
            # all nine reverse-order stops plus all three overlay retirements.
            # Invoke it even after a partial lifecycle start.
            try:
                if producer_fleet is None:
                    lifecycle_stop_result = self.lifecycle.stop()
                else:
                    if not callable(producer_fleet.cleanup_producer):
                        _fail("cleanup_observer_producer_missing")
                    if production_diff is None:
                        _fail("production_after_observation_missing")
                    lifecycle_stop_result = self.lifecycle.stop(
                        cleanup_producer=lambda publication: (
                            producer_fleet.cleanup_producer(
                                _observer_cleanup_payload(
                                    publication,
                                    fixture=fixture,
                                    production_diff=production_diff,
                                )
                            )
                        ),
                        cleanup_run_id=fixture.value["run_id"],
                        producer_activation_retirer=lambda: (
                            self.producer_fleet_retirer(producer_fleet)
                        ),
                    )
                    if not isinstance(lifecycle_stop_result, Mapping):
                        _fail("capability_lifecycle_stop_invalid")
                    producer_retirement = lifecycle_stop_result.get(
                        "producer_fleet_retirement"
                    )
                    cleanup_finalization = lifecycle_stop_result.get(
                        "cleanup_finalization"
                    )
                    if (
                        not isinstance(producer_retirement, Mapping)
                        or producer_retirement.get("schema")
                        != "muncho-production-capability-fleet-retirement.v1"
                        or producer_retirement.get("run_id")
                        != fixture.value["run_id"]
                        or producer_retirement.get("readiness_sha256")
                        != producer_fleet.readiness.get("readiness_sha256")
                        or producer_retirement.get("retired") is not True
                        or producer_retirement.get("absence_verified") is not True
                    ):
                        _fail("producer_retirement_invalid")
                    if (
                        not isinstance(cleanup_finalization, Mapping)
                        or cleanup_finalization.get("schema")
                        != evidence_contract.CLEANUP_FINALIZATION_SCHEMA
                        or cleanup_finalization.get("run_id")
                        != fixture.value["run_id"]
                        or cleanup_finalization.get("fixture_sha256")
                        != fixture.sha256
                        or type(
                            cleanup_finalization.get("finalized_at_unix_ms")
                        )
                        is not int
                        or not isinstance(
                            cleanup_finalization.get("finalization_sha256"), str
                        )
                    ):
                        _fail("cleanup_finalization_invalid")
                services_started = False
            except BaseException as exc:
                cleanup_errors.append(exc)
            if lifecycle_start_attempted:
                try:
                    cleanup = inbox.wait(
                        "cleanup",
                        deadline=time.monotonic()
                        + self.receipt_timeout_seconds,
                    )
                    evidence_contract._validate_cleanup(
                        cleanup,
                        fixture=fixture.value,
                        fixture_sha256=fixture.sha256,
                    )
                    if (
                        isinstance(lifecycle_stop_result, Mapping)
                        and lifecycle_stop_result.get("cleanup_receipt")
                        and lifecycle_stop_result["cleanup_receipt"] != cleanup
                    ):
                        _fail("cleanup_receipt_lifecycle_mismatch")
                except BaseException as exc:
                    cleanup_errors.append(exc)
            # The observer remains alive until it has witnessed stopped state
            # and exact install-bound retirement of every canary lease.
            try:
                self.collector.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            errors = ([] if primary is None else [primary]) + cleanup_errors
            raise BaseExceptionGroup(
                "capability live run cleanup failed closed", errors
            )
        if primary is not None:
            raise RuntimeError("capability live run failed closed") from primary
        if services_started:
            _fail("capability_services_not_stopped")
        assert runtime is not None
        assert bundles is not None
        assert conversation is not None
        assert api_started_at is not None
        assert restart_receipt is not None
        assert cleanup is not None
        assert cleanup_finalization is not None
        assert producer_readiness is not None
        assert producer_retirement is not None
        assert production_before is not None
        assert production_diff is not None
        receipt_times = [
            _observed_at(value)
            for value in evidence_contract._all_evidence_receipts(
                runtime, bundles, cleanup
            )
        ]
        if _observed_at(cleanup) != max(receipt_times):
            _fail("cleanup_receipt_order_invalid")
        completed_at = max(
            self.now_ms(),
            _observed_at(cleanup),
            cleanup_finalization["finalized_at_unix_ms"],
        )
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "execution_mode": "live_production_shaped_canary",
            "synthetic": False,
            "fixture_sha256": fixture.sha256,
            "release_sha": fixture.value["release_sha"],
            "release_artifact_sha256": fixture.value["release_artifact_sha256"],
            "installed_wheel_manifest_sha256": fixture.value[
                "installed_wheel_manifest_sha256"
            ],
            "producer_readiness_sha256": producer_readiness[
                "readiness_sha256"
            ],
            "run_id": fixture.value["run_id"],
            "started_at_unix_ms": started_at,
            "api_started_at_unix_ms": api_started_at,
            "api_completed_at_unix_ms": conversation.completed_at_unix_ms,
            "completed_at_unix_ms": completed_at,
            "runtime_receipt": runtime,
            "bundles": bundles,
            "cleanup_receipt": cleanup,
            "cleanup_finalization": copy.deepcopy(
                dict(cleanup_finalization)
            ),
        }
        evidence_path, evidence_sha256 = self.evidence_publisher(fixture, evidence)
        verification = self.verifier(
            fixture_path=fixture.path,
            fixture_sha256=fixture.sha256,
            evidence_path=evidence_path,
            evidence_sha256=evidence_sha256,
        )
        return {
            "schema": LIVE_DRIVER_SCHEMA,
            "ok": True,
            "release_sha": self.plan.revision,
            "capability_plan_sha256": self.plan.sha256,
            "full_canary_plan_sha256": self.full_plan.sha256,
            "run_id": fixture.value["run_id"],
            "reviewed_objective_ids": [
                item.objective_id for item in REVIEWED_OBJECTIVES
            ],
            "api_session_id": conversation.session_id,
            "assistant_response_sha256": _sha256_bytes(
                _canonical_bytes(conversation.assistant_completed)
            ),
            "worker_restart_receipt": copy.deepcopy(dict(restart_receipt)),
            "producer_retirement_receipt": copy.deepcopy(
                dict(producer_retirement)
            ),
            "production_before_observation_sha256": production_before[
                "observation_sha256"
            ],
            "production_diff_sha256": production_diff["diff_sha256"],
            "fixture_path": str(fixture.path),
            "fixture_sha256": fixture.sha256,
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha256,
            "offline_verification_receipt": copy.deepcopy(dict(verification)),
        }


def _build_driver() -> HonestCapabilityCanaryDriver:
    plan = load_capability_plan()
    full_plan = load_full_canary_plan()
    capability_approval = load_capability_approval()
    lifecycle = CapabilityCanaryLifecycle(plan, full_plan)
    collector = RootEvidenceCollector(full_plan)
    owner_subject_sha256 = capability_approval.value[
        "owner_subject_sha256"
    ]
    published_markers: dict[str, Mapping[str, Any]] = {}
    observer_source_state: dict[str, Mapping[str, Any]] = {}

    def production_observation_gate(
        phase: str,
        fixture: PublishedFixture,
        deadline: float,
    ) -> Mapping[str, Any]:
        if phase not in {"before", "after"}:
            _fail("production_observation_phase_invalid")
        installed = load_installed_producer_foundation()
        endpoint = installed.value["endpoints"]["gateway_observer"]
        observer_gid = endpoint.get("gid")
        if type(observer_gid) is not int or observer_gid <= 0:
            _fail("production_observer_identity_invalid")
        if phase not in published_markers:
            published_markers[phase] = (
                publish_capability_production_observation_marker(
                    plan,
                    phase=phase,
                    fixture_sha256=fixture.sha256,
                    run_id=fixture.value["run_id"],
                    owner_subject_sha256=owner_subject_sha256,
                    observer_gid=observer_gid,
                )
            )
        path = (
            DEFAULT_RECEIPT_ROOT
            / fixture.value["run_id"]
            / (
                f"production-observation-{phase}.json"
                if phase == "before"
                else "production-diff.json"
            )
        )
        while not os.path.lexists(path):
            if time.monotonic() >= deadline:
                _fail(f"production_{phase}_observation_timeout")
            time.sleep(0.05)
        if phase == "before":
            return load_staged_owner_signed_production_observation(
                plan=plan,
                phase="before",
                fixture_sha256=fixture.sha256,
                run_id=fixture.value["run_id"],
                owner_subject_sha256=owner_subject_sha256,
                observer_gid=observer_gid,
            )
        return load_published_capability_production_diff(
            plan=plan,
            fixture_sha256=fixture.sha256,
            run_id=fixture.value["run_id"],
            owner_subject_sha256=owner_subject_sha256,
            observer_gid=observer_gid,
        )

    def gateway_observer_source_publisher(
        foundation: TrustedProducerFoundation,
        fixture: PublishedFixture,
        conversation: SSEConversation,
        restart_receipt: Mapping[str, Any],
        lifecycle_start_result: Mapping[str, Any],
        producer_readiness: Mapping[str, Any],
        deadline: float,
    ) -> Mapping[str, Any]:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            _fail("gateway_observer_source_publication_timeout")
        self_collector = collector
        self_collector.wait_session_end(timeout=remaining)
        frames = self_collector.frames
        readiness = getattr(self_collector, "_collector_readiness", None)
        if not isinstance(readiness, Mapping) or not frames:
            _fail("gateway_observer_source_unavailable")
        live_preflight = collect_capability_preflight(
            plan,
            full_plan,
            phase="live",
        )
        evidence = live_preflight.get("evidence")
        gateway = (
            evidence.get("gateway.readiness")
            if isinstance(evidence, Mapping)
            else None
        )
        connector = (
            evidence.get("discord_connector.runtime")
            if isinstance(evidence, Mapping)
            else None
        )
        discord_ready = (
            connector.get("discord_gateway_ready")
            if isinstance(connector, Mapping)
            else None
        )
        if (
            live_preflight.get("ok") is not True
            or not isinstance(gateway, Mapping)
            or not isinstance(connector, Mapping)
            or not isinstance(discord_ready, Mapping)
            or not isinstance(discord_ready.get("bot_user_id"), str)
            or not isinstance(lifecycle_start_result.get("receipt_sha256"), str)
            or producer_readiness.get("run_id") != fixture.value["run_id"]
        ):
            _fail("gateway_observer_runtime_source_invalid")
        runtime_source_identity = {
            "gateway_process_identity_sha256": _sha256_bytes(
                _canonical_bytes(
                    {
                        "service_unit": "hermes-cloud-gateway.service",
                        "main_pid": gateway.get("gateway_pid"),
                        "gateway_readiness_receipt_sha256": gateway.get(
                            "receipt_sha256"
                        ),
                        "lifecycle_start_receipt_sha256": (
                            lifecycle_start_result["receipt_sha256"]
                        ),
                    }
                )
            ),
            "discord_connector_readiness_sha256": connector.get(
                "receipt_sha256"
            ),
            "connector_bot_user_id": discord_ready["bot_user_id"],
            "connector_bot_user_id_provenance": (
                "discord_gateway_ready_user_id"
            ),
        }
        observed_at = max(
            int(time.time() * 1000),
            conversation.completed_at_unix_ms,
            restart_receipt.get("completed_at_unix_ms", 0),
            *(frame.value["observed_at_unix_ms"] for frame in frames),
        )
        projection = build_gateway_observer_source_projection(
            foundation=foundation.value,
            fixture_sha256=fixture.sha256,
            run_id=fixture.value["run_id"],
            producer_readiness=producer_readiness,
            collector_readiness=readiness,
            runtime_source_identity=runtime_source_identity,
            frames=frames,
            worker_restart_receipt=restart_receipt,
            api_terminal_event_identity=(
                build_api_terminal_event_identity(conversation)
            ),
            observed_at_unix_ms=observed_at,
        )
        run_id = fixture.value["run_id"]
        if run_id in observer_source_state:
            _fail("gateway_observer_source_replaced")
        observer_source_state[run_id] = {
            "projection": copy.deepcopy(dict(projection)),
            "model_proposal_cores": (
                extract_gateway_observer_model_proposal_cores(frames)
            ),
        }
        return publish_gateway_observer_source_projection(
            projection,
            foundation=foundation.value,
            fixture_sha256=fixture.sha256,
            run_id=fixture.value["run_id"],
        )

    def foundation_check() -> TrustedProducerFoundation:
        try:
            installed = load_installed_producer_foundation()
        except Exception as exc:
            raise CapabilityLiveDriverError(
                "producer_foundation_unavailable"
            ) from exc
        return trust_producer_foundation(
            installed.value,
            pinned_owner_public_key_ed25519_hex=(
                installed.pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                installed.pinned_owner_public_key_source_sha256
            ),
        )

    def fixture_publisher(
        foundation: TrustedProducerFoundation,
    ) -> PublishedFixture:
        return publish_reviewed_fixture(
            plan=plan,
            full_plan=full_plan,
            producer_foundation=foundation.value,
            pinned_owner_public_key_ed25519_hex=(
                foundation.pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                foundation.pinned_owner_public_key_source_sha256
            ),
        )

    def inbox_factory(fixture: PublishedFixture) -> FixedReceiptInbox:
        try:
            installed = load_installed_producer_foundation()
            receipt = installed.value["receipt_contract"]
        except Exception as exc:
            raise CapabilityLiveDriverError(
                "producer_foundation_unavailable"
            ) from exc
        if (
            fixture.value.get("producer_foundation_sha256")
            != installed.sha256
            or receipt.get("base_root") != str(DEFAULT_RECEIPT_ROOT)
        ):
            _fail("fixture_producer_foundation_mismatch")
        return FixedReceiptInbox(
            fixture=fixture,
            root=Path(receipt["base_root"]),
            role_identities={
                **{
                    role: (
                        installed.value["endpoints"][role]["uid"],
                        installed.value["endpoints"][role]["gid"],
                    )
                    for role in ENDPOINT_ROLES
                },
                "owner": (0, 0),
            },
            run_identity=(
                receipt["run_directory_uid"],
                receipt["run_directory_gid"],
                receipt["run_directory_mode"],
            ),
        )

    def fleet_activator(
        foundation: TrustedProducerFoundation,
        fixture: PublishedFixture,
    ) -> ActivatedProducerFleet:
        installed = InstalledProducerFoundation(
            value=foundation.value,
            pinned_owner_public_key_ed25519_hex=(
                foundation.pinned_owner_public_key_ed25519_hex
            ),
            pinned_owner_public_key_source_sha256=(
                foundation.pinned_owner_public_key_source_sha256
            ),
        )
        try:
            endpoint_clients = production_endpoint_clients(installed.value)
            activated: ProductionFleetActivation = activate_production_fleet(
                plan=plan,
                full_plan=full_plan,
                installed_foundation=installed,
                fixture=fixture.value,
                fixture_sha256=fixture.sha256,
                endpoint_clients=endpoint_clients,
            )
        except Exception as exc:
            raise CapabilityLiveDriverError(
                "producer_activation_unavailable"
            ) from exc
        receipt_pump = ProductionReceiptPump(
            installed_foundation=installed,
            readiness=activated.readiness,
            endpoint_clients=endpoint_clients,
        )
        native_pump = FixedNativePublicationPump(pump=receipt_pump)
        non_observer_slots = tuple(
            slot
            for slot in PRODUCTION_PRE_CLEANUP_PUMP_SLOTS
            if SLOT_ROLE[slot] != "gateway_observer"
        )
        observer_slots = tuple(
            slot
            for slot in PRODUCTION_PRE_CLEANUP_PUMP_SLOTS
            if SLOT_ROLE[slot] == "gateway_observer"
        )
        non_observer_receipts: dict[str, Mapping[str, Any]] = {}

        def pump_non_observer(
            deadline: float,
            cancel: threading.Event,
        ) -> Mapping[str, Mapping[str, Any]]:
            produced = native_pump.pump_slots(
                non_observer_slots,
                deadline=deadline,
                cancel=cancel,
            )
            if set(produced) != set(non_observer_slots):
                _fail("producer_pre_cleanup_receipts_incomplete")
            non_observer_receipts.update(produced)
            return produced

        def pump_observer(
            deadline: float,
            cancel: threading.Event,
        ) -> Mapping[str, Mapping[str, Any]]:
            if cancel.is_set():
                _fail("gateway_observer_pump_cancelled")
            if time.monotonic() >= deadline:
                _fail("gateway_observer_pump_timeout")
            run_id = fixture.value["run_id"]
            state = observer_source_state.get(run_id)
            if (
                not isinstance(state, Mapping)
                or set(non_observer_receipts) != set(non_observer_slots)
            ):
                _fail("gateway_observer_payload_sources_unavailable")
            payloads = _build_gateway_observer_payloads(
                fixture=fixture,
                source_projection=state["projection"],
                model_proposal_cores=state["model_proposal_cores"],
                non_observer_receipts=non_observer_receipts,
            )
            if tuple(payloads) != observer_slots:
                _fail("gateway_observer_payload_slots_invalid")
            return {
                slot: receipt_pump.produce(
                    slot=slot,
                    payload=payloads[slot],
                )
                for slot in observer_slots
            }

        return ActivatedProducerFleet(
            readiness=activated.readiness,
            endpoint_activation_receipts=(
                activated.endpoint_activation_receipts
            ),
            pre_cleanup_pump=pump_non_observer,
            observer_pump=pump_observer,
            cleanup_producer=lambda payload: (
                native_pump.pump_cleanup_payload(payload=payload)
            ),
        )

    def fleet_retirer(
        activated: ActivatedProducerFleet,
    ) -> Mapping[str, Any]:
        try:
            return retire_fleet_readiness(
                expected_readiness_sha256=activated.readiness[
                    "readiness_sha256"
                ]
            )
        except Exception as exc:
            raise CapabilityLiveDriverError(
                "producer_retirement_unavailable"
            ) from exc

    return HonestCapabilityCanaryDriver(
        plan=plan,
        full_plan=full_plan,
        lifecycle=lifecycle,
        capability_approval=capability_approval,
        full_approval=load_full_canary_approval(),
        producer_foundation_check=foundation_check,
        fixture_publisher=fixture_publisher,
        producer_fleet_activator=fleet_activator,
        producer_fleet_retirer=fleet_retirer,
        production_observation_gate=production_observation_gate,
        gateway_observer_source_publisher=(
            gateway_observer_source_publisher
        ),
        inbox_factory=inbox_factory,
        collector=collector,
        client_factory=lambda control, session: LoopbackCanaryClient(
            control_key=control,
            session_key=session,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the signed production-shaped Muncho capability canary"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("publish-fixture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = _build_driver().run()
        else:
            raw = sys.stdin.buffer.read(MAX_FIXTURE_BYTES + 1)
            if not raw or len(raw) > MAX_FIXTURE_BYTES or sys.stdin.buffer.read(1):
                _fail("fixture_publication_authority_invalid")
            authority = _strict_json(
                raw, "fixture_publication_authority_invalid"
            )
            plan = load_capability_plan()
            full_plan = load_full_canary_plan()
            installed = load_installed_producer_foundation()
            result = install_reviewed_fixture(
                authority,
                producer_foundation=installed.value,
                pinned_owner_public_key_ed25519_hex=(
                    installed.pinned_owner_public_key_ed25519_hex
                ),
                pinned_owner_public_key_source_sha256=(
                    installed.pinned_owner_public_key_source_sha256
                ),
                plan=plan,
                full_plan=full_plan,
            )
    except BaseException as exc:
        code = (
            exc.code
            if isinstance(exc, CapabilityLiveDriverError)
            else "capability_live_run_failed_closed"
        )
        print(
            json.dumps(
                {"schema": LIVE_DRIVER_SCHEMA, "ok": False, "failure_code": code},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "CapabilityLiveDriverError",
    "FIXTURE_PUBLICATION_AUTHORITY_SCHEMA",
    "FIXTURE_PUBLICATION_RECEIPT_SCHEMA",
    "FixedReceiptInbox",
    "HonestCapabilityCanaryDriver",
    "LIVE_DRIVER_SCHEMA",
    "PublishedFixture",
    "RECEIPT_SLOTS",
    "RESTART_CHECKPOINT_SCHEMA",
    "REVIEWED_OBJECTIVES",
    "build_reviewed_fixture",
    "install_reviewed_fixture",
    "publish_evidence",
    "publish_reviewed_fixture",
    "reviewed_objective_prompt",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
