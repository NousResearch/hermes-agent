#!/usr/bin/env python3
"""Deterministic offline installer for the private Muncho owner gate.

The installer has no Cloud or network client.  It accepts one digest-bound
bundle transferred over IAP, re-verifies the stable fork-pinned release signer
and every byte on the target, installs an immutable versioned release, and
emits a signed install receipt.  Inert systemd selection is a separate command;
the topology/IAM activation seal and Caddy cutover are never created here.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import grp
import hashlib
import json
import os
import platform
import pwd
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_bootstrap_journal as bootstrap_journal
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_stage0 as stage0
from scripts.canary import owner_gate_trust as trust
from scripts.canary import direct_iam_identity_authority as direct_iam


BOOTSTRAP_RECEIPT_SCHEMA = "muncho-owner-gate-offline-install-receipt.v1"
BOOTSTRAP_TRANSACTION_SCHEMA = "muncho-owner-gate-offline-install-transaction.v1"
EXECUTOR_HOSTS_RECEIPT_SCHEMA = "muncho-owner-gate-executor-hosts-file.v1"
MIGRATION_SCHEMA = "muncho-owner-gate-host-attested-credential-migration.v1"
OWNER_DISCORD_USER_ID = "1279454038731264061"
EXPECTED_CREDENTIAL_ID_SHA256 = (
    "63bbfca0778101d21dddf2b53cc774460565042391b918eb2d1c87b9d6d19860"
)
EXPECTED_PUBLIC_KEY_SHA256 = (
    "478c0bd2ee54f733dbb63acd329ad35188a7f091f9c6bdc4b6e64e7d59d5db89"
)
EXPECTED_USER_HANDLE_SHA256 = (
    "a72512de5fcd7fa3e679fcca570c9b4db6ff1e403b6329586ddad90c093ad983"
)
EXPECTED_SOURCE_SERVICE_SHA256 = (
    "da8fc7823f378b77791c291a8c949d8c7e59d872cedb9e4a43501ff41200b9ff"
)
AUTHORITY_UID = 29102
EXECUTOR_UID = 29103
WEB_UID = 29101
MAX_JSON_BYTES = 4 * 1024 * 1024
COMPUTE_API_PRIVATE_VIP = "199.36.153.8"
COMPUTE_API_HOST = "compute.googleapis.com"
PRIVATE_GOOGLE_API_HOSTS = (
    "compute.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
)
COMPUTE_API_HOSTS_LINE = b"".join(
    f"{COMPUTE_API_PRIVATE_VIP} {host}\n".encode("ascii")
    for host in PRIVATE_GOOGLE_API_HOSTS
)
EXECUTOR_HOSTS_FILENAME = "compute-api-hosts"
INSTALL_PHASES = (
    "reverify_bundle_and_runtime",
    "install_fixed_identities_and_directories",
    "generate_or_verify_authority_receipt_key",
    "install_root_owned_configuration_units_firewall_and_hosts",
    "bootstrap_and_verify_canonical_databases",
    "seal_and_publish_immutable_release",
    "emit_signed_inert_install_receipt",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class OwnerGateBootstrapError(RuntimeError):
    """Stable, secret-free offline bootstrap failure."""


@dataclass(frozen=True)
class InstallLayout:
    release_base: Path = Path("/opt/muncho-owner-gate/releases")
    current_link: Path = Path("/opt/muncho-owner-gate/current")
    etc_root: Path = Path("/etc/muncho-owner-gate")
    state_root: Path = Path("/var/lib/muncho-owner-gate")
    run_root: Path = Path("/run/muncho-owner-gate")
    systemd_root: Path = Path("/etc/systemd/system")
    sysusers_root: Path = Path("/usr/lib/sysusers.d")
    tmpfiles_root: Path = Path("/usr/lib/tmpfiles.d")
    sudoers_root: Path = Path("/etc/sudoers.d")
    python: Path = Path("/usr/bin/python3")
    os_release: Path = Path("/etc/os-release")


PRODUCTION_LAYOUT = InstallLayout()


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest: Mapping[str, Any]
    authority: Mapping[str, Any]
    migration: Mapping[str, Any]
    direct_iam_identity: Mapping[str, Any] = field(default_factory=dict)

    @property
    def revision(self) -> str:
        return str(self.manifest["release_revision"])


def _read_regular(
    path: Path,
    *,
    maximum: int,
    expected_uid: int,
    allowed_modes: frozenset[int],
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink not in allowed_nlinks
            or opened.st_uid != expected_uid
            or stat.S_IMODE(opened.st_mode) not in allowed_modes
            or opened.st_size < 1
            or opened.st_size > maximum
        ):
            raise OwnerGateBootstrapError("owner_gate_bootstrap_file_invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_file_read_invalid"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OwnerGateBootstrapError("owner_gate_bootstrap_file_changed")
        return b"".join(chunks)
    except OSError as exc:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_file_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _canonical_json(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_json_invalid") from None
    if (
        not isinstance(value, Mapping)
        or foundation.canonical_json_bytes(value) != raw
    ):
        raise OwnerGateBootstrapError("owner_gate_bootstrap_json_invalid")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_directory_sync_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_canonical_state(
    path: Path,
    value: Mapping[str, Any],
    *,
    mode: int = 0o600,
    expected_uid: int = 0,
    expected_gid: int | None = None,
) -> None:
    raw = foundation.canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != expected_uid
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_state_directory_invalid"
        )
    target_gid = parent.st_gid if expected_gid is None else expected_gid
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, expected_uid, target_gid)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        readback = _read_regular(
            path,
            maximum=MAX_JSON_BYTES,
            expected_uid=expected_uid,
            allowed_modes=frozenset({mode}),
        )
        if readback != raw:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_state_readback_invalid"
            )
    except OSError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_state_write_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _new_transaction_manifest(
    bundle: VerifiedBundle,
    *,
    started_at_unix: int | None = None,
) -> Mapping[str, Any]:
    timestamp = int(time.time()) if started_at_unix is None else started_at_unix
    if type(timestamp) is not int or timestamp <= 0:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_time_invalid")
    base = {
        "schema": "muncho-owner-gate-offline-install-manifest.v2",
        "release_revision": bundle.revision,
        "package_sha256": bundle.manifest["package_sha256"],
        "started_at_unix": timestamp,
        "phase_order": list(INSTALL_PHASES),
        "activation_performed": False,
        "cloud_mutation_performed": False,
    }
    with_transaction = {
        **base,
        "transaction_id": foundation.sha256_json(base),
    }
    return {
        **with_transaction,
        "manifest_sha256": foundation.sha256_json(with_transaction),
    }


def _validate_transaction_manifest(
    value: Mapping[str, Any],
    *,
    bundle: VerifiedBundle,
) -> Mapping[str, Any]:
    fields = frozenset({
        "schema",
        "release_revision",
        "package_sha256",
        "started_at_unix",
        "phase_order",
        "activation_performed",
        "cloud_mutation_performed",
        "transaction_id",
        "manifest_sha256",
    })
    base = {
        key: item
        for key, item in value.items()
        if key not in {"transaction_id", "manifest_sha256"}
    }
    with_transaction = {
        **base,
        "transaction_id": value.get("transaction_id"),
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema")
        != "muncho-owner-gate-offline-install-manifest.v2"
        or value.get("release_revision") != bundle.revision
        or value.get("package_sha256") != bundle.manifest["package_sha256"]
        or type(value.get("started_at_unix")) is not int
        or value["started_at_unix"] <= 0
        or value.get("phase_order") != list(INSTALL_PHASES)
        or value.get("activation_performed") is not False
        or value.get("cloud_mutation_performed") is not False
        or value.get("transaction_id") != foundation.sha256_json(base)
        or value.get("manifest_sha256")
        != foundation.sha256_json(with_transaction)
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    return dict(value)


def _phase_intent(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    index: int,
    prior_head_sha256: str,
    intent: Mapping[str, Any],
) -> Mapping[str, Any]:
    foundation.canonical_json_bytes(intent)
    unsigned = {
        "schema": "muncho-owner-gate-offline-install-phase-intent.v2",
        "transaction_id": manifest["transaction_id"],
        "phase": phase,
        "phase_index": index,
        "prior_head_sha256": prior_head_sha256,
        "intent": dict(intent),
        "intent_payload_sha256": foundation.sha256_json(intent),
    }
    return {
        **unsigned,
        "phase_intent_sha256": foundation.sha256_json(unsigned),
    }


def _phase_success(
    manifest: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    foundation.canonical_json_bytes(evidence)
    unsigned = {
        "schema": "muncho-owner-gate-offline-install-phase-success.v2",
        "transaction_id": manifest["transaction_id"],
        "phase": intent["phase"],
        "phase_index": intent["phase_index"],
        "prior_head_sha256": intent["prior_head_sha256"],
        "phase_intent_sha256": intent["phase_intent_sha256"],
        "evidence": dict(evidence),
        "evidence_sha256": foundation.sha256_json(evidence),
    }
    return {
        **unsigned,
        "phase_success_sha256": foundation.sha256_json(unsigned),
    }


def _validate_phase_intent(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    phase: str,
    index: int,
    prior_head_sha256: str,
) -> Mapping[str, Any]:
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "phase_intent_sha256"
    }
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "transaction_id",
            "phase",
            "phase_index",
            "prior_head_sha256",
            "intent",
            "intent_payload_sha256",
            "phase_intent_sha256",
        }
        or value.get("schema")
        != "muncho-owner-gate-offline-install-phase-intent.v2"
        or value.get("transaction_id") != manifest["transaction_id"]
        or value.get("phase") != phase
        or value.get("phase_index") != index
        or value.get("prior_head_sha256") != prior_head_sha256
        or not isinstance(value.get("intent"), Mapping)
        or value.get("intent_payload_sha256")
        != foundation.sha256_json(value["intent"])
        or value.get("phase_intent_sha256")
        != foundation.sha256_json(unsigned)
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    return dict(value)


def _validate_phase_success(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "phase_success_sha256"
    }
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "transaction_id",
            "phase",
            "phase_index",
            "prior_head_sha256",
            "phase_intent_sha256",
            "evidence",
            "evidence_sha256",
            "phase_success_sha256",
        }
        or value.get("schema")
        != "muncho-owner-gate-offline-install-phase-success.v2"
        or value.get("transaction_id") != manifest["transaction_id"]
        or value.get("phase") != intent["phase"]
        or value.get("phase_index") != intent["phase_index"]
        or value.get("prior_head_sha256") != intent["prior_head_sha256"]
        or value.get("phase_intent_sha256")
        != intent["phase_intent_sha256"]
        or not isinstance(value.get("evidence"), Mapping)
        or value.get("evidence_sha256")
        != foundation.sha256_json(value["evidence"])
        or value.get("phase_success_sha256")
        != foundation.sha256_json(unsigned)
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    return dict(value)


def _load_transaction_from_journal(
    journal: bootstrap_journal.BootstrapInstallJournal,
    *,
    bundle: VerifiedBundle,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    manifest_value = journal.read("manifest")
    if manifest_value is None:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    manifest = _validate_transaction_manifest(
        manifest_value,
        bundle=bundle,
    )
    completed: list[Mapping[str, Any]] = []
    prior_head = str(manifest["manifest_sha256"])
    pending: Mapping[str, Any] | None = None
    gap = False
    for index, phase in enumerate(INSTALL_PHASES):
        intent_value = journal.read(f"p{index}-intent")
        success_value = journal.read(f"p{index}-success")
        if intent_value is None:
            if success_value is not None:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_transaction_invalid"
                )
            gap = True
            continue
        if gap or pending is not None:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_transaction_invalid"
            )
        intent = _validate_phase_intent(
            intent_value,
            manifest=manifest,
            phase=phase,
            index=index,
            prior_head_sha256=prior_head,
        )
        if success_value is None:
            pending = intent
            continue
        success = _validate_phase_success(
            success_value,
            manifest=manifest,
            intent=intent,
        )
        completed.append(
            {
                "phase": phase,
                "evidence": dict(success["evidence"]),
                "evidence_sha256": success["evidence_sha256"],
                "phase_intent_sha256": intent["phase_intent_sha256"],
                "phase_success_sha256": success["phase_success_sha256"],
            }
        )
        prior_head = str(success["phase_success_sha256"])
    complete = len(completed) == len(INSTALL_PHASES)
    terminal = journal.read("terminal-success")
    if complete:
        unsigned_terminal = {
            "schema": "muncho-owner-gate-offline-install-terminal.v2",
            "transaction_id": manifest["transaction_id"],
            "prior_head_sha256": prior_head,
            "completed_phase_success_sha256": [
                item["phase_success_sha256"] for item in completed
            ],
            "activation_performed": False,
            "cloud_mutation_performed": False,
        }
        expected_terminal = {
            **unsigned_terminal,
            "terminal_sha256": foundation.sha256_json(unsigned_terminal),
        }
        if terminal is None:
            terminal = journal.publish("terminal-success", expected_terminal)
        elif terminal != expected_terminal:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_transaction_invalid"
            )
        prior_head = str(terminal["terminal_sha256"])
    elif terminal is not None:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    projection = {
        "schema": BOOTSTRAP_TRANSACTION_SCHEMA,
        "release_revision": bundle.revision,
        "package_sha256": bundle.manifest["package_sha256"],
        "started_at_unix": manifest["started_at_unix"],
        "phase_order": list(INSTALL_PHASES),
        "completed_phases": completed,
        "complete": complete,
        "activation_performed": False,
        "cloud_mutation_performed": False,
        "transaction_sha256": prior_head,
        "journal_transaction_id": manifest["transaction_id"],
        "journal_manifest_sha256": manifest["manifest_sha256"],
    }
    return projection, pending


def _open_journal(
    path: Path,
    *,
    expected_uid: int,
) -> bootstrap_journal.BootstrapInstallJournal:
    if os.path.lexists(path):
        # The former mutable summary cannot prove an intent-before-effect
        # history and therefore cannot be migrated into canonical truth.
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    expected_gid = int(os.getegid()) if expected_uid == int(os.geteuid()) else 0  # windows-footgun: ok — Debian root boundary
    return bootstrap_journal.BootstrapInstallJournal(
        path,
        owner_uid=expected_uid,
        owner_gid=expected_gid,
    )


def load_or_create_transaction(
    path: Path,
    *,
    bundle: VerifiedBundle,
    expected_uid: int = 0,
    started_at_unix: int | None = None,
) -> Mapping[str, Any]:
    journal = _open_journal(path, expected_uid=expected_uid)
    try:
        with journal.transaction_lease(create=True):
            manifest = journal.read("manifest")
            if manifest is None:
                manifest = journal.publish(
                    "manifest",
                    _new_transaction_manifest(
                        bundle,
                        started_at_unix=started_at_unix,
                    ),
                )
            _validate_transaction_manifest(manifest, bundle=bundle)
            projection, _pending = _load_transaction_from_journal(
                journal,
                bundle=bundle,
            )
            return projection
    except bootstrap_journal.BootstrapJournalError as exc:
        raise OwnerGateBootstrapError(str(exc)) from None


# Private compatibility seam for focused tests; production truth is the
# append-only manifest published by ``load_or_create_transaction``.
_new_transaction = _new_transaction_manifest


@contextmanager
def _exclusive_transaction_lock(
    journal_path: Path,
    *,
    expected_uid: int,
) -> Iterator[None]:
    """Compatibility wrapper around the journal's whole-transaction lease."""

    journal = _open_journal(journal_path, expected_uid=expected_uid)
    try:
        with journal.transaction_lease(create=True):
            yield
    except bootstrap_journal.BootstrapJournalError as exc:
        raise OwnerGateBootstrapError(str(exc)) from None


def run_install_transaction(
    *,
    bundle: VerifiedBundle,
    journal_path: Path,
    handlers: Mapping[str, Callable[[], Mapping[str, Any]]],
    revalidators: Mapping[
        str,
        Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ],
    expected_uid: int = 0,
    started_at_unix: int | None = None,
    transaction_context: dict[str, Any] | None = None,
    intent_builders: Mapping[str, Callable[[], Mapping[str, Any]]] | None = None,
    fresh_target_guard: Callable[[frozenset[str]], None] | None = None,
) -> Mapping[str, Any]:
    """Run/replay an immutable intent-before-effect transaction journal."""

    if (
        set(handlers) != set(INSTALL_PHASES)
        or set(revalidators) != set(INSTALL_PHASES)
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_handlers_invalid"
        )
    if intent_builders is not None and set(intent_builders) != set(INSTALL_PHASES):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_handlers_invalid"
        )
    journal = _open_journal(journal_path, expected_uid=expected_uid)
    try:
        with journal.transaction_lease(create=True):
            manifest = journal.read("manifest")
            if manifest is None:
                manifest = journal.publish(
                    "manifest",
                    _new_transaction_manifest(
                        bundle,
                        started_at_unix=started_at_unix,
                    ),
                )
            manifest = _validate_transaction_manifest(manifest, bundle=bundle)
            value, pending = _load_transaction_from_journal(
                journal,
                bundle=bundle,
            )
            completed = list(value["completed_phases"])
            if transaction_context is not None:
                transaction_context.clear()
                transaction_context.update(
                    {
                        "started_at_unix": value["started_at_unix"],
                        "transaction_id": manifest["transaction_id"],
                        "journal_manifest_sha256": manifest[
                            "manifest_sha256"
                        ],
                        "phase_evidence": {
                            str(item["phase"]): dict(item["evidence"])
                            for item in completed
                        },
                        "phase_intents": {
                            phase: dict(intent["intent"])
                            for index, phase in enumerate(INSTALL_PHASES)
                            if (
                                intent := journal.read(f"p{index}-intent")
                            )
                            is not None
                        },
                    }
                )
            if fresh_target_guard is not None:
                authorized_phases = {
                    str(item["phase"]) for item in completed
                }
                if pending is not None:
                    authorized_phases.add(str(pending["phase"]))
                fresh_target_guard(frozenset(authorized_phases))
            for item in completed:
                phase = str(item["phase"])
                observed = revalidators[phase](dict(item["evidence"]))
                if (
                    not isinstance(observed, Mapping)
                    or foundation.sha256_json(observed)
                    != item["evidence_sha256"]
                ):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_committed_phase_drift"
                    )
            for index in range(len(completed), len(INSTALL_PHASES)):
                phase = INSTALL_PHASES[index]
                if pending is None:
                    prior_head = (
                        str(completed[-1]["phase_success_sha256"])
                        if completed
                        else str(manifest["manifest_sha256"])
                    )
                    if transaction_context is not None:
                        transaction_context["next_prior_head_sha256"] = (
                            prior_head
                        )
                    payload = (
                        intent_builders[phase]()
                        if intent_builders is not None
                        else {
                            "schema": "muncho-owner-gate-generic-phase-intent.v1",
                            "phase": phase,
                        }
                    )
                    if not isinstance(payload, Mapping):
                        raise OwnerGateBootstrapError(
                            "owner_gate_bootstrap_phase_intent_invalid"
                        )
                    pending = journal.publish(
                        f"p{index}-intent",
                        _phase_intent(
                            manifest,
                            phase=phase,
                            index=index,
                            prior_head_sha256=prior_head,
                            intent=payload,
                        ),
                    )
                    pending = _validate_phase_intent(
                        pending,
                        manifest=manifest,
                        phase=phase,
                        index=index,
                        prior_head_sha256=prior_head,
                    )
                elif pending.get("phase") != phase:
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_transaction_invalid"
                    )
                if transaction_context is not None:
                    transaction_context["active_phase_intent"] = dict(
                        pending["intent"]
                    )
                    transaction_context["active_phase_intent_sha256"] = (
                        pending["phase_intent_sha256"]
                    )
                    transaction_context["active_prior_head_sha256"] = (
                        pending["prior_head_sha256"]
                    )
                evidence = handlers[phase]()
                if not isinstance(evidence, Mapping):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_phase_evidence_invalid"
                    )
                foundation.canonical_json_bytes(evidence)
                if transaction_context is not None:
                    transaction_context["phase_evidence"][phase] = dict(
                        evidence
                    )
                success = journal.publish(
                    f"p{index}-success",
                    _phase_success(
                        manifest,
                        intent=pending,
                        evidence=evidence,
                    ),
                )
                success = _validate_phase_success(
                    success,
                    manifest=manifest,
                    intent=pending,
                )
                completed.append(
                    {
                        "phase": phase,
                        "evidence": dict(evidence),
                        "evidence_sha256": success["evidence_sha256"],
                        "phase_intent_sha256": pending[
                            "phase_intent_sha256"
                        ],
                        "phase_success_sha256": success[
                            "phase_success_sha256"
                        ],
                    }
                )
                pending = None
            value, pending = _load_transaction_from_journal(
                journal,
                bundle=bundle,
            )
            if pending is not None or not value["complete"]:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_transaction_invalid"
                )
            return value
    except bootstrap_journal.BootstrapJournalError as exc:
        raise OwnerGateBootstrapError(str(exc)) from None


def _b64url(value: Any, *, maximum: int) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_base64_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_base64_invalid") from None
    if not raw or len(raw) > maximum:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_base64_invalid")
    return raw


def validate_migration(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    host_public_key: Ed25519PublicKey,
    host_key_id: str,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "release_revision",
        "source_service_sha256",
        "owner_discord_user_id",
        "credential_id_b64url",
        "credential_id_sha256",
        "public_key_cose_b64url",
        "public_key_cose_sha256",
        "expected_user_handle_b64url",
        "expected_user_handle_sha256",
        "initial_sign_count",
        "initial_credential_backed_up",
        "source_receipt_sha256",
        "collected_at_unix",
        "host_collector_public_key_id",
        "envelope_sha256",
        "signature_ed25519_b64url",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_migration_invalid")
    credential_id = _b64url(value.get("credential_id_b64url"), maximum=4096)
    public_key = _b64url(value.get("public_key_cose_b64url"), maximum=4096)
    user_handle = _b64url(
        value.get("expected_user_handle_b64url"), maximum=256
    )
    unsigned = {
        key: item
        for key, item in value.items()
        if key not in {"envelope_sha256", "signature_ed25519_b64url"}
    }
    signed = {**unsigned, "envelope_sha256": value["envelope_sha256"]}
    if (
        value.get("schema") != MIGRATION_SCHEMA
        or value.get("release_revision") != release_revision
        or value.get("source_service_sha256") != EXPECTED_SOURCE_SERVICE_SHA256
        or value.get("owner_discord_user_id") != OWNER_DISCORD_USER_ID
        or hashlib.sha256(credential_id).hexdigest()
        != EXPECTED_CREDENTIAL_ID_SHA256
        or value.get("credential_id_sha256")
        != EXPECTED_CREDENTIAL_ID_SHA256
        or hashlib.sha256(public_key).hexdigest()
        != EXPECTED_PUBLIC_KEY_SHA256
        or value.get("public_key_cose_sha256") != EXPECTED_PUBLIC_KEY_SHA256
        or hashlib.sha256(user_handle).hexdigest()
        != EXPECTED_USER_HANDLE_SHA256
        or value.get("expected_user_handle_sha256")
        != EXPECTED_USER_HANDLE_SHA256
        or value.get("initial_sign_count") != 0
        or value.get("initial_credential_backed_up") is not True
        or _SHA256.fullmatch(str(value.get("source_receipt_sha256", "")))
        is None
        or type(value.get("collected_at_unix")) is not int
        or value["collected_at_unix"] <= 0
        or value.get("host_collector_public_key_id") != host_key_id
        or value.get("envelope_sha256") != foundation.sha256_json(unsigned)
    ):
        raise OwnerGateBootstrapError("owner_gate_bootstrap_migration_invalid")
    signature = _b64url(value.get("signature_ed25519_b64url"), maximum=64)
    if len(signature) != 64:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_migration_invalid")
    try:
        host_public_key.verify(
            signature,
            foundation.canonical_json_bytes(signed),
        )
    except InvalidSignature as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_migration_signature_invalid"
        ) from None
    return dict(value)


def _load_raw_ed25519(path: Path, *, expected_uid: int) -> Ed25519PublicKey:
    raw = _read_regular(
        path,
        maximum=32,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if len(raw) != 32:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_collector_key_invalid")
    try:
        return Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_collector_key_invalid"
        ) from None


def verify_bundle(root: Path, *, expected_uid: int = 0) -> VerifiedBundle:
    if not root.is_absolute() or ".." in root.parts or not root.is_dir():
        raise OwnerGateBootstrapError("owner_gate_bootstrap_bundle_invalid")
    authority = trust.load_pinned_release_trust(
        manifest_path=root / "trust/release-trust.json",
        public_key_path=root / "trust/release-trust-signing.pub",
        expected_uid=expected_uid,
    )
    manifest_raw = _read_regular(
        root / "package-manifest.json",
        maximum=MAX_JSON_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    manifest = _canonical_json(manifest_raw)
    try:
        package.validate_authorized_manifest(manifest, authority=authority)
    except package.OwnerGatePackageError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_package_invalid"
        ) from None
    for item in (*manifest["payloads"], *manifest["wheels"]):
        if "release_relative" in item:
            path = root / "payload" / item["release_relative"]
        else:
            path = root / "wheels" / item["filename"]
        digest, size = package._sha256_file(path)
        if digest != item["sha256"] or size != item["size"]:
            raise OwnerGateBootstrapError("owner_gate_bootstrap_bundle_digest_invalid")
    collector_keys: dict[str, Ed25519PublicKey] = {}
    for name, key_id in authority["collector_public_key_ids"].items():
        path = root / "trust" / f"{name}-observation-attestation.pub"
        raw = _read_regular(
            path,
            maximum=32,
            expected_uid=expected_uid,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
        if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != key_id:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_collector_key_invalid"
            )
        collector_keys[name] = Ed25519PublicKey.from_public_bytes(raw)
    migration_raw = _read_regular(
        root / "migration/credential.json",
        maximum=MAX_JSON_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if (
        hashlib.sha256(migration_raw).hexdigest()
        != authority["credential_migration_envelope_sha256"]
    ):
        raise OwnerGateBootstrapError("owner_gate_bootstrap_migration_invalid")
    migration = validate_migration(
        _canonical_json(migration_raw),
        release_revision=str(manifest["release_revision"]),
        host_public_key=collector_keys["host"],
        host_key_id=authority["collector_public_key_ids"]["host"],
    )
    direct_iam_raw = _read_regular(
        root / "trust/direct-iam-identity-authority.json",
        maximum=direct_iam.MAX_BYTES,
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if (
        hashlib.sha256(direct_iam_raw).hexdigest()
        != authority["direct_iam_identity_authority_sha256"]
        or hashlib.sha256(direct_iam_raw).hexdigest()
        != manifest["direct_iam_identity_authority_sha256"]
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_direct_iam_identity_authority_invalid"
        )
    try:
        direct_iam_identity = direct_iam.decode_canonical(
            direct_iam_raw,
            release_revision=str(authority["foundation_source_revision"]),
        )
    except direct_iam.DirectIamIdentityAuthorityError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_direct_iam_identity_authority_invalid"
        ) from None
    if (
        authority["foundation_source_revision"]
        != manifest["foundation_source_revision"]
        or authority["foundation_source_tree_oid"]
        != manifest["foundation_source_tree_oid"]
        or authority["foundation_source_revision"]
        == manifest["release_revision"]
        or direct_iam_identity["release_revision"]
        != authority["foundation_source_revision"]
        or direct_iam_identity["release_revision"]
        != manifest["foundation_source_revision"]
        or direct_iam_identity["pre_foundation_authority_sha256"]
        != authority["pre_foundation_authority_sha256"]
        or direct_iam_identity["pre_foundation_authority_sha256"]
        != manifest["pre_foundation_authority_sha256"]
        or direct_iam_identity["foundation_apply_receipt_sha256"]
        != authority["foundation_apply_receipt_sha256"]
        or direct_iam_identity["foundation_apply_receipt_sha256"]
        != manifest["foundation_apply_receipt_sha256"]
        or direct_iam_identity["resource_ancestor_chain"]
        != authority["resource_ancestor_chain"]
        or direct_iam_identity["resource_ancestor_chain"]
        != manifest["resource_ancestor_chain"]
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_foundation_chain_invalid"
        )
    return VerifiedBundle(
        root=root,
        manifest=manifest,
        authority=authority,
        migration=migration,
        direct_iam_identity=direct_iam_identity,
    )


def _default_runner(argv: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_command_failed") from None
    if completed.returncode != 0 or len(completed.stdout) > MAX_JSON_BYTES:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_command_failed")
    return completed.stdout


def validate_target_runtime(
    bundle: VerifiedBundle,
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    runner: Callable[[Sequence[str]], bytes] = _default_runner,
) -> Mapping[str, Any]:
    executable_paths = (
        layout.python,
        Path("/usr/bin/systemd"),
        Path("/usr/sbin/iptables-nft"),
        Path("/usr/sbin/iptables-nft-save"),
    )
    try:
        stage0._read_exact_os_release(layout.os_release)
        executable_identities = tuple(
            stage0._capture_executable_identity(path)
            for path in executable_paths
        )
    except stage0.OwnerGateStage0Error as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_runtime_identity_invalid"
        ) from None
    python_sha = str(executable_identities[0]["target_sha256"])
    version = runner((str(layout.python), "--version")).decode("ascii").strip()
    systemd = runner(("/usr/bin/systemd", "--version")).decode("ascii").splitlines()
    try:
        executable_identities_after = tuple(
            stage0._capture_executable_identity(path)
            for path in executable_paths
        )
    except stage0.OwnerGateStage0Error as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_runtime_identity_changed"
        ) from None
    if (
        python_sha != bundle.manifest["interpreter_sha256"]
        or version != f"Python {package.PYTHON_VERSION}"
        or platform.machine() != "x86_64"
        or not systemd
        or not systemd[0].startswith("systemd 252 ")
        or executable_identities_after != executable_identities
    ):
        raise OwnerGateBootstrapError("owner_gate_bootstrap_runtime_mismatch")
    unsigned = {
        "schema": "muncho-owner-gate-target-runtime-preflight.v1",
        "release_revision": bundle.revision,
        "python_version": package.PYTHON_VERSION,
        "python_sha256": python_sha,
        "machine": "x86_64",
        "debian_version": "12",
        "systemd_major": 252,
        "iptables_backend": "iptables-nft",
        "executable_identities_sha256": foundation.sha256_json(
            executable_identities
        ),
        "network_install_required": False,
    }
    return {**unsigned, "preflight_sha256": foundation.sha256_json(unsigned)}


IDENTITIES = (
    ("muncho-passkey-web", WEB_UID, "/usr/sbin/nologin"),
    ("muncho-passkey-authority", AUTHORITY_UID, "/usr/sbin/nologin"),
    ("muncho-storage-executor", EXECUTOR_UID, "/usr/sbin/nologin"),
)
IDENTITY_DIRECTORY_REQUIREMENTS = (
    (Path("/etc/muncho-owner-gate/keys"), 0, 0, 0o700),
    (Path("/etc/muncho-owner-gate/executor-keys"), 0, EXECUTOR_UID, 0o710),
    (Path("/etc/muncho-owner-gate/public"), 0, 0, 0o755),
    (
        Path("/var/lib/muncho-owner-gate/authority"),
        AUTHORITY_UID,
        AUTHORITY_UID,
        0o700,
    ),
    (
        Path("/var/lib/muncho-owner-gate/executor"),
        EXECUTOR_UID,
        EXECUTOR_UID,
        0o700,
    ),
    (
        Path("/var/lib/muncho-owner-gate/executor/cloud-attestations"),
        EXECUTOR_UID,
        EXECUTOR_UID,
        0o700,
    ),
    (Path("/var/lib/muncho-owner-gate/bootstrap-receipts"), 0, 0, 0o700),
    (
        Path(
            "/var/lib/muncho-owner-gate/"
            "activation-evidence-staging-receipts"
        ),
        0,
        0,
        0o700,
    ),
)
SYSTEMD_ASSETS = (
    "muncho-owner-gate-metadata-firewall.service",
    "muncho-owner-gate-firewall-readiness.service",
    "muncho-owner-gate-firewall-readiness.timer",
    "muncho-passkey-authority.service",
    "muncho-passkey-authority.socket",
    "muncho-passkey-web.service",
    "muncho-privileged-executor.service",
    "muncho-privileged-executor.socket",
)


def _install_exact_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    uid: int = 0,
    gid: int = 0,
    _checkpoint: Callable[[str], None] | None = None,
    _write_chunk_bytes: int = 64 * 1024,
) -> Mapping[str, Any]:
    """Publish immutable bytes with a same-filesystem no-replace hard link."""

    digest = hashlib.sha256(payload).hexdigest()
    if not payload or _write_chunk_bytes < 1:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_installed_file_write_failed"
        )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    pending_prefix = f".{path.name}.bootstrap."
    pending = sorted(
        item
        for item in path.parent.iterdir()
        if item.name.startswith(pending_prefix)
        and item.name.endswith(".pending")
    )
    if len(pending) > 1:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_installed_file_pending_ambiguous"
        )

    def metadata(item: Path, *, links: frozenset[int]) -> os.stat_result:
        state = item.lstat()
        if (
            not stat.S_ISREG(state.st_mode)
            or stat.S_ISLNK(state.st_mode)
            or state.st_uid != uid
            or state.st_gid != gid
            or stat.S_IMODE(state.st_mode) != mode
            or state.st_nlink not in links
            or state.st_size > max(MAX_JSON_BYTES, len(payload))
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_installed_file_pending_invalid"
            )
        return state

    if pending:
        scratch = pending[0]
        scratch_state = metadata(scratch, links=frozenset({1, 2}))
        final_exists = path.exists() or path.is_symlink()
        if not final_exists and scratch_state.st_nlink != 1:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_installed_file_pending_invalid"
            )
        if not final_exists and scratch_state.st_nlink == 1:
            try:
                scratch_raw = _read_regular(
                    scratch,
                    maximum=max(MAX_JSON_BYTES, len(payload)),
                    expected_uid=uid,
                    allowed_modes=frozenset({mode}),
                )
            except OwnerGateBootstrapError:
                # The unique nlink=1 scratch is not published truth.  Under
                # the transaction lease, a bounded transaction-owned partial
                # write can be discarded and rebuilt after SIGKILL.
                scratch.unlink()
                _fsync_directory(path.parent)
                pending = []
            else:
                if scratch_raw != payload:
                    scratch.unlink()
                    _fsync_directory(path.parent)
                    pending = []
                else:
                    os.link(scratch, path, follow_symlinks=False)
                    _fsync_directory(path.parent)
                    final_exists = True
        if pending and final_exists:
            final_state = metadata(path, links=frozenset({2}))
            scratch_state = metadata(scratch, links=frozenset({2}))
            scratch_raw = _read_regular(
                scratch,
                maximum=max(MAX_JSON_BYTES, len(payload)),
                expected_uid=uid,
                allowed_modes=frozenset({mode}),
                allowed_nlinks=frozenset({2}),
            )
            final_raw = _read_regular(
                path,
                maximum=max(MAX_JSON_BYTES, len(payload)),
                expected_uid=uid,
                allowed_modes=frozenset({mode}),
                allowed_nlinks=frozenset({2}),
            )
            if (
                scratch_raw != payload
                or final_raw != payload
                or (scratch_state.st_dev, scratch_state.st_ino)
                != (final_state.st_dev, final_state.st_ino)
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_installed_file_pending_invalid"
                )
            scratch.unlink()
            _fsync_directory(path.parent)

    if path.exists() or path.is_symlink():
        raw = _read_regular(
            path,
            maximum=max(MAX_JSON_BYTES, len(payload)),
            expected_uid=uid,
            allowed_modes=frozenset({mode}),
        )
        state = path.lstat()
        if raw != payload or state.st_gid != gid:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_installed_file_conflict"
            )
        return {
            "path": str(path),
            "sha256": digest,
            "mode": f"{mode:04o}",
            "uid": uid,
            "gid": gid,
            "created": False,
        }
    temporary = path.with_name(
        f"{pending_prefix}{os.getpid()}.{os.urandom(16).hex()}.pending"
    )
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view[:_write_chunk_bytes])
            if written <= 0:
                raise OSError
            view = view[written:]
            if view and _checkpoint is not None:
                _checkpoint("scratch_write_progress")
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, uid, gid)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(path.parent)
        if _checkpoint is not None:
            _checkpoint("scratch_fsynced")
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        if _checkpoint is not None:
            _checkpoint("final_linked")
        temporary_state = metadata(temporary, links=frozenset({2}))
        final_state = metadata(path, links=frozenset({2}))
        if (temporary_state.st_dev, temporary_state.st_ino) != (
            final_state.st_dev,
            final_state.st_ino,
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_installed_file_readback_invalid"
            )
        temporary.unlink()
        _fsync_directory(path.parent)
        if _checkpoint is not None:
            _checkpoint("scratch_unlinked")
    except OSError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_installed_file_write_failed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # A published nlink=2 pair is deliberately retained for the next
        # locked replay to reconcile.  Only an unpublished scratch is safe to
        # remove here.
        try:
            if temporary.exists() and not path.exists():
                temporary.unlink()
                _fsync_directory(path.parent)
        except OSError:
            pass
    raw = _read_regular(
        path,
        maximum=max(MAX_JSON_BYTES, len(payload)),
        expected_uid=uid,
        allowed_modes=frozenset({mode}),
    )
    state = path.lstat()
    if raw != payload or state.st_gid != gid:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_installed_file_readback_invalid"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "mode": f"{mode:04o}",
        "uid": uid,
        "gid": gid,
        "created": True,
    }


def _asset(release: Path, name: str) -> Path:
    return release / "ops/muncho/owner-gate" / name


def _private_api_hosts_mapping() -> str:
    return COMPUTE_API_HOSTS_LINE.decode("ascii").removesuffix("\n")


def _executor_hosts_binding() -> str:
    return (
        f"{PRODUCTION_LAYOUT.etc_root / EXECUTOR_HOSTS_FILENAME}:/etc/hosts"
    )


def _validate_executor_hosts_receipt(
    receipt: Any,
    *,
    path: Path,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_receipt_invalid"
        )
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    expected_sha256 = hashlib.sha256(COMPUTE_API_HOSTS_LINE).hexdigest()
    if (
        set(receipt)
        != {
            "schema",
            "path",
            "source_asset",
            "mapping",
            "content_sha256",
            "size_bytes",
            "mode",
            "uid",
            "gid",
            "nlink",
            "bind_read_only",
            "global_etc_hosts_mutated",
            "receipt_sha256",
        }
        or receipt.get("schema") != EXECUTOR_HOSTS_RECEIPT_SCHEMA
        or receipt.get("path") != str(path)
        or receipt.get("source_asset") != "compute-api-hosts.fragment"
        or receipt.get("mapping") != _private_api_hosts_mapping()
        or receipt.get("content_sha256") != expected_sha256
        or receipt.get("size_bytes") != len(COMPUTE_API_HOSTS_LINE)
        or receipt.get("mode") != "0444"
        or receipt.get("uid") != expected_uid
        or receipt.get("gid") != expected_gid
        or receipt.get("nlink") != 1
        or receipt.get("bind_read_only") != _executor_hosts_binding()
        or receipt.get("global_etc_hosts_mutated") is not False
        or receipt.get("receipt_sha256") != foundation.sha256_json(unsigned)
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_receipt_invalid"
        )
    raw = _read_regular(
        path,
        maximum=len(COMPUTE_API_HOSTS_LINE),
        expected_uid=expected_uid,
        allowed_modes=frozenset({0o444}),
    )
    state = path.lstat()
    if (
        raw != COMPUTE_API_HOSTS_LINE
        or state.st_gid != expected_gid
        or state.st_nlink != 1
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_readback_invalid"
        )
    return dict(receipt)


def _install_executor_hosts_file(
    release: Path,
    *,
    layout: InstallLayout,
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = _asset(release, "compute-api-hosts.fragment")
    payload = source.read_bytes()
    try:
        manifest = json.loads(
            _asset(release, "bootstrap-manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_manifest_invalid"
        ) from None
    if (
        payload != COMPUTE_API_HOSTS_LINE
        or not isinstance(manifest, Mapping)
        or manifest.get("compute_api_hosts_fragment")
        != _private_api_hosts_mapping()
        or manifest.get("compute_api_hosts_path")
        != str(PRODUCTION_LAYOUT.etc_root / EXECUTOR_HOSTS_FILENAME)
        or manifest.get("compute_api_hosts_install")
        != "executor_only_root_owned_file"
        or manifest.get("compute_api_hosts_bind_read_only")
        != _executor_hosts_binding()
        or manifest.get("global_etc_hosts_mutation") is not False
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_manifest_invalid"
        )
    path = layout.etc_root / EXECUTOR_HOSTS_FILENAME
    installed_file = _install_exact_bytes(
        path,
        payload,
        mode=0o444,
        uid=expected_uid,
        gid=expected_gid,
    )
    # This fixed, service-specific path is owned by the bootstrap contract on
    # every replay.  Normalizing ownership makes a crash after file publication
    # but before journal commit indistinguishable from a clean idempotent retry.
    file_evidence = {**installed_file, "created": True}
    state = path.lstat()
    unsigned = {
        "schema": EXECUTOR_HOSTS_RECEIPT_SCHEMA,
        "path": str(path),
        "source_asset": "compute-api-hosts.fragment",
        "mapping": _private_api_hosts_mapping(),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "mode": "0444",
        "uid": expected_uid,
        "gid": expected_gid,
        "nlink": state.st_nlink,
        "bind_read_only": _executor_hosts_binding(),
        "global_etc_hosts_mutated": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": foundation.sha256_json(unsigned),
    }
    return file_evidence, _validate_executor_hosts_receipt(
        receipt,
        path=path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )


def _attest_executor_hosts_service_isolation(
    *,
    layout: InstallLayout,
) -> None:
    source = str(PRODUCTION_LAYOUT.etc_root / EXECUTOR_HOSTS_FILENAME)
    binding = f"BindReadOnlyPaths={source}:/etc/hosts"
    hidden = f"InaccessiblePaths={source}"
    executor_path = (
        layout.systemd_root / "muncho-privileged-executor.service"
    )
    executor_raw = _read_regular(
        executor_path,
        maximum=MAX_JSON_BYTES,
        expected_uid=0,
        allowed_modes=frozenset({0o444}),
    )
    try:
        executor = executor_raw.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_unit_invalid"
        ) from None
    if (
        executor.splitlines().count(binding) != 1
        or "ReadOnlyPaths=/etc/hosts" in executor.splitlines()
        or hidden in executor.splitlines()
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_hosts_unit_invalid"
        )
    for name in (
        "muncho-passkey-authority.service",
        "muncho-passkey-web.service",
        "muncho-owner-gate-firewall-readiness.service",
        "muncho-owner-gate-metadata-firewall.service",
    ):
        raw = _read_regular(
            layout.systemd_root / name,
            maximum=MAX_JSON_BYTES,
            expected_uid=0,
            allowed_modes=frozenset({0o444}),
        )
        try:
            unit = raw.decode("ascii", errors="strict").splitlines()
        except UnicodeError as exc:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_executor_hosts_unit_invalid"
            ) from None
        if (
            unit.count(hidden) != 1
            or any(line.startswith("BindReadOnlyPaths=") for line in unit)
            or "ReadOnlyPaths=/etc/hosts" in unit
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_executor_hosts_unit_invalid"
            )


def install_identities_and_directories(
    release: Path,
    *,
    runner: Callable[[Sequence[str]], bytes] = _default_runner,
) -> Mapping[str, Any]:
    sysusers = _asset(release, "muncho-owner-gate.sysusers")
    tmpfiles = _asset(release, "muncho-owner-gate.tmpfiles")
    runner(("/usr/bin/systemd-sysusers", str(sysusers)))
    users: list[dict[str, Any]] = []
    for name, numeric_id, shell in IDENTITIES:
        try:
            user = pwd.getpwnam(name)
            group = grp.getgrnam(name)
        except KeyError as exc:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_identity_missing"
            ) from None
        if (
            user.pw_uid != numeric_id
            or user.pw_gid != numeric_id
            or group.gr_gid != numeric_id
            or user.pw_dir != "/nonexistent"
            or user.pw_shell != shell
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_identity_conflict"
            )
        users.append({
            "name": name,
            "uid": numeric_id,
            "gid": numeric_id,
            "home": "/nonexistent",
            "shell": shell,
        })
    runner(("/usr/bin/systemd-tmpfiles", "--create", str(tmpfiles)))
    directories: list[dict[str, Any]] = []
    for path, uid, gid, mode in IDENTITY_DIRECTORY_REQUIREMENTS:
        state = path.lstat()
        if (
            not stat.S_ISDIR(state.st_mode)
            or state.st_uid != uid
            or state.st_gid != gid
            or stat.S_IMODE(state.st_mode) != mode
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_directory_identity_invalid"
            )
        directories.append({
            "path": str(path),
            "uid": uid,
            "gid": gid,
            "mode": f"{mode:04o}",
        })
    return {
        "schema": "muncho-owner-gate-identities-directories.v1",
        "users": users,
        "directories": directories,
    }


def generate_or_verify_receipt_key(
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    transaction_intent: Mapping[str, Any] | None = None,
    _checkpoint: Callable[[str], None] | None = None,
    _expected_uid: int = 0,
    _expected_gid: int = 0,
) -> Mapping[str, Any]:
    private_path = layout.etc_root / "keys/receipt-signing-key.pem"
    public_path = layout.etc_root / "public/authority-receipt-public.pem"

    def recover_published_pair(path: Path, *, mode: int) -> bytes:
        state = path.lstat()
        if state.st_nlink == 2:
            linked_raw = _read_regular(
                path,
                maximum=4096,
                expected_uid=_expected_uid,
                allowed_modes=frozenset({mode}),
                allowed_nlinks=frozenset({2}),
            )
            _install_exact_bytes(
                path,
                linked_raw,
                mode=mode,
                uid=_expected_uid,
                gid=_expected_gid,
            )
        return _read_regular(
            path,
            maximum=4096,
            expected_uid=_expected_uid,
            allowed_modes=frozenset({mode}),
        )

    if private_path.exists():
        private_raw = recover_published_pair(private_path, mode=0o400)
        try:
            private_key = serialization.load_pem_private_key(
                private_raw,
                password=None,
            )
        except (TypeError, ValueError) as exc:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_receipt_key_invalid"
            ) from None
        if not isinstance(private_key, Ed25519PrivateKey):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_receipt_key_invalid"
            )
        public_key = private_key.public_key()
        expected_public_raw = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        if public_path.exists():
            public_raw = recover_published_pair(public_path, mode=0o444)
            if public_raw != expected_public_raw:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_receipt_key_invalid"
                )
            created = False
        else:
            public_raw = expected_public_raw
            _install_exact_bytes(
                public_path,
                public_raw,
                mode=0o444,
                uid=_expected_uid,
                gid=_expected_gid,
                _checkpoint=(
                    None
                    if _checkpoint is None
                    else lambda label: _checkpoint(f"public_{label}")
                ),
            )
            created = True
    elif public_path.exists():
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_key_invalid"
        )
    else:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        private_raw = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_raw = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        _install_exact_bytes(
            private_path,
            private_raw,
            mode=0o400,
            uid=_expected_uid,
            gid=_expected_gid,
            _checkpoint=(
                None
                if _checkpoint is None
                else lambda label: _checkpoint(f"private_{label}")
            ),
        )
        _install_exact_bytes(
            public_path,
            public_raw,
            mode=0o444,
            uid=_expected_uid,
            gid=_expected_gid,
            _checkpoint=(
                None
                if _checkpoint is None
                else lambda label: _checkpoint(f"public_{label}")
            ),
        )
        created = True
    created_by_transaction = created
    if transaction_intent is not None:
        targets = transaction_intent.get("targets")
        expected = {
            str(private_path): True,
            str(public_path): True,
        }
        if (
            transaction_intent.get("schema")
            != "muncho-owner-gate-receipt-key-phase-intent.v1"
            or not isinstance(targets, list)
            or {
                str(item.get("path")): item.get("created_by_transaction")
                for item in targets
                if isinstance(item, Mapping)
            }
            != expected
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        created_by_transaction = True
    return {
        "schema": "muncho-owner-gate-authority-receipt-key.v1",
        "public_key_sha256": hashlib.sha256(public_raw).hexdigest(),
        "public_key_id": hashlib.sha256(
            public_key.public_bytes_raw()
        ).hexdigest(),
        "private_key_path": str(private_path),
        "public_key_path": str(public_path),
        "generated_on_target": True,
        "created": created_by_transaction,
        "created_by_transaction": created_by_transaction,
    }


def _validate_web_config_asset(value: Any) -> Mapping[str, Any]:
    expected = {
        "schema": "muncho-owner-gate-web-config.v1",
        "listen_host": "0.0.0.0",
        "listen_port": foundation.WEB_LISTEN_PORT,
        "origin": "https://auth.lomliev.com",
        "rp_id": "lomliev.com",
        "owner_discord_user_id": OWNER_DISCORD_USER_ID,
        "authority_socket": str(foundation.PASSKEY_AUTHORITY_SOCKET),
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_web_config_invalid"
        )
    return dict(value)


def install_configuration_units_firewall_and_hosts(
    bundle: VerifiedBundle,
    release: Path,
    key_receipt: Mapping[str, Any],
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    runner: Callable[[Sequence[str]], bytes] = _default_runner,
    transaction_intent: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    try:
        executor = json.loads(_asset(release, "executor.json").read_text("utf-8"))
        web = json.loads(_asset(release, "web.json").read_text("utf-8"))
        cloud_attestor = json.loads(
            _asset(release, "cloud-observation-attestor.json").read_text(
                "utf-8"
            )
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_config_invalid"
        ) from None
    _validate_web_config_asset(web)
    replacements = {
        "receipt_public_key_sha256": key_receipt["public_key_sha256"],
        "cloud_observation_public_key_id": bundle.authority[
            "collector_public_key_ids"
        ]["cloud"],
        "cloud_observation_public_key_sha256": bundle.authority[
            "collector_public_key_ids"
        ]["cloud"],
        "host_observation_public_key_id": bundle.authority[
            "collector_public_key_ids"
        ]["host"],
        "host_observation_public_key_sha256": bundle.authority[
            "collector_public_key_ids"
        ]["host"],
        "direct_iam_runtime_service_account_unique_id": bundle.direct_iam_identity.get(
            "owner_gate_service_account_unique_id"
        ),
        "direct_iam_target_service_account_unique_id": bundle.direct_iam_identity.get(
            "target_service_account_unique_id"
        ),
        "direct_iam_runtime_instance_numeric_id": bundle.direct_iam_identity.get(
            "owner_gate_vm_numeric_id"
        ),
        "direct_iam_resource_ancestor_chain": bundle.direct_iam_identity.get(
            "resource_ancestor_chain"
        ),
        "direct_iam_project_read_role": bundle.direct_iam_identity.get(
            "project_read_role"
        ),
        "direct_iam_project_read_role_title": bundle.direct_iam_identity.get(
            "project_read_role_title"
        ),
        "direct_iam_project_read_role_description": bundle.direct_iam_identity.get(
            "project_read_role_description"
        ),
        "direct_iam_ancestor_read_role": bundle.direct_iam_identity.get(
            "ancestor_read_role"
        ),
        "direct_iam_ancestor_read_role_title": bundle.direct_iam_identity.get(
            "ancestor_read_role_title"
        ),
        "direct_iam_ancestor_read_role_description": bundle.direct_iam_identity.get(
            "ancestor_read_role_description"
        ),
        "direct_iam_mutation_role": bundle.direct_iam_identity.get(
            "mutation_role"
        ),
        "direct_iam_mutation_role_title": bundle.direct_iam_identity.get(
            "mutation_role_title"
        ),
        "direct_iam_mutation_role_description": bundle.direct_iam_identity.get(
            "mutation_role_description"
        ),
        "direct_iam_mutation_condition": bundle.direct_iam_identity.get(
            "mutation_condition"
        ),
        "direct_iam_mutation_binding_member": bundle.direct_iam_identity.get(
            "mutation_binding_member"
        ),
        "direct_iam_external_gcp_admin_trust_root": (
            bundle.direct_iam_identity.get("external_gcp_admin_trust_root")
        ),
    }
    expected_placeholders = {
        "receipt_public_key_sha256": "@AUTHORITY_RECEIPT_PUBLIC_KEY_SHA256@",
        "cloud_observation_public_key_id": "@CLOUD_OBSERVATION_PUBLIC_KEY_ID@",
        "cloud_observation_public_key_sha256": "@CLOUD_OBSERVATION_PUBLIC_KEY_SHA256@",
        "host_observation_public_key_id": "@HOST_OBSERVATION_PUBLIC_KEY_ID@",
        "host_observation_public_key_sha256": "@HOST_OBSERVATION_PUBLIC_KEY_SHA256@",
        "direct_iam_runtime_service_account_unique_id": "@OWNER_GATE_SERVICE_ACCOUNT_UNIQUE_ID@",
        "direct_iam_target_service_account_unique_id": "@TARGET_CANARY_SERVICE_ACCOUNT_UNIQUE_ID@",
        "direct_iam_runtime_instance_numeric_id": "@OWNER_GATE_VM_NUMERIC_ID@",
        "direct_iam_resource_ancestor_chain": "@DIRECT_IAM_RESOURCE_ANCESTOR_CHAIN@",
        "direct_iam_project_read_role": "@DIRECT_IAM_PROJECT_READ_ROLE@",
        "direct_iam_project_read_role_title": "@DIRECT_IAM_PROJECT_READ_ROLE_TITLE@",
        "direct_iam_project_read_role_description": "@DIRECT_IAM_PROJECT_READ_ROLE_DESCRIPTION@",
        "direct_iam_ancestor_read_role": "@DIRECT_IAM_ANCESTOR_READ_ROLE@",
        "direct_iam_ancestor_read_role_title": "@DIRECT_IAM_ANCESTOR_READ_ROLE_TITLE@",
        "direct_iam_ancestor_read_role_description": "@DIRECT_IAM_ANCESTOR_READ_ROLE_DESCRIPTION@",
        "direct_iam_mutation_role": "@DIRECT_IAM_MUTATION_ROLE@",
        "direct_iam_mutation_role_title": "@DIRECT_IAM_MUTATION_ROLE_TITLE@",
        "direct_iam_mutation_role_description": "@DIRECT_IAM_MUTATION_ROLE_DESCRIPTION@",
        "direct_iam_mutation_condition": "@DIRECT_IAM_MUTATION_CONDITION@",
        "direct_iam_mutation_binding_member": "@DIRECT_IAM_MUTATION_BINDING_MEMBER@",
        "direct_iam_external_gcp_admin_trust_root": "@DIRECT_IAM_EXTERNAL_GCP_ADMIN_TRUST_ROOT@",
    }
    if any(executor.get(key) != value for key, value in expected_placeholders.items()):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_config_invalid"
        )
    executor.update(replacements)
    direct_identity = bundle.direct_iam_identity
    if (
        not direct_identity
        or executor.get("direct_iam_project_number")
        != direct_identity.get("project_number")
        or executor.get("direct_iam_runtime_service_account_email")
        != direct_identity.get("owner_gate_service_account_email")
        or executor.get("direct_iam_target_service_account_email")
        != direct_identity.get("target_service_account_email")
        or executor.get("direct_iam_metadata_oauth_scopes")
        != direct_identity.get("metadata_oauth_scopes")
        or executor.get("direct_iam_fixed_api_hosts")
        != direct_identity.get("private_google_api_hosts")
        or executor.get("direct_iam_project_read_permissions")
        != direct_identity.get("project_read_permissions")
        or executor.get("direct_iam_ancestor_read_permissions")
        != direct_identity.get("ancestor_read_permissions")
        or executor.get("direct_iam_project_read_role")
        != direct_identity.get("project_read_role")
        or executor.get("direct_iam_project_read_role_title")
        != direct_identity.get("project_read_role_title")
        or executor.get("direct_iam_project_read_role_description")
        != direct_identity.get("project_read_role_description")
        or executor.get("direct_iam_ancestor_read_role")
        != direct_identity.get("ancestor_read_role")
        or executor.get("direct_iam_ancestor_read_role_title")
        != direct_identity.get("ancestor_read_role_title")
        or executor.get("direct_iam_ancestor_read_role_description")
        != direct_identity.get("ancestor_read_role_description")
        or executor.get("direct_iam_mutation_role")
        != direct_identity.get("mutation_role")
        or executor.get("direct_iam_mutation_role_title")
        != direct_identity.get("mutation_role_title")
        or executor.get("direct_iam_mutation_role_description")
        != direct_identity.get("mutation_role_description")
        or executor.get("direct_iam_mutation_condition")
        != direct_identity.get("mutation_condition")
        or executor.get("direct_iam_mutation_binding_member")
        != direct_identity.get("mutation_binding_member")
        or executor.get("direct_iam_mutation_binding_present")
        is not direct_identity.get("mutation_binding_present")
        or executor.get("direct_iam_mutation_activation_seal")
        != direct_identity.get("mutation_activation_seal")
        or executor.get("direct_iam_mutation_activation_seal_present")
        is not direct_identity.get("mutation_activation_seal_present")
        or executor.get("direct_iam_allowed_owner_gate_impersonators")
        != direct_identity.get("allowed_owner_gate_impersonators")
        or executor.get("direct_iam_owner_gate_user_managed_key_inventory")
        != direct_identity.get("owner_gate_user_managed_key_inventory")
        or executor.get("direct_iam_external_gcp_admin_trust_root")
        != direct_identity.get("external_gcp_admin_trust_root")
        or any(
            isinstance(value, str) and value.startswith("@")
            for value in replacements.values()
        )
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_direct_iam_identity_authority_invalid"
        )
    if (
        not isinstance(cloud_attestor, dict)
        or cloud_attestor.get("public_key_id")
        != "@CLOUD_OBSERVATION_PUBLIC_KEY_ID@"
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_executor_config_invalid"
        )
    cloud_attestor["public_key_id"] = replacements[
        "cloud_observation_public_key_id"
    ]
    files: list[Mapping[str, Any]] = []
    static_configs = {
        "authority.json": layout.etc_root / "authority.json",
        "web.json": layout.etc_root / "web.json",
        "metadata-firewall.rules": layout.etc_root / "metadata-firewall.rules",
    }
    for name, destination in static_configs.items():
        files.append(_install_exact_bytes(
            destination,
            _asset(release, name).read_bytes(),
            mode=0o444,
        ))
    files.append(_install_exact_bytes(
        layout.etc_root / "executor.json",
        foundation.canonical_json_bytes(executor),
        mode=0o444,
    ))
    files.append(_install_exact_bytes(
        layout.etc_root / "cloud-observation-attestor.json",
        foundation.canonical_json_bytes(cloud_attestor),
        mode=0o444,
    ))
    python_sha, _ = package._sha256_file(layout.python)
    if python_sha != bundle.manifest["interpreter_sha256"]:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_runtime_mismatch")
    files.append(_install_exact_bytes(
        layout.etc_root / "python3.sha256",
        f"{python_sha}  {layout.python}\n".encode("ascii"),
        mode=0o444,
    ))
    for name in ("cloud", "host"):
        source = bundle.root / "trust" / f"{name}-observation-attestation.pub"
        files.append(_install_exact_bytes(
            layout.etc_root / "public" / f"{name}-observation-attestation.pub",
            source.read_bytes(),
            mode=0o444,
        ))
    files.append(_install_exact_bytes(
        layout.sysusers_root / "muncho-owner-gate.conf",
        _asset(release, "muncho-owner-gate.sysusers").read_bytes(),
        mode=0o444,
    ))
    files.append(_install_exact_bytes(
        layout.tmpfiles_root / "muncho-owner-gate.conf",
        _asset(release, "muncho-owner-gate.tmpfiles").read_bytes(),
        mode=0o444,
    ))
    sudoers_source = _asset(release, "muncho-owner-gate.sudoers")
    runner(("/usr/sbin/visudo", "-cf", str(sudoers_source)))
    files.append(_install_exact_bytes(
        layout.sudoers_root / "muncho-owner-gate",
        sudoers_source.read_bytes(),
        mode=0o440,
    ))
    provisioning_template = _asset(
        release, "muncho-owner-gate-provisioning.sudoers.in"
    ).read_bytes()
    if (
        provisioning_template.count(b"@RELEASE_SHA@") < 1
        or b"/usr/bin/python3" in provisioning_template
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_sudoers_template_invalid"
        )
    provisioning_sudoers = provisioning_template.replace(
        b"@RELEASE_SHA@", bundle.revision.encode("ascii")
    )
    if b"@" in provisioning_sudoers or (
        str(layout.release_base / bundle.revision / "venv/bin/python").encode("ascii")
        not in provisioning_sudoers
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_sudoers_template_invalid"
        )
    provisioning_validation = (
        layout.sudoers_root
        / f".muncho-owner-gate-provisioning.{os.getpid()}.validate"
    )
    validation_descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        validation_descriptor = os.open(provisioning_validation, flags, 0o600)
        view = memoryview(provisioning_sudoers)
        while view:
            written = os.write(validation_descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(validation_descriptor)
        os.close(validation_descriptor)
        validation_descriptor = None
        runner(("/usr/sbin/visudo", "-cf", str(provisioning_validation)))
    except OSError as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_sudoers_template_invalid"
        ) from None
    finally:
        if validation_descriptor is not None:
            os.close(validation_descriptor)
        try:
            provisioning_validation.unlink(missing_ok=True)
        except OSError:
            pass
    files.append(_install_exact_bytes(
        layout.sudoers_root / "muncho-owner-gate-provisioning",
        provisioning_sudoers,
        mode=0o440,
    ))
    executor_hosts_file, executor_hosts_receipt = (
        _install_executor_hosts_file(release, layout=layout)
    )
    files.append(executor_hosts_file)
    for name in SYSTEMD_ASSETS:
        files.append(_install_exact_bytes(
            layout.systemd_root / name,
            _asset(release, name).read_bytes(),
            mode=0o444,
        ))
    _attest_executor_hosts_service_isolation(layout=layout)
    if transaction_intent is not None:
        targets = transaction_intent.get("targets")
        if (
            transaction_intent.get("schema")
            != "muncho-owner-gate-system-files-phase-intent.v1"
            or not isinstance(targets, list)
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        ownership: dict[str, bool] = {}
        for item in targets:
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {"path", "created_by_transaction", "reversible"}
                or item.get("created_by_transaction") is not True
                or type(item.get("reversible")) is not bool
                or not Path(str(item.get("path", ""))).is_absolute()
                or str(item["path"]) in ownership
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_phase_intent_invalid"
                )
            ownership[str(item["path"])] = True
        if ownership != {str(item["path"]): True for item in files}:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        files = [
            {
                **item,
                "created": ownership[str(item["path"])],
                "created_by_transaction": ownership[str(item["path"])],
            }
            for item in files
        ]
    return {
        "schema": "muncho-owner-gate-installed-system-files.v1",
        "files": sorted(files, key=lambda item: str(item["path"])),
        "executor_hosts": executor_hosts_receipt,
        "systemd_units_enabled": [],
        "current_release_selected": False,
        "activation_seal_created": False,
    }


def _sqlite_sidecars(path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{path}{suffix}") for suffix in ("-journal", "-wal", "-shm"))


def _discard_owned_database_stage(path: Path, *, uid: int) -> None:
    for target in (*_sqlite_sidecars(path), path):
        if not os.path.lexists(target):
            continue
        state = target.lstat()
        if (
            not stat.S_ISREG(state.st_mode)
            or stat.S_ISLNK(state.st_mode)
            or state.st_nlink != 1
        or state.st_uid not in {0, uid, int(os.geteuid())}  # windows-footgun: ok — Debian root boundary
            or stat.S_IMODE(state.st_mode) not in {0o600, 0o644}
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_database_stage_invalid"
            )
        target.unlink()
    _fsync_directory(path.parent)


def _publish_database_stage(
    stage_path: Path,
    final_path: Path,
    *,
    uid: int,
    gid: int,
    validate: Callable[[Path], Mapping[str, Any]],
    checkpoint: Callable[[str], None] | None,
    label: str,
) -> Mapping[str, Any]:
    """No-replace publish one fully closed and validated SQLite database."""

    if any(os.path.lexists(path) for path in _sqlite_sidecars(final_path)):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_database_sidecar_present"
        )
    if os.path.lexists(final_path):
        final_state = final_path.lstat()
        if os.path.lexists(stage_path):
            stage_state = stage_path.lstat()
            if (
                (stage_state.st_dev, stage_state.st_ino)
                != (final_state.st_dev, final_state.st_ino)
                or stage_state.st_nlink != 2
                or final_state.st_nlink != 2
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_database_publish_conflict"
                )
            stage_path.unlink()
            _fsync_directory(final_path.parent)
        return validate(final_path)
    if any(os.path.lexists(path) for path in _sqlite_sidecars(stage_path)):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_database_sidecar_present"
        )
    stage_evidence = validate(stage_path)
    stage_state = stage_path.lstat()
    if (
        stage_state.st_uid != uid
        or stage_state.st_gid != gid
        or stat.S_IMODE(stage_state.st_mode) != 0o600
        or stage_state.st_nlink != 1
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_database_stage_invalid"
        )
    os.link(stage_path, final_path, follow_symlinks=False)
    _fsync_directory(final_path.parent)
    if checkpoint is not None:
        checkpoint(f"{label}_final_linked")
    linked_stage = stage_path.lstat()
    linked_final = final_path.lstat()
    if (
        (linked_stage.st_dev, linked_stage.st_ino)
        != (linked_final.st_dev, linked_final.st_ino)
        or linked_stage.st_nlink != 2
        or linked_final.st_nlink != 2
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_database_publish_conflict"
        )
    stage_path.unlink()
    _fsync_directory(final_path.parent)
    if checkpoint is not None:
        checkpoint(f"{label}_stage_unlinked")
    final_evidence = validate(final_path)
    if (
        stage_evidence.get("sqlite_master_sha256")
        != final_evidence.get("sqlite_master_sha256")
        or final_path.lstat().st_nlink != 1
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_database_publish_invalid"
        )
    return final_evidence


def bootstrap_and_verify_databases(
    bundle: VerifiedBundle,
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    key_receipt: Mapping[str, Any],
    now_unix: int,
    require_root: bool = True,
    authority_uid: int = AUTHORITY_UID,
    authority_gid: int = AUTHORITY_UID,
    executor_uid: int = EXECUTOR_UID,
    executor_gid: int = EXECUTOR_UID,
    transaction_intent: Mapping[str, Any] | None = None,
    _checkpoint: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    from scripts.canary import passkey_v2_sqlite as sqlite_backend
    from scripts.canary import passkey_v2_webauthn as webauthn_backend

    authority_path = layout.state_root / "authority/passkey-v2.sqlite3"
    executor_path = layout.state_root / "executor/execution-v2.sqlite3"
    default_authority_stage = authority_path.with_name(
        f".{authority_path.name}.bootstrap-stage"
    )
    default_executor_stage = executor_path.with_name(
        f".{executor_path.name}.bootstrap-stage"
    )
    imported_by_transaction: bool | None = None
    if transaction_intent is None:
        authority_stage = default_authority_stage
        executor_stage = default_executor_stage
    else:
        targets = transaction_intent.get("targets")
        if (
            transaction_intent.get("schema")
            != "muncho-owner-gate-databases-phase-intent.v1"
            or not isinstance(targets, list)
            or transaction_intent.get("credential_imported_by_transaction")
            is not True
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        by_path = {
            str(item.get("path")): item
            for item in targets
            if isinstance(item, Mapping)
        }
        authority_target = by_path.get(str(authority_path))
        executor_target = by_path.get(str(executor_path))
        if (
            len(by_path) != 2
            or not isinstance(authority_target, Mapping)
            or not isinstance(executor_target, Mapping)
            or authority_target.get("created_by_transaction") is not True
            or executor_target.get("created_by_transaction") is not True
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        authority_stage = Path(str(authority_target.get("stage_path", "")))
        executor_stage = Path(str(executor_target.get("stage_path", "")))
        if (
            authority_stage.parent != authority_path.parent
            or executor_stage.parent != executor_path.parent
            or not authority_stage.name.startswith(f".{authority_path.name}.")
            or not executor_stage.name.startswith(f".{executor_path.name}.")
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        imported_by_transaction = True
    migration = bundle.migration
    credential = webauthn_backend.build_migrated_credential(
        owner_discord_user_id=OWNER_DISCORD_USER_ID,
        credential_id=_b64url(migration["credential_id_b64url"], maximum=4096),
        public_key_cose=_b64url(
            migration["public_key_cose_b64url"], maximum=4096
        ),
        rp_id="lomliev.com",
        origin="https://auth.lomliev.com",
        imported_at_unix=int(migration["collected_at_unix"]),
        migration_receipt_sha256=str(migration["envelope_sha256"]),
        initial_sign_count=int(migration["initial_sign_count"]),
        initial_credential_backed_up=bool(
            migration["initial_credential_backed_up"]
        ),
        expected_user_handle=_b64url(
            migration["expected_user_handle_b64url"], maximum=256
        ),
    )
    try:
        receipt_public_key = serialization.load_pem_public_key(
            Path(str(key_receipt["public_key_path"])).read_bytes()
        )
    except (OSError, TypeError, ValueError) as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_key_invalid"
        ) from None
    if not isinstance(receipt_public_key, Ed25519PublicKey):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_key_invalid"
        )
    def authority_database(path: Path):
        return sqlite_backend.PasskeyV2AuthorityDatabase(
            path,
            authority_uid=authority_uid,
            authority_gid=authority_gid,
        )

    def executor_database(path: Path):
        return sqlite_backend.PasskeyV2ExecutorDatabase(
            path,
            executor_uid=executor_uid,
            executor_gid=executor_gid,
            pinned_authority_receipt_public_key=receipt_public_key,
            pinned_authority_receipt_key_id=str(key_receipt["public_key_id"]),
        )

    imported_this_attempt = False
    authority_created_this_attempt = False
    executor_created_this_attempt = False
    if not os.path.lexists(authority_path):
        rebuild = not os.path.lexists(authority_stage)
        if not rebuild:
            try:
                staged_authority = authority_database(authority_stage)
                rebuild = staged_authority.read_active_credentials() != (
                    credential,
                )
            except Exception:
                rebuild = True
        if rebuild:
            _discard_owned_database_stage(
                authority_stage,
                uid=authority_uid,
            )
            sqlite_backend.bootstrap_authority_database(
                authority_stage,
                authority_uid=authority_uid,
                authority_gid=authority_gid,
                now_unix=now_unix,
                require_root=require_root,
            )
            staged_authority = authority_database(authority_stage)
            staged_authority.import_migrated_credential(credential)
            imported_this_attempt = True
            authority_created_this_attempt = True
            if staged_authority.read_active_credentials() != (credential,):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_authority_credential_invalid"
                )
            staged_authority.preflight()
            if _checkpoint is not None:
                _checkpoint("authority_stage_validated")
    _publish_database_stage(
        authority_stage,
        authority_path,
        uid=authority_uid,
        gid=authority_gid,
        validate=lambda path: authority_database(path).preflight(),
        checkpoint=_checkpoint,
        label="authority",
    )
    authority = authority_database(authority_path)
    if authority.read_active_credentials() != (credential,):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_authority_credential_conflict"
        )

    if not os.path.lexists(executor_path):
        rebuild = not os.path.lexists(executor_stage)
        if not rebuild:
            try:
                executor_database(executor_stage).preflight()
            except Exception:
                rebuild = True
        if rebuild:
            _discard_owned_database_stage(
                executor_stage,
                uid=executor_uid,
            )
            sqlite_backend.bootstrap_executor_database(
                executor_stage,
                executor_uid=executor_uid,
                executor_gid=executor_gid,
                now_unix=now_unix,
                require_root=require_root,
            )
            executor_database(executor_stage).preflight()
            executor_created_this_attempt = True
            if _checkpoint is not None:
                _checkpoint("executor_stage_validated")
    _publish_database_stage(
        executor_stage,
        executor_path,
        uid=executor_uid,
        gid=executor_gid,
        validate=lambda path: executor_database(path).preflight(),
        checkpoint=_checkpoint,
        label="executor",
    )
    executor = executor_database(executor_path)
    authority_preflight = authority.preflight()
    executor_preflight = executor.preflight()
    if authority.read_active_credentials() != (credential,):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_authority_credential_invalid"
        )
    return {
        "schema": "muncho-owner-gate-canonical-databases-bootstrap.v1",
        "authority_preflight": authority_preflight,
        "executor_preflight": executor_preflight,
        "credential_id_sha256": credential["credential_id_sha256"],
        "credential_record_sha256": credential["credential_record_sha256"],
        "credential_count": 1,
        "credential_imported_this_attempt": (
            imported_this_attempt
            if imported_by_transaction is None
            else imported_by_transaction
        ),
        "credential_imported_by_transaction": (
            imported_this_attempt
            if imported_by_transaction is None
            else imported_by_transaction
        ),
        "authority_database_created_by_transaction": (
            transaction_intent is not None or authority_created_this_attempt
        ),
        "executor_database_created_by_transaction": (
            transaction_intent is not None or executor_created_this_attempt
        ),
        "append_only_truth": True,
    }


def seal_and_publish_release(
    release: Path,
    bundle: VerifiedBundle,
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    transaction_intent: Mapping[str, Any] | None = None,
    _checkpoint: Callable[[str], None] | None = None,
    _expected_uid: int = 0,
    _expected_gid: int = 0,
) -> Mapping[str, Any]:
    final = layout.release_base / bundle.revision
    staging = layout.release_base / f".{bundle.revision}.bootstrap"
    if release not in {staging, final}:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_release_path_invalid")
    created_by_transaction = not final.exists()
    if transaction_intent is not None:
        if (
            transaction_intent.get("schema")
            != "muncho-owner-gate-release-phase-intent.v1"
            or transaction_intent.get("staging_path") != str(staging)
            or transaction_intent.get("final_path") != str(final)
            or transaction_intent.get("created_by_transaction") is not True
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        created_by_transaction = True
    if final.exists() and staging.exists():
        raise OwnerGateBootstrapError("owner_gate_bootstrap_release_conflict")
    if final.exists() and release == staging and transaction_intent is None:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_release_conflict")
    target = final if final.exists() else staging
    state = target.lstat()
    if (
        not stat.S_ISDIR(state.st_mode)
        or state.st_uid != _expected_uid
        or state.st_gid != _expected_gid
    ):
        raise OwnerGateBootstrapError("owner_gate_bootstrap_release_invalid")
    for item in bundle.manifest["payloads"]:
        path = target / item["release_relative"]
        digest, size = package._sha256_file(path)
        if digest != item["sha256"] or size != item["size"]:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_payload_invalid"
            )
    wheelhouse = target / ".bootstrap-wheelhouse"
    if wheelhouse.exists():
        if wheelhouse.is_symlink() or wheelhouse.parent != target:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_invalid"
            )
        shutil.rmtree(wheelhouse)
    (target / ".bootstrap-wheelhouse-installed.json").unlink(missing_ok=True)
    for cache in sorted(target.rglob("__pycache__"), reverse=True):
        if cache.is_symlink():
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_invalid"
            )
        shutil.rmtree(cache)
    projection: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*"), key=lambda item: str(item.relative_to(target))):
        relative = str(path.relative_to(target))
        item_state = path.lstat()
        if (
            item_state.st_uid != _expected_uid
            or item_state.st_gid != _expected_gid
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_owner_invalid"
            )
        if stat.S_ISLNK(item_state.st_mode):
            link = os.readlink(path)
            if os.path.isabs(link) or ".." in Path(link).parts:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_release_symlink_invalid"
                )
            projection.append({"path": relative, "type": "symlink", "target": link})
            continue
        if stat.S_ISDIR(item_state.st_mode):
            projection.append({"path": relative, "type": "directory", "mode": "0555"})
            continue
        if not stat.S_ISREG(item_state.st_mode):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_node_invalid"
            )
        executable = bool(stat.S_IMODE(item_state.st_mode) & 0o111)
        mode = 0o555 if executable else 0o444
        os.chmod(path, mode, follow_symlinks=False)
        digest, size = package._sha256_file(path)
        projection.append({
            "path": relative,
            "type": "file",
            "mode": f"{mode:04o}",
            "sha256": digest,
            "size": size,
        })
    for directory in sorted(
        (item for item in target.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    target.chmod(0o555)
    if target == staging:
        if os.path.lexists(final):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_conflict"
            )
        os.rename(staging, final)
        _fsync_directory(layout.release_base)
        if _checkpoint is not None:
            _checkpoint("release_renamed")
    readback = final.lstat()
    if (
        not stat.S_ISDIR(readback.st_mode)
        or readback.st_uid != _expected_uid
        or readback.st_gid != _expected_gid
        or stat.S_IMODE(readback.st_mode) != 0o555
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_release_publish_invalid"
        )
    return {
        "schema": "muncho-owner-gate-immutable-release.v1",
        "release_revision": bundle.revision,
        "release_path": str(final),
        "release_tree_sha256": foundation.sha256_json(projection),
        "release_node_count": len(projection),
        "owner": "root:root",
        "mode": "0555",
        "created_by_transaction": created_by_transaction,
        "current_release_selected": False,
    }


def _build_install_receipt_unsigned(
    bundle: VerifiedBundle,
    *,
    transaction_prefix_sha256: str,
    phase_evidence: Mapping[str, Mapping[str, Any]],
    layout: InstallLayout,
    now_unix: int,
) -> Mapping[str, Any]:
    if (
        _SHA256.fullmatch(transaction_prefix_sha256 or "") is None
        or type(now_unix) is not int
        or now_unix <= 0
        or set(phase_evidence) != set(INSTALL_PHASES[:-1])
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_evidence_invalid"
        )
    key_evidence = phase_evidence["generate_or_verify_authority_receipt_key"]
    return {
        "schema": BOOTSTRAP_RECEIPT_SCHEMA,
        "release_revision": bundle.revision,
        "package_sha256": bundle.manifest["package_sha256"],
        "source_tree_oid": bundle.manifest["source_tree_oid"],
        "pre_foundation_authority_sha256": bundle.authority[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": bundle.authority[
            "foundation_apply_receipt_sha256"
        ],
        "project_ancestry_evidence_sha256": bundle.authority[
            "project_ancestry_evidence_sha256"
        ],
        "project_ancestry_chain_sha256": bundle.authority[
            "project_ancestry_chain_sha256"
        ],
        "resource_ancestor_chain": list(
            bundle.authority["resource_ancestor_chain"]
        ),
        "installed_at_unix": now_unix,
        "release_path": str(layout.release_base / bundle.revision),
        "release_tree_sha256": phase_evidence[
            "seal_and_publish_immutable_release"
        ]["release_tree_sha256"],
        "transaction_prefix_sha256": transaction_prefix_sha256,
        "phase_evidence_sha256": {
            phase: foundation.sha256_json(phase_evidence[phase])
            for phase in INSTALL_PHASES[:-1]
        },
        "authority_receipt_public_key_sha256": key_evidence[
            "public_key_sha256"
        ],
        "authority_receipt_public_key_id": key_evidence["public_key_id"],
        "credential_id_sha256": phase_evidence[
            "bootstrap_and_verify_canonical_databases"
        ]["credential_id_sha256"],
        "executor_hosts_receipt_sha256": phase_evidence[
            "install_root_owned_configuration_units_firewall_and_hosts"
        ]["executor_hosts"]["receipt_sha256"],
        "current_release_selected": False,
        "systemd_units_enabled": [],
        "activation_performed": False,
        "activation_seal_created": False,
        "iam_binding_created": False,
        "cloud_mutation_performed": False,
        "caddy_cutover_performed": False,
    }


def emit_signed_install_receipt(
    bundle: VerifiedBundle,
    *,
    transaction_prefix_sha256: str,
    phase_evidence: Mapping[str, Mapping[str, Any]],
    layout: InstallLayout = PRODUCTION_LAYOUT,
    now_unix: int,
    transaction_intent: Mapping[str, Any] | None = None,
    _checkpoint: Callable[[str], None] | None = None,
) -> Mapping[str, Any]:
    required = set(INSTALL_PHASES[:-1])
    if set(phase_evidence) != required:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_evidence_invalid"
        )
    key_evidence = phase_evidence["generate_or_verify_authority_receipt_key"]
    try:
        private_raw = Path(str(key_evidence["private_key_path"])).read_bytes()
        private_key = serialization.load_pem_private_key(private_raw, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_key_invalid"
        ) from None
    if not isinstance(private_key, Ed25519PrivateKey):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_key_invalid"
        )
    unsigned = _build_install_receipt_unsigned(
        bundle,
        transaction_prefix_sha256=transaction_prefix_sha256,
        phase_evidence=phase_evidence,
        layout=layout,
        now_unix=now_unix,
    )
    signed = {**unsigned, "receipt_sha256": foundation.sha256_json(unsigned)}
    signature = private_key.sign(foundation.canonical_json_bytes(signed))
    receipt = {
        **signed,
        "signer_key_id": key_evidence["public_key_id"],
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }
    receipt_path = (
        layout.state_root
        / "bootstrap-receipts"
        / f"install-{bundle.revision}.json"
    )
    created_by_transaction = not os.path.lexists(receipt_path)
    if transaction_intent is not None:
        if (
            transaction_intent.get("schema")
            != "muncho-owner-gate-install-receipt-phase-intent.v1"
            or transaction_intent.get("path") != str(receipt_path)
            or transaction_intent.get("created_by_transaction") is not True
            or transaction_intent.get("transaction_prefix_sha256")
            != transaction_prefix_sha256
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_phase_intent_invalid"
            )
        created_by_transaction = True
    _install_exact_bytes(
        receipt_path,
        foundation.canonical_json_bytes(receipt),
        mode=0o400,
        _checkpoint=_checkpoint,
    )
    return {
        "schema": "muncho-owner-gate-signed-install-receipt-reference.v1",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "pre_foundation_authority_sha256": receipt[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": receipt[
            "foundation_apply_receipt_sha256"
        ],
        "project_ancestry_evidence_sha256": receipt[
            "project_ancestry_evidence_sha256"
        ],
        "project_ancestry_chain_sha256": receipt[
            "project_ancestry_chain_sha256"
        ],
        "resource_ancestor_chain": receipt["resource_ancestor_chain"],
        "signed_receipt_file_sha256": hashlib.sha256(
            foundation.canonical_json_bytes(receipt)
        ).hexdigest(),
        "signer_key_id": receipt["signer_key_id"],
        "created_by_transaction": created_by_transaction,
        "activation_performed": False,
        "cloud_mutation_performed": False,
    }


def _attest_installed_file_evidence(evidence: Mapping[str, Any]) -> None:
    allowed_fields = {"path", "sha256", "mode", "uid", "gid", "created"}
    transaction_fields = allowed_fields | {"created_by_transaction"}
    if (
        not isinstance(evidence, Mapping)
        or frozenset(evidence)
        not in {frozenset(allowed_fields), frozenset(transaction_fields)}
        or _SHA256.fullmatch(str(evidence.get("sha256", ""))) is None
        or type(evidence.get("uid")) is not int
        or type(evidence.get("gid")) is not int
        or type(evidence.get("created")) is not bool
        or (
            "created_by_transaction" in evidence
            and evidence.get("created_by_transaction")
            is not evidence.get("created")
        )
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_committed_phase_drift"
        )
    try:
        mode = int(str(evidence["mode"]), 8)
        path = Path(str(evidence["path"]))
    except (TypeError, ValueError) as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_committed_phase_drift"
        ) from None
    if not path.is_absolute() or mode not in {0o400, 0o440, 0o444}:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_committed_phase_drift"
        )
    raw = _read_regular(
        path,
        maximum=MAX_JSON_BYTES,
        expected_uid=int(evidence["uid"]),
        allowed_modes=frozenset({mode}),
    )
    state = path.lstat()
    if (
        state.st_gid != evidence["gid"]
        or hashlib.sha256(raw).hexdigest() != evidence["sha256"]
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_committed_phase_drift"
        )


def _attest_immutable_release(
    bundle: VerifiedBundle,
    *,
    layout: InstallLayout,
    created_by_transaction: bool = True,
) -> Mapping[str, Any]:
    final = layout.release_base / bundle.revision
    state = final.lstat()
    if (
        not stat.S_ISDIR(state.st_mode)
        or state.st_uid != 0
        or state.st_gid != 0
        or stat.S_IMODE(state.st_mode) != 0o555
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_committed_phase_drift"
        )
    for item in bundle.manifest["payloads"]:
        digest, size = package._sha256_file(final / item["release_relative"])
        if digest != item["sha256"] or size != item["size"]:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
    if (
        (final / ".bootstrap-wheelhouse").exists()
        or (final / ".bootstrap-wheelhouse-installed.json").exists()
        or any(final.rglob("__pycache__"))
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_committed_phase_drift"
        )
    projection: list[dict[str, Any]] = []
    for path in sorted(
        final.rglob("*"),
        key=lambda item: str(item.relative_to(final)),
    ):
        relative = str(path.relative_to(final))
        item_state = path.lstat()
        if item_state.st_uid != 0 or item_state.st_gid != 0:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        if stat.S_ISLNK(item_state.st_mode):
            link = os.readlink(path)
            if os.path.isabs(link) or ".." in Path(link).parts:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
            projection.append({"path": relative, "type": "symlink", "target": link})
        elif stat.S_ISDIR(item_state.st_mode):
            if stat.S_IMODE(item_state.st_mode) != 0o555:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
            projection.append({"path": relative, "type": "directory", "mode": "0555"})
        elif stat.S_ISREG(item_state.st_mode):
            mode = stat.S_IMODE(item_state.st_mode)
            if mode not in {0o444, 0o555}:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
            digest, size = package._sha256_file(path)
            projection.append({
                "path": relative,
                "type": "file",
                "mode": f"{mode:04o}",
                "sha256": digest,
                "size": size,
            })
        else:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
    return {
        "schema": "muncho-owner-gate-immutable-release.v1",
        "release_revision": bundle.revision,
        "release_path": str(final),
        "release_tree_sha256": foundation.sha256_json(projection),
        "release_node_count": len(projection),
        "owner": "root:root",
        "mode": "0555",
        "created_by_transaction": created_by_transaction,
        "current_release_selected": False,
    }


def _revalidate_committed_phase(
    phase: str,
    stored: Mapping[str, Any],
    *,
    bundle: VerifiedBundle,
    layout: InstallLayout,
    runner: Callable[[Sequence[str]], bytes],
    all_evidence: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Read-only re-attestation for a phase already committed to the journal."""

    if phase == "reverify_bundle_and_runtime":
        observed = validate_target_runtime(bundle, layout=layout, runner=runner)
        if observed != stored:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        return dict(stored)

    if phase == "install_fixed_identities_and_directories":
        expected_users = [
            {
                "name": name,
                "uid": numeric_id,
                "gid": numeric_id,
                "home": "/nonexistent",
                "shell": shell,
            }
            for name, numeric_id, shell in IDENTITIES
        ]
        expected_directories = [
            {
                "path": str(path),
                "uid": uid,
                "gid": gid,
                "mode": f"{mode:04o}",
            }
            for path, uid, gid, mode in IDENTITY_DIRECTORY_REQUIREMENTS
        ]
        if (
            stored.get("schema") != "muncho-owner-gate-identities-directories.v1"
            or stored.get("users") != expected_users
            or stored.get("directories") != expected_directories
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        for expected in expected_users:
            try:
                name = str(expected["name"])
                user = pwd.getpwnam(name)
                group = grp.getgrnam(name)
            except KeyError as exc:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                ) from None
            if (
                user.pw_uid != expected["uid"]
                or user.pw_gid != expected["gid"]
                or group.gr_gid != expected["gid"]
                or user.pw_dir != expected["home"]
                or user.pw_shell != expected["shell"]
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
        for directory in expected_directories:
            if (
                not isinstance(directory, Mapping)
                or set(directory) != {"path", "uid", "gid", "mode"}
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
            path = Path(str(directory["path"]))
            state = path.lstat()
            if (
                not stat.S_ISDIR(state.st_mode)
                or state.st_uid != directory["uid"]
                or state.st_gid != directory["gid"]
                or f"{stat.S_IMODE(state.st_mode):04o}" != directory["mode"]
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
        return dict(stored)

    if phase == "generate_or_verify_authority_receipt_key":
        if (
            stored.get("schema") != "muncho-owner-gate-authority-receipt-key.v1"
            or stored.get("private_key_path")
            != str(layout.etc_root / "keys/receipt-signing-key.pem")
            or stored.get("public_key_path")
            != str(layout.etc_root / "public/authority-receipt-public.pem")
            or stored.get("generated_on_target") is not True
            or type(stored.get("created")) is not bool
            or stored.get("created_by_transaction")
            is not stored.get("created")
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        private_raw = _read_regular(
            Path(str(stored["private_key_path"])),
            maximum=4096,
            expected_uid=0,
            allowed_modes=frozenset({0o400}),
        )
        public_raw = _read_regular(
            Path(str(stored["public_key_path"])),
            maximum=4096,
            expected_uid=0,
            allowed_modes=frozenset({0o444}),
        )
        try:
            private_key = serialization.load_pem_private_key(private_raw, password=None)
        except (TypeError, ValueError) as exc:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            ) from None
        if not isinstance(private_key, Ed25519PrivateKey):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        expected_public = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        if (
            public_raw != expected_public
            or hashlib.sha256(public_raw).hexdigest()
            != stored.get("public_key_sha256")
            or hashlib.sha256(private_key.public_key().public_bytes_raw()).hexdigest()
            != stored.get("public_key_id")
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        return dict(stored)

    if phase == "install_root_owned_configuration_units_firewall_and_hosts":
        files = stored.get("files")
        if (
            set(stored)
            != {
                "schema",
                "files",
                "executor_hosts",
                "systemd_units_enabled",
                "current_release_selected",
                "activation_seal_created",
            }
            or stored.get("schema")
            != "muncho-owner-gate-installed-system-files.v1"
            or not isinstance(files, list)
            or stored.get("systemd_units_enabled") != []
            or stored.get("current_release_selected") is not False
            or stored.get("activation_seal_created") is not False
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        paths: dict[str, Mapping[str, Any]] = {}
        for file_evidence in files:
            _attest_installed_file_evidence(file_evidence)
            path = str(file_evidence["path"])
            if path in paths:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_committed_phase_drift"
                )
            paths[path] = file_evidence
        executor_hosts_path = layout.etc_root / EXECUTOR_HOSTS_FILENAME
        executor_hosts = _validate_executor_hosts_receipt(
            stored.get("executor_hosts"),
            path=executor_hosts_path,
        )
        file_evidence = paths.get(str(executor_hosts_path))
        if (
            not isinstance(file_evidence, Mapping)
            or file_evidence.get("sha256")
            != executor_hosts["content_sha256"]
            or file_evidence.get("mode") != executor_hosts["mode"]
            or file_evidence.get("uid") != executor_hosts["uid"]
            or file_evidence.get("gid") != executor_hosts["gid"]
            or file_evidence.get("created") is not True
            or file_evidence.get("created_by_transaction") is not True
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        _attest_executor_hosts_service_isolation(layout=layout)
        return dict(stored)

    if phase == "bootstrap_and_verify_canonical_databases":
        from scripts.canary import passkey_v2_sqlite as sqlite_backend
        from scripts.canary import passkey_v2_webauthn as webauthn_backend

        key_evidence = all_evidence.get(
            "generate_or_verify_authority_receipt_key"
        )
        if not isinstance(key_evidence, Mapping):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        public_raw = _read_regular(
            Path(str(key_evidence["public_key_path"])),
            maximum=4096,
            expected_uid=0,
            allowed_modes=frozenset({0o444}),
        )
        try:
            receipt_public_key = serialization.load_pem_public_key(public_raw)
        except (TypeError, ValueError) as exc:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            ) from None
        if not isinstance(receipt_public_key, Ed25519PublicKey):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        authority = sqlite_backend.PasskeyV2AuthorityDatabase(
            layout.state_root / "authority/passkey-v2.sqlite3",
            authority_uid=AUTHORITY_UID,
            authority_gid=AUTHORITY_UID,
        )
        executor = sqlite_backend.PasskeyV2ExecutorDatabase(
            layout.state_root / "executor/execution-v2.sqlite3",
            executor_uid=EXECUTOR_UID,
            executor_gid=EXECUTOR_UID,
            pinned_authority_receipt_public_key=receipt_public_key,
            pinned_authority_receipt_key_id=str(key_evidence["public_key_id"]),
        )
        migration = bundle.migration
        credential = webauthn_backend.build_migrated_credential(
            owner_discord_user_id=OWNER_DISCORD_USER_ID,
            credential_id=_b64url(migration["credential_id_b64url"], maximum=4096),
            public_key_cose=_b64url(
                migration["public_key_cose_b64url"], maximum=4096
            ),
            rp_id="lomliev.com",
            origin="https://auth.lomliev.com",
            imported_at_unix=int(migration["collected_at_unix"]),
            migration_receipt_sha256=str(migration["envelope_sha256"]),
            initial_sign_count=int(migration["initial_sign_count"]),
            initial_credential_backed_up=bool(
                migration["initial_credential_backed_up"]
            ),
            expected_user_handle=_b64url(
                migration["expected_user_handle_b64url"], maximum=256
            ),
        )
        if (
            stored.get("schema")
            != "muncho-owner-gate-canonical-databases-bootstrap.v1"
            or stored.get("authority_preflight") != authority.preflight()
            or stored.get("executor_preflight") != executor.preflight()
            or authority.read_active_credentials() != (credential,)
            or stored.get("credential_id_sha256")
            != credential["credential_id_sha256"]
            or stored.get("credential_record_sha256")
            != credential["credential_record_sha256"]
            or stored.get("credential_count") != 1
            or type(stored.get("credential_imported_this_attempt")) is not bool
            or stored.get("credential_imported_by_transaction")
            is not stored.get("credential_imported_this_attempt")
            or type(
                stored.get("authority_database_created_by_transaction")
            )
            is not bool
            or type(
                stored.get("executor_database_created_by_transaction")
            )
            is not bool
            or stored.get("append_only_truth") is not True
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        return dict(stored)

    if phase == "seal_and_publish_immutable_release":
        if (
            type(stored.get("created_by_transaction")) is not bool
            or _attest_immutable_release(
                bundle,
                layout=layout,
                created_by_transaction=bool(
                    stored["created_by_transaction"]
                ),
            )
            != stored
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        return dict(stored)

    if phase == "emit_signed_inert_install_receipt":
        key_evidence = all_evidence.get(
            "generate_or_verify_authority_receipt_key"
        )
        if not isinstance(key_evidence, Mapping):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        receipt_path = Path(str(stored.get("receipt_path", "")))
        if receipt_path != (
            layout.state_root
            / "bootstrap-receipts"
            / f"install-{bundle.revision}.json"
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        raw = _read_regular(
            receipt_path,
            maximum=MAX_JSON_BYTES,
            expected_uid=0,
            allowed_modes=frozenset({0o400}),
        )
        receipt = _canonical_json(raw)
        signature_text = receipt.get("signature_ed25519_b64url")
        signature = _b64url(signature_text, maximum=64)
        if (
            len(signature) != 64
            or base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
            != signature_text
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        signed = {
            key: value
            for key, value in receipt.items()
            if key not in {"signer_key_id", "signature_ed25519_b64url"}
        }
        unsigned = {
            key: value for key, value in signed.items() if key != "receipt_sha256"
        }
        public_raw = _read_regular(
            Path(str(key_evidence["public_key_path"])),
            maximum=4096,
            expected_uid=0,
            allowed_modes=frozenset({0o444}),
        )
        try:
            public_key = serialization.load_pem_public_key(public_raw)
            if not isinstance(public_key, Ed25519PublicKey):
                raise TypeError
            public_key.verify(signature, foundation.canonical_json_bytes(signed))
        except (InvalidSignature, TypeError, ValueError) as exc:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            ) from None
        if (
            receipt.get("receipt_sha256") != foundation.sha256_json(unsigned)
            or receipt.get("signer_key_id") != key_evidence.get("public_key_id")
            or hashlib.sha256(raw).hexdigest()
            != stored.get("signed_receipt_file_sha256")
            or stored.get("receipt_sha256") != receipt.get("receipt_sha256")
            or stored.get("signer_key_id") != receipt.get("signer_key_id")
            or receipt.get("pre_foundation_authority_sha256")
            != bundle.authority["pre_foundation_authority_sha256"]
            or receipt.get("foundation_apply_receipt_sha256")
            != bundle.authority["foundation_apply_receipt_sha256"]
            or receipt.get("project_ancestry_evidence_sha256")
            != bundle.authority["project_ancestry_evidence_sha256"]
            or receipt.get("project_ancestry_chain_sha256")
            != bundle.authority["project_ancestry_chain_sha256"]
            or receipt.get("resource_ancestor_chain")
            != bundle.authority["resource_ancestor_chain"]
            or stored.get("pre_foundation_authority_sha256")
            != receipt.get("pre_foundation_authority_sha256")
            or stored.get("foundation_apply_receipt_sha256")
            != receipt.get("foundation_apply_receipt_sha256")
            or stored.get("project_ancestry_evidence_sha256")
            != receipt.get("project_ancestry_evidence_sha256")
            or stored.get("project_ancestry_chain_sha256")
            != receipt.get("project_ancestry_chain_sha256")
            or stored.get("resource_ancestor_chain")
            != receipt.get("resource_ancestor_chain")
            or type(stored.get("created_by_transaction")) is not bool
            or stored.get("activation_performed") is not False
            or stored.get("cloud_mutation_performed") is not False
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_committed_phase_drift"
            )
        return dict(stored)

    raise OwnerGateBootstrapError(
        "owner_gate_bootstrap_transaction_handlers_invalid"
    )


def build_install_plan(bundle: VerifiedBundle) -> Mapping[str, Any]:
    """Return the fixed executable phases; this is not an activation receipt."""

    unsigned = {
        "schema": "muncho-owner-gate-offline-install-plan.v1",
        "release_revision": bundle.revision,
        "package_sha256": bundle.manifest["package_sha256"],
        "phases": [
            "reverify_signed_release_trust_and_all_bundle_bytes",
            "attest_exact_debian12_python3112_systemd252_iptables_nft",
            "create_fixed_no_shell_identities_and_private_directories",
            "copy_versioned_release_and_create_offline_venv",
            "install_exact_wheels_no_index_no_deps_only_binary",
            "generate_authority_receipt_key_and_pin_public_key",
            "install_root_owned_configs_collector_keys_units_and_firewall",
            "install_exact_executor_only_private_api_hosts_file",
            "bootstrap_both_canonical_sqlite_databases",
            "import_one_host_attested_public_webauthn_credential",
            "emit_signed_install_receipt_with_activation_false",
        ],
        "network_install_allowed": False,
        "cloud_mutation_allowed": False,
        "iam_binding_allowed": False,
        "activation_seal_creation_allowed": False,
        "caddy_cutover_allowed": False,
        "inert_selection_separate_command": True,
        "rollback_deletes_databases_or_receipts": False,
        "rollback_reports_only_committed_existing_artifacts": True,
        "global_etc_hosts_mutation_allowed": False,
        "executor_hosts_file_is_service_specific": True,
    }
    return {**unsigned, "plan_sha256": foundation.sha256_json(unsigned)}


def _ensure_bootstrap_state_roots(layout: InstallLayout) -> None:
    layout.state_root.mkdir(parents=True, exist_ok=True, mode=0o711)
    os.chmod(layout.state_root, 0o711)
    os.chown(layout.state_root, 0, 0)
    receipts = layout.state_root / "bootstrap-receipts"
    receipts.mkdir(exist_ok=True, mode=0o700)
    os.chmod(receipts, 0o700)
    os.chown(receipts, 0, 0)
    for path, mode in ((layout.state_root, 0o711), (receipts, 0o700)):
        state = path.lstat()
        if (
            not stat.S_ISDIR(state.st_mode)
            or state.st_uid != 0
            or state.st_gid != 0
            or stat.S_IMODE(state.st_mode) != mode
        ):
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_state_directory_invalid"
            )


def _require_fresh_targets_absent(
    paths: Sequence[Path],
    *,
    phase: str,
) -> None:
    for path in paths:
        if not path.is_absolute() or os.path.lexists(path):
            raise OwnerGateBootstrapError(
                f"owner_gate_bootstrap_fresh_target_conflict:{phase}"
            )


def _configuration_target_paths(layout: InstallLayout) -> tuple[Path, ...]:
    return (
        layout.etc_root / "authority.json",
        layout.etc_root / "web.json",
        layout.etc_root / "metadata-firewall.rules",
        layout.etc_root / "executor.json",
        layout.etc_root / "cloud-observation-attestor.json",
        layout.etc_root / "python3.sha256",
        layout.etc_root / "public/cloud-observation-attestation.pub",
        layout.etc_root / "public/host-observation-attestation.pub",
        layout.sysusers_root / "muncho-owner-gate.conf",
        layout.tmpfiles_root / "muncho-owner-gate.conf",
        layout.sudoers_root / "muncho-owner-gate",
        layout.sudoers_root / "muncho-owner-gate-provisioning",
        layout.etc_root / EXECUTOR_HOSTS_FILENAME,
        *(layout.systemd_root / name for name in SYSTEMD_ASSETS),
    )


def _production_phase_intent_builders(
    *,
    bundle: VerifiedBundle,
    release: Path,
    layout: InstallLayout,
    transaction_context: Mapping[str, Any],
) -> Mapping[str, Callable[[], Mapping[str, Any]]]:
    """Build fixed fresh-VM intents before any managed side effect."""

    private_key = layout.etc_root / "keys/receipt-signing-key.pem"
    public_key = layout.etc_root / "public/authority-receipt-public.pem"
    configuration_targets = _configuration_target_paths(layout)
    authority_database = layout.state_root / "authority/passkey-v2.sqlite3"
    executor_database = layout.state_root / "executor/execution-v2.sqlite3"
    final_release = layout.release_base / bundle.revision
    staging_release = layout.release_base / f".{bundle.revision}.bootstrap"
    install_receipt = (
        layout.state_root
        / "bootstrap-receipts"
        / f"install-{bundle.revision}.json"
    )

    def runtime() -> Mapping[str, Any]:
        return {
            "schema": "muncho-owner-gate-runtime-phase-intent.v1",
            "release_revision": bundle.revision,
            "package_sha256": bundle.manifest["package_sha256"],
            "side_effects": [],
        }

    def identities() -> Mapping[str, Any]:
        return {
            "schema": "muncho-owner-gate-identities-phase-intent.v1",
            "identities": [
                {"name": name, "uid": uid, "gid": uid, "shell": shell}
                for name, uid, shell in IDENTITIES
            ],
            "systemd_sysusers_tmpfiles_exact_or_create": True,
        }

    def receipt_key() -> Mapping[str, Any]:
        _require_fresh_targets_absent(
            (private_key, public_key),
            phase="generate_or_verify_authority_receipt_key",
        )
        return {
            "schema": "muncho-owner-gate-receipt-key-phase-intent.v1",
            "targets": [
                {
                    "path": str(path),
                    "created_by_transaction": True,
                    "preserve_on_rollback": True,
                }
                for path in (private_key, public_key)
            ],
        }

    def system_files() -> Mapping[str, Any]:
        _require_fresh_targets_absent(
            configuration_targets,
            phase="install_root_owned_configuration_units_firewall_and_hosts",
        )
        return {
            "schema": "muncho-owner-gate-system-files-phase-intent.v1",
            "targets": [
                {
                    "path": str(path),
                    "created_by_transaction": True,
                    "reversible": True,
                }
                for path in configuration_targets
            ],
        }

    def databases() -> Mapping[str, Any]:
        transaction_id = str(transaction_context["transaction_id"])
        authority_stage = authority_database.with_name(
            f".{authority_database.name}.{transaction_id}.stage"
        )
        executor_stage = executor_database.with_name(
            f".{executor_database.name}.{transaction_id}.stage"
        )
        _require_fresh_targets_absent(
            (
                authority_database,
                executor_database,
                authority_stage,
                executor_stage,
                *_sqlite_sidecars(authority_database),
                *_sqlite_sidecars(executor_database),
                *_sqlite_sidecars(authority_stage),
                *_sqlite_sidecars(executor_stage),
            ),
            phase="bootstrap_and_verify_canonical_databases",
        )
        return {
            "schema": "muncho-owner-gate-databases-phase-intent.v1",
            "targets": [
                {
                    "path": str(authority_database),
                    "stage_path": str(authority_stage),
                    "created_by_transaction": True,
                    "preserve_on_rollback": True,
                },
                {
                    "path": str(executor_database),
                    "stage_path": str(executor_stage),
                    "created_by_transaction": True,
                    "preserve_on_rollback": True,
                },
            ],
            "credential_imported_by_transaction": True,
            "credential_id_sha256": EXPECTED_CREDENTIAL_ID_SHA256,
            "preserve_append_only_truth_on_rollback": True,
        }

    def immutable_release() -> Mapping[str, Any]:
        _require_fresh_targets_absent(
            (final_release,),
            phase="seal_and_publish_immutable_release",
        )
        if release != staging_release or not staging_release.is_dir():
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_release_path_invalid"
            )
        return {
            "schema": "muncho-owner-gate-release-phase-intent.v1",
            "staging_path": str(staging_release),
            "final_path": str(final_release),
            "created_by_transaction": True,
            "preserve_on_rollback": True,
        }

    def signed_receipt() -> Mapping[str, Any]:
        _require_fresh_targets_absent(
            (install_receipt,),
            phase="emit_signed_inert_install_receipt",
        )
        return {
            "schema": "muncho-owner-gate-install-receipt-phase-intent.v1",
            "path": str(install_receipt),
            "created_by_transaction": True,
            "preserve_on_rollback": True,
            "transaction_prefix_sha256": transaction_context[
                "next_prior_head_sha256"
            ],
        }

    return {
        "reverify_bundle_and_runtime": runtime,
        "install_fixed_identities_and_directories": identities,
        "generate_or_verify_authority_receipt_key": receipt_key,
        "install_root_owned_configuration_units_firewall_and_hosts": (
            system_files
        ),
        "bootstrap_and_verify_canonical_databases": databases,
        "seal_and_publish_immutable_release": immutable_release,
        "emit_signed_inert_install_receipt": signed_receipt,
    }


def _production_fresh_target_guard(
    *,
    bundle: VerifiedBundle,
    layout: InstallLayout,
    transaction_context: Mapping[str, Any],
) -> Callable[[frozenset[str]], None]:
    """Reject every unexplained managed target before the first side effect."""

    key_targets = (
        layout.etc_root / "keys/receipt-signing-key.pem",
        layout.etc_root / "public/authority-receipt-public.pem",
    )
    authority_database = layout.state_root / "authority/passkey-v2.sqlite3"
    executor_database = layout.state_root / "executor/execution-v2.sqlite3"
    final_release = layout.release_base / bundle.revision
    install_receipt = (
        layout.state_root
        / "bootstrap-receipts"
        / f"install-{bundle.revision}.json"
    )

    def guard(authorized_phases: frozenset[str]) -> None:
        transaction_id = str(transaction_context["transaction_id"])
        authority_stage = authority_database.with_name(
            f".{authority_database.name}.{transaction_id}.stage"
        )
        executor_stage = executor_database.with_name(
            f".{executor_database.name}.{transaction_id}.stage"
        )
        targets_by_phase = {
            INSTALL_PHASES[2]: key_targets,
            INSTALL_PHASES[3]: _configuration_target_paths(layout),
            INSTALL_PHASES[4]: (
                authority_database,
                executor_database,
                authority_stage,
                executor_stage,
                *_sqlite_sidecars(authority_database),
                *_sqlite_sidecars(executor_database),
                *_sqlite_sidecars(authority_stage),
                *_sqlite_sidecars(executor_stage),
            ),
            INSTALL_PHASES[5]: (final_release,),
            INSTALL_PHASES[6]: (install_receipt,),
        }
        for phase, targets in targets_by_phase.items():
            if phase not in authorized_phases:
                _require_fresh_targets_absent(targets, phase=phase)

        database_intent = transaction_context.get("phase_intents", {}).get(
            INSTALL_PHASES[4]
        )
        allowed_stages: set[Path] = set()
        if isinstance(database_intent, Mapping):
            allowed_stages = {
                Path(str(item.get("stage_path")))
                for item in database_intent.get("targets", [])
                if isinstance(item, Mapping)
            }
        observed_stages = {
            *authority_database.parent.glob(
                f".{authority_database.name}.*.stage"
            ),
            *executor_database.parent.glob(
                f".{executor_database.name}.*.stage"
            ),
        }
        if observed_stages - allowed_stages:
            raise OwnerGateBootstrapError(
                "owner_gate_bootstrap_fresh_target_conflict:database_stage"
            )

    return guard


def install_after_stage0(
    bundle_root: Path,
    release: Path,
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    runner: Callable[[Sequence[str]], bytes] = _default_runner,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Complete/replay the inert target transaction after stage-zero venv setup."""

    if layout != PRODUCTION_LAYOUT or os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise OwnerGateBootstrapError("owner_gate_bootstrap_root_required")
    bundle = verify_bundle(bundle_root, expected_uid=0)
    expected_paths = {
        layout.release_base / f".{bundle.revision}.bootstrap",
        layout.release_base / bundle.revision,
    }
    if release not in expected_paths:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_release_path_invalid")
    timestamp = int(time.time()) if now_unix is None else now_unix
    if timestamp <= 0:
        raise OwnerGateBootstrapError("owner_gate_bootstrap_time_invalid")
    _ensure_bootstrap_state_roots(layout)
    journal_path = (
        layout.state_root
        / "bootstrap-receipts"
        / f"transaction-{bundle.revision}.json"
    )
    transaction_context: dict[str, Any] = {}
    handlers: dict[str, Callable[[], Mapping[str, Any]]] = {
        "reverify_bundle_and_runtime": lambda: validate_target_runtime(
            bundle,
            layout=layout,
            runner=runner,
        ),
        "install_fixed_identities_and_directories": lambda: (
            install_identities_and_directories(release, runner=runner)
        ),
        "generate_or_verify_authority_receipt_key": lambda: (
            generate_or_verify_receipt_key(
                layout=layout,
                transaction_intent=transaction_context[
                    "active_phase_intent"
                ],
            )
        ),
        "install_root_owned_configuration_units_firewall_and_hosts": lambda: (
            install_configuration_units_firewall_and_hosts(
                bundle,
                release,
                transaction_context["phase_evidence"][
                    "generate_or_verify_authority_receipt_key"
                ],
                layout=layout,
                runner=runner,
                transaction_intent=transaction_context[
                    "active_phase_intent"
                ],
            )
        ),
        "bootstrap_and_verify_canonical_databases": lambda: (
            bootstrap_and_verify_databases(
                bundle,
                layout=layout,
                key_receipt=transaction_context["phase_evidence"][
                    "generate_or_verify_authority_receipt_key"
                ],
                now_unix=int(transaction_context["started_at_unix"]),
                transaction_intent=transaction_context[
                    "active_phase_intent"
                ],
            )
        ),
        "seal_and_publish_immutable_release": lambda: seal_and_publish_release(
            release,
            bundle,
            layout=layout,
            transaction_intent=transaction_context[
                "active_phase_intent"
            ],
        ),
        "emit_signed_inert_install_receipt": lambda: emit_signed_install_receipt(
                bundle,
                transaction_prefix_sha256=transaction_context[
                    "active_prior_head_sha256"
                ],
                phase_evidence=transaction_context["phase_evidence"],
                layout=layout,
                now_unix=int(transaction_context["started_at_unix"]),
                transaction_intent=transaction_context[
                    "active_phase_intent"
                ],
        ),
    }
    revalidators = {
        phase: (
            lambda stored, committed_phase=phase: _revalidate_committed_phase(
                committed_phase,
                stored,
                bundle=bundle,
                layout=layout,
                runner=runner,
                all_evidence=transaction_context["phase_evidence"],
            )
        )
        for phase in INSTALL_PHASES
    }
    transaction = run_install_transaction(
        bundle=bundle,
        journal_path=journal_path,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=0,
        started_at_unix=timestamp,
        transaction_context=transaction_context,
        intent_builders=_production_phase_intent_builders(
            bundle=bundle,
            release=release,
            layout=layout,
            transaction_context=transaction_context,
        ),
        fresh_target_guard=_production_fresh_target_guard(
            bundle=bundle,
            layout=layout,
            transaction_context=transaction_context,
        ),
    )
    reference = transaction["completed_phases"][-1]["evidence"]
    receipt_path = Path(str(reference["receipt_path"]))
    receipt_raw = _read_regular(
        receipt_path,
        maximum=MAX_JSON_BYTES,
        expected_uid=0,
        allowed_modes=frozenset({0o400}),
    )
    if hashlib.sha256(receipt_raw).hexdigest() != reference[
        "signed_receipt_file_sha256"
    ]:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_receipt_readback_invalid"
        )
    return _canonical_json(receipt_raw)


def _remove_exact_installed_file(
    evidence: Mapping[str, Any],
    *,
    allowed_roots: Sequence[Path],
) -> bool:
    path = Path(str(evidence.get("path", "")))
    if not path.is_absolute() or not any(
        path.parent == root for root in allowed_roots
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_rollback_file_invalid"
        )
    if not path.exists():
        return False
    try:
        mode = int(str(evidence["mode"]), 8)
        uid = int(evidence["uid"])
        gid = int(evidence["gid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_rollback_file_invalid"
        ) from None
    raw = _read_regular(
        path,
        maximum=MAX_JSON_BYTES,
        expected_uid=uid,
        allowed_modes=frozenset({mode}),
    )
    state = path.lstat()
    if (
        state.st_gid != gid
        or hashlib.sha256(raw).hexdigest() != evidence.get("sha256")
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_rollback_file_changed"
        )
    path.unlink()
    _fsync_directory(path.parent)
    return True


def _rollback_executor_hosts_file(
    system: Mapping[str, Any],
    *,
    layout: InstallLayout,
) -> Mapping[str, Any]:
    path = layout.etc_root / EXECUTOR_HOSTS_FILENAME
    receipt = _validate_executor_hosts_receipt(
        system.get("executor_hosts"),
        path=path,
    )
    file_evidence = next(
        (
            item
            for item in system.get("files", [])
            if isinstance(item, Mapping) and item.get("path") == str(path)
        ),
        None,
    )
    if (
        not isinstance(file_evidence, Mapping)
        or file_evidence.get("created") is not True
        or file_evidence.get("sha256") != receipt["content_sha256"]
        or file_evidence.get("mode") != "0444"
        or file_evidence.get("uid") != 0
        or file_evidence.get("gid") != 0
    ):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_rollback_executor_hosts_invalid"
        )
    removed = _remove_exact_installed_file(
        file_evidence,
        allowed_roots=(layout.etc_root,),
    )
    if not removed or path.exists() or path.is_symlink():
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_rollback_executor_hosts_invalid"
        )
    unsigned = {
        "schema": "muncho-owner-gate-executor-hosts-rollback.v1",
        "path": str(path),
        "removed": removed,
        "global_etc_hosts_mutated": False,
    }
    return {**unsigned, "receipt_sha256": foundation.sha256_json(unsigned)}


def _rollback_preservation_projection(
    phase_evidence: Mapping[str, Mapping[str, Any]],
    *,
    bundle: VerifiedBundle,
    layout: InstallLayout,
) -> Mapping[str, bool]:
    database = phase_evidence.get(
        "bootstrap_and_verify_canonical_databases"
    )
    release = phase_evidence.get("seal_and_publish_immutable_release")
    receipt = phase_evidence.get("emit_signed_inert_install_receipt")
    authority_path = layout.state_root / "authority/passkey-v2.sqlite3"
    executor_path = layout.state_root / "executor/execution-v2.sqlite3"
    expected_release = layout.release_base / bundle.revision
    expected_receipt = (
        layout.state_root
        / "bootstrap-receipts"
        / f"install-{bundle.revision}.json"
    )
    return {
        "authority_database_preserved": (
            isinstance(database, Mapping) and authority_path.is_file()
        ),
        "executor_database_preserved": (
            isinstance(database, Mapping) and executor_path.is_file()
        ),
        "install_receipt_preserved": (
            isinstance(receipt, Mapping)
            and receipt.get("receipt_path") == str(expected_receipt)
            and expected_receipt.is_file()
        ),
        "immutable_release_preserved": (
            isinstance(release, Mapping)
            and release.get("release_path") == str(expected_release)
            and expected_release.is_dir()
        ),
    }


def rollback_inert_install(
    bundle_root: Path,
    *,
    layout: InstallLayout = PRODUCTION_LAYOUT,
    runner: Callable[[Sequence[str]], bytes] = _default_runner,
    _checkpoint: Callable[[str], None] | None = None,
    _verified_bundle: VerifiedBundle | None = None,
    _expected_uid: int = 0,
    _expected_gid: int = 0,
    _require_root: bool = True,
) -> Mapping[str, Any]:
    """Replay-safe rollback of transaction-owned reversible system files."""

    if _require_root and (layout != PRODUCTION_LAYOUT or os.geteuid() != 0):  # windows-footgun: ok — Debian root boundary
        raise OwnerGateBootstrapError("owner_gate_bootstrap_root_required")
    bundle = (
        verify_bundle(bundle_root, expected_uid=_expected_uid)
        if _verified_bundle is None
        else _verified_bundle
    )
    if foundation.MUTATION_ENABLE_SEAL.exists():
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_rollback_activation_present"
        )
    journal_path = (
        layout.state_root
        / "bootstrap-receipts"
        / f"transaction-{bundle.revision}.json"
    )
    if os.path.lexists(journal_path):
        raise OwnerGateBootstrapError(
            "owner_gate_bootstrap_transaction_invalid"
        )
    journal = bootstrap_journal.BootstrapInstallJournal(
        journal_path,
        owner_uid=_expected_uid,
        owner_gid=_expected_gid,
    )
    rollback_path = (
        layout.state_root
        / "bootstrap-receipts"
        / f"rollback-{bundle.revision}.json"
    )
    try:
        with journal.transaction_lease(create=False):
            transaction, pending = _load_transaction_from_journal(
                journal,
                bundle=bundle,
            )
            if pending is not None or not transaction["complete"]:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_install_incomplete"
                )
            if os.path.lexists(layout.current_link):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_selection_present"
                )
            evidence = {
                str(item["phase"]): dict(item["evidence"])
                for item in transaction["completed_phases"]
            }
            system = evidence.get(
                "install_root_owned_configuration_units_firewall_and_hosts"
            )
            if not isinstance(system, Mapping):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_install_incomplete"
                )
            phase_intent = journal.read("p3-intent")
            if (
                not isinstance(phase_intent, Mapping)
                or not isinstance(phase_intent.get("intent"), Mapping)
                or phase_intent["intent"].get("schema")
                != "muncho-owner-gate-system-files-phase-intent.v1"
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_intent_invalid"
                )
            reversible = {
                str(item.get("path"))
                for item in phase_intent["intent"].get("targets", [])
                if isinstance(item, Mapping)
                and item.get("created_by_transaction") is True
                and item.get("reversible") is True
            }
            files = {
                str(item.get("path")): dict(item)
                for item in system.get("files", [])
                if isinstance(item, Mapping)
                and item.get("created_by_transaction") is True
            }
            if reversible != set(files) or not reversible:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_intent_invalid"
                )
            expected_intent_unsigned = {
                "schema": "muncho-owner-gate-inert-rollback-intent.v2",
                "transaction_id": transaction["journal_transaction_id"],
                "install_terminal_sha256": transaction[
                    "transaction_sha256"
                ],
                "release_revision": bundle.revision,
                "package_sha256": bundle.manifest["package_sha256"],
                "targets": [files[path] for path in sorted(files)],
                "preserve_authority_key": True,
                "preserve_databases": True,
                "preserve_append_only_receipts": True,
                "preserve_immutable_release": True,
                "activation_seal_present": False,
                "cloud_mutation_performed": False,
            }
            expected_intent = {
                **expected_intent_unsigned,
                "rollback_intent_sha256": foundation.sha256_json(
                    expected_intent_unsigned
                ),
            }
            rollback_intent = journal.read("rollback-intent")
            if rollback_intent is None:
                for item in transaction["completed_phases"]:
                    phase = str(item["phase"])
                    _revalidate_committed_phase(
                        phase,
                        item["evidence"],
                        bundle=bundle,
                        layout=layout,
                        runner=runner,
                        all_evidence=evidence,
                    )
                rollback_intent = journal.publish(
                    "rollback-intent",
                    expected_intent,
                )
                if _checkpoint is not None:
                    _checkpoint("rollback_intent_published")
            elif rollback_intent != expected_intent:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_intent_invalid"
                )

            rollback_success = journal.read("rollback-success")
            if rollback_success is None:
                allowed_roots = (
                    layout.etc_root,
                    layout.etc_root / "public",
                    layout.systemd_root,
                    layout.sudoers_root,
                    layout.sysusers_root,
                    layout.tmpfiles_root,
                )
                for index, item in enumerate(rollback_intent["targets"]):
                    path = Path(str(item["path"]))
                    if os.path.lexists(path):
                        _remove_exact_installed_file(
                            item,
                            allowed_roots=allowed_roots,
                        )
                    if os.path.lexists(path):
                        raise OwnerGateBootstrapError(
                            "owner_gate_bootstrap_rollback_file_changed"
                        )
                    if _checkpoint is not None:
                        _checkpoint(f"rollback_target_{index}_absent")
                preservation = _rollback_preservation_projection(
                    evidence,
                    bundle=bundle,
                    layout=layout,
                )
                if not all(preservation.values()):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_rollback_preservation_invalid"
                    )
                key_raw = _read_regular(
                    layout.etc_root / "keys/receipt-signing-key.pem",
                    maximum=4096,
                    expected_uid=_expected_uid,
                    allowed_modes=frozenset({0o400}),
                )
                try:
                    private_key = serialization.load_pem_private_key(
                        key_raw,
                        password=None,
                    )
                except (TypeError, ValueError) as exc:
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_receipt_key_invalid"
                    ) from None
                if not isinstance(private_key, Ed25519PrivateKey):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_receipt_key_invalid"
                    )
                hosts_path = str(layout.etc_root / EXECUTOR_HOSTS_FILENAME)
                unsigned = {
                    "schema": "muncho-owner-gate-inert-install-rollback.v2",
                    "release_revision": bundle.revision,
                    "package_sha256": bundle.manifest["package_sha256"],
                    "transaction_sha256": transaction[
                        "transaction_sha256"
                    ],
                    "rollback_intent_sha256": rollback_intent[
                        "rollback_intent_sha256"
                    ],
                    "current_release_selection_removed": False,
                    "removed_entry_files": sorted(files),
                    "executor_hosts_rollback": {
                        "schema": (
                            "muncho-owner-gate-executor-hosts-rollback.v2"
                        ),
                        "path": hosts_path,
                        "removed_by_transaction": hosts_path in files,
                        "global_etc_hosts_mutated": False,
                    },
                    **preservation,
                    "activation_seal_present": False,
                    "cloud_mutation_performed": False,
                    "caddy_cutover_performed": False,
                }
                signed = {
                    **unsigned,
                    "receipt_sha256": foundation.sha256_json(unsigned),
                }
                receipt = {
                    **signed,
                    "signer_key_id": hashlib.sha256(
                        private_key.public_key().public_bytes_raw()
                    ).hexdigest(),
                    "signature_ed25519_b64url": base64.urlsafe_b64encode(
                        private_key.sign(
                            foundation.canonical_json_bytes(signed)
                        )
                    )
                    .rstrip(b"=")
                    .decode("ascii"),
                }
                receipt_raw = foundation.canonical_json_bytes(receipt)
                _install_exact_bytes(
                    rollback_path,
                    receipt_raw,
                    mode=0o400,
                    uid=_expected_uid,
                    gid=_expected_gid,
                    _checkpoint=(
                        None
                        if _checkpoint is None
                        else lambda label: _checkpoint(
                            f"rollback_receipt_{label}"
                        )
                    ),
                )
                if _checkpoint is not None:
                    _checkpoint("rollback_receipt_published")
                success_evidence = {
                    "rollback_receipt_path": str(rollback_path),
                    "rollback_receipt_file_sha256": hashlib.sha256(
                        receipt_raw
                    ).hexdigest(),
                    "rollback_receipt_sha256": receipt["receipt_sha256"],
                    "removed_entry_files": sorted(files),
                    **preservation,
                }
                success_unsigned = {
                    "schema": "muncho-owner-gate-inert-rollback-success.v2",
                    "install_terminal_sha256": transaction[
                        "transaction_sha256"
                    ],
                    "rollback_intent_sha256": rollback_intent[
                        "rollback_intent_sha256"
                    ],
                    "evidence": success_evidence,
                    "evidence_sha256": foundation.sha256_json(
                        success_evidence
                    ),
                }
                rollback_success = journal.publish(
                    "rollback-success",
                    {
                        **success_unsigned,
                        "rollback_success_sha256": foundation.sha256_json(
                            success_unsigned
                        ),
                    },
                )
                if _checkpoint is not None:
                    _checkpoint("rollback_success_published")
            else:
                if (
                    rollback_success.get("schema")
                    != "muncho-owner-gate-inert-rollback-success.v2"
                    or rollback_success.get("install_terminal_sha256")
                    != transaction["transaction_sha256"]
                    or rollback_success.get("rollback_intent_sha256")
                    != rollback_intent["rollback_intent_sha256"]
                    or not isinstance(
                        rollback_success.get("evidence"), Mapping
                    )
                    or rollback_success["evidence"].get(
                        "rollback_receipt_path"
                    )
                    != str(rollback_path)
                    or rollback_success["evidence"].get(
                        "removed_entry_files"
                    )
                    != sorted(files)
                    or any(
                        rollback_success["evidence"].get(name) is not True
                        for name in (
                            "authority_database_preserved",
                            "executor_database_preserved",
                            "install_receipt_preserved",
                            "immutable_release_preserved",
                        )
                    )
                ):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_rollback_success_invalid"
                    )
                success_unsigned = {
                    key: value
                    for key, value in rollback_success.items()
                    if key != "rollback_success_sha256"
                }
                if (
                    rollback_success.get("evidence_sha256")
                    != foundation.sha256_json(
                        rollback_success["evidence"]
                    )
                    or rollback_success.get("rollback_success_sha256")
                    != foundation.sha256_json(success_unsigned)
                ):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_rollback_success_invalid"
                    )

            receipt_raw = _read_regular(
                rollback_path,
                maximum=MAX_JSON_BYTES,
                expected_uid=_expected_uid,
                allowed_modes=frozenset({0o400}),
            )
            receipt = _canonical_json(receipt_raw)
            preservation = _rollback_preservation_projection(
                evidence,
                bundle=bundle,
                layout=layout,
            )
            signature_text = receipt.get("signature_ed25519_b64url")
            signature = _b64url(signature_text, maximum=64)
            signed_receipt = {
                key: value
                for key, value in receipt.items()
                if key
                not in {"signer_key_id", "signature_ed25519_b64url"}
            }
            unsigned_receipt = {
                key: value
                for key, value in signed_receipt.items()
                if key != "receipt_sha256"
            }
            key_evidence = evidence.get(
                "generate_or_verify_authority_receipt_key"
            )
            if not isinstance(key_evidence, Mapping):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_success_invalid"
                )
            public_raw = _read_regular(
                Path(str(key_evidence["public_key_path"])),
                maximum=4096,
                expected_uid=_expected_uid,
                allowed_modes=frozenset({0o444}),
            )
            try:
                public_key = serialization.load_pem_public_key(public_raw)
                if not isinstance(public_key, Ed25519PublicKey):
                    raise TypeError
                public_key.verify(
                    signature,
                    foundation.canonical_json_bytes(signed_receipt),
                )
            except (InvalidSignature, TypeError, ValueError) as exc:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_success_invalid"
                ) from None
            if (
                hashlib.sha256(receipt_raw).hexdigest()
                != rollback_success["evidence"][
                    "rollback_receipt_file_sha256"
                ]
                or receipt.get("receipt_sha256")
                != rollback_success["evidence"][
                    "rollback_receipt_sha256"
                ]
                or receipt.get("receipt_sha256")
                != foundation.sha256_json(unsigned_receipt)
                or receipt.get("signer_key_id")
                != key_evidence.get("public_key_id")
                or receipt.get("transaction_sha256")
                != transaction["transaction_sha256"]
                or receipt.get("rollback_intent_sha256")
                != rollback_intent["rollback_intent_sha256"]
                or receipt.get("removed_entry_files") != sorted(files)
                or receipt.get("current_release_selection_removed")
                is not False
                or receipt.get("activation_seal_present") is not False
                or receipt.get("cloud_mutation_performed") is not False
                or receipt.get("caddy_cutover_performed") is not False
                or not all(preservation.values())
                or any(
                    receipt.get(name) != value
                    for name, value in preservation.items()
                )
            ):
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_success_invalid"
                )
            for item in rollback_intent["targets"]:
                if os.path.lexists(Path(str(item["path"]))):
                    raise OwnerGateBootstrapError(
                        "owner_gate_bootstrap_rollback_replay_drift"
                    )
            terminal_unsigned = {
                "schema": "muncho-owner-gate-inert-rollback-terminal.v2",
                "install_terminal_sha256": transaction[
                    "transaction_sha256"
                ],
                "rollback_intent_sha256": rollback_intent[
                    "rollback_intent_sha256"
                ],
                "rollback_success_sha256": rollback_success[
                    "rollback_success_sha256"
                ],
                "rollback_receipt_file_sha256": hashlib.sha256(
                    receipt_raw
                ).hexdigest(),
                "activation_seal_present": False,
                "cloud_mutation_performed": False,
            }
            expected_terminal = {
                **terminal_unsigned,
                "rollback_terminal_sha256": foundation.sha256_json(
                    terminal_unsigned
                ),
            }
            terminal = journal.read("rollback-terminal")
            if terminal is None:
                journal.publish("rollback-terminal", expected_terminal)
                if _checkpoint is not None:
                    _checkpoint("rollback_terminal_published")
            elif terminal != expected_terminal:
                raise OwnerGateBootstrapError(
                    "owner_gate_bootstrap_rollback_terminal_invalid"
                )
            return receipt
    except bootstrap_journal.BootstrapJournalError as exc:
        raise OwnerGateBootstrapError(str(exc)) from None


def runtime_install_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("install-after-stage0", "rollback-inert"),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    arguments = parser.parse_args(argv)
    release = Path(__file__).resolve(strict=True).parents[2]
    receipt = (
        install_after_stage0(arguments.bundle, release)
        if arguments.operation == "install-after-stage0"
        else rollback_inert_install(arguments.bundle)
    )
    print(foundation.canonical_json_bytes(receipt).decode("ascii"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("preflight", "plan"):
        child = subparsers.add_parser(name)
        child.add_argument("--bundle", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise OwnerGateBootstrapError("owner_gate_bootstrap_root_required")
    bundle = verify_bundle(arguments.bundle, expected_uid=0)
    value = (
        validate_target_runtime(bundle)
        if arguments.operation == "preflight"
        else build_install_plan(bundle)
    )
    print(foundation.canonical_json_bytes(value).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
