#!/usr/bin/env python3
"""Sealed owner orchestration for one inert owner-gate observation pair.

The production entrypoint accepts no paths, evidence, keys, journal roots, or
remote commands.  Package preparation is a separate operation which places
two exact mode-0400 tree streams and their canonical digest pins below the
fixed release-derived owner-only root.  This module only validates and
consumes those inputs, loads the already-successful Foundation chain, runs one
HOST/CLOUD composite, and returns its immediately validated inert preflight.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import re
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_author_and_apply as foundation_author
from scripts.canary import owner_gate_author_journal as author_journal
from scripts.canary import owner_gate_cloud_observation_author as cloud_author
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_stage0 as cloud_stage0
from scripts.canary import owner_gate_stage0_iap as stage0_iap
from scripts.canary import owner_gate_trust_author as trust_author
from scripts.canary import trusted_signer_author as signer_author


INPUT_PINS_SCHEMA = "muncho-owner-gate-inert-observation-input-pins.v1"
RECEIPT_SCHEMA = "muncho-owner-gate-inert-observation-receipt.v1"
EVIDENCE_SET_ID_SCHEMA = "muncho-owner-gate-inert-observation-set-id.v1"
OWNER_HOME = Path(pwd.getpwuid(os.geteuid()).pw_dir)  # windows-footgun: ok
INPUT_ROOT = OWNER_HOME / ".hermes" / "owner-gate-inert-observation-inputs"
EVIDENCE_ROOT = OWNER_HOME / ".hermes" / "owner-gate-inert-observation-evidence"
EVIDENCE_PHASE = "inert"
PINS_NAME = "stream-pins.json"
KIT_STREAM_NAME = "outer-stage0.tree-stream"
BUNDLE_STREAM_NAME = "owner-gate-bundle.tree-stream"
INERT_CLOUD_OBSERVATION_NAME = "inert-cloud-observation.json"
INERT_HOST_OBSERVATION_NAME = "inert-host-observation.json"
INERT_PREFLIGHT_NAME = "inert-preflight.json"
RECEIPT_NAME = "receipt.json"
PENDING_NAME = ".pending"
MAX_PINS_BYTES = 16 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_LOCK = threading.Lock()

_EVIDENCE_NAMES = (
    INERT_CLOUD_OBSERVATION_NAME,
    INERT_HOST_OBSERVATION_NAME,
    INERT_PREFLIGHT_NAME,
)
_TRANSACTION_NAMES = frozenset((*_EVIDENCE_NAMES, RECEIPT_NAME))
_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "phase",
    "release_revision",
    "source_tree_oid",
    "package_sha256",
    "plan_sha256",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "input_pins_sha256",
    "observed_at_unix",
    "evidence_files",
    "cloud_observation_sha256",
    "host_observation_sha256",
    "preflight_report_sha256",
    "evidence_set_sha256",
    "caller_authored_evidence_accepted",
    "cloud_mutation_performed",
    "service_activation_performed",
    "receipt_sha256",
})


def _error(code: str, _cause: BaseException | None = None) -> None:
    raise launcher.OwnerLauncherError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_inert_observation_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode_document(raw: bytes, *, code: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _error(code)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
    except launcher.OwnerLauncherError:
        raise
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error(code, exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error(code)
    return dict(value)


def _fsync_directory(path: Path, *, code: str) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        _error(code, exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error(code, exc)


def _ensure_private_directory(path: Path, *, parent: Path) -> None:
    if not path.is_absolute() or ".." in path.parts or path.parent != parent:
        _error("owner_gate_inert_observation_evidence_root_invalid")
    if not os.path.lexists(path):
        try:
            os.mkdir(path, 0o700)
            _fsync_directory(
                parent,
                code="owner_gate_inert_observation_evidence_root_invalid",
            )
        except FileExistsError:
            pass
        except OSError as exc:
            _error("owner_gate_inert_observation_evidence_root_invalid", exc)
    _require_directory(
        path,
        parent=parent,
        code="owner_gate_inert_observation_evidence_root_invalid",
    )


def _evidence_phase_root(release_revision: str) -> Path:
    if _REVISION.fullmatch(release_revision or "") is None:
        _error("owner_gate_inert_observation_release_invalid")
    hermes = OWNER_HOME / ".hermes"
    _require_directory(
        hermes,
        parent=OWNER_HOME,
        code="owner_gate_inert_observation_evidence_root_invalid",
    )
    _ensure_private_directory(EVIDENCE_ROOT, parent=hermes)
    release_root = EVIDENCE_ROOT / release_revision
    _ensure_private_directory(release_root, parent=EVIDENCE_ROOT)
    phase_root = release_root / EVIDENCE_PHASE
    _ensure_private_directory(phase_root, parent=release_root)
    return phase_root


@contextmanager
def _evidence_lease(release_revision: str) -> Iterator[Path]:
    descriptor: int | None = None
    acquired = False
    _PROCESS_LOCK.acquire()
    try:
        phase_root = _evidence_phase_root(release_revision)
        descriptor = os.open(
            phase_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok
            or opened.st_gid != os.getegid()  # windows-footgun: ok
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            _error("owner_gate_inert_observation_evidence_root_invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        phase_identity = _require_directory(
            phase_root,
            parent=phase_root.parent,
            code="owner_gate_inert_observation_evidence_root_invalid",
        )
        if (opened.st_dev, opened.st_ino) != (
            phase_identity[3],
            phase_identity[4],
        ):
            _error("owner_gate_inert_observation_evidence_root_invalid")
        yield phase_root
        final_identity = _require_directory(
            phase_root,
            parent=phase_root.parent,
            code="owner_gate_inert_observation_manual_reconciliation_required",
        )
        if (opened.st_dev, opened.st_ino) != (
            final_identity[3],
            final_identity[4],
        ):
            _error("owner_gate_inert_observation_manual_reconciliation_required")
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error("owner_gate_inert_observation_evidence_lock_failed", exc)
    finally:
        failed = False
        if descriptor is not None:
            if acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    failed = True
            try:
                os.close(descriptor)
            except OSError:
                failed = True
        _PROCESS_LOCK.release()
        if failed:
            _error("owner_gate_inert_observation_evidence_lock_failed")


def _require_directory(
    path: Path,
    *,
    parent: Path,
    code: str,
    mode: int = 0o700,
) -> tuple[Any, ...]:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        opened = resolved.stat()
    except OSError as exc:
        _error(code, exc)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.parent != parent
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or resolved != path
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        or opened.st_uid != os.geteuid()  # windows-footgun: ok
        or opened.st_gid != os.getegid()  # windows-footgun: ok
        or stat.S_IMODE(opened.st_mode) != mode
    ):
        _error(code)
    return (
        opened.st_mode,
        opened.st_uid,
        opened.st_gid,
        opened.st_dev,
        opened.st_ino,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )


def _read_pinned_file(
    path: Path,
    *,
    maximum: int,
    code: str,
) -> tuple[tuple[Any, ...], bytes]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok
            or opened.st_gid != os.getegid()  # windows-footgun: ok
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o400
            or not 0 < opened.st_size <= maximum
        ):
            _error(code)
        raw = bytearray()
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                _error(code)
            raw.extend(chunk)
        after = os.fstat(descriptor)
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _error(code)
        return identity, bytes(raw)
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error(code, exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _pinned_regular_identity(
    path: Path,
    *,
    maximum: int,
    expected_sha256: str,
    code: str,
) -> tuple[Any, ...]:
    descriptor: int | None = None
    try:
        if _SHA256.fullmatch(expected_sha256 or "") is None:
            _error(code)
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (
            not path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok
            or opened.st_gid != os.getegid()  # windows-footgun: ok
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o400
            or not 0 < opened.st_size <= maximum
            or identity
            != (
                before.st_mode,
                before.st_uid,
                before.st_gid,
                before.st_dev,
                before.st_ino,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        ):
            _error(code)
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _error(code)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) or digest.hexdigest() != expected_sha256:
            _error(code)
        after = os.fstat(descriptor)
        if identity != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _error(code)
        return identity
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error(code, exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _decode_pins(raw: bytes, *, release_revision: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _error("owner_gate_inert_observation_pins_invalid")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
    except launcher.OwnerLauncherError:
        raise
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_inert_observation_pins_invalid", exc)
    fields = {
        "schema",
        "release_revision",
        "kit_release_id",
        "kit_tree_manifest_sha256",
        "kit_stream_sha256",
        "bundle_tree_manifest_sha256",
        "bundle_stream_sha256",
        "pins_sha256",
    }
    unsigned = {key: item for key, item in value.items() if key != "pins_sha256"}
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != INPUT_PINS_SCHEMA
        or value.get("release_revision") != release_revision
        or _SHA256.fullmatch(str(value.get("kit_release_id", ""))) is None
        or _SHA256.fullmatch(
            str(value.get("kit_tree_manifest_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(value.get("bundle_tree_manifest_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(str(value.get("kit_stream_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("bundle_stream_sha256", ""))) is None
        or value.get("pins_sha256") != foundation.sha256_json(unsigned)
        or _canonical(value) != raw
    ):
        _error("owner_gate_inert_observation_pins_invalid")
    return dict(value)


@dataclass(frozen=True)
class _PinnedObservationInputs:
    release_root: Path
    root_identity: tuple[Any, ...]
    release_identity: tuple[Any, ...]
    pins_identity: tuple[Any, ...]
    kit_identity: tuple[Any, ...]
    bundle_identity: tuple[Any, ...]
    pins_raw: bytes
    pins: Mapping[str, Any]
    kit_stream: stage0_iap.PinnedExactTreeStream
    bundle_stream: stage0_iap.PinnedExactTreeStream

    @classmethod
    def load(cls, release_revision: str) -> "_PinnedObservationInputs":
        if _REVISION.fullmatch(release_revision or "") is None:
            _error("owner_gate_inert_observation_release_invalid")
        hermes = OWNER_HOME / ".hermes"
        _require_directory(
            hermes,
            parent=OWNER_HOME,
            code="owner_gate_inert_observation_input_root_invalid",
        )
        root_identity = _require_directory(
            INPUT_ROOT,
            parent=hermes,
            code="owner_gate_inert_observation_input_root_invalid",
        )
        release_root = INPUT_ROOT / release_revision
        release_identity = _require_directory(
            release_root,
            parent=INPUT_ROOT,
            code="owner_gate_inert_observation_input_release_invalid",
        )
        try:
            inventory = set(os.listdir(release_root))
        except OSError as exc:
            _error("owner_gate_inert_observation_input_release_invalid", exc)
        if inventory != {PINS_NAME, KIT_STREAM_NAME, BUNDLE_STREAM_NAME}:
            _error("owner_gate_inert_observation_input_release_invalid")
        pins_identity, pins_raw = _read_pinned_file(
            release_root / PINS_NAME,
            maximum=MAX_PINS_BYTES,
            code="owner_gate_inert_observation_pins_invalid",
        )
        pins = _decode_pins(pins_raw, release_revision=release_revision)
        kit_identity = _pinned_regular_identity(
            release_root / KIT_STREAM_NAME,
            maximum=stage0_iap.MAX_STREAM_BYTES,
            expected_sha256=str(pins["kit_stream_sha256"]),
            code="owner_gate_inert_observation_stream_invalid",
        )
        bundle_identity = _pinned_regular_identity(
            release_root / BUNDLE_STREAM_NAME,
            maximum=stage0_iap.MAX_STREAM_BYTES,
            expected_sha256=str(pins["bundle_stream_sha256"]),
            code="owner_gate_inert_observation_stream_invalid",
        )
        try:
            kit_stream = stage0_iap.PinnedExactTreeStream(
                release_root / KIT_STREAM_NAME,
                purpose="outer-stage0-kit",
                release_id=str(pins["kit_release_id"]),
                expected_manifest_sha256=str(
                    pins["kit_tree_manifest_sha256"]
                ),
            )
            bundle_stream = stage0_iap.PinnedExactTreeStream(
                release_root / BUNDLE_STREAM_NAME,
                purpose="owner-gate-bundle",
                release_id=release_revision,
                expected_manifest_sha256=str(
                    pins["bundle_tree_manifest_sha256"]
                ),
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_inert_observation_stream_invalid", exc)
        value = cls(
            release_root=release_root,
            root_identity=root_identity,
            release_identity=release_identity,
            pins_identity=pins_identity,
            kit_identity=kit_identity,
            bundle_identity=bundle_identity,
            pins_raw=pins_raw,
            pins=pins,
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        value.assert_stable()
        return value

    def assert_stable(self) -> None:
        hermes = OWNER_HOME / ".hermes"
        try:
            inventory = set(os.listdir(self.release_root))
        except OSError as exc:
            _error("owner_gate_inert_observation_input_changed", exc)
        if (
            inventory != {PINS_NAME, KIT_STREAM_NAME, BUNDLE_STREAM_NAME}
            or
            _require_directory(
                INPUT_ROOT,
                parent=hermes,
                code="owner_gate_inert_observation_input_changed",
            )
            != self.root_identity
            or _require_directory(
                self.release_root,
                parent=INPUT_ROOT,
                code="owner_gate_inert_observation_input_changed",
            )
            != self.release_identity
        ):
            _error("owner_gate_inert_observation_input_changed")
        identity, raw = _read_pinned_file(
            self.release_root / PINS_NAME,
            maximum=MAX_PINS_BYTES,
            code="owner_gate_inert_observation_input_changed",
        )
        if identity != self.pins_identity or raw != self.pins_raw:
            _error("owner_gate_inert_observation_input_changed")
        if (
            _pinned_regular_identity(
                self.release_root / KIT_STREAM_NAME,
                maximum=stage0_iap.MAX_STREAM_BYTES,
                expected_sha256=str(self.pins["kit_stream_sha256"]),
                code="owner_gate_inert_observation_input_changed",
            )
            != self.kit_identity
            or _pinned_regular_identity(
                self.release_root / BUNDLE_STREAM_NAME,
                maximum=stage0_iap.MAX_STREAM_BYTES,
                expected_sha256=str(self.pins["bundle_stream_sha256"]),
                code="owner_gate_inert_observation_input_changed",
            )
            != self.bundle_identity
        ):
            _error("owner_gate_inert_observation_input_changed")
        self.kit_stream.assert_stable()
        self.bundle_stream.assert_stable()


@dataclass(frozen=True)
class _LoadedFoundation:
    chain: foundation_apply.ValidatedFoundationApplyChain
    raw_artifacts: stage0_iap.RawFoundationChainArtifacts


def _load_successful_foundation(release_revision: str) -> _LoadedFoundation:
    try:
        release_public = pre_foundation.load_pinned_public_key(
            trust_author.KEY_DIRECTORY / trust_author.PUBLIC_KEY_NAME,
            expected_uid=os.geteuid(),  # windows-footgun: ok
        )
        network_public = foundation_author._load_network_public_key(
            release_revision
        )
        journal = author_journal.OwnerGateAuthorJournal()
        with journal.release_lease(release_revision):
            transactions = journal.list_transactions(release_revision)
            successful: list[tuple[str, Mapping[str, Mapping[str, Any]]]] = []
            for transaction_id, artifacts in transactions.items():
                if "terminal" not in artifacts:
                    _error("owner_gate_inert_observation_foundation_incomplete")
                receipt = foundation_author._validate_terminal(
                    release_revision=release_revision,
                    transaction_id=transaction_id,
                    artifacts=artifacts,
                    release_public=release_public,
                    network_public=network_public,
                )
                if receipt is not None:
                    successful.append((transaction_id, artifacts))
            if len(successful) != 1:
                _error("owner_gate_inert_observation_foundation_invalid")
            transaction_id, artifacts = successful[0]
            foundation_a = foundation_author._historical_foundation_chain(
                artifacts=artifacts,
                release_public=release_public,
                network_public=network_public,
            )
            chain = foundation_apply.load_validated_foundation_apply_chain(
                foundation_a
            )
            transaction_root = journal.root / release_revision / transaction_id
            network_public_key_path = (
                signer_author._public_path(
                    release_revision,
                    "network",
                )
            )
            raw_artifacts = stage0_iap.RawFoundationChainArtifacts(
                pre_foundation_authority_path=transaction_root / "authority.json",
                owner_reauthentication_receipt_path=(
                    transaction_root / "owner-reauth.json"
                ),
                network_evidence_path=(
                    transaction_root / "network-evidence.json"
                ),
                network_collector_public_key_path=(
                    network_public_key_path
                ),
                project_ancestry_evidence_path=(
                    transaction_root / "ancestry-evidence.json"
                ),
                project_ancestry_collector_public_key_path=(
                    network_public_key_path
                ),
                release_public_key_path=(
                    trust_author.KEY_DIRECTORY / trust_author.PUBLIC_KEY_NAME
                ),
            )
    except launcher.OwnerLauncherError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        author_journal.OwnerGateAuthorJournalError,
        foundation_author.OwnerGateAuthorAndApplyError,
        foundation_apply.OwnerGateFoundationApplyError,
        pre_foundation.OwnerGatePreFoundationError,
    ) as exc:
        _error("owner_gate_inert_observation_foundation_invalid", exc)
    if (
        type(chain) is not foundation_apply.ValidatedFoundationApplyChain
        or chain._marker is not foundation_apply._CHAIN_MARKER
        or chain.foundation_source_revision != release_revision
    ):
        _error("owner_gate_inert_observation_foundation_invalid")
    return _LoadedFoundation(chain=chain, raw_artifacts=raw_artifacts)


def _final_plan(
    chain: foundation_apply.ValidatedFoundationApplyChain,
    bundle_stream: stage0_iap.PinnedExactTreeStream,
) -> tuple[foundation.OwnerGateFoundationPlan, Mapping[str, Any]]:
    try:
        package_raw = bundle_stream.member("package-manifest.json")
        package = cloud_author._decode_json(package_raw)
        unsigned = {
            key: item for key, item in package.items() if key != "package_sha256"
        }
        inventory = {
            key: package[key] for key in cloud_stage0.INVENTORY_FIELDS
        }
        collectors = package.get("collector_public_key_ids")
        if (
            set(package) != cloud_stage0.MANIFEST_FIELDS
            or package.get("schema") != cloud_stage0.PACKAGE_SCHEMA
            or package.get("release_revision") != chain.foundation_source_revision
            or package.get("source_tree_oid") != chain.foundation_source_tree_oid
            or package.get("package_sha256") != foundation.sha256_json(unsigned)
            or package.get("package_inventory_sha256")
            != foundation.sha256_json(inventory)
            or not isinstance(collectors, Mapping)
            or set(collectors) != {"network", "cloud", "host"}
            or any(_SHA256.fullmatch(str(item)) is None for item in collectors.values())
        ):
            _error("owner_gate_inert_observation_package_invalid")
        foundation_a = chain.foundation_a
        spec = replace(
            foundation_a.plan.spec,
            source_tree_oid=None,
            boot_image_numeric_id=None,
            package_inventory_sha256=str(package["package_inventory_sha256"]),
            cloud_collector_public_key_id=str(collectors["cloud"]),
            host_collector_public_key_id=str(collectors["host"]),
        )
        plan = foundation.build_plan(
            spec=spec,
            network_evidence=foundation_a.network_evidence,
            network_collector_public_key=(
                foundation_a.network_collector_public_key
            ),
            now_unix=foundation_a.network_evidence.collected_at_unix,
        )
    except launcher.OwnerLauncherError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        foundation.OwnerGateFoundationError,
    ) as exc:
        _error("owner_gate_inert_observation_package_invalid", exc)
    return plan, package


def _collector_key(release_revision: str, *, role: str) -> Ed25519PublicKey:
    snapshot = cloud_author._public_signer_snapshot(
        release_revision,
        role=role,
    )
    try:
        return Ed25519PublicKey.from_public_bytes(snapshot.public_raw)
    except ValueError as exc:
        _error("owner_gate_inert_observation_collector_key_invalid", exc)


def _evidence_set_sha256(
    *,
    release_revision: str,
    plan_sha256: str,
    evidence_files: Mapping[str, str],
) -> str:
    return foundation.sha256_json({
        "schema": EVIDENCE_SET_ID_SCHEMA,
        "release_revision": release_revision,
        "phase": EVIDENCE_PHASE,
        "plan_sha256": plan_sha256,
        "evidence_files": dict(evidence_files),
    })


def _build_receipt(
    *,
    release_revision: str,
    inputs: _PinnedObservationInputs,
    loaded: _LoadedFoundation,
    package: Mapping[str, Any],
    plan: foundation.OwnerGateFoundationPlan,
    cloud_observation: Mapping[str, Any],
    host_observation: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    payloads = {
        INERT_CLOUD_OBSERVATION_NAME: _canonical(cloud_observation),
        INERT_HOST_OBSERVATION_NAME: _canonical(host_observation),
        INERT_PREFLIGHT_NAME: _canonical(report),
    }
    evidence_files = {
        name: _sha256(payloads[name]) for name in _EVIDENCE_NAMES
    }
    evidence_set_sha256 = _evidence_set_sha256(
        release_revision=release_revision,
        plan_sha256=plan.sha256,
        evidence_files=evidence_files,
    )
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "ok": True,
        "phase": EVIDENCE_PHASE,
        "release_revision": release_revision,
        "source_tree_oid": loaded.chain.foundation_source_tree_oid,
        "package_sha256": package["package_sha256"],
        "plan_sha256": plan.sha256,
        "pre_foundation_authority_sha256": (
            loaded.chain.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": (
            loaded.chain.foundation_apply_receipt_sha256
        ),
        "input_pins_sha256": inputs.pins["pins_sha256"],
        "observed_at_unix": report["observed_at_unix"],
        "evidence_files": evidence_files,
        "cloud_observation_sha256": cloud_observation["report_sha256"],
        "host_observation_sha256": host_observation["report_sha256"],
        "preflight_report_sha256": report["report_sha256"],
        "evidence_set_sha256": evidence_set_sha256,
        "caller_authored_evidence_accepted": False,
        "cloud_mutation_performed": False,
        "service_activation_performed": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": foundation.sha256_json(unsigned),
    }
    receipt_raw = _canonical(receipt)
    if len(receipt_raw) > launcher._MAX_JSON_LINE_BYTES:
        _error("owner_gate_inert_observation_receipt_too_large")
    return receipt, {**payloads, RECEIPT_NAME: receipt_raw}


def _validate_receipt(
    *,
    transaction_name: str,
    receipt: Mapping[str, Any],
    receipt_raw: bytes,
    evidence_raw: Mapping[str, bytes],
    release_revision: str,
    inputs: _PinnedObservationInputs,
    loaded: _LoadedFoundation,
    package: Mapping[str, Any],
    plan: foundation.OwnerGateFoundationPlan,
) -> None:
    unsigned = {
        key: item for key, item in receipt.items() if key != "receipt_sha256"
    }
    evidence_files = {
        name: _sha256(evidence_raw[name]) for name in _EVIDENCE_NAMES
    }
    if (
        set(receipt) != _RECEIPT_FIELDS
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("ok") is not True
        or receipt.get("phase") != EVIDENCE_PHASE
        or receipt.get("release_revision") != release_revision
        or receipt.get("source_tree_oid")
        != loaded.chain.foundation_source_tree_oid
        or receipt.get("package_sha256") != package.get("package_sha256")
        or receipt.get("plan_sha256") != plan.sha256
        or receipt.get("pre_foundation_authority_sha256")
        != loaded.chain.pre_foundation_authority_sha256
        or receipt.get("foundation_apply_receipt_sha256")
        != loaded.chain.foundation_apply_receipt_sha256
        or receipt.get("input_pins_sha256") != inputs.pins.get("pins_sha256")
        or type(receipt.get("observed_at_unix")) is not int
        or receipt["observed_at_unix"] <= 0
        or not isinstance(receipt.get("evidence_files"), Mapping)
        or dict(receipt["evidence_files"]) != evidence_files
        or any(_SHA256.fullmatch(item) is None for item in evidence_files.values())
        or _SHA256.fullmatch(str(receipt.get("cloud_observation_sha256", "")))
        is None
        or _SHA256.fullmatch(str(receipt.get("host_observation_sha256", "")))
        is None
        or _SHA256.fullmatch(str(receipt.get("preflight_report_sha256", "")))
        is None
        or receipt.get("caller_authored_evidence_accepted") is not False
        or receipt.get("cloud_mutation_performed") is not False
        or receipt.get("service_activation_performed") is not False
        or receipt.get("receipt_sha256") != foundation.sha256_json(unsigned)
        or _canonical(receipt) != receipt_raw
    ):
        _error("owner_gate_inert_observation_manual_reconciliation_required")
    evidence_set_sha256 = _evidence_set_sha256(
        release_revision=release_revision,
        plan_sha256=plan.sha256,
        evidence_files=evidence_files,
    )
    if (
        transaction_name != evidence_set_sha256
        or receipt.get("evidence_set_sha256") != evidence_set_sha256
    ):
        _error("owner_gate_inert_observation_manual_reconciliation_required")


def _load_evidence_transaction(
    *,
    phase_root: Path,
    transaction_name: str,
    release_revision: str,
    inputs: _PinnedObservationInputs,
    loaded: _LoadedFoundation,
    package: Mapping[str, Any],
    plan: foundation.OwnerGateFoundationPlan,
    cloud_key: Ed25519PublicKey,
    host_key: Ed25519PublicKey,
    now_unix: int,
) -> tuple[Mapping[str, Any], bool]:
    transaction_root = phase_root / transaction_name
    try:
        transaction_identity = _require_directory(
            transaction_root,
            parent=phase_root,
            code="owner_gate_inert_observation_manual_reconciliation_required",
            mode=0o500,
        )
        inventory = set(os.listdir(transaction_root))
        if inventory != _TRANSACTION_NAMES:
            _error("owner_gate_inert_observation_manual_reconciliation_required")
        raw: dict[str, bytes] = {}
        identities: dict[str, tuple[Any, ...]] = {}
        for name in (*_EVIDENCE_NAMES, RECEIPT_NAME):
            identities[name], raw[name] = _read_pinned_file(
                transaction_root / name,
                maximum=(
                    launcher._MAX_JSON_LINE_BYTES
                    if name == RECEIPT_NAME
                    else MAX_EVIDENCE_BYTES
                ),
                code="owner_gate_inert_observation_manual_reconciliation_required",
            )
        cloud_observation = _decode_document(
            raw[INERT_CLOUD_OBSERVATION_NAME],
            code="owner_gate_inert_observation_manual_reconciliation_required",
        )
        host_observation = _decode_document(
            raw[INERT_HOST_OBSERVATION_NAME],
            code="owner_gate_inert_observation_manual_reconciliation_required",
        )
        stored_report = _decode_document(
            raw[INERT_PREFLIGHT_NAME],
            code="owner_gate_inert_observation_manual_reconciliation_required",
        )
        receipt = _decode_document(
            raw[RECEIPT_NAME],
            code="owner_gate_inert_observation_manual_reconciliation_required",
        )
        _validate_receipt(
            transaction_name=transaction_name,
            receipt=receipt,
            receipt_raw=raw[RECEIPT_NAME],
            evidence_raw={name: raw[name] for name in _EVIDENCE_NAMES},
            release_revision=release_revision,
            inputs=inputs,
            loaded=loaded,
            package=package,
            plan=plan,
        )
        observed_at_unix = receipt["observed_at_unix"]
        rebuilt = preflight.build_preflight_report(
            plan=plan,
            cloud_observation=cloud_observation,
            host_observation=host_observation,
            cloud_collector_public_key=cloud_key,
            host_collector_public_key=host_key,
            now_unix=observed_at_unix,
        )
        if (
            rebuilt != stored_report
            or receipt["cloud_observation_sha256"]
            != cloud_observation.get("report_sha256")
            or receipt["host_observation_sha256"]
            != host_observation.get("report_sha256")
            or receipt["preflight_report_sha256"]
            != stored_report.get("report_sha256")
            or stored_report.get("observed_at_unix") != observed_at_unix
            or now_unix < observed_at_unix
        ):
            _error("owner_gate_inert_observation_manual_reconciliation_required")
        try:
            preflight.build_preflight_report(
                plan=plan,
                cloud_observation=cloud_observation,
                host_observation=host_observation,
                cloud_collector_public_key=cloud_key,
                host_collector_public_key=host_key,
                now_unix=now_unix,
            )
        except preflight.OwnerGatePreflightError as exc:
            if str(exc) == "owner_gate_preflight_stale":
                is_fresh = False
            else:
                raise
        else:
            is_fresh = True
        if (
            set(os.listdir(transaction_root)) != _TRANSACTION_NAMES
            or _require_directory(
                transaction_root,
                parent=phase_root,
                code="owner_gate_inert_observation_manual_reconciliation_required",
                mode=0o500,
            )
            != transaction_identity
        ):
            _error("owner_gate_inert_observation_manual_reconciliation_required")
        for name in (*_EVIDENCE_NAMES, RECEIPT_NAME):
            identity, observed_raw = _read_pinned_file(
                transaction_root / name,
                maximum=(
                    launcher._MAX_JSON_LINE_BYTES
                    if name == RECEIPT_NAME
                    else MAX_EVIDENCE_BYTES
                ),
                code="owner_gate_inert_observation_manual_reconciliation_required",
            )
            if identity != identities[name] or observed_raw != raw[name]:
                _error("owner_gate_inert_observation_manual_reconciliation_required")
        if not is_fresh:
            return receipt, False
        return receipt, True
    except launcher.OwnerLauncherError:
        raise
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        preflight.OwnerGatePreflightError,
    ) as exc:
        _error("owner_gate_inert_observation_manual_reconciliation_required", exc)


def _find_fresh_replay(
    *,
    phase_root: Path,
    release_revision: str,
    inputs: _PinnedObservationInputs,
    loaded: _LoadedFoundation,
    package: Mapping[str, Any],
    plan: foundation.OwnerGateFoundationPlan,
    cloud_key: Ed25519PublicKey,
    host_key: Ed25519PublicKey,
    now_unix: int,
) -> Mapping[str, Any] | None:
    try:
        inventory = set(os.listdir(phase_root))
    except OSError as exc:
        _error("owner_gate_inert_observation_manual_reconciliation_required", exc)
    if any(_SHA256.fullmatch(name) is None for name in inventory):
        _error("owner_gate_inert_observation_manual_reconciliation_required")
    fresh: list[Mapping[str, Any]] = []
    for transaction_name in sorted(inventory):
        receipt, is_fresh = _load_evidence_transaction(
            phase_root=phase_root,
            transaction_name=transaction_name,
            release_revision=release_revision,
            inputs=inputs,
            loaded=loaded,
            package=package,
            plan=plan,
            cloud_key=cloud_key,
            host_key=host_key,
            now_unix=now_unix,
        )
        if is_fresh:
            fresh.append(receipt)
    if len(fresh) > 1:
        _error("owner_gate_inert_observation_manual_reconciliation_required")
    try:
        final_inventory = set(os.listdir(phase_root))
    except OSError as exc:
        _error("owner_gate_inert_observation_manual_reconciliation_required", exc)
    if final_inventory != inventory:
        _error("owner_gate_inert_observation_manual_reconciliation_required")
    return fresh[0] if fresh else None


def _write_staged_file(path: Path, raw: bytes) -> None:
    descriptor: int | None = None
    try:
        if not raw or len(raw) > MAX_EVIDENCE_BYTES:
            _error("owner_gate_inert_observation_evidence_publish_failed")
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(raw)
        written = 0
        while written < len(raw):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                _error("owner_gate_inert_observation_evidence_publish_failed")
            written += count
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok
            or opened.st_gid != os.getegid()  # windows-footgun: ok
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_size != len(raw)
        ):
            _error("owner_gate_inert_observation_evidence_publish_failed")
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error("owner_gate_inert_observation_evidence_publish_failed", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error("owner_gate_inert_observation_evidence_publish_failed", exc)


def _publish_evidence(
    *,
    phase_root: Path,
    receipt: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> None:
    pending = phase_root / PENDING_NAME
    destination = phase_root / str(receipt["evidence_set_sha256"])
    if set(payloads) != _TRANSACTION_NAMES or os.path.lexists(pending):
        _error("owner_gate_inert_observation_manual_reconciliation_required")
    pending_created = False
    try:
        os.mkdir(pending, 0o700)
        pending_created = True
        _fsync_directory(
            phase_root,
            code="owner_gate_inert_observation_evidence_publish_failed",
        )
        for name in (*_EVIDENCE_NAMES, RECEIPT_NAME):
            _write_staged_file(pending / name, payloads[name])
        _fsync_directory(
            pending,
            code="owner_gate_inert_observation_evidence_publish_failed",
        )
        os.chmod(pending, 0o500)
        _fsync_directory(
            pending,
            code="owner_gate_inert_observation_evidence_publish_failed",
        )
        _require_directory(
            pending,
            parent=phase_root,
            code="owner_gate_inert_observation_evidence_publish_failed",
            mode=0o500,
        )
        launcher._atomic_rename_no_replace(
            str(pending),
            str(destination),
            exists_code="owner_gate_inert_observation_evidence_exists",
            failed_code="owner_gate_inert_observation_evidence_publish_failed",
        )
        _fsync_directory(
            phase_root,
            code="owner_gate_inert_observation_evidence_publish_failed",
        )
    except (launcher.OwnerLauncherError, OSError) as exc:
        if pending_created or os.path.lexists(pending) or os.path.lexists(destination):
            _error("owner_gate_inert_observation_manual_reconciliation_required", exc)
        _error("owner_gate_inert_observation_evidence_publish_failed", exc)


def _collect_inert_observation(
    *,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
) -> Mapping[str, Any]:
    inputs = _PinnedObservationInputs.load(release_revision)
    loaded = _load_successful_foundation(release_revision)
    plan, package = _final_plan(loaded.chain, inputs.bundle_stream)
    now_unix = int(time.time())
    if now_unix <= 0:
        _error("owner_gate_inert_observation_time_invalid")
    cloud_key = _collector_key(release_revision, role="cloud")
    host_key = _collector_key(release_revision, role="host")
    with _evidence_lease(release_revision) as phase_root:
        replay = _find_fresh_replay(
            phase_root=phase_root,
            release_revision=release_revision,
            inputs=inputs,
            loaded=loaded,
            package=package,
            plan=plan,
            cloud_key=cloud_key,
            host_key=host_key,
            now_unix=now_unix,
        )
        if replay is not None:
            inputs.assert_stable()
            return replay
        transport = stage0_iap.OwnerGateStage0IapTransport(
            release_sha=release_revision,
            owner_identity=owner_identity,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
            foundation_artifacts=loaded.raw_artifacts,
        )
        pair = cloud_author.collect_and_author_bound_pair(
            plan=plan,
            foundation_apply_chain=loaded.chain,
            phase=EVIDENCE_PHASE,
            collected_at_unix=None,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
            owner_identity=owner_identity,
            stage0_transport=transport,
            kit_stream=inputs.kit_stream,
            bundle_stream=inputs.bundle_stream,
        )
        cloud_observation, host_observation = (
            cloud_author.consume_bound_observation_pair(
                pair,
                plan=plan,
                phase=EVIDENCE_PHASE,
            )
        )
        try:
            report = preflight.build_preflight_report(
                plan=plan,
                cloud_observation=cloud_observation,
                host_observation=host_observation,
                cloud_collector_public_key=cloud_key,
                host_collector_public_key=host_key,
                now_unix=now_unix,
            )
        except preflight.OwnerGatePreflightError as exc:
            _error("owner_gate_inert_observation_preflight_invalid", exc)
        inputs.assert_stable()
        if (
            report.get("mutation_performed") is not False
            or report.get("mutation_iam_binding_present") is not False
            or report.get("executor_activation_seal_present") is not False
            or report.get("plan_sha256") != plan.sha256
        ):
            _error("owner_gate_inert_observation_preflight_invalid")
        receipt, payloads = _build_receipt(
            release_revision=release_revision,
            inputs=inputs,
            loaded=loaded,
            package=package,
            plan=plan,
            cloud_observation=cloud_observation,
            host_observation=host_observation,
            report=report,
        )
        _publish_evidence(
            phase_root=phase_root,
            receipt=receipt,
            payloads=payloads,
        )
        persisted = _find_fresh_replay(
            phase_root=phase_root,
            release_revision=release_revision,
            inputs=inputs,
            loaded=loaded,
            package=package,
            plan=plan,
            cloud_key=cloud_key,
            host_key=host_key,
            now_unix=now_unix,
        )
        inputs.assert_stable()
        if persisted != receipt:
            _error("owner_gate_inert_observation_manual_reconciliation_required")
        return persisted


def collect_inert_observation(
    *,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
) -> Mapping[str, Any]:
    """Run only from the sealed, release-proven owner launcher action."""

    if (
        _REVISION.fullmatch(release_revision or "") is None
        or type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or owner_identity.gcloud_configuration is not gcloud_configuration
        or getattr(owner_identity, "_gcloud_executable", None)
        is not gcloud_executable
    ):
        _error("owner_gate_inert_observation_capability_invalid")
    launcher.require_trusted_owner_support_activation(
        gcloud_executable,
        release_sha=release_revision,
    )
    launcher.require_local_launcher_provenance(release_revision)
    receipt = _collect_inert_observation(
        release_revision=release_revision,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        owner_identity=owner_identity,
    )
    launcher.require_trusted_owner_support_activation(
        gcloud_executable,
        release_sha=release_revision,
    )
    launcher.require_local_launcher_provenance(release_revision)
    return receipt


__all__ = [
    "BUNDLE_STREAM_NAME",
    "EVIDENCE_ROOT",
    "EVIDENCE_SET_ID_SCHEMA",
    "INPUT_PINS_SCHEMA",
    "INPUT_ROOT",
    "INERT_CLOUD_OBSERVATION_NAME",
    "INERT_HOST_OBSERVATION_NAME",
    "INERT_PREFLIGHT_NAME",
    "KIT_STREAM_NAME",
    "PINS_NAME",
    "RECEIPT_NAME",
    "RECEIPT_SCHEMA",
    "collect_inert_observation",
]
