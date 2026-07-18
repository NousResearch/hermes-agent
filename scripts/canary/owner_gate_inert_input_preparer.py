#!/usr/bin/env python3
"""Prepare the fixed owner-side input set for one inert observation.

The public boundary accepts no paths.  It requires two immutable prerequisites
below fixed owner-only roots: an exact clean checkout of the release and an
already-materialized offline owner-gate bundle.  Package authoring is
deliberately outside this module.  The existing reviewed kit and tree-stream
builders create the two streams, and one atomic no-replace directory rename
publishes those streams with their canonical self-hashed pins.

The preflight performs no persistent write.  A crash before the atomic rename
leaves only the fixed ``.pending`` directory and fails closed for manual
reconciliation; a crash after the rename is an exact replay through the normal
pinned-input reader.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import pwd
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Never, Protocol

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_observation as inert
from scripts.canary import owner_gate_outer_stage0 as outer
from scripts.canary import owner_gate_stage0 as stage0
from scripts.canary import owner_gate_stage0_iap as stage0_iap


PREFLIGHT_SCHEMA = "muncho-owner-gate-inert-input-preflight.v1"
RECEIPT_SCHEMA = "muncho-owner-gate-inert-input-preparation-receipt.v1"
OWNER_HOME = Path(pwd.getpwuid(os.geteuid()).pw_dir)  # windows-footgun: ok
TRUSTED_ROOT = OWNER_HOME / ".hermes" / "trusted"
RELEASE_SOURCE_BASE = TRUSTED_ROOT / "owner-gate-release-sources"
BUNDLE_SOURCE_BASE = TRUSTED_ROOT / "owner-gate-offline-bundles"
LOCK_ROOT = OWNER_HOME / ".hermes" / "owner-gate-inert-input-locks"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SealedOwnerSupportRuntime(Protocol):
    def trusted_owner_support_paths(self) -> tuple[str, str]: ...

    def sealed_owner_support_manifest(
        self,
        *,
        expected_release_sha: str,
    ) -> Mapping[str, Any]: ...


def _error(code: str, _cause: BaseException | None = None) -> Never:
    raise launcher.OwnerLauncherError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_inert_input_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
            os.close(descriptor)


def _require_directory(
    path: Path,
    *,
    parent: Path,
    mode: int,
    code: str,
) -> tuple[int, ...]:
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


def _ensure_directory(
    path: Path,
    *,
    parent: Path,
    mode: int,
    code: str,
) -> tuple[int, ...]:
    if not os.path.lexists(path):
        try:
            os.mkdir(path, mode)
            os.chown(path, os.geteuid(), os.getegid())  # windows-footgun: ok
            os.chmod(path, mode)
            _fsync_directory(parent, code=code)
        except FileExistsError:
            pass
        except OSError as exc:
            _error(code, exc)
    return _require_directory(path, parent=parent, mode=mode, code=code)


def _hermes_root() -> Path:
    hermes = OWNER_HOME / ".hermes"
    _require_directory(
        hermes,
        parent=OWNER_HOME,
        mode=0o700,
        code="owner_gate_inert_input_owner_root_invalid",
    )
    return hermes


def _read_regular(
    path: Path,
    *,
    maximum: int,
    modes: frozenset[int],
    code: str,
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
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
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok
            or opened.st_gid != os.getegid()  # windows-footgun: ok
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) not in modes
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
        return bytes(raw)
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error(code, exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fixed_release_source(
    release_revision: str,
    runtime: SealedOwnerSupportRuntime,
) -> tuple[Path, str]:
    launcher.require_trusted_owner_support_activation(
        runtime,  # type: ignore[arg-type]
        release_sha=release_revision,
    )
    runtime.trusted_owner_support_paths()
    support_manifest = runtime.sealed_owner_support_manifest(
        expected_release_sha=release_revision,
    )
    expected_tree = support_manifest.get("source_tree_oid")
    if (
        support_manifest.get("release_sha") != release_revision
        or _REVISION.fullmatch(str(expected_tree or "")) is None
    ):
        _error("owner_gate_inert_input_sealed_source_invalid")
    _require_directory(
        TRUSTED_ROOT,
        parent=_hermes_root(),
        mode=0o700,
        code="owner_gate_inert_input_trusted_root_invalid",
    )
    if not os.path.lexists(RELEASE_SOURCE_BASE):
        _error("owner_gate_inert_input_release_source_missing")
    _require_directory(
        RELEASE_SOURCE_BASE,
        parent=TRUSTED_ROOT,
        mode=0o700,
        code="owner_gate_inert_input_release_source_invalid",
    )
    source = RELEASE_SOURCE_BASE / release_revision
    if not os.path.lexists(source):
        _error("owner_gate_inert_input_release_checkout_missing")
    _require_directory(
        source,
        parent=RELEASE_SOURCE_BASE,
        mode=0o700,
        code="owner_gate_inert_input_release_checkout_invalid",
    )
    try:
        observed_tree = outer.verify_local_provenance(
            source,
            release_revision=release_revision,
        )
    except outer.OwnerGateOuterStage0Error as exc:
        _error("owner_gate_inert_input_release_checkout_invalid", exc)
    if observed_tree != expected_tree:
        _error("owner_gate_inert_input_release_checkout_mismatch")
    return source, observed_tree


def _expected_bundle_inventory(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, int], set[str]]:
    files = {
        "package-manifest.json": 0o444,
        "trust/release-trust.json": 0o444,
        "trust/release-trust-signing.pub": 0o444,
        "trust/direct-iam-identity-authority.json": 0o444,
        "migration/credential.json": 0o400,
    }
    collectors = manifest.get("collector_public_key_ids")
    payloads = manifest.get("payloads")
    wheels = manifest.get("wheels")
    bootstrap = manifest.get("bootstrap_pip")
    if (
        not isinstance(collectors, Mapping)
        or set(collectors) != {"network", "cloud", "host"}
        or not isinstance(payloads, list)
        or not isinstance(wheels, list)
        or not isinstance(bootstrap, Mapping)
    ):
        _error("owner_gate_inert_input_bundle_invalid")
    files.update({
        f"trust/{role}-observation-attestation.pub": 0o444 for role in collectors
    })
    try:
        for item in payloads:
            mode = str(item["mode"])
            if mode not in {"0444", "0555"}:
                _error("owner_gate_inert_input_bundle_invalid")
            files[f"payload/{item['release_relative']}"] = int(mode, 8)
        files.update({f"wheels/{item['filename']}": 0o444 for item in wheels})
        files[f"bootstrap/{bootstrap['filename']}"] = 0o444
        normalized = {
            str(outer._safe_relative(relative)): mode
            for relative, mode in files.items()
        }
    except (KeyError, TypeError, outer.OwnerGateOuterStage0Error) as exc:
        _error("owner_gate_inert_input_bundle_invalid", exc)
    directories = {
        str(parent)
        for relative in normalized
        for parent in Path(relative).parents
        if str(parent) != "."
    }
    return normalized, directories


def _require_exact_bundle_inventory(
    bundle: Path,
    manifest: Mapping[str, Any],
) -> None:
    expected_files, expected_directories = _expected_bundle_inventory(manifest)
    observed_files: dict[str, int] = {}
    observed_directories: set[str] = set()
    try:
        for path in sorted(bundle.rglob("*")):
            state = path.lstat()
            relative = str(path.relative_to(bundle))
            if (
                stat.S_ISLNK(state.st_mode)
                or state.st_uid != os.geteuid()  # windows-footgun: ok
                or state.st_gid != os.getegid()  # windows-footgun: ok
            ):
                _error("owner_gate_inert_input_bundle_invalid")
            if stat.S_ISDIR(state.st_mode):
                if stat.S_IMODE(state.st_mode) != 0o555:
                    _error("owner_gate_inert_input_bundle_invalid")
                observed_directories.add(relative)
            elif stat.S_ISREG(state.st_mode) and state.st_nlink == 1:
                observed_files[relative] = stat.S_IMODE(state.st_mode)
            else:
                _error("owner_gate_inert_input_bundle_invalid")
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error("owner_gate_inert_input_bundle_invalid", exc)
    if observed_files != expected_files or observed_directories != expected_directories:
        _error("owner_gate_inert_input_bundle_invalid")


def _fixed_bundle(
    release_revision: str,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], str]:
    if not os.path.lexists(BUNDLE_SOURCE_BASE):
        _error("owner_gate_inert_input_bundle_source_missing")
    _require_directory(
        BUNDLE_SOURCE_BASE,
        parent=TRUSTED_ROOT,
        mode=0o700,
        code="owner_gate_inert_input_bundle_source_invalid",
    )
    bundle = BUNDLE_SOURCE_BASE / release_revision
    if not os.path.lexists(bundle):
        _error("owner_gate_inert_input_bundle_missing")
    _require_directory(
        bundle,
        parent=BUNDLE_SOURCE_BASE,
        mode=0o555,
        code="owner_gate_inert_input_bundle_invalid",
    )
    try:
        manifest = stage0.verify_bundle_stage0(
            bundle,
            expected_uid=os.geteuid(),  # windows-footgun: ok
        )
    except stage0.OwnerGateStage0Error as exc:
        _error("owner_gate_inert_input_bundle_invalid", exc)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("release_revision") != release_revision
        or _REVISION.fullmatch(str(manifest.get("source_tree_oid", ""))) is None
        or _SHA256.fullmatch(str(manifest.get("package_sha256", ""))) is None
        or _SHA256.fullmatch(
            str(manifest.get("credential_migration_envelope_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(str(manifest.get("trust_manifest_sha256", ""))) is None
    ):
        _error("owner_gate_inert_input_bundle_invalid")
    _require_exact_bundle_inventory(bundle, manifest)
    package_manifest_raw = _read_regular(
        bundle / "package-manifest.json",
        maximum=stage0.MAX_JSON_BYTES,
        modes=frozenset({0o444}),
        code="owner_gate_inert_input_bundle_invalid",
    )
    try:
        tree_manifest = outer.build_tree_stream_manifest(
            bundle,
            purpose="owner-gate-bundle",
            release_id=release_revision,
        )
    except outer.OwnerGateOuterStage0Error as exc:
        _error("owner_gate_inert_input_bundle_invalid", exc)
    return bundle, dict(manifest), dict(tree_manifest), _sha256(package_manifest_raw)


def _receipt(
    *,
    schema: str,
    state: str,
    release_revision: str,
    source_tree_oid: str,
    package_manifest: Mapping[str, Any],
    package_manifest_file_sha256: str,
    kit_release_id: str,
    bundle_tree_manifest_sha256: str,
    pins: Mapping[str, Any] | None,
    publication_performed: bool,
) -> Mapping[str, Any]:
    unsigned: dict[str, Any] = {
        "schema": schema,
        "ok": True,
        "state": state,
        "release_revision": release_revision,
        "source_tree_oid": source_tree_oid,
        "fixed_release_source": str(RELEASE_SOURCE_BASE / release_revision),
        "fixed_bundle_source": str(BUNDLE_SOURCE_BASE / release_revision),
        "fixed_input_release_root": str(inert.INPUT_ROOT / release_revision),
        "package_sha256": package_manifest["package_sha256"],
        "package_manifest_file_sha256": package_manifest_file_sha256,
        "release_trust_manifest_sha256": package_manifest["trust_manifest_sha256"],
        "credential_migration_envelope_sha256": package_manifest[
            "credential_migration_envelope_sha256"
        ],
        "kit_release_id": kit_release_id,
        "bundle_tree_manifest_sha256": bundle_tree_manifest_sha256,
        "caller_selected_path_accepted": False,
        "cloud_mutation_performed": False,
        "service_activation_performed": False,
        "inputs_ready": pins is not None,
        "input_publication_performed": publication_performed,
    }
    if pins is not None:
        unsigned.update({
            "pins_sha256": pins["pins_sha256"],
            "kit_tree_manifest_sha256": pins["kit_tree_manifest_sha256"],
            "kit_stream_sha256": pins["kit_stream_sha256"],
            "bundle_stream_sha256": pins["bundle_stream_sha256"],
        })
    return {**unsigned, "receipt_sha256": foundation.sha256_json(unsigned)}


def _existing_inputs(
    release_revision: str,
) -> inert._PinnedObservationInputs | None:
    if os.path.lexists(inert.INPUT_ROOT / f".{release_revision}.pending"):
        _error("owner_gate_inert_input_manual_reconciliation_required")
    final = inert.INPUT_ROOT / release_revision
    if not os.path.lexists(final):
        return None
    try:
        return inert._PinnedObservationInputs.load(release_revision)
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_inert_input_existing_invalid", exc)


def _validated_prerequisites(
    *,
    release_revision: str,
    gcloud_executable: SealedOwnerSupportRuntime,
) -> tuple[Path, str, Path, Mapping[str, Any], Mapping[str, Any], str, str]:
    if _REVISION.fullmatch(release_revision or "") is None:
        _error("owner_gate_inert_input_release_invalid")
    source, source_tree_oid = _fixed_release_source(
        release_revision,
        gcloud_executable,
    )
    bundle, package_manifest, bundle_tree, package_file_sha256 = _fixed_bundle(
        release_revision
    )
    if package_manifest.get("source_tree_oid") != source_tree_oid:
        _error("owner_gate_inert_input_bundle_release_mismatch")
    try:
        kit_manifest = outer.build_manifest(
            source,
            release_revision=release_revision,
            source_tree_oid=source_tree_oid,
        )
    except outer.OwnerGateOuterStage0Error as exc:
        _error("owner_gate_inert_input_kit_invalid", exc)
    kit_release_id = _sha256(outer.canonical_json_bytes(kit_manifest))
    return (
        source,
        source_tree_oid,
        bundle,
        package_manifest,
        bundle_tree,
        package_file_sha256,
        kit_release_id,
    )


def preflight_inert_observation_inputs(
    *,
    release_revision: str,
    gcloud_executable: SealedOwnerSupportRuntime,
) -> Mapping[str, Any]:
    """Validate the fixed sources and any final input without writing state."""

    (
        _source,
        source_tree_oid,
        _bundle,
        package_manifest,
        bundle_tree,
        package_file_sha256,
        kit_release_id,
    ) = _validated_prerequisites(
        release_revision=release_revision,
        gcloud_executable=gcloud_executable,
    )
    existing = _existing_inputs(release_revision)
    bundle_manifest_sha256 = _sha256(outer.canonical_json_bytes(bundle_tree))
    if existing is not None and (
        existing.pins["kit_release_id"] != kit_release_id
        or existing.pins["bundle_tree_manifest_sha256"] != bundle_manifest_sha256
    ):
        _error("owner_gate_inert_input_existing_invalid")
    return _receipt(
        schema=PREFLIGHT_SCHEMA,
        state=("inputs_ready" if existing is not None else "prerequisites_ready"),
        release_revision=release_revision,
        source_tree_oid=source_tree_oid,
        package_manifest=package_manifest,
        package_manifest_file_sha256=package_file_sha256,
        kit_release_id=kit_release_id,
        bundle_tree_manifest_sha256=bundle_manifest_sha256,
        pins=(existing.pins if existing is not None else None),
        publication_performed=False,
    )


def _write_pins(path: Path, pins: Mapping[str, Any]) -> None:
    raw = _canonical(pins)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        os.fchown(descriptor, os.geteuid(), os.getegid())  # windows-footgun: ok
        os.fchmod(descriptor, 0o400)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        _error("owner_gate_inert_input_publish_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_materialized_kit(path: Path, *, parent: Path) -> None:
    if not path.is_absolute() or path.parent != parent:
        _error("owner_gate_inert_input_kit_cleanup_failed")
    try:
        entries = sorted(path.rglob("*"), key=lambda item: len(item.parts))
        directories = [path]
        for entry in entries:
            state = entry.lstat()
            if (
                stat.S_ISLNK(state.st_mode)
                or state.st_uid != os.geteuid()  # windows-footgun: ok
                or state.st_gid != os.getegid()  # windows-footgun: ok
            ):
                _error("owner_gate_inert_input_kit_cleanup_failed")
            if stat.S_ISDIR(state.st_mode):
                directories.append(entry)
            elif not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
                _error("owner_gate_inert_input_kit_cleanup_failed")
        for directory in directories:
            directory.chmod(0o700)
        for entry in sorted(entries, key=lambda item: len(item.parts), reverse=True):
            if entry.is_dir():
                entry.rmdir()
            else:
                entry.unlink()
        path.rmdir()
        _fsync_directory(parent, code="owner_gate_inert_input_kit_cleanup_failed")
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error("owner_gate_inert_input_kit_cleanup_failed", exc)


@contextmanager
def _preparation_lock(release_revision: str) -> Iterator[None]:
    _ensure_directory(
        LOCK_ROOT,
        parent=_hermes_root(),
        mode=0o700,
        code="owner_gate_inert_input_lock_invalid",
    )
    path = LOCK_ROOT / f"{release_revision}.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchown(descriptor, os.geteuid(), os.getegid())  # windows-footgun: ok
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != os.geteuid()  # windows-footgun: ok
            or opened.st_gid != os.getegid()  # windows-footgun: ok
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            _error("owner_gate_inert_input_lock_invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except launcher.OwnerLauncherError:
        raise
    except OSError as exc:
        _error("owner_gate_inert_input_lock_invalid", exc)
    finally:
        if descriptor is not None:
            failed = False
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                failed = True
            try:
                os.close(descriptor)
            except OSError:
                failed = True
            if failed:
                _error("owner_gate_inert_input_lock_release_failed")


def prepare_inert_observation_inputs(
    *,
    release_revision: str,
    gcloud_executable: SealedOwnerSupportRuntime,
) -> Mapping[str, Any]:
    """Atomically publish or exactly replay the fixed three-file input set."""

    (
        source,
        source_tree_oid,
        bundle,
        package_manifest,
        bundle_tree,
        package_file_sha256,
        kit_release_id,
    ) = _validated_prerequisites(
        release_revision=release_revision,
        gcloud_executable=gcloud_executable,
    )
    bundle_manifest_sha256 = _sha256(outer.canonical_json_bytes(bundle_tree))
    with _preparation_lock(release_revision):
        _ensure_directory(
            inert.INPUT_ROOT,
            parent=_hermes_root(),
            mode=0o700,
            code="owner_gate_inert_input_root_invalid",
        )
        existing = _existing_inputs(release_revision)
        if existing is not None:
            if (
                existing.pins["kit_release_id"] != kit_release_id
                or existing.pins["bundle_tree_manifest_sha256"]
                != bundle_manifest_sha256
            ):
                _error("owner_gate_inert_input_existing_invalid")
            return _receipt(
                schema=RECEIPT_SCHEMA,
                state="exact_replay",
                release_revision=release_revision,
                source_tree_oid=source_tree_oid,
                package_manifest=package_manifest,
                package_manifest_file_sha256=package_file_sha256,
                kit_release_id=kit_release_id,
                bundle_tree_manifest_sha256=bundle_manifest_sha256,
                pins=existing.pins,
                publication_performed=False,
            )

        pending = inert.INPUT_ROOT / f".{release_revision}.pending"
        try:
            os.mkdir(pending, 0o700)
            os.chown(pending, os.geteuid(), os.getegid())  # windows-footgun: ok
            os.chmod(pending, 0o700)
            _fsync_directory(
                inert.INPUT_ROOT,
                code="owner_gate_inert_input_publish_failed",
            )
        except FileExistsError:
            _error("owner_gate_inert_input_manual_reconciliation_required")
        except OSError as exc:
            _error("owner_gate_inert_input_publish_failed", exc)

        kit = pending / ".outer-stage0-kit"
        try:
            kit_manifest = outer.materialize_kit(
                source,
                kit,
                release_revision=release_revision,
                source_tree_oid=source_tree_oid,
            )
            if _sha256(outer.canonical_json_bytes(kit_manifest)) != kit_release_id:
                _error("owner_gate_inert_input_kit_changed")
            kit_build = outer.write_tree_stream(
                kit,
                pending / inert.KIT_STREAM_NAME,
                purpose="outer-stage0-kit",
                release_id=kit_release_id,
            )
            _remove_materialized_kit(kit, parent=pending)
            bundle_build = outer.write_tree_stream(
                bundle,
                pending / inert.BUNDLE_STREAM_NAME,
                purpose="owner-gate-bundle",
                release_id=release_revision,
            )
        except launcher.OwnerLauncherError:
            raise
        except (outer.OwnerGateOuterStage0Error, OSError) as exc:
            _error("owner_gate_inert_input_build_failed", exc)
        if bundle_build.get("stream_manifest_sha256") != bundle_manifest_sha256:
            _error("owner_gate_inert_input_bundle_changed")
        try:
            _kit_identity, kit_raw = inert._read_pinned_file(
                pending / inert.KIT_STREAM_NAME,
                maximum=stage0_iap.MAX_STREAM_BYTES,
                code="owner_gate_inert_input_build_failed",
            )
            _bundle_identity, bundle_raw = inert._read_pinned_file(
                pending / inert.BUNDLE_STREAM_NAME,
                maximum=stage0_iap.MAX_STREAM_BYTES,
                code="owner_gate_inert_input_build_failed",
            )
            pinned_bundle = stage0_iap.PinnedExactTreeStream(
                pending / inert.BUNDLE_STREAM_NAME,
                purpose="owner-gate-bundle",
                release_id=release_revision,
                expected_manifest_sha256=bundle_manifest_sha256,
            )
            inert._load_release_binding(release_revision, pinned_bundle)
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_inert_input_build_failed", exc)
        unsigned_pins = {
            "schema": inert.INPUT_PINS_SCHEMA,
            "release_revision": release_revision,
            "kit_release_id": kit_release_id,
            "kit_tree_manifest_sha256": kit_build["stream_manifest_sha256"],
            "kit_stream_sha256": _sha256(kit_raw),
            "bundle_tree_manifest_sha256": bundle_manifest_sha256,
            "bundle_stream_sha256": _sha256(bundle_raw),
        }
        pins = {
            **unsigned_pins,
            "pins_sha256": foundation.sha256_json(unsigned_pins),
        }
        _write_pins(pending / inert.PINS_NAME, pins)
        _fsync_directory(pending, code="owner_gate_inert_input_publish_failed")
        try:
            launcher._atomic_rename_no_replace(
                str(pending),
                str(inert.INPUT_ROOT / release_revision),
                exists_code="owner_gate_inert_input_destination_exists",
                failed_code="owner_gate_inert_input_publish_failed",
            )
        except launcher.OwnerLauncherError as exc:
            if exc.code == "owner_gate_inert_input_destination_exists":
                _error("owner_gate_inert_input_manual_reconciliation_required", exc)
            raise
        _fsync_directory(
            inert.INPUT_ROOT,
            code="owner_gate_inert_input_publish_failed",
        )
        try:
            loaded = inert._PinnedObservationInputs.load(release_revision)
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_inert_input_postwrite_invalid", exc)
        if loaded.pins != pins:
            _error("owner_gate_inert_input_postwrite_invalid")
        return _receipt(
            schema=RECEIPT_SCHEMA,
            state="inputs_prepared",
            release_revision=release_revision,
            source_tree_oid=source_tree_oid,
            package_manifest=package_manifest,
            package_manifest_file_sha256=package_file_sha256,
            kit_release_id=kit_release_id,
            bundle_tree_manifest_sha256=bundle_manifest_sha256,
            pins=loaded.pins,
            publication_performed=True,
        )


__all__ = [
    "BUNDLE_SOURCE_BASE",
    "LOCK_ROOT",
    "PREFLIGHT_SCHEMA",
    "RECEIPT_SCHEMA",
    "RELEASE_SOURCE_BASE",
    "prepare_inert_observation_inputs",
    "preflight_inert_observation_inputs",
]
