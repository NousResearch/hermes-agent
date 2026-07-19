#!/usr/bin/env python3
"""Root-only fixed-path publisher for owner-signed production cutover inputs.

The caller supplies canonical public JSON on stdin.  This module never accepts
paths, credentials, or secret bytes.  It validates every typed object against
the production coordinator and atomically creates (or exactly resumes) only
the fixed root-owned 0400 staging files consumed by the cutover service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from gateway import canonical_writer_production_cutover as cutover
from scripts.canary import package_production_cutover_artifacts as package
from scripts.canary import production_cutover_passkey as passkey


PUBLICATION_SCHEMA = "muncho-production-cutover-publication.v1"
RECEIPT_SCHEMA = "muncho-production-cutover-publication-receipt.v1"
MAX_INPUT = 16 * 1024 * 1024
STAGED_MODE = 0o400
_ACTIONS = frozenset({"unit-input-authority", "freeze-authority", "cutover-plan"})
_FIELDS = frozenset({
    "schema",
    "action",
    "release_revision",
    "documents",
    "secret_material_recorded",
    "secret_digest_recorded",
    "publication_sha256",
})


class PublicStagingError(RuntimeError):
    """One stable public staging failure."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in items:
            if name in value:
                raise PublicStagingError("public_staging_duplicate_key")
            value[name] = item
        return value

    def constant(_value: str) -> None:
        raise PublicStagingError("public_staging_nonfinite_number")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except PublicStagingError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PublicStagingError("public_staging_json_invalid") from exc
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise PublicStagingError("public_staging_json_not_canonical")
    return value


def _publication(
    value: Mapping[str, Any],
    *,
    now_unix: int | None = None,
) -> tuple[str, dict[Path, bytes]]:
    current = int(time.time()) if now_unix is None else now_unix
    if set(value) != _FIELDS:
        raise PublicStagingError("public_staging_fields_invalid")
    unsigned = {
        name: item for name, item in value.items() if name != "publication_sha256"
    }
    if (
        value.get("schema") != PUBLICATION_SCHEMA
        or value.get("action") not in _ACTIONS
        or not isinstance(value.get("release_revision"), str)
        or package.REVISION.fullmatch(value["release_revision"]) is None
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("publication_sha256") != _sha(_canonical(unsigned))
        or not isinstance(value.get("documents"), Mapping)
    ):
        raise PublicStagingError("public_staging_identity_invalid")
    action = str(value["action"])
    revision = str(value["release_revision"])
    documents = dict(value["documents"])
    outputs: dict[Path, bytes]
    try:
        if action == "unit-input-authority":
            if set(documents) != {"plan", "approval"}:
                raise PublicStagingError("public_staging_documents_invalid")
            plan = package.validate_unit_input_plan(documents["plan"])
            approval = package.validate_unit_input_approval(
                documents["approval"],
                plan=plan,
                now_unix=current,
            )
            if plan["release_revision"] != revision:
                raise PublicStagingError("public_staging_revision_invalid")
            outputs = {
                package.STAGED_UNIT_INPUT_PLAN_PATH: _canonical(plan),
                package.STAGED_UNIT_INPUT_APPROVAL_PATH: _canonical(approval),
            }
        elif action == "freeze-authority":
            if set(documents) != {"plan", "approval"}:
                raise PublicStagingError("public_staging_documents_invalid")
            plan = cutover.FreezePlan.from_mapping(documents["plan"])
            approval = cutover.CutoverApproval.from_mapping(
                documents["approval"],
                plan=plan,
                now_unix=current,
            )
            if plan.value["release_revision"] != revision:
                raise PublicStagingError("public_staging_revision_invalid")
            outputs = {
                cutover.STAGED_FREEZE_PLAN_PATH: _canonical(plan.to_mapping()),
                cutover.STAGED_FREEZE_APPROVAL_PATH: _canonical(approval.value),
            }
        else:
            if set(documents) != {"plan"}:
                raise PublicStagingError("public_staging_documents_invalid")
            plan = cutover.CutoverPlan.from_mapping(documents["plan"])
            if plan.value["release_revision"] != revision:
                raise PublicStagingError("public_staging_revision_invalid")
            outputs = {
                cutover.STAGED_CUTOVER_PLAN_PATH: _canonical(plan.to_mapping()),
            }
    except PublicStagingError:
        raise
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        raise PublicStagingError("public_staging_documents_invalid") from exc
    return action, outputs


def build_publication(
    *,
    action: str,
    release_revision: str,
    documents: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": PUBLICATION_SCHEMA,
        "action": action,
        "release_revision": release_revision,
        "documents": dict(documents),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    value = {**unsigned, "publication_sha256": _sha(_canonical(unsigned))}
    _publication(value, now_unix=now_unix)
    return value


def _read_exact(path: Path, state: os.stat_result, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (state.st_dev, state.st_ino)
            or opened.st_size != state.st_size
            or opened.st_size < 0
            or opened.st_size > maximum
        ):
            raise PublicStagingError("public_staging_file_changed")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != opened.st_size or len(payload) > maximum:
            raise PublicStagingError("public_staging_file_changed")
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PublicStagingError("public_staging_file_changed")
        return payload
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _install_exact(path: Path, payload: bytes, *, uid: int, gid: int) -> bool:
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
        raise PublicStagingError("public_staging_path_invalid")
    parent = path.parent.lstat()
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != uid
        or parent.st_gid != gid
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise PublicStagingError("public_staging_parent_invalid")
    temporary = path.with_name(f".{path.name}.stage.{os.getpid()}")
    created = False
    descriptor: int | None = None
    try:
        if not os.path.lexists(path):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600)
            os.fchown(descriptor, uid, gid)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short public staging write")
                view = view[written:]
            os.fchmod(descriptor, STAGED_MODE)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(temporary, path, follow_symlinks=False)
                created = True
                cutover.activation._fsync_directory(path.parent)
            except FileExistsError:
                pass
            temporary.unlink()
        state = path.lstat()
        if (
            path.resolve(strict=True) != path
            or stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or state.st_nlink != 1
            or state.st_uid != uid
            or state.st_gid != gid
            or stat.S_IMODE(state.st_mode) != STAGED_MODE
            or _read_exact(path, state, maximum=MAX_INPUT) != payload
        ):
            raise PublicStagingError("public_staging_conflict")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return created


def _remove_exact_created(path: Path, payload: bytes, *, uid: int, gid: int) -> None:
    try:
        state = path.lstat()
        if (
            path.resolve(strict=True) != path
            or stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or state.st_nlink != 1
            or state.st_uid != uid
            or state.st_gid != gid
            or stat.S_IMODE(state.st_mode) != STAGED_MODE
            or _read_exact(path, state, maximum=MAX_INPUT) != payload
        ):
            raise PublicStagingError("public_staging_rollback_conflict")
        path.unlink()
        cutover.activation._fsync_directory(path.parent)
    except FileNotFoundError:
        return
    except PublicStagingError:
        raise
    except OSError as exc:
        raise PublicStagingError("public_staging_rollback_failed") from exc


def stage_publication(
    value: Mapping[str, Any],
    *,
    require_root: bool = True,
    now_unix: int | None = None,
    journal: cutover.CutoverJournal | None = None,
    lock_factory: Any | None = None,
) -> Mapping[str, Any]:
    if require_root and (
        not sys.platform.startswith("linux") or os.geteuid() != 0  # windows-footgun: ok — Linux production/canary boundary
    ):
        raise PublicStagingError("public_staging_requires_linux_root")
    uid = 0 if require_root else os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    gid = 0 if require_root else os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    current = int(time.time()) if now_unix is None else now_unix
    proof: Mapping[str, Any] | None = None
    replay_only = False
    publication = value
    if value.get("schema") == passkey.CUTOVER_CLAIM_FRAME_SCHEMA:
        try:
            publication, proof = passkey.validate_claim_frame(
                value, now_unix=current
            )
        except passkey.ProductionCutoverPasskeyError:
            try:
                publication, proof = (
                    passkey.validate_claim_frame_for_recorded_replay(value)
                )
                replay_only = True
            except passkey.ProductionCutoverPasskeyError:
                raise PublicStagingError(
                    "public_staging_passkey_claim_invalid"
                ) from None
    publication_validation_time = (
        proof["authorization_receipt"]["consumed_at_unix"]
        if replay_only and proof is not None
        else current
    )
    action, outputs = _publication(
        publication,
        now_unix=publication_validation_time,
    )
    if action == "freeze-authority" and proof is None:
        raise PublicStagingError("public_staging_passkey_claim_required")
    if action != "freeze-authority" and proof is not None:
        raise PublicStagingError("public_staging_passkey_claim_unexpected")
    needs_claim = action in {"freeze-authority", "cutover-plan"}
    if needs_claim and journal is None:
        journal = cutover.RootCutoverJournal()
    if lock_factory is None:
        lock_factory = cutover.activation._host_activation_lock

    def require_or_record_claim() -> None:
        assert journal is not None
        if action == "freeze-authority":
            assert proof is not None
            documents = publication["documents"]
            plan = cutover.FreezePlan.from_mapping(documents["plan"])
            approval = cutover.CutoverApproval.from_mapping(
                documents["approval"],
                plan=plan,
                now_unix=publication_validation_time,
            )
            receipt = proof["authorization_receipt"]
            action_envelope = proof["action_envelope"]
            evidence = {
                "schema": cutover.PASSKEY_CLAIM_SCHEMA,
                "freeze_plan_sha256": plan.sha256,
                "freeze_approval_sha256": approval.sha256,
                "freeze_publication_sha256": publication[
                    "publication_sha256"
                ],
                "passkey_proof_sha256": proof["proof_sha256"],
                "authorization_receipt_sha256": receipt[
                    "receipt_sha256"
                ],
                "action_envelope_sha256": action_envelope[
                    "envelope_sha256"
                ],
                "action_payload_sha256": action_envelope[
                    "action_payload_sha256"
                ],
                "request_id": action_envelope["request_id"],
                "consume_attempt_id": receipt["consume_attempt_id"],
                "authority_release_sha": action_envelope[
                    "authority_release_sha"
                ],
                "execution_window_expires_at_unix": receipt[
                    "execution_window_expires_at_unix"
                ],
            }
            cutover.validate_passkey_claim_evidence(
                evidence,
                plan_sha256=plan.sha256,
                approval_sha256=approval.sha256,
                release_revision=plan.value["release_revision"],
            )
            existing_entry: cutover.JournalEntry | None = None
            existing_claim: Mapping[str, Any] | None = None
            if journal.load(plan.sha256):
                try:
                    existing_entry, existing_claim = (
                        cutover.require_recorded_passkey_claim(
                            journal,
                            plan_sha256=plan.sha256,
                            approval_sha256=approval.sha256,
                            release_revision=plan.value[
                                "release_revision"
                            ],
                        )
                    )
                except (PermissionError, ValueError):
                    raise PublicStagingError(
                        "public_staging_passkey_claim_conflict"
                    ) from None
            if existing_entry is not None and existing_claim == evidence:
                if (
                    existing_entry.value["recorded_at_unix"]
                    < receipt["consumed_at_unix"]
                    or existing_entry.value["recorded_at_unix"]
                    >= receipt["execution_window_expires_at_unix"]
                    or existing_entry.value["recorded_at_unix"]
                    >= approval.value["expires_at_unix"]
                ):
                    raise PublicStagingError(
                        "public_staging_passkey_claim_conflict"
                    )
            elif existing_entry is not None:
                if replay_only:
                    raise PublicStagingError(
                        "public_staging_passkey_claim_expired"
                    )
                try:
                    cutover.supersede_recorded_passkey_claim(
                        journal,
                        plan_sha256=plan.sha256,
                        approval_sha256=approval.sha256,
                        release_revision=plan.value["release_revision"],
                        new_claim_value=evidence,
                        now_unix=current,
                    )
                except (PermissionError, ValueError):
                    raise PublicStagingError(
                        "public_staging_passkey_claim_conflict"
                    ) from None
            else:
                if replay_only:
                    raise PublicStagingError(
                        "public_staging_passkey_claim_expired"
                    )
                journal.append(
                    plan.sha256, "passkey_claim", evidence, current
                )
            # Persist the Ed25519 authority at the same first-write boundary.
            # Later exact recovery therefore never depends on a short lease.
            cutover._append_authority(
                journal, plan.sha256, approval, current
            )
        else:
            plan = cutover.CutoverPlan.from_mapping(
                publication["documents"]["plan"]
            )
            cutover.require_recorded_passkey_claim(
                journal,
                plan_sha256=plan.value["freeze_plan_sha256"],
                approval_sha256=plan.value["freeze_approval_sha256"],
                release_revision=plan.value["release_revision"],
            )

    records = []
    created_outputs: list[tuple[Path, bytes]] = []
    context = lock_factory() if needs_claim else None
    try:
        if context is not None:
            context.__enter__()
            require_or_record_claim()
        for path, payload in outputs.items():
            created = _install_exact(path, payload, uid=uid, gid=gid)
            if created:
                created_outputs.append((path, payload))
            records.append({
                "path": str(path),
                "sha256": _sha(payload),
                "created": created,
            })
    except Exception:
        for path, payload in reversed(created_outputs):
            _remove_exact_created(path, payload, uid=uid, gid=gid)
        raise
    finally:
        if context is not None:
            context.__exit__(*sys.exc_info())
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "action": action,
        "release_revision": publication["release_revision"],
        "publication_sha256": publication["publication_sha256"],
        "files": records,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage one fixed owner-signed production cutover publication",
    )
    parser.parse_args(argv)
    raw = sys.stdin.buffer.read(MAX_INPUT + 1)
    try:
        if not raw or len(raw) > MAX_INPUT or raw.endswith(b"\n"):
            raise PublicStagingError("public_staging_input_invalid")
        receipt = stage_publication(_decode(raw))
    except (OSError, PublicStagingError):
        print('{"error_code":"public_staging_failed","ok":false}', file=sys.stderr)
        return 2
    print(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
