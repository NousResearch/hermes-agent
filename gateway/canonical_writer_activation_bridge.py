"""Fixed stdin boundary for stopped-only writer activation authority.

The owner launcher authors the existing ``muncho-writer-owner-approval.v1``
receipt and projects fresh read-only IAM evidence.  This packaged module does
not grant approval, collect cloud state, interpret intent, accept paths, or
start a service.  It only validates one bounded canonical frame and stages the
two already-supported activation inputs at their production-pinned paths.

The final-plan action is an explicit, journaled native-to-final replacement.
Both old generations are archived before either staged file is replaced, and
a retry may only finish the same byte-identical transition.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway import canonical_writer_activation as activation
from gateway.canonical_writer_host_authority import (
    ExternalIAMReceipt,
    OwnerApprovalReceipt,
)


FRAME_MAGIC = b"MWA1"
FRAME_SCHEMA = "muncho-writer-owner-authority-frame.v1"
STAGE_RECEIPT_SCHEMA = "muncho-writer-owner-authority-stage-receipt.v1"
INTENT_SCHEMA = "muncho-writer-owner-authority-stage-intent.v1"
MAX_FRAME_BYTES = 192 * 1024
PINNED_APPROVAL_SOURCE_SHA256 = hashlib.sha256(
    b"SHA256:7Ea5WNys9ui7FL/p0FlOnL1ZLr6NPFuewekwqRw/rdw"
).hexdigest()
EVIDENCE_ROOT = Path("/var/lib/muncho-writer-canary-evidence/authority-bridge")

_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
_REVISION = __import__("re").compile(r"^[0-9a-f]{40}$")
_FRAME_KEYS = frozenset({
    "schema",
    "action",
    "scope",
    "revision",
    "plan_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "owner_approval",
    "external_iam_receipt",
    "previous_owner_approval_sha256",
    "previous_external_iam_receipt_sha256",
    "framed_at_unix",
    "frame_sha256",
})
_STAGE_RECEIPT_KEYS = frozenset({
    "schema",
    "ok",
    "state",
    "action",
    "scope",
    "revision",
    "plan_sha256",
    "frame_sha256",
    "owner_subject_sha256",
    "approval_source_sha256",
    "owner_approval_sha256",
    "external_iam_receipt_sha256",
    "external_iam_policy_sha256",
    "previous_owner_approval_sha256",
    "previous_external_iam_receipt_sha256",
    "archive",
    "owner_staged_present",
    "external_iam_staged_present",
    "services_started",
    "services_stopped",
    "intent_path",
    "intent_sha256",
    "receipt_path",
    "completed_at_unix",
    "receipt_sha256",
})


@dataclass(frozen=True)
class AuthorityPaths:
    staged_owner_approval: Path = activation.DEFAULT_STAGED_OWNER_APPROVAL_PATH
    staged_external_iam: Path = activation.DEFAULT_STAGED_EXTERNAL_IAM_PATH
    evidence_root: Path = EVIDENCE_ROOT


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not lowercase SHA-256")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_frame(value: Mapping[str, Any]) -> bytes:
    """Encode one already-complete authority frame for the fixed stdin API."""

    validated = validate_frame(value)
    payload = _canonical_bytes(validated)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("owner authority frame is oversized")
    return FRAME_MAGIC + struct.pack(">I", len(payload)) + payload


def read_frame(stream: Any = None) -> Mapping[str, Any]:
    source = sys.stdin.buffer if stream is None else stream
    raw = source.read(MAX_FRAME_BYTES + 9)
    if not isinstance(raw, bytes) or len(raw) < 9 or len(raw) > MAX_FRAME_BYTES + 8:
        raise ValueError("owner authority frame size is invalid")
    if raw[:4] != FRAME_MAGIC:
        raise ValueError("owner authority frame magic is invalid")
    declared = struct.unpack(">I", raw[4:8])[0]
    payload = raw[8:]
    if declared != len(payload) or declared == 0 or declared > MAX_FRAME_BYTES:
        raise ValueError("owner authority frame length is invalid")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=activation._reject_duplicate_keys,
            parse_constant=activation._reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("owner authority frame is not strict JSON") from exc
    if not isinstance(value, Mapping) or payload != _canonical_bytes(value):
        raise ValueError("owner authority frame is not canonical JSON")
    return validate_frame(value)


def _load_plan(action: str) -> tuple[Any, str, str]:
    if action == "stage-native-authority":
        plan = activation.load_native_observation_plan(
            activation.DEFAULT_NATIVE_PLAN_PATH
        )
        return plan, "native_observation", str(plan.value["revision"])
    if action == "replace-final-authority":
        plan = activation.load_activation_plan(activation.DEFAULT_PLAN_PATH)
        return plan, "activation", str(plan.revision)
    raise ValueError("owner authority action is invalid")


def _plan_policy_sha256(plan: Any, scope: str) -> str:
    if scope == "native_observation":
        return _digest(
            plan.value.get("external_iam_policy_sha256"),
            "native external IAM policy",
        )
    return _digest(
        plan.digests.external_iam_policy_sha256,
        "activation external IAM policy",
    )


def validate_frame(
    value: Mapping[str, Any],
    *,
    now_unix: int | None = None,
    plan_loader: Callable[[str], tuple[Any, str, str]] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FRAME_KEYS:
        raise ValueError("owner authority frame fields are not exact")
    action = value.get("action")
    if value.get("schema") != FRAME_SCHEMA or action not in {
        "stage-native-authority",
        "replace-final-authority",
    }:
        raise ValueError("owner authority frame schema or action is invalid")
    loader = _load_plan if plan_loader is None else plan_loader
    plan, expected_scope, revision = loader(str(action))
    if (
        value.get("scope") != expected_scope
        or value.get("revision") != revision
        or _REVISION.fullmatch(revision) is None
        or value.get("plan_sha256") != plan.sha256
    ):
        raise ValueError("owner authority frame plan binding is invalid")
    owner_subject = _digest(value.get("owner_subject_sha256"), "owner subject")
    source = _digest(value.get("approval_source_sha256"), "approval source")
    if source != PINNED_APPROVAL_SOURCE_SHA256:
        raise PermissionError("owner authority approval source is not pinned")
    approval = OwnerApprovalReceipt.from_mapping(value.get("owner_approval"))
    current = int(time.time()) if now_unix is None else now_unix
    approval.require(
        scope=expected_scope,
        plan_sha256=plan.sha256,
        now_unix=current,
    )
    if (
        approval.value.get("owner_subject_sha256") != owner_subject
        or approval.value.get("approval_source_sha256") != source
    ):
        raise PermissionError("owner authority approval lineage is invalid")
    iam = ExternalIAMReceipt.from_mapping(value.get("external_iam_receipt"))
    iam.require_fresh(current, minimum_remaining_seconds=720)
    if (
        iam.value.get("source_approval_sha256") != approval.sha256
        or iam.policy_sha256 != _plan_policy_sha256(plan, expected_scope)
    ):
        raise PermissionError("owner authority IAM lineage is invalid")
    framed_at = value.get("framed_at_unix")
    if type(framed_at) is not int or not 0 <= current - framed_at <= 60:
        raise ValueError("owner authority frame time is invalid")
    previous_approval = value.get("previous_owner_approval_sha256")
    previous_iam = value.get("previous_external_iam_receipt_sha256")
    if action == "stage-native-authority":
        if previous_approval is not None or previous_iam is not None:
            raise ValueError("native authority frame cannot replace prior authority")
    else:
        _digest(previous_approval, "previous owner approval")
        _digest(previous_iam, "previous external IAM receipt")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "frame_sha256"
    }
    if value.get("frame_sha256") != _sha256(_canonical_bytes(unsigned)):
        raise ValueError("owner authority frame digest is invalid")
    return copy.deepcopy(dict(value))


def _read_optional(path: Path) -> bytes | None:
    if not os.path.lexists(path):
        return None
    return activation._read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )


def _atomic_replace_exact(
    path: Path,
    *,
    expected_previous: bytes | None,
    payload: bytes,
) -> bool:
    current = _read_optional(path)
    if current == payload:
        return False
    if current != expected_previous:
        raise RuntimeError("staged authority generation conflicts with transition")
    activation._validate_root_parent_chain(path.parent)
    temporary = path.with_name(f".{path.name}.authority.{os.getpid()}")
    activation._install_exact_bytes(
        temporary,
        payload,
        uid=0,
        gid=0,
        mode=0o400,
    )
    try:
        reached = os.lstat(temporary)
        if (
            not stat.S_ISREG(reached.st_mode)
            or stat.S_ISLNK(reached.st_mode)
            or reached.st_uid != 0
            or reached.st_gid != 0
            or stat.S_IMODE(reached.st_mode) != 0o400
            or reached.st_nlink != 1
        ):
            raise RuntimeError("staged authority temporary identity is invalid")
        os.replace(temporary, path)
        activation._fsync_directory(path.parent)
    except BaseException:
        try:
            if os.path.lexists(temporary):
                temporary.unlink()
                activation._fsync_directory(temporary.parent)
        except BaseException:
            pass
        raise
    if _read_optional(path) != payload:
        raise RuntimeError("staged authority replacement readback failed")
    return True


def _require_stopped() -> None:
    activation._require_off_or_absent(
        activation.WRITER_UNIT,
        runner=activation._runner,
    )
    activation._require_off_or_absent(
        activation.GATEWAY_UNIT,
        runner=activation._runner,
    )
    activation._require_off_or_absent(
        activation.PHASE_B_READINESS_UNIT,
        runner=activation._runner,
    )
    activation._require_off_disabled(
        activation.EXPORTER_UNIT,
        runner=activation._runner,
        absent=True,
    )
    activation._require_off_disabled(
        activation.DISCORD_UNIT,
        runner=activation._runner,
        absent=True,
    )


def _archive_previous(
    root: Path,
    *,
    owner_raw: bytes,
    iam_raw: bytes,
) -> Mapping[str, Any]:
    owner_sha = _sha256(owner_raw)
    iam_sha = _sha256(iam_raw)
    archive_root = root / "retired" / owner_sha / iam_sha
    activation._ensure_root_directory(archive_root)
    owner_path = archive_root / "owner-approval.json"
    iam_path = archive_root / "external-iam-receipt.json"
    activation._install_exact_bytes(owner_path, owner_raw, uid=0, gid=0, mode=0o400)
    activation._install_exact_bytes(iam_path, iam_raw, uid=0, gid=0, mode=0o400)
    return {
        "owner_approval_path": str(owner_path),
        "owner_approval_file_sha256": owner_sha,
        "external_iam_path": str(iam_path),
        "external_iam_file_sha256": iam_sha,
    }


def _load_archived_previous(
    root: Path,
    *,
    owner_sha256: str,
    iam_sha256: str,
) -> tuple[bytes, bytes, Mapping[str, Any]] | None:
    archive_root = root / "retired" / owner_sha256 / iam_sha256
    owner_path = archive_root / "owner-approval.json"
    iam_path = archive_root / "external-iam-receipt.json"
    if not os.path.lexists(owner_path) and not os.path.lexists(iam_path):
        return None
    owner_raw = _read_optional(owner_path)
    iam_raw = _read_optional(iam_path)
    if (
        owner_raw is None
        or iam_raw is None
        or _sha256(owner_raw) != owner_sha256
        or _sha256(iam_raw) != iam_sha256
    ):
        raise RuntimeError("retired staged authority archive is incomplete")
    return owner_raw, iam_raw, {
        "owner_approval_path": str(owner_path),
        "owner_approval_file_sha256": owner_sha256,
        "external_iam_path": str(iam_path),
        "external_iam_file_sha256": iam_sha256,
    }


def _load_terminal_receipt(
    path: Path,
    *,
    frame: Mapping[str, Any],
    owner_payload: bytes,
    iam_payload: bytes,
    paths: AuthorityPaths,
) -> Mapping[str, Any] | None:
    raw = _read_optional(path)
    if raw is None:
        return None
    value = activation._decode_strict_json(raw, label="authority bridge receipt")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "receipt_sha256"
    }
    if (
        set(value) != _STAGE_RECEIPT_KEYS
        or value.get("schema") != STAGE_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "authority_staged_services_stopped"
        or value.get("frame_sha256") != frame.get("frame_sha256")
        or value.get("plan_sha256") != frame.get("plan_sha256")
        or value.get("services_started") is not False
        or value.get("services_stopped") is not True
        or value.get("receipt_path") != str(path)
        or value.get("receipt_sha256") != _sha256(_canonical_bytes(unsigned))
        or _read_optional(paths.staged_owner_approval) != owner_payload
        or _read_optional(paths.staged_external_iam) != iam_payload
    ):
        raise RuntimeError("authority bridge terminal receipt conflicts")
    return copy.deepcopy(dict(value))


def stage_authority(
    frame: Mapping[str, Any],
    *,
    paths: AuthorityPaths = AuthorityPaths(),
    now_unix: int | None = None,
    stopped_guard: Callable[[], None] = _require_stopped,
) -> Mapping[str, Any]:
    activation._require_root_linux()
    validated = validate_frame(frame, now_unix=now_unix)
    plan, scope, revision = _load_plan(str(validated["action"]))
    approval = OwnerApprovalReceipt.from_mapping(validated["owner_approval"])
    iam = ExternalIAMReceipt.from_mapping(validated["external_iam_receipt"])
    owner_payload = _canonical_bytes(approval.to_mapping())
    iam_payload = _canonical_bytes(iam.to_mapping())
    action = str(validated["action"])
    intent_root = paths.evidence_root / revision / plan.sha256 / validated["frame_sha256"]
    intent_path = intent_root / "intent.json"
    receipt_path = intent_root / "receipt.json"
    stopped_guard()
    with activation._host_activation_lock():
        stopped_guard()
        existing_terminal = _load_terminal_receipt(
            receipt_path,
            frame=validated,
            owner_payload=owner_payload,
            iam_payload=iam_payload,
            paths=paths,
        )
        if existing_terminal is not None:
            return existing_terminal
        current_owner = _read_optional(paths.staged_owner_approval)
        current_iam = _read_optional(paths.staged_external_iam)
        archive: Mapping[str, Any] | None = None
        expected_owner: bytes | None = None
        expected_iam: bytes | None = None
        if action == "stage-native-authority":
            if current_owner not in {None, owner_payload} or current_iam not in {
                None,
                iam_payload,
            }:
                raise RuntimeError("native staged authority is not pristine")
        else:
            previous_owner_sha = str(
                validated["previous_owner_approval_sha256"]
            )
            previous_iam_sha = str(
                validated["previous_external_iam_receipt_sha256"]
            )
            if (
                current_owner is not None
                and current_iam is not None
                and _sha256(current_owner) == previous_owner_sha
                and _sha256(current_iam) == previous_iam_sha
            ):
                archive = _archive_previous(
                    paths.evidence_root,
                    owner_raw=current_owner,
                    iam_raw=current_iam,
                )
                previous_owner_raw, previous_iam_raw = current_owner, current_iam
            else:
                archived = _load_archived_previous(
                    paths.evidence_root,
                    owner_sha256=previous_owner_sha,
                    iam_sha256=previous_iam_sha,
                )
                if archived is None:
                    raise RuntimeError(
                        "final authority replacement has no trusted native archive"
                    )
                previous_owner_raw, previous_iam_raw, archive = archived
            previous_approval = OwnerApprovalReceipt.from_mapping(
                activation._decode_strict_json(
                    previous_owner_raw,
                    label="previous staged owner approval",
                )
            )
            previous_iam = ExternalIAMReceipt.from_mapping(
                activation._decode_strict_json(
                    previous_iam_raw,
                    label="previous staged external IAM",
                )
            )
            if (
                previous_approval.value.get("scope") != "native_observation"
                or previous_approval.sha256 != previous_owner_sha
                or previous_iam.sha256 != previous_iam_sha
                or previous_iam.value.get("source_approval_sha256")
                != previous_approval.sha256
                or previous_approval.value.get("owner_subject_sha256")
                != approval.value.get("owner_subject_sha256")
                or previous_approval.value.get("approval_source_sha256")
                != approval.value.get("approval_source_sha256")
                or current_owner not in {previous_owner_raw, owner_payload}
                or current_iam not in {previous_iam_raw, iam_payload}
            ):
                raise PermissionError("native-to-final owner lineage drifted")
            expected_owner = previous_owner_raw
            expected_iam = previous_iam_raw
        intent_unsigned = {
            "schema": INTENT_SCHEMA,
            "action": action,
            "scope": scope,
            "revision": revision,
            "plan_sha256": plan.sha256,
            "frame_sha256": validated["frame_sha256"],
            "owner_approval_sha256": approval.sha256,
            "external_iam_receipt_sha256": iam.sha256,
            "previous_owner_approval_sha256": validated[
                "previous_owner_approval_sha256"
            ],
            "previous_external_iam_receipt_sha256": validated[
                "previous_external_iam_receipt_sha256"
            ],
            "staged_owner_approval_path": str(paths.staged_owner_approval),
            "staged_external_iam_path": str(paths.staged_external_iam),
        }
        intent = {
            **intent_unsigned,
            "intent_sha256": _sha256(_canonical_bytes(intent_unsigned)),
        }
        activation._ensure_root_directory(intent_root)
        activation._install_exact_bytes(
            intent_path,
            _canonical_bytes(intent),
            uid=0,
            gid=0,
            mode=0o400,
        )
        _atomic_replace_exact(
            paths.staged_owner_approval,
            expected_previous=expected_owner,
            payload=owner_payload,
        )
        _atomic_replace_exact(
            paths.staged_external_iam,
            expected_previous=expected_iam,
            payload=iam_payload,
        )
        stopped_guard()
        completed = int(time.time()) if now_unix is None else now_unix
        receipt_unsigned = {
            "schema": STAGE_RECEIPT_SCHEMA,
            "ok": True,
            "state": "authority_staged_services_stopped",
            "action": action,
            "scope": scope,
            "revision": revision,
            "plan_sha256": plan.sha256,
            "frame_sha256": validated["frame_sha256"],
            "owner_subject_sha256": validated["owner_subject_sha256"],
            "approval_source_sha256": validated["approval_source_sha256"],
            "owner_approval_sha256": approval.sha256,
            "external_iam_receipt_sha256": iam.sha256,
            "external_iam_policy_sha256": iam.policy_sha256,
            "previous_owner_approval_sha256": validated[
                "previous_owner_approval_sha256"
            ],
            "previous_external_iam_receipt_sha256": validated[
                "previous_external_iam_receipt_sha256"
            ],
            "archive": None if archive is None else copy.deepcopy(dict(archive)),
            "owner_staged_present": True,
            "external_iam_staged_present": True,
            "services_started": False,
            "services_stopped": True,
            "intent_path": str(intent_path),
            "intent_sha256": intent["intent_sha256"],
            "receipt_path": str(receipt_path),
            "completed_at_unix": completed,
        }
        receipt = {
            **receipt_unsigned,
            "receipt_sha256": _sha256(_canonical_bytes(receipt_unsigned)),
        }
        activation._install_exact_bytes(
            receipt_path,
            _canonical_bytes(receipt),
            uid=0,
            gid=0,
            mode=0o400,
        )
    stopped_guard()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage fixed stopped-only writer activation authority",
    )
    parser.add_argument(
        "action",
        choices=("stage-native-authority", "replace-final-authority"),
    )
    arguments = parser.parse_args(argv)
    frame = read_frame()
    if frame.get("action") != arguments.action:
        raise ValueError("owner authority command does not match frame")
    receipt = stage_authority(frame)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "AuthorityPaths",
    "EVIDENCE_ROOT",
    "FRAME_MAGIC",
    "FRAME_SCHEMA",
    "INTENT_SCHEMA",
    "MAX_FRAME_BYTES",
    "PINNED_APPROVAL_SOURCE_SHA256",
    "STAGE_RECEIPT_SCHEMA",
    "build_frame",
    "main",
    "read_frame",
    "stage_authority",
    "validate_frame",
]


if __name__ == "__main__":
    raise SystemExit(main())
