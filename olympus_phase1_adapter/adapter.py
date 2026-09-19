"""Synchronous, fake-only Phase 1 adapter with no registration surface."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import (
    PHASE0_RUNTIME_BINDING_DIGEST,
    ContractValidationError,
    JSONValue,
    Operation,
    canonical_bytes,
    canonical_digest,
    parse_operation,
    transient_rejection,
    validate_hex_digest,
)
from .receipt_store import (
    OLYMPUS_CHECKOUT,
    OwnerHandle,
    PersistenceError,
    ReceiptStore,
    RootSafetyError,
    StoreIndeterminate,
    StoreIndeterminateUnavailable,
    StorePreInvokeUnavailable,
    StoreSealedUnavailable,
    open_root_set,
)

_EXPECTED_OLYMPUS_ORIGIN = (
    OLYMPUS_CHECKOUT / "src" / "olympus_engine" / "__init__.py"
)
_CONTROL_FLOW = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)

_PHASE0_FILES = (
    (
        "pyproject.toml",
        "100644",
        304,
        "7393770eb30641478439da2a595be33bff034cb462a4fef5ff5fff0e5f525af7",
        "5ac8a57913e39cdbe6e35c66fc4a6f7f55aeebf4",
    ),
    (
        "schemas/v1/repo-analysis-authorization-v1.schema.json",
        "100644",
        1605,
        "4592e578f8f66922f1112d8c4886c0cd1c0c6634b9b064906903d6a919d84de6",
        "2637bf45bef2dcd9ddbe7ab24e63f054ef09ac95",
    ),
    (
        "schemas/v1/repo-analysis-comparison-v1.schema.json",
        "100644",
        1352,
        "b50046ea7711070df3140ae4e8e0ee2f461b47d5243e0d7e0735e44d49f2315d",
        "0126077d27c806194404a46aaede0367ed47cd1b",
    ),
    (
        "schemas/v1/repo-analysis-evidence-manifest-v1.schema.json",
        "100644",
        3803,
        "03de62db10054291bdf1c80462204d1da2198f283f28edae4a1fee2fbef43fab",
        "1798908e534b0228a90ef4562dcd8f417be4b13b",
    ),
    (
        "schemas/v1/repo-analysis-request-v1.schema.json",
        "100644",
        3627,
        "4568da179616ac76e83b7b81e9bfc37d81c9908d021d86d9c8db2a9b315e7780",
        "7fbb70bf7ddf3ecf52c707e6f998775d8e4bab5d",
    ),
    (
        "schemas/v1/repo-analysis-reviewer-result-v1.schema.json",
        "100644",
        1594,
        "b91ace11fc8cbecd2e4b1c7c038cfe59c29a51bce6a2b4c9a07529ed530c1556",
        "9bf4676ccd12e9e985bdfcb3688d8198876b1ba3",
    ),
    (
        "schemas/v1/repo-analysis-terminal-report-v1.schema.json",
        "100644",
        1601,
        "68d458043ce4f9e400ef0240197dc89502069a7bc57a95d5cc060a5327c76ace",
        "fc0cffacf7ef7100d9a9147f56cc7c6599cd3d16",
    ),
    (
        "schemas/v1/repo-analysis-worker-result-v1.schema.json",
        "100644",
        1863,
        "006a912dd8cde1164115dab0cd6baa09b0cc88d6e90abb61412f0d595e2d738c",
        "239ab07b907b269674042a172d793aa663444b8d",
    ),
    (
        "src/olympus_engine/__init__.py",
        "100644",
        2370,
        "ce1720a097ae19543f8bf30d6766fa31c5eec7d3753c93f90305eea431b9b1ba",
        "ecd0247e97111f96d2becf2385a8a34afa55dc1b",
    ),
    (
        "src/olympus_engine/authorization.py",
        "100644",
        3080,
        "d1a35cbde6214c4a7c67039d7c5592955a17eca78757abde1e5bf7301c4dcd1e",
        "3effe27bb1ba8a217e47e33876fdbdf4fd744a80",
    ),
    (
        "src/olympus_engine/canonical.py",
        "100644",
        5727,
        "7dd308bcadf615e1f45cbc602bd3b0dce44f1ebdfbcf30c7a9769a6ef2779697",
        "1f7450296685c1a792273d5eb39dc208d3980117",
    ),
    (
        "src/olympus_engine/comparator.py",
        "100644",
        6506,
        "ae8039ae77c9075561ad98bc0ee6bb9f47bbeef8e4a7757d957b5c78cc35ea7b",
        "e982e22ab6b96a27c11ae8ebbbb2ee0b571a3508",
    ),
    (
        "src/olympus_engine/contracts.py",
        "100644",
        36584,
        "87170fa2ee3099009e7327ec32f4c2d26287da4b94e28ea78336729347205955",
        "7740a6101cd95ff67740abfd9a0587cde0551b4e",
    ),
    (
        "src/olympus_engine/evidence.py",
        "100644",
        41545,
        "db496c33bd6bd248cc322fcea4f39cfe57ffdebf945fc69e0781f010f9d85d32",
        "b7a87f18afb779d7e805bec955d6b15008272c42",
    ),
    (
        "src/olympus_engine/packetizer.py",
        "100644",
        15645,
        "3ace4275c4a68f329f7b1657a398119ddf068c826c68b14d1445301c77b18489",
        "47f0e1ba86fefa56796a1c1b9830b69d21214c22",
    ),
    (
        "src/olympus_engine/renderer.py",
        "100644",
        3314,
        "19db15a2bdcd43605358eb901439ada5338f64a729aab65f49b4b3a2bea74b99",
        "10c0815a7d6af409bfaa58a06bb549c552f81cd4",
    ),
    (
        "src/olympus_engine/safe_fs.py",
        "100644",
        31624,
        "850b822b5c0968b7d17e449ee05a84be6f170915d9cd68b47818b0df563e36bc",
        "8216434a0f4a1aaeb74f59989ed4162f6e24dd44",
    ),
    (
        "src/olympus_engine/transports.py",
        "100644",
        2119,
        "4c1a92ae90f530e82f1225cd381f3d40b151d3c6693dca9135515929d58d53fe",
        "b0d6453814fab7f2b4831614b9d032edc7b001c2",
    ),
    (
        "src/olympus_engine/workflow.py",
        "100644",
        18073,
        "e31aa321ef1235d687df36a973d65a97c96697dd4e2ec92fe89c6abdf14fa120",
        "7947e9b6f7908759f8dbfa0cfb36657fab4d1526",
    ),
)


class IndeterminateReceiptUnavailable(RuntimeError):
    """A consumed attempt has no safely returnable sealed receipt."""

    def __init__(
        self, *, idempotency_key_digest: str, operation_digest: str
    ) -> None:
        self.idempotency_key_digest = validate_hex_digest(
            idempotency_key_digest, "idempotency_key_digest"
        )
        self.operation_digest = validate_hex_digest(
            operation_digest, "operation_digest"
        )
        self.receipt_state = "INDETERMINATE_NO_RETRY"
        self.reason_code = "PERSISTENCE_CORRUPTION"
        self.automatic_engine_retry_permitted = False
        super().__init__(
            "Phase 1 attempt is indeterminate and no sealed receipt is "
            "available; automatic retry is forbidden"
        )


class PreInvokeReceiptUnavailable(RuntimeError):
    """Ownership exists, invocation did not occur, and no receipt is available."""

    def __init__(
        self, *, idempotency_key_digest: str, operation_digest: str
    ) -> None:
        self.idempotency_key_digest = validate_hex_digest(
            idempotency_key_digest, "idempotency_key_digest"
        )
        self.operation_digest = validate_hex_digest(
            operation_digest, "operation_digest"
        )
        self.receipt_state = "REJECTED_PRE_INVOKE"
        self.reason_code = "PERSISTENCE_CORRUPTION"
        self.engine_invocation_occurred = False
        self.automatic_engine_retry_permitted = False
        super().__init__(
            "Phase 1 invocation did not occur, but no sealed rejection receipt "
            "is available; automatic retry is forbidden"
        )


class SealedReceiptUnavailable(RuntimeError):
    """A durably sealed receipt cannot be safely returned."""

    def __init__(
        self,
        *,
        idempotency_key_digest: str,
        operation_digest: str,
        receipt_digest: str,
        receipt_state: Literal[
            "REJECTED_PRE_INVOKE", "ENGINE_TERMINAL", "INDETERMINATE_NO_RETRY"
        ],
    ) -> None:
        self.idempotency_key_digest = validate_hex_digest(
            idempotency_key_digest, "idempotency_key_digest"
        )
        self.operation_digest = validate_hex_digest(
            operation_digest, "operation_digest"
        )
        self.receipt_digest = validate_hex_digest(
            receipt_digest, "receipt_digest"
        )
        if receipt_state not in {
            "REJECTED_PRE_INVOKE",
            "ENGINE_TERMINAL",
            "INDETERMINATE_NO_RETRY",
        }:
            raise ValueError("invalid sealed receipt state")
        self.receipt_state = receipt_state
        self.reason_code = "PERSISTENCE_CORRUPTION"
        self.receipt_was_durably_sealed = True
        self.automatic_engine_retry_permitted = False
        super().__init__(
            "A Phase 1 receipt was durably sealed but cannot be safely "
            "returned; automatic retry is forbidden"
        )


class _FreshRejection(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class _EvidenceFailure(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class _Phase0API:
    FakePairATransport: type
    FakePairBTransport: type
    OlympusWorkflow: type
    canonical_bytes: Any
    canonical_digest: Any
    strict_loads: Any
    verify_evidence_package: Any


def _phase0_manifest() -> list[dict[str, JSONValue]]:
    return [
        {
            "path": path,
            "mode": mode,
            "raw_bytes": size,
            "raw_sha256": digest,
            "git_blob": blob,
        }
        for path, mode, size, digest, blob in _PHASE0_FILES
    ]


def _read_frozen_phase0_file(
    relative: str, expected_size: int, expected_digest: str
) -> None:
    path = OLYMPUS_CHECKOUT / relative
    try:
        entry = os.lstat(path)
        if (
            not stat.S_ISREG(entry.st_mode)
            or stat.S_ISLNK(entry.st_mode)
            or entry.st_uid != os.geteuid()
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o644
            or entry.st_size != expected_size
        ):
            raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
            os, "O_CLOEXEC", 0
        )
        fd = os.open(path, flags)
        try:
            held = os.fstat(fd)
            hasher = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(fd, 65_536)
                if not chunk:
                    break
                total += len(chunk)
                hasher.update(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except _FreshRejection:
        raise
    except OSError as exc:
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH") from exc
    if (
        (entry.st_dev, entry.st_ino)
        != (held.st_dev, held.st_ino)
        or (held.st_dev, held.st_ino) != (after.st_dev, after.st_ino)
        or total != expected_size
        or hasher.hexdigest() != expected_digest
    ):
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH")


def _object_origin(value: Any) -> Path | None:
    try:
        source = inspect.getsourcefile(value)
        return None if source is None else Path(source).resolve(strict=True)
    except (OSError, TypeError, RuntimeError):
        return None


def _verify_phase0_provenance() -> _Phase0API:
    if canonical_digest("phase1-phase0-binding", _phase0_manifest()) != (
        PHASE0_RUNTIME_BINDING_DIGEST
    ):
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH")
    for relative, _mode, size, digest, _blob in _PHASE0_FILES:
        _read_frozen_phase0_file(relative, size, digest)

    existing = sys.modules.get("olympus_engine")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        try:
            resolved = None if origin is None else Path(origin).resolve(strict=True)
        except (OSError, RuntimeError):
            resolved = None
        if resolved != _EXPECTED_OLYMPUS_ORIGIN:
            raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH")
    try:
        module = (
            existing
            if existing is not None
            else importlib.import_module("olympus_engine")
        )
    except Exception as exc:
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH") from exc
    try:
        module_origin = Path(module.__file__).resolve(strict=True)
    except (AttributeError, OSError, RuntimeError) as exc:
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH") from exc
    if module_origin != _EXPECTED_OLYMPUS_ORIGIN:
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH")

    names = (
        "FakePairATransport",
        "FakePairBTransport",
        "OlympusWorkflow",
        "canonical_bytes",
        "canonical_digest",
        "strict_loads",
        "verify_evidence_package",
    )
    try:
        values = {name: getattr(module, name) for name in names}
    except AttributeError as exc:
        raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH") from exc
    expected_origins = {
        "FakePairATransport": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "transports.py",
        "FakePairBTransport": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "transports.py",
        "OlympusWorkflow": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "workflow.py",
        "canonical_bytes": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "canonical.py",
        "canonical_digest": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "canonical.py",
        "strict_loads": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "canonical.py",
        "verify_evidence_package": OLYMPUS_CHECKOUT
        / "src"
        / "olympus_engine"
        / "evidence.py",
    }
    for name, expected in expected_origins.items():
        if _object_origin(values[name]) != expected:
            raise _FreshRejection("PHASE0_PROVENANCE_MISMATCH")
    return _Phase0API(**values)


def _validate_fakes(api: _Phase0API, pair_a: Any, pair_b: Any) -> None:
    if type(pair_a) is not api.FakePairATransport:
        raise _FreshRejection("FAKE_TRANSPORT_INVALID")
    if type(pair_b) is not api.FakePairBTransport:
        raise _FreshRejection("FAKE_TRANSPORT_INVALID")
    if (
        type(pair_a.call_count) is not int
        or pair_a.call_count != 0
        or type(pair_a.received_packets) is not list
        or pair_a.received_packets
        or type(pair_b.call_count) is not int
        or pair_b.call_count != 0
        or type(pair_b.received_packets) is not list
        or pair_b.received_packets
        or type(pair_b.received_worker_results) is not list
        or pair_b.received_worker_results
    ):
        raise _FreshRejection("FAKE_TRANSPORT_INVALID")


def _preflight(pair_a: Any, pair_b: Any) -> _Phase0API:
    api = _verify_phase0_provenance()
    _validate_fakes(api, pair_a, pair_b)
    return api


def _translate_store_error(error: BaseException) -> BaseException:
    if isinstance(error, StoreSealedUnavailable):
        return SealedReceiptUnavailable(
            idempotency_key_digest=error.idempotency_key_digest,
            operation_digest=error.operation_digest,
            receipt_digest=error.receipt_digest,
            receipt_state=error.receipt_state,
        )
    if isinstance(error, StorePreInvokeUnavailable):
        return PreInvokeReceiptUnavailable(
            idempotency_key_digest=error.idempotency_key_digest,
            operation_digest=error.operation_digest,
        )
    if isinstance(
        error, (StoreIndeterminate, StoreIndeterminateUnavailable)
    ):
        return IndeterminateReceiptUnavailable(
            idempotency_key_digest=error.idempotency_key_digest,
            operation_digest=error.operation_digest,
        )
    return error


def _refresh_or_indeterminate(owner: OwnerHandle) -> str:
    try:
        owner.refresh_active()
    except StoreIndeterminate as exc:
        raise IndeterminateReceiptUnavailable(
            idempotency_key_digest=exc.idempotency_key_digest,
            operation_digest=exc.operation_digest,
        ) from None
    return str(owner.last_record["state"])


def _seal_preinvoke(owner: OwnerHandle, reason_code: str) -> bytes:
    state = _refresh_or_indeterminate(owner)
    if state == "OWNERSHIP_ACQUIRED":
        try:
            owner.append("PREINVOKE_REJECTED", reason_code=reason_code)
        except Exception:
            state = _refresh_or_indeterminate(owner)
            if state != "PREINVOKE_REJECTED":
                if state in {
                    "INVOCATION_CLAIMED",
                    "ENGINE_EVIDENCE_VERIFIED",
                    "INDETERMINATE_NO_RETRY",
                }:
                    return _seal_indeterminate(owner, "PERSISTENCE_CORRUPTION")
                raise PreInvokeReceiptUnavailable(
                    idempotency_key_digest=owner.operation.idempotency_key_digest,
                    operation_digest=owner.operation.operation_digest,
                ) from None
    elif state != "PREINVOKE_REJECTED":
        if state in {
            "INVOCATION_CLAIMED",
            "ENGINE_EVIDENCE_VERIFIED",
            "INDETERMINATE_NO_RETRY",
        }:
            return _seal_indeterminate(owner, "PERSISTENCE_CORRUPTION")
        raise PreInvokeReceiptUnavailable(
            idempotency_key_digest=owner.operation.idempotency_key_digest,
            operation_digest=owner.operation.operation_digest,
        ) from None
    persisted_reason = str(owner.last_record["reason_code"])
    try:
        return owner.seal(
            receipt_state="REJECTED_PRE_INVOKE",
            reason_code=persisted_reason,
        )
    except StoreSealedUnavailable as exc:
        raise _translate_store_error(exc) from None
    except Exception:
        if owner.sealed_raw is not None:
            owner.best_effort_finalize()
            return owner.sealed_raw
        raise PreInvokeReceiptUnavailable(
            idempotency_key_digest=owner.operation.idempotency_key_digest,
            operation_digest=owner.operation.operation_digest,
        ) from None


def _seal_indeterminate(owner: OwnerHandle, reason_code: str) -> bytes:
    if owner.sealed_raw is not None:
        owner.best_effort_finalize()
        return owner.sealed_raw
    state = _refresh_or_indeterminate(owner)
    if state in {"INVOCATION_CLAIMED", "ENGINE_EVIDENCE_VERIFIED"}:
        try:
            owner.append("INDETERMINATE_NO_RETRY", reason_code=reason_code)
        except Exception:
            state = _refresh_or_indeterminate(owner)
            if state != "INDETERMINATE_NO_RETRY":
                raise IndeterminateReceiptUnavailable(
                    idempotency_key_digest=owner.operation.idempotency_key_digest,
                    operation_digest=owner.operation.operation_digest,
                ) from None
    elif state != "INDETERMINATE_NO_RETRY":
        raise IndeterminateReceiptUnavailable(
            idempotency_key_digest=owner.operation.idempotency_key_digest,
            operation_digest=owner.operation.operation_digest,
        ) from None
    persisted_reason = str(owner.last_record["reason_code"])
    try:
        return owner.seal(
            receipt_state="INDETERMINATE_NO_RETRY",
            reason_code=persisted_reason,
        )
    except StoreSealedUnavailable as exc:
        raise _translate_store_error(exc) from None
    except Exception:
        if owner.sealed_raw is not None:
            owner.best_effort_finalize()
            return owner.sealed_raw
        raise IndeterminateReceiptUnavailable(
            idempotency_key_digest=owner.operation.idempotency_key_digest,
            operation_digest=owner.operation.operation_digest,
        ) from None


def _best_effort_control_flow(
    owner: OwnerHandle,
    *,
    invocation_claimed: bool,
) -> None:
    try:
        if owner.sealed_raw is not None:
            owner.best_effort_finalize()
        elif invocation_claimed:
            _seal_indeterminate(owner, "CANCELLED_AFTER_INVOCATION_CLAIM")
        else:
            _seal_preinvoke(owner, "CANCELLED_BEFORE_INVOCATION_CLAIM")
    except BaseException:
        pass


def _verify_phase0_result(
    *,
    api: _Phase0API,
    operation: Operation,
    owner: OwnerHandle,
    result: Any,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    destination = owner.evidence_destination
    manifest_object = getattr(result, "evidence_manifest", None)
    if manifest_object is None or not destination.is_dir():
        raise _EvidenceFailure("EVIDENCE_MISSING")
    try:
        request_value = result.request.to_dict()
        terminal = result.terminal_report.to_dict()
        manifest = manifest_object.to_dict()
        verified = api.verify_evidence_package(destination)
        verified_manifest = verified.to_dict()
        if request_value != operation.payload:
            raise ContractValidationError("Phase 0 request result mismatch")
        request_bytes = api.canonical_bytes(operation.payload)
        if api.strict_loads(request_bytes) != operation.payload:
            raise ContractValidationError("Phase 0 request canonical mismatch")
        if api.canonical_digest("request", request_value) != (
            operation.payload_request_digest
        ):
            raise ContractValidationError("Phase 0 request digest mismatch")
        if verified_manifest != manifest:
            raise ContractValidationError("Phase 0 verified manifest mismatch")
        terminal_bytes = api.canonical_bytes(terminal)
        terminal_digest = api.canonical_digest("terminal-report", terminal)
        if terminal_digest != result.terminal_report.digest:
            raise ContractValidationError("Phase 0 terminal digest mismatch")
        manifest_digest = api.canonical_digest("evidence-manifest", manifest)
        if manifest_digest != manifest_object.digest:
            raise ContractValidationError("Phase 0 manifest digest mismatch")
        if (
            terminal.get("request_id") != operation.payload_request_id
            or verified_manifest.get("request_id") != operation.payload_request_id
        ):
            raise ContractValidationError("Phase 0 manifest request mismatch")
        if verified_manifest.get("terminal_report_digest") != terminal_digest:
            raise ContractValidationError("Phase 0 manifest terminal mismatch")
        artifacts = verified_manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ContractValidationError("Phase 0 manifest artifacts mismatch")
        artifacts_by_name = {
            artifact.get("name"): artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
        }
        request_artifact = artifacts_by_name.get("01-request.json")
        terminal_artifact = artifacts_by_name.get("07-terminal-report.json")
        if not isinstance(request_artifact, dict) or (
            request_artifact.get("digest") != operation.payload_request_digest
            or request_artifact.get("size") != len(request_bytes)
        ):
            raise ContractValidationError("Phase 0 request artifact mismatch")
        if not isinstance(terminal_artifact, dict) or (
            terminal_artifact.get("digest") != terminal_digest
            or terminal_artifact.get("size") != len(terminal_bytes)
        ):
            raise ContractValidationError("Phase 0 terminal artifact mismatch")
        identity_fields = (
            "request_id",
            "packetization_status",
            "pair_a_status",
            "pair_b_status",
            "repository_postcheck_status",
            "artifacts",
            "events",
            "repository_pre_digest",
            "repository_post_digest",
            "terminal_report_digest",
        )
        expected_package_id = api.canonical_digest(
            "evidence-package",
            {field: verified_manifest.get(field) for field in identity_fields},
        )
        if verified_manifest.get("package_id") != expected_package_id:
            raise ContractValidationError("Phase 0 evidence package mismatch")
        # Local canonicalization must be byte-identical to the frozen engine.
        if terminal_bytes != canonical_bytes(terminal):
            raise ContractValidationError("terminal canonicalization drift")
        if api.canonical_bytes(manifest) != canonical_bytes(manifest):
            raise ContractValidationError("manifest canonicalization drift")
    except _CONTROL_FLOW:
        raise
    except Exception as exc:
        raise _EvidenceFailure("EVIDENCE_VERIFICATION_FAILED") from exc
    return terminal, manifest


def _execute_first_owner(
    *,
    owner: OwnerHandle,
    pair_a: Any,
    pair_b: Any,
) -> bytes:
    api = owner.preflight_result
    if not isinstance(api, _Phase0API):
        return _seal_preinvoke(owner, "PERSISTENCE_CORRUPTION")
    invocation_claimed = False
    try:
        try:
            owner.roots.revalidate_all()
            owner.append("INVOCATION_CLAIMED")
            invocation_claimed = True
        except _CONTROL_FLOW:
            try:
                owner.refresh_active()
                invocation_claimed = str(owner.last_record["state"]) in {
                    "INVOCATION_CLAIMED",
                    "ENGINE_EVIDENCE_VERIFIED",
                    "INDETERMINATE_NO_RETRY",
                }
            except BaseException:
                pass
            raise
        except Exception:
            return _seal_preinvoke(owner, "PERSISTENCE_CORRUPTION")

        request_bytes = api.canonical_bytes(owner.operation.payload)
        try:
            workflow = api.OlympusWorkflow(
                request_json=request_bytes,
                repository_root=owner.roots.repository.path,
                pair_a=pair_a,
                pair_b=pair_b,
                evidence_destination=owner.evidence_destination,
            )
        except _CONTROL_FLOW:
            raise
        except Exception:
            return _seal_indeterminate(owner, "WORKFLOW_CONSTRUCTION_FAILED")
        try:
            result = workflow.run()
        except _CONTROL_FLOW:
            raise
        except Exception:
            return _seal_indeterminate(
                owner, "ENGINE_EXCEPTION_WITHOUT_SEALED_RECEIPT"
            )
        try:
            terminal, manifest = _verify_phase0_result(
                api=api,
                operation=owner.operation,
                owner=owner,
                result=result,
            )
        except _CONTROL_FLOW:
            raise
        except _EvidenceFailure as exc:
            return _seal_indeterminate(owner, exc.reason_code)

        terminal_digest = canonical_digest("terminal-report", terminal)
        manifest_digest = canonical_digest("evidence-manifest", manifest)
        try:
            owner.roots.revalidate_all()
            owner.append(
                "ENGINE_EVIDENCE_VERIFIED",
                reason_code="PHASE0_TERMINAL",
                phase0_terminal_report_digest=terminal_digest,
                phase0_evidence_manifest_digest=manifest_digest,
                phase0_evidence_directory_name=owner.evidence_destination.name,
            )
            return owner.seal(
                receipt_state="ENGINE_TERMINAL",
                reason_code="PHASE0_TERMINAL",
                phase0_terminal_report=terminal,
                phase0_evidence_manifest=manifest,
            )
        except _CONTROL_FLOW:
            raise
        except StoreSealedUnavailable as exc:
            raise _translate_store_error(exc) from None
        except Exception:
            return _seal_indeterminate(owner, "PERSISTENCE_CORRUPTION")
    except _CONTROL_FLOW:
        _best_effort_control_flow(
            owner, invocation_claimed=invocation_claimed
        )
        raise


def execute_operation(
    envelope_json: str | bytes,
    *,
    repository_root: str | Path,
    receipt_root: str | Path,
    evidence_root: str | Path,
    pair_a: olympus_engine.FakePairATransport,
    pair_b: olympus_engine.FakePairBTransport,
) -> bytes:
    """Execute or replay exactly one frozen fake-only Phase 1 operation."""

    try:
        operation = parse_operation(envelope_json)
    except ContractValidationError as exc:
        reason = getattr(exc, "reason_code", "ENVELOPE_INVALID_JSON")
        return transient_rejection(reason)

    try:
        roots = open_root_set(
            repository_root=repository_root,
            receipt_root=receipt_root,
            evidence_root=evidence_root,
        )
    except RootSafetyError as exc:
        return transient_rejection(exc.reason_code)

    with roots:
        store = ReceiptStore(roots)
        try:
            claim = store.claim(
                operation,
                fresh_preflight=lambda: _preflight(pair_a, pair_b),
            )
        except _FreshRejection as exc:
            return transient_rejection(exc.reason_code)
        except RootSafetyError as exc:
            return transient_rejection(exc.reason_code)
        except (StoreIndeterminate, StoreIndeterminateUnavailable) as exc:
            raise _translate_store_error(exc) from None
        except StorePreInvokeUnavailable as exc:
            raise _translate_store_error(exc) from None
        except StoreSealedUnavailable as exc:
            raise _translate_store_error(exc) from None
        except _CONTROL_FLOW:
            raise
        except (PersistenceError, OSError):
            return transient_rejection("PERSISTENCE_CORRUPTION")

        if claim.response is not None:
            return claim.response
        owner = claim.owner
        if owner is None:
            raise AssertionError("claim returned no outcome")
        with owner:
            return _execute_first_owner(
                owner=owner,
                pair_a=pair_a,
                pair_b=pair_b,
            )
