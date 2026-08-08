from __future__ import annotations

import ast
import asyncio
import copy
import dataclasses
import hashlib
import json
import multiprocessing
import os
import socket
import stat
import subprocess
import threading
import tomllib
import types
from pathlib import Path
from typing import Literal, get_type_hints

import pytest

from olympus_engine.transports import FakePairATransport, FakePairBTransport
from olympus_engine.workflow import OlympusWorkflow

from olympus_phase1_adapter import adapter
from olympus_phase1_adapter.adapter import (
    IndeterminateReceiptUnavailable,
    PreInvokeReceiptUnavailable,
    SealedReceiptUnavailable,
    execute_operation,
)
from olympus_phase1_adapter.contracts import (
    ENGINE_CONTRACT_DIGEST,
    GATE1_PACKET_SHA256,
    GATE2_PACKET_SHA256,
    PHASE0_RUNTIME_BINDING_DIGEST,
    PHASE1_CONTRACT_DIGEST,
    ContractValidationError,
    build_record,
    canonical_bytes,
    canonical_digest,
    digest_vectors,
    parse_operation,
    parse_record_bytes,
    sealed_receipt,
    strict_loads,
    transient_conflict,
    transient_rejection,
    validate_receipt,
    verify_frozen_schemas,
)
from olympus_phase1_adapter.receipt_store import OwnerHandle, PersistenceError

from conftest import (
    operation_bytes,
    operation_value,
    request_value,
    valid_transports,
)


def _call(
    roots: dict[str, Path],
    envelope: bytes,
    pair_a: object,
    pair_b: object,
) -> bytes:
    return execute_operation(
        envelope,
        repository_root=roots["repository"],
        receipt_root=roots["receipt"],
        evidence_root=roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )


def _tree_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    snapshot: list[tuple[object, ...]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: item.as_posix()):
        entry = path.lstat()
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if stat.S_ISREG(entry.st_mode)
            else None
        )
        snapshot.append(
            (
                path.relative_to(root).as_posix(),
                stat.S_IFMT(entry.st_mode),
                stat.S_IMODE(entry.st_mode),
                entry.st_dev,
                entry.st_ino,
                entry.st_size,
                entry.st_mtime_ns,
                digest,
            )
        )
    return tuple(snapshot)


class _DictObject:
    def __init__(self, value: dict[str, object], *, digest: str | None = None) -> None:
        self._value = copy.deepcopy(value)
        self.digest = digest

    def to_dict(self) -> dict[str, object]:
        return copy.deepcopy(self._value)


def _over_depth_json() -> bytes:
    return b"[" * 66 + b"0" + b"]" * 66


def _wrong_schema_operation() -> bytes:
    value = operation_value()
    value["schema_version"] = "olympus.hermes.phase1.operation/v2"
    return canonical_bytes(value)


def _invalid_phase0_request_operation() -> bytes:
    payload = request_value()
    payload["requested_paths"] = []
    return operation_bytes(payload=payload)


def _forbidden_payload_field_operation() -> bytes:
    payload = request_value()
    payload["provider"] = "forbidden"
    return operation_bytes(payload=payload)


def test_t01_valid_allow_fixture_is_one_shot_and_engine_terminal(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_preflight = adapter._preflight
    original_pair_a = FakePairATransport.analyze
    original_pair_b = FakePairBTransport.review

    def pair_a_once(self: FakePairATransport, packet: bytes) -> bytes:
        events.append("pair_a")
        return original_pair_a(self, packet)

    def pair_b_once(
        self: FakePairBTransport, packet: bytes, worker: bytes
    ) -> bytes:
        events.append("pair_b")
        return original_pair_b(self, packet, worker)

    def instrumented_preflight(pair_a_value: object, pair_b_value: object):
        api = original_preflight(pair_a_value, pair_b_value)
        workflow_type = api.OlympusWorkflow
        verifier = api.verify_evidence_package

        class InstrumentedWorkflow:
            def __init__(self, **kwargs: object) -> None:
                events.append("construct")
                self._inner = workflow_type(**kwargs)

            def run(self):
                events.append("run")
                return self._inner.run()

        def verify_once(destination: Path):
            events.append("verify")
            return verifier(destination)

        return dataclasses.replace(
            api,
            OlympusWorkflow=InstrumentedWorkflow,
            verify_evidence_package=verify_once,
        )

    monkeypatch.setattr(FakePairATransport, "analyze", pair_a_once)
    monkeypatch.setattr(FakePairBTransport, "review", pair_b_once)
    monkeypatch.setattr(adapter, "_preflight", instrumented_preflight)
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    raw = _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    receipt = strict_loads(raw)
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert receipt["durability"] == "SEALED"
    assert receipt["reason_codes"] == ["PHASE0_TERMINAL"]
    assert receipt["phase0_terminal_report"]["terminal_state"] == "COMPLETED"
    assert pair_a.call_count == pair_b.call_count == 1
    assert events == ["construct", "run", "pair_a", "pair_b", "verify"]
    key = parse_operation(operation_bytes()).idempotency_key_digest
    key_root = phase1_roots["receipt"] / "v1" / key[:2] / key
    assert (key_root / "receipt.json").read_bytes() == raw
    assert (stat_mode(key_root) == 0o500)
    assert stat_mode(key_root / "records") == 0o500
    assert stat_mode(key_root / "receipt.json") == 0o400
    assert all(
        stat_mode(path) == 0o400
        for path in (key_root / "records").iterdir()
    )


def stat_mode(path: Path) -> int:
    return path.stat(follow_symlinks=False).st_mode & 0o7777


def test_t02_native_denial_is_preserved_without_pair_calls(
    phase1_roots: dict[str, Path],
) -> None:
    payload = request_value()
    payload["controls"]["offline"] = False
    pair_a = FakePairATransport(b"{}")
    pair_b = FakePairBTransport(b"{}")
    raw = _call(
        phase1_roots,
        operation_bytes(payload=payload),
        pair_a,
        pair_b,
    )
    receipt = strict_loads(raw)
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert receipt["phase0_terminal_report"]["terminal_state"] == "DENIED"
    assert pair_a.call_count == pair_b.call_count == 0


@pytest.mark.parametrize(
    ("variant", "native_reason"),
    [
        ("timeout", "PAIR_A_TIMEOUT"),
        ("malformed", "PAIR_A_MALFORMED"),
        ("transport", "PAIR_A_TRANSPORT_FAILURE"),
    ],
)
def test_t03_pair_a_failures_remain_native_phase0_terminal_results(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    native_reason: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    if variant == "timeout":
        pair_a.timeout = True
    elif variant == "malformed":
        pair_a.response = b"{}"
    else:
        def fail_transport(
            _self: FakePairATransport, _packet: bytes
        ) -> bytes:
            raise RuntimeError("injected Pair A transport failure")

        monkeypatch.setattr(FakePairATransport, "analyze", fail_transport)
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert receipt["reason_codes"] == ["PHASE0_TERMINAL"]
    assert receipt["phase0_terminal_report"]["terminal_state"] == "FAILED"
    assert native_reason in receipt["phase0_terminal_report"]["reason_codes"]
    assert pair_b.call_count == 0
    assert pair_a.call_count == (0 if variant == "transport" else 1)


@pytest.mark.parametrize(
    ("variant", "native_reason"),
    [
        ("timeout", "PAIR_B_TIMEOUT"),
        ("malformed", "PAIR_B_MALFORMED"),
        ("transport", "PAIR_B_TRANSPORT_FAILURE"),
    ],
)
def test_t04_pair_b_failures_remain_native_phase0_terminal_results(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    native_reason: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    if variant == "timeout":
        pair_b.timeout = True
    elif variant == "malformed":
        pair_b.response = b"{}"
    else:
        def fail_transport(
            _self: FakePairBTransport,
            _packet: bytes,
            _worker: bytes,
        ) -> bytes:
            raise RuntimeError("injected Pair B transport failure")

        monkeypatch.setattr(FakePairBTransport, "review", fail_transport)
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert receipt["reason_codes"] == ["PHASE0_TERMINAL"]
    assert receipt["phase0_terminal_report"]["terminal_state"] == "FAILED"
    assert native_reason in receipt["phase0_terminal_report"]["reason_codes"]
    assert pair_a.call_count == 1
    assert pair_b.call_count == (0 if variant == "transport" else 1)


def test_t05_sealed_replay_is_byte_identical_and_does_not_consult_fakes(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    envelope = operation_bytes()
    first = _call(phase1_roots, envelope, pair_a, pair_b)
    operation = parse_operation(envelope)
    key_root = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    receipt_before = _tree_snapshot(key_root)
    evidence_before = _tree_snapshot(phase1_roots["evidence"])

    def forbidden_runtime(*_args: object, **_kwargs: object):
        pytest.fail("replay attempted Phase 0 construction or execution")

    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda _name: pytest.fail("replay initiated an Olympus import"),
    )
    monkeypatch.setattr(
        adapter,
        "_preflight",
        lambda *_args: pytest.fail("replay consulted fresh-owner preflight"),
    )
    monkeypatch.setattr(OlympusWorkflow, "__init__", forbidden_runtime)
    monkeypatch.setattr(OlympusWorkflow, "run", forbidden_runtime)
    second = _call(phase1_roots, envelope, object(), object())
    assert second == first
    assert pair_a.call_count == pair_b.call_count == 1
    assert (key_root / "receipt.json").read_bytes() == first
    assert _tree_snapshot(key_root) == receipt_before
    assert _tree_snapshot(phase1_roots["evidence"]) == evidence_before


def test_t06_same_key_different_operation_is_immutable_anchor_conflict(
    phase1_roots: dict[str, Path],
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    first_envelope = operation_bytes()
    _call(phase1_roots, first_envelope, pair_a, pair_b)
    changed = operation_value(correlation_id="corr-phase1-other")
    raw = _call(
        phase1_roots,
        canonical_bytes(changed),
        object(),
        object(),
    )
    receipt = strict_loads(raw)
    first_operation = parse_operation(first_envelope)
    current_operation = parse_operation(canonical_bytes(changed))
    assert receipt["receipt_state"] == "IDEMPOTENCY_CONFLICT"
    assert receipt["durability"] == "TRANSIENT"
    assert receipt["operation_digest"] == current_operation.operation_digest
    assert receipt["bound_operation_digest"] == first_operation.operation_digest
    assert receipt["bound_operation_digest"] != receipt["operation_digest"]
    key_root = (
        phase1_roots["receipt"]
        / "v1"
        / first_operation.idempotency_key_digest[:2]
        / first_operation.idempotency_key_digest
    )
    anchor = parse_record_bytes(
        (key_root / "records" / "000001-OWNERSHIP_ACQUIRED.json").read_bytes()
    )
    assert anchor["sequence"] == 1
    assert anchor["state"] == "OWNERSHIP_ACQUIRED"
    assert receipt["ownership_record_digest"] == anchor["record_digest"]
    assert receipt["bound_operation_digest"] == anchor["operation_digest"]
    assert pair_a.call_count == pair_b.call_count == 1


def test_t06_different_operation_conflict_validates_current_and_anchor_bindings(
) -> None:
    bound_operation = parse_operation(operation_bytes())
    current_operation = parse_operation(
        operation_bytes(
            correlation_id="corr-phase1-other",
            payload=request_value(request_id="req-phase1-other"),
        )
    )
    anchor, _ = build_record(
        bound_operation,
        sequence=1,
        state="OWNERSHIP_ACQUIRED",
        previous_record_digest=None,
    )
    receipt = strict_loads(
        transient_conflict(
            current_operation,
            bound_operation_digest=bound_operation.operation_digest,
            ownership_record_digest=str(anchor["record_digest"]),
        )
    )

    assert validate_receipt(
        receipt, chain=[anchor], operation=current_operation
    ) == receipt
    assert receipt["correlation_id"] == current_operation.correlation_id
    assert receipt["payload_request_id"] == current_operation.payload_request_id
    assert receipt["payload_request_digest"] == current_operation.payload_request_digest
    assert receipt["bound_operation_digest"] == anchor["operation_digest"]
    assert receipt["ownership_record_digest"] == anchor["record_digest"]


def test_t06_sealed_receipt_retains_full_anchor_identity_validation() -> None:
    operation = parse_operation(operation_bytes())
    anchor, _ = build_record(
        operation,
        sequence=1,
        state="OWNERSHIP_ACQUIRED",
        previous_record_digest=None,
    )
    predecessor, _ = build_record(
        operation,
        sequence=2,
        state="PREINVOKE_REJECTED",
        previous_record_digest=str(anchor["record_digest"]),
        reason_code="PERSISTENCE_CORRUPTION",
    )
    _, raw = sealed_receipt(
        operation,
        receipt_state="REJECTED_PRE_INVOKE",
        predecessor=predecessor,
        reason_code="PERSISTENCE_CORRUPTION",
    )
    receipt = strict_loads(raw)
    receipt["correlation_id"] = "corr-phase1-other"
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = canonical_digest("phase1-receipt", body)

    with pytest.raises(
        ContractValidationError,
        match="receipt anchor mismatch: correlation_id",
    ):
        validate_receipt(receipt, chain=[anchor, predecessor])


def test_t06_t13_sealed_receipt_state_must_match_predecessor_in_all_directions(
    phase1_roots: dict[str, Path],
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    terminal_raw = _call(phase1_roots, envelope, pair_a, pair_b)
    terminal_receipt = strict_loads(terminal_raw)
    assert isinstance(terminal_receipt, dict)

    key = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    stored = [
        parse_record_bytes(path.read_bytes())
        for path in sorted((key / "records").iterdir())
    ]
    anchor, invocation, engine_predecessor = stored[:3]
    assert engine_predecessor["state"] == "ENGINE_EVIDENCE_VERIFIED"

    preinvoke_predecessor, _ = build_record(
        operation,
        sequence=2,
        state="PREINVOKE_REJECTED",
        previous_record_digest=str(anchor["record_digest"]),
        reason_code="PERSISTENCE_CORRUPTION",
    )
    indeterminate_predecessor, _ = build_record(
        operation,
        sequence=3,
        state="INDETERMINATE_NO_RETRY",
        previous_record_digest=str(invocation["record_digest"]),
        reason_code="PERSISTENCE_CORRUPTION",
    )
    chains = {
        "PREINVOKE_REJECTED": [anchor, preinvoke_predecessor],
        "ENGINE_EVIDENCE_VERIFIED": [anchor, invocation, engine_predecessor],
        "INDETERMINATE_NO_RETRY": [anchor, invocation, indeterminate_predecessor],
    }

    rejected_receipt, _ = sealed_receipt(
        operation,
        receipt_state="REJECTED_PRE_INVOKE",
        predecessor=preinvoke_predecessor,
        reason_code="PERSISTENCE_CORRUPTION",
    )
    indeterminate_receipt, _ = sealed_receipt(
        operation,
        receipt_state="INDETERMINATE_NO_RETRY",
        predecessor=indeterminate_predecessor,
        reason_code="PERSISTENCE_CORRUPTION",
    )
    receipts = {
        "REJECTED_PRE_INVOKE": rejected_receipt,
        "ENGINE_TERMINAL": terminal_receipt,
        "INDETERMINATE_NO_RETRY": indeterminate_receipt,
    }
    expected_predecessors = {
        "REJECTED_PRE_INVOKE": "PREINVOKE_REJECTED",
        "ENGINE_TERMINAL": "ENGINE_EVIDENCE_VERIFIED",
        "INDETERMINATE_NO_RETRY": "INDETERMINATE_NO_RETRY",
    }

    checked: set[tuple[str, str]] = set()
    for receipt_state, valid_receipt in receipts.items():
        for predecessor_state, chain in chains.items():
            if predecessor_state == expected_predecessors[receipt_state]:
                continue
            predecessor = chain[-1]
            candidate = copy.deepcopy(valid_receipt)
            candidate["ownership_record_digest"] = predecessor["record_digest"]
            candidate["receipt_digest"] = canonical_digest(
                "phase1-receipt",
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "receipt_digest"
                },
            )
            assert canonical_bytes(strict_loads(canonical_bytes(candidate))) == (
                canonical_bytes(candidate)
            )
            with pytest.raises(ContractValidationError) as caught:
                validate_receipt(candidate, chain=chain, operation=operation)
            if {
                receipt_state,
                predecessor_state,
            } == {"REJECTED_PRE_INVOKE", "INDETERMINATE_NO_RETRY"}:
                assert "predecessor state mismatch" in str(caught.value)
            checked.add((receipt_state, predecessor_state))

    assert checked == {
        (receipt_state, predecessor_state)
        for receipt_state, expected in expected_predecessors.items()
        for predecessor_state in chains
        if predecessor_state != expected
    }


@pytest.mark.parametrize(
    ("record_field", "replacement", "message"),
    [
        (
            "phase0_terminal_report_digest",
            "0" * 64,
            r"terminal.*predecessor|predecessor.*terminal",
        ),
        (
            "phase0_evidence_manifest_digest",
            "0" * 64,
            r"manifest.*predecessor|predecessor.*manifest",
        ),
        (
            "phase0_evidence_directory_name",
            "phase0-" + "0" * 64,
            r"directory.*predecessor|predecessor.*directory",
        ),
    ],
)
def test_t13_engine_receipt_binds_each_engine_verified_record_field(
    phase1_roots: dict[str, Path],
    record_field: str,
    replacement: str,
    message: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    envelope = operation_bytes()
    operation = parse_operation(envelope)
    raw = _call(phase1_roots, envelope, pair_a, pair_b)
    receipt = strict_loads(raw)
    assert isinstance(receipt, dict)

    key = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    chain = [
        parse_record_bytes(path.read_bytes())
        for path in sorted((key / "records").iterdir())
    ][:3]
    predecessor = copy.deepcopy(chain[-1])
    assert predecessor["state"] == "ENGINE_EVIDENCE_VERIFIED"
    assert predecessor[record_field] != replacement
    predecessor[record_field] = replacement
    predecessor["record_digest"] = canonical_digest(
        "phase1-ownership-record",
        {
            key: value
            for key, value in predecessor.items()
            if key != "record_digest"
        },
    )
    chain[-1] = parse_record_bytes(canonical_bytes(predecessor))

    candidate = copy.deepcopy(receipt)
    candidate["ownership_record_digest"] = predecessor["record_digest"]
    candidate["receipt_digest"] = canonical_digest(
        "phase1-receipt",
        {
            key: value
            for key, value in candidate.items()
            if key != "receipt_digest"
        },
    )
    with pytest.raises(ContractValidationError, match=message):
        validate_receipt(candidate, chain=chain, operation=operation)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b" " * 65_537, "ENVELOPE_TOO_LARGE"),
        (b"{", "ENVELOPE_INVALID_JSON"),
        (b"\xff", "ENVELOPE_INVALID_JSON"),
        (_over_depth_json(), "ENVELOPE_INVALID_JSON"),
        (
            b'{"engine_contract_digest":"x","engine_contract_digest":"y"}',
            "ENVELOPE_INVALID_JSON",
        ),
        (b'{"engine_contract_digest":1.5}', "ENVELOPE_INVALID_JSON"),
        (b"[]", "ENVELOPE_SCHEMA_MISMATCH"),
        (b"{}", "ENVELOPE_SCHEMA_MISMATCH"),
        (b'{"engine_contract_digest":7}', "ENVELOPE_SCHEMA_MISMATCH"),
        (_wrong_schema_operation(), "ENVELOPE_SCHEMA_MISMATCH"),
        (_invalid_phase0_request_operation(), "ENVELOPE_SCHEMA_MISMATCH"),
        (_forbidden_payload_field_operation(), "ENVELOPE_SCHEMA_MISMATCH"),
        (
            b'{"engine_contract_digest":"ffffffffffffffffffffffffffffffff'
            b'ffffffffffffffffffffffffffffffff"}',
            "ENGINE_CONTRACT_MISMATCH",
        ),
        (
            canonical_bytes(
                {
                    **operation_value(),
                    "unexpected": True,
                }
            ),
            "ENVELOPE_SCHEMA_MISMATCH",
        ),
    ],
)
def test_t10_bad_envelope_classes_are_deterministic(
    raw: bytes,
    reason: str,
) -> None:
    receipt = strict_loads(
        execute_operation(
            raw,
            repository_root="relative",
            receipt_root="relative",
            evidence_root="relative",
            pair_a=object(),
            pair_b=object(),
        )
    )
    assert receipt["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert receipt["durability"] == "TRANSIENT"
    assert receipt["reason_codes"] == [reason]
    assert receipt["idempotency_key_digest"] is None


def test_t11_provenance_tamper_fails_before_ownership(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    monkeypatch.setattr(
        adapter,
        "_read_frozen_phase0_file",
        lambda *_args: (_ for _ in ()).throw(
            adapter._FreshRejection("PHASE0_PROVENANCE_MISMATCH")
        ),
    )
    raw = _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    receipt = strict_loads(raw)
    assert receipt["reason_codes"] == ["PHASE0_PROVENANCE_MISMATCH"]
    operation = parse_operation(operation_bytes())
    key = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    assert not key.exists()
    assert pair_a.call_count == pair_b.call_count == 0


def test_t11_wrong_preloaded_import_origin_fails_before_construction(
    phase1_roots: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    wrong_origin = tmp_path / "wrong-olympus-engine.py"
    wrong_origin.write_text("# wrong origin\n", encoding="utf-8")
    wrong_module = types.SimpleNamespace(__file__=str(wrong_origin))

    monkeypatch.setitem(adapter.sys.modules, "olympus_engine", wrong_module)
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda _name: pytest.fail("wrong-origin rejection attempted an import"),
    )
    monkeypatch.setattr(
        OlympusWorkflow,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "wrong-origin rejection constructed a workflow"
        ),
    )

    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    operation = parse_operation(operation_bytes())
    key_root = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    assert receipt["reason_codes"] == ["PHASE0_PROVENANCE_MISMATCH"]
    assert not key_root.exists()
    assert pair_a.call_count == pair_b.call_count == 0


@pytest.mark.parametrize("tamper", ["mode", "size", "hash", "source-type"])
def test_t11_actual_frozen_file_tamper_classes_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    relative = "frozen-source.py"
    source = tmp_path / relative
    original = b"frozen-source\n"
    source.write_bytes(original)
    source.chmod(0o644)
    expected_digest = hashlib.sha256(original).hexdigest()
    monkeypatch.setattr(adapter, "OLYMPUS_CHECKOUT", tmp_path)

    if tamper == "mode":
        source.chmod(0o600)
    elif tamper == "size":
        source.write_bytes(original + b"x")
    elif tamper == "hash":
        source.write_bytes(b"x" * len(original))
    else:
        source.unlink()
        target = tmp_path / "actual-source.py"
        target.write_bytes(original)
        target.chmod(0o644)
        source.symlink_to(target)

    with pytest.raises(adapter._FreshRejection) as caught:
        adapter._read_frozen_phase0_file(
            relative,
            len(original),
            expected_digest,
        )
    assert caught.value.reason_code == "PHASE0_PROVENANCE_MISMATCH"


@pytest.mark.parametrize(
    "variant",
    [
        "pair-a-subclass",
        "pair-b-subclass",
        "pair-a-call-count",
        "pair-b-call-count",
        "pair-a-received-packet",
        "pair-b-received-packet",
        "pair-b-received-worker",
    ],
)
def test_t12_fake_subclass_and_reused_instances_are_rejected(
    phase1_roots: dict[str, Path],
    variant: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])

    class PairASubclass(FakePairATransport):
        pass

    class PairBSubclass(FakePairBTransport):
        pass

    if variant == "pair-a-subclass":
        pair_a = PairASubclass(pair_a.response)
    elif variant == "pair-b-subclass":
        pair_b = PairBSubclass(pair_b.response)
    elif variant == "pair-a-call-count":
        pair_a.call_count = 1
    elif variant == "pair-b-call-count":
        pair_b.call_count = 1
    elif variant == "pair-a-received-packet":
        pair_a.received_packets.append(b"reused")
    elif variant == "pair-b-received-packet":
        pair_b.received_packets.append(b"reused")
    else:
        pair_b.received_worker_results.append(b"reused")

    raw = _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    assert strict_loads(raw)["reason_codes"] == ["FAKE_TRANSPORT_INVALID"]
    operation = parse_operation(operation_bytes())
    key_root = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    assert not key_root.exists()


def test_t13_root_overlap_and_wrong_mode_fail_closed(
    phase1_roots: dict[str, Path],
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=phase1_roots["receipt"],
        evidence_root=phase1_roots["receipt"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    assert strict_loads(raw)["reason_codes"] == ["ROOTS_OVERLAP"]
    phase1_roots["receipt"].chmod(0o755)
    raw = _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    assert strict_loads(raw)["reason_codes"] == ["UNSAFE_RECEIPT_ROOT"]


def test_t13_symlinked_and_traversing_roots_are_rejected(
    phase1_roots: dict[str, Path],
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    receipt_link = phase1_roots["receipt"].parent / "receipt-link"
    receipt_link.symlink_to(phase1_roots["receipt"], target_is_directory=True)
    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=receipt_link,
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    assert strict_loads(raw)["reason_codes"] == ["UNSAFE_RECEIPT_ROOT"]
    traversal_parent = phase1_roots["receipt"].parent / "traversal"
    traversal_parent.mkdir(mode=0o700)
    traversal = (
        f"{traversal_parent}/../{phase1_roots['receipt'].name}"
    )
    raw = execute_operation(
        operation_bytes(),
        repository_root=phase1_roots["repository"],
        receipt_root=traversal,
        evidence_root=phase1_roots["evidence"],
        pair_a=pair_a,
        pair_b=pair_b,
    )
    assert strict_loads(raw)["reason_codes"] == ["UNSAFE_RECEIPT_ROOT"]


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("missing", "EVIDENCE_MISSING"),
        ("invalid", "EVIDENCE_VERIFICATION_FAILED"),
        ("wrong-request", "EVIDENCE_VERIFICATION_FAILED"),
        ("wrong-terminal", "EVIDENCE_VERIFICATION_FAILED"),
        ("wrong-package", "EVIDENCE_VERIFICATION_FAILED"),
        ("changed", "EVIDENCE_VERIFICATION_FAILED"),
    ],
)
def test_t15_evidence_verification_mismatch_never_becomes_engine_terminal(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    reason: str,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_preflight = adapter._preflight

    def poisoned_preflight(pair_a_value: object, pair_b_value: object):
        api = original_preflight(pair_a_value, pair_b_value)
        workflow_type = api.OlympusWorkflow
        verifier = api.verify_evidence_package

        class PoisonedWorkflow:
            def __init__(self, **kwargs: object) -> None:
                self._destination = Path(kwargs["evidence_destination"])
                self._inner = workflow_type(**kwargs)

            def run(self):
                result = self._inner.run()
                if variant == "missing":
                    return dataclasses.replace(result, evidence_manifest=None)
                if variant == "wrong-request":
                    value = result.request.to_dict()
                    value["request_id"] = "req-phase1-wrong"
                    return dataclasses.replace(result, request=_DictObject(value))
                if variant == "wrong-terminal":
                    value = result.terminal_report.to_dict()
                    value["request_id"] = "req-phase1-wrong"
                    return dataclasses.replace(
                        result,
                        terminal_report=_DictObject(
                            value,
                            digest=result.terminal_report.digest,
                        ),
                    )
                if variant == "changed":
                    artifact = self._destination / "01-request.json"
                    artifact.write_bytes(artifact.read_bytes() + b"\n")
                return result

        def verify_evidence(destination: Path):
            if variant == "invalid":
                raise ValueError("injected evidence mismatch")
            verified = verifier(destination)
            if variant == "wrong-package":
                value = verified.to_dict()
                value["package_id"] = "f" * 64
                return _DictObject(value)
            return verified

        return dataclasses.replace(
            api,
            OlympusWorkflow=PoisonedWorkflow,
            verify_evidence_package=verify_evidence,
        )

    monkeypatch.setattr(adapter, "_preflight", poisoned_preflight)
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert receipt["receipt_state"] == "INDETERMINATE_NO_RETRY"
    assert receipt["reason_codes"] == [reason]
    assert receipt["phase0_terminal_report"] is None
    assert receipt["phase0_evidence_manifest"] is None
    assert pair_a.call_count == pair_b.call_count == 1


def test_t15_operation_agnostic_terminal_validation_binds_request_artifact_digest(
    phase1_roots: dict[str, Path],
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert isinstance(receipt, dict)
    manifest = receipt["phase0_evidence_manifest"]
    assert isinstance(manifest, dict)
    artifacts = manifest["artifacts"]
    events = manifest["events"]
    assert isinstance(artifacts, list) and isinstance(events, list)

    request_artifact = next(
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("name") == "01-request.json"
    )
    assert isinstance(request_artifact, dict)
    assert request_artifact["digest"] == receipt["payload_request_digest"]
    request_artifact["digest"] = "f" * 64

    previous_event_digest: str | None = None
    for artifact, event in zip(artifacts, events, strict=True):
        assert isinstance(artifact, dict) and isinstance(event, dict)
        event["artifact_digest"] = artifact["digest"]
        event["previous_event_digest"] = previous_event_digest
        event_body = {
            "sequence": event["sequence"],
            "event_type": event["event_type"],
            "artifact_name": event["artifact_name"],
            "artifact_digest": event["artifact_digest"],
            "previous_event_digest": event["previous_event_digest"],
        }
        event["event_digest"] = canonical_digest("evidence-event", event_body)
        previous_event_digest = str(event["event_digest"])

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
    manifest["package_id"] = canonical_digest(
        "evidence-package",
        {field: manifest.get(field) for field in identity_fields},
    )
    receipt["phase0_evidence_manifest_digest"] = canonical_digest(
        "evidence-manifest", manifest
    )
    receipt["phase0_evidence_package_id"] = manifest["package_id"]
    receipt["receipt_digest"] = canonical_digest(
        "phase1-receipt",
        {key: value for key, value in receipt.items() if key != "receipt_digest"},
    )

    # This deliberately omits ``operation=``. The sealed receipt still has to
    # bind 01-request.json to its own validated payload_request_digest.
    with pytest.raises(ContractValidationError, match="request artifact binding"):
        validate_receipt(receipt)


def test_t15_self_consistent_returned_terminal_must_match_verified_manifest(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_preflight = adapter._preflight

    def poisoned_preflight(pair_a_value: object, pair_b_value: object):
        api = original_preflight(pair_a_value, pair_b_value)
        workflow_type = api.OlympusWorkflow

        class ReplacedTerminalWorkflow:
            def __init__(self, **kwargs: object) -> None:
                self._inner = workflow_type(**kwargs)

            def run(self):
                result = self._inner.run()
                terminal = result.terminal_report.to_dict()
                terminal["summary"] = (
                    f"{terminal['summary']} replaced after verification"
                )
                replacement_digest = canonical_digest("terminal-report", terminal)
                assert replacement_digest != result.terminal_report.digest
                return dataclasses.replace(
                    result,
                    terminal_report=_DictObject(
                        terminal,
                        digest=replacement_digest,
                    ),
                )

        return dataclasses.replace(api, OlympusWorkflow=ReplacedTerminalWorkflow)

    monkeypatch.setattr(adapter, "_preflight", poisoned_preflight)
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert receipt["receipt_state"] == "INDETERMINATE_NO_RETRY"
    assert receipt["reason_codes"] == ["EVIDENCE_VERIFICATION_FAILED"]
    assert receipt["phase0_terminal_report"] is None
    assert receipt["phase0_evidence_manifest"] is None

    operation = parse_operation(operation_bytes())
    key = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    record_names = sorted(path.name for path in (key / "records").iterdir())
    assert not any("ENGINE_EVIDENCE_VERIFIED" in name for name in record_names)
    assert any("INDETERMINATE_NO_RETRY" in name for name in record_names)
    assert pair_a.call_count == pair_b.call_count == 1


def test_t16_static_runtime_exclusion_guard() -> None:
    package = Path(adapter.__file__).parent
    forbidden_imports = {
        "_thread",
        "agent",
        "agents",
        "aiohttp",
        "anthropic",
        "concurrent",
        "config",
        "configparser",
        "configs",
        "configuration",
        "ctypes",
        "delivery",
        "dotenv",
        "fastapi",
        "ftplib",
        "gateway",
        "gateways",
        "grpc",
        "hermes_cli",
        "http",
        "httpx",
        "imaplib",
        "model_tools",
        "multiprocessing",
        "openai",
        "plugin",
        "plugins",
        "pluggy",
        "poplib",
        "profile",
        "profiles",
        "provider",
        "providers",
        "pty",
        "requests",
        "route",
        "routes",
        "service",
        "services",
        "smtplib",
        "socket",
        "ssl",
        "starlette",
        "subprocess",
        "telnetlib",
        "tool",
        "tools",
        "tomllib",
        "urllib",
        "webbrowser",
        "websockets",
        "yaml",
    }
    forbidden_calls = {
        "add_api_route",
        "add_route",
        "create_service",
        "deliver",
        "get_config",
        "get_profile",
        "include_router",
        "load_config",
        "load_profile",
        "mount",
        "read_config",
        "register",
        "register_plugin",
        "register_route",
        "register_tool",
        "send",
        "send_message",
        "start_service",
    }
    forbidden_qualified_calls = {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "asyncio.create_task",
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "builtins.__import__",
        "compile",
        "eval",
        "exec",
        "__import__",
        "os.fork",
        "os.forkpty",
        "os.popen",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.startfile",
        "os.system",
        "threading.Thread",
        "_thread.start_new_thread",
    }

    def qualified_name(node: ast.expr, aliases: dict[str, str]) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = qualified_name(node.value, aliases)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    paths = sorted(package.rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".", 1)[0])
                    aliases[alias.asname or alias.name.split(".", 1)[0]] = (
                        alias.name
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
                for alias in node.names:
                    aliases[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        assert not (imported & forbidden_imports), (path, imported & forbidden_imports)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = qualified_name(node.func, aliases)
            leaf = name.rsplit(".", 1)[-1]
            assert leaf not in forbidden_calls, (path, name)
            assert name not in forbidden_qualified_calls, (path, name)
            assert not name.startswith(
                ("socket.", "subprocess.", "multiprocessing.")
            ), (path, name)
            assert not name.startswith(("os.exec", "os.spawn")), (path, name)
            if name.startswith("importlib."):
                assert name == "importlib.import_module", (path, name)
                assert len(node.args) == 1 and not node.keywords, (path, name)
                assert isinstance(node.args[0], ast.Constant), (path, name)
                assert node.args[0].value == "olympus_engine", (path, name)

    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in (
        "AIAgent",
        "PluginManager",
        "config.yaml",
        "gateway/",
        "/v1/runs",
        "provider_factory",
        "run_conversation",
        "send_message_tool",
    ):
        assert forbidden not in source


def test_t16_runtime_forbidden_capabilities_are_unreachable(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    violations: list[str] = []

    def denied(label: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            violations.append(label)
            raise AssertionError(f"forbidden Phase 1 capability reached: {label}")

        return fail

    for module, names in (
        (
            socket,
            ("socket", "socketpair", "create_connection", "create_server"),
        ),
        (
            subprocess,
            (
                "Popen",
                "run",
                "call",
                "check_call",
                "check_output",
                "getoutput",
                "getstatusoutput",
            ),
        ),
        (multiprocessing, ("Process", "Pool", "get_context")),
        (
            asyncio,
            ("create_task", "create_subprocess_exec", "create_subprocess_shell"),
        ),
        (threading, ("Thread",)),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, denied(f"{module.__name__}.{name}"))

    for name in (
        "fork",
        "forkpty",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "startfile",
        "system",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, denied(f"os.{name}"))

    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert pair_a.call_count == pair_b.call_count == 1
    assert violations == []


def test_t17_package_is_excluded_from_distribution_metadata() -> None:
    from setuptools import find_namespace_packages, find_packages

    root = Path(adapter.__file__).parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    manifest_path = root / "MANIFEST.in"
    manifest = (
        manifest_path.read_text(encoding="utf-8")
        if manifest_path.is_file()
        else ""
    )
    setuptools_metadata = pyproject["tool"]["setuptools"]
    finder = setuptools_metadata["packages"]["find"]
    all_packages = set(find_packages(where=str(root)))
    configured_finder = (
        find_namespace_packages
        if finder.get("namespaces", True)
        else find_packages
    )
    configured_packages = set(
        configured_finder(
            where=str(root),
            include=tuple(finder["include"]),
            exclude=tuple(finder.get("exclude", ())),
        )
    )
    assert "olympus_phase1_adapter" in all_packages
    assert not any(
        package == "olympus_phase1_adapter"
        or package.startswith("olympus_phase1_adapter.")
        for package in configured_packages
    )

    published_metadata = {
        "scripts": pyproject["project"].get("scripts", {}),
        "gui-scripts": pyproject["project"].get("gui-scripts", {}),
        "entry-points": pyproject["project"].get("entry-points", {}),
        "py-modules": setuptools_metadata.get("py-modules", []),
        "package-data": setuptools_metadata.get("package-data", {}),
        "data-files": setuptools_metadata.get("data-files", {}),
    }
    assert "olympus_phase1_adapter" not in json.dumps(
        published_metadata, sort_keys=True
    )
    assert "olympus_phase1_adapter" not in manifest


def test_t17_in_process_wheel_and_sdist_filelists_exclude_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from setuptools import Distribution, find_namespace_packages
    from setuptools.command.build_py import build_py
    from setuptools.command.egg_info import egg_info
    from setuptools.command.sdist import sdist

    root = Path(adapter.__file__).parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools_metadata = pyproject["tool"]["setuptools"]
    finder = setuptools_metadata["packages"]["find"]
    packages = sorted(
        find_namespace_packages(
            where=str(root),
            include=tuple(finder["include"]),
            exclude=tuple(finder.get("exclude", ())),
        )
    )
    scripts = pyproject["project"].get("scripts", {})
    distribution = Distribution(
        {
            "name": pyproject["project"]["name"],
            "version": pyproject["project"]["version"],
            "packages": packages,
            "package_dir": {"": str(root)},
            "py_modules": setuptools_metadata.get("py-modules", []),
            "entry_points": {
                "console_scripts": [
                    f"{name} = {target}" for name, target in scripts.items()
                ]
            },
        }
    )
    distribution.script_name = str(root / "pyproject.toml")

    wheel_command = build_py(distribution)
    wheel_command.build_lib = str(tmp_path / "wheel-tree")
    wheel_command.ensure_finalized()
    wheel_sources = {
        Path(path).resolve() for path in wheel_command.get_source_files()
    }

    egg_base = tmp_path / "egg-info"
    egg_base.mkdir()
    egg_command = egg_info(distribution)
    egg_command.egg_base = str(egg_base)
    egg_command.ensure_finalized()
    distribution.command_obj["egg_info"] = egg_command
    sdist_command = sdist(distribution)
    distribution.command_obj["sdist"] = sdist_command
    before_repo_egg_info = sorted(root.glob("*.egg-info"))
    monkeypatch.chdir(root)
    sdist_command.ensure_finalized()
    sdist_command.run_command("egg_info")
    sdist_command.filelist = egg_command.filelist
    sdist_sources = {
        (root / path).resolve() for path in sdist_command.filelist.files
    }

    adapter_root = (root / "olympus_phase1_adapter").resolve()

    def belongs_to_adapter(path: Path) -> bool:
        return path == adapter_root or adapter_root in path.parents

    assert not any(belongs_to_adapter(path) for path in wheel_sources)
    assert not any(belongs_to_adapter(path) for path in sdist_sources)
    assert "olympus_phase1_adapter" not in str(distribution.entry_points)
    assert sorted(root.glob("*.egg-info")) == before_repo_egg_info
    assert any(egg_base.rglob("SOURCES.txt"))


def test_t18_frozen_digests_vectors_and_schema_bytes() -> None:
    packet_path = Path(
        "/Users/macmini/Hermes-Handoff/reviews/"
        "olympus-phase1-gate2-design-freeze-20260806T130942Z/"
        "OLYMPUS_PHASE1_GATE2_IMPLEMENTATION_DESIGN_FREEZE_PACKET.json"
    )
    packet_raw = packet_path.read_bytes()
    assert hashlib.sha256(packet_raw).hexdigest() == GATE2_PACKET_SHA256
    packet = strict_loads(packet_raw)
    assert isinstance(packet, dict)

    integrity = packet["integrity"]
    assert isinstance(integrity, dict)
    integrity_payload = canonical_bytes(
        {key: value for key, value in packet.items() if key != "integrity"}
    )
    assert len(integrity_payload) == integrity["payload_bytes"] == 148_364
    assert integrity["payload_sha256"] == (
        "94f7628b0ac1dd95289fdffeed6d661c151a6c543cbd5ca12c686fcf5646842c"
    )
    assert hashlib.sha256(integrity_payload).hexdigest() == integrity[
        "payload_sha256"
    ]

    contract = packet["contract"]
    assert isinstance(contract, dict)
    assert contract["phase1_contract_digest"] == PHASE1_CONTRACT_DIGEST
    assert canonical_digest("phase1-contract", contract["surface"]) == (
        PHASE1_CONTRACT_DIGEST
    )
    phase0_files = packet["phase0_provenance_and_import"][
        "source_and_schema_files"
    ]
    assert canonical_digest("phase1-phase0-binding", phase0_files) == (
        PHASE0_RUNTIME_BINDING_DIGEST
    )
    gate1 = packet["gate1_binding"]
    assert gate1["raw_sha256"] == GATE1_PACKET_SHA256
    assert gate1["engine_contract_digest"] == ENGINE_CONTRACT_DIGEST

    root = Path(adapter.__file__).parents[1]
    schemas = packet["schemas"]
    assert isinstance(schemas, list) and len(schemas) == 3
    for schema in schemas:
        assert isinstance(schema, dict)
        schema_raw = (root / str(schema["path"])).read_bytes()
        assert len(schema_raw) == schema["raw_bytes"]
        assert hashlib.sha256(schema_raw).hexdigest() == schema["raw_sha256"]
        document = strict_loads(schema_raw)
        assert document == schema["document"]
        assert isinstance(document, dict)
        assert document["$id"] == schema["schema_id"]
        assert canonical_digest("json-schema", document) == (
            schema["canonical_schema_digest"]
        )

    vectors = packet["digest_vectors"]
    operation_vector = vectors["operation"]
    operation = parse_operation(canonical_bytes(operation_vector["envelope"]))
    assert operation.semantic_body == operation_vector["semantic_body"]
    assert canonical_bytes(operation.semantic_body).hex() == (
        operation_vector["canonical_semantic_body_hex"]
    )
    assert operation.idempotency_key_digest == (
        operation_vector["idempotency_key_digest"]
    )
    assert operation.operation_digest == operation_vector["operation_digest"]
    assert operation.payload_request_digest == (
        operation_vector["payload_request_digest"]
    )

    record_vector = vectors["first_ownership_record"]
    record, record_raw = build_record(
        operation,
        sequence=1,
        state="OWNERSHIP_ACQUIRED",
        previous_record_digest=None,
    )
    assert record == record_vector["record"]
    assert record_raw.hex() == record_vector["canonical_record_hex"]

    receipt_vector = vectors["transient_receipt"]
    transient_raw = transient_rejection("ENVELOPE_INVALID_JSON")
    assert strict_loads(transient_raw) == receipt_vector["receipt"]
    assert transient_raw.hex() == receipt_vector["canonical_receipt_hex"]

    verify_frozen_schemas()
    assert GATE1_PACKET_SHA256 == (
        "6ef02d1e63d19853e0794b6c324f0e7233ec082744419fae529ed16e6e11cacf"
    )
    assert GATE2_PACKET_SHA256 == (
        "445f0ce7bbf3c063ce84d7b1ae47154241e5bb6f148969790a5d63dfca4f6d25"
    )
    assert PHASE1_CONTRACT_DIGEST == (
        "e5db4065e1e39134a5274152a01363d2ed3a0c79fb5d4bdb9c1773d36a2b1de1"
    )
    assert ENGINE_CONTRACT_DIGEST == (
        "2c5860582fcc73192b4e301544e043189dac1079b47078bad3c1f93371b8ee85"
    )
    assert PHASE0_RUNTIME_BINDING_DIGEST == (
        "366f41af5292272b183605cb1f6f5ab75b3a259c453d3396eab34a34bb83cb12"
    )
    assert digest_vectors() == {
        "idempotency_key_digest": operation_vector["idempotency_key_digest"],
        "operation_digest": operation_vector["operation_digest"],
        "payload_request_digest": operation_vector["payload_request_digest"],
        "first_record_digest": record_vector["record"]["record_digest"],
        "transient_invalid_json_receipt_digest": receipt_vector["receipt"][
            "receipt_digest"
        ],
    }


def test_t18_sealed_receipt_unavailable_annotation_is_exact() -> None:
    hints = get_type_hints(SealedReceiptUnavailable.__init__)
    assert hints["receipt_state"] == Literal[
        "REJECTED_PRE_INVOKE",
        "ENGINE_TERMINAL",
        "INDETERMINATE_NO_RETRY",
    ]


def test_t20_postclaim_receipt_unavailable_raises_exact_exception(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    operation = parse_operation(operation_bytes())
    monkeypatch.setattr(
        OwnerHandle,
        "seal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PersistenceError("injected receipt failure")
        ),
    )
    with pytest.raises(IndeterminateReceiptUnavailable) as caught:
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    error = caught.value
    assert str(error) == (
        "Phase 1 attempt is indeterminate and no sealed receipt is available; "
        "automatic retry is forbidden"
    )
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert error.idempotency_key_digest == operation.idempotency_key_digest
    assert error.operation_digest == operation.operation_digest
    assert error.receipt_state == "INDETERMINATE_NO_RETRY"
    assert error.reason_code == "PERSISTENCE_CORRUPTION"
    assert error.automatic_engine_retry_permitted is False
    calls = (pair_a.call_count, pair_b.call_count)
    monkeypatch.undo()
    recovered = strict_loads(
        _call(phase1_roots, operation_bytes(), object(), object())
    )
    assert recovered["receipt_state"] == "INDETERMINATE_NO_RETRY"
    assert recovered["reason_codes"] == ["PERSISTENCE_CORRUPTION"]
    assert (pair_a.call_count, pair_b.call_count) == calls


def test_t21_preinvoke_receipt_unavailable_raises_exact_exception(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    operation = parse_operation(operation_bytes())
    original_append = OwnerHandle.append

    def fail_claim(self: OwnerHandle, state: str, **kwargs: object):
        if state == "INVOCATION_CLAIMED":
            raise PersistenceError("injected claim failure")
        return original_append(self, state, **kwargs)

    monkeypatch.setattr(OwnerHandle, "append", fail_claim)
    monkeypatch.setattr(
        OwnerHandle,
        "seal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PersistenceError("injected receipt failure")
        ),
    )
    with pytest.raises(PreInvokeReceiptUnavailable) as caught:
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    error = caught.value
    assert str(error) == (
        "Phase 1 invocation did not occur, but no sealed rejection receipt is "
        "available; automatic retry is forbidden"
    )
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert error.idempotency_key_digest == operation.idempotency_key_digest
    assert error.operation_digest == operation.operation_digest
    assert error.engine_invocation_occurred is False
    assert error.receipt_state == "REJECTED_PRE_INVOKE"
    assert error.reason_code == "PERSISTENCE_CORRUPTION"
    assert error.automatic_engine_retry_permitted is False
    monkeypatch.undo()
    recovered = strict_loads(
        _call(phase1_roots, operation_bytes(), object(), object())
    )
    assert recovered["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert recovered["reason_codes"] == ["PERSISTENCE_CORRUPTION"]
    assert pair_a.call_count == pair_b.call_count == 0


def _control_flow_objects() -> list[BaseException]:
    return [
        KeyboardInterrupt("injected"),
        SystemExit(73),
        asyncio.CancelledError("injected"),
    ]


@pytest.mark.parametrize("injected", _control_flow_objects())
def test_t23_control_flow_before_ownership_is_identical_and_writes_no_key(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])

    def interrupt_preflight(_pair_a: object, _pair_b: object):
        raise injected

    monkeypatch.setattr(adapter, "_preflight", interrupt_preflight)
    with pytest.raises(type(injected)) as caught:
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    assert caught.value is injected
    operation = parse_operation(operation_bytes())
    key = (
        phase1_roots["receipt"]
        / "v1"
        / operation.idempotency_key_digest[:2]
        / operation.idempotency_key_digest
    )
    assert not key.exists()
    assert pair_a.call_count == pair_b.call_count == 0
    assert sorted(path.name for path in phase1_roots["receipt"].iterdir()) == [
        ".phase1-store.lock",
        "v1",
    ]
    assert list(phase1_roots["evidence"].iterdir()) == []

    monkeypatch.undo()
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    )
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert pair_a.call_count == pair_b.call_count == 1


@pytest.mark.parametrize("injected", _control_flow_objects())
def test_t23_control_flow_after_ownership_before_claim_is_identical(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_append = OwnerHandle.append

    def interrupt_before_claim(
        self: OwnerHandle, state: str, **kwargs: object
    ):
        if state == "INVOCATION_CLAIMED":
            raise injected
        return original_append(self, state, **kwargs)

    monkeypatch.setattr(OwnerHandle, "append", interrupt_before_claim)
    with pytest.raises(type(injected)) as caught:
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    assert caught.value is injected
    assert pair_a.call_count == pair_b.call_count == 0
    monkeypatch.undo()
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), object(), object())
    )
    assert receipt["receipt_state"] == "REJECTED_PRE_INVOKE"
    assert receipt["reason_codes"] == ["CANCELLED_BEFORE_INVOCATION_CLAIM"]


@pytest.mark.parametrize("injected", _control_flow_objects())
def test_t23_control_flow_after_claim_is_identical_and_never_reinvokes(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    original_append = OwnerHandle.append

    def interrupt_after_claim(
        self: OwnerHandle, state: str, **kwargs: object
    ):
        result = original_append(self, state, **kwargs)
        if state == "INVOCATION_CLAIMED":
            raise injected
        return result

    monkeypatch.setattr(OwnerHandle, "append", interrupt_after_claim)
    with pytest.raises(type(injected)) as caught:
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    assert caught.value is injected
    assert pair_a.call_count == pair_b.call_count == 0
    monkeypatch.undo()
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), object(), object())
    )
    assert receipt["receipt_state"] == "INDETERMINATE_NO_RETRY"
    assert receipt["reason_codes"] == ["CANCELLED_AFTER_INVOCATION_CLAIM"]
    assert pair_a.call_count == pair_b.call_count == 0


@pytest.mark.parametrize("injected", _control_flow_objects())
def test_t23_control_flow_after_receipt_seal_is_identical_and_replayable(
    phase1_roots: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    pair_a, pair_b = valid_transports(phase1_roots["repository"])
    finalize_calls = 0

    def interrupt_finalization(_self: OwnerHandle) -> None:
        nonlocal finalize_calls
        finalize_calls += 1
        raise injected

    monkeypatch.setattr(OwnerHandle, "_finalize_once", interrupt_finalization)
    with pytest.raises(type(injected)) as caught:
        _call(phase1_roots, operation_bytes(), pair_a, pair_b)
    assert caught.value is injected
    assert finalize_calls == 1
    assert pair_a.call_count == pair_b.call_count == 1
    monkeypatch.undo()
    receipt = strict_loads(
        _call(phase1_roots, operation_bytes(), object(), object())
    )
    assert receipt["receipt_state"] == "ENGINE_TERMINAL"
    assert receipt["reason_codes"] == ["PHASE0_TERMINAL"]
    assert pair_a.call_count == pair_b.call_count == 1
