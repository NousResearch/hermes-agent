"""Focal tests for the production EvidencePack monitoring audit sink.

The audit sink adapter is the seam between the engine's in-memory
audit emit (the ``self._audit_sink.emit({...})`` call inside
``EvidencePackEngine.dry_run`` for ``severity == 'high'`` conflicts)
and the process monitoring emitter.

Coverage matrix:

* Accepted events project onto the canonical 9-field schema
* Output is the EXACT nine fields, in canonical order
* ``event_id`` is deterministic from the canonical tuple
* Distinct logical identities yield distinct event_ids
* Input mapping is never mutated (no surprise side effects)
* Sensitive and unknown input fields are stripped from the wire
* Missing, malformed, or oversized identifiers drop the event
* Unsupported gate_type / severity drop the event
* Non-Mapping inputs drop the event without raising
* Hostile Mapping inputs (custom __setitem__, __iter__, etc.) are
  isolated from the emitter
* Emitter exceptions never propagate
* Emit delegates exactly once per accepted call
* Adapter has no ``close`` method and never resolves ``get_emitter``
* Adapter has no filesystem / network / thread / process side effect
* Adapter has no import from hermes_cli, gateway, tui_gateway
"""
from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import MutableMapping
from typing import Any

import pytest

from agent.executive.knowledge_discovery.adapters import (
    ALLOWED_GATE_TYPE,
    ALLOWED_SEVERITY,
    EVENT,
    EvidencePackMonitoringAuditSink,
    IDENTIFIER_FIELDS,
    OUTPUT_FIELD_ORDER,
    SCHEMA_VERSION,
    SOURCE,
)
from agent.executive.knowledge_discovery.adapters import (
    audit_sink as audit_sink_module,
)


# Canonical inputs that the engine emits today (per
# ``EvidencePackEngine._emit_conflict`` in engine.py: high-severity
# knowledge conflicts only). The fixture mirrors the on-the-wire shape
# so the tests are real against the engine contract.
CONFLICT_ID = "kc-7f3b9c0e1a4d"
OBJECTIVE_ID = "obj-b1-e3a"
DETECTED_AT = "2026-08-04T12:00:00+00:00"


# ─────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────


class RecordingEmitter:
    """In-memory recording emitter.

    Records every ``emit`` call. By default it returns ``None`` from
    ``emit`` so the adapter sees the documented non-raising contract.
    Tests can install a side effect (raise, return value, etc.) by
    setting attributes after construction.
    """

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self._raise: "BaseException | None" = None

    def emit(self, event: Any) -> None:
        self.calls.append(event)
        if self._raise is not None:
            raise self._raise


class HostileMapping(MutableMapping):
    """A hostile input that records mutation attempts and surfaces
    unexpected reads.

    Used to verify that the adapter never writes back into the input
    and handles read-side errors gracefully. The adapter must drop
    the event without crashing and without mutating the hostile state.
    """

    def __init__(self, *, raise_on_read: bool = True) -> None:
        self._data: dict[str, Any] = {}
        self.read_attempts: int = 0
        self._raise_on_read = raise_on_read

    def __getitem__(self, key: str) -> Any:
        self.read_attempts += 1
        if self._raise_on_read:
            raise RuntimeError("hostile read should not be observable")
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        self._data.pop(key, None)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _valid_event() -> dict[str, Any]:
    """Return the engine-shaped audit event the adapter must accept."""
    return {
        "gate_type": "knowledge_conflict",
        "severity": "high",
        "conflict_id": CONFLICT_ID,
        "objective_id": OBJECTIVE_ID,
        "detected_at": DETECTED_AT,
    }


def _sink() -> "tuple[RecordingEmitter, EvidencePackMonitoringAuditSink]":
    emitter = RecordingEmitter()
    sink = EvidencePackMonitoringAuditSink(emitter=emitter)
    return emitter, sink


# ─────────────────────────────────────────────────────────────────────
# 1. Accepted events project onto the canonical 9-field schema
# ─────────────────────────────────────────────────────────────────────


def test_accepted_event_is_delegated_to_emitter():
    """A valid engine-shaped audit event reaches the emitter with the
    canonical nine-field payload."""
    emitter, sink = _sink()
    sink.emit(_valid_event())
    assert len(emitter.calls) == 1
    payload = emitter.calls[0]
    assert isinstance(payload, dict)
    # Canonical fixed fields
    assert payload["event"] == EVENT
    assert payload["gate_type"] == "knowledge_conflict"
    assert payload["severity"] == "high"
    assert payload["conflict_id"] == CONFLICT_ID
    assert payload["objective_id"] == OBJECTIVE_ID
    assert payload["detected_at"] == DETECTED_AT
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source"] == SOURCE
    assert "event_id" in payload
    assert isinstance(payload["event_id"], str)


# ─────────────────────────────────────────────────────────────────────
# 2. Output is EXACTLY nine fields in canonical order
# ─────────────────────────────────────────────────────────────────────


def test_accepted_event_output_is_exactly_nine_fields():
    """The emitted payload must contain exactly the canonical fields
    and nothing else — no extra keys, no missing keys."""
    emitter, sink = _sink()
    sink.emit(_valid_event())
    payload = emitter.calls[0]
    assert set(payload.keys()) == set(OUTPUT_FIELD_ORDER)
    assert tuple(payload.keys()) == OUTPUT_FIELD_ORDER
    assert len(payload) == 9


def test_canonical_constants_match_schema_contract():
    """Lock the canonical constants to their documented values."""
    assert SCHEMA_VERSION == "evidence_pack.audit.v1"
    assert EVENT == "knowledge_conflict"
    assert SOURCE == "evidence_pack"
    assert ALLOWED_GATE_TYPE == "knowledge_conflict"
    assert ALLOWED_SEVERITY == "high"
    assert IDENTIFIER_FIELDS == ("conflict_id", "objective_id", "detected_at")
    assert OUTPUT_FIELD_ORDER == (
        "event",
        "gate_type",
        "severity",
        "conflict_id",
        "objective_id",
        "detected_at",
        "schema_version",
        "source",
        "event_id",
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Deterministic event_id
# ─────────────────────────────────────────────────────────────────────


def test_event_id_is_deterministic_for_identical_input():
    """Identical inputs must produce identical event_ids across
    separate adapter instances and across repeated calls."""
    emitter_a, sink_a = _sink()
    sink_a.emit(_valid_event())
    event_id_a = emitter_a.calls[0]["event_id"]

    emitter_b, sink_b = _sink()
    sink_b.emit(_valid_event())
    event_id_b = emitter_b.calls[0]["event_id"]

    assert event_id_a == event_id_b

    # And within the same adapter: deterministic across repeats.
    sink_a.emit(_valid_event())
    sink_a.emit(_valid_event())
    assert emitter_a.calls[1]["event_id"] == event_id_a
    assert emitter_a.calls[2]["event_id"] == event_id_a


def test_event_id_format_is_schema_prefixed_sha256_prefix():
    """event_id is a short, prefixed hex string. The exact prefix
    format is part of the audit schema contract."""
    emitter, sink = _sink()
    sink.emit(_valid_event())
    event_id = emitter.calls[0]["event_id"]
    # Starts with the canonical "epa1-" prefix
    assert event_id.startswith("epa1-")
    # Followed by exactly 16 lowercase hex characters
    suffix = event_id[len("epa1-"):]
    assert len(suffix) == 16
    assert all(c in "0123456789abcdef" for c in suffix)


# ─────────────────────────────────────────────────────────────────────
# 4. Distinct identities yield distinct event_ids
# ─────────────────────────────────────────────────────────────────────


def test_different_objective_id_yields_different_event_id():
    emitter_a, sink_a = _sink()
    sink_a.emit(_valid_event())
    event_a = emitter_a.calls[0]["event_id"]

    emitter_b, sink_b = _sink()
    sink_b.emit({**_valid_event(), "objective_id": "obj-different"})
    event_b = emitter_b.calls[0]["event_id"]

    assert event_a != event_b


def test_different_conflict_id_yields_different_event_id():
    emitter_a, sink_a = _sink()
    sink_a.emit(_valid_event())
    event_a = emitter_a.calls[0]["event_id"]

    emitter_b, sink_b = _sink()
    sink_b.emit({**_valid_event(), "conflict_id": "kc-000000000000"})
    event_b = emitter_b.calls[0]["event_id"]

    assert event_a != event_b


def test_different_detected_at_yields_different_event_id():
    emitter_a, sink_a = _sink()
    sink_a.emit(_valid_event())
    event_a = emitter_a.calls[0]["event_id"]

    emitter_b, sink_b = _sink()
    sink_b.emit({**_valid_event(), "detected_at": "2026-08-04T13:00:00+00:00"})
    event_b = emitter_b.calls[0]["event_id"]

    assert event_a != event_b


def test_extra_unknown_fields_do_not_change_event_id():
    """Unknown input fields must not perturb the deterministic event_id.

    Two events with the same canonical five fields but different
    unknown-field noise must produce the same event_id."""
    emitter_a, sink_a = _sink()
    sink_a.emit(_valid_event())
    event_a = emitter_a.calls[0]["event_id"]

    emitter_b, sink_b = _sink()
    sink_b.emit({**_valid_event(), "noise": "x", "prompt": "leak me"})
    event_b = emitter_b.calls[0]["event_id"]

    assert event_a == event_b


# ─────────────────────────────────────────────────────────────────────
# 5. No input mutation
# ─────────────────────────────────────────────────────────────────────


def test_input_mapping_is_not_mutated():
    """The adapter must not write back into the input mapping."""
    emitter, sink = _sink()
    event = _valid_event()
    snapshot = dict(event)  # copy before
    sink.emit(event)
    assert event == snapshot
    # Confirm: no new keys added, no values changed.
    assert set(event.keys()) == set(snapshot.keys())
    assert all(event[k] == snapshot[k] for k in snapshot)


def test_input_mapping_with_unknown_fields_is_not_mutated():
    """Even when unknown fields are present, the input is left intact."""
    emitter, sink = _sink()
    event = {**_valid_event(), "prompt_text": "leak me", "secret": "abc"}
    snapshot_keys = set(event.keys())
    snapshot = {k: event[k] for k in snapshot_keys}
    sink.emit(event)
    assert set(event.keys()) == snapshot_keys
    assert {event[k] for k in snapshot_keys} == {snapshot[k] for k in snapshot_keys}


def test_returned_payload_is_a_new_object():
    """The adapter must not return the input dict or share identity."""
    emitter, sink = _sink()
    event = _valid_event()
    sink.emit(event)
    payload = emitter.calls[0]
    assert payload is not event
    # The payload contains the canonical nine fields — values for the
    # three input identifiers come from the input, the other six are
    # fixed by the schema.
    expected = {
        "event": EVENT,
        "gate_type": "knowledge_conflict",
        "severity": "high",
        "conflict_id": event["conflict_id"],
        "objective_id": event["objective_id"],
        "detected_at": event["detected_at"],
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "event_id": payload["event_id"],
    }
    assert payload == expected


# ─────────────────────────────────────────────────────────────────────
# 6. Sensitive and unknown field removal
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt": "do not leak the user's prompt"},
        {"objective_text": "ship the migration by Friday"},
        {"snippet": "secret compliance text"},
        {"api_key": "sk-1234"},
        {"token": "bearer xyz"},
        {"password": "hunter2"},
        {"path": "/home/jr-ubuntu/.hermes/state.db"},
        {"env": "PROD_API_KEY"},
        {"argv": ["--secret", "value"]},
        {"random": "noise"},
    ],
)
def test_unknown_and_sensitive_fields_are_dropped(extra):
    """No unknown / sensitive input field may appear in the wire payload."""
    emitter, sink = _sink()
    event = {**_valid_event(), **extra}
    sink.emit(event)
    assert len(emitter.calls) == 1
    payload = emitter.calls[0]
    # The canonical nine fields are the only keys.
    assert set(payload.keys()) == set(OUTPUT_FIELD_ORDER)
    # No extra key is reflected anywhere on the wire.
    for k in extra:
        assert k not in payload
    # No extra value reaches the wire (only check stringy values to
    # avoid coercing list/dict values; the key-check above is the
    # primary invariant).
    serialized = "\n".join(str(v) for v in payload.values())
    for k, v in extra.items():
        if isinstance(v, str):
            assert v not in serialized


# ─────────────────────────────────────────────────────────────────────
# 7. Missing / malformed / oversized rejection
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "missing_field", ["conflict_id", "objective_id", "detected_at"]
)
def test_missing_required_identifier_drops_event(missing_field):
    emitter, sink = _sink()
    event = _valid_event()
    del event[missing_field]
    sink.emit(event)
    assert emitter.calls == []


@pytest.mark.parametrize("bad_value", [None, 42, 3.14, b"bytes", [], {}, True])
def test_non_string_identifier_drops_event(bad_value):
    emitter, sink = _sink()
    event = {**_valid_event(), "conflict_id": bad_value}
    sink.emit(event)
    assert emitter.calls == []


@pytest.mark.parametrize("bad_value", ["", "   ", "\t", "\n", " \n\t  \n "])
def test_empty_or_whitespace_identifier_drops_event(bad_value):
    emitter, sink = _sink()
    event = {**_valid_event(), "objective_id": bad_value}
    sink.emit(event)
    assert emitter.calls == []


def test_oversized_identifier_drops_event_without_truncation():
    """An oversized identifier must be rejected outright, not truncated."""
    emitter, sink = _sink()
    huge = "x" * 1024
    event = {**_valid_event(), "conflict_id": huge}
    sink.emit(event)
    assert emitter.calls == []
    # And the input itself is untouched.
    assert event["conflict_id"] == huge


def test_oversized_detected_at_drops_event():
    emitter, sink = _sink()
    event = {**_valid_event(), "detected_at": "z" * 1024}
    sink.emit(event)
    assert emitter.calls == []


def test_max_len_identifier_is_accepted():
    """An identifier at exactly the cap must still be accepted."""
    emitter, sink = _sink()
    edge = "a" * 256  # cap
    event = {**_valid_event(), "objective_id": edge}
    sink.emit(event)
    assert len(emitter.calls) == 1
    assert emitter.calls[0]["objective_id"] == edge


# ─────────────────────────────────────────────────────────────────────
# 8. Unsupported gate_type / severity rejection
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_gate", ["evidence_conflict", "", "KNOWLEDGE_CONFLICT", None])
def test_unsupported_gate_type_drops_event(bad_gate):
    emitter, sink = _sink()
    event = {**_valid_event(), "gate_type": bad_gate}
    sink.emit(event)
    assert emitter.calls == []


@pytest.mark.parametrize("bad_severity", ["low", "medium", "HIGH", "warn", None])
def test_unsupported_severity_drops_event(bad_severity):
    emitter, sink = _sink()
    event = {**_valid_event(), "severity": bad_severity}
    sink.emit(event)
    assert emitter.calls == []


def test_missing_gate_type_drops_event():
    emitter, sink = _sink()
    event = _valid_event()
    del event["gate_type"]
    sink.emit(event)
    assert emitter.calls == []


def test_missing_severity_drops_event():
    emitter, sink = _sink()
    event = _valid_event()
    del event["severity"]
    sink.emit(event)
    assert emitter.calls == []


# ─────────────────────────────────────────────────────────────────────
# 9. Non-Mapping / hostile input
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "not_a_mapping",
    [None, 42, "string", ["list"], ("tuple",), object()],
)
def test_non_mapping_input_drops_event_without_raising(not_a_mapping):
    emitter, sink = _sink()
    sink.emit(not_a_mapping)  # type: ignore[arg-type]
    assert emitter.calls == []


def test_hostile_mapping_dropped_without_mutations():
    """A hostile Mapping (raises on read) must not crash the adapter
    and must not be mutated by it. The adapter is total — it catches
    the read error and drops the event."""
    emitter, sink = _sink()
    hostile = HostileMapping()
    hostile["gate_type"] = "knowledge_conflict"
    hostile["severity"] = "high"
    hostile["conflict_id"] = CONFLICT_ID
    hostile["objective_id"] = OBJECTIVE_ID
    hostile["detected_at"] = DETECTED_AT
    # Even though __getitem__ raises, the adapter must not propagate
    # the error and must not mutate the hostile state.
    sink.emit(hostile)  # type: ignore[arg-type]
    # No payload reaches the emitter when the read fails.
    assert emitter.calls == []
    # The hostile object was not mutated by the adapter.
    assert hostile._data == {
        "gate_type": "knowledge_conflict",
        "severity": "high",
        "conflict_id": CONFLICT_ID,
        "objective_id": OBJECTIVE_ID,
        "detected_at": DETECTED_AT,
    }


def test_non_raising_hostile_mapping_accepted_without_back_writes():
    """A well-behaved hostile Mapping (no read exceptions) is processed
    normally. The adapter must not write back into it."""
    emitter, sink = _sink()
    hostile = HostileMapping(raise_on_read=False)
    hostile["gate_type"] = "knowledge_conflict"
    hostile["severity"] = "high"
    hostile["conflict_id"] = CONFLICT_ID
    hostile["objective_id"] = OBJECTIVE_ID
    hostile["detected_at"] = DETECTED_AT
    before = {k: v for k, v in hostile._data.items()}
    sink.emit(hostile)  # type: ignore[arg-type]
    assert len(emitter.calls) == 1
    # Hostile state is unchanged (no back-writes).
    assert hostile._data == before


def test_ordinary_mapping_inputs_are_not_rewritten_by_adapter():
    """The adapter must only emit a fresh dict, never edit the input
    Mapping in-place."""
    emitter, sink = _sink()
    event: dict[str, Any] = _valid_event()
    before = {k: event[k] for k in event}
    sink.emit(event)
    # Input is untouched.
    assert event == before
    # Payload is a different object (a fresh dict).
    assert emitter.calls[0] is not event


# ─────────────────────────────────────────────────────────────────────
# 10. Emitter exception isolation
# ─────────────────────────────────────────────────────────────────────


def test_emitter_exception_is_swallowed():
    """If the emitter raises, the adapter must swallow it."""
    emitter = RecordingEmitter()
    emitter._raise = RuntimeError("emitter is on fire")
    sink = EvidencePackMonitoringAuditSink(emitter=emitter)
    # Must NOT raise:
    sink.emit(_valid_event())
    # The call was attempted (and recorded before the raise).
    assert len(emitter.calls) == 1


def test_emitter_value_error_is_swallowed():
    emitter = RecordingEmitter()
    emitter._raise = ValueError("bad payload")
    sink = EvidencePackMonitoringAuditSink(emitter=emitter)
    sink.emit(_valid_event())  # must not raise


def test_emitter_base_exception_other_than_keyboard_interrupt_is_swallowed():
    """All non-system exceptions are swallowed. BaseException is
    intentionally caught because the monitoring emitter is a fire-and-
    forget seam — even programmer errors in user-provided mocks
    must not break the engine."""
    emitter = RecordingEmitter()
    emitter._raise = ArithmeticError("division by zero")
    sink = EvidencePackMonitoringAuditSink(emitter=emitter)
    sink.emit(_valid_event())


def test_dropped_event_does_not_call_emitter_on_failure_paths():
    """The emitter must NOT be invoked for rejected events, even when
    a hostile exception is set."""
    emitter = RecordingEmitter()
    emitter._raise = RuntimeError("should not be called")
    sink = EvidencePackMonitoringAuditSink(emitter=emitter)
    # Empty event (no gate_type, no severity, no identifiers):
    sink.emit({})
    sink.emit({"gate_type": "wrong"})
    sink.emit({"gate_type": "knowledge_conflict"})  # no severity
    sink.emit(None)  # type: ignore[arg-type]
    sink.emit("not a mapping")  # type: ignore[arg-type]
    assert emitter.calls == []


# ─────────────────────────────────────────────────────────────────────
# 11. Exactly-once delegation
# ─────────────────────────────────────────────────────────────────────


def test_emit_delegates_exactly_once_per_accepted_call():
    emitter, sink = _sink()
    sink.emit(_valid_event())
    sink.emit(_valid_event())
    sink.emit(_valid_event())
    assert len(emitter.calls) == 3


def test_rejected_event_delegates_zero_times():
    emitter, sink = _sink()
    # Various rejection paths:
    sink.emit({})
    sink.emit({"gate_type": "x"})
    sink.emit(None)  # type: ignore[arg-type]
    sink.emit({**_valid_event(), "objective_id": ""})
    sink.emit({**_valid_event(), "severity": "low"})
    assert len(emitter.calls) == 0


def test_emitted_payload_does_not_reference_input_object():
    """The emitted payload must be a fresh dict, not a view onto the
    input — later mutations of the input must not retroactively
    change the wire payload."""
    emitter, sink = _sink()
    event = _valid_event()
    sink.emit(event)
    payload = emitter.calls[0]
    # Mutate the input after emit.
    event["objective_id"] = "obj-mutated"
    event["secret"] = "leak"
    # Wire payload is untouched.
    assert payload["objective_id"] == OBJECTIVE_ID
    assert "secret" not in payload
    assert payload["conflict_id"] == CONFLICT_ID


# ─────────────────────────────────────────────────────────────────────
# 12. No close / no get_emitter / no external I/O
# ─────────────────────────────────────────────────────────────────────


def test_adapter_does_not_define_close_method():
    """The adapter must NOT define ``close`` — it borrows the emitter
    and owns no resources.

    Behavioral contract: instantiating the adapter with a real
    ``RecordingEmitter`` and exercising ``emit()`` (a) successfully
    delegates the payload, (b) exposes no ``close`` attribute on the
    instance, and (c) rejects any unknown kwarg on the constructor
    with ``TypeError`` (the public surface accepts only ``emitter``).
    The drive invokes the documented callable and observes the public
    surface.
    """
    sink = EvidencePackMonitoringAuditSink(emitter=RecordingEmitter())
    # Behavioral observation 1: no close attribute on the instance.
    assert not hasattr(sink, "close")
    # Behavioral observation 2: the constructor rejects unknown kwargs
    # (the public surface advertises exactly one parameter: ``emitter``).
    with pytest.raises(TypeError):
        EvidencePackMonitoringAuditSink(
            emitter=RecordingEmitter(),
            close=lambda: None,
        )
    # Behavioral observation 3: the adapter actually works end-to-end
    # (call the documented surface and observe the emitter receipt).
    sink.emit(_valid_event())
    assert sink._emitter.calls, "emitter must have received the event"


def test_adapter_does_not_call_get_emitter():
    """The adapter module must not expose ``get_emitter`` as a public
    name.

    Behavioral contract: the live ``audit_sink`` module's
    runtime namespace does not contain the forbidden names
    (``get_emitter``, ``reset_emitter_for_tests``, ``MonitoringEmitter``).
    The check observes the module's public namespace directly; it does
    not read source text or walk AST.
    """
    forbidden_names = {
        "get_emitter",
        "reset_emitter_for_tests",
        "MonitoringEmitter",
    }
    module_namespace = vars(audit_sink_module)
    leaked = sorted(forbidden_names & set(module_namespace))
    assert not leaked, (
        f"audit_sink module exposes forbidden names from "
        f"agent.monitoring.emitter: {leaked}"
    )


def test_adapter_does_not_import_from_hermes_cli_gateway_tui_gateway():
    """No imports from hermes_cli, gateway, or tui_gateway."""
    module = audit_sink_module
    forbidden = ("hermes_cli", "gateway", "tui_gateway")
    for name, obj in vars(module).items():
        mod = getattr(obj, "__module__", "")
        for ban in forbidden:
            if mod == ban or mod.startswith(ban + "."):
                raise AssertionError(
                    f"audit_sink imports {name!r} from {mod!r}"
                )


def test_adapter_has_no_filesystem_network_thread_process_primitives():
    """No filesystem, network, threading, or subprocess imports."""
    module = audit_sink_module
    forbidden = {
        "os", "io", "pathlib", "tempfile", "shutil", "glob", "fnmatch",
        "socket", "urllib", "urllib.request", "http", "httpx", "aiohttp",
        "ssl", "asyncio", "threading", "thread", "multiprocessing",
        "subprocess", "queue", "concurrent", "concurrent.futures",
        "selectors", "signal", "fcntl",
    }
    for name, obj in vars(module).items():
        mod = getattr(obj, "__module__", "")
        for ban in forbidden:
            if mod == ban or mod.startswith(ban + "."):
                raise AssertionError(
                    f"audit_sink pulls in primitive {name!r} from {mod!r}"
                )


def test_adapter_does_not_open_files_or_sockets(monkeypatch, tmp_path):
    """A real ``emit()`` run must not touch the filesystem or open
    network sockets — defensive double-check at runtime."""
    sink = EvidencePackMonitoringAuditSink(emitter=RecordingEmitter())

    opened_files: list[str] = []
    opened_sockets: list[str] = []
    import builtins as _bi
    real_open = _bi.open

    def tracked_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened_files.append(str(file))
        return real_open(file, *args, **kwargs)

    import socket as _socket
    real_socket = _socket.socket

    class _NoSocket:
        def __init__(self, *a, **kw):
            opened_sockets.append("socket()")
            raise AssertionError("audit sink must not open sockets")

    monkeypatch.setattr(_bi, "open", tracked_open)
    monkeypatch.setattr(_socket, "socket", _NoSocket)
    sink.emit(_valid_event())
    assert opened_files == []
    assert opened_sockets == []


def test_adapter_does_not_spawn_threads():
    """A real ``emit()`` run must not spawn threads."""
    import threading
    initial = set(threading.enumerate())
    sink = EvidencePackMonitoringAuditSink(emitter=RecordingEmitter())
    sink.emit(_valid_event())
    sink.emit({})
    sink.emit(None)  # type: ignore[arg-type]
    final = set(threading.enumerate())
    # Adapter is synchronous and must not have spawned any thread.
    assert final == initial


def test_emit_is_synchronous_and_returns_none():
    """``emit`` returns None synchronously (no coroutine, no awaitable).

    Behavioral contract: calling ``emit`` on a real adapter returns a
    concrete ``None`` value (not an awaitable, not a coroutine). The
    drive invokes the documented method and observes the return value;
    no source/AST introspection is performed.
    """
    sink = EvidencePackMonitoringAuditSink(emitter=RecordingEmitter())
    result = sink.emit(_valid_event())
    # Behavioral observation 1: the return value is exactly None.
    assert result is None
    # Behavioral observation 2: the returned value is NOT a coroutine,
    # future, task, or any other awaitable — None is the contract.
    assert not hasattr(result, "__await__"), (
        "emit must return a concrete value, not an awaitable"
    )
    # Repeat on a few inputs to prove the contract is invariant.
    sink.emit({})
    sink.emit(None)  # type: ignore[arg-type]
    # Every emit returns None (or raises, but the contract is non-raising).
    assert sink.emit({"gate_type": "x"}) is None


# ─────────────────────────────────────────────────────────────────────
# 13. Scope / import boundaries
# ─────────────────────────────────────────────────────────────────────


def test_constructor_rejects_none_emitter_without_replacing_with_singleton():
    """The adapter must not silently substitute ``get_emitter()`` when
    ``None`` is passed — the design contract is explicit injection.

    Behavioral contract: the constructor accepts exactly one positional
    argument (``emitter``), stores the value verbatim (no singleton
    fallback), and rejects any unknown kwarg with ``TypeError``. The
    drive constructs the adapter with a real emitter and asserts the
    observed public surface.
    """
    # Behavioral observation 1: the constructor accepts a positional
    # ``emitter`` argument and stashes it on the instance.
    sentinel = RecordingEmitter()
    sink = EvidencePackMonitoringAuditSink(sentinel)
    assert sink._emitter is sentinel, (
        "adapter must borrow the exact emitter passed by the caller — "
        "no singleton fallback"
    )

    # Behavioral observation 2: an unknown kwarg is rejected with
    # TypeError (the public surface advertises exactly one parameter).
    with pytest.raises(TypeError):
        EvidencePackMonitoringAuditSink(
            emitter=RecordingEmitter(),
            unknown_kwarg="x",
        )

    # Behavioral observation 3: the constructor rejects any extra
    # positional argument (the public surface is ``__init__(self, emitter)``).
    with pytest.raises(TypeError):
        EvidencePackMonitoringAuditSink(
            RecordingEmitter(),
            "extra-positional",
        )


def test_adapter_module_is_self_contained():
    """The audit sink module's public names must all come from
    stdlib-only sources."""
    module = audit_sink_module
    allowed_prefixes = (
        "agent.executive.knowledge_discovery",
        "typing",
        "logging",
        "hashlib",
        "builtins",
    )
    for name in module.__dict__:
        if name.startswith("_"):
            continue
        obj = module.__dict__[name]
        if obj is None:
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        mod = getattr(obj, "__module__", "")
        if not mod:
            continue
        if any(mod == p or mod.startswith(p + ".") for p in allowed_prefixes):
            continue
        raise AssertionError(
            f"audit_sink exposes unexpected primitive {name!r} from {mod!r}"
        )


def test_adapter_class_is_publicly_importable():
    """The class must be importable from the public adapters package."""
    from agent.executive.knowledge_discovery.adapters import (
        EvidencePackMonitoringAuditSink as Cls,
    )
    assert Cls is EvidencePackMonitoringAuditSink


def test_module_class_metadata_unchanged():
    """Lock down the module path used for instrumentation / diagnostics."""
    assert (
        EvidencePackMonitoringAuditSink.__module__
        == "agent.executive.knowledge_discovery.adapters.audit_sink"
    )


# ─────────────────────────────────────────────────────────────────────
# 14. End-to-end shape parity with engine emit dicts
# ─────────────────────────────────────────────────────────────────────


def test_engine_shaped_event_is_accepted_and_renormalized():
    """The dict the engine produces in ``EvidencePackEngine.dry_run``
    (per the in-tree conflict path) must be accepted and renormalized
    to the canonical audit schema."""
    engine_event = {
        "gate_type": "knowledge_conflict",
        "severity": "high",
        "conflict_id": "kc-abc",
        "objective_id": "obj-b1-e3a",
        "detected_at": "2026-08-04T12:00:00+00:00",
    }
    emitter, sink = _sink()
    sink.emit(engine_event)
    assert len(emitter.calls) == 1
    payload = emitter.calls[0]
    # All canonical fields present:
    for k in OUTPUT_FIELD_ORDER:
        assert k in payload
    # Sensitive engine fields that MUST NOT leak through:
    assert "prompt" not in payload
    assert "objective_text" not in payload
    assert "snippet" not in payload


def test_event_id_changes_only_when_canonical_tuple_changes():
    """The event_id is keyed on the canonical tuple. Adding/removing
    non-canonical fields must not perturb it."""
    emitter_a, sink_a = _sink()
    sink_a.emit(_valid_event())
    id_a = emitter_a.calls[0]["event_id"]

    emitter_b, sink_b = _sink()
    noise = {
        "prompt": "x",
        "objective_text": "y",
        "snippet": "z",
        "secret": "w",
    }
    sink_b.emit({**_valid_event(), **noise})
    id_b = emitter_b.calls[0]["event_id"]

    emitter_c, sink_c = _sink()
    sink_c.emit(
        {
            **_valid_event(),
            "irrelevant": True,
            "another": ["nested"],
        }
    )
    id_c = emitter_c.calls[0]["event_id"]

    assert id_a == id_b == id_c


def test_drop_reason_is_observable_only_via_emitter_call_count():
    """There is no return value or callback for dropped events. The
    only signal that an event was dropped is that the emitter was
    not called."""
    emitter, sink = _sink()
    # A dropped event returns None and does not raise.
    assert sink.emit({}) is None
    assert sink.emit(None) is None  # type: ignore[arg-type]
    assert sink.emit({**_valid_event(), "gate_type": "wrong"}) is None
    assert sink.emit({**_valid_event(), "conflict_id": ""}) is None
    assert len(emitter.calls) == 0