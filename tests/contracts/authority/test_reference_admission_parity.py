"""Reference admission payloads accept only plain dicts."""

import dataclasses
from array import array
from collections import UserDict
import sqlite3
from types import MappingProxyType

import pytest

from persistence import AuthorityRefusal, FenceStale, PermanentStoreError, SessionId, UnknownExecution
from persistence.authority.admissions import AdmissionOutcome, AdmissionRecord
from persistence.conversation.sessions import NewSession
from tests.contracts._reference import InMemoryBackend
from tests.contracts._scopes import profile_scope


def _plain(claimed):
    return AdmissionRecord(**{f.name: getattr(claimed, f.name) for f in dataclasses.fields(AdmissionRecord)})


def _counters(backend, scope, session_id):
    row = backend.open(scope).sessions.get(session_id)
    return None if row is None else (row.runtime_revision, row.runtime_generation)


@pytest.mark.parametrize("mapping", [UserDict, MappingProxyType])
def test_admission_payloads_reject_non_dict_mappings(mapping):
    backend = InMemoryBackend()
    scope = profile_scope("payloads")
    session_id = SessionId("s")
    try:
        backend.open(scope).sessions.create(NewSession(session_id=session_id, source="test"))
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        request = dict(epoch=epoch, principal_id="p", session_id=session_id, request_id="r")
        with pytest.raises(ValueError):
            backend.open(scope).admissions.admit(**request, payload=mapping({"input": "value"}))
        assert backend.open(scope).admissions.list(session_id=session_id, pending_only=False) == []
        row = backend.open(scope).admissions.admit(**request, payload={"input": "value"})[0]
        claimed = backend.open(scope).admissions.claim(epoch=epoch, session_id=session_id)
        settlement = dict(epoch=epoch, admission_id=row.admission_id, generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED)
        # A mapping result is stored as its dict.
        backend.open(scope).admissions.settle(**settlement, result=mapping({"answer": "value"}))
        saved = backend.open(scope).admissions.settled_result(admission_id=row.admission_id)
        assert saved.result == {"answer": "value"}
    finally:
        backend.close()


@pytest.mark.parametrize("case", ["principal-empty", "session-none", "request-int", "payload-list", "payload-mapping", "payload-nan", "cancelled-outcome", "begin-empty", "begin-none", "begin-int"])
def test_reference_preflight_preserves_legacy_error(case):
    backend = InMemoryBackend()
    scope = profile_scope("envelopes")
    try:
        backend.open(scope).sessions.create(NewSession(session_id="s", source="test"))
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        args = dict(epoch=epoch, principal_id="p", session_id="s", request_id="r", payload={})
        row = backend.open(scope).admissions.admit(**args)[0]
        claimed = backend.open(scope).admissions.claim(epoch=epoch, session_id="s")
        with pytest.raises(AuthorityRefusal) as refused:
            if case.startswith("begin-"):
                backend.open(scope).runtime.begin_epoch(instance_id={"begin-empty": "", "begin-none": None, "begin-int": 7}[case])
            elif case.startswith("result-") or case == "cancelled-outcome":
                backend.open(scope).admissions.settle(epoch=0, admission_id=row.admission_id, generation=claimed.generation, outcome=AdmissionOutcome.CANCELLED if case == "cancelled-outcome" else AdmissionOutcome.COMPLETED, result={"n": float("nan")} if case == "result-nan" else [])
            else:
                field, value = {"principal-empty": ("principal_id", ""), "session-none": ("session_id", None), "request-int": ("request_id", 7), "payload-list": ("payload", []), "payload-mapping": ("payload", UserDict({})), "payload-nan": ("payload", {"n": float("nan")})}[case]
                backend.open(scope).admissions.admit(**(args | {field: value, "epoch": 0}))
        assert type(refused.value) is AuthorityRefusal
        assert refused.value.args == ("invalid_params",)
        assert refused.value.reason == "invalid_params"
        if case.endswith("-nan"):
            assert type(refused.value.__cause__) is ValueError
            assert "Out of range float values" in str(refused.value.__cause__)
        else:
            assert refused.value.__cause__ is None
        assert backend.open(scope).admissions.get(admission_id=row.admission_id) == _plain(claimed)
    finally:
        backend.close()


@pytest.mark.parametrize("case,expected,reason", [("epoch", FenceStale, "stale_epoch"), ("missing", AuthorityRefusal, "not_found"), ("conflict", AuthorityRefusal, "admission_conflict"), ("generation", FenceStale, "stale_generation")])
def test_reference_refusal_carries_the_legacy_reason(case, expected, reason):
    backend = InMemoryBackend()
    scope = profile_scope("causes")
    try:
        backend.open(scope).sessions.create(NewSession(session_id="s", source="test"))
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        args = dict(epoch=epoch, principal_id="p", session_id="s", request_id="r", payload={})
        row = backend.open(scope).admissions.admit(**args)[0]
        with pytest.raises(expected) as refused:
            if case == "epoch":
                backend.open(scope).admissions.claim(epoch=0, session_id="s")
            elif case == "missing":
                backend.open(scope).admissions.claim(epoch=epoch, session_id="missing")
            elif case == "conflict":
                backend.open(scope).admissions.admit(**(args | {"payload": {"changed": True}}))
            else:
                backend.open(scope).admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=1, outcome=AdmissionOutcome.COMPLETED)
        assert isinstance(refused.value, AuthorityRefusal)
        assert refused.value.args == (reason,)
        assert refused.value.reason == reason
        assert backend.open(scope).admissions.get(admission_id=row.admission_id) == row
    finally:
        backend.close()


@pytest.mark.parametrize("pending_only,visible", [("0", False), (0.5, False), (False, True), (0, True), ("", True), (None, True), ("1", False), (True, False), (7, False)])
def test_reference_list_filters_on_the_pending_flags_truthiness(pending_only, visible):
    backend = InMemoryBackend()
    scope = profile_scope("pending")
    try:
        backend.open(scope).sessions.create(NewSession(session_id="s", source="test"))
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        row = backend.open(scope).admissions.admit(epoch=epoch, principal_id="p", session_id="s", request_id="r", payload={})[0]
        terminal = backend.open(scope).admissions.cancel(epoch=epoch, admission_id=row.admission_id)
        rows = backend.open(scope).admissions.list(session_id="s", pending_only=pending_only)
        assert rows == ([terminal] if visible else [])
        assert backend.open(scope).admissions.get(admission_id=row.admission_id) == terminal
    finally:
        backend.close()



@pytest.mark.parametrize("admission_id,expected", [(None, TypeError), (7, TypeError), (True, TypeError), (b"s", TypeError), ([], sqlite3.ProgrammingError)])
def test_reference_get_preserves_identifier_lookup_refusals(admission_id, expected):
    backend = InMemoryBackend()
    scope = profile_scope("lookup")
    try:
        with pytest.raises(expected) as refused:
            backend.open(scope).admissions.get(admission_id=admission_id)
        assert type(refused.value) is expected
        assert refused.value.__cause__ is None
        assert backend.open(scope).admissions.get(admission_id="") is None
    finally:
        backend.close()


@pytest.mark.parametrize("field", ["principal_id", "request_id"])
def test_reference_find_preserves_filter_binding_order(field):
    backend = InMemoryBackend()
    scope = profile_scope("find-bind")
    try:
        backend.open(scope).sessions.create(NewSession(session_id="s", source="test"))
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        row = backend.open(scope).admissions.admit(epoch=epoch, principal_id="7", session_id="s", request_id="7", payload={})[0]
        assert backend.open(scope).admissions.match(**{field: 7}) == [row]
        assert backend.open(scope).admissions.match(**{field: b"7"}) == []
        with pytest.raises(PermanentStoreError) as refused:
            backend.open(scope).admissions.match(**{field: []})
        assert type(refused.value.__cause__) is sqlite3.ProgrammingError
        assert refused.value.__cause__.args == ("Error binding parameter 1: type 'list' is not supported",)
        assert backend.open(scope).admissions.get(admission_id=row.admission_id) == row
    finally:
        backend.close()


@pytest.mark.parametrize("intent", ["", None, 7, True, [], b"queue", "invalid"])
@pytest.mark.parametrize("epoch", [0, 1])
def test_reference_admit_checks_intent_before_payload_and_epoch(intent, epoch):
    backend = InMemoryBackend()
    scope = profile_scope("intent-order")
    try:
        backend.open(scope).sessions.create(NewSession(session_id="s", source="test"))
        backend.open(scope).runtime.begin_epoch(instance_id="owner")
        with pytest.raises(AuthorityRefusal) as refused:
            backend.open(scope).admissions.admit(epoch=epoch, principal_id="p", session_id="s", request_id="r", payload={"n": float("nan")}, intent=intent)
        assert type(refused.value) is AuthorityRefusal
        assert refused.value.args == ("invalid_params",)
        assert refused.value.__cause__ is None
        assert backend.open(scope).admissions.list(session_id="s", pending_only=False) == []
    finally:
        backend.close()


@pytest.mark.parametrize("outcome", ["", None, 7, True, [], b"completed", "invalid"])
@pytest.mark.parametrize("epoch", [0, 1])
def test_reference_settle_checks_outcome_before_result_and_epoch(outcome, epoch):
    backend = InMemoryBackend()
    scope = profile_scope("outcome-order")
    try:
        backend.open(scope).sessions.create(NewSession(session_id="s", source="test"))
        current = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        row = backend.open(scope).admissions.admit(epoch=current, principal_id="p", session_id="s", request_id="r", payload={})[0]
        claimed = backend.open(scope).admissions.claim(epoch=current, session_id="s")
        before = _counters(backend, scope, "s")
        with pytest.raises(AuthorityRefusal) as refused:
            backend.open(scope).admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=claimed.generation, outcome=outcome, result={"n": float("nan")})
        assert type(refused.value) is AuthorityRefusal
        assert refused.value.args == ("invalid_params",)
        assert refused.value.__cause__ is None
        assert backend.open(scope).admissions.get(admission_id=row.admission_id) == _plain(claimed)
        assert _counters(backend, scope, "s") == before
    finally:
        backend.close()


@pytest.mark.parametrize("identifier,session_id", [(True, "1"), (False, "0"), (1, "1"), (0, "0"), (1e20, "1.0e+20"), (-0.0, "0.0")])
@pytest.mark.parametrize("unknown", [False, True])
def test_reference_claim_keeps_worker_barriers_for_bound_identifiers(identifier, session_id, unknown):
    from persistence.workers.executions import WorkerKind

    backend = InMemoryBackend()
    scope = profile_scope("worker-binding")
    try:
        backend.open(scope).sessions.create(NewSession(session_id=session_id, source="test"))
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        row = backend.open(scope).admissions.admit(epoch=epoch, principal_id="p", session_id=session_id, request_id="r", payload={})[0]
        backend.open(scope).worker_executions.register(epoch=epoch, execution_id="worker", session_id=session_id, generation=0, kind=WorkerKind.COMPUTE, adoption_secret="proof")
        if unknown:
            epoch = backend.open(scope).runtime.begin_epoch(instance_id="next-owner")
            backend.open(scope).admissions.recover(epoch=epoch)
        before = _counters(backend, scope, session_id)
        if unknown:
            with pytest.raises(UnknownExecution) as refused:
                backend.open(scope).admissions.claim(epoch=epoch, session_id=identifier)
            assert refused.value.reason == "unknown_execution"
        else:
            assert backend.open(scope).admissions.claim(epoch=epoch, session_id=identifier) is None
        assert backend.open(scope).admissions.get(admission_id=row.admission_id) == row
        assert _counters(backend, scope, session_id) == before
    finally:
        backend.close()


class ChangedInt(int):
    def __int__(self):
        return 8


@pytest.mark.parametrize("verb", ["list", "find-principal", "find-request", "claim"])
def test_reference_integer_subclass_uses_underlying_identifier(verb):
    backend = InMemoryBackend()
    scope = profile_scope("integer-binding")
    try:
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        rows = {}
        for session_id in ("7", "8"):
            backend.open(scope).sessions.create(NewSession(session_id=session_id, source="test"))
            rows[session_id] = backend.open(scope).admissions.admit(epoch=epoch, principal_id=session_id, session_id=session_id, request_id=session_id, payload={})[0]
        before = _counters(backend, scope, "8")
        if verb == "claim":
            claimed = backend.open(scope).admissions.claim(epoch=epoch, session_id=ChangedInt(7))
            assert claimed.admission_id == rows["7"].admission_id
            assert claimed.target_session_id == "7"
            assert claimed.generation == 1
        elif verb == "list":
            assert backend.open(scope).admissions.list(session_id=ChangedInt(7), pending_only=False) == [rows["7"]]
        else:
            field = "principal_id" if verb == "find-principal" else "request_id"
            assert backend.open(scope).admissions.match(**{field: ChangedInt(7)}) == [rows["7"]]
        assert backend.open(scope).admissions.get(admission_id=rows["8"].admission_id) == rows["8"]
        assert _counters(backend, scope, "8") == before
    finally:
        backend.close()


@pytest.mark.parametrize("identifier,expected,neighbor", [(1.2345678901234567, "1.2345678901234567", "1.23456789012346"), (1000000000000001.0, "1000000000000001.0", "1.0e+15")])
@pytest.mark.parametrize("verb", ["list", "find-principal", "find-request", "claim"])
def test_reference_float_identifiers_keep_colliding_identities_distinct(identifier, expected, neighbor, verb):
    backend = InMemoryBackend()
    scope = profile_scope("float-binding")
    try:
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        rows = {}
        for session_id in (expected, neighbor):
            backend.open(scope).sessions.create(NewSession(session_id=session_id, source="test"))
            rows[session_id] = backend.open(scope).admissions.admit(epoch=epoch, principal_id=session_id, session_id=session_id, request_id=session_id, payload={})[0]
        before = _counters(backend, scope, neighbor)
        if verb == "claim":
            claimed = backend.open(scope).admissions.claim(epoch=epoch, session_id=identifier)
            assert claimed.admission_id == rows[expected].admission_id
            assert claimed.target_session_id == expected
            assert claimed.generation == 1
        elif verb == "list":
            assert backend.open(scope).admissions.list(session_id=identifier, pending_only=False) == [rows[expected]]
        else:
            field = "principal_id" if verb == "find-principal" else "request_id"
            assert backend.open(scope).admissions.match(**{field: identifier}) == [rows[expected]]
        assert backend.open(scope).admissions.get(admission_id=rows[neighbor].admission_id) == rows[neighbor]
        assert _counters(backend, scope, neighbor) == before
    finally:
        backend.close()


@pytest.mark.parametrize("identifier,session_id", [(array("B", b"7"), "7"), (array("b", [115]), "s")])
@pytest.mark.parametrize("verb", ["list", "find-principal", "find-request", "get", "claim"])
def test_reference_buffer_identifiers_remain_binary(identifier, session_id, verb):
    backend = InMemoryBackend()
    scope = profile_scope("buffer-binding")
    try:
        epoch = backend.open(scope).runtime.begin_epoch(instance_id="owner")
        backend.open(scope).sessions.create(NewSession(session_id=session_id, source="test"))
        row = backend.open(scope).admissions.admit(epoch=epoch, principal_id=session_id, session_id=session_id, request_id=session_id, payload={})[0]
        before = _counters(backend, scope, session_id)
        if verb == "get":
            with pytest.raises(TypeError) as refused:
                backend.open(scope).admissions.get(admission_id=identifier)
            assert str(refused.value) == 'can only concatenate str (not "array.array") to str'
            assert refused.value.__cause__ is None
        elif verb == "claim":
            with pytest.raises(AuthorityRefusal, match="^not_found$"):
                backend.open(scope).admissions.claim(epoch=epoch, session_id=identifier)
        elif verb == "list":
            assert backend.open(scope).admissions.list(session_id=identifier, pending_only=False) == []
        else:
            field = "principal_id" if verb == "find-principal" else "request_id"
            assert backend.open(scope).admissions.match(**{field: identifier}) == []
        assert backend.open(scope).admissions.list(session_id=session_id, pending_only=False) == [row]
        assert _counters(backend, scope, session_id) == before
    finally:
        backend.close()
