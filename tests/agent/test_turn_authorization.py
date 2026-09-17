from __future__ import annotations

import json
import pickle
import threading
import time
from types import SimpleNamespace

import pytest


PERSON_A = "a" * 64
PERSON_B = "b" * 64
ADMISSION_A = "1" * 32


def test_person_token_requires_stable_principal_id():
    from agent.turn_authorization import TurnAuthorization

    with pytest.raises(ValueError, match="principal id is required"):
        TurnAuthorization.from_raw("person-token", expires_at=time.time() + 3600)


def test_person_prompt_requires_well_formed_private_admission_id():
    from agent.turn_authorization import TurnAuthorization

    with pytest.raises(ValueError, match="admission id is required"):
        TurnAuthorization.from_raw(
            "person-token",
            expires_at=time.time() + 3600,
            principal_id=PERSON_A,
            require_admission=True,
        )
    with pytest.raises(ValueError, match="opaque identifier"):
        TurnAuthorization.from_raw(
            "person-token",
            expires_at=time.time() + 3600,
            principal_id=PERSON_A,
            admission_id="browser-controlled",
            require_admission=True,
        )

    holder = TurnAuthorization.from_raw(
        "person-token",
        expires_at=time.time() + 3600,
        principal_id=PERSON_A,
        admission_id=ADMISSION_A,
        require_admission=True,
    )
    assert holder._fizko_admission_id() == ADMISSION_A


@pytest.mark.parametrize("principal_id", ["short", "A" * 64, 123])
def test_person_token_rejects_malformed_principal_id(principal_id):
    from agent.turn_authorization import TurnAuthorization

    with pytest.raises(ValueError, match="SHA-256 identifier"):
        TurnAuthorization.from_raw(
            "person-token",
            expires_at=time.time() + 3600,
            principal_id=principal_id,
        )


def test_stable_principal_owns_rotated_credentials_without_sharing_bearer():
    from agent.turn_authorization import TurnAuthorization

    first = TurnAuthorization.from_raw(
        "person-token-t1", expires_at=time.time() + 3600, principal_id=PERSON_A
    )
    reconnected = TurnAuthorization.from_raw(
        "person-token-t2", expires_at=time.time() + 3600, principal_id=PERSON_A
    )
    stranger = TurnAuthorization.from_raw(
        "person-token-t3", expires_at=time.time() + 3600, principal_id=PERSON_B
    )

    assert first.same_principal(reconnected)
    assert not first.same_credential(reconnected)
    assert not first.same_principal(stranger)


def test_turn_authorization_holder_is_opaque_redacted_and_nonserializable():
    from agent.turn_authorization import TurnAuthorization

    token = "person-token.sentinel-123"
    holder = TurnAuthorization.from_raw(
        token, expires_at=time.time() + 3600, principal_id=PERSON_A
    )

    assert token not in repr(holder)
    assert token not in str(holder)
    with pytest.raises(TypeError):
        pickle.dumps(holder)
    with pytest.raises(TypeError):
        json.dumps(holder)


def test_person_token_requires_expiry():
    from agent.turn_authorization import TurnAuthorization

    with pytest.raises(ValueError, match="expiry is required"):
        TurnAuthorization.from_raw("person-token")


@pytest.mark.parametrize(
    "expires_at", [True, False, "123", float("inf"), float("nan"), 0, -1, 10**1000]
)
def test_person_token_rejects_invalid_expiry(expires_at):
    from agent.turn_authorization import TurnAuthorization

    with pytest.raises(ValueError, match="Unix timestamp"):
        TurnAuthorization.from_raw(
            "person-token", expires_at=expires_at, principal_id=PERSON_A
        )


def test_personal_descendant_scope_is_blocked_not_static_fallback():
    from agent.turn_authorization import (
        TurnAuthorization,
        current_fizko_authorization_state,
        reset_current_turn_authorization,
        set_current_turn_authorization,
        without_turn_authorization,
    )

    token = set_current_turn_authorization(
        TurnAuthorization.from_raw(
            "person-token", expires_at=time.time() + 3600, principal_id=PERSON_A
        )
    )
    try:
        assert current_fizko_authorization_state()[0] is True
        with without_turn_authorization():
            assert current_fizko_authorization_state() == (True, "")
        assert current_fizko_authorization_state()[0] is True
    finally:
        reset_current_turn_authorization(token)


def test_expired_person_authorization_remains_personal_but_has_no_header():
    from agent.turn_authorization import (
        TurnAuthorization,
        current_fizko_authorization_state,
        reset_current_turn_authorization,
        set_current_turn_authorization,
    )

    holder = TurnAuthorization.from_raw(
        "expired-person", expires_at=1.0, principal_id=PERSON_A
    )
    token = set_current_turn_authorization(holder)
    try:
        assert holder.has_token is True
        assert holder.is_expired is True
        assert current_fizko_authorization_state() == (True, "")
    finally:
        reset_current_turn_authorization(token)


def test_real_delegate_child_worker_does_not_inherit_parent_turn_authorization(monkeypatch):
    from agent.turn_authorization import (
        TurnAuthorization,
        current_fizko_authorization_header,
        current_fizko_authorization_state,
        reset_current_turn_authorization,
        set_current_turn_authorization,
    )
    from tools.delegate_tool_child_run import _ChildRun

    seen = []
    child = SimpleNamespace(
        session_id="child-session",
        run_conversation=lambda **_kwargs: seen.append(current_fizko_authorization_header())
        or {"final_response": "done", "completed": True},
    )
    run = _ChildRun(child, SimpleNamespace(), 0, "work", None, None)
    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: None)

    states = []
    child.run_conversation = lambda **_kwargs: (
        states.append(current_fizko_authorization_state()),
        seen.append(current_fizko_authorization_header()),
    )[-1] or {"final_response": "done", "completed": True}
    token = set_current_turn_authorization(
        TurnAuthorization.from_raw(
            "person-token", expires_at=time.time() + 3600, principal_id=PERSON_A
        )
    )
    try:
        result, error, deferred = run.await_child()
        assert current_fizko_authorization_header() == "Bearer person-token"
    finally:
        reset_current_turn_authorization(token)

    assert result == {"final_response": "done", "completed": True}
    assert error is None
    assert deferred is False
    assert seen == [""]
    assert states == [(True, "")]


def test_detached_process_reader_does_not_retain_parent_turn_authorization(monkeypatch, tmp_path):
    from agent.turn_authorization import (
        TurnAuthorization,
        current_fizko_authorization_header,
        current_fizko_authorization_state,
        reset_current_turn_authorization,
        set_current_turn_authorization,
    )
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(registry, "_prune_if_needed", lambda: None)
    session = registry._new_session("noop", "task", "task", "session", str(tmp_path))
    observed = []
    finished = threading.Event()

    def reader(_session):
        observed.append(
            (current_fizko_authorization_state(), current_fizko_authorization_header())
        )
        finished.set()

    token = set_current_turn_authorization(
        TurnAuthorization.from_raw(
            "detached-parent-token",
            expires_at=time.time() + 3600,
            principal_id=PERSON_A,
        )
    )
    try:
        registry._track_started(session, reader, "test-detached-reader")
        assert finished.wait(timeout=2)
        assert current_fizko_authorization_header() == "Bearer detached-parent-token"
    finally:
        reset_current_turn_authorization(token)

    assert session._reader_thread is not None
    session._reader_thread.join(timeout=2)
    assert observed == [((True, ""), "")]
