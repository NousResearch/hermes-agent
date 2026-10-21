"""Presentation authority belongs to one live canonical cron invocation."""

import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from cron.approval_relay import ApprovalUnavailable
from cron.approval_relay_scope import bind_agent, current_scope, presentation_scope
from cron.execution_identity import enter_cron_execution, exit_cron_execution
from tests.waiting import wait_until


@pytest.fixture
def execution(tmp_path, monkeypatch):
    from gateway import session_cron
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    service = PluginContext(PluginManifest(name="guard", version="0.1.0"), manager).cron_approvals
    owner = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / "state.db"),
                            runner=SimpleNamespace(_draining=False), retired=False)
    agent = SimpleNamespace(session_id="session")
    cancel = threading.Event()
    identity = enter_cron_execution({"id": "daily"}, "fire", {})
    owner_token = session_cron.bind_execution(owner, "session", "daily", "fire")
    state = SimpleNamespace(service=service, agent=agent, owner=owner, cancel=cancel,
                            requests=[], lifetimes=[], entered=threading.Event(), retired=threading.Event())

    @contextlib.contextmanager
    def present(request, lifetime):
        state.requests.append(request)
        state.lifetimes.append(lifetime)
        state.entered.set()
        try:
            yield
        finally:
            state.retired.set()

    service.register_presenter(present)
    try:
        with presentation_scope(cancel):
            bind_agent(agent)
            yield state
    finally:
        service.close()
        session_cron._execution.reset(owner_token)
        exit_cron_execution(identity)


def pending(state, call="call", **kwargs):
    args = dict(session_id=state.agent.session_id, external_session_id="audit", call_id=call,
                deadline=time.monotonic() + 30, check=lambda: None)
    args.update(kwargs)
    return state.service.pending(**args)


@pytest.mark.parametrize("stop", ["cancel", "retire", "drain", "unload", "invocation", "finish"])
def test_lifetime_rejects_late_callbacks_and_retires_prompt(execution, stop):
    state = execution
    invocation_cancelled = threading.Event()

    def check():
        if invocation_cancelled.is_set():
            raise RuntimeError("cancelled invocation")

    with pending(state, check=check):
        assert state.entered.wait(2)
        lifetime = state.lifetimes[0]
        {
            "cancel": state.cancel.set,
            "retire": lambda: setattr(state.owner, "retired", True),
            "drain": lambda: setattr(state.owner.runner, "_draining", True),
            "unload": state.service.close,
            "invocation": invocation_cancelled.set,
            "finish": current_scope().close,
        }[stop]()
        assert state.retired.wait(2)
        effects = []
        with pytest.raises(ApprovalUnavailable):
            lifetime.request(lambda: effects.append("approve"))
        assert effects == []
        invocation_cancelled.clear()  # restoration cannot revive a latched presentation
        with pytest.raises(ApprovalUnavailable):
            lifetime.check()


@pytest.mark.parametrize("mismatch", ["session", "job", "execution", "profile", "agent"])
def test_foreign_binding_cannot_present(execution, tmp_path, mismatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    state = execution
    identity = profile = None
    try:
        if mismatch in ("job", "execution"):
            identity = enter_cron_execution({"id": "other" if mismatch == "job" else "daily"},
                                            "other" if mismatch == "execution" else "fire", {})
        elif mismatch == "profile":
            profile = set_hermes_home_override(tmp_path / "foreign")
        elif mismatch == "agent":
            bind_agent(SimpleNamespace(session_id="foreign"))
        with pending(state, session_id="foreign" if mismatch in ("session", "agent") else "session"):
            pass
        assert state.requests == []
    finally:
        if identity is not None:
            exit_cron_execution(identity)
        if profile is not None:
            reset_hermes_home_override(profile)


def test_rotation_follows_bound_agent_and_duplicate_call_stays_fenced(execution):
    state = execution
    with pending(state):
        assert state.entered.wait(2)
    assert state.retired.wait(2)
    state.agent.session_id = "compressed-session"
    with pending(state, session_id="forged-session"):
        pass
    with pending(state):  # same call after compression must not produce another prompt
        pass
    state.entered.clear()
    with pending(state, "next-call"):
        assert state.entered.wait(2)
    assert [(r.session_id, r.call_id) for r in state.requests] == [
        ("session", "call"), ("compressed-session", "next-call")]


def test_presentation_failure_keeps_audit_wait_and_duplicate_fence(execution):
    state = execution
    scope = current_scope()
    attempts = []

    @contextlib.contextmanager
    def ambiguous_send(request, lifetime):
        attempts.append(request.call_id)
        raise OSError("send outcome unknown")
        yield

    scope._presenters["guard"].callback = ambiguous_send
    audit_waits = []
    with pending(state):
        wait_until(lambda: attempts == ["call"], timeout=2)
        audit_waits.append("polled")
    with pending(state):
        audit_waits.append("polled")
    assert attempts == ["call"]
    assert audit_waits == ["polled", "polled"]


def test_capacity_releases_slots_without_evicting_duplicate_fences(execution, monkeypatch):
    from cron import approval_relay_scope

    monkeypatch.setattr(approval_relay_scope, "MAX_ACTIVE", 1)
    monkeypatch.setattr(approval_relay_scope, "MAX_INVOCATIONS", 2)
    state = execution
    with pending(state):
        assert state.entered.wait(2)
        with pending(state, "blocked"):
            pass
        assert [r.call_id for r in state.requests] == ["call"]
    scope = current_scope()
    wait_until(lambda: not scope._waits, timeout=2)
    state.entered.clear()
    with pending(state, "second"):
        assert state.entered.wait(2)
    wait_until(lambda: not scope._waits, timeout=2)
    with pending(state, "third"):
        pass
    with pending(state):
        pass
    assert [r.call_id for r in state.requests] == ["call", "second"]


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), -1])
def test_invalid_deadline_keeps_external_wait_without_prompt(execution, deadline):
    with pending(execution, deadline=deadline):
        pass
    assert execution.requests == []


@pytest.mark.platforms("posix")
def test_forked_subprocess_cannot_inherit_presentation_authority(execution):
    import os

    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            os.write(writer, b"available" if execution.service.available else b"refused")
        finally:
            os._exit(0)
    os.close(writer)
    try:
        assert os.read(reader, 64) == b"refused"
        assert os.waitpid(pid, 0)[1] == 0
        assert execution.service.available
    finally:
        os.close(reader)
