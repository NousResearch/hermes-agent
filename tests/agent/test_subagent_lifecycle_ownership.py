"""Public lifecycle ownership must follow the originating conversation across rotation."""
import dataclasses
import threading
import uuid
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import (
    _REGISTRY,
    SubagentLaunchRequest,
    SubagentLifecycleService,
    SubagentState,
)
from hermes_cli.plugin_host_wire import decode, encode
from hermes_state import SessionDB
from tools.delegate_tool_registry import _register_subagent, _unregister_subagent


class Parent:
    def __init__(self, session_id, db=None):
        self.session_id = session_id
        self._session_db = db
        self.enabled_toolsets = None


@pytest.fixture
def owned_child(monkeypatch, tmp_path, request):
    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session("owner-old", "cli")
    owner = Parent(getattr(request, "param", "owner-old"), db)
    active = [owner]
    service = SubagentLifecycleService(lambda: active[0])
    started, release, interrupted = threading.Event(), threading.Event(), threading.Event()
    queued = []
    child = SimpleNamespace(_subagent_id="sa-ownership-" + uuid.uuid4().hex[:8],
                            provider="test", model="test-model", _delegate_role="leaf", _delegate_depth=1,
                            hard_interrupt=lambda *args, **kwargs: interrupted.set(),
                            steer=lambda text: queued.append(text) is None)

    def worker(*args):
        _register_subagent({"subagent_id": child._subagent_id, "agent": child})
        try:
            started.set()
            assert release.wait(15), "test did not release its child"
            return {"status": "interrupted" if interrupted.is_set() else "completed", "summary": "partial report"}
        finally:
            _unregister_subagent(child._subagent_id, agent=child)

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", lambda **kwargs: child)
    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", worker)
    monkeypatch.setattr("tools.delegate_tool._resolve_delegation_credentials", lambda *args: {
        "model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None})
    handle = service.launch(SubagentLaunchRequest(goal="inspect ownership"))
    try:
        assert started.wait(5)
        yield SimpleNamespace(db=db, owner=owner, active=active, service=service, handle=handle,
                              release=release, queued=queued, interrupted=interrupted)
    finally:
        release.set()
        with _REGISTRY.lock:
            record = _REGISTRY.records.get(handle.subagent_id)
        if record is not None:
            record.future.result(timeout=5)
            with _REGISTRY.lock:
                _REGISTRY.records.pop(handle.subagent_id, None)
        db.close()


@pytest.mark.parametrize("rebuilt", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_rotated_owner_keeps_live_controls_and_terminal_receipt(owned_child, rebuilt, wire):
    rig = owned_child
    snapshot = dataclasses.asdict(rig.handle)
    rig.db.end_session("owner-old", "compression")
    rig.db.create_session("owner-new", "cli", parent_session_id="owner-old")
    rig.db.append_message("owner-new", "user", "continue")
    assert rig.db.resolve_resume_session_id("owner-old") == "owner-new"
    if rebuilt:
        rig.active[0] = Parent("owner-new", rig.db)
    else:
        rig.owner.session_id = "owner-new"
        rig.owner._session_db = None  # native identity ownership also works without a DB
    handle = decode(encode(rig.handle)) if wire else rig.handle
    service = rig.service
    assert service.status(handle).state is SubagentState.RUNNING
    assert service.wait(handle, timeout_seconds=0).timed_out
    assert service.result(handle).error_classification == "NOT_READY"
    assert service.reconnect(handle).connected
    assert service.steer(handle, "focus on the parser")
    assert rig.queued == ["focus on the parser"]
    assert service.cancel(handle, reason="stop now").accepted
    assert service.status(handle).state is SubagentState.CANCEL_REQUESTED
    rig.release.set()
    assert service.wait(handle, timeout_seconds=5).state is SubagentState.CANCELLED
    assert service.result(handle).summary == "partial report"
    assert service.reconnect(handle).connected
    assert dataclasses.asdict(rig.handle) == snapshot  # never remint the capability for a new session id


@pytest.mark.parametrize("owned_child", [None], indirect=True)
def test_sessionless_owner_requires_live_identity(owned_child):
    rig = owned_child
    assert rig.service.status(rig.handle).state is SubagentState.RUNNING
    for unrelated in (None, Parent(None)):
        rig.active[0] = unrelated
        assert rig.service.status(rig.handle).state is SubagentState.UNKNOWN
        assert not rig.service.steer(rig.handle, "not the owner")
        assert not rig.service.cancel(rig.handle, reason="not the owner").accepted
    rig.active[0] = rig.owner
    rig.release.set()
    assert rig.service.wait(rig.handle, timeout_seconds=5).state is SubagentState.SUCCEEDED
    assert rig.service.result(rig.handle).summary == "partial report"


@pytest.mark.parametrize("caller", ["foreign", "branch", "reset", "missing", "forged"])
def test_nonowners_and_invalid_capabilities_still_fail_closed(owned_child, caller):
    rig = owned_child
    handle = rig.handle
    if caller == "forged":
        handle = dataclasses.replace(handle, capability="forged")
    elif caller == "missing":
        rig.active[0] = None
    else:
        if caller in {"branch", "reset"}:
            rig.db.end_session("owner-old", "compression")
            rig.db.create_session("owner-new", "cli", parent_session_id="owner-old")
            rig.db.append_message("owner-new", "user", "continuation")
        config = {"_branched_from" if caller == "branch" else "_reset_from": "owner-old"} if caller != "foreign" else {}
        rig.db.create_session(caller, "cli", parent_session_id=None if caller == "foreign" else "owner-old", model_config=config)
        rig.db.append_message(caller, "user", "different conversation")
        if caller in {"branch", "reset"}:
            assert rig.db.resolve_resume_session_id("owner-old") == "owner-new"
        rig.active[0] = Parent(caller, rig.db)
    service = rig.service
    assert service.status(handle).state is SubagentState.UNKNOWN
    assert service.wait(handle, timeout_seconds=0).diagnostic == "UNKNOWN_HANDLE"
    assert service.result(handle).error_classification == "UNKNOWN_HANDLE"
    assert not service.reconnect(handle).connected
    assert not service.steer(handle, "foreign instruction")
    assert not service.cancel(handle, reason="foreign stop").accepted
    assert rig.queued == [] and not rig.interrupted.is_set()
