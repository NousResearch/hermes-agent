"""A gateway-owned room says which tool a working member is in.

The member's turn runs on the gateway's canonical session path, whose frames never pass through
tui_gateway's room projection: the member authority's ``info`` reports its live tool, and the room
driver's existing info poll carries it into ``groups.state`` ``member_activity``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from tests.waiting import wait_until


def test_member_info_reports_the_running_executions_tool_and_last_frame(hosted_owner):
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from hermes_state_runtime import claim_session_input
    authority, loop, principal, _agent = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='scout', profile='default',
                                 principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    rpc.submit(**coords, session_id=sid, prompt='check the logs', task=TaskIdentity('room', 'task', 'thread', 'turn'),
               execution_generation=1, on_terminal=lambda receipt: None)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    authority.executing[sid] = row['generation']  # what the canonical claim stamps before running
    assert 'tool' not in rpc.info(**coords, session_id=sid)

    authority.publish_execution(sid, row['generation'], 'tool.start', {'tool_id': 'c1', 'name': 'terminal'})
    info = rpc.info(**coords, session_id=sid)
    assert (info['task_id'], info['tool']) == ('task', 'terminal') and info['progress_at'] > 0

    # A frame from another execution generation (a late callback) moves nothing.
    authority.publish_execution(sid, row['generation'] + 1, 'tool.start', {'tool_id': 'c2', 'name': 'web_search'})
    assert rpc.info(**coords, session_id=sid)['tool'] == 'terminal'

    authority.publish_execution(sid, row['generation'], 'tool.complete', {'tool_id': 'c1', 'name': 'terminal'})
    info = rpc.info(**coords, session_id=sid)
    assert 'tool' not in info and info['progress_at'] > 0


class _ToolRPC:
    """Real driver path over a member transport whose ``info`` reports the member's tool, like the
    gateway's ``HostedRoomAuthorityRPC``; its frames never reach this process's room projection."""

    def __init__(self) -> None:
        self.task = None
        self.on_terminal = None
        self.submitted = threading.Event()
        self.sessions = {}

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(self, *, task, on_terminal, **_kwargs):
        self.task, self.on_terminal = task, on_terminal
        self.submitted.set()
        return {"accepted": True}

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        if self.task is None:
            return {"active": False, "task_id": None}
        return {"active": True, "status": "started", "task_id": self.task.task_id, "execution_generation": 1,
                "tool": "terminal", "progress_at": 1234.5}


def test_groups_state_names_the_tool_a_member_transport_reports(tmp_path: Path):
    from tui_gateway.hosted_room_service import HostedRoomService
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=tmp_path / "state.db")
    service.rpc = service.runtime.rpc = _ToolRPC()
    service.local_profiles = lambda: ("default", "scout")
    service.create_room(room_id="room-1", name="Release room", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "scout", "profile": "scout", "handle": "scout"}])
    service.start()
    try:
        service.send(room_id="room-1", event_id="user-1",
                     payload={"text": "@scout check the build logs", "thread_id": "thread-1"})
        assert service.rpc.submitted.wait(timeout=5)

        def scout_activity():
            return next((a for a in service.status("room-1")["member_activity"] if a["member_id"] == "scout"), {})
        wait_until(lambda: scout_activity().get("tool") == "terminal", timeout=5)
        assert scout_activity()["progress_at"] == 1234.5

        service.rpc.on_terminal({"status": "settled", "text": "Build logs are clean."})
        wait_until(lambda: scout_activity() == {}, timeout=5)
        from tui_gateway.hosted_room_member_activity import task_progress
        assert task_progress("room-1", service.rpc.task.task_id) == {}
    finally:
        service.stop(timeout=5)
