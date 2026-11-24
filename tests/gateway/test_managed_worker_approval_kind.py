"""A schedule approval keeps its card kind across a managed worker: the worker's frame and the
gateway's frame check carry ``kind``, so the owner sees a schedule card, not a command card."""
from types import SimpleNamespace

from agent.managed_worker import WorkerControls
from gateway.session_managed_worker import _prompt_frame


def test_schedule_kind_crosses_the_worker_frame():
    sent = []
    controls = WorkerControls.__new__(WorkerControls)  # no reader thread: only the outbound frame is under test
    controls.channel = SimpleNamespace(send=lambda kind, data: sent.append({"type": kind, "data": data}))
    controls.route = "agent:main:local:dm:x"
    controls.approval({"request_id": "r1", "command": "<cronjob_manage>", "description": "Schedule water",
                       "allow_session": True, "allow_permanent": True, "kind": "schedule"})
    assert sent[0]["data"]["kind"] == "schedule"

    registered = []
    live = SimpleNamespace(controls=SimpleNamespace(remote_responders={}, pending={}), route="r")
    authority = SimpleNamespace(sessions={"s": live},
                                register_approval=lambda sid, gen, route, data: registered.append(data))
    assert _prompt_frame(authority, SimpleNamespace(session_id="s"), {"generation": 1},
                         SimpleNamespace(respond=None), sent[0])
    assert registered[0]["kind"] == "schedule"
