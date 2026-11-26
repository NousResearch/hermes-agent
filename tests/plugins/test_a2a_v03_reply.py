"""Legacy ``message/send`` callers get an A2A v0.3-shaped reply (kind discriminators, lowercase
states/roles). Telnyx AI Assistants parse only v0.3 and report "no answer" for v1.0 replies."""
from plugins.platforms.a2a import protocol


def test_to_v03_task_completed_reply():
    task = protocol.build_task("task-1", "ctx-1", protocol.STATE_COMPLETED, "PONG")
    out = protocol.to_v03_task(task)
    assert out["kind"] == "task" and out["id"] == "task-1" and out["contextId"] == "ctx-1"
    assert out["status"]["state"] == "completed"
    msg = out["status"]["message"]
    assert msg["kind"] == "message" and msg["role"] == "agent"
    assert msg["parts"] == [{"text": "PONG", "mediaType": "text/plain", "kind": "text"}]
    assert out["artifacts"][0]["parts"][0]["kind"] == "text"
    # the v1.0 task itself is untouched
    assert task["status"]["state"] == protocol.STATE_COMPLETED and "kind" not in task


def test_to_v03_task_maps_every_state():
    for v1, v03 in [(protocol.STATE_FAILED, "failed"), (protocol.STATE_WORKING, "working"),
                    (protocol.STATE_INPUT_REQUIRED, "input-required"), (protocol.STATE_REJECTED, "rejected")]:
        assert protocol.to_v03_task(protocol.build_task("t", "c", v1))["status"]["state"] == v03
