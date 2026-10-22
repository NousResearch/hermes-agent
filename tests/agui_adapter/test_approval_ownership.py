"""Approval waits belong to an application, not a caller-chosen global id."""

import copy
import json
import shlex

from fastapi.testclient import TestClient
import pytest

from agui_adapter.server import create_app
from hermes_cli.config import save_config
from tests.agui_adapter.test_modernization_probe import body, config_for, events
from tests.agui_adapter.test_conversations import next_user


@pytest.mark.parametrize(
    "approved,status",
    [(False, "resolved"), (1, "resolved"), ("true", "resolved"), (True, "cancelled")],
)
def test_independent_apps_can_park_the_same_thread_id(peer, tmp_path, approved, status):
    target = tmp_path / "must-not-be-removed"
    target.mkdir()
    peer.tool_name = "terminal"
    peer.tool_args = {"command": "rm -r -- " + shlex.quote(str(target))}
    save_config(
        {
            "approvals": {"mode": "manual", "timeout": 10},
            "terminal": {"env": "local"},
            "auxiliary": {"title_generation": {"enabled": False}},
        }
    )
    apps = [create_app(config_for(peer, ["terminal"])) for _ in range(2)]
    with TestClient(apps[0], base_url="http://127.0.0.1") as one:
        with TestClient(apps[1], base_url="http://127.0.0.1") as two:
            request = body("same-client-thread-id")
            first = events(one.post("/", json=request))
            second = events(two.post("/", json=request))
            assert first[-1]["outcome"]["type"] == "interrupt"
            assert second[-1]["outcome"]["type"] == "interrupt"
            assert len(peer.requests) == 2
            # A response addressed to the other app must not consume either wait.
            foreign = copy.deepcopy(request)
            foreign["runId"] = "foreign-resume"
            foreign["resume"] = [
                {
                    "interruptId": first[-1]["outcome"]["interrupts"][0]["id"],
                    "status": "resolved",
                    "payload": {"approved": False},
                }
            ]
            response = two.post("/", json=foreign)
            assert "RUN_ERROR" in response.text
            for client, interrupted in [(two, second), (one, first)]:
                resumed = copy.deepcopy(request)
                resumed["runId"] = "denied-resume"
                resumed["resume"] = [
                    {
                        "interruptId": interrupted[-1]["outcome"]["interrupts"][0][
                            "id"
                        ],
                        "status": status,
                        "payload": {"approved": approved, "scope": [1]},
                    }
                ]
                assert (
                    events(client.post("/", json=resumed))[-1]["outcome"]["type"]
                    == "success"
                )
    assert target.exists()
    assert len(peer.requests) == 4


@pytest.mark.parametrize(
    "scope", [None, [], [1], {}, {"x": 1}, "unknown", "once", "session"]
)
def test_session_approval_is_not_shared_by_apps_or_reused_thread_ids(
    peer, tmp_path, scope
):
    from tools import approval

    target = tmp_path / "owned-session-probe"
    target.mkdir()
    peer.tool_name = "terminal"
    peer.tool_args = {"command": "rm -r -- " + shlex.quote(str(target))}
    save_config(
        {
            "approvals": {"mode": "manual", "timeout": 10},
            "terminal": {"env": "local"},
            "auxiliary": {"title_generation": {"enabled": False}},
        }
    )
    apps = [create_app(config_for(peer, ["terminal"])) for _ in range(2)]
    request = body("same-scope-id")
    first_record = None
    try:
        with TestClient(apps[0], base_url="http://127.0.0.1") as one:
            first = events(one.post("/", json=request))
            first_record = apps[0].state.executions.conversations.get(
                request["threadId"]
            )
            resumed = copy.deepcopy(request)
            resumed["runId"] = "session-approved"
            resumed["resume"] = [
                {
                    "interruptId": first[-1]["outcome"]["interrupts"][0]["id"],
                    "status": "resolved",
                    "payload": {"approved": True, "scope": scope},
                }
            ]
            events(one.post("/", json=resumed))
            assert not target.exists()
            target.mkdir()  # Re-create only this test's disposable fixture.
            with TestClient(apps[1], base_url="http://127.0.0.1") as two:
                second = events(two.post("/", json=request))
                assert second[-1]["outcome"]["type"] == "interrupt", (
                    "session approval leaked across apps"
                )
                assert target.exists()
            # Only a session scope keeps its grant on the next user turn.
            peer.script = lambda incoming, index: (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "session-second",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps(peer.tool_args),
                            },
                        }
                    ],
                }
                if incoming["messages"][-1]["role"] == "user"
                else {"role": "assistant", "content": "SESSION_DONE"}
            )
            warmed = events(one.post("/", json=next_user(request)))
            assert warmed[-1]["outcome"]["type"] == (
                "success" if scope == "session" else "interrupt"
            )
            assert target.exists() is (scope != "session")
        assert first_record is not None
        assert first_record.approval_key not in approval._session_approved
    finally:
        # The unfixed implementation uses the client id as a global grant key.
        approval.clear_session(request["threadId"])
