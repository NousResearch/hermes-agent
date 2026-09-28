"""A native reaction resumes an encrypted split reply after gateway restart."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
    _wait_for,
)


@pytest.fixture
def gateway_config() -> str:
    return (
        "platforms:\n  matrix:\n    enabled: true\n    max_message_length: 500\n"
        "streaming:\n  enabled: false\nupdates:\n  check: false\n"
    )


def test_reaction_to_encrypted_split_final_resumes_after_restart(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
) -> None:
    parts = [
        f"[part:{label}] "
        + (f"The {label} decision belongs to the logical final reply. " * 6).strip()
        for label in ("alpha", "beta", "gamma")
    ]
    final = "\n".join(parts)
    gateway.model.push(
        ToolCall("tool_search", {"queries": ["Matrix reaction follow-up"]}),
        ToolCall(
            "tool_call",
            {
                "calls": [
                    {
                        "name": "matrix_followup",
                        "arguments": {"enabled": True, "emoji": ["👍"]},
                    }
                ]
            },
        ),
        Text(final),
        Text("Reaction follow-up"),
    )

    def gateway_logs() -> str:
        return (
            gateway.container
            .get_wrapped_container()
            .logs()
            .decode(errors="replace")[-6000:]
        )

    def failure_details() -> str:
        code = (
            "import json; from pathlib import Path; "
            "from reaction_followup_state import failure_state; "
            f"print(json.dumps(failure_state(Path('/opt/data'), {live_room.room_id!r})))"
        )
        result = gateway.container.get_wrapped_container().exec_run(
            ["/opt/hermes/.venv/bin/python", "-c", code],
            environment={"PYTHONPATH": "/matrix_live"},
        )
        requests = [
            {
                "model": request.get("model"),
                "messages": [
                    {
                        "role": message.get("role"),
                        "content": str(message.get("content", ""))[:4000],
                        "tool_calls": message.get("tool_calls"),
                        "tool_call_id": message.get("tool_call_id"),
                    }
                    for message in request["messages"][-24:]
                ],
            }
            for request in gateway.model.main_requests()[-12:]
        ]
        logs = "\n".join(
            f"{path.name}:\n{path.read_text(errors='replace')[-131072:]}"
            for path in sorted((gateway.home / "logs").glob("gateway.log*"))[-2:]
        )
        details = (
            f"Model requests ({len(gateway.model.main_requests())}):\n"
            f"{json.dumps(requests)}\nPersisted state (exit {result.exit_code}):\n"
            f"{result.output.decode(errors='replace')[-65536:]}\n"
            f"Gateway file logs:\n{logs}\nGateway container logs:\n{gateway_logs()}"
        )
        for secret in (live_room.bot.access_token, live_room.observer.access_token):
            details = details.replace(secret, "<redacted>")
        return details

    def exchange(**kwargs) -> dict:
        code = (
            "import asyncio, json; "
            "from reaction_followup_client import _exchange; "
            f"args = json.loads({json.dumps(kwargs)!r}); "
            "print(json.dumps(asyncio.run(asyncio.wait_for(_exchange("
            f"{live_room.room_id!r}, {live_room.bot.user_id!r}, {parts!r}, **args), timeout=25))))"
        )
        output = linux_nio_observer.run_python(code)
        return json.loads(output.splitlines()[-1])

    def persisted_state(room_id: str, thread_id: str) -> dict:
        code = (
            "import sys, json; from pathlib import Path; "
            "sys.path.insert(0, '/matrix_live'); "
            "from reaction_followup_state import persisted_state; "
            f"print(json.dumps(persisted_state(Path('/opt/data'), {room_id!r}, {thread_id!r})))"
        )
        result = gateway.container.get_wrapped_container().exec_run(
            ["/opt/hermes/.venv/bin/python", "-c", code]
        )
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, output
        return json.loads(output.splitlines()[-1])

    started = time.monotonic()
    try:
        delivered = exchange()
        root = delivered["root"]
        event_ids = delivered["event_ids"]
        assert len(set(event_ids)) == 3

        def final_persisted() -> bool:
            state = persisted_state(live_room.room_id, root)
            return {row[0] for row in state["watches"]} == set(event_ids) and any(
                row[1:] == ["assistant", final] for row in state["messages"]
            )

        _wait_for(
            final_persisted,
            "split reply watch and transcript",
            timeout=10,
            details=gateway_logs,
        )
        before = persisted_state(live_room.room_id, root)
        assert len(before["sessions"]) == 1
        session_id, session_key = before["sessions"][0]
        assert before["watches"] == [
            [event_id, session_key, session_id, final] for event_id in sorted(event_ids)
        ]
        assert len(gateway.model.main_requests()) == 4

        gateway.restart()
        restored = persisted_state(live_room.room_id, root)
        assert restored["watches"] == before["watches"]
        assert restored["sessions"] == before["sessions"]
        reacted = exchange(root=root, target=event_ids[-1])

        def followup_persisted() -> bool:
            state = persisted_state(live_room.room_id, root)
            return [session_id, "assistant", "Reaction follow-up"] in state["messages"]

        _wait_for(
            followup_persisted,
            "reaction follow-up transcript",
            timeout=10,
            details=gateway_logs,
        )
        after = persisted_state(live_room.room_id, root)
        assert after["sessions"] == before["sessions"]
        assert after["watches"] == []
        reaction_turns = [
            row
            for row in after["messages"]
            if row[1] == "user" and "Matrix reaction by " in row[2]
        ]
        assert len(reaction_turns) == 1
        assert reaction_turns[0][0] == session_id

        requests = gateway.model.main_requests()
        assert len(requests) == 5, [
            [message.get("content") for message in request["messages"] if message.get("role") == "user"]
            for request in requests
        ]
        messages = requests[-1]["messages"]
        assert [
            message["content"]
            for message in messages
            if message["role"] == "assistant" and message.get("content") == final
        ] == [final]
        latest_user = [
            message["content"] for message in messages if message["role"] == "user"
        ][-1]
        logical_prefix = " ".join(final[:500].split())
        assert f'[Replying to your previous message: "{logical_prefix}"]' in latest_user
        assert (
            f"Matrix reaction by {live_room.observer.user_id}: 👍 on reply {event_ids[-1]}"
            in latest_user
        )
        assert f"reaction event {reacted['reaction_id']}" in latest_user
    except (Exception, pytest.fail.Exception) as exc:
        details = failure_details()
        record_property("failure_diagnostics", details)
        message = str(exc)
        for secret in (live_room.bot.access_token, live_room.observer.access_token):
            message = message.replace(secret, "<redacted>")
        raise AssertionError(f"{message}\n{details}") from exc
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
