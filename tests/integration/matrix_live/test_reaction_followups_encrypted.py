"""A native reaction resumes an encrypted split reply after gateway restart."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from plugins.platforms.matrix.followup_context import REPLY_EXCERPT_CHARS
from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveRoom,
    _wait_for,
)
from tests.integration.matrix_live.live_gateway import LiveGateway


@pytest.fixture
def gateway_config() -> str:
    # The client's reply deadline spans a restart. A delivered restart notice is one more
    # encrypted send inside that window, and this case does not test notices.
    return (
        "platforms:\n  matrix:\n    enabled: true\n    max_message_length: 500\n"
        "    gateway_restart_notification: false\n"
        "streaming:\n  enabled: false\nupdates:\n  check: false\n"
        "plugins:\n  enabled: [matrix-restart-barriers]\n"
    )


@pytest.fixture
def gateway_home_setup() -> Callable[[Path], None]:
    def setup(home: Path) -> None:
        plugin = home / "plugins" / "matrix-restart-barriers"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            "name: matrix-restart-barriers\nversion: 0.1.0\n", encoding="utf-8"
        )
        (plugin / "__init__.py").write_text(
            Path(__file__).with_name("restart_barriers.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    return setup


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
        barriers = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(gateway.home.glob("*-diagnostics.json"))
        }
        checkpoints = {
            str(path.relative_to(gateway.home)): json.loads(
                path.read_text(encoding="utf-8")
            )
            for path in sorted(gateway.home.rglob("sync-*.json"))
        }
        details = (
            f"Model requests ({len(gateway.model.main_requests())}):\n"
            f"{json.dumps(requests)}\nPersisted state (exit {result.exit_code}):\n"
            f"{result.output.decode(errors='replace')[-65536:]}\n"
            f"Native barriers:\n{json.dumps(barriers)}\n"
            f"Sync checkpoints:\n{json.dumps(checkpoints)}\n"
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
        result = gateway.container.get_wrapped_container().exec_run([
            "/opt/hermes/.venv/bin/python",
            "-c",
            code,
        ])
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, output
        return json.loads(output.splitlines()[-1])

    started = time.monotonic()
    try:

        def key_import_blocked() -> bool:
            shared = gateway.home / "key-shared"
            imported = gateway.home / "key-import-blocked"
            if not shared.exists() or not imported.exists():
                return False
            session_id = shared.read_text(encoding="utf-8")
            return bool(session_id) and session_id == imported.read_text(
                encoding="utf-8"
            )

        with ThreadPoolExecutor(max_workers=1) as clients:
            delivery = clients.submit(exchange, barrier_home="/gateway-barriers")
            _wait_for(
                key_import_blocked,
                "native key import before restart",
                timeout=10,
                details=failure_details,
            )
            shared_session = (gateway.home / "key-shared").read_text(encoding="utf-8")
            assert (gateway.home / "key-import-blocked").read_text(
                encoding="utf-8"
            ) == shared_session
            assert not (gateway.home / "room-input-published").exists()
            assert gateway.model.main_requests() == []
            gateway.restart()
            assert (gateway.home / "key-import-blocked-cancelled").read_text(
                encoding="utf-8"
            ) == shared_session
            assert (gateway.home / "key-import-restored").read_text(
                encoding="utf-8"
            ) == shared_session
            assert not (gateway.home / "room-input-published").exists()
            assert gateway.model.main_requests() == []
            (gateway.home / "room-input-release").touch()
            _wait_for(
                lambda: (gateway.home / "text-buffered").exists(),
                "default text buffer before admission",
                timeout=10,
                details=failure_details,
            )
            buffered_id = (gateway.home / "text-buffered").read_text(encoding="utf-8")
            buffered_checkpoints = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in gateway.home.rglob("sync-*.json")
            ]
            assert len(buffered_checkpoints) == 1
            assert buffered_id not in buffered_checkpoints[0].get("accepted_events", [])
            assert gateway.model.main_requests() == []
            gateway.restart()
            assert (gateway.home / "text-buffered-cancelled").read_text(
                encoding="utf-8"
            ) == buffered_id
            delivered = delivery.result()
        root = delivered["root"]
        assert (gateway.home / "room-input-published").read_text(
            encoding="utf-8"
        ) == root
        intake_id = delivered["intake_id"]
        assert delivered["prime_id"] == buffered_id
        assert intake_id != buffered_id
        replayed = json.loads(
            (gateway.home / "startup-replayed-diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
        assert (replayed["event_id"], replayed["adapter_running"]) == (
            buffered_id,
            True,
        )
        event_ids = delivered["event_ids"]
        assert len(set(event_ids)) == 3
        assert (gateway.home / "replacement-delivery").read_text(
            encoding="utf-8"
        ).splitlines() == event_ids

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
            [event_id, session_key, session_id, final[:REPLY_EXCERPT_CHARS]]
            for event_id in sorted(event_ids)
        ]
        assert len(gateway.model.main_requests()) == 4

        _wait_for(
            lambda: (gateway.home / "intake-blocked").exists(),
            "durable intake with incomplete native sibling",
            timeout=10,
            details=failure_details,
        )
        assert (gateway.home / "intake-blocked").read_text(
            encoding="utf-8"
        ) == intake_id
        blocked = json.loads(
            (gateway.home / "intake-blocked-diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
        assert (blocked["event_id"], blocked["adapter_running"]) == (intake_id, True)
        assert intake_id in blocked["accepted_events"]
        record_property(
            "native_barriers",
            json.dumps({
                "buffered_event_id": buffered_id,
                "startup_replay": replayed,
                "unfinished_native_sibling": blocked,
            }),
        )
        checkpoints = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in gateway.home.rglob("sync-*.json")
        ]
        assert len(checkpoints) == 1
        assert intake_id in checkpoints[0]["accepted_events"]
        gateway.restart()
        assert (gateway.home / "intake-blocked-cancelled").exists()
        assert all(
            "accepted_events" not in json.loads(path.read_text(encoding="utf-8"))
            for path in gateway.home.rglob("sync-*.json")
        )
        restored = persisted_state(live_room.room_id, root)
        assert restored == before
        assert len(gateway.model.main_requests()) == 4
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
            [
                message.get("content")
                for message in request["messages"]
                if message.get("role") == "user"
            ]
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
        excerpt = final[:REPLY_EXCERPT_CHARS]
        assert f'[Replying to your previous message: "{excerpt}"]' in latest_user
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
