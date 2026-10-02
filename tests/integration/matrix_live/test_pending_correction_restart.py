"""Native correction originals resume only after current homeserver validation."""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from pathlib import Path
from textwrap import dedent
from typing import Callable, Any

import pytest

from tests.fakes.fake_llm_provider import Text
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
    MatrixAccount,
    _wait_for,
)
from tests.integration.matrix_live.edit_client import begin, edits, final, send
from tests.integration.matrix_live.test_edit_followups import marker

pytest_plugins = ["tests.integration.matrix_live.context_client"]


@pytest.fixture
def model_responder() -> Callable[[dict], Text]:
    def respond(body: dict) -> Text:
        content = json.dumps(body["body"]["messages"][-1]["content"])
        if "[in:latest]" in content:
            return Text("Restored correction response")
        if "[in:original]" in content:
            return Text("Matrix live reply")
        return Text("Restored earlier response")

    return respond


@pytest.mark.parametrize("gateway", ["pause-edit-followups"], indirect=True)
@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_pending_correction_keeps_typed_identity_authority_and_user_receipt_across_restart(
    tmp_path: Path,
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    group_member: MatrixAccount,
    encrypted: bool,
) -> None:
    async def foreign_message() -> str:
        client = group_member.client(live_room.homeserver)
        try:
            return await send(
                client,
                live_room.room_id,
                {"msgtype": "m.text", "body": "Foreign original"},
            )
        finally:
            await client.close()

    foreign = asyncio.run(asyncio.wait_for(foreign_message(), 15))
    prefix = (
        f"ROOM={live_room.room_id!r}\nBOT={live_room.bot.user_id!r}\nDEVICE={live_room.bot.device_id!r}\n"
        + dedent("""
            import asyncio, json
            from client import open_encrypted_client
            from edit_client import begin, edits, final
            async def run(operation):
                client = open_encrypted_client()
                try:
                    return await operation(client)
                finally:
                    await client.close()
        """)
    )

    def encrypted_call(expression: str) -> Any:
        output = linux_nio_observer.run_python(
            prefix
            + f"print(json.dumps(asyncio.run(asyncio.wait_for(run(lambda client: {expression}), 30))))\n"
        )
        return json.loads(output.strip().splitlines()[-1])

    async def plain_call(operation):
        client = live_room.observer.client(live_room.homeserver)
        try:
            return await operation(client)
        finally:
            await client.close()

    ids = (
        encrypted_call("begin(client, ROOM, BOT, DEVICE, encrypted=True)")
        if encrypted
        else asyncio.run(
            asyncio.wait_for(
                plain_call(
                    lambda client: begin(
                        client,
                        live_room.room_id,
                        live_room.bot.user_id,
                        live_room.bot.device_id,
                        encrypted=False,
                    )
                ),
                30,
            )
        )
    )
    _wait_for(
        lambda: marker(gateway, "context-started"),
        "real second-turn preparation barrier",
    )
    before = copy.deepcopy(gateway.model.main_requests())
    assert len(before) == 1
    edit_ids = (
        encrypted_call(f"edits(client, ROOM, BOT, {ids!r}, {foreign!r})")
        if encrypted
        else asyncio.run(
            asyncio.wait_for(
                plain_call(
                    lambda client: edits(
                        client, live_room.room_id, live_room.bot.user_id, ids, foreign
                    )
                ),
                15,
            )
        )
    )
    _wait_for(
        lambda: marker(gateway, "edits-observed", "\n".join(edit_ids)),
        "real corrections admitted or refused",
    )
    assert gateway.model.main_requests() == before
    home = gateway.home
    gateway.container.get_wrapped_container().stop(timeout=20)
    paths = list((home / "pending_messages").glob("*.json"))
    saved = [json.loads(path.read_text()) for path in paths]
    records = [record for payload in saved for record in payload.get("events", [])]
    corrections = [
        record
        for record in records
        if record["event"]["metadata"].get("edited_message")
    ]
    observed = [
        (
            record["event"]["message_id"],
            record["event"]["text"],
            record["native"].get("correction"),
            record["event"]["source"]["thread_id"],
            record["event"]["source"]["user_id"],
        )
        for record in corrections
    ]
    assert observed == [
        (
            edit_ids[-1],
            "Latest correction [in:latest]",
            {"original_event_id": ids["original"]},
            ids["root"],
            live_room.observer.user_id,
        )
    ]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    record = corrections[0]
    owner, uid = record["input_owner"]["owner"], record["uid"]
    artifact = tmp_path / "pending-correction-restart.json"
    artifact.write_text(
        json.dumps(
            {"shutdown": saved, "correction_uid": uid, "input_owner": owner}, indent=2
        )
        + "\n"
    )

    def receipts() -> list[tuple[str, str]]:
        with sqlite3.connect(home / "state.db") as db:
            rows = db.execute(
                "SELECT content, display_metadata FROM messages WHERE role='user'"
            ).fetchall()
        return [
            (content, json.loads(metadata)["gateway_input_owner"])
            for content, metadata in rows
            if metadata and json.loads(metadata).get("gateway_input_owner") == owner
        ]

    assert receipts() == []
    (home / "context-release").write_text("release", encoding="utf-8")
    gateway.restart()
    _wait_for(
        lambda: len(receipts()) == 1,
        "canonical persisted user-input ownership receipt",
        timeout=30,
    )
    _wait_for(
        lambda: any(
            "[in:latest]" in json.dumps(request["messages"][-1]["content"])
            for request in gateway.model.main_requests()[len(before) :]
        ),
        "actual restored correction model turn",
        timeout=30,
    )
    relation = (
        encrypted_call(
            "final(client, ROOM, BOT, 'Restored correction response', encrypted=True)"
        )
        if encrypted
        else asyncio.run(
            asyncio.wait_for(
                plain_call(
                    lambda client: final(
                        client,
                        live_room.room_id,
                        live_room.bot.user_id,
                        "Restored correction response",
                        encrypted=False,
                    )
                ),
                30,
            )
        )
    )
    assert relation == {
        "rel_type": "m.thread",
        "event_id": ids["root"],
        "is_falling_back": False,
        "m.in_reply_to": {"event_id": ids["original"]},
    }

    def retained() -> list[dict]:
        return [
            row
            for path in (home / "pending_messages").glob("*.json")
            for row in json.loads(path.read_text()).get("events", [])
            if row["uid"] == uid
        ]

    _wait_for(
        lambda: retained() == [],
        "forensic body cleanup after canonical receipt",
        timeout=30,
    )
    requests = gateway.model.main_requests()[len(before) :]
    corrections_sent = [
        request
        for request in requests
        if "[in:latest]" in json.dumps(request["messages"][-1]["content"])
    ]
    assert len(corrections_sent) == 1
    content = json.dumps(corrections_sent[0]["messages"][-1]["content"])
    assert "[in:obsolete]" not in content and "[in:forgery]" not in content
    assert f"[Correction to earlier message {ids['original']}]" in content
    final_receipts = receipts()
    normalized_receipts = [
        {
            "owner": recorded_owner,
            "latest": "[in:latest]" in recorded_content,
            "original_target": f"[Correction to earlier message {ids['original']}]"
            in recorded_content,
        }
        for recorded_content, recorded_owner in final_receipts
    ]
    assert normalized_receipts == [
        {"owner": owner, "latest": True, "original_target": True}
    ]
    artifact.write_text(
        json.dumps(
            {
                "shutdown": saved,
                "correction_uid": uid,
                "input_owner": owner,
                "receipts": final_receipts,
                "restored_requests": requests,
                "native_reply_relation": relation,
                "retained": retained(),
            },
            indent=2,
        )
        + "\n"
    )
