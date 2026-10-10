"""Native pending originals resume only after current homeserver validation."""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import pytest
from tests.integration.matrix_live.conftest import _wait_for
from tests.integration.matrix_live.context_client import hand_off


@pytest.fixture
def gateway_home_setup():
    def setup(home):
        with (home / "plugins/matrix-live-context/__init__.py").open("a") as stream:
            stream.write(OBSERVE)

    return setup


@pytest.mark.parametrize("gateway", ["pause-image-context"], indirect=True)
@pytest.mark.parametrize("encrypted", [False, True])
def test_native_buffered_emote_and_stickers_resume_with_durable_user_receipts(
    tmp_path, gateway, live_room, linux_nio_observer, encrypted
):
    home = gateway.home
    prefix = (
        "import asyncio, json\nfrom rich_content_client import accepted_exchange, paused_emote, queued_stickers, open_encrypted_client, relation, send, assert_native\n"
        f"ROOM={live_room.room_id!r}\nBOT={live_room.bot.user_id!r}\nDEVICE={live_room.bot.device_id!r}\nENCRYPTED={encrypted!r}\n"
    )
    received = json.loads(
        linux_nio_observer
        .run_python(
            prefix
            + "print(json.dumps(asyncio.run(accepted_exchange(ROOM,BOT,DEVICE,encrypted=ENCRYPTED))))\n"
        )
        .strip()
        .splitlines()[-1]
    )
    prefix += f"ROOT={received['root']!r}\nURL={received['url']!r}\n"
    linux_nio_observer.run_python(
        prefix + "print(asyncio.run(paused_emote(ROOM,ROOT,encrypted=ENCRYPTED)))\n"
    )
    _wait_for(
        lambda: (home / "context-started").exists(), "active emote barrier", timeout=10
    )
    body = f"{live_room.bot.user_id} Checkpoint buffered emote"
    code = (
        "async def emit():\n    client=open_encrypted_client()\n    try:\n        await client.sync(timeout=0,full_state=True)\n"
        f"        content={{'msgtype':'m.emote','body':{body!r},'m.relates_to':relation(ROOT,ROOT)}}\n"
        "        target=await send(client,ROOM,'m.room.message',content)\n        await assert_native(client,ROOM,target,'m.room.message',content,encrypted=ENCRYPTED)\n        return target\n    finally:\n        await client.close()\nprint(json.dumps(asyncio.run(emit())))\n"
    )
    target = json.loads(
        linux_nio_observer.run_python(prefix + code).strip().splitlines()[-1]
    )
    _wait_for(
        lambda: (home / "checkpoint-blocked").exists(),
        "real batch receipt barrier",
        timeout=10,
    )
    blocked = json.loads((home / "checkpoint-blocked").read_text())
    cursor_path = home / Path(blocked["store_path"]).relative_to("/opt/data")
    assert json.loads(cursor_path.read_text()) == blocked["cursor"]
    assert target not in blocked["cursor"].get("accepted_events", [])
    assert not blocked["receipt_done"]
    queued = json.loads(
        linux_nio_observer
        .run_python(
            prefix
            + "print(json.dumps(asyncio.run(queued_stickers(ROOM,ROOT,URL,BOT,encrypted=ENCRYPTED))))\n"
        )
        .strip()
        .splitlines()[-1]
    )
    _wait_for(
        lambda: (
            (home / "rich-events-queued").exists()
            and sorted((home / "rich-events-queued").read_text().splitlines())
            == sorted(queued)
        ),
        "native stickers admitted",
        timeout=10,
    )
    hand_off(home / "checkpoint-release", "release")
    _wait_for(
        lambda: (home / "checkpoint-admitted").exists(),
        "batch admission complete",
        timeout=10,
    )
    _wait_for(
        lambda: (
            json.loads(cursor_path.read_text())["next_batch"]
            != blocked["cursor"]["next_batch"]
        ),
        "durable cursor after receipt",
        timeout=10,
    )
    assert len(gateway.model.main_requests()) == 2
    gateway.container.get_wrapped_container().stop(timeout=20)
    payloads = [
        json.loads(p.read_text()) for p in (home / "pending_messages").glob("*.json")
    ]
    structured = [p for p in payloads if p.get("events")]
    records = [r for p in structured for r in p["events"]]
    identities = {
        value
        for r in records
        for value in [
            r["event"]["message_id"],
            *r["event"].get("merged_message_ids", []),
        ]
    }
    rich = [
        c
        for r in records
        for s in r.get("context", {}).get("snapshots", [])
        for c in s.get("contributions", [])
    ]
    artifact = (
        Path(__file__).resolve().parents[3].parent
        / "queue-pending-restoration-native.json"
    )
    artifact.write_text(
        json.dumps({"blocked": blocked, "shutdown": structured}, indent=2) + "\n"
    )
    assert target in identities or target in {c["event_id"] for c in rich}
    assert set(queued) <= {c["event_id"] for c in rich}
    emote = next(c for c in rich if c["event_id"] == target)
    assert body.replace(live_room.bot.user_id, "").strip() in emote["original_text"]
    assert all(
        c["sender"] == live_room.observer.user_id and c["media_paths"]
        for c in rich
        if c["event_id"] in queued
    )
    assert all(
        p.stat().st_mode & 0o777 == 0o600
        for p in (home / "pending_messages").glob("*.json")
    )
    before = len(gateway.model.main_requests())
    owners = {record["input_owner"]["owner"] for record in records}
    hand_off(home / "context-release", "release")
    gateway.restart()

    def receipts():
        with sqlite3.connect(home / "state.db") as db:
            rows = db.execute(
                "SELECT role, content, display_metadata FROM messages WHERE role = 'user'"
            ).fetchall()
        return [
            (content, json.loads(metadata).get("gateway_input_owner"))
            for role, content, metadata in rows
            if metadata and json.loads(metadata).get("gateway_input_owner") in owners
        ]

    _wait_for(
        lambda: len(receipts()) == len(owners),
        "canonical user receipts for restored originals",
        timeout=30,
    )
    _wait_for(
        lambda: len(gateway.model.main_requests()) >= before + len(owners),
        "actual restored model turns",
        timeout=30,
    )
    _wait_for(
        lambda: not list((home / "pending_messages").glob("*.json")),
        "body cleanup after user receipts",
        timeout=30,
    )
    after = [
        json.loads(p.read_text()) for p in (home / "pending_messages").glob("*.json")
    ]
    assert sorted(owner for content, owner in receipts()) == sorted(owners)
    assert len(gateway.model.main_requests()) == before + len(owners)
    restored = gateway.model.main_requests()[before:]
    rendered = [json.dumps(request["messages"][-1]["content"]) for request in restored]
    assert all(
        any(caption in text for text in rendered)
        for caption in (
            "Checkpoint buffered emote",
            "Retained queued sticker",
            "Withdrawn queued sticker",
        )
    )
    artifact = (
        Path(__file__).resolve().parents[3].parent
        / "queue-pending-restoration-native.json"
    )
    artifact.write_text(
        json.dumps(
            {
                "blocked": blocked,
                "shutdown": structured,
                "restarted": after,
                "model_requests": len(gateway.model.main_requests()),
                "receipts": receipts(),
            },
            indent=2,
        )
        + "\n"
    )


OBSERVE = r"""
import json
from plugins.platforms.matrix.intake_mixin import MatrixIntakeMixin
original_dispatch_batch = MatrixIntakeMixin._dispatch_text_batch
async def observed_dispatch_batch(self,event):
    observed = 'Checkpoint buffered emote' in event.text
    if observed:
        store=self._client.sync_store
        receipts=self._text_batch_intakes.get(id(event),[])
        signal('checkpoint-blocked',json.dumps({'store_path':str(store.path),'cursor':json.loads(store.path.read_text()),'receipt_done':any(receipt.done() for _,receipt in receipts)}))
        while not (get_hermes_home()/'checkpoint-release').exists():
            await asyncio.sleep(0.01)
    result=await original_dispatch_batch(self,event)
    if observed:
        signal('checkpoint-admitted','admitted')
    return result
MatrixIntakeMixin._dispatch_text_batch=observed_dispatch_batch
"""
