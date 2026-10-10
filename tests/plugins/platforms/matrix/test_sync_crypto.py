"""SDK decryption failures and failed key imports have different sync outcomes."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from plugins.platforms.matrix.sync_transport import DurableSyncStore, create_sync_client


@pytest_asyncio.fixture
async def sdk_client(tmp_path):
    pytest.importorskip("mautrix.crypto")
    from mautrix.crypto.store.asyncpg import PgCryptoStore
    from mautrix.util.async_db import Database
    from mautrix.client.state_store import MemoryStateStore
    from mautrix.util.logging import TraceLogger
    from plugins.platforms.matrix.sync_transport import create_sync_olm_machine

    store = DurableSyncStore(
        tmp_path, "https://matrix.test", "@bot:matrix.test", "BOT", "secret"
    )
    client = create_sync_client(
        mxid="@bot:matrix.test",
        device_id="BOT",
        api=SimpleNamespace(log=TraceLogger("crypto-test")),
        state_store=MemoryStateStore(),
        sync_store=store,
    )
    db = Database.create(
        f"sqlite:///{tmp_path / 'crypto.db'}", upgrade_table=PgCryptoStore.upgrade_table
    )
    await db.start()
    crypto_store = PgCryptoStore("test", "test-pickle", db)
    machine = create_sync_olm_machine(client, crypto_store, SimpleNamespace())
    await machine.load()
    machine.share_keys = AsyncMock()
    client.crypto = machine
    try:
        yield client, machine, crypto_store
    finally:
        await db.stop()


def sync_response(token, *to_device):
    return {
        "next_batch": token,
        "to_device": {"events": list(to_device)},
        "rooms": {
            "join": {
                "!room:matrix.test": {
                    "timeline": {
                        "events": [
                            {
                                "type": "m.room.message",
                                "event_id": "$valid",
                                "sender": "@alice:matrix.test",
                                "origin_server_ts": 1700000000000,
                                "content": {
                                    "msgtype": "m.text",
                                    "body": "valid room intake",
                                },
                            }
                        ]
                    }
                }
            }
        },
    }


def ciphertext(machine, *, own=True):
    return {
        "type": "m.room.encrypted",
        "sender": "@alice:matrix.test",
        "content": {
            "algorithm": "m.olm.v1.curve25519-aes-sha2",
            "sender_key": "sender-key",
            "ciphertext": {
                machine.account.identity_key: {"type": 0, "body": "ciphertext"}
            }
            if own
            else {},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("unreadable", ["other-device", "malformed-payload"])
async def test_unreadable_olm_does_not_poison_room_intake(sdk_client, unreadable):
    from mautrix.types import EventType

    client, machine, _ = sdk_client
    intake = AsyncMock()
    client.add_event_handler(EventType.ROOM_MESSAGE, intake)
    if unreadable == "malformed-payload":
        machine._decrypt_olm_ciphertext = AsyncMock(return_value="{broken")
    await client.sync_store.put_next_batch("before")
    response = sync_response(
        "after", ciphertext(machine, own=unreadable != "other-device")
    )
    await client.hermes_sync.dispatch_sync(response)
    await client.sync_store.put_next_batch(response["next_batch"])
    assert (intake.await_count, await client.sync_store.get_next_batch()) == (
        1,
        "after",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["failure", "cancel"])
async def test_partial_real_key_import_keeps_cursor_and_retries_before_room_intake(
    sdk_client,
    monkeypatch,
    interruption,
):
    from mautrix.crypto import OlmAccount, OutboundGroupSession
    from mautrix.errors import DecryptionError
    from mautrix.types import EventType

    client, machine, crypto_store = sdk_client
    session = OutboundGroupSession("!room:matrix.test")
    payload = {
        "type": "m.room_key",
        "sender": "@alice:matrix.test",
        "sender_device": "ALICE",
        "recipient": str(client.mxid),
        "recipient_keys": {"ed25519": machine.account.signing_key},
        "keys": {"ed25519": "alice-signing-key"},
        "content": {
            "algorithm": "m.megolm.v1.aes-sha2",
            "room_id": "!room:matrix.test",
            "session_id": session.id,
            "session_key": session.session_key,
            "com.beeper.max_age_ms": 60000,
            "com.beeper.max_messages": 100,
        },
    }
    sender = OlmAccount()
    machine.account.generate_one_time_keys(1)
    await crypto_store.put_account(machine.account)
    one_time_key = next(iter(machine.account.one_time_keys["curve25519"].values()))
    sender_session = sender.new_outbound_session(
        machine.account.identity_key, one_time_key
    )
    encrypted = sender_session.encrypt(json.dumps(payload))
    event = ciphertext(machine)
    event["content"]["sender_key"] = sender.identity_key
    event["content"]["ciphertext"][machine.account.identity_key] = encrypted.serialize()
    intake = AsyncMock()
    client.add_event_handler(EventType.ROOM_MESSAGE, intake)
    original = crypto_store.put_group_session
    imported, release = asyncio.Event(), asyncio.Event()
    attempts = []

    async def partial_import(*args):
        await original(*args)
        attempts.append("import")
        imported.set()
        if len(attempts) == 1:
            await release.wait()
            raise DecryptionError("application key import failed")

    monkeypatch.setattr(crypto_store, "put_group_session", partial_import)
    await client.sync_store.put_next_batch("before")
    response = sync_response("after", event)
    dispatch = asyncio.create_task(client.hermes_sync.dispatch_sync(response))
    await asyncio.wait_for(imported.wait(), timeout=2)
    if interruption == "cancel":
        dispatch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch
    else:
        release.set()
        with pytest.raises(DecryptionError, match="application key import failed"):
            await dispatch
    assert (intake.await_count, await client.sync_store.get_next_batch()) == (
        0,
        "before",
    )
    assert await crypto_store.get_group_session("!room:matrix.test", session.id) is None
    await client.hermes_sync.dispatch_sync(response)
    await client.sync_store.put_next_batch(response["next_batch"])
    assert (attempts, intake.await_count, await client.sync_store.get_next_batch()) == (
        ["import", "import"],
        1,
        "after",
    )
