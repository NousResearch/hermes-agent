"""Independent matrix-nio exchanges for explicit thread creation."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from nio import Api, Event, RoomMessageText, RoomPutStateResponse, RoomSendResponse

from client import open_encrypted_client


class ThreadCreationProbe:
    def __init__(self, room_id: str, bot_id: str, encrypted: bool):
        self.room_id = room_id
        self.bot_id = bot_id
        self.encrypted = encrypted
        self.client = open_encrypted_client()
        self.events: asyncio.Queue[RoomMessageText] = asyncio.Queue()
        self.client.add_event_callback(self._on_message, RoomMessageText)
        self.sync_task: asyncio.Task | None = None

    async def _on_message(self, room: Any, event: Event) -> None:
        if (
            isinstance(event, RoomMessageText)
            and room.room_id == self.room_id
            and event.sender == self.bot_id
        ):
            self.events.put_nowait(event)

    async def start(self) -> None:
        await asyncio.wait_for(self.client.sync(timeout=0, full_state=True), timeout=15)
        if self.client.should_upload_keys:
            await self.client.keys_upload()
        self.sync_task = asyncio.create_task(self.client.sync_forever(timeout=250))

    async def close(self) -> None:
        if self.sync_task is not None:
            self.sync_task.cancel()
            await asyncio.gather(self.sync_task, return_exceptions=True)
        await self.client.close()

    async def send(
        self,
        body: str,
        *,
        root: str | None = None,
        notice: bool = False,
        mention: bool = True,
    ) -> str:
        content: dict[str, Any] = {
            "msgtype": "m.notice" if notice else "m.text",
            "body": body,
        }
        if mention:
            content["m.mentions"] = {"user_ids": [self.bot_id]}
        if root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": root},
            }
        result = await asyncio.wait_for(
            self.client.room_send(
                self.room_id,
                "m.room.message",
                content,
                ignore_unverified_devices=True,
            ),
            timeout=15,
        )
        assert isinstance(result, RoomSendResponse), result
        return result.event_id

    async def next_bot(self) -> RoomMessageText:
        return await asyncio.wait_for(self.events.get(), timeout=15)

    async def assert_wire_type(self, event_id: str) -> None:
        method, path = Api.room_get_event(
            self.client.access_token, self.room_id, event_id
        )
        response = await self.client.send(method, path, timeout=15)
        async with response:
            assert response.status == 200
            raw = await response.json()
        assert raw["type"] == (
            "m.room.encrypted" if self.encrypted else "m.room.message"
        )

    async def prepare(self) -> dict[str, str]:
        await self.start()
        try:
            if self.encrypted:
                response = await self.client.room_put_state(
                    self.room_id,
                    "m.room.encryption",
                    {"algorithm": "m.megolm.v1.aes-sha2"},
                )
                assert isinstance(response, RoomPutStateResponse), response
                await asyncio.wait_for(
                    self.client.sync(timeout=0, full_state=True), timeout=15
                )
            original = await self.send(
                "Original thread [old:sentinel]", notice=True, mention=False
            )
            await self.send("Prime original conversation [old:question]", root=original)
            first = await self.next_bot()
            assert first.body == "Matrix live reply"
            assert first.source["content"]["m.relates_to"]["event_id"] == original
            existing = await self.send(
                "Existing root [new:root]", notice=True, mention=False
            )
            return {"original": original, "existing": existing}
        finally:
            await self.close()

    async def create_and_follow(
        self, original: str, existing: str | None
    ) -> dict[str, Any]:
        await self.start()
        try:
            await self.send(
                "Create the explicit thread now [old:create]", root=original
            )
            root = existing
            if root is None:
                event = await self.next_bot()
                assert event.body == "New root [new:root]"
                assert "m.relates_to" not in event.source["content"]
                assert event.decrypted is self.encrypted
                root = event.event_id
                await self.assert_wire_type(root)
            first = await self.next_bot()
            assert first.body == "Initial thread message [new:initial]"
            assert first.decrypted is self.encrypted
            assert first.source["content"]["m.relates_to"] == {
                "rel_type": "m.thread",
                "event_id": root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": root},
            }
            await self.assert_wire_type(first.event_id)
            original_final = await self.next_bot()
            assert original_final.body == "Created in original"
            assert (
                original_final.source["content"]["m.relates_to"]["event_id"] == original
            )
            reply = await self.send(
                "Follow the new thread [new:question]", root=root, mention=False
            )
            final = await self.next_bot()
            assert final.body == "Thread final"
            assert final.decrypted is self.encrypted
            assert final.source["content"]["m.relates_to"] == {
                "rel_type": "m.thread",
                "event_id": root,
                "is_falling_back": False,
                "m.in_reply_to": {"event_id": reply},
            }
            await self.assert_wire_type(final.event_id)
            return {
                "success": True,
                "room_id": self.room_id,
                "root_event_id": root,
                "initial_reply_event_id": first.event_id,
            }
        finally:
            await self.close()

    async def deny_bot_messages(self) -> None:
        from nio import RoomGetStateEventResponse

        response = await self.client.room_get_state_event(
            self.room_id, "m.room.power_levels"
        )
        assert isinstance(response, RoomGetStateEventResponse), response
        content = response.content
        content["events_default"] = 50
        content["users"][self.bot_id] = 0
        changed = await self.client.room_put_state(
            self.room_id, "m.room.power_levels", content
        )
        assert isinstance(changed, RoomPutStateResponse), changed

    async def assert_no_thread_replies(self, root: str) -> None:
        path = f"/_matrix/client/v1/rooms/{quote(self.room_id, safe='')}/relations/{quote(root, safe='')}/m.thread"
        response = await self.client.send(
            "GET",
            path,
            headers={
                "Authorization": f"Bearer {self.client.access_token}",
            },
            timeout=15,
        )
        async with response:
            assert response.status == 200
            raw = await response.json()
        assert raw["chunk"] == []

    async def request_denied_creation(self, original: str) -> None:
        await self.start()
        try:
            await self.deny_bot_messages()
            await self.send("Attempt thread creation [old:denied]", root=original)
        finally:
            await self.close()

    async def assert_admission_encryption_transition_refused(
        self, bot_token: str
    ) -> str:
        from mautrix.client import Client
        from mautrix.client.state_store.memory import MemoryStateStore
        from mautrix.types import RoomID
        from nio import MessageDirection, RoomMessagesResponse
        from gateway.config import PlatformConfig
        from plugins.platforms.matrix.adapter import MatrixAdapter
        from plugins.platforms.matrix.read_context import (
            MatrixSessionAccess,
            MatrixSessionError,
        )

        await self.start()
        sdk = Client(
            mxid=self.bot_id,
            base_url=self.client.homeserver,
            token=bot_token,
            state_store=MemoryStateStore(),
        )
        sdk.api.default_retry_count = 0
        await sdk.join_room(RoomID(self.room_id))
        observer = self.client
        changed = False

        class AdmissionAdapter(MatrixAdapter):
            async def _is_dm_room(
                self, room_id: str, *, owner=None, require_classification: bool = False,
            ) -> bool:
                nonlocal changed
                if not changed:
                    response = await observer.room_put_state(
                        room_id,
                        "m.room.encryption",
                        {"algorithm": "m.megolm.v1.aes-sha2"},
                    )
                    assert isinstance(response, RoomPutStateResponse), response
                    changed = True
                    await sdk.get_state(RoomID(room_id))
                return await super()._is_dm_room(
                    room_id, owner=owner, require_classification=require_classification,
                )

        adapter = AdmissionAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "homeserver": self.client.homeserver,
                    "user_id": self.bot_id,
                    "allowed_users": self.client.user_id,
                    "e2ee_mode": "off",
                },
            )
        )
        adapter._client = sdk
        adapter._joined_rooms.add(self.room_id)
        adapter.set_authorization_check(
            lambda user, chat_type, chat_id: user == self.client.user_id
        )
        try:
            access = MatrixSessionAccess.capture(
                adapter, self.room_id, self.client.user_id
            )
            try:
                await access.send_message({
                    "msgtype": "m.text",
                    "body": "matrix-admission-secret",
                })
            except MatrixSessionError as exc:
                assert str(exc) == "missing encryption keys"
            else:
                raise AssertionError(
                    "Plaintext was accepted after encryption admission"
                )
            response = await self.client.room_messages(
                self.room_id,
                direction=MessageDirection.back,
                limit=100,
            )
            assert isinstance(response, RoomMessagesResponse), response
            assert not any(
                event.source.get("content", {}).get("body") == "matrix-admission-secret"
                for event in response.chunk
            )
            return "missing encryption keys"
        finally:
            await sdk.api.session.close()
            await self.close()
