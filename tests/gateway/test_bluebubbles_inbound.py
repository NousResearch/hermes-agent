"""BlueBubbles same-GUID ownership, late media, routing, and gateway admission."""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageType
from tests.gateway.bluebubbles_test_support import _make_adapter, _FakeBlueBubblesRequest


class TestBlueBubblesInboundRegression:
    @staticmethod
    def _payload(event_type="new-message", guid="msg-1", *, chat=None, **fields):
        data = {
            "guid": guid,
            "text": "hello",
            "handle": {"address": "+155****0100"},
            "isFromMe": False,
            **fields,
        }
        if chat is None:
            data["chatIdentifier"] = "+155****0100"
        else:
            data["chats"] = [chat]
        return {"type": event_type, "data": data}

    @staticmethod
    def _capture(monkeypatch, **extra):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False, require_mention=False, **extra)
        handled = []

        async def capture(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", capture)
        return adapter, handled

    @pytest.mark.asyncio
    async def test_inbound_lifecycle_routes_once_without_losing_rich_metadata(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            started.set()
            await release.wait()
            return "/tmp/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100", "style": 45}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("new-message", guid="enriched", chat=dm,
                          attachments=[{"guid": "att-1", "mimeType": ""}]))))
        await started.wait()
        second = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            "updated-message", guid="enriched", chat=dm,
            attachments=[{"guid": "att-1", "mimeType": "image/jpeg", "transferName": "photo.jpg"}]))))
        await asyncio.sleep(0)
        release.set()
        responses = await asyncio.gather(first, second)
        assert [response.status for response in responses] == [200, 200]
        assert calls == ["att-1"]
        assert [(event.source.chat_id, event.source.chat_type, event.message_type, event.media_urls)
                for event in handled] == [
            ("+155****0100", "dm", MessageType.PHOTO, ["/tmp/photo.jpg"])]

        # Receipt-only updates do not claim the GUID needed by a subsequent real event.
        receipt_adapter, receipt_handled = self._capture(monkeypatch)
        receipt = await receipt_adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("updated-message", guid="receipt", chat=dm, text="", dateRead=1789859535544)))
        message = await receipt_adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("new-message", guid="receipt", chat=dm)))
        await asyncio.sleep(0)
        assert receipt.status == message.status == 200
        assert [(event.source.chat_id, event.source.chat_type) for event in receipt_handled] == [
            ("+155****0100", "dm")]

        # Sparse private-API group records retry hydration inside this webhook and never become DMs.
        group_adapter, group_handled = self._capture(monkeypatch)
        group_adapter.client = AsyncMock()
        get = AsyncMock(side_effect=[
            httpx.ReadTimeout("temporary"),
            {"data": {"chats": []}},
            {"data": {"chats": [{
                "[auth-key]": "any;+;family-group",
                "style": 43,
                "chatIdentifier": "family-group",
            }]}},
        ])
        monkeypatch.setattr(group_adapter, "_api_get", get)
        response = await group_adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("updated-message", guid="group")))
        await asyncio.sleep(0)
        assert response.status == 200
        assert get.await_count == 3
        assert [(event.source.chat_id, event.source.chat_type) for event in group_handled] == [
            ("any;+;family-group", "group")]

        # A rich group event followed by a sparse echo remains one group dispatch.
        echo_adapter, echo_handled = self._capture(monkeypatch)
        echo_adapter.client = AsyncMock()
        monkeypatch.setattr(echo_adapter, "_api_get", AsyncMock(return_value={"data": {
            "chats": [{"guid": "any;+;same-group", "style": 43}],
        }}))
        await echo_adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            "new-message", guid="echo", chat={"guid": "any;+;same-group", "style": 43})))
        await echo_adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            "updated-message", guid="echo")))
        await asyncio.sleep(0)
        assert [(event.source.chat_id, event.source.chat_type) for event in echo_handled] == [
            ("any;+;same-group", "group")]

    @pytest.mark.asyncio
    async def test_webhook_reconciliation_preserves_delivery_registration(self, monkeypatch):
        # BlueBubbles POST is idempotent by URL, so migration must delete stale state before creation.
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        url = adapter._webhook_register_url
        state = [{"id": 1, "url": url, "events": ["new-message"]}]
        order = []

        async def find(_url):
            return list(state)

        async def post(_path, payload):
            order.append(("post", payload["events"]))
            if state:
                return {"status": 200, "data": state[0]}
            created = {"id": 2, "url": url, "events": payload["events"]}
            state.append(created)
            return {"status": 200, "data": created}

        async def delete(webhook_id):
            order.append(("delete", webhook_id))
            state[:] = [item for item in state if item["id"] != webhook_id]

        monkeypatch.setattr(adapter, "_find_registered_webhooks", find)
        monkeypatch.setattr(adapter, "_api_post", post)
        monkeypatch.setattr(adapter, "_delete_webhook_id", delete)
        assert await adapter._register_webhook() is True
        assert order == [("delete", 1), ("post", ["new-message", "updated-message"])]
        assert state == [{"id": 2, "url": url, "events": ["new-message", "updated-message"]}]

        # If replacement fails after deletion, restore the prior event set best-effort.
        rollback = _make_adapter(monkeypatch)
        rollback.client = AsyncMock()
        rollback_state = [{"id": 3, "url": url, "events": ["new-message"]}]
        posts = []

        async def rollback_delete(webhook_id):
            rollback_state[:] = [item for item in rollback_state if item["id"] != webhook_id]

        async def rollback_post(_path, payload):
            posts.append(payload["events"])
            if len(posts) == 1:
                raise httpx.ReadTimeout("replacement failed")
            restored = {"id": 4, "url": url, "events": payload["events"]}
            rollback_state.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(rollback, "_find_registered_webhooks", AsyncMock(return_value=list(rollback_state)))
        monkeypatch.setattr(rollback, "_api_post", rollback_post)
        monkeypatch.setattr(rollback, "_delete_webhook_id", rollback_delete)
        assert await rollback._register_webhook() is False
        assert posts == [["new-message", "updated-message"], ["new-message"]]
        assert rollback_state == [{"id": 4, "url": url, "events": ["new-message"]}]


class TestBlueBubblesAdmissionBoundaries:
    _payload = staticmethod(TestBlueBubblesInboundRegression._payload)
    _capture = staticmethod(TestBlueBubblesInboundRegression._capture)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["ordinary group message", ""])
    async def test_group_mentions_gate_known_text_before_media_hydration(self, monkeypatch, text):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False, require_mention=True)
        admitted, requests = [], []
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8"
            "AAwMCAO+jR7kAAAAASUVORK5CYII=")
        chat = {"guid": "any;+;group", "style": 43}
        attachment = {"guid": "photo", "mimeType": "image/png", "transferState": 5}

        async def accept(event):
            event._gateway_accepted = True
            admitted.append(event)

        def transport(request):
            requests.append(request)
            assert request.method == "GET"
            if request.url.path == "/api/v1/message/msg-1":
                assert request.url.params["with"] == "chats,attachments"
                return httpx.Response(200, json={"data": {
                    "text": "@hermes photo", "chats": [chat], "attachments": [attachment],
                }})
            assert request.url.path == "/api/v1/attachment/photo/download"
            return httpx.Response(200, content=png)

        adapter.handle_message = accept
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            response = await adapter._handle_webhook(_FakeBlueBubblesRequest(
                self._payload(chat=chat, text=text, attachments=[attachment])))
        assert response.status == 200
        if text:
            assert admitted == []
            assert requests == []
        else:
            assert len(admitted) == 1
            assert admitted[0].text == "photo"
            assert admitted[0].source.chat_id == chat["guid"]
            assert admitted[0].media_types == ["image/png"]
            assert Path(admitted[0].media_urls[0]).read_bytes() == png

    @pytest.mark.asyncio
    @pytest.mark.parametrize("caption", ["caption", ""])
    async def test_failed_download_can_complete_without_losing_caption(self, monkeypatch, caption):
        adapter, handled = self._capture(monkeypatch)
        ready = False
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            return "/tmp/photo.jpg" if ready else None

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100", "style": 45}
        data = self._payload(chat=dm, text=caption,
                             attachments=[{"guid": "photo", "mimeType": "image/jpeg"}])
        refused = await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        assert refused.status == 503
        assert handled == []
        ready = True
        data["type"] = "updated-message"
        data["data"]["isDelivered"] = True
        accepted = await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        assert accepted.status == 200
        assert [(e.text, e.media_urls) for e in handled] == [(caption or "(attachment)", ["/tmp/photo.jpg"])]

    @pytest.mark.asyncio
    async def test_partial_downloads_survive_refused_admission(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        ready = False
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            return "/tmp/" + guid + ".jpg" if guid == "first" or ready else None

        monkeypatch.setattr(adapter, "_download_attachment", download)
        data = self._payload(chat={"guid": "any;-;+155****0100"}, attachments=[
            {"guid": "first", "mimeType": "image/jpeg"}, {"guid": "second", "mimeType": "image/jpeg"}])
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
        ready = True
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert calls.count("first") == 1
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/first.jpg", "/tmp/second.jpg"]

    @pytest.mark.asyncio
    async def test_actual_gateway_refusal_does_not_consume_message(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        data = self._payload(chat={"guid": "any;-;+155****0100"})
        refused = await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        assert refused.status == 503  # Real BasePlatformAdapter: no gateway handler installed.
        accepted = []

        async def admit(event):
            event._gateway_accepted = True
            accepted.append(event)

        monkeypatch.setattr(adapter, "handle_message", admit)
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(accepted) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refresh", [{"transferState": 2}, {"mimeType": "image/png"}])
    async def test_stale_cached_attachment_is_not_claimed_without_delivery(self, monkeypatch, refresh):
        adapter, handled = self._capture(monkeypatch)
        ready = False

        async def download(guid, metadata):
            if ready or (guid == "first" and metadata.get("mimeType") == "image/jpeg"):
                return "/tmp/" + guid + ".jpg"
            return None

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100"}
        data = self._payload(chat=dm, attachments=[
            {"guid": "first", "mimeType": "image/jpeg"}, {"guid": "second", "mimeType": "image/jpeg"}])
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
        data["data"]["attachments"][0].update(refresh)
        ready = True
        if "mimeType" in refresh:
            ready = False
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
        assert handled == []
        ready = True
        data["data"]["attachments"][0]["transferState"] = 5
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/first.jpg", "/tmp/second.jpg"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("accepted_before_cancel", [False, True])
    async def test_cancellation_preserves_admission_boundary(self, monkeypatch, accepted_before_cancel):
        adapter, handled = self._capture(monkeypatch)
        data = self._payload(chat={"guid": "any;-;+155****0100"})
        original = adapter.handle_message

        async def cancel(event):
            event._gateway_accepted = accepted_before_cancel
            raise asyncio.CancelledError

        monkeypatch.setattr(adapter, "handle_message", cancel)
        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        monkeypatch.setattr(adapter, "handle_message", original)
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(handled) == (0 if accepted_before_cancel else 1)
        assert adapter._inbound_chat_tails == {}

    @pytest.mark.asyncio
    async def test_late_attachment_is_delivered_without_replaying_caption(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value="/tmp/photo.jpg"))
        dm = {"guid": "any;-;+155****0100"}
        initial = self._payload(chat=dm)
        late = self._payload("updated-message", chat=dm, isDelivered=True, attachments=[
            {"guid": "photo", "mimeType": "image/jpeg"}])
        for data in (initial, late, late):
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert [(e.text, e.media_urls) for e in handled] == [
            ("hello", []), ("(attachment)", ["/tmp/photo.jpg"])]
        assert adapter._download_attachment.await_count == 1

    @pytest.mark.asyncio
    async def test_same_guid_completion_across_download_has_one_owner(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            if guid == "first":
                started.set()
                await release.wait()
            return "/tmp/" + guid + ".jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100"}
        first_att = {"guid": "first", "mimeType": "image/jpeg"}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(chat=dm, attachments=[first_att]))))
        await started.wait()
        second = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("updated-message", chat=dm, isDelivered=True, attachments=[
                first_att, {"guid": "second", "mimeType": "image/jpeg"}]))))
        await asyncio.sleep(0)
        release.set()
        assert [r.status for r in await asyncio.gather(first, second)] == [200, 200]
        assert calls == ["first", "second"]
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/first.jpg", "/tmp/second.jpg"]

    @pytest.mark.asyncio
    async def test_distinct_messages_keep_order_without_blocking_other_chats(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()

        async def download(guid, metadata):
            started.set()
            await release.wait()
            return "/tmp/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        group = {"guid": "any;+;group", "style": 43}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="first", chat=group, attachments=[{"guid": "photo", "mimeType": "image/jpeg"}]))))
        await started.wait()
        second = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="second", chat=group))))
        await asyncio.sleep(0)
        unrelated = await adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            guid="unrelated", chat={"guid": "any;-;+155****0100"})))
        assert unrelated.status == 200
        assert [e.message_id for e in handled] == ["unrelated"]
        release.set()
        await asyncio.gather(first, second)
        assert [e.message_id for e in handled] == ["unrelated", "first", "second"]
        assert [e.source.chat_type for e in handled] == ["dm", "group", "group"]

    @pytest.mark.asyncio
    async def test_cancelled_fifo_waiter_cannot_release_its_successor_early(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()

        async def download(guid, metadata):
            started.set()
            await release.wait()
            return "/tmp/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100"}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            guid="first", chat=dm, attachments=[{"guid": "photo", "mimeType": "image/jpeg"}]))))
        await started.wait()
        cancelled = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="cancelled", chat=dm))))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        third = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="third", chat=dm))))
        await asyncio.sleep(0)
        assert handled == []
        release.set()
        await asyncio.gather(first, third)
        assert [e.message_id for e in handled] == ["first", "third"]
        assert adapter._inbound_chat_tails == {}

    @pytest.mark.asyncio
    async def test_known_replays_do_not_wait_or_download_again(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        download = AsyncMock(return_value="/tmp/photo.jpg")
        monkeypatch.setattr(adapter, "_download_attachment", download)
        sleep = AsyncMock(side_effect=AssertionError("unexpected fixed delay"))
        monkeypatch.setattr(asyncio, "sleep", sleep)
        data = self._payload(chat={"guid": "any;-;+155****0100"}, attachments=[
            {"guid": "photo", "mimeType": "image/jpeg"}])
        for _ in range(3):
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(handled) == 1
        assert download.await_count == 1
        assert sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_rest_hydration_requests_relationships_and_preserves_group_identity(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        requests = []

        def transport(request):
            requests.append(request)
            assert request.url.params['password'] == 'secret'
            with_fields = request.url.params.get('with', '').split(',')
            chats = [{'guid': 'any;+;group', 'style': 43, 'chatIdentifier': 'group'}] if 'chats' in with_fields else []
            return httpx.Response(200, json={'status': 200, 'data': {'chats': chats}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            data = self._payload('updated-message', isDelivered=True)
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert requests and all('chats' in r.url.params['with'].split(',') for r in requests)
        assert [(e.source.chat_id, e.source.chat_type, e.source.user_id) for e in handled] == [
            ('any;+;group', 'group', '+155****0100')]

    @pytest.mark.asyncio
    async def test_attachment_readiness_uses_rest_metadata_before_download(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        snapshots = iter([2, 2, 5])
        downloads = []

        def transport(request):
            if request.url.path.startswith('/api/v1/message/'):
                assert set(request.url.params['with'].split(',')) >= {'chats', 'attachments'}
                return httpx.Response(200, json={'status': 200, 'data': {
                    'chats': [{'guid': 'any;-;+155****0100', 'style': 45}],
                    'attachments': [{'guid': 'photo', 'mimeType': 'image/png', 'transferState': next(snapshots)}]}})
            assert request.url.path == '/api/v1/attachment/photo/download'
            downloads.append(request.url.path)
            return httpx.Response(200, content=b'probe-photo')

        from gateway.platforms import bluebubbles as module
        monkeypatch.setattr(module, 'cache_image_from_bytes_async', AsyncMock(return_value='/tmp/photo.png'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            response = await adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
                chat={'guid': 'any;-;+155****0100'}, attachments=[{'guid': 'photo', 'mimeType': ''}])))
        assert response.status == 200
        assert downloads == ['/api/v1/attachment/photo/download']
        assert [(e.message_type, e.media_types) for e in handled] == [(MessageType.PHOTO, ['image/png'])]
        module.cache_image_from_bytes_async.assert_awaited_once_with(b'probe-photo', '.png')

    @pytest.mark.asyncio
    async def test_dm_reply_retains_inbound_route_and_recovers_after_cache_eviction(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        address = '+15555550100'
        guid = 'any;-;' + address
        chats = [{'guid': 'any;-;+1555555' + str(i).zfill(4),
                  'chatIdentifier': '+1555555' + str(i).zfill(4)} for i in range(100)]
        chats.append({'guid': guid, 'chatIdentifier': address})
        query_offsets, sent = [], []

        def transport(request):
            body = json.loads(request.content)
            if request.url.path == '/api/v1/chat/query':
                query_offsets.append(body['offset'])
                return httpx.Response(200, json={'status': 200,
                    'data': chats[body['offset']:body['offset'] + body['limit']]})
            assert request.url.path == '/api/v1/message/text'
            sent.append(body['chatGuid'])
            return httpx.Response(200, json={'status': 200, 'data': {'guid': 'reply'}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            for message_id, chat in [('sparse', {'guid': guid, 'style': 45, 'displayName': 'Alice'}),
                                     ('rich', {'guid': guid, 'style': 45, 'chatIdentifier': address})]:
                await adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
                    guid=message_id, chat=chat, handle={'address': address})))
            assert [e.source.chat_id for e in handled] == [address, address]
            assert handled[0].source.chat_name == 'Alice'
            assert (await adapter.send(handled[0].source.chat_id, 'reply')).success
            assert query_offsets == []
            adapter._guid_cache.clear()
            assert (await adapter.send(handled[1].source.chat_id, 'reply')).success
        assert query_offsets == [0, 100]
        assert sent == [guid, guid]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('busy_mode', ['inline_command', 'fifo', 'debounce'])
    async def test_actual_base_busy_admission_prevents_replay(self, monkeypatch, busy_mode):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False, require_mention=False)
        data = self._payload(chat={'guid': 'any;-;+155****0100'},
                             text='/status' if busy_mode == 'inline_command' else 'hello')
        seen = []

        async def handler(event):
            seen.append(event.message_id)
            return None

        adapter.set_message_handler(handler)
        source = adapter.build_source(chat_id='+155****0100', chat_type='dm', user_id='+155****0100')
        session_key = adapter._source_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()
        if busy_mode == 'fifo':
            async def busy(event, key):
                event._gateway_accepted = True
                seen.append(event.message_id)
                return True
            adapter._busy_session_handler = busy
        elif busy_mode == 'debounce':
            adapter._busy_text_mode = 'queue'
            adapter._busy_text_debounce_seconds = 60
        try:
            for _ in range(2):
                assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
            if busy_mode == 'debounce':
                assert adapter._text_debounce_store()[session_key].event.text == 'hello'
            else:
                assert seen == ['msg-1']
        finally:
            adapter._discard_text_debounce(session_key)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', ['queue', 'steer', 'interrupt'])
    async def test_real_busy_runner_claims_only_accepted_input(self, monkeypatch, mode):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setenv('GATEWAY_ALLOW_ALL_USERS', 'true')
        monkeypatch.setenv('HERMES_GATEWAY_BUSY_ACK_ENABLED', 'false')
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.set_message_handler(AsyncMock(return_value=None))
        runner = GatewayRunner(GatewayConfig())
        runner.adapters = {Platform.BLUEBUBBLES: adapter}
        runner._busy_input_mode = mode
        runner._busy_text_mode = 'interrupt'
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        data = self._payload(chat={'guid': 'any;-;+155****0100'})
        source = adapter.build_source(chat_id='+155****0100', chat_type='dm', user_id='+155****0100')
        key = adapter._source_session_key(source)
        adapter._active_sessions[key] = asyncio.Event()
        calls = []

        class Receiver:
            _supports_active_turn_redirect = True

            def steer(self, text):
                calls.append(text)
                return True

            redirect = steer

        runner._session_state(key).turn.agent = Receiver()
        if mode == 'queue':
            runner._BUSY_QUEUE_MAX_PENDING = 0
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
            assert key not in adapter._pending_messages
            runner._BUSY_QUEUE_MAX_PENDING = 1
        for _ in range(2):
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        if mode == 'queue':
            assert adapter._pending_messages[key].text == 'hello'
            assert runner._queue_depth(key, adapter=adapter) == 1
        else:
            assert len(calls) == 1 and calls[0].endswith('\n\nhello')
