"""BlueBubbles registration behavior."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.gateway.bluebubbles_test_support import _make_adapter

pytestmark = pytest.mark.usefixtures("_isolate_bluebubbles_environment")


class TestBlueBubblesWebhookRegistration:
    """Tests for _register_webhook, _unregister_webhook, _find_registered_webhooks."""

    @staticmethod
    def _mock_client(get_response=None, post_response=None, delete_ok=True):
        """Build a tiny mock httpx.AsyncClient."""

        async def mock_get(*args, **kwargs):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return get_response or {"status": 200, "data": []}

            return R()

        async def mock_post(*args, **kwargs):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return post_response or {"status": 200, "data": {}}

            return R()

        async def mock_delete(*args, **kwargs):
            class R:
                status_code = 200 if delete_ok else 500

                def raise_for_status(self_inner):
                    if not delete_ok:
                        raise Exception("delete failed")

            return R()

        return type(
            "MockClient",
            (),
            {"get": mock_get, "post": mock_post, "delete": mock_delete},
        )()

    # -- _find_registered_webhooks --

    def test_find_registered_webhooks_returns_matches(self, monkeypatch):
        import asyncio

        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_url
        adapter.client = self._mock_client(
            get_response={
                "status": 200,
                "data": [
                    {"id": 1, "url": url, "events": ["new-message"]},
                    {"id": 2, "url": "http://other:9999/hook", "events": ["message"]},
                ],
            }
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(url)
        )
        assert len(result) == 1
        assert result[0]["id"] == 1

    # -- _register_webhook --

    def test_register_fresh(self, monkeypatch):
        """No existing webhook → POST creates one."""
        import asyncio

        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 200, "data": {"id": 42}},
        )
        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())
        assert ok is True

    @pytest.mark.asyncio
    async def test_register_fails_closed_when_lookup_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(adapter, "client", self._mock_client())
        monkeypatch.setattr(
            adapter, "_api_get", AsyncMock(side_effect=RuntimeError("lookup failed"))
        )
        post = AsyncMock(return_value={"status": 200})
        monkeypatch.setattr(adapter, "_api_post", post)

        ok = await adapter._register_webhook()

        assert ok is False
        post.assert_not_awaited()

    def test_register_reuses_existing(self, monkeypatch):
        """Crash resilience — existing registration is reused, no POST needed."""
        import asyncio

        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={
                "status": 200,
                "data": [
                    {"id": 7, "url": url, "events": ["new-message"]},
                ],
            },
        )

        # Track whether POST was called
        post_called = False
        orig_api_post = adapter._api_post

        async def tracking_post(path, payload):
            nonlocal post_called
            post_called = True
            return await orig_api_post(path, payload)

        adapter._api_post = tracking_post

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())
        assert ok is True
        assert not post_called, "Should reuse existing, not POST again"

    @pytest.mark.asyncio
    async def test_register_migrates_existing_hook_without_inbound_event(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [{"id": 7, "url": url, "events": ["updated-message"]}]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete
        calls = []

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            calls.append(("delete", args[0]))
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def tracking_post(path, payload):
            calls.append(("post", payload))
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": created}

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", tracking_post)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert calls == [
            ("delete", adapter._api_url("/api/v1/webhook/7")),
            ("post", {"url": url, "events": ["new-message"]}),
        ]
        assert registrations == [{"id": 8, "url": url, "events": ["new-message"]}]

    @pytest.mark.asyncio
    async def test_register_migrates_realistic_idempotent_same_url_webhook(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        calls = []
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_post(path, payload):
            calls.append(("post", payload))
            # BlueBubbles addWebhook() returns an existing same-URL row
            # unchanged instead of updating its event list.
            existing = next(
                (item for item in registrations if item["url"] == payload["url"]),
                None,
            )
            if existing:
                return {"status": 200, "data": dict(existing)}
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": dict(created)}

        async def tracking_delete(*args, **kwargs):
            calls.append(("delete", args[0]))
            registrations[:] = [item for item in registrations if item["id"] != 7]
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        monkeypatch.setattr(adapter, "_api_post", tracking_post)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert calls == [
            ("delete", adapter._api_url("/api/v1/webhook/7")),
            ("post", {"url": url, "events": ["new-message"]}),
        ]
        assert registrations == [{"id": 8, "url": url, "events": ["new-message"]}]

    @pytest.mark.asyncio
    async def test_reused_registration_is_not_owned_or_removed_on_disconnect(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted = []
        adapter.client = self._mock_client(
            get_response={
                "status": 200,
                "data": [{"id": 7, "url": url, "events": ["new-message"]}],
            }
        )
        original_delete = adapter.client.delete

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == []

    @pytest.mark.asyncio
    async def test_cancelled_fresh_registration_remains_durable(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        started = asyncio.Event()
        release = asyncio.Event()
        registrations = []
        deleted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def delayed_post(path, payload):
            started.set()
            await release.wait()
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": created}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", delayed_post)
        adapter.client.delete = tracking_delete

        registration = asyncio.create_task(adapter._register_webhook())
        await started.wait()
        registration.cancel()
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await registration

        assert await adapter._unregister_webhook() is False
        assert deleted == []

    @pytest.mark.asyncio
    async def test_cancelled_registration_post_leaves_durable_replacement(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        post_started = asyncio.Event()
        release_post = asyncio.Event()
        deleted = []
        registrations = [
            {"id": 7, "url": url, "events": ["new-message", "updated-message"]}
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return list(registrations)

        async def delayed_post(path, payload):
            post_started.set()
            await release_post.wait()
            registrations.append({
                "id": 8,
                "url": url,
                "events": list(payload["events"]),
            })
            return {"status": 200, "data": {"id": 8}}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        monkeypatch.setattr(adapter, "_api_post", delayed_post)
        adapter.client.delete = tracking_delete

        registration = asyncio.create_task(adapter._register_webhook())
        await post_started.wait()
        registration.cancel()
        release_post.set()

        with pytest.raises(asyncio.CancelledError):
            await registration

        assert await adapter._unregister_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]

    @pytest.mark.asyncio
    async def test_partial_stale_delete_failure_restores_when_url_is_empty(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {"id": 7, "url": url, "events": ["updated-message"]},
            {"id": 8, "url": url, "events": ["updated-message"]},
        ]
        posted = []
        adapter.client = self._mock_client()

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def uncertain_delete(endpoint, **kwargs):
            webhook_id = int(endpoint.split("/api/v1/webhook/")[1].split("?")[0])
            registrations[:] = [
                item for item in registrations if item["id"] != webhook_id
            ]
            if webhook_id == 8:
                raise TimeoutError("delete response lost after commit")

            class Response:
                def raise_for_status(self):
                    return None

            return Response()

        async def restore_post(path, payload):
            posted.append(payload)
            restored = {"id": 9, **payload}
            registrations.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = uncertain_delete
        adapter._api_post = restore_post

        assert await adapter._register_webhook() is False
        assert posted == [{"url": url, "events": ["updated-message"]}]
        assert registrations == [{"id": 9, "url": url, "events": ["updated-message"]}]

    @pytest.mark.asyncio
    async def test_register_post_failure_restores_stale_hook(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted = []
        posted = []
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def replacement_then_rollback(path, payload):
            posted.append(payload)
            if len(posted) == 1:
                return {"status": 500, "message": "internal error"}
            restored = {"id": 9, **payload}
            registrations.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = replacement_then_rollback

        assert await adapter._register_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]
        assert posted == [
            {"url": url, "events": ["new-message"]},
            {"url": url, "events": ["new-message", "updated-message"]},
        ]
        assert registrations == [
            {
                "id": 9,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]

    @pytest.mark.asyncio
    async def test_register_reconciles_replacement_committed_before_timeout(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        posted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def committed_then_timed_out(path, payload):
            posted.append(payload)
            committed = {"id": 8, **payload}
            registrations.append(committed)
            raise TimeoutError("response lost after commit")

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = committed_then_timed_out

        assert await adapter._register_webhook() is True
        assert posted == [{"url": url, "events": ["new-message"]}]
        assert registrations == [{"id": 8, "url": url, "events": ["new-message"]}]

    @pytest.mark.asyncio
    async def test_register_does_not_delete_unexpected_post_failure_owner(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def unexpected_owner(path, payload):
            occupied = {
                "id": 8,
                "url": url,
                "events": ["updated-message"],
            }
            registrations.append(occupied)
            return {"status": 200, "data": occupied}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = unexpected_owner

        assert await adapter._register_webhook() is False
        assert registrations == [{"id": 8, "url": url, "events": ["updated-message"]}]

    @pytest.mark.asyncio
    async def test_connect_cancellation_cleans_listener_without_deleting_hook(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, webhook_port="0")
        cleanup_calls = []

        async def fake_api_get(path):
            if path == "/api/v1/server/info":
                return {"data": {}}
            return {"status": 200}

        async def cancelled_registration():
            raise asyncio.CancelledError

        async def tracking_unregister():
            cleanup_calls.append("unregister")
            return False

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(adapter, "_register_webhook", cancelled_registration)
        monkeypatch.setattr(adapter, "_unregister_webhook", tracking_unregister)

        with pytest.raises(asyncio.CancelledError):
            await adapter.connect()

        assert cleanup_calls == ["unregister"]
        assert adapter.client is None
        assert adapter._runner is None

    @pytest.mark.asyncio
    async def test_fresh_concurrent_winner_is_never_deleted(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = []
        deleted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def concurrent_winner(path, payload):
            winner = {"id": 99, **payload}
            registrations.append(winner)
            return {"status": 200, "data": winner}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", concurrent_winner)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == []
        assert registrations == [{"id": 99, "url": url, "events": ["new-message"]}]

    @pytest.mark.asyncio
    async def test_migration_concurrent_winner_is_never_deleted(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [{"id": 7, "url": url, "events": ["updated-message"]}]
        deleted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def concurrent_winner(path, payload):
            winner = {"id": 99, **payload}
            registrations.append(winner)
            return {"status": 200, "data": winner}

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", concurrent_winner)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]
        assert registrations == [{"id": 99, "url": url, "events": ["new-message"]}]

    # -- _unregister_webhook --

    def test_unregister_preserves_durable_registration(self, monkeypatch):
        """Disconnect never deletes an ambiguous fixed-URL registration."""
        import asyncio

        adapter = _make_adapter(monkeypatch)
        deleted_urls = []

        async def mock_delete(*args, **kwargs):
            deleted_urls.append(args[0] if args else "")

        adapter.client = self._mock_client()
        adapter.client.delete = mock_delete

        ok = asyncio.get_event_loop().run_until_complete(adapter._unregister_webhook())

        assert ok is False
        assert deleted_urls == []
