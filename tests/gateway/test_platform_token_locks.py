"""Token-lock guards on the credential/port-binding platform adapters.

Each adapter must refuse to connect when its unique identity is already in use
by another profile (``gateway.status.acquire_scoped_lock`` returning False) and
release the lock on disconnect. These four adapters historically had no lock, so
two profiles could share one WhatsApp phone number, BlueBubbles server, or
webhook bind.

Port of the fork's ``73efc1cfaa`` tests, adapted to the generic
``BasePlatformAdapter._acquire_platform_lock`` / ``_release_platform_lock``
helpers upstream now provides (the fork called ``acquire_scoped_lock`` directly).
Shared-listener secondary profiles bind nothing and must skip the lock, so the
adapters guard on ``shared_ingress_profile``; these tests exercise the
default-profile (binding) path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig


async def _connect_refuses_when_locked(adapter, scope: str, identity: str) -> None:
    with patch(
        "gateway.status.acquire_scoped_lock", return_value=(False, None)
    ) as mock_acquire:
        result = await adapter.connect()

    assert result is False
    assert adapter.fatal_error_code == f"{scope}_lock"
    # The base helper forwards a metadata kwarg; assert scope/identity positionally.
    assert mock_acquire.call_args.args[:2] == (scope, identity)


async def _disconnect_releases_lock(adapter, scope: str, identity: str) -> None:
    adapter._platform_lock_scope = scope
    adapter._platform_lock_identity = identity
    with patch("gateway.status.release_scoped_lock") as mock_release:
        await adapter.disconnect()
    mock_release.assert_called_once_with(scope, identity)


class TestWhatsAppCloudLock:
    @pytest.mark.anyio
    async def test_connect_refuses_when_locked(self):
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

        adapter = WhatsAppCloudAdapter(
            PlatformConfig(
                enabled=True,
                extra={"phone_number_id": "1234567890", "access_token": "tok"},
            )
        )
        await _connect_refuses_when_locked(adapter, "whatsapp_cloud", "1234567890")

    @pytest.mark.anyio
    async def test_disconnect_releases_lock(self):
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

        adapter = WhatsAppCloudAdapter(
            PlatformConfig(
                enabled=True,
                extra={"phone_number_id": "1234567890", "access_token": "tok"},
            )
        )
        await _disconnect_releases_lock(adapter, "whatsapp_cloud", "1234567890")


class TestBlueBubblesLock:
    @pytest.mark.anyio
    async def test_connect_refuses_when_locked(self):
        from gateway.platforms.bluebubbles import BlueBubblesAdapter

        adapter = BlueBubblesAdapter(
            PlatformConfig(
                enabled=True,
                extra={"server_url": "http://localhost:1234", "password": "secret"},
            )
        )
        await _connect_refuses_when_locked(
            adapter, "bluebubbles", "http://localhost:1234"
        )

    @pytest.mark.anyio
    async def test_disconnect_releases_lock(self):
        from gateway.platforms.bluebubbles import BlueBubblesAdapter

        adapter = BlueBubblesAdapter(
            PlatformConfig(
                enabled=True,
                extra={"server_url": "http://localhost:1234", "password": "secret"},
            )
        )
        await _disconnect_releases_lock(
            adapter, "bluebubbles", "http://localhost:1234"
        )


class TestMSGraphWebhookLock:
    @pytest.mark.anyio
    async def test_connect_refuses_when_locked(self):
        from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter

        adapter = MSGraphWebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 8000,
                    "client_state": "expected-client-state",
                    "accepted_resources": ["communications/onlineMeetings"],
                },
            )
        )
        await _connect_refuses_when_locked(
            adapter, "msgraph_webhook", "127.0.0.1:8000"
        )

    @pytest.mark.anyio
    async def test_disconnect_releases_lock(self):
        from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter

        adapter = MSGraphWebhookAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 8000,
                    "client_state": "expected-client-state",
                    "accepted_resources": ["communications/onlineMeetings"],
                },
            )
        )
        await _disconnect_releases_lock(
            adapter, "msgraph_webhook", "127.0.0.1:8000"
        )


class TestAPIServerLock:
    @pytest.mark.anyio
    async def test_connect_refuses_when_locked(self):
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8000})
        )
        with patch.object(
            adapter, "_api_key_passes_startup_guard", return_value=True
        ):
            await _connect_refuses_when_locked(adapter, "api_server", "127.0.0.1:8000")

    @pytest.mark.anyio
    async def test_disconnect_releases_lock(self):
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8000})
        )
        await _disconnect_releases_lock(adapter, "api_server", "127.0.0.1:8000")
