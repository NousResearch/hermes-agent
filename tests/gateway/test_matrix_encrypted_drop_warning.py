"""Matrix fails loud when encrypted room events arrive with no E2EE decryptor (#131778)."""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


class TestMatrixEncryptedDropWarning:

    @staticmethod
    def _cryptoless_client():
        """Client with no decryptor attached (E2EE off, or the optional mode degraded)."""
        fake_client = MagicMock()
        fake_client.crypto = None
        fake_client.sync_store = MagicMock()
        fake_client.sync_store.put_next_batch = AsyncMock()
        fake_client.handle_sync = MagicMock(return_value=[])
        return fake_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode, hint", [
        ("off", "set MATRIX_E2EE_MODE=optional"),
        # Degraded optional mode: the install/enable advice is wrong; point at the setup failure.
        ("optional", "E2EE mode is optional but the decryptor was not set up"),
    ])
    async def test_absorb_sync_warns_once_per_room_for_encrypted_events_without_crypto(
        self, caplog, mode, hint
    ):
        """Encrypted events with no decryptor must fail loud, once per room (#131778)."""
        adapter = _make_adapter()
        adapter._closing = False
        adapter._e2ee_mode = mode
        fake_client = self._cryptoless_client()
        adapter._client = fake_client

        sync_data = {
            "rooms": {"join": {
                "!enc:example.org": {"timeline": {"events": [
                    {"type": "m.room.encrypted", "event_id": "$e1"}]}},
                "!plain:example.org": {"timeline": {"events": [
                    {"type": "m.room.message", "event_id": "$m1"}]}},
            }},
            "next_batch": "s1",
        }
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
            await adapter._absorb_sync(fake_client, sync_data)
            await adapter._absorb_sync(fake_client, sync_data)  # rate-limited: one warning per room

        warnings = [r.getMessage() for r in caplog.records if "encrypted" in r.getMessage()]
        assert len(warnings) == 1
        assert "!enc:example.org" in warnings[0] and hint in warnings[0]
        assert (mode == "off") == ("pip install" in warnings[0])

    @pytest.mark.asyncio
    async def test_absorb_sync_no_encrypted_warning_when_crypto_attached(self, caplog):
        """With a decryptor attached, mautrix's own machinery reports failures instead."""
        adapter = _make_adapter()
        adapter._closing = False
        fake_client = self._cryptoless_client()
        # OlmMachine attached: mautrix's DecryptionDispatcher handles encrypted events.
        fake_client.crypto = MagicMock()
        adapter._client = fake_client

        sync_data = {
            "rooms": {"join": {"!enc:example.org": {
                "timeline": {"events": [{"type": "m.room.encrypted", "event_id": "$e1"}]}}}},
            "next_batch": "s1",
        }
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
            await adapter._absorb_sync(fake_client, sync_data)

        assert not [r for r in caplog.records if "encrypted" in r.getMessage()]
