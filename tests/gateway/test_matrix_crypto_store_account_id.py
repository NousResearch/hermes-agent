"""Matrix crypto store account id after password login."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from tests.gateway.matrix_helpers import make_fake_mautrix


class TestCryptoStoreAccountId:
    @pytest.mark.asyncio
    async def test_password_login_keeps_configured_store_account_and_adopts_canonical_id(self):
        """An E2EE store written before the upgrade is keyed by the configured MATRIX_USER_ID,
        while self-identity checks use the homeserver's spelling of it."""
        from types import SimpleNamespace
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@Bot:example.org",
            "password": "secret",
            "device_id": "STABLE",
            "encryption": True,
        }))

        fake = make_fake_mautrix()
        opened = []
        login_identifiers = []

        class RecordingStore:
            upgrade_table = MagicMock()

            def __init__(self, account_id="", **_kw):
                opened.append(account_id)

            async def open(self):
                pass

            async def get_device_id(self):
                return "STABLE"

            async def put_device_id(self, device_id):
                pass

            async def get_account(self):
                return MagicMock()

        fake["mautrix.crypto.store.asyncpg"].PgCryptoStore = RecordingStore

        client = MagicMock()
        client.mxid = "@Bot:example.org"
        client.device_id = None
        client.crypto = None

        async def login(**kwargs):
            login_identifiers.append(kwargs["identifier"])
            client.mxid = "@bot:example.org"
            return SimpleNamespace(device_id="STABLE", user_id="@bot:example.org")

        client.login = login
        client.sync = AsyncMock(return_value={"rooms": {"join": {}}})
        client.api.token = ""
        client.api.session.close = AsyncMock()
        olm = MagicMock()
        olm.load = AsyncMock()
        olm.share_keys = AsyncMock()
        fake["mautrix.client"].Client = MagicMock(return_value=client)
        fake["mautrix.crypto"].OlmMachine = MagicMock(return_value=olm)

        import plugins.platforms.matrix.adapter as matrix_mod
        with patch.object(matrix_mod, "_check_e2ee_deps", return_value=True), \
                patch.dict("sys.modules", fake), \
                patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
                patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
                patch.object(adapter, "_verify_device_keys_on_server", AsyncMock(return_value=True)), \
                patch.object(adapter, "_verify_or_bootstrap_cross_signing", AsyncMock()):
            assert await adapter.connect() is True
            assert await adapter.connect(is_reconnect=True) is True

        await adapter.disconnect()
        assert (login_identifiers, opened, adapter._user_id) == (
            ["@Bot:example.org"] * 2, ["@Bot:example.org"] * 2, "@bot:example.org",
        )
