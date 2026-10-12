"""Regressions for room-key recovery, crash-safe OTK upload and store ownership."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from gateway.config import PlatformConfig
from tests.gateway.test_matrix import _make_adapter, _make_fake_mautrix


class TestMatrixStoreOwnership:
    @pytest.mark.asyncio
    async def test_second_e2ee_process_stops_before_opening_crypto_store(self):
        """A held store lock must reject a second OlmMachine owner."""
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="syt_test_access_token",
                extra={
                    "homeserver": "https://matrix.example.org",
                    "user_id": "@bot:example.org",
                    "encryption": True,
                },
            )
        )
        adapter._acquire_platform_lock = MagicMock(return_value=False)

        fake_mautrix_mods = _make_fake_mautrix()
        mock_client = MagicMock()
        mock_client.mxid = "@bot:example.org"
        mock_client.device_id = None
        mock_client.whoami = AsyncMock(
            return_value=MagicMock(
                user_id="@bot:example.org", device_id="DEV123"
            )
        )
        fake_mautrix_mods["mautrix.client"].Client = MagicMock(
            return_value=mock_client
        )
        database_cls = fake_mautrix_mods["mautrix.util.async_db"].Database
        database_create = MagicMock(wraps=database_cls.create)
        database_cls.create = database_create

        import plugins.platforms.matrix.adapter as matrix_mod

        with patch.object(
            matrix_mod, "_check_e2ee_deps", return_value=True
        ), patch.dict("sys.modules", fake_mautrix_mods):
            assert await adapter.connect() is False

        adapter._acquire_platform_lock.assert_called_once_with(
            "matrix-crypto-store",
            str(adapter._crypto_db_path.resolve()),
            "Matrix crypto store",
        )
        database_create.assert_not_called()


class TestMatrixRoomKeyRecovery:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._allowed_user_ids = {"@alice:example.org"}
        self.adapter._allowed_room_ids = set()
        self.adapter._joined_rooms = {"!room:example.org"}

        self.olm = MagicMock()
        self.olm.share_keys_min_trust = 0
        self.olm.resolve_trust = AsyncMock(return_value=0)
        self.olm.default_allow_key_share = AsyncMock(return_value=True)

        self.client = MagicMock()
        self.client.mxid = "@bot:example.org"
        self.client.crypto = self.olm
        self.client.get_joined_members = AsyncMock(
            return_value={
                "@bot:example.org": MagicMock(),
                "@alice:example.org": MagicMock(),
            }
        )
        self.adapter._client = self.client

        self.device = MagicMock()
        self.device.user_id = "@alice:example.org"
        self.device.device_id = "ALICEPHONE"
        self.device.deleted = False
        self.request = MagicMock()
        self.request.room_id = "!room:example.org"

    @pytest.mark.asyncio
    async def test_allows_explicitly_allowed_current_room_member(self):
        assert await self.adapter._allow_matrix_key_share(self.device, self.request)
        self.olm.resolve_trust.assert_awaited_once_with(self.device)

    @pytest.mark.asyncio
    async def test_rejects_user_not_in_explicit_allowlist(self):
        self.device.user_id = "@mallory:example.org"

        assert not await self.adapter._allow_matrix_key_share(self.device, self.request)
        self.client.get_joined_members.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_user_who_is_no_longer_joined(self):
        self.client.get_joined_members.return_value = {
            "@bot:example.org": MagicMock()
        }

        assert not await self.adapter._allow_matrix_key_share(self.device, self.request)

    @pytest.mark.asyncio
    async def test_rejects_room_the_bot_has_not_joined(self):
        self.request.room_id = "!other:example.org"

        assert not await self.adapter._allow_matrix_key_share(self.device, self.request)
        self.client.get_joined_members.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_when_membership_cannot_be_verified(self):
        self.client.get_joined_members.side_effect = RuntimeError(
            "homeserver unavailable"
        )

        assert not await self.adapter._allow_matrix_key_share(self.device, self.request)

    @pytest.mark.asyncio
    async def test_same_user_uses_mautrix_default_policy(self):
        self.device.user_id = "@bot:example.org"

        assert await self.adapter._allow_matrix_key_share(self.device, self.request)
        self.olm.default_allow_key_share.assert_awaited_once_with(
            self.device, self.request
        )
        self.client.get_joined_members.assert_not_awaited()


class TestMatrixCrashSafeOtkUpload:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.call_order = []

        self.account = MagicMock()
        self.account.shared = True
        self.account.max_one_time_keys = 100
        self.account.one_time_keys = {"curve25519": {}}
        self.account.get_one_time_keys.return_value = {
            "signed_curve25519:AAAA": {"key": "public"}
        }
        self.account.mark_keys_as_published.side_effect = (
            lambda: self.call_order.append("mark")
        )

        self.store = MagicMock()
        self.store.put_account = AsyncMock(
            side_effect=lambda account: self.call_order.append("persist")
        )
        self.client = MagicMock()
        self.client.mxid = "@bot:example.org"
        self.client.device_id = "BOTDEVICE"
        self.client.upload_keys = AsyncMock(
            side_effect=lambda **kwargs: self.call_order.append("upload") or {}
        )
        self.olm = MagicMock()
        self.olm.account = self.account
        self.olm.client = self.client
        self.olm.crypto_store = self.store
        self.olm._last_key_share = 0.0

    @pytest.mark.asyncio
    async def test_persists_private_otks_before_network_upload(self):
        await self.adapter._share_matrix_keys_crash_safely(self.olm, 0)

        assert self.call_order == ["persist", "upload", "mark", "persist"]

    @pytest.mark.asyncio
    async def test_upload_failure_keeps_generated_private_otks_durable(self):
        async def fail_upload(**kwargs):
            self.call_order.append("upload")
            raise RuntimeError("response lost")

        self.client.upload_keys.side_effect = fail_upload

        with pytest.raises(RuntimeError, match="response lost"):
            await self.adapter._share_matrix_keys_crash_safely(self.olm, 0)

        assert self.call_order == ["persist", "upload"]
        self.account.mark_keys_as_published.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_existing_unpublished_keys_without_new_generation(self):
        self.account.one_time_keys = {
            "curve25519": {"AAAA": "public", "AAAB": "public"}
        }

        await self.adapter._share_matrix_keys_crash_safely(self.olm, 0)

        self.account.get_one_time_keys.assert_called_once_with(
            "@bot:example.org", "BOTDEVICE", 50
        )



@pytest.mark.platforms("posix")
def test_real_store_lock_refuses_same_process_owner_and_recovers(tmp_path):
    first, second = _make_adapter(), _make_adapter()
    for adapter in (first, second):
        adapter._store_dir = tmp_path
    try:
        assert first._acquire_matrix_store_lock()
        assert not second._acquire_matrix_store_lock()
        second._release_matrix_store_lock()
        assert not second._acquire_matrix_store_lock()
        first._release_matrix_store_lock()
        assert second._acquire_matrix_store_lock()
    finally:
        first._release_matrix_store_lock()
        second._release_matrix_store_lock()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["optional", "required"])
async def test_encrypted_standalone_sender_refuses_plaintext(mode):
    from plugins.platforms.matrix.adapter import _standalone_send
    from types import SimpleNamespace
    result = await _standalone_send(SimpleNamespace(extra={"e2ee_mode": mode}), "!room:example.org", "hello")
    assert "live gateway" in result["error"]
