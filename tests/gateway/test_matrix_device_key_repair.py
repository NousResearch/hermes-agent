"""Regression tests: automatic repair of mismatched server device records.

When the homeserver's record for our device diverges from the local crypto
store, the adapter rebinds the record to the intact local identity: delete
the mismatched record (completing password UIA when the homeserver requires
it), re-authenticate the same device ID if the deletion invalidated the
token, re-upload the local keys, and verify. Accounts that already shared
keys without a configured password stay fail-closed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig

USER_ID = "@bot:example.org"
DEVICE_ID = "DEV123"


def _make_adapter(password: str = "pw"):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_access_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": USER_ID,
            "encryption": True,
            "device_id": DEVICE_ID,
        },
    )
    adapter = MatrixAdapter(config)
    adapter._password = password
    return adapter


def _record(keys: dict):
    dev = MagicMock()
    dev.keys = keys
    resp = MagicMock()
    resp.device_keys = {USER_ID: {DEVICE_ID: dev}}
    return resp


def _empty_resp():
    resp = MagicMock()
    resp.device_keys = {USER_ID: {}}
    return resp


def _mock_client():
    client = MagicMock()
    client.mxid = USER_ID
    client.device_id = DEVICE_ID
    client.api = MagicMock()
    client.api.token = "syt_test_access_token"
    client.api.request = AsyncMock(return_value=None)
    client.login = AsyncMock()
    return client


def _mock_olm(shared: bool = True):
    olm = MagicMock()
    olm.account = MagicMock()
    olm.account.shared = shared
    olm.account.identity_keys = {"ed25519": "local_new", "curve25519": "c1"}
    olm.share_keys = AsyncMock()
    return olm


class _UnknownTokenError(Exception):
    errcode = "M_UNKNOWN_TOKEN"


class _UiaRequired(Exception):
    http_status = 401
    text = '{"session": "uia-session-abc"}'


@pytest.mark.asyncio
async def test_mismatch_with_password_repairs_and_reauthenticates_same_device():
    adapter = _make_adapter(password="pw")
    client = _mock_client()
    olm = _mock_olm(shared=True)

    mismatched = _record({"ed25519:DEV123": "server_old", "curve25519:DEV123": "c1"})
    fixed = _record({"ed25519:DEV123": "local_new", "curve25519:DEV123": "c1"})
    client.query_keys = AsyncMock(side_effect=[mismatched, fixed])
    client.login = AsyncMock(side_effect=lambda **kw: setattr(client.api, "token", "fresh-token"))

    with patch.object(
        type(adapter), "_has_valid_device_self_signature", staticmethod(lambda *a: True)
    ):
        result = await adapter._verify_device_keys_on_server(client, olm)

    assert result is True
    # Server record deleted, token re-bound to the SAME device ID via password
    # login, local identity re-uploaded and re-verified.
    client.api.request.assert_awaited_once()
    assert "/devices/DEV123" in client.api.request.await_args.args[1]
    client.login.assert_awaited_once()
    assert client.login.await_args.kwargs["device_id"] == DEVICE_ID
    assert client.login.await_args.kwargs["password"] == "pw"
    assert adapter._access_token == "fresh-token"
    olm.share_keys.assert_awaited_once()
    assert olm.account.shared is False


@pytest.mark.asyncio
async def test_delete_server_device_completes_password_uia():
    adapter = _make_adapter(password="pw")
    client = _mock_client()
    client.api.request = AsyncMock(side_effect=[_UiaRequired(), None])

    assert await adapter._delete_server_device(client) is True

    assert client.api.request.await_count == 2
    uia_call = client.api.request.await_args_list[1]
    auth = uia_call.args[2]["auth"]
    assert auth["type"] == "m.login.password"
    assert auth["session"] == "uia-session-abc"
    assert auth["password"] == "pw"
    assert auth["identifier"] == {"type": "m.id.user", "user": USER_ID}
    assert uia_call.kwargs.get("sensitive") is True


@pytest.mark.asyncio
async def test_uia_without_password_fails_closed():
    adapter = _make_adapter(password="")
    client = _mock_client()
    client.api.request = AsyncMock(side_effect=[_UiaRequired()])

    assert await adapter._delete_server_device(client) is False
    assert client.api.request.await_count == 1  # no passwordless retry


@pytest.mark.asyncio
async def test_shared_account_without_password_stays_fail_closed():
    adapter = _make_adapter(password="")
    client = _mock_client()
    olm = _mock_olm(shared=True)

    mismatched = _record({"ed25519:DEV123": "server_old", "curve25519:DEV123": "c1"})
    client.query_keys = AsyncMock(return_value=mismatched)

    with patch.object(
        type(adapter), "_has_valid_device_self_signature", staticmethod(lambda *a: True)
    ):
        result = await adapter._verify_device_keys_on_server(client, olm)

    assert result is False
    client.api.request.assert_not_awaited()  # no repair attempted
    olm.share_keys.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_token_on_upload_triggers_reauth_and_retry():
    adapter = _make_adapter(password="pw")
    client = _mock_client()
    olm = _mock_olm(shared=False)

    fixed = _record({"ed25519:DEV123": "local_new", "curve25519:DEV123": "c1"})
    client.query_keys = AsyncMock(side_effect=[_empty_resp(), fixed])
    olm.share_keys = AsyncMock(side_effect=[_UnknownTokenError("M_UNKNOWN_TOKEN"), None])
    client.login = AsyncMock(side_effect=lambda **kw: setattr(client.api, "token", "fresh-token"))

    with patch.object(
        type(adapter), "_has_valid_device_self_signature", staticmethod(lambda *a: True)
    ):
        result = await adapter._verify_device_keys_on_server(client, olm)

    assert result is True
    assert olm.share_keys.await_count == 2
    client.login.assert_awaited_once()
    assert client.login.await_args.kwargs["device_id"] == DEVICE_ID


@pytest.mark.asyncio
async def test_repair_fails_closed_when_server_delete_fails():
    adapter = _make_adapter(password="pw")
    client = _mock_client()
    olm = _mock_olm(shared=False)
    client.api.request = AsyncMock(side_effect=ConnectionError("homeserver unreachable"))

    result = await adapter._repair_server_device_keys(client, olm, "local_new", "c1")

    assert result is False
    olm.share_keys.assert_not_awaited()
    client.login.assert_not_awaited()
