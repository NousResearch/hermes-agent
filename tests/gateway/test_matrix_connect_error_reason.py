"""A refused Matrix connect() must carry its reason into the delivery error (#111333).

``MatrixAdapter.connect()`` keeps the ``bool`` contract, so every caller used to
collapse three different causes -- no homeserver, failed auth, failed E2EE setup
(e.g. stale one-time keys on the homeserver) -- into ``Matrix connect failed``,
which left ``last_delivery_error`` with nothing for an operator to act on.
"""
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.outbound import senders
from plugins.platforms.matrix import adapter as matrix_adapter


def _stub_mautrix(monkeypatch):
    """Import-only stubs: connect() bails out before it would use any of them."""
    for name in ("mautrix", "mautrix.api", "mautrix.client", "mautrix.client.state_store"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(
            HTTPAPI=MagicMock(), Client=MagicMock(),
            MemoryStateStore=MagicMock(), MemorySyncStore=MagicMock()))


def _make_adapter(**extra) -> matrix_adapter.MatrixAdapter:
    extra.setdefault("homeserver", "https://matrix.example.org")
    extra.setdefault("user_id", "@bot:example.org")
    return matrix_adapter.MatrixAdapter(
        PlatformConfig(enabled=True, token="syt_test_token", extra=extra))


def test_connect_without_homeserver_reports_reason(monkeypatch):
    adapter = matrix_adapter.MatrixAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    # connect() imports mautrix before it checks the homeserver; other tests stub that package.
    _stub_mautrix(monkeypatch)
    assert asyncio.run(adapter.connect()) is False
    assert "MATRIX_HOMESERVER" in adapter.last_connect_error


def test_whoami_failure_reports_reason():
    adapter = _make_adapter()
    client, api = MagicMock(), MagicMock()
    client.whoami = AsyncMock(side_effect=RuntimeError("M_UNKNOWN_TOKEN"))
    api.session.close = AsyncMock()
    assert asyncio.run(adapter._connect_authenticate(client, api)) is False
    assert adapter.last_connect_error.startswith("whoami failed")
    assert "M_UNKNOWN_TOKEN" in adapter.last_connect_error


def test_missing_credentials_reports_reason():
    adapter = matrix_adapter.MatrixAdapter(
        PlatformConfig(enabled=True, token="", extra={"homeserver": "https://matrix.example.org"}))
    client, api = MagicMock(), MagicMock()
    api.session.close = AsyncMock()
    assert asyncio.run(adapter._connect_authenticate(client, api)) is False
    assert "MATRIX_ACCESS_TOKEN" in adapter.last_connect_error


def test_stale_one_time_keys_report_reason(tmp_path, monkeypatch):
    """The reported failure mode: share_keys() raises "already exists" for stale OTKs."""
    adapter = _make_adapter()
    adapter._store_dir = tmp_path / "store"  # _crypto_db_path resolves from here
    adapter._encryption, adapter._e2ee_mode = True, "required"
    monkeypatch.setattr(matrix_adapter, "_check_e2ee_deps", lambda: True)
    monkeypatch.setattr(adapter, "_reset_crypto_store_if_device_changed", AsyncMock(return_value=False))
    monkeypatch.setattr(adapter, "_migrate_legacy_crypto_pickle", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_verify_device_keys_on_server", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_verify_or_bootstrap_cross_signing", AsyncMock(return_value=None))

    class _Db:
        async def start(self):
            return None

        async def stop(self):
            return None

    class _Store:
        upgrade_table = MagicMock()

        def __init__(self, **_kw):
            pass

        async def open(self):
            return None

        async def put_device_id(self, _device_id):
            return None

    class _Olm:
        def __init__(self, *_a, **_kw):
            pass

        async def load(self):
            return None

        async def share_keys(self):
            raise RuntimeError("One-time key with same ID already exists")

    for name in ("mautrix", "mautrix.crypto", "mautrix.crypto.store",
                 "mautrix.crypto.store.asyncpg", "mautrix.util", "mautrix.util.async_db"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(
            OlmMachine=_Olm, PgCryptoStore=_Store,
            Database=types.SimpleNamespace(create=lambda *a, **kw: _Db())))

    client = MagicMock(device_id="DEVICEID")
    api = MagicMock(session=MagicMock(close=AsyncMock()))
    assert asyncio.run(adapter._connect_setup_e2ee(client, api, MagicMock())) is False
    assert "stale one-time keys" in adapter.last_connect_error


REASONS = (
    "MATRIX_HOMESERVER is not configured",
    "whoami failed: M_UNKNOWN_TOKEN",
    "device DEVICEID has stale one-time keys on the server signed with a previous identity key",
)


class _RefusingAdapter:
    def __init__(self, reason):
        self.last_connect_error = reason

    async def connect(self):
        return False


class _RefusingAdapterNoReason:
    async def connect(self):
        return False


@pytest.mark.parametrize("reason", REASONS)
def test_delivery_error_carries_the_connect_reason(monkeypatch, reason):
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (None, None))
    monkeypatch.setitem(sys.modules, "plugins.platforms.matrix.adapter",
                        types.SimpleNamespace(MatrixAdapter=lambda _p: _RefusingAdapter(reason)))
    result = asyncio.run(senders._send_matrix_via_adapter(MagicMock(), "!room:example.org", "hi"))
    assert reason in result["error"], result
    assert result["error"].startswith("Matrix connect failed: ")


def test_delivery_error_falls_back_when_adapter_has_no_reason(monkeypatch):
    """Older/3rd-party adapters without the attribute keep the old text."""
    monkeypatch.setattr(senders, "_live_adapter", lambda *a, **kw: (None, None))
    monkeypatch.setitem(sys.modules, "plugins.platforms.matrix.adapter",
                        types.SimpleNamespace(MatrixAdapter=lambda _p: _RefusingAdapterNoReason()))
    result = asyncio.run(senders._send_matrix_via_adapter(MagicMock(), "!room:example.org", "hi"))
    assert result["error"] == "Matrix connect failed"
