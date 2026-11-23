"""Matrix crypto store must be pinned per profile at connect(), not at import.

Under ``gateway.multiplex_profiles`` one process imports
``plugins.platforms.matrix.adapter`` once; the old module-level
``_STORE_DIR``/``_CRYPTO_DB_PATH`` resolved against the root HERMES_HOME at
import time, so every profile's adapter opened the SAME crypto.db and inbound
E2EE failed with "no session found" (#89168). ``connect()`` calls
``_resolve_store_dir()`` inside ``_profile_runtime_scope`` (context-local
HERMES_HOME), so resolving there -- and caching on the instance -- gives each
profile its own store. Exercised via ``_resolve_store_dir`` directly so the
test needs no mautrix install.
"""
import asyncio

from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.matrix import adapter as matrix_adapter


def _make_adapter() -> matrix_adapter.MatrixAdapter:
    return matrix_adapter.MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
        )
    )


def test_store_dir_pinned_to_each_profile_home(tmp_path):
    """Two profiles resolving in one process get two stores, and each
    adapter keeps reporting its own store after the scope is gone."""
    stores = {}
    for profile in ("accountant", "engineering-lead"):
        home = tmp_path / "profiles" / profile
        home.mkdir(parents=True)
        adapter = _make_adapter()
        token = set_hermes_home_override(str(home))
        try:
            adapter._resolve_store_dir().mkdir(parents=True, exist_ok=True)
        finally:
            reset_hermes_home_override(token)
        # Cached on the instance: correct even when read outside the scope.
        path = adapter.get_diagnostics()["e2ee"]["crypto_store_path"]
        assert path.startswith(str(home)), f"store not profile-scoped: {path}"
        assert adapter._store_dir.is_dir()
        stores[profile] = path

    assert stores["accountant"] != stores["engineering-lead"]


import pytest


@pytest.mark.parametrize("move", ("delete", "rename"))
def test_store_dir_and_recovery_key_skip_a_deleted_or_renamed_profile_home(tmp_path, monkeypatch, move):
    """Neither the crypto store nor a recovery-key file under the home recreates it (#155595 U3)."""
    from tests.fixtures.dead_profile_home import dead_named_home

    home = dead_named_home(tmp_path, move)
    monkeypatch.setattr(matrix_adapter, "_recovery_key_output_path", lambda: home / "matrix" / "recovery.key")
    path, reason = matrix_adapter._get_matrix_recovery_key_output_target()
    assert path is None and reason.startswith("unusable: Named profile home does not exist")
    with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
        matrix_adapter._write_matrix_recovery_key_output_file("key")
    from unittest.mock import patch
    from tests.gateway.test_matrix import _make_fake_mautrix

    adapter = _make_adapter()
    token = set_hermes_home_override(str(home))
    try:
        with patch.dict("sys.modules", _make_fake_mautrix()), \
                pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            asyncio.run(adapter.connect())  # resolves and creates the store before any network call
    finally:
        reset_hermes_home_override(token)
    assert not home.exists()


@pytest.mark.parametrize("move", ("delete", "rename"))
def test_e2ee_setup_refuses_a_store_under_a_deleted_or_renamed_profile_home(tmp_path, monkeypatch, move):
    """E2EE setup's store dir goes through the boundary: required mode aborts, the home stays gone."""
    import sys
    import types
    from unittest.mock import AsyncMock, MagicMock
    from tests.fixtures.dead_profile_home import dead_named_home

    home = dead_named_home(tmp_path, move)
    adapter = _make_adapter()
    adapter._store_dir = home / "platforms" / "matrix" / "store"
    adapter._encryption, adapter._e2ee_mode = True, "required"
    monkeypatch.setattr(matrix_adapter, "_check_e2ee_deps", lambda: True)
    for name in ("mautrix", "mautrix.crypto", "mautrix.crypto.store", "mautrix.crypto.store.asyncpg",
                 "mautrix.util", "mautrix.util.async_db"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(
            OlmMachine=object, PgCryptoStore=object, Database=object))
    api = MagicMock(session=MagicMock(close=AsyncMock()))
    assert asyncio.run(adapter._connect_setup_e2ee(MagicMock(), api, MagicMock())) is False
    assert not home.exists()
