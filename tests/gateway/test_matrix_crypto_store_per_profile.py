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
