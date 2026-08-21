"""Regression test: Matrix crypto store must be resolved per-instance, not
at module scope, so multiplexed profiles never share one crypto.db.

Under ``gateway.multiplex_profiles`` a single gateway process imports
``plugins.platforms.matrix.adapter`` ONCE. The old module-level
``_STORE_DIR``/``_CRYPTO_DB_PATH`` resolved against the root HERMES_HOME
at import time, so every profile's adapter opened the SAME crypto.db — all
bots' Olm identities landed in one store and inbound E2EE failed with
"Error decrypting megolm event, no session found". The fix mirrors the
pairing-store migration (a6397c379): resolve the store path per instance
through the active profile's HERMES_HOME (``get_hermes_dir`` honors the
context-local override installed by ``_profile_runtime_scope``).

These tests exercise the resolver directly with ``set_hermes_home_override``,
the same contextvar the multiplexer uses, so no network or mautrix needed.
"""
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.matrix.adapter import MatrixAdapter
from gateway.config import PlatformConfig


def _make_adapter() -> MatrixAdapter:
    return MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )


def test_two_profiles_resolve_distinct_crypto_stores(tmp_path):
    """Two profile homes must yield different crypto.db paths."""
    prof_a = tmp_path / "profiles" / "accountant"
    prof_b = tmp_path / "profiles" / "engineering-lead"
    prof_a.mkdir(parents=True)
    prof_b.mkdir(parents=True)

    adapter = _make_adapter()

    token_a = set_hermes_home_override(str(prof_a))
    try:
        path_a = adapter._get_store_path()
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(prof_b))
    try:
        path_b = adapter._get_store_path()
    finally:
        reset_hermes_home_override(token_b)

    assert path_a != path_b
    assert path_a.name == "crypto.db"
    assert path_b.name == "crypto.db"
    # Each path lives under its own profile home — never a shared root.
    assert str(path_a).replace("\\", "/").startswith(
        str(prof_a).replace("\\", "/")
    ), f"store not profile-scoped: {path_a}"
    assert str(path_b).replace("\\", "/").startswith(
        str(prof_b).replace("\\", "/")
    ), f"store not profile-scoped: {path_b}"


def test_store_path_is_resolved_per_call_not_cached_at_module_scope(tmp_path):
    """Changing the active profile changes the resolved path — no module-level pin."""
    prof_a = tmp_path / "profiles" / "a"
    prof_b = tmp_path / "profiles" / "b"
    prof_a.mkdir(parents=True)
    prof_b.mkdir(parents=True)

    adapter = _make_adapter()

    token_a = set_hermes_home_override(str(prof_a))
    try:
        first = adapter._get_store_path()
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(prof_b))
    try:
        second = adapter._get_store_path()
    finally:
        reset_hermes_home_override(token_b)

    assert first != second
    assert first.parent.name == "store"
    assert second.parent.name == "store"
