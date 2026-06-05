"""Tests for the 1Password SDK-based secret source.

These tests exercise the shared cache substrate (``_cache.py``) and
the 1Password secret source (``onepassword.py``).  They run with
``HERMES_HOME`` redirected to a temp directory so they never touch
the real ``~/.hermes/``.

Integration tests that need a live 1Password service account token
are skipped when ``OP_SERVICE_ACCOUNT_TOKEN`` is not set in the
environment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.secret_sources._cache import (
    CachedFetch,
    FetchResult,
    TwoLayerCache,
    is_valid_env_name,
    resolve_cache_home,
)
from agent.secret_sources.onepassword import (
    _cache_key,
    _refs_fingerprint,
    _sanitise_env_name,
    _sdk_available,
    _token_fingerprint,
    apply_onepassword_secrets,
    json_dumps_stable,
    _reset_cache_for_tests,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path: Path) -> None:
    """Redirect HERMES_HOME to a temp dir for every test."""
    os.environ["HERMES_HOME"] = str(tmp_path)
    yield
    os.environ.pop("HERMES_HOME", None)


@pytest.fixture
def cache() -> TwoLayerCache[str]:
    """Fresh TwoLayerCache instance for each test."""
    c = TwoLayerCache[str](basename="test_cache.json")
    yield c
    c.clear()


# ---------------------------------------------------------------------------
# is_valid_env_name
# ---------------------------------------------------------------------------


class TestIsValidEnvName:
    def test_valid_names(self):
        assert is_valid_env_name("OPENAI_API_KEY")
        assert is_valid_env_name("_PRIVATE_KEY")
        assert is_valid_env_name("A")
        assert is_valid_env_name("A_B_C_123")

    def test_invalid_names(self):
        assert not is_valid_env_name("")
        assert not is_valid_env_name("1LEADING_DIGIT")
        assert not is_valid_env_name("has spaces")
        assert not is_valid_env_name("has-dashes")
        assert not is_valid_env_name("has.dots")
        assert not is_valid_env_name(None)  # type: ignore


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------


class TestFetchResult:
    def test_defaults(self):
        fr = FetchResult()
        assert fr.secrets == {}
        assert fr.applied == []
        assert fr.skipped == []
        assert fr.warnings == []
        assert fr.error is None
        assert fr.ok is True

    def test_error(self):
        fr = FetchResult(error="something went wrong")
        assert fr.ok is False

    def test_cache_hit(self):
        fr = FetchResult(cache_hit=True, secrets={"KEY": "val"})
        assert fr.cache_hit is True


# ---------------------------------------------------------------------------
# CachedFetch
# ---------------------------------------------------------------------------


class TestCachedFetch:
    def test_is_fresh(self):
        cf = CachedFetch(secrets={"A": "1"}, fetched_at=time.time())
        assert cf.is_fresh(300) is True

    def test_is_stale(self):
        cf = CachedFetch(secrets={"A": "1"}, fetched_at=time.time() - 301)
        assert cf.is_fresh(300) is False

    def test_ttl_zero_never_fresh(self):
        cf = CachedFetch(secrets={"A": "1"}, fetched_at=time.time())
        assert cf.is_fresh(0) is False


# ---------------------------------------------------------------------------
# TwoLayerCache
# ---------------------------------------------------------------------------


class TestTwoLayerCache:
    def test_read_miss_empty_cache(self, cache):
        assert cache.read("key1", ttl_seconds=300) is None

    def test_read_write_roundtrip(self, cache, tmp_path):
        home = tmp_path / "hermes"
        home.mkdir()
        entry = CachedFetch(secrets={"KEY": "value"}, fetched_at=time.time())
        cache.write("key1", entry, home_path=home)

        result = cache.read("key1", ttl_seconds=300, home_path=home)
        assert result is not None
        assert result.secrets == {"KEY": "value"}

    def test_read_l1_hit(self, cache):
        """Second read within TTL hits L1 (in-process), not L2 (disk)."""
        entry = CachedFetch(secrets={"K": "V"}, fetched_at=time.time())
        cache.write("k1", entry)

        # First read — fills L1
        cache.read("k1", ttl_seconds=300)

        # Second read — L1 hit (instant, even with no disk)
        result = cache.read("k1", ttl_seconds=300)
        assert result is not None
        assert result.secrets == {"K": "V"}

    def test_ttl_zero_disables_cache(self, cache):
        entry = CachedFetch(secrets={"K": "V"}, fetched_at=time.time())
        cache.write("k1", entry)
        assert cache.read("k1", ttl_seconds=0) is None

    def test_disk_persistence_across_instances(self, tmp_path):
        """Two separate cache instances share the same disk file."""
        home = tmp_path / "hermes"
        home.mkdir()

        c1 = TwoLayerCache[str](basename="shared.json")
        c2 = TwoLayerCache[str](basename="shared.json")

        entry = CachedFetch(secrets={"SHARED": "yes"}, fetched_at=time.time())
        c1.write("shared-key", entry, home_path=home)

        result = c2.read("shared-key", ttl_seconds=300, home_path=home)
        assert result is not None
        assert result.secrets == {"SHARED": "yes"}

        c1.clear(home_path=home)
        c2.clear(home_path=home)

    def test_merge_entries_no_overwrite(self, cache, tmp_path):
        """Writing a new key preserves existing sibling entries."""
        home = tmp_path / "hermes"
        home.mkdir()

        e1 = CachedFetch(secrets={"A": "1"}, fetched_at=time.time())
        e2 = CachedFetch(secrets={"B": "2"}, fetched_at=time.time())

        cache.write("first", e1, home_path=home)
        cache.write("second", e2, home_path=home)

        # Both should be readable
        assert cache.read("first", ttl_seconds=300, home_path=home).secrets == {"A": "1"}
        assert cache.read("second", ttl_seconds=300, home_path=home).secrets == {"B": "2"}


# ---------------------------------------------------------------------------
# Rate-limit cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_disabled_by_default(self, cache):
        assert cache.is_cooldown_active("k") is False
        cache.record_cooldown("k")
        assert cache.is_cooldown_active("k") is False  # still disabled

    def test_cooldown_blocks_reads(self, tmp_path):
        home = tmp_path / "hermes"
        home.mkdir()
        c = TwoLayerCache[str](
            basename="cooldown.json",
            cooldown_enabled=True,
            cooldown_seconds=3600,
        )
        c.record_cooldown("key123")
        assert c.is_cooldown_active("key123") is True

        # Writes still work, reads are blocked
        entry = CachedFetch(secrets={"K": "V"}, fetched_at=time.time())
        c.write("key123", entry, home_path=home)
        assert c.read("key123", ttl_seconds=300, home_path=home) is None

        c.clear(home_path=home)

    def test_cooldown_remaining(self):
        c = TwoLayerCache[str](
            basename="cd.json", cooldown_enabled=True, cooldown_seconds=10
        )
        c.record_cooldown("k")
        remaining = c.cooldown_remaining("k")
        assert 0 < remaining <= 10


# ---------------------------------------------------------------------------
# resolve_cache_home
# ---------------------------------------------------------------------------


class TestResolveCacheHome:
    def test_explicit_path(self, tmp_path):
        p = tmp_path / "custom"
        assert resolve_cache_home(p) == p

    def test_falls_back_to_env(self, tmp_path):
        os.environ["HERMES_HOME"] = str(tmp_path)
        assert resolve_cache_home() == tmp_path

    def test_falls_back_to_dot_hermes(self):
        os.environ.pop("HERMES_HOME", None)
        result = resolve_cache_home()
        assert result.name == ".hermes"


# ---------------------------------------------------------------------------
# onepassword helpers
# ---------------------------------------------------------------------------


class TestTokenFingerprint:
    def test_consistent(self):
        assert _token_fingerprint("abc") == _token_fingerprint("abc")

    def test_different_tokens(self):
        assert _token_fingerprint("abc") != _token_fingerprint("xyz")

    def test_sha256_hex_prefix(self):
        fp = _token_fingerprint("my-token")
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)


class TestCacheKey:
    def test_includes_fingerprint_and_vault(self):
        key = _cache_key("token", "My Vault")
        fp = _token_fingerprint("token")
        assert key == f"{fp}|My Vault"

    def test_includes_refs_fingerprint(self):
        refs = {"A": "op://V/A/cred"}
        refs_fp = _refs_fingerprint(refs)
        key = _cache_key("token", "V", refs_fp)
        assert refs_fp in key

    def test_different_configs_produce_different_keys(self):
        key1 = _cache_key("token", "V", _refs_fingerprint({"A": "ref1"}))
        key2 = _cache_key("token", "V", _refs_fingerprint({"B": "ref2"}))
        assert key1 != key2


class TestRefsFingerprint:
    def test_stable(self):
        refs = {"B": "op://V/B/cred", "A": "op://V/A/cred"}
        assert _refs_fingerprint(refs) == _refs_fingerprint(refs)

    def test_empty(self):
        assert _refs_fingerprint({}) == ""


class TestSanitiseEnvName:
    def test_uppercase_and_underscores(self):
        assert _sanitise_env_name("my api key") == "MY_API_KEY"

    def test_handles_dashes(self):
        assert _sanitise_env_name("my-api-key") == "MY_API_KEY"

    def test_strips_non_alphanumeric(self):
        assert _sanitise_env_name("API Key (prod)!") == "API_KEY_PROD"

    def test_empty_and_none(self):
        assert _sanitise_env_name("") == ""
        assert _sanitise_env_name(None) == ""

    def test_leading_digit_rejected(self):
        assert _sanitise_env_name("1password_token") == ""


class TestJsonDumpsStable:
    def test_sorted_keys(self):
        assert json_dumps_stable({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    def test_deterministic(self):
        d = {"c": 3, "b": 2, "a": 1}
        assert json_dumps_stable(d) == json_dumps_stable(d)


# ---------------------------------------------------------------------------
# apply_onepassword_secrets — start-up injection
# ---------------------------------------------------------------------------


class TestApplyOnepasswordSecrets:
    def test_disabled_returns_empty(self):
        result = apply_onepassword_secrets(enabled=False)
        assert result.ok is True
        assert result.secrets == {}
        assert result.applied == []

    def test_missing_token(self):
        os.environ.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
        result = apply_onepassword_secrets(
            enabled=True,
            vault="TestVault",
        )
        assert result.ok is False
        assert "not set" in result.error

    def test_no_vault_and_no_refs(self):
        os.environ["OP_SERVICE_ACCOUNT_TOKEN"] = "fake-token"
        result = apply_onepassword_secrets(
            enabled=True,
            auto_discover=False,
            env_refs={},
        )
        os.environ.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
        assert result.ok is False
        assert "nothing to fetch" in result.error

    def test_sdk_not_installed(self):
        """When the SDK is absent, returns error, never raises."""
        os.environ["OP_SERVICE_ACCOUNT_TOKEN"] = "fake-token"
        with patch(
            "agent.secret_sources.onepassword._sdk_available",
            return_value=False,
        ):
            result = apply_onepassword_secrets(
                enabled=True,
                vault="TestVault",
                auto_discover=True,
            )
        os.environ.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
        assert result.ok is False
        assert "not installed" in result.error

    def test_cache_hit(self, tmp_path):
        """Cache hit returns without touching the SDK."""
        os.environ["OP_SERVICE_ACCOUNT_TOKEN"] = "fake-token"

        # Pre-populate the cache
        _reset_cache_for_tests(tmp_path)
        from agent.secret_sources.onepassword import _cache as _onep_cache

        from agent.secret_sources._cache import CachedFetch
        key = _cache_key("fake-token", "TestVault")
        _onep_cache.write(
            key,
            CachedFetch(secrets={"TEST_KEY": "cached-value"}, fetched_at=time.time()),
            home_path=tmp_path,
        )

        with patch(
            "agent.secret_sources.onepassword._sdk_available",
            return_value=True,
        ):
            result = apply_onepassword_secrets(
                enabled=True,
                vault="TestVault",
                auto_discover=True,
                home_path=tmp_path,
            )

        os.environ.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
        assert result.cache_hit is True
        assert result.secrets == {"TEST_KEY": "cached-value"}

    def test_override_existing(self):
        """When override_existing=True, already-set vars are overwritten."""
        os.environ["OP_SERVICE_ACCOUNT_TOKEN"] = "fake-token"
        os.environ["EXISTING_KEY"] = "old-value"

        _reset_cache_for_tests()
        from agent.secret_sources.onepassword import _cache as _onep_cache
        from agent.secret_sources._cache import CachedFetch

        key = _cache_key("fake-token", "TestVault")
        _onep_cache.write(
            key,
            CachedFetch(
                secrets={"EXISTING_KEY": "new-value"},
                fetched_at=time.time(),
            ),
        )

        with patch(
            "agent.secret_sources.onepassword._sdk_available",
            return_value=True,
        ):
            result = apply_onepassword_secrets(
                enabled=True,
                vault="TestVault",
                auto_discover=True,
                override_existing=True,
            )

        os.environ.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
        os.environ.pop("EXISTING_KEY", None)

        assert "EXISTING_KEY" in result.applied

    def test_token_env_never_overwritten(self):
        """The bootstrap token env var is always skipped."""
        os.environ["OP_SERVICE_ACCOUNT_TOKEN"] = "fake-token"

        _reset_cache_for_tests()
        from agent.secret_sources.onepassword import _cache as _onep_cache
        from agent.secret_sources._cache import CachedFetch

        key = _cache_key("fake-token", "TestVault")
        _onep_cache.write(
            key,
            CachedFetch(
                secrets={"OP_SERVICE_ACCOUNT_TOKEN": "should-not-overwrite"},
                fetched_at=time.time(),
            ),
        )

        with patch(
            "agent.secret_sources.onepassword._sdk_available",
            return_value=True,
        ):
            result = apply_onepassword_secrets(
                enabled=True,
                vault="TestVault",
                auto_discover=True,
                override_existing=True,
            )

        os.environ.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
        assert "OP_SERVICE_ACCOUNT_TOKEN" in result.skipped
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in result.applied


# ---------------------------------------------------------------------------
# Live integration tests (skipped without token)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OP_SERVICE_ACCOUNT_TOKEN"),
    reason="OP_SERVICE_ACCOUNT_TOKEN not set — skipping live 1Password tests",
)
class TestLive1Password:
    """Tests that require a real 1Password service account token.

    These are skipped by default in CI.  Run locally with:
        OP_SERVICE_ACCOUNT_TOKEN=ops_... pytest tests/test_onepassword_secrets.py -v
    """

    def test_sdk_is_available(self):
        assert _sdk_available() is True

    def test_apply_auto_discover(self, tmp_path):
        """Auto-discovery fetches secrets from a vault."""
        token = os.environ["OP_SERVICE_ACCOUNT_TOKEN"]

        # Clear caches for a fresh fetch
        _reset_cache_for_tests(tmp_path)

        result = apply_onepassword_secrets(
            enabled=True,
            token_env="OP_SERVICE_ACCOUNT_TOKEN",
            vault="Jarvis's Vault",
            auto_discover=True,
            cache_ttl_seconds=0,  # bypass cache
            home_path=tmp_path,
        )

        assert result.ok, result.error
        assert isinstance(result.secrets, dict)
        # We expect at least some secrets
        assert len(result.secrets) > 0

    def test_apply_explicit_mapping(self, tmp_path):
        """Explicit env mapping resolves op:// references."""
        _reset_cache_for_tests(tmp_path)

        # This test uses a reference that should exist in any configured vault.
        # Adjust the ref to match your test vault.
        result = apply_onepassword_secrets(
            enabled=True,
            token_env="OP_SERVICE_ACCOUNT_TOKEN",
            vault="Jarvis's Vault",
            env_refs={
                "TEST_BUILDKITE": "op://Jarvis's Vault/Buildkite Token/credential",
            },
            cache_ttl_seconds=0,
            home_path=tmp_path,
        )

        assert result.ok, result.error
        assert "TEST_BUILDKITE" in result.secrets
        assert result.secrets["TEST_BUILDKITE"]

    def test_both_modes(self, tmp_path):
        """Explicit + auto-discover; explicit wins on collisions."""
        _reset_cache_for_tests(tmp_path)

        result = apply_onepassword_secrets(
            enabled=True,
            token_env="OP_SERVICE_ACCOUNT_TOKEN",
            vault="Jarvis's Vault",
            auto_discover=True,
            env_refs={
                "EXPLICIT_KEY": "op://Jarvis's Vault/Buildkite Token/credential",
            },
            cache_ttl_seconds=0,
            home_path=tmp_path,
        )

        assert result.ok, result.error
        assert "EXPLICIT_KEY" in result.secrets
        assert len(result.secrets) > 1  # auto-discover found others too
