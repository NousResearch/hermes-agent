"""Tests for ``agent.secret_sources.protonpass.cache``.

The two-layer cache (in-process + disk): hit/miss behaviour, the 0600 disk file
in a 0700 dir, the token-fingerprint cache key (token never persisted),
fingerprint invalidation on token change, TTL expiry, ``cache_ttl_seconds <= 0``
disabling both layers, and the transient-empty-not-cached rule.  Driven through
``fetch_protonpass_secrets`` with a mocked ``_run_pass_cli``.
"""

from __future__ import annotations

import json
import os
import time
from unittest import mock

from tests._protonpass_helpers import (  # noqa: F401
    _SIMPLE_VAULT,
    _ok,
    _patch_run,
    _reset_caches,
    _vault_runner,
    hermes_home,
    pp,
    pp_cache,
    pp_session,
)


def test_inprocess_cache_hit_avoids_subprocess(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=60, home_path=hermes_home,
    )
    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=60, home_path=hermes_home,
    )
    assert counter["n"] == 1  # second call served from cache


def test_disk_cache_is_0600_and_omits_token(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    token = "the-secret-token"
    secrets, _ = pp.fetch_protonpass_secrets(
        service_token=token, vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert secrets == {"K_PASSWORD": "v"}

    cache_path = pp_cache._disk_cache_path(hermes_home)
    assert cache_path.exists()
    mode = os.stat(cache_path).st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"

    raw = cache_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert set(payload.keys()) == {"key", "secrets", "fetched_at"}
    assert payload["secrets"] == {"K_PASSWORD": "v"}
    # Critically, the raw token must NOT appear anywhere in the file.
    assert token not in raw
    # The key embeds the token fingerprint (not the token itself).
    assert pp_session._token_fingerprint(token) in payload["key"]


def test_disk_cache_short_circuits_when_fresh(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert counter["n"] == 1

    # Simulate a fresh process: clear ONLY the in-process cache.
    pp_cache._CACHE.clear()

    secrets, _ = pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert secrets == {"K_PASSWORD": "v"}
    assert counter["n"] == 1  # served from disk, no subprocess


def test_token_change_invalidates_cache(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="token-A", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert counter["n"] == 1
    pp_cache._CACHE.clear()

    # New token → new fingerprint → different cache key → refetch.
    pp.fetch_protonpass_secrets(
        service_token="token-B", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert counter["n"] == 2


def test_disk_cache_ttl_expiry_refetches(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert counter["n"] == 1

    # Backdate the disk cache past the TTL window.
    cache_path = pp_cache._disk_cache_path(hermes_home)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["fetched_at"] = time.time() - 10_000
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    pp_cache._CACHE.clear()

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert counter["n"] == 2


def test_use_cache_false_skips_both_layers(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, use_cache=True, home_path=hermes_home,
    )
    assert counter["n"] == 1
    pp_cache._CACHE.clear()

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, use_cache=False, home_path=hermes_home,
    )
    assert counter["n"] == 2


def test_ttl_zero_disables_cache(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=0, home_path=hermes_home,
    )
    assert counter["n"] == 1
    assert not pp_cache._disk_cache_path(hermes_home).exists()

    pp_cache._CACHE.clear()
    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=0, home_path=hermes_home,
    )
    assert counter["n"] == 2


def test_disk_cache_write_wrapper_honors_disabled_ttl(hermes_home):
    cache_key = pp_cache.build_cache_key("svc", "V", {}, hermes_home)
    entry = pp_cache._CachedFetch(secrets={"K_PASSWORD": "v"}, fetched_at=time.time())

    pp_cache._write_disk_cache(cache_key, entry, 0, hermes_home)

    assert not pp_cache._disk_cache_path(hermes_home).exists()


def test_session_and_cache_dirs_are_0700(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )

    # A7: the session dir is token-fingerprinted, so the dir that was actually
    # created during the fetch is the suffixed one (token="svc").
    session_dir = pp_session._session_dir("svc")
    assert session_dir.exists()
    assert (os.stat(session_dir).st_mode & 0o777) == 0o700

    cache_dir = pp_cache._disk_cache_path(hermes_home).parent
    assert cache_dir.exists()
    assert (os.stat(cache_dir).st_mode & 0o777) == 0o700


# ---------------------------------------------------------------------------
# empty-with-warnings results are NOT cached (transient failures retry)
# ---------------------------------------------------------------------------


def test_empty_with_warnings_not_cached(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        counter["n"] += 1
        return mock.Mock(returncode=2, stdout="", stderr="rejected")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert secrets == {}
    assert warnings
    assert not pp_cache._disk_cache_path(hermes_home).exists()
    pp_cache._CACHE.clear()
    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert counter["n"] == 2


def test_l1_cache_key_includes_home_path(tmp_path):
    """N1: the in-process L1 cache key folds in ``home_path`` so a single
    long-lived process serving two Hermes profiles can't return a stale L1 entry
    across profiles."""
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    key_a = pp_cache.build_cache_key("svc", "V", {}, home_a)
    key_b = pp_cache.build_cache_key("svc", "V", {}, home_b)
    # Same token / vault / refs, different home → different L1 keys.
    assert key_a != key_b
    assert str(home_a) in key_a
    assert str(home_b) in key_b
    # But the persisted disk-key string DELIBERATELY omits the home element
    # (the file already lives under that home), so the on-disk format is stable.
    assert pp_cache._cache_key_str(key_a) == pp_cache._cache_key_str(key_b)


def test_l1_cache_distinct_homes_do_not_collide(monkeypatch, tmp_path):
    """N1 end-to-end: two fetches with the SAME token/vault but DIFFERENT
    ``home_path`` each perform their own fetch (no cross-profile L1 reuse)."""
    home_a = tmp_path / "home-a" / ".hermes"
    home_b = tmp_path / "home-b" / ".hermes"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    pp._reset_cache_for_tests()

    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    counter = {"n": 0}
    _vault_runner(monkeypatch, _SIMPLE_VAULT, counter)

    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=60, home_path=home_a,
    )
    # A different profile (home_b) must NOT be served from home_a's L1 entry.
    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=60, home_path=home_b,
    )
    assert counter["n"] == 2
    # The same profile (home_a) IS served from L1 on a repeat.
    pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=60, home_path=home_a,
    )
    assert counter["n"] == 2


def test_empty_without_warnings_is_cached(hermes_home, monkeypatch, tmp_path):
    """An intentionally empty success (no warnings) IS safe to cache."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=json.dumps({"items": []}), stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc", vault="V", binary=binary,
        cache_ttl_seconds=300, home_path=hermes_home,
    )
    assert secrets == {}
    assert warnings == []
    assert pp_cache._disk_cache_path(hermes_home).exists()
