"""C1 bootstrap-token leak fix + V8-B fail-closed tests for
``agent.secret_sources.protonpass.fetch``.

Covers the "never cache or apply the bootstrap token" invariant:

* C1 — a MODE A derived name or a MODE B ref KEY equal to a protected bootstrap
  env var is dropped before the cache write / return.
* V8-B — the canonical default token env var is protected by DEFAULT (fail
  closed) even with no ``bootstrap_env``; a custom ``bootstrap_env`` is honored
  on top of it; and a legacy disk-cache entry carrying the bootstrap key is
  dropped on read.

Plus two v9 cache-path regressions (see the section banner below): a CACHE HIT
now surfaces the accumulated bootstrap-strip warning instead of ``[]``, and a
DISK cache hit promotes a bootstrap-DROPPED copy into the in-process cache.

Split out of the former monolithic ``test_protonpass_fetch.py`` (>1000 lines);
the MODE A and MODE B sections live in sibling modules.  Shared fixtures/helpers
come from ``tests._protonpass_helpers`` (do NOT duplicate them).
"""

from __future__ import annotations

import json
import time
from unittest import mock

from tests._protonpass_helpers import (  # noqa: F401
    _ok,
    _patch_run,
    _reset_caches,
    hermes_home,
    pp,
    pp_cache,
    pp_fetch,
)


# ---------------------------------------------------------------------------
# C1 — the bootstrap-token leak fix (MODE A derived name + MODE B ref)
# ---------------------------------------------------------------------------


def test_mode_a_bootstrap_named_value_not_returned_or_cached(
    hermes_home, monkeypatch, tmp_path
):
    """C1 (the MODE A leak): a vault item whose DERIVED env name equals
    ``bootstrap_env`` must NOT appear in the returned secrets NOR be written to
    the plaintext disk cache file — even though MODE A fetched it."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    # A vault item that derives PROTON_TOKEN (title "Proton", field "Token") plus
    # a harmless second field that must survive.
    payload = json.dumps({
        "items": [
            {
                "content": {
                    "title": "Proton",
                    "content": {"Login": {"token": "pst_LEAKED", "other": "keep"}},
                }
            }
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        vault="V",
        binary=binary,
        use_cache=True,            # exercise the cache-WRITE path
        cache_ttl_seconds=300,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )
    # The bootstrap-named derived value is dropped; the sibling field survives.
    assert "PROTON_TOKEN" not in secrets
    assert secrets.get("PROTON_OTHER") == "keep"
    assert any("PROTON_TOKEN" in w for w in warnings)
    assert "pst_LEAKED" not in " ".join(warnings)

    # And it must NOT be persisted to the plaintext disk cache JSON.
    cache_path = pp_cache._disk_cache_path(hermes_home)
    assert cache_path.exists()
    raw = cache_path.read_text(encoding="utf-8")
    cached = json.loads(raw)
    assert "PROTON_TOKEN" not in cached["secrets"]
    assert "pst_LEAKED" not in raw
    assert cached["secrets"].get("PROTON_OTHER") == "keep"


def test_mode_b_bootstrap_ref_stripped_with_single_warning(
    hermes_home, monkeypatch, tmp_path
):
    """C1 (MODE B): a ref whose KEY equals ``bootstrap_env`` is stripped before
    fetching — exactly ONE warning, no ``item view`` for it, and a sibling ref
    still resolves."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    captured = []

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        captured.append(cmd)
        return mock.Mock(returncode=0, stdout="ok\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={
            "PROTON_TOKEN": "pass://S/I/token",
            "OPENAI_API_KEY": "pass://S/I/api_key",
        },
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )
    assert secrets == {"OPENAI_API_KEY": "ok"}
    # Exactly one bootstrap warning (no duplicate).
    boot_warnings = [w for w in warnings if "PROTON_TOKEN" in w]
    assert len(boot_warnings) == 1
    # No item view was ever issued for the stripped bootstrap ref.
    for cmd in captured:
        assert "token" not in cmd  # the bootstrap ref's FIELD never reached argv


def test_mode_b_prefiltered_bootstrap_ref_no_double_warning(
    hermes_home, monkeypatch, tmp_path
):
    """C1: passing ALREADY-FILTERED refs (the apply.py path) plus ``bootstrap_env``
    must NOT produce a duplicate warning — strip is a no-op on filtered refs."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout="ok\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    # Caller already removed the bootstrap ref; only OPENAI remains.
    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"OPENAI_API_KEY": "pass://S/I/api_key"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )
    assert secrets == {"OPENAI_API_KEY": "ok"}
    assert not any("PROTON_TOKEN" in w for w in warnings)


# ---------------------------------------------------------------------------
# V8-B — fail closed by DEFAULT for the canonical token env name
# ---------------------------------------------------------------------------


def test_fetch_protects_default_token_env_without_bootstrap_env(
    hermes_home, monkeypatch, tmp_path
):
    """V8-B (a): a MODE A item that DERIVES the canonical default token env var
    must be dropped from the returned secrets AND from the on-disk cache JSON
    even when the caller passes NO ``bootstrap_env`` — fail closed by default."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    # Title "Proton Pass Personal Access" + field "Token" derives
    # PROTON_PASS_PERSONAL_ACCESS_TOKEN (the canonical default), plus a harmless
    # sibling that must survive.
    payload = json.dumps({
        "items": [
            {
                "content": {
                    "title": "Proton Pass Personal Access",
                    "content": {
                        "Login": {"token": "pst_LEAKED_DEFAULT", "other": "keep"}
                    },
                }
            }
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        vault="V",
        binary=binary,
        use_cache=True,            # exercise the cache-WRITE path
        cache_ttl_seconds=300,
        home_path=hermes_home,
        # NOTE: no bootstrap_env passed — the default must still be protected.
    )
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" not in secrets
    assert secrets.get("PROTON_PASS_PERSONAL_ACCESS_OTHER") == "keep"
    assert any("PROTON_PASS_PERSONAL_ACCESS_TOKEN" in w for w in warnings)
    assert "pst_LEAKED_DEFAULT" not in " ".join(warnings)

    # And it must NOT be persisted to the plaintext disk cache JSON.
    cache_path = pp_cache._disk_cache_path(hermes_home)
    assert cache_path.exists()
    raw = cache_path.read_text(encoding="utf-8")
    cached = json.loads(raw)
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" not in cached["secrets"]
    assert "pst_LEAKED_DEFAULT" not in raw


def test_fetch_custom_bootstrap_env_still_honored(hermes_home, monkeypatch, tmp_path):
    """V8-B (b): a custom ``bootstrap_env`` is still protected ON TOP of the
    canonical default — both the custom-named MODE A derivation and the default
    are dropped, and harmless siblings survive."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    # Title "Proton" + field "token" → PROTON_TOKEN (the custom bootstrap_env);
    # a harmless sibling must survive.
    payload = json.dumps({
        "items": [
            {
                "content": {
                    "title": "Proton",
                    "content": {"Login": {"token": "pst_CUSTOM", "other": "keep"}},
                }
            }
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        vault="V",
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )
    assert "PROTON_TOKEN" not in secrets
    assert secrets.get("PROTON_OTHER") == "keep"
    assert any("PROTON_TOKEN" in w for w in warnings)
    assert "pst_CUSTOM" not in " ".join(warnings)


def test_fetch_drops_bootstrap_from_legacy_disk_cache_on_read(
    hermes_home, monkeypatch, tmp_path
):
    """V8-B (c) LEGACY-CACHE-READ regression: a disk cache entry written before
    the guard existed can carry the bootstrap key.  A fetch that HITS that cache
    entry (use_cache=True) must drop it via the ``_drop_bootstrap`` read guard,
    never hand it back to the caller."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    env_refs = {"OPENAI_API_KEY": "pass://S/I/api_key"}
    # Pre-seed the disk cache with an entry that (legacy) carries the bootstrap
    # key alongside a harmless value.  Build the SAME cache key fetch will use.
    cache_key = pp_cache.build_cache_key("svc", "", env_refs, hermes_home)
    legacy = pp_cache._CachedFetch(
        secrets={"PROTON_TOKEN": "pst_LEGACY_LEAK", "OPENAI_API_KEY": "ok"},
        fetched_at=__import__("time").time(),
    )
    pp_cache._write_disk_cache(cache_key, legacy, 300, hermes_home)

    def fake_run(cmd, env):  # pragma: no cover - the cache hit short-circuits
        raise AssertionError("fetch must serve the disk cache, not run pass-cli")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs=env_refs,
        binary=binary,
        use_cache=True,
        cache_ttl_seconds=300,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )
    # The legacy-cached bootstrap key is dropped on read; the sibling survives.
    assert "PROTON_TOKEN" not in secrets
    assert secrets.get("OPENAI_API_KEY") == "ok"


# ---------------------------------------------------------------------------
# v9 cache-path regressions — warnings on a CACHE HIT + a bootstrap-dropped
# in-process promotion from disk
# ---------------------------------------------------------------------------


def test_cache_hit_surfaces_bootstrap_strip_warning(
    hermes_home, monkeypatch, tmp_path
):
    """v9 regression (cache HIT): the in-process cache-hit return now surfaces
    the ACCUMULATED bootstrap-strip warning (``list(warnings)``) instead of an
    empty ``[]``.  A ref whose KEY equals ``bootstrap_env`` is stripped on every
    call (producing a warning), so a second fetch that HITS the cache must still
    return that warning, not drop it."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    view_calls = []

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        view_calls.append(cmd)
        return mock.Mock(returncode=0, stdout="ok\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    env_refs = {
        "PROTON_TOKEN": "pass://S/I/token",   # the stripped bootstrap ref
        "OPENAI_API_KEY": "pass://S/I/api_key",
    }
    kwargs = dict(
        service_token="svc",
        env_refs=env_refs,
        binary=binary,
        use_cache=True,
        cache_ttl_seconds=300,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )

    # First fetch: misses the cache, fetches OPENAI, primes the in-process cache.
    secrets1, warnings1 = pp.fetch_protonpass_secrets(**kwargs)
    assert secrets1 == {"OPENAI_API_KEY": "ok"}
    assert any("PROTON_TOKEN" in w for w in warnings1)
    calls_after_first = len(view_calls)
    assert calls_after_first >= 1  # OPENAI was actually fetched

    # Second fetch: HITS the in-process cache (no new item view), and the
    # bootstrap-strip warning is STILL surfaced — the v9 fix returns
    # ``list(warnings)`` here, not ``[]``.
    secrets2, warnings2 = pp.fetch_protonpass_secrets(**kwargs)
    assert secrets2 == {"OPENAI_API_KEY": "ok"}
    assert len(view_calls) == calls_after_first  # served from cache, no re-fetch
    assert warnings2 != []
    assert any("PROTON_TOKEN" in w for w in warnings2)


def test_disk_cache_hit_promotes_bootstrap_dropped_entry_into_inprocess_cache(
    hermes_home, monkeypatch, tmp_path
):
    """v9 regression (DISK hit): a legacy disk entry carrying a bootstrap-named
    key is dropped on read, and the v9 fix promotes the bootstrap-DROPPED copy
    into the in-process ``_CACHE`` (``_CachedFetch(dropped, fetched_at)``) — so a
    subsequent in-process hit can never re-surface the protected key.  Assert
    both the in-process ``_CACHE`` entry AND the returned secrets exclude it."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    env_refs = {"OPENAI_API_KEY": "pass://S/I/api_key"}
    # Seed ONLY the disk cache (the autouse fixture clears the in-process cache),
    # with a legacy entry that carries the bootstrap key alongside a sibling.
    # Build the SAME cache key fetch will use (the bootstrap ref is not among the
    # refs, so the strip is a no-op and the key matches).
    cache_key = pp_cache.build_cache_key("svc", "", env_refs, hermes_home)
    legacy = pp_cache._CachedFetch(
        secrets={"PROTON_TOKEN": "pst_LEGACY_LEAK", "OPENAI_API_KEY": "ok"},
        fetched_at=time.time(),
    )
    pp_cache._write_disk_cache(cache_key, legacy, 300, hermes_home)
    # Sanity: the in-process cache is empty, so the fetch must take the disk path.
    assert cache_key not in pp_fetch._CACHE

    def fake_run(cmd, env):  # pragma: no cover - the disk cache hit short-circuits
        raise AssertionError("fetch must serve the disk cache, not run pass-cli")

    _patch_run(monkeypatch, fake_run)

    secrets, _warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs=env_refs,
        binary=binary,
        use_cache=True,
        cache_ttl_seconds=300,
        home_path=hermes_home,
        bootstrap_env="PROTON_TOKEN",
    )

    # The in-process entry promoted from disk must NOT carry the bootstrap key.
    promoted = pp_fetch._CACHE[cache_key]
    assert "PROTON_TOKEN" not in promoted.secrets
    assert promoted.secrets.get("OPENAI_API_KEY") == "ok"
    # And the returned secrets exclude it too.
    assert "PROTON_TOKEN" not in secrets
    assert secrets.get("OPENAI_API_KEY") == "ok"
