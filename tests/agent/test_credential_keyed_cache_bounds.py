"""Regression tests: a process cache keyed by a credential FINGERPRINT must not keep one entry per
credential the process ever presents.

A TTL bounds REUSE of an entry, never its LIFETIME: a key that is never presented again is kept —
with whatever it references — until the process exits. Codex access tokens, provider api keys and
MoA profile/provider/model combinations all reach that in normal operation (a token refresh, a key
rotation, a profile leaving the multiplex set), so each of these caches grew by one entry per
rotation — sometimes keeping the credential itself.

Each test asserts the same three-part contract, and no test reads the cap constant: the run must
stay bounded, the retired credential's entry must be gone, and the credential in use must still be
served. The last two together cannot be satisfied by simply clearing the cache.

Fixes #130241 (endpoint model metadata), #130243 (MoA slot runtime), #130246 (Codex OAuth catalogue
context windows).
"""

import base64
import json

import pytest

import hermes_constants
from hermes_constants import hermes_home_key

# Comfortably past every cap, so "kept one entry per rotation" and "bounded" cannot both be true.
_ROTATIONS = 400


@pytest.fixture(autouse=True)
def _clear_caches():
    from agent import model_metadata as mm
    from agent import moa_loop as moa

    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    mm._codex_oauth_context_cache.clear()
    mm._codex_oauth_max_context_cache.clear()
    moa._runtime_cache.clear()
    yield
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    mm._codex_oauth_context_cache.clear()
    mm._codex_oauth_max_context_cache.clear()
    moa._runtime_cache.clear()


def _unsigned_jwt(marker: str) -> str:
    """A decodable JWT, so the Codex probe/header helpers accept the token. The claim carries the
    marker, so each rotation is a genuinely distinct token string (and a distinct cache key)."""
    def part(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'sub': marker})}.sig"


def _stub_codex_catalog(monkeypatch, mm):
    monkeypatch.setattr(mm, "fetch_codex_catalog_entries", lambda get, base_url="": ([
        {"slug": "gpt-x", "context_window": 400_000, "max_context_window": 900_000},
    ], 200))


# ── agent/model_metadata.py: endpoint model metadata (#130241) ──────────

def test_endpoint_metadata_cache_is_bounded_across_rotated_credentials():
    """A rotating provider key mints a new (url, fingerprint) key every time. The memo must not
    grow with the rotation count, must release the retired key, and must keep serving the current
    one — the two value dicts included, or a stamp left behind still pins its key."""
    from agent import model_metadata as mm

    first_key = last_key = mm._endpoint_memo_key("https://gw.example.com/v1", "cred-rotated-0")
    for rotation in range(_ROTATIONS):
        last_key = mm._endpoint_memo_key("https://gw.example.com/v1", f"cred-rotated-{rotation}")
        mm._remember_endpoint_models(last_key, {"m": {"context_length": 1024}})

    assert len(mm._endpoint_model_metadata_cache) < _ROTATIONS, "grew one entry per rotation"
    assert len(mm._endpoint_model_metadata_cache) == len(mm._endpoint_model_metadata_cache_time)
    assert first_key not in mm._endpoint_model_metadata_cache, "retired credential is still cached"
    assert last_key in mm._endpoint_model_metadata_cache, "the in-use credential must stay cached"
    assert mm._endpoint_model_metadata_cache[last_key] == {"m": {"context_length": 1024}}


def test_endpoint_metadata_cache_releases_entries_past_their_ttl():
    """An entry nobody reads again is past the TTL and must be released on the next write, not
    pinned until process exit."""
    from agent import model_metadata as mm

    retired_key = mm._endpoint_memo_key("https://gw.example.com/v1", "cred-retired")
    mm._remember_endpoint_models(retired_key, {"m": {"context_length": 1024}})
    live_key = mm._endpoint_memo_key("https://gw.example.com/v1", "cred-live")

    # Age the retired entry past the TTL, then write an unrelated one.
    aged = mm._endpoint_model_metadata_cache_time[retired_key] - mm._ENDPOINT_MODEL_CACHE_TTL - 1
    mm._endpoint_model_metadata_cache_time[retired_key] = aged
    mm._remember_endpoint_models(live_key, {"m": {"context_length": 2048}})

    assert retired_key not in mm._endpoint_model_metadata_cache
    assert retired_key not in mm._endpoint_model_metadata_cache_time
    assert live_key in mm._endpoint_model_metadata_cache


# ── agent/model_metadata.py: Codex OAuth catalogue (#130246) ────────────

def test_codex_oauth_context_cache_is_bounded_across_token_refreshes(monkeypatch):
    """A Codex OAuth refresh mints a new fingerprint and refreshed tokens never come back. Both the
    context map and its max-context companion — a second copy of the same catalogue — must stay
    bounded, and the live token must still be served."""
    from agent import model_metadata as mm

    _stub_codex_catalog(monkeypatch, mm)

    for rotation in range(_ROTATIONS):
        lengths, from_http = mm._fetch_codex_oauth_context_lengths_with_source(
            _unsigned_jwt(f"rotation-{rotation}"))
        assert lengths == {"gpt-x": 400_000} and from_http is True

    live_key = mm._codex_oauth_token_fingerprint(_unsigned_jwt(f"rotation-{_ROTATIONS - 1}"))
    assert len(mm._codex_oauth_context_cache) < _ROTATIONS, "grew one entry per token refresh"
    assert len(mm._codex_oauth_max_context_cache) == len(mm._codex_oauth_context_cache)
    assert live_key in mm._codex_oauth_context_cache, "the live token's catalogue must stay cached"
    assert mm._codex_oauth_max_context_cache[live_key] == {"gpt-x": 900_000}


def test_codex_oauth_context_cache_releases_entries_past_their_ttl(monkeypatch):
    """A token retired mid-window is past the TTL and must be released on the next write."""
    from agent import model_metadata as mm

    _stub_codex_catalog(monkeypatch, mm)

    retired_token = _unsigned_jwt("retired")
    mm._fetch_codex_oauth_context_lengths_with_source(retired_token)
    retired_key = mm._codex_oauth_token_fingerprint(retired_token)
    assert retired_key in mm._codex_oauth_context_cache

    lengths, stored_at = mm._codex_oauth_context_cache[retired_key]
    mm._codex_oauth_context_cache[retired_key] = (
        lengths, stored_at - mm._CODEX_OAUTH_CONTEXT_CACHE_TTL - 1,
    )
    mm._fetch_codex_oauth_context_lengths_with_source(_unsigned_jwt("fresh"))

    assert retired_key not in mm._codex_oauth_context_cache
    assert retired_key not in mm._codex_oauth_max_context_cache


# ── agent/moa_loop.py: slot runtime (#130243) ───────────────────────────

def test_moa_slot_runtime_cache_is_bounded_across_profiles(monkeypatch, tmp_path):
    """A multiplex gateway resolves (profile home, provider, model) tuples. Every distinct profile
    home adds a key, and each entry holds that slot's resolved api_key — so the cache must stay
    bounded instead of retaining credentials for profiles that are long gone."""
    import agent.moa_loop as moa
    import hermes_cli.runtime_provider as rt_mod

    monkeypatch.setattr(rt_mod, "resolve_runtime_provider", lambda **kw: {
        "base_url": "https://x", "api_mode": None,
        "api_key": f"key-{hermes_constants.get_hermes_home().name}",
    })

    for index in range(_ROTATIONS):
        home = tmp_path / f"p{index}"
        home.mkdir()
        token = hermes_constants.set_hermes_home_override(str(home))
        try:
            moa._slot_runtime({"provider": "openai", "model": "gpt-5"})
        finally:
            hermes_constants.reset_hermes_home_override(token)

    retained = {entry[1].get("api_key") for entry in moa._runtime_cache.values()}
    assert len(moa._runtime_cache) < _ROTATIONS, "grew one entry per profile"
    assert "key-p0" not in retained, "the first profile's resolved api_key is still retained"
    assert f"key-p{_ROTATIONS - 1}" in retained, "the in-use profile's runtime must stay cached"


def test_moa_slot_runtime_cache_releases_entries_past_their_ttl(monkeypatch, tmp_path):
    """A retired (profile, provider, model) entry is past the TTL and must be released on the next
    write rather than holding its api_key until process exit."""
    import agent.moa_loop as moa
    import hermes_cli.runtime_provider as rt_mod

    monkeypatch.setattr(rt_mod, "resolve_runtime_provider", lambda **kw: {
        "base_url": "https://x", "api_key": "key-retired", "api_mode": None,
    })
    slot = {"provider": "openai", "model": "gpt-5"}
    moa._slot_runtime(slot)

    retired_key = (hermes_home_key(), "openai", "gpt-5")
    assert retired_key in moa._runtime_cache
    stamped_at, cached = moa._runtime_cache[retired_key]
    moa._runtime_cache[retired_key] = (stamped_at - moa._RUNTIME_CACHE_TTL_SECONDS - 1, cached)

    fresh_home = tmp_path / "fresh"
    fresh_home.mkdir()
    monkeypatch.setattr(rt_mod, "resolve_runtime_provider", lambda **kw: {
        "base_url": "https://x", "api_key": "key-fresh", "api_mode": None,
    })
    token = hermes_constants.set_hermes_home_override(str(fresh_home))
    try:
        moa._slot_runtime(slot)
        fresh_key = (hermes_home_key(), "openai", "gpt-5")
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert retired_key not in moa._runtime_cache
    assert fresh_key in moa._runtime_cache
    assert moa._runtime_cache[fresh_key][1]["api_key"] == "key-fresh"
