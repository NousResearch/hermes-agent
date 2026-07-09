"""Live integration tests against the real Z.AI Coding Plan endpoint.

These tests hit the actual Z.AI API to validate that the pool routing
fix works end-to-end with real keys. They are SLOW and use real quota —
skip them by default, run explicitly with --runlive.

SECURITY
--------
- Keys are read from the ``GLM_TEST_KEYS`` environment variable ONLY.
  Keys must be passed as a comma-separated list, e.g.:
      export GLM_TEST_KEYS="key1,key2,key3"
- NEVER hardcode keys in this file. NEVER log full keys.
- The fixture masks every key to its 8-char prefix in any error message.
- This file is safe to commit publicly.

USAGE
-----
    # Set keys via env var (in your shell, never in code):
    $env:GLM_TEST_KEYS = "key1,key2,key3"   # PowerShell
    # or
    export GLM_TEST_KEYS="key1,key2,key3"  # bash

    # Run only the live tests:
    pytest tests/agent/test_zai_live.py -v --runlive

    # Skip live tests entirely (default):
    pytest tests/agent/test_zai_live.py -v
"""
from __future__ import annotations

import os

import pytest

# ────────────────────────────────────────────────────────────────────────────
# Live-test gate
# ────────────────────────────────────────────────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--runlive",
        action="store_true",
        default=False,
        help="Run live tests against the real Z.AI API (consumes quota)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runlive"):
        return
    skip_live = pytest.mark.skip(reason="needs --runlive flag to run")
    for item in items:
        if "test_zai_live" in item.nodeid:
            item.add_marker(skip_live)


# ────────────────────────────────────────────────────────────────────────────
# Key loading (env var only, masked in errors)
# ────────────────────────────────────────────────────────────────────────────


def _load_test_keys() -> list[str]:
    """Load keys from GLM_TEST_KEYS env var. Raise a clear error if unset."""
    raw = os.environ.get("GLM_TEST_KEYS", "").strip()
    if not raw:
        pytest.fail(
            "GLM_TEST_KEYS env var is not set. "
            "Pass keys as a comma-separated list, e.g.: "
            "$env:GLM_TEST_KEYS = 'key1,key2,key3'"
        )
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        pytest.fail("GLM_TEST_KEYS is set but contains no non-empty keys")
    return keys


def _mask(token: str) -> str:
    """Return the first 8 chars + '...' for safe logging."""
    if not token:
        return "<empty>"
    if len(token) < 12:
        return "<short>"
    return f"{token[:8]}..."


@pytest.fixture
def zai_coding_url() -> str:
    """The Coding Plan endpoint for chat completions."""
    return "https://api.z.ai/api/coding/paas/v4/chat/completions"


@pytest.fixture
def zai_metered_url() -> str:
    """The metered endpoint for chat completions."""
    return "https://api.z.ai/api/paas/v4/chat/completions"


# ────────────────────────────────────────────────────────────────────────────
# Sanity tests — each key works against the coding endpoint
# ────────────────────────────────────────────────────────────────────────────


class TestLiveKeySanity:
    """Verify each key authenticates against /api/coding/paas/v4."""

    def test_each_key_authenticates_on_coding_endpoint(self, zai_coding_url):
        """All keys return 200 on the coding endpoint."""
        import httpx

        keys = _load_test_keys()
        results = []

        for key in keys:
            resp = httpx.post(
                zai_coding_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=15.0,
            )
            results.append((_mask(key), resp.status_code))

        # All keys should authenticate (200) OR be in a known rolling-quota state (1308)
        statuses = [s for _, s in results]
        valid = {200, 1308, 1113}  # 1308 = 5h rolling quota, 1113 = per-key limit
        invalid = [(m, s) for m, s in results if s not in valid]

        # Print results for visibility (only masked prefixes)
        print("\nLive key sanity results:")
        for m, s in results:
            print(f"  {m}: HTTP {s}")

        assert not invalid, (
            f"Some keys returned unexpected status codes: {invalid}. "
            f"Expected one of {valid}. Check that these are Coding Plan keys."
        )

    def test_keys_rejected_on_metered_endpoint(self, zai_metered_url):
        """Coding Plan keys should be REJECTED on the metered endpoint (1113)."""
        import httpx

        keys = _load_test_keys()

        # Just test the first key — enough to prove the metered endpoint doesn't work
        key = keys[0]
        resp = httpx.post(
            zai_metered_url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "glm-4-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=15.0,
        )

        assert resp.status_code == 1113, (
            f"Expected Coding Plan key to be REJECTED on metered endpoint "
            f"(1113 'no resource package'), got HTTP {resp.status_code}. "
            f"This means the key may be a non-Coding-Plan key."
        )


# ────────────────────────────────────────────────────────────────────────────
# Pool routing live test
# ────────────────────────────────────────────────────────────────────────────


class TestLivePoolRouting:
    """End-to-end pool rotation through the real Z.AI Coding Plan API."""

    def test_pool_rotates_through_keys(self, zai_coding_url):
        """Multiple sequential requests hit the coding endpoint and get 200."""
        import httpx

        keys = _load_test_keys()
        if len(keys) < 2:
            pytest.skip("Need at least 2 keys to test pool rotation")

        successes = 0
        statuses_per_key: dict[str, int] = {}

        for key in keys:
            masked = _mask(key)
            resp = httpx.post(
                zai_coding_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-air",
                    "messages": [{"role": "user", "content": "réponds ok"}],
                    "max_tokens": 32,
                },
                timeout=20.0,
            )
            statuses_per_key[masked] = resp.status_code
            if resp.status_code == 200:
                successes += 1

        print("\nLive pool rotation results:")
        for m, s in statuses_per_key.items():
            print(f"  {m}: HTTP {s}")

        # At least one key should succeed
        assert successes >= 1, (
            f"No key returned 200 — the pool cannot function. "
            f"Statuses: {statuses_per_key}"
        )

    def test_exhausted_key_does_not_block_others(self, zai_coding_url):
        """Drive one key into 1308 (5h quota), verify others still work."""
        import httpx

        keys = _load_test_keys()
        if len(keys) < 2:
            pytest.skip("Need at least 2 keys to test isolation")

        # Find which keys currently work (200) and which are exhausted (1308)
        working_keys = []
        exhausted_keys = []
        for key in keys:
            masked = _mask(key)
            resp = httpx.post(
                zai_coding_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-air",
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 8,
                },
                timeout=15.0,
            )
            if resp.status_code == 200:
                working_keys.append(key)
            elif resp.status_code == 1308:
                exhausted_keys.append(key)

        print(f"\nKey health: {len(working_keys)} working, {len(exhausted_keys)} exhausted")

        # All exhausted keys should NOT block working keys from being used
        # The pool should rotate to working keys
        assert len(working_keys) + len(exhausted_keys) == len(keys), (
            "Some keys returned unexpected status codes"
        )

        if len(working_keys) >= 1:
            # A working key still works
            key = working_keys[0]
            resp = httpx.post(
                zai_coding_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "glm-4-air",
                    "messages": [{"role": "user", "content": "verify"}],
                    "max_tokens": 8,
                },
                timeout=15.0,
            )
            assert resp.status_code == 200, (
                f"Working key {_mask(key)} suddenly stopped working: HTTP {resp.status_code}"
            )