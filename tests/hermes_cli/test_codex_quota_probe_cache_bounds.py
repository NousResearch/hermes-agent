"""Regression test: the Codex quota-probe throttle must not keep an entry per token rotation.

The throttle cache is keyed by a SHA-256 prefix of the Codex access token. Access tokens rotate on
every OAuth refresh, so each rotation minted a key that was never presented again — and the probe
INTERVAL, which only gates a read, never removed it. A long-lived process accumulated one entry per
token it ever held (#130249).

The contract: a long run of rotating tokens stays bounded, a token retired past the interval is
released, and the throttle still works for the token in use.

Fixes #130249.
"""

import base64
import json

import pytest

import hermes_cli.auth as auth_mod
import hermes_cli.auth_codex as auth_codex

_ROTATIONS = 400


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    auth_mod._codex_quota_probe_cache.clear()
    yield
    auth_mod._codex_quota_probe_cache.clear()


def _jwt(marker: str) -> str:
    """A decodable JWT — the probe refuses non-JWT tokens — carrying a per-rotation claim, so every
    marker is a distinct token string and therefore a distinct throttle key."""
    def part(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'sub': marker})}.sig"


@pytest.fixture
def usage_endpoint(monkeypatch):
    """Answer the usage probe locally and record every request, so the throttle's effect on the
    wire is observable. The window reads 1% used, i.e. quota restored."""
    requests: list[str] = []

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"rate_limit": {"primary_window": {"used_percent": 1.0}}}

    class _Client:
        def get(self, url, headers=None):
            requests.append(url)
            return _Response()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kwargs: _Client())
    return requests


def test_quota_probe_cache_is_bounded_across_token_rotations(usage_endpoint):
    """Each rotation presents a token the process will never see again, and each is still probed
    exactly once. The throttle must not keep one entry per token it ever held."""
    for rotation in range(_ROTATIONS):
        auth_codex._probe_codex_quota_restored(_jwt(f"rotation-{rotation}"))

    cache = auth_mod._codex_quota_probe_cache
    assert len(usage_endpoint) == _ROTATIONS, "every distinct token is probed exactly once"
    assert len(cache) < _ROTATIONS, "kept one entry per token rotation"
    live_key = auth_codex._codex_quota_probe_cache_key(_jwt(f"rotation-{_ROTATIONS - 1}"))
    assert live_key in cache, "the in-use token must stay throttled"


def test_quota_probe_cache_releases_a_token_past_its_interval(usage_endpoint):
    """A token retired mid-window is past the throttle interval and must not be retained by the
    next probe — while the token in use stays throttled."""
    retired_token, live_token = _jwt("retired"), _jwt("live")
    retired_key = auth_codex._codex_quota_probe_cache_key(retired_token)

    auth_codex._probe_codex_quota_restored(retired_token)
    auth_codex._probe_codex_quota_restored(live_token)
    assert retired_key in auth_mod._codex_quota_probe_cache
    probes_before = len(usage_endpoint)

    # Age the retired token's entry past the interval the next probe judges it by.
    stamped_at, verdict = auth_mod._codex_quota_probe_cache[retired_key]
    auth_mod._codex_quota_probe_cache[retired_key] = (stamped_at - 10_000, verdict)
    auth_codex._probe_codex_quota_restored(_jwt("fresh"))

    assert retired_key not in auth_mod._codex_quota_probe_cache
    assert len(usage_endpoint) == probes_before + 1, "only the fresh token reached the endpoint"

    # The live token is still inside its interval, so it is answered from the cache.
    assert auth_codex._probe_codex_quota_restored(live_token) is True
    assert len(usage_endpoint) == probes_before + 1, "the in-use token must stay throttled"
