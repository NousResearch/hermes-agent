"""Tests for Hindsight api_url normalization and fail-fast validation.

Covers the schemeless-URL failure mode (a config like
``"api_url": "host:8888"`` made every hindsight_* tool fail with a cryptic
"Failed to search memory: host:8888/v1/..." error) and the disabled-provider
guard for genuinely invalid URLs.
"""

import json
import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _diagnose_scheme_mismatch,
    _normalize_api_url,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no stale Hindsight env vars leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_MODE",
        "HINDSIGHT_BANK_ID", "HINDSIGHT_BUDGET", "HINDSIGHT_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)
    # Per-process diagnosis cache must not leak between tests that monkeypatch
    # the probe with different behaviors.
    from plugins.memory.hindsight import _scheme_diagnosis_cache
    _scheme_diagnosis_cache.clear()


def _make_provider(tmp_path, monkeypatch, config: dict) -> HindsightMemoryProvider:
    """Write *config* as the profile config.json and build an initialized provider."""
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    return provider


# ---------------------------------------------------------------------------
# _normalize_api_url unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["local_external", "local_embedded"])
def test_normalize_adds_http_for_local_modes(mode):
    assert _normalize_api_url("pve1-docker.ts.rmg7.com:8888", mode) == \
        "http://pve1-docker.ts.rmg7.com:8888"
    assert _normalize_api_url("localhost:8888", mode) == "http://localhost:8888"
    assert _normalize_api_url("127.0.0.1:8888", mode) == "http://127.0.0.1:8888"


def test_normalize_adds_https_for_cloud():
    assert _normalize_api_url("api.hindsight.vectorize.io", "cloud") == \
        "https://api.hindsight.vectorize.io"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8888",
        "https://api.hindsight.vectorize.io",
        "http://pve1-docker.ts.rmg7.com:8888/",
    ],
)
def test_normalize_keeps_explicit_scheme(url):
    assert _normalize_api_url(url, "cloud") == url


def test_normalize_strips_surrounding_whitespace():
    assert _normalize_api_url("  localhost:8888  ", "local_external") == \
        "http://localhost:8888"


@pytest.mark.parametrize("empty", [None, ""])
def test_normalize_passthrough_empty(empty):
    assert _normalize_api_url(empty, "cloud") is None


def test_normalize_whitespace_only_returns_none():
    # Whitespace-only input is "unset": None lets callers fall back to defaults.
    assert _normalize_api_url("   ", "cloud") is None


@pytest.mark.parametrize("bad", ["ftp://example.com", "gopher://example.com", "hindsight://local"])
def test_normalize_rejects_invalid_scheme(bad):
    with pytest.raises(ValueError, match="http:// or https://"):
        _normalize_api_url(bad, "cloud")


# ---------------------------------------------------------------------------
# Provider-level behavior
# ---------------------------------------------------------------------------


def test_provider_normalizes_schemeless_local_url(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "pve1-docker.ts.rmg7.com:8888",
        "apiKey": "test-key",
    })
    assert provider._api_url == "http://pve1-docker.ts.rmg7.com:8888"


def test_provider_normalizes_schemeless_cloud_url(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "cloud",
        "api_url": "api.hindsight.vectorize.io",
        "apiKey": "test-key",
    })
    assert provider._api_url == "https://api.hindsight.vectorize.io"


def test_provider_whitespace_api_url_falls_back_to_default(tmp_path, monkeypatch):
    """Whitespace-only api_url is \"unset\" — provider uses the mode default URL."""
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "   ",
        "apiKey": "test-key",
    })
    assert provider._api_url == "http://localhost:8888"


def test_provider_whitespace_api_url_cloud_falls_back_to_default(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "cloud",
        "api_url": " \t ",
        "apiKey": "test-key",
    })
    assert provider._api_url == "https://api.hindsight.vectorize.io"


def test_is_available_consistent_with_whitespace_api_url(tmp_path, monkeypatch):
    """is_available must agree with initialize(): whitespace-only -> default URL."""
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "   ",
        "apiKey": "test-key",
    })
    assert provider.is_available()
    assert provider._api_url == "http://localhost:8888"


def test_provider_disables_on_invalid_scheme(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "ftp://example.com:8888",
        "apiKey": "test-key",
    })
    assert provider._mode == "disabled"


def test_disabled_provider_tools_return_actionable_error(tmp_path, monkeypatch):
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "ftp://example.com:8888",
        "apiKey": "test-key",
    })
    err = provider.handle_tool_call("hindsight_recall", {"query": "anything"})
    assert "disabled" in err
    assert "config.json" in err


def test_is_available_false_on_invalid_scheme(tmp_path, monkeypatch):
    """A broken api_url must hide the provider before tools are registered."""
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "ftp://example.com:8888",
        "apiKey": "test-key",
    })
    assert not provider.is_available()


def test_is_available_true_on_schemeless_local_url(tmp_path, monkeypatch):
    """A schemeless local URL is auto-normalized, so the provider stays available."""
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "pve1-docker.ts.rmg7.com:8888",
        "apiKey": "test-key",
    })
    assert provider.is_available()


def test_recall_error_includes_exception_type(tmp_path, monkeypatch):
    """Tool errors surface the exception type, not just a bare message."""
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "http://localhost:9999",
        "apiKey": "test-key",
    })

    class _BoomClient:
        async def arecall(self, **kwargs):
            raise ValueError("boom")

    provider._client = _BoomClient()
    err = provider.handle_tool_call("hindsight_recall", {"query": "anything"})
    assert "ValueError: boom" in err


# ---------------------------------------------------------------------------
# Scheme-mismatch diagnosis (https:// to a plain-HTTP daemon)
# ---------------------------------------------------------------------------


def _fake_probe_http_only(url, key=None, timeout=5.0):
    """Probe that answers only on plain HTTP — the user's https-vs-http case."""
    return "0.8.6" if url.startswith("http://") else None


def test_diagnose_no_mismatch_when_configured_works(monkeypatch):
    calls = []
    def fake_probe(url, key=None, timeout=5.0):
        calls.append(url)
        return "0.8.6"
    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", fake_probe)
    assert _diagnose_scheme_mismatch("http://host:8888", "k") is None
    assert calls == ["http://host:8888"]


def test_diagnose_detects_https_to_http_server(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version", _fake_probe_http_only
    )
    hint = _diagnose_scheme_mismatch("https://pve1-docker.ts.rmg7.com:8888", "k")
    assert hint is not None
    assert "https://" in hint
    assert "http://pve1-docker.ts.rmg7.com:8888" in hint


def test_diagnose_detects_http_to_https_server(monkeypatch):
    def fake_probe(url, key=None, timeout=5.0):
        return "0.8.6" if url.startswith("https://") else None
    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", fake_probe)
    hint = _diagnose_scheme_mismatch("http://host:8443", "k")
    assert hint is not None
    assert "https://host:8443" in hint


def test_diagnose_both_schemes_fail_returns_none(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version",
        lambda url, key=None, timeout=5.0: None,
    )
    assert _diagnose_scheme_mismatch("https://host:8888", "k") is None


def test_diagnose_swapped_probe_never_receives_api_key(monkeypatch):
    """The swapped-scheme host must never receive credentials (key leak #2)."""
    seen = {}

    def fake_probe(url, key=None, timeout=5.0):
        seen[url] = key
        return "0.8.6" if url.startswith("http://") else None

    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", fake_probe)
    hint = _diagnose_scheme_mismatch("https://host:8888", "secret-key")
    assert hint is not None
    # The configured URL is the intended target — the key is safe there.
    assert seen.get("https://host:8888") == "secret-key"
    # The swapped host (http:// here, a downgrade to plaintext) gets no key.
    assert seen.get("http://host:8888") is None


def test_diagnose_cached_per_process(monkeypatch):
    """The probe runs once per URL per process — no re-probe on later calls."""
    calls = []

    def fake_probe(url, key=None, timeout=5.0):
        calls.append(url)
        return None  # both schemes fail -> hint None

    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", fake_probe)
    assert _diagnose_scheme_mismatch("https://host:8888") is None
    assert len(calls) == 2  # configured + swapped
    assert _diagnose_scheme_mismatch("https://host:8888") is None
    assert len(calls) == 2  # served from cache — no extra probes


def test_diagnose_cached_hint_reused(monkeypatch):
    """A cached positive diagnosis is returned without re-probing."""
    calls = []

    def fake_probe(url, key=None, timeout=5.0):
        calls.append(url)
        return "0.8.6" if url.startswith("http://") else None

    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", fake_probe)
    first = _diagnose_scheme_mismatch("https://host:8888", "k")
    second = _diagnose_scheme_mismatch("https://host:8888", "k")
    assert first is not None
    assert second == first
    assert len(calls) == 2  # only the first call probed both schemes


def test_diagnose_ignores_non_http_scheme(monkeypatch):
    calls = []
    def fake_probe(url, key=None, timeout=5.0):
        calls.append(url)
        return "0.8.6"
    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", fake_probe)
    assert _diagnose_scheme_mismatch("ftp://host:8888", "k") is None
    assert calls == []


def test_provider_logs_scheme_mismatch_but_stays_enabled(tmp_path, monkeypatch, caplog):
    """https-to-http daemon: log the exact fix, do NOT disable the provider."""
    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version", _fake_probe_http_only
    )
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "https://pve1-docker.ts.rmg7.com:8888",
        "apiKey": "test-key",
    })
    assert provider._mode == "local_external"  # not disabled
    assert "http://pve1-docker.ts.rmg7.com:8888" in caplog.text


def test_provider_does_not_probe_plain_http_url(tmp_path, monkeypatch):
    """http:// configs must never touch the network during init."""
    def boom_probe(url, key=None, timeout=5.0):
        raise AssertionError(f"probe called for {url}")
    monkeypatch.setattr("plugins.memory.hindsight._fetch_hindsight_api_version", boom_probe)
    provider = _make_provider(tmp_path, monkeypatch, {
        "mode": "local_external",
        "api_url": "http://localhost:9999",
        "apiKey": "test-key",
    })
    assert provider._api_url == "http://localhost:9999"
