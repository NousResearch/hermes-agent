"""OpenViking endpoint always-blocked floor."""

import pytest

from plugins.memory.openviking import (
    _OpenVikingEndpointError,
    _normalize_openviking_url,
    _openviking_endpoint_is_always_blocked,
)


def test_openviking_blocks_metadata_endpoint():
    with pytest.raises(_OpenVikingEndpointError, match="blocked metadata address"):
        _normalize_openviking_url("http://169.254.169.254/")


def test_openviking_keeps_default_loopback():
    assert _normalize_openviking_url("http://127.0.0.1:1933") == "http://127.0.0.1:1933"


def test_openviking_blocks_ecs_metadata_hostname():
    with pytest.raises(_OpenVikingEndpointError, match="blocked metadata address"):
        _normalize_openviking_url("http://metadata.google.internal/computeMetadata/v1/")


def test_openviking_rejects_endpoint_credentials_and_query():
    with pytest.raises(_OpenVikingEndpointError, match="cannot contain user info"):
        _normalize_openviking_url("https://user:secret@example.com?api_key=secret")


def test_openviking_validates_shorthand_ipv6_port():
    assert _normalize_openviking_url("::1:1934") == "http://[::1]:1934"
    with pytest.raises(_OpenVikingEndpointError, match="Port could not be cast"):
        _normalize_openviking_url("::1:not-a-port")


def test_openviking_caches_safety_check_for_unchanged_endpoint(monkeypatch):
    import tools.url_safety as url_safety

    calls = []
    _openviking_endpoint_is_always_blocked.cache_clear()
    monkeypatch.setattr(
        url_safety,
        "is_always_blocked_url",
        lambda value: calls.append(value) or False,
    )

    assert _normalize_openviking_url("https://openviking.example.test") == (
        "https://openviking.example.test"
    )
    assert _normalize_openviking_url("https://openviking.example.test") == (
        "https://openviking.example.test"
    )
    assert calls == ["https://openviking.example.test"]
    _openviking_endpoint_is_always_blocked.cache_clear()
