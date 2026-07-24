"""Supermemory BASE_URL always-blocked floor."""

from plugins.memory.supermemory import _DEFAULT_BASE_URL, _resolve_base_url


def test_supermemory_blocks_metadata_base_url(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)
    assert _resolve_base_url("http://169.254.169.254/latest/meta-data/") == _DEFAULT_BASE_URL


def test_supermemory_allows_localhost_self_host(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)
    assert _resolve_base_url("http://localhost:6767") == "http://localhost:6767"


def test_supermemory_env_poisoning_falls_back(monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_BASE_URL", "http://metadata.google.internal/")
    assert _resolve_base_url("") == _DEFAULT_BASE_URL
