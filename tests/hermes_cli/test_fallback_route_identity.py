"""Configured fallback routes retain case-sensitive endpoint identity."""

from pathlib import Path

import pytest

from hermes_cli.config import atomic_config_write
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.fallback_config import get_fallback_chain


def _load_chain(tmp_path, monkeypatch, urls):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    entries = [{"provider": "custom", "model": "model", "base_url": url} for url in urls]
    atomic_config_write(home / "config.yaml", {
        "fallback_providers": entries[:1], "fallback_model": entries[1:],
    })
    return get_fallback_chain(load_user_config_effective())


@pytest.mark.parametrize("urls", [
    ("https://proxy.example/Tenant/v1", "https://proxy.example/tenant/v1"),
    ("https://proxy.example/v1?route=Fast", "https://proxy.example/v1?route=fast"),
])
def test_distinct_endpoint_paths_and_queries_survive_config_load(tmp_path, monkeypatch, urls):
    chain = _load_chain(tmp_path, monkeypatch, urls)

    assert [entry["base_url"] for entry in chain] == list(urls)


def test_equivalent_endpoint_spellings_share_one_fallback_slot(tmp_path, monkeypatch):
    urls = ("https://PROXY.example:443/Tenant/v1", "https://proxy.example/Tenant/v1/")

    chain = _load_chain(tmp_path, monkeypatch, urls)

    assert [entry["base_url"] for entry in chain] == [urls[0]]
