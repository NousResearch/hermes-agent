"""Credential-pool discovery must not run a disabled provider's profile seeding."""

from __future__ import annotations

import providers
from providers.base import ProviderProfile


class _Seeding(ProviderProfile):
    calls: list = []

    def seed_pool_credentials(self, *, suppressed_sources, explicitly_configured):
        type(self).calls.append(1)
        return [{"source": "env:ACME_TOKEN", "access_token": "minted", "base_url": None, "label": "ACME"}]


def _setup(tmp_path, monkeypatch, config_yaml):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (home / "auth.json").write_text('{"version": 1, "credential_pool": {}}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _Seeding.calls = []
    profile = _Seeding(name="acme", base_url="https://api.acme.example/v1")
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile if name == "acme" else None)


def test_disabled_provider_skips_profile_seeding(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, "providers:\n  acme:\n    enabled: false\n")
    from agent.credential_pool import load_pool

    assert load_pool("acme").entries() == []
    assert _Seeding.calls == []


def test_enabled_provider_still_runs_profile_seeding(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, "providers:\n  acme:\n    enabled: true\n")
    from agent.credential_pool import load_pool

    assert [e.access_token for e in load_pool("acme").entries()] == ["minted"]
    assert _Seeding.calls == [1]
