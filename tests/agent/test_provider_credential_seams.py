"""Generic credential seams on ``ProviderProfile``: a profile that seeds, removes and re-mints its own
credentials is honoured by core with no provider name in core, and a plain profile keeps the generic path."""

from __future__ import annotations

from types import SimpleNamespace

import providers
from providers.base import ProviderProfile
from utils import base_url_host_matches


class _OwnCreds(ProviderProfile):
    def seed_pool_credentials(self, *, suppressed_sources, explicitly_configured):
        return [] if "env:ACME_TOKEN" in suppressed_sources else [
            {"source": "env:ACME_TOKEN", "access_token": "minted", "base_url": None, "label": "ACME_TOKEN"}]

    def remove_credential_source(self, source):
        return (("env:ACME_TOKEN", "acme_cli"), ("both gone",)) if source.startswith("env:") else None

    def refresh_runtime_credentials(self, *, reason):
        return ("fresh", None) if reason == "unauthorized" else None

    def request_rejection_hint(self, *, status_code, base_url):
        return ("capped",) if status_code == 413 and "acme.example" in base_url else ()


_ACME = _OwnCreds(name="acme", base_url="https://api.acme.example/v1", hostnames=("acme.example",))
_PLAIN = ProviderProfile(name="plain", base_url="https://api.plain.example/v1")


def _registry(monkeypatch):
    profiles = {"acme": _ACME, "plain": _PLAIN}
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profiles.get(name))
    monkeypatch.setattr(providers, "list_providers", lambda: list(profiles.values()))
    monkeypatch.setattr(providers, "profile_for_host", lambda url, metadata=False: next(
        (p for p in profiles.values() if any(base_url_host_matches(url, h) for h in p.hostnames)), None))


def test_profile_seeded_rows_land_in_the_pool_and_skip_the_generic_env_loop(monkeypatch):
    from agent import credential_pool as cp

    _registry(monkeypatch)
    monkeypatch.setattr(cp, "_is_source_suppressed_fn", lambda: lambda provider, source: False)
    entries: list = []
    cp._seed_from_singletons("acme", entries)
    assert [(e.source, e.access_token) for e in entries] == [("env:ACME_TOKEN", "minted")]
    assert cp._profile_seeds_pool("acme") is True
    assert cp._profile_seeds_pool("plain") is False


def test_profile_claimed_removal_suppresses_every_variant(monkeypatch):
    from hermes_cli import auth
    from agent.credential_sources import find_removal_step

    _registry(monkeypatch)
    suppressed: list = []
    monkeypatch.setattr(auth, "suppress_credential_source", lambda p, s: suppressed.append((p, s)))
    step = find_removal_step("acme", "env:ACME_TOKEN")
    result = step.remove_fn("acme", SimpleNamespace(source="env:ACME_TOKEN"))
    assert suppressed == [("acme", "env:ACME_TOKEN"), ("acme", "acme_cli")]
    assert result.hints == ["both gone"]
    # An unclaimed source keeps the generic registry step.
    assert find_removal_step("plain", "env:PLAIN_KEY").source_id == "env:"


def test_runtime_refresh_owner_is_found_by_name_or_by_host_of_a_custom_route(monkeypatch):
    from agent.conversation_loop import _credential_refresh_profile

    _registry(monkeypatch)
    assert _credential_refresh_profile(SimpleNamespace(provider="acme", base_url="")) is _ACME
    assert _credential_refresh_profile(SimpleNamespace(provider="custom", base_url="https://api.acme.example/v1")) is _ACME
    # A profile that does not re-mint credentials never owns a refresh.
    assert _credential_refresh_profile(SimpleNamespace(provider="plain", base_url="https://api.plain.example")) is None


def test_endpoint_rejection_hint_reaches_any_route(monkeypatch):
    from agent.turn_overflow import _request_rejection_hint

    _registry(monkeypatch)
    assert _request_rejection_hint(413, "https://api.acme.example/v1") == ("capped",)
    assert _request_rejection_hint(400, "https://api.acme.example/v1") == ()
