"""No-send regressions for physical metadata and local probe routes."""

from __future__ import annotations

import pytest


def _restricted_profile(tmp_path, deny: str):
    profile = tmp_path / "restricted"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n" + deny, encoding="utf-8",
    )
    return profile



def test_anthropic_metadata_denial_sends_no_headers_or_request(tmp_path, monkeypatch):
    """A denied Anthropic catalog lookup cannot leak its API-key header."""
    from agent import model_metadata as metadata
    from hermes_cli.routing_policy import RoutingPolicyError

    profile = _restricted_profile(tmp_path, "    providers: [anthropic]\n")
    sent = []
    monkeypatch.setattr(metadata, "_ensure_requests", lambda: None)
    monkeypatch.setattr(metadata, "requests", type("Requests", (), {"get": lambda *a, **k: sent.append((a, k))})())
    with pytest.raises(RoutingPolicyError):
        metadata._query_anthropic_context_length("claude-test", "https://api.anthropic.com", "secret", profile_home=profile)
    assert sent == []


def test_codex_context_catalog_denial_sends_no_authorization_header(tmp_path, monkeypatch):
    """A denied Codex context catalog must not open requests with its bearer token."""
    from agent import model_metadata as metadata
    from hermes_cli.routing_policy import RoutingPolicyError

    profile = _restricted_profile(tmp_path, "    providers: [openai-codex]\n")
    sent = []
    monkeypatch.setattr(metadata, "_ensure_requests", lambda: None)
    monkeypatch.setattr(metadata, "requests", type("Requests", (), {"get": lambda *a, **k: sent.append((a, k))})())
    with pytest.raises(RoutingPolicyError):
        metadata._fetch_codex_oauth_context_lengths_with_source("secret", profile_home=profile)
    assert sent == []
