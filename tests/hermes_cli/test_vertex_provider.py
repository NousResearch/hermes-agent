"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, alias canonicalization,
resolve_runtime_provider(vertex) minting an OAuth token, and the friendly
AuthError when credentials can't be resolved. No network calls.
"""

from __future__ import annotations

import pytest








def test_resolve_runtime_provider_raises_autherror_when_unresolved(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(va, "get_vertex_config", lambda: (None, None))
    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(requested="vertex")
    msg = str(exc.value)
    assert "OAuth2" in msg
    assert "not a static API key" in msg


def test_vertex_registered_in_provider_registry():
    """PROVIDER_REGISTRY (hermes_cli.auth) is what agent/auxiliary_client.py's
    resolve_provider_client() looks up before dispatching on auth_type. Without
    an entry here, the ``elif pconfig.auth_type == "vertex":`` branch there is
    unreachable dead code — every auxiliary Vertex call (vision, title
    generation, MoA reference/aggregator slots, ...) fails at the
    ``pconfig is None`` guard before ever reaching it."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get("vertex")
    assert cfg is not None
    assert cfg.auth_type == "vertex"


def test_vertex_registered_in_hermes_overlays():
    """hermes_cli.providers.get_provider("vertex") backs
    _preserve_provider_with_base_url() in agent/auxiliary_client.py, which
    decides whether a MoA slot's resolved Vertex (base_url, api_key) pair
    keeps its "vertex" provider identity or silently collapses to "custom" —
    losing the identity _refresh_provider_credentials() needs to re-mint an
    expired OAuth2 token on a 401."""
    from hermes_cli.providers import get_provider

    resolved = get_provider("vertex")
    assert resolved is not None
    assert resolved.auth_type == "vertex"


# ---------------------------------------------------------------------------
# Claude-on-Vertex: dual-path routing (Anthropic Messages vs OpenAI-compat).
# ---------------------------------------------------------------------------

def test_claude_on_vertex_routes_to_anthropic_messages(monkeypatch):
    """A Claude model on the vertex provider must route through the
    AnthropicVertex SDK path (api_mode=anthropic_messages), carrying the
    google-auth Credentials object and project/region — NOT a static token."""
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    class _Creds:
        token = "ya29.TOKEN"

    fake_creds = _Creds()
    monkeypatch.setattr(
        va, "get_vertex_anthropic_config",
        lambda *a, **k: (fake_creds, "my-proj", "us-east5"),
    )
    rt = rp.resolve_runtime_provider(
        requested="vertex", target_model="claude-sonnet-4-5@20250929",
    )
    assert rt["provider"] == "vertex"
    assert rt["api_mode"] == "anthropic_messages"
    assert rt["vertex_anthropic"] is True
    assert rt["vertex_project_id"] == "my-proj"
    assert rt["region"] == "us-east5"
    assert rt["vertex_credentials"] is fake_creds
    # regional base_url shape (Anthropic SDK appends the rawPredict path itself)
    assert rt["base_url"] == "https://us-east5-aiplatform.googleapis.com/v1"


def test_claude_on_vertex_global_region_base_url(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        va, "get_vertex_anthropic_config",
        lambda *a, **k: (object(), "my-proj", "global"),
    )
    rt = rp.resolve_runtime_provider(
        requested="vertex", target_model="claude-opus-4-1@20250805",
    )
    assert rt["base_url"] == "https://aiplatform.googleapis.com/v1"


def test_gemini_on_vertex_still_uses_openai_compat(monkeypatch):
    """Non-Claude models must keep the OpenAI-compat (chat_completions) path."""
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        va, "get_vertex_config",
        lambda: ("ya29.TOKEN", "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"),
    )
    rt = rp.resolve_runtime_provider(
        requested="vertex", target_model="gemini-2.5-flash",
    )
    assert rt["api_mode"] == "chat_completions"
    assert rt.get("vertex_anthropic") is None
    assert rt["api_key"] == "ya29.TOKEN"


def test_claude_on_vertex_raises_autherror_when_unresolved(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(va, "get_vertex_anthropic_config", lambda *a, **k: (None, None, None))
    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(requested="vertex", target_model="claude-sonnet-4-5@20250929")
    assert "Claude" in str(exc.value)


def test_build_anthropic_vertex_client_shape():
    """The AnthropicVertex client must be built with self-refreshing creds,
    max_retries=0 (hermes owns retry), and NO 1M-context beta."""
    pytest.importorskip("anthropic")
    from unittest.mock import MagicMock
    from agent.anthropic_adapter import build_anthropic_vertex_client

    creds = MagicMock()
    client = build_anthropic_vertex_client("my-proj", "us-east5", credentials=creds)
    assert type(client).__name__ == "AnthropicVertex"
    assert client.project_id == "my-proj"
    assert client.region == "us-east5"
    assert client.max_retries == 0
    beta = client._custom_headers.get("anthropic-beta", "")
    assert "context-1m" not in beta
    assert "interleaved-thinking-2025-05-14" in beta


# ── /model picker visibility (list_authenticated_providers) ─────────────────
#
# Vertex has auth_type "vertex" and env_vars=() — no API key to detect — so
# without a dedicated credential check (mirroring bedrock's aws_sdk special
# case) the picker omits the provider row entirely whenever vertex isn't the
# configured model.provider. Regression: switching model.provider to `moa`
# made "Google Vertex AI" vanish from the desktop picker despite working ADC.

def test_picker_lists_vertex_when_credentials_present(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import model_switch as ms

    monkeypatch.setattr(va, "has_vertex_credentials", lambda: True)
    rows = ms.list_authenticated_providers(current_provider="moa")
    vertex_rows = [r for r in rows if r.get("slug") == "vertex"]
    assert vertex_rows, "vertex row missing from picker despite credentials"
    models = vertex_rows[0].get("models") or []
    assert "claude-fable-5-1" in models


def test_picker_hides_vertex_without_credentials(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import model_switch as ms

    monkeypatch.setattr(va, "has_vertex_credentials", lambda: False)
    rows = ms.list_authenticated_providers(current_provider="moa")
    assert not [r for r in rows if r.get("slug") == "vertex"]


def test_vertex_explicitly_configured_via_config_section(monkeypatch):
    """A `vertex:` config section with project_id is the explicit opt-in
    signal (vertex has no API key for check 3 to find)."""
    import hermes_cli.auth as auth
    import hermes_cli.config as config

    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr(
        config, "load_config",
        lambda *a, **k: {"model": {"provider": "moa"}, "vertex": {"project_id": "my-proj"}},
    )
    monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    assert auth.is_provider_explicitly_configured("vertex") is True


def test_vertex_not_explicitly_configured_when_unset(monkeypatch):
    import hermes_cli.auth as auth
    import hermes_cli.config as config

    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr(
        config, "load_config",
        lambda *a, **k: {"model": {"provider": "moa"}},
    )
    monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    assert auth.is_provider_explicitly_configured("vertex") is False


# ── model-ID normalization (primary path) ────────────────────────────────────
#
# The vertex model space is mixed: Claude rides the AnthropicVertex SDK
# (bare publisher IDs — the SDK URL-injects publishers/anthropic/models/<id>),
# while Gemini/partner-MaaS ride the OpenAI-compat endpoint (REQUIRES the
# publisher/ prefix). Only the anthropic/ prefix may be stripped, and the
# primary path must agree with the auxiliary path's strip.

@pytest.mark.parametrize("given,expected", [
    ("anthropic/claude-fable-5", "claude-fable-5"),
    ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
    ("Anthropic/Claude-Fable-5", "Claude-Fable-5"),  # case-insensitive prefix
    ("anthropic/claude-sonnet-4-5@20250929", "claude-sonnet-4-5@20250929"),
    ("claude-fable-5", "claude-fable-5"),                       # already bare
    ("google/gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview"),
    ("moonshotai/kimi-k2-thinking-maas", "moonshotai/kimi-k2-thinking-maas"),
    ("deepseek-ai/deepseek-v3.2-maas", "deepseek-ai/deepseek-v3.2-maas"),
])
def test_vertex_normalization_strips_only_anthropic_prefix(given, expected):
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider(given, "vertex") == expected


def test_vertex_normalization_agrees_with_anthropic_detection():
    """Whatever the classifier accepts, normalization must reduce to an ID the
    AnthropicVertex SDK can serve — the primary/auxiliary disagreement bug."""
    from agent.vertex_adapter import is_anthropic_vertex_model
    from hermes_cli.model_normalize import normalize_model_for_provider

    for alias in ("anthropic/claude-fable-5", "claude-fable-5"):
        assert is_anthropic_vertex_model(alias)
        normalized = normalize_model_for_provider(alias, "vertex")
        assert normalized == "claude-fable-5"
        assert is_anthropic_vertex_model(normalized)


@pytest.mark.parametrize("short_alias", ["sonnet", "fable", "opus"])
def test_short_claude_aliases_resolve_on_vertex(short_alias):
    """Family-specific Claude aliases must resolve to a bare,
    is_anthropic_vertex_model-recognized ID on vertex.

    A missing MODEL_ALIASES entry (e.g. "fable" was absent) makes
    resolve_alias() return None, so switch_model() falls through and passes
    the literal short alias straight to the wire. Vertex's OpenAI-compatible
    /openapi endpoint then 400s with "Malformed publisher model (model:
    'fable')" because the bare short name was never routed through the
    AnthropicVertex SDK path in the first place.

    Invariant: each alias here names exactly ONE family in the curated
    vertex catalog, so resolution must be unambiguous. Listing two
    generations of the same family (e.g. opus-5 and opus-4-8) would make
    resolve_alias raise AmbiguousAliasError instead — see
    test_bare_claude_alias_is_ambiguous_on_vertex for the deliberate
    multi-family case.

    Note: "haiku" is deliberately excluded — Vertex's curated catalog
    doesn't currently list a haiku model, so that alias legitimately
    fails to resolve there.
    """
    from agent.vertex_adapter import is_anthropic_vertex_model
    from hermes_cli.model_switch import resolve_alias

    result = resolve_alias(short_alias, "vertex")
    assert result is not None, f"'{short_alias}' did not resolve on vertex — check MODEL_ALIASES"
    provider, resolved_model, _alias_name = result
    assert provider == "vertex"
    assert is_anthropic_vertex_model(resolved_model)


def test_bare_claude_alias_is_ambiguous_on_vertex():
    """The bare "claude" alias spans every Claude family in the curated
    vertex catalog, so it must raise rather than silently pick one.

    resolve_alias() deliberately refuses to guess among multiple matches:
    version-sort heuristics have repeatedly landed on the wrong model
    (dated snapshots outranking point releases, suffix tiebreaks picking
    the cheapest tier). Asserting the raise pins that contract so a future
    catalog edit can't reintroduce silent selection.
    """
    from hermes_cli.model_switch import AmbiguousAliasError, resolve_alias

    with pytest.raises(AmbiguousAliasError) as exc:
        resolve_alias("claude", "vertex")

    # Every candidate offered to the user must be a real Claude id.
    assert all("claude" in m.lower() for m in exc.value.candidates)
    assert len(exc.value.candidates) > 1
