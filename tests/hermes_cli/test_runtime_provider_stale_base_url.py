"""Stale cross-provider ``model.base_url`` admission (#135676).

A provider switch can leave the previous provider's canonical URL in ``model.base_url`` while
``model.provider`` moves on; honoring it sends every request to the wrong route (openai-codex on
``https://api.anthropic.com`` 404s behind an outage-looking retry message). Judged by hostname
for every registered provider: only an override sitting on a different-family vendor's canonical
host (or a subdomain of it) is dropped — sibling region endpoints (minimax → minimax-cn),
same-family variants and third-party proxies are deliberate overrides and stay (#6039)."""

from types import SimpleNamespace

from hermes_cli import runtime_provider as rp


def _pool_with(url):
    entry = SimpleNamespace(access_token=f"pool-token", source="manual", base_url=url)
    return SimpleNamespace(has_credentials=lambda: True, select=lambda **_k: entry)


def test_codex_pool_ignores_stale_foreign_canonical_base_url(monkeypatch, caplog):
    """model.base_url that is another provider's canonical endpoint is dropped at the
    config-admission gate with a warning instead of 404ing every request on the mismatched route."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openai-codex")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _pool_with("https://chatgpt.com/backend-api/codex"))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "openai-codex", "default": "gpt-5.3-codex", "base_url": "https://api.anthropic.com"})

    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        resolved = rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert "another provider's canonical endpoint" in caplog.text


def test_key_provider_keeps_cross_region_model_base_url(monkeypatch, caplog):
    """A key-based provider pointing at a sibling region's canonical URL (minimax → minimax-cn)
    is a deliberate override, not provider-switch residue (#6039)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _pool_with("https://api.minimax.io/anthropic"))
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", f"test-minimax-key")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "minimax", "base_url": "https://api.minimaxi.com/anthropic"})

    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["base_url"] == "https://api.minimaxi.com/anthropic"
    assert "another provider's canonical endpoint" not in caplog.text


def test_key_provider_drops_base_url_on_foreign_vendor_host(monkeypatch, caplog):
    """A key-based provider pointing at another vendor's canonical host (xai on Google AI Studio)
    cannot be a deliberate proxy — the host belongs to the other vendor — so it is dropped like
    the OAuth case instead of 404ing every request (#135676)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "xai")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _pool_with("https://api.x.ai/v1"))
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    monkeypatch.setenv("XAI_API_KEY", f"test-xai-key")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "xai", "default": "grok-4", "base_url": "https://generativelanguage.googleapis.com/v1beta"})

    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        resolved = rp.resolve_runtime_provider(requested="xai")

    assert resolved["base_url"] == "https://api.x.ai/v1"
    assert "another provider's canonical endpoint" in caplog.text


def test_key_provider_keeps_proxy_base_url(monkeypatch, caplog):
    """A base_url whose host is not any registered vendor's endpoint stays honored for key-based
    providers — Ramp Router / relay deployments are deliberate user choices (#6039)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "xai")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _pool_with("https://api.x.ai/v1"))
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    monkeypatch.setenv("XAI_API_KEY", f"test-xai-key")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "xai", "default": "grok-4", "base_url": "https://my-proxy.example/v1"})

    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        resolved = rp.resolve_runtime_provider(requested="xai")

    assert resolved["base_url"] == "https://my-proxy.example/v1"
    assert "another provider's canonical endpoint" not in caplog.text


def test_same_family_variant_base_url_kept():
    """A base_url pointing at a same-family sibling variant's canonical URL (opencode-zen on the
    opencode-go route — the resolve layer then normalizes the route itself) is a routing choice,
    not provider-switch residue, so the admission gate lets it through."""
    assert not rp._stale_cross_provider_config_base_url("opencode-zen", "https://opencode.ai/zen/go/v1")
    assert not rp._stale_cross_provider_config_base_url("minimax-oauth", "https://api.minimax.io/anthropic")
    # A different-family vendor host on the same URL shape is still stale residue.
    assert rp._stale_cross_provider_config_base_url("xai", "https://opencode.ai/zen/go/v1")
