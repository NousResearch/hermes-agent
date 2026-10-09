"""Stale cross-provider ``model.base_url`` admission (#135676).

A provider switch can leave the previous provider's canonical URL in ``model.base_url`` while
``model.provider`` moves on; honoring it sends every request to the wrong route (openai-codex on
``https://api.anthropic.com`` 404s behind an outage-looking retry message). Only subscription-
routed (OAuth) providers are judged — key-based providers keep every override (region endpoints
like minimax → minimax-cn are deliberate, #6039)."""

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
