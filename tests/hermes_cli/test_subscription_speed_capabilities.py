"""Subscription menus must reflect speed modes the request path can apply."""

from hermes_cli.inventory import _apply_capabilities
from hermes_cli.models import resolve_fast_mode_overrides


def test_astra_has_distinct_speed_tiers_and_daybreak_has_none():
    rows = [{"slug": "openai-codex", "models": ["gpt-6-astra-900k", "gpt-daybreak-blue-latest-900k"]}]
    _apply_capabilities(rows)
    astra = rows[0]["capabilities"]["gpt-6-astra-900k"]
    daybreak = rows[0]["capabilities"]["gpt-daybreak-blue-latest-900k"]
    assert astra["fast"] and astra["ultrafast"]
    assert not daybreak["fast"] and not daybreak.get("ultrafast", False)
    assert resolve_fast_mode_overrides("gpt-6-astra-900k", provider="openai-codex", tier="ultrafast") == {"service_tier": "ultrafast"}
    assert resolve_fast_mode_overrides("gpt-daybreak-blue-latest-900k", provider="openai-codex") is None


def test_proxy_rows_do_not_offer_first_party_speed_parameters():
    rows = [{"slug": "openrouter", "models": ["openai/gpt-6-astra"]}]
    _apply_capabilities(rows)
    caps = rows[0]["capabilities"]["openai/gpt-6-astra"]
    assert not caps["fast"] and not caps.get("ultrafast", False)


def test_daybreak_options_follow_account_catalog_and_context_alias(monkeypatch):
    from agent import model_metadata
    from hermes_cli import auth_codex
    monkeypatch.setattr(auth_codex, "resolve_codex_runtime_credentials", lambda **_: {"api_key": "account-token", "base_url": "https://chatgpt.com/backend-api/codex"})
    monkeypatch.setattr(model_metadata, "codex_access_programs", lambda *args: {
        "gpt-6-astra": ["standard"], "gpt-6.1-sol": ["standard"],
        "gpt-6-sol": ["standard", "daybreak_blue"], "gpt-daybreak-blue-latest": ["daybreak_blue"]})
    row = {"slug": "openai-codex", "models": ["gpt-6-astra", "gpt-6.1-sol-900k", "gpt-6-sol-900k", "gpt-daybreak-blue-latest"]}
    _apply_capabilities([row])
    assert not row["capabilities"]["gpt-6-astra"]["daybreak"]
    assert not row["capabilities"]["gpt-6.1-sol-900k"]["daybreak"]
    assert row["capabilities"]["gpt-6-sol-900k"]["daybreak"]
    assert row["capabilities"]["gpt-daybreak-blue-latest"]["daybreak"]

    # The Codex app-server runtime keeps Codex's own model settings (#75186): no Daybreak switch.
    native = {"slug": "openai-codex", "models": ["gpt-6-sol-900k"]}
    _apply_capabilities([native], metadata_config={"model": {"openai_runtime": "codex_app_server"}})
    assert not native["capabilities"]["gpt-6-sol-900k"]["daybreak"]
