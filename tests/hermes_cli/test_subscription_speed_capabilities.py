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
