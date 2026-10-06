from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.setup_summary import _WEB_MISSING, _web_row


def _features(*, available=False, managed=False, explicit=True, provider="exa"):
    return SimpleNamespace(web=SimpleNamespace(
        available=available,
        managed_by_nous=managed,
        explicit_configured=explicit,
        current_provider=provider,
    ))


def _provider(name, display_name=None, *, keyed=False, keyless=False):
    return SimpleNamespace(
        name=name,
        display_name=display_name or name.title(),
        is_available=lambda: keyed,
        is_keyless_available=lambda: keyless,
    )


def _resolve_row(features, search, extract):
    with patch("tools.web_tools._ensure_web_plugins_loaded"), \
         patch("agent.web_search_registry.get_active_search_provider", return_value=search), \
         patch("agent.web_search_registry.get_active_extract_provider", return_value=extract):
        return _web_row({"web": {"backend": "exa"}}, features)


def test_explicit_keyless_exa_reports_search_and_extract_ready():
    exa = _provider("exa", "Exa", keyless=True)

    assert _resolve_row(_features(), exa, exa) == (
        "Web Search & Extract (Exa)", True, None,
    )


def test_explicit_keyless_search_does_not_hide_unavailable_extract():
    exa = _provider("exa", "Exa", keyless=True)
    unavailable = _provider("paid-extract", "Paid Extract")

    assert _resolve_row(_features(), exa, unavailable) == (
        "Web Search & Extract", False, _WEB_MISSING,
    )


def test_unavailable_selected_provider_stays_missing_despite_other_credentials():
    exa = _provider("exa", "Exa")

    # A different web credential may make the subscription feature snapshot available,
    # but it cannot make the explicitly selected Exa route ready.
    assert _resolve_row(_features(available=True), exa, exa) == (
        "Web Search & Extract", False, _WEB_MISSING,
    )


def test_split_search_and_extract_backends_are_shown_separately():
    exa = _provider("exa", "Exa", keyless=True)
    tavily = _provider("tavily", "Tavily", keyless=True)

    assert _resolve_row(_features(provider="exa/tavily"), exa, tavily) == (
        "Web Search & Extract (Exa search, Tavily extract)", True, None,
    )


def test_managed_web_route_keeps_managed_summary():
    features = _features(available=True, managed=True)
    with patch("tools.web_tools._ensure_web_plugins_loaded") as discover:
        assert _web_row({}, features) == ("Web Search & Extract (Nous subscription)", True, None)
        discover.assert_not_called()


def test_unconfigured_summary_keeps_feature_snapshot_behavior():
    features = _features(available=True, explicit=False, provider="")
    with patch("tools.web_tools._ensure_web_plugins_loaded") as discover:
        assert _web_row({}, features) == ("Web Search & Extract", True, None)
        discover.assert_not_called()
