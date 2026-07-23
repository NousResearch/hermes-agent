"""Regression test for issue #64503.

Disabling the ``browser`` toolset (e.g. in headless/Docker deployments
without Chromium) used to silently also remove ``web_search`` because
``web_search`` was statically listed inside the ``browser`` toolset.
With ``web_search`` moved out of the ``browser`` toolset, disabling
``browser`` should leave ``web_search`` available as long as the ``web``
toolset itself is still enabled.
"""
from model_tools import get_tool_definitions


def _names(tools):
    return {t["function"]["name"] for t in tools}


def test_disabling_browser_keeps_web_search(monkeypatch):
    """Disabling browser toolset must not strip web_search (#64503)."""
    # Mock web backend so web_search check_fn passes even without Tavily key
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-mock")

    tools_with_browser = get_tool_definitions(
        enabled_toolsets=["web", "browser"],
        quiet_mode=True,
    )
    tools_without_browser = get_tool_definitions(
        enabled_toolsets=["web", "browser"],
        disabled_toolsets=["browser"],
        quiet_mode=True,
    )
    names_with = _names(tools_with_browser)
    names_without = _names(tools_without_browser)

    # web_search must survive disabling browser
    assert "web_search" in names_with, (
        "precondition: web_search present with browser enabled"
    )
    assert "web_search" in names_without, (
        "web_search was removed when browser was disabled — it should survive "
        "because web_search belongs to the web toolset, not browser"
    )

    # browser_* tools must be removed
    browser_tools = {n for n in names_with if n.startswith("browser_")}
    removed = names_with - names_without
    assert browser_tools <= removed, (
        f"browser tools not fully removed: {browser_tools - removed}"
    )


def test_disabling_browser_no_web_enabled_still_ok(monkeypatch):
    """disabling browser when web isn't enabled is a clean no-op."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-mock")

    tools = get_tool_definitions(
        enabled_toolsets=[],
        disabled_toolsets=["browser"],
        quiet_mode=True,
    )
    assert "web_search" not in _names(tools)
