"""Tests for the ``web.prefer_fetch_over_browser`` toggle (issue #34545).

When both ``browser_navigate`` and ``web_extract`` are available the model
picks which to use, and it often reaches for the (slower, costlier) browser
even though ``browser_navigate``'s own description advises preferring the web
tools. The advisory hint is non-binding.

This toggle, when enabled, rewrites the ``browser_navigate`` description from
an advisory hint into a directive so non-interactive fetches route to
``web_extract``. The browser tools stay registered.

These tests pin:
- ``_prefer_fetch_over_browser`` reads config + honours the env override
- the directive is appended only when the flag is on AND web_extract exists
- default behaviour (flag off) leaves the description untouched
- the directive is never duplicated on repeated computation
"""
from __future__ import annotations

import pytest

import model_tools


_BROWSER_DESC = (
    "Navigate to a URL in the browser. For simple information retrieval, "
    "prefer web_search or web_extract (faster, cheaper)."
)


def _fake_definitions():
    """Minimal browser_navigate + web_extract schemas for the rewrite path."""
    return [
        {
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "description": _BROWSER_DESC,
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_extract",
                "description": "Extract content from a URL.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _compute(monkeypatch, prefer: bool):
    """Run the uncached computation with a controlled registry + flag."""
    monkeypatch.setattr(
        model_tools.registry,
        "get_definitions",
        lambda *a, **k: _fake_definitions(),
    )
    monkeypatch.setattr(
        model_tools, "_prefer_fetch_over_browser", lambda: prefer
    )
    defs = model_tools._compute_tool_definitions(
        enabled_toolsets=["web", "browser"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    by_name = {d["function"]["name"]: d for d in defs}
    return by_name["browser_navigate"]["function"]["description"]


class TestPreferFetchHelper:
    def test_env_truthy_overrides_config(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREFER_FETCH_OVER_BROWSER", "1")
        assert model_tools._prefer_fetch_over_browser() is True
        monkeypatch.setenv("HERMES_PREFER_FETCH_OVER_BROWSER", "true")
        assert model_tools._prefer_fetch_over_browser() is True

    def test_env_falsy_overrides_config(self, monkeypatch):
        # Even if config said true, an explicit env "0" wins.
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"web": {"prefer_fetch_over_browser": True}},
        )
        monkeypatch.setenv("HERMES_PREFER_FETCH_OVER_BROWSER", "0")
        assert model_tools._prefer_fetch_over_browser() is False

    def test_config_true_when_no_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFER_FETCH_OVER_BROWSER", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"web": {"prefer_fetch_over_browser": True}},
        )
        assert model_tools._prefer_fetch_over_browser() is True

    def test_default_false(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFER_FETCH_OVER_BROWSER", raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        assert model_tools._prefer_fetch_over_browser() is False

    def test_config_load_failure_is_false(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFER_FETCH_OVER_BROWSER", raising=False)

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        assert model_tools._prefer_fetch_over_browser() is False


class TestBrowserNavigateDescriptionRewrite:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        model_tools._tool_defs_cache.clear()
        yield
        model_tools._tool_defs_cache.clear()

    def test_directive_appended_when_enabled(self, monkeypatch):
        desc = _compute(monkeypatch, prefer=True)
        assert "prefer web_extract" in desc  # directive text
        assert "configured to prefer web_extract" in desc
        # Original advisory text is retained alongside the directive.
        assert "faster, cheaper" in desc

    def test_directive_absent_when_disabled(self, monkeypatch):
        desc = _compute(monkeypatch, prefer=False)
        assert "configured to prefer web_extract" not in desc
        # Untouched advisory description.
        assert desc == _BROWSER_DESC

    def test_directive_not_duplicated(self, monkeypatch):
        # Computing twice must not stack the directive twice.
        monkeypatch.setattr(
            model_tools.registry,
            "get_definitions",
            lambda *a, **k: _fake_definitions(),
        )
        monkeypatch.setattr(
            model_tools, "_prefer_fetch_over_browser", lambda: True
        )
        defs1 = model_tools._compute_tool_definitions(
            enabled_toolsets=["web", "browser"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        # Feed the (already-rewritten) output back in as the registry result.
        rewritten = [dict(d) for d in defs1]
        monkeypatch.setattr(
            model_tools.registry, "get_definitions", lambda *a, **k: rewritten
        )
        defs2 = model_tools._compute_tool_definitions(
            enabled_toolsets=["web", "browser"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        desc = {d["function"]["name"]: d for d in defs2}[
            "browser_navigate"
        ]["function"]["description"]
        assert desc.count("configured to prefer web_extract") == 1
