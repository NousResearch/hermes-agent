"""Tests for the ``web.prefer_fetch_over_browser`` toggle (issue #34545).

When both the browser and ``web_extract`` are available the model picks which
to use, and it often reaches for the browser even though a plain HTTP fetch
would do.

The toggle, when enabled, appends a directive to ``browser_exec`` (the current
browser surface; ``browser_navigate`` is gone) so non-interactive fetches route
to ``web_extract``. The browser tool stays registered. With the flag off, the
description is left unchanged.
"""
from __future__ import annotations

import pytest

import model_tools


_BROWSER_DESC = (
    "Drive a real browser for JS pages, logins, forms and clicks."
)


def _fake_definitions(*, include_extract: bool = True):
    """Minimal browser_exec schema, plus the tools the rewrite may name."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_exec",
                "description": _BROWSER_DESC,
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    if include_extract:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_extract",
                    "description": "Extract content from a URL.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def _compute(monkeypatch, prefer: bool, *, include_extract: bool = True):
    """Run the uncached computation with a controlled registry + flag."""
    monkeypatch.setattr(
        model_tools.registry,
        "get_definitions",
        lambda *a, **k: _fake_definitions(include_extract=include_extract),
    )
    monkeypatch.setattr(model_tools, "_prefer_fetch_over_browser", lambda: prefer)
    defs = model_tools._compute_tool_definitions(
        enabled_toolsets=["web", "browser"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    by_name = {d["function"]["name"]: d for d in defs}
    return by_name, by_name["browser_exec"]["function"]["description"]


class TestPreferFetchHelper:
    def test_env_truthy_overrides_config(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREFER_FETCH_OVER_BROWSER", "1")
        assert model_tools._prefer_fetch_over_browser() is True
        monkeypatch.setenv("HERMES_PREFER_FETCH_OVER_BROWSER", "true")
        assert model_tools._prefer_fetch_over_browser() is True

    def test_env_falsy_overrides_config(self, monkeypatch):
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
        monkeypatch.setattr("hermes_cli.config.load_config", dict)
        assert model_tools._prefer_fetch_over_browser() is False

    def test_config_load_failure_is_false(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFER_FETCH_OVER_BROWSER", raising=False)

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        assert model_tools._prefer_fetch_over_browser() is False

    def test_non_dict_web_section_is_false(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFER_FETCH_OVER_BROWSER", raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"web": "yes"})
        assert model_tools._prefer_fetch_over_browser() is False


class TestBrowserExecDescriptionRewrite:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        model_tools._tool_defs_cache.clear()
        yield
        model_tools._tool_defs_cache.clear()

    def test_directive_appended_when_enabled(self, monkeypatch):
        by_name, desc = _compute(monkeypatch, prefer=True)
        assert "browser_exec" in by_name
        assert "web_extract" in by_name
        assert "configured to prefer web_extract" in desc
        assert _BROWSER_DESC in desc

    def test_directive_absent_when_disabled(self, monkeypatch):
        _by_name, desc = _compute(monkeypatch, prefer=False)
        assert "configured to prefer web_extract" not in desc
        assert desc == _BROWSER_DESC

    def test_directive_absent_without_web_extract(self, monkeypatch):
        by_name, desc = _compute(monkeypatch, prefer=True, include_extract=False)
        assert "web_extract" not in by_name
        assert "configured to prefer web_extract" not in desc
        assert "browser_exec" in by_name

    def test_directive_not_duplicated(self, monkeypatch):
        monkeypatch.setattr(
            model_tools.registry,
            "get_definitions",
            lambda *a, **k: _fake_definitions(),
        )
        monkeypatch.setattr(model_tools, "_prefer_fetch_over_browser", lambda: True)
        defs1 = model_tools._compute_tool_definitions(
            enabled_toolsets=["web", "browser"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        rewritten = [dict(d) for d in defs1]
        monkeypatch.setattr(
            model_tools.registry,
            "get_definitions",
            lambda *a, **k: rewritten,
        )
        defs2 = model_tools._compute_tool_definitions(
            enabled_toolsets=["web", "browser"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        desc = {d["function"]["name"]: d for d in defs2}["browser_exec"]["function"]["description"]
        assert desc.count("configured to prefer web_extract") == 1

    def test_cache_key_tracks_prefer_fetch_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFER_FETCH_OVER_BROWSER", raising=False)
        off = model_tools._tool_defs_cache_key(["web"], None, True)
        monkeypatch.setenv("HERMES_PREFER_FETCH_OVER_BROWSER", "1")
        on = model_tools._tool_defs_cache_key(["web"], None, True)
        if off is None or on is None:
            pytest.skip("tool-def cache bypassed in this profile scope")
        assert off != on
