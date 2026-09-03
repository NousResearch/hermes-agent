"""Regression tests for ``plugins.enabled`` type handling.

A quoted-string ``plugins.enabled`` (e.g. ``enabled: '["web/x"]'``) used to
make ``_get_enabled_plugins()`` return ``None``, which is the same signal as
"not configured yet" and therefore silently disabled EVERY plugin. For
capability-providing plugins (web search/extract backends) that meant a
silent fall-through to the paid default backend — a billing leak with no
error message anywhere.
"""

from __future__ import annotations

import importlib


def _enabled_for(tmp_path, monkeypatch, value: str):
    (tmp_path / "config.yaml").write_text(f"plugins:\n  enabled: {value}\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.config as config_mod
    import hermes_cli.plugins as plugins_mod

    importlib.reload(config_mod)
    importlib.reload(plugins_mod)
    return plugins_mod._get_enabled_plugins()


def test_proper_yaml_list_is_read(tmp_path, monkeypatch):
    got = _enabled_for(tmp_path, monkeypatch, "\n    - web/local-extract\n    - web/rotating-search")
    assert got == {"web/local-extract", "web/rotating-search"}


def test_quoted_json_string_is_coerced_not_dropped(tmp_path, monkeypatch):
    """The regression: a quoted list must not disable every plugin."""
    got = _enabled_for(tmp_path, monkeypatch, "'[\"profanity-relay\",\"web/rotating-search\"]'")
    assert got == {"profanity-relay", "web/rotating-search"}


def test_comma_joined_string_is_coerced(tmp_path, monkeypatch):
    got = _enabled_for(tmp_path, monkeypatch, "'web/local-extract, web/rotating-search'")
    assert got == {"web/local-extract", "web/rotating-search"}


def test_explicit_empty_list_disables_everything(tmp_path, monkeypatch):
    assert _enabled_for(tmp_path, monkeypatch, "[]") == set()


def test_bare_scalar_is_not_mistaken_for_a_plugin_name(tmp_path, monkeypatch):
    """A scalar is a config error; it must not become a 1-element allow-list."""
    assert _enabled_for(tmp_path, monkeypatch, "'true'") is None


def test_missing_key_returns_none(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("plugins:\n  disabled: []\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.config as config_mod
    import hermes_cli.plugins as plugins_mod

    importlib.reload(config_mod)
    importlib.reload(plugins_mod)
    assert plugins_mod._get_enabled_plugins() is None
