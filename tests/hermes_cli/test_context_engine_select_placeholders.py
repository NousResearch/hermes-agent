"""Regression for dashboard/desktop context.engine placeholder options (#107664).

UI used to offer ``default``/``custom`` (and desktop also listed ``compressor``).
Runtime only treats literal ``compressor`` as built-in; every other name is a
plugin lookup. Selecting the placeholders wrote a name nothing implements and
logged a bogus ``not found`` warning.
"""

from __future__ import annotations

import pytest
import yaml


def test_schema_offers_only_placeholders_on_main():
    from hermes_cli.web_server_config import CONFIG_SCHEMA
    opts = CONFIG_SCHEMA["context.engine"]["options"]
    assert "compressor" in opts
    assert "default" not in opts and "custom" not in opts


def test_legacy_default_alias_silent(caplog):
    from agent.agent_init import _select_context_engine
    with caplog.at_level("WARNING"):
        assert _select_context_engine({"context": {"engine": "default"}}) is None
    assert "not found" not in caplog.text


def test_legacy_custom_alias_silent(caplog):
    from agent.agent_init import _select_context_engine
    with caplog.at_level("WARNING"):
        assert _select_context_engine({"context": {"engine": "custom"}}) is None
    assert "not found" not in caplog.text


def test_legacy_default_alias_case_insensitive(caplog):
    from agent.agent_init import _select_context_engine
    with caplog.at_level("WARNING"):
        assert _select_context_engine({"context": {"engine": "  DeFaUlT  "}}) is None
    assert "not found" not in caplog.text


def test_compressor_still_returns_none_without_warning(caplog):
    from agent.agent_init import _select_context_engine
    with caplog.at_level("WARNING"):
        assert _select_context_engine({"context": {"engine": "compressor"}}) is None
    assert "not found" not in caplog.text


def test_missing_plugin_name_still_warns_not_found(caplog):
    from agent.agent_init import _select_context_engine
    with caplog.at_level("WARNING"):
        assert _select_context_engine({"context": {"engine": "definitely-missing-xyz"}}) is None
    assert "not found" in caplog.text


def test_substring_default_is_not_an_alias(caplog):
    from agent.agent_init import _select_context_engine
    with caplog.at_level("WARNING"):
        assert _select_context_engine({"context": {"engine": "my-default-engine"}}) is None
    assert "not found" in caplog.text


def test_dynamic_schema_excludes_placeholders_and_includes_discovered(monkeypatch):
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli import web_server_config as wsc

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"context": {"engine": "compressor"}},
    )
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_context_engines",
        lambda: [("freshly_installed", "test engine")],
    )

    fields = wsc._schema_with_dynamic_provider_options()
    opts = fields["context.engine"]["options"]
    assert "compressor" in opts
    assert "freshly_installed" in opts
    assert "default" not in opts
    assert "custom" not in opts
    assert wsc.CONFIG_SCHEMA["context.engine"]["options"] == ["compressor"]


def test_dynamic_schema_keeps_configured_undiscovered_plugin(monkeypatch):
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli import web_server_config as wsc

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"context": {"engine": "vanished-engine"}},
    )
    monkeypatch.setattr(plugins_cmd, "_discover_context_engines", lambda: [])

    opts = wsc._schema_with_dynamic_provider_options()["context.engine"]["options"]
    assert "compressor" in opts
    assert "vanished-engine" in opts
    assert "default" not in opts
    assert "custom" not in opts


def test_dynamic_schema_fail_open_when_discovery_raises(monkeypatch):
    import hermes_cli.plugins_cmd as plugins_cmd
    from hermes_cli import web_server_config as wsc

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"context": {"engine": "compressor"}},
    )

    def _boom():
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(plugins_cmd, "_discover_context_engines", _boom)

    opts = wsc._schema_with_dynamic_provider_options()["context.engine"]["options"]
    assert "compressor" in opts
    assert "default" not in opts
    assert "custom" not in opts


def test_save_context_engine_default_persists_compressor(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text("context:\n  engine: compressor\n", encoding="utf-8")
    from hermes_cli.plugins_cmd import _save_context_engine

    _save_context_engine("default")
    content = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert content["context"]["engine"] == "compressor"


@pytest.mark.parametrize("raw", ["custom", "CUSTOM", "", None])
def test_save_context_engine_aliases_persist_compressor(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text("context:\n  engine: compressor\n", encoding="utf-8")
    from hermes_cli.plugins_cmd import _save_context_engine

    _save_context_engine(raw)
    content = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert content["context"]["engine"] == "compressor"


def test_save_context_engine_keeps_real_plugin_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text("context:\n  engine: compressor\n", encoding="utf-8")
    from hermes_cli.plugins_cmd import _save_context_engine

    _save_context_engine("lcm")
    content = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert content["context"]["engine"] == "lcm"
