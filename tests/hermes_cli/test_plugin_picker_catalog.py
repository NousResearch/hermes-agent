"""The picker catalog extension crosses real plugin discovery and profile scope."""

from pathlib import Path

import hermes_cli.plugins as plugins
from hermes_cli.model_switch_providers import (
    _apply_plugin_picker_catalogs,
    _picker_catalog_base_url,
)
from hermes_cli.config_providers import (
    _custom_provider_entry_to_provider_config,
    _normalize_custom_provider_entry,
)


def _install(home: Path, label: str) -> None:
    plugin = home / "plugins" / "sample-catalog"
    plugin.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [sample-catalog]\n", encoding="utf-8",
    )
    (plugin / "plugin.yaml").write_text(
        "name: sample-catalog\nversion: 1.0.0\nprovides_hooks:\n"
        "  - picker_model_catalog\n", encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def catalog(provider_key, base_url, provider_config, row_models, non_blocking):\n"
        "        if not provider_config['plugin_options'].get('sample-catalog'):\n"
        "            return None\n"
        f"        return {{'provider_key': provider_key, 'models': [{label!r}]}}\n"
        "    ctx.register_hook('picker_model_catalog', catalog)\n",
        encoding="utf-8",
    )


def test_real_plugin_discovery_is_scoped_a_b_a(tmp_path, monkeypatch):
    """A secondary profile cannot reuse the first profile's catalog or plugin instance."""
    a, b = tmp_path / "a", tmp_path / "b"
    _install(a, "org/A")
    _install(b, "org/B")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty"))
    plugins._reset_plugin_managers_for_tests()
    try:
        for home, expected in ((a, "org/A"), (b, "org/B"), (a, "org/A")):
            monkeypatch.setenv("HERMES_HOME", str(home))
            rows = [{"slug": "custom:local", "aliases": ["local"],
                     "models": ["saved"], "total_models": 1, "is_current": True}]
            providers = {"local": {
                "base_url": "http://user:password@127.0.0.1:8084/v1?token=hidden",
                "plugin_options": {"sample-catalog": {"enabled": True}},
            }}
            _apply_plugin_picker_catalogs(rows, providers, non_blocking=True)
            assert rows[0]["models"] == [expected]
            assert rows[0]["total_models"] == 1
    finally:
        plugins._reset_plugin_managers_for_tests()


def test_authoritative_empty_catalog_and_missing_plugin(tmp_path, monkeypatch):
    home = tmp_path / "a"
    _install(home, "org/A")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty"))
    plugins._reset_plugin_managers_for_tests()
    try:
        rows = [{"slug": "local", "models": ["saved"], "total_models": 1,
                 "is_current": True}]
        providers = {"local": {"plugin_options": {"other-plugin": {}}}}
        _apply_plugin_picker_catalogs(rows, providers, non_blocking=True)
        assert rows[0]["models"] == ["saved"]

        def _empty(**kwargs):
            return [{"provider_key": "local", "models": []}]

        monkeypatch.setattr(plugins, "invoke_hook", lambda *_args, **kw: _empty(**kw))
        _apply_plugin_picker_catalogs(rows, providers, non_blocking=True)
        assert rows[0]["models"] == []
        assert rows[0]["native_catalog_empty"] is True
    finally:
        plugins._reset_plugin_managers_for_tests()


def test_catalog_url_omits_credentials_query_fragment():
    assert _picker_catalog_base_url({
        "api": "https://u:p@example.com:8443/v1?token=secret#fragment",
    }) == "https://example.com:8443/v1"


def test_plugin_options_survive_legacy_provider_normalization_without_mutation():
    original = {
        "name": "local", "base_url": "http://127.0.0.1:8084/v1",
        "plugin_options": {"example": {"root": "/models"}},
    }
    normalized = _normalize_custom_provider_entry(original, provider_key="local")
    assert normalized["plugin_options"] == original["plugin_options"]
    assert normalized["plugin_options"] is not original["plugin_options"]
    converted = _custom_provider_entry_to_provider_config(original, provider_key="local")
    assert converted["plugin_options"] == original["plugin_options"]
