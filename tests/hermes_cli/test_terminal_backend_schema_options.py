"""``terminal.backend`` schema options must be discovery-driven.

The Desktop Settings dropdown for ``terminal.backend`` consumes
``schema.options``; the backend schema merges the installed plugin backends
(and, since this regression, the currently configured one) per request, so a
backend installed after server start still appears — the same contract as
``memory.provider``. These tests pin the merge itself:

- a plugin-registered backend is appended to the built-ins;
- a configured backend whose plugin is gone stays selectable (it must not
  vanish from the dropdown while it is still the active backend);
- nothing is overlaid (no needless shallow copy) when there is nothing to add.
"""

import pytest


def _schema_with(monkeypatch, *, plugin_rows, cfg):
    from hermes_cli import web_server_config
    from hermes_cli import web_server_profiles

    monkeypatch.setattr(web_server_profiles, "_plugin_terminal_backend_rows",
                        lambda: plugin_rows)
    from hermes_cli import config as config_module

    monkeypatch.setattr(config_module, "load_config", lambda: cfg)
    return web_server_config._schema_with_dynamic_provider_options()


class TestTerminalBackendSchemaOptions:
    @pytest.fixture(autouse=True)
    def _home(self, _isolate_hermes_home):
        pass

    def test_a_plugin_backend_is_offered_beside_the_built_ins(self, monkeypatch):
        schema = _schema_with(
            monkeypatch,
            plugin_rows=[{"name": "apple_container", "label": "Apple Container",
                          "description": "…"}],
            cfg={},
        )
        options = schema["terminal.backend"]["options"]
        assert "apple_container" in options
        for builtin in ("local", "docker", "ssh"):
            assert builtin in options

    def test_a_configured_backend_stays_selectable_when_its_plugin_is_gone(self, monkeypatch):
        schema = _schema_with(
            monkeypatch,
            plugin_rows=[],
            cfg={"terminal": {"backend": "apple_container"}},
        )
        # Still the active backend, so it must remain in the dropdown even
        # though nothing discovers it anymore.
        assert "apple_container" in schema["terminal.backend"]["options"]
        assert "local" in schema["terminal.backend"]["options"]

    def test_nothing_extra_leaves_the_schema_entry_alone(self, monkeypatch):
        from hermes_cli.web_server_config import CONFIG_SCHEMA

        schema = _schema_with(monkeypatch, plugin_rows=[], cfg={})
        assert schema["terminal.backend"] is CONFIG_SCHEMA["terminal.backend"]
