"""Plugin-declared config field bridge tests."""

from types import SimpleNamespace


def test_plugin_config_fields_are_added_to_dashboard_schema(monkeypatch):
    from hermes_cli import web_server

    manifest = SimpleNamespace(
        config_schema={
            "stt.bifrost.model": {
                "label": "Bifrost STT Model",
                "type": "select",
                "options": ["small", "large"],
                "description": "Model used by the provider",
            }
        }
    )
    loaded = SimpleNamespace(manifest=manifest)
    manager = SimpleNamespace(
        _plugins={"bifrost": loaded},
        discover_and_load=lambda: None,
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    fields = web_server._plugin_config_schema_fields()

    assert fields["stt.bifrost.model"] == {
        "type": "select",
        "label": "Bifrost STT Model",
        "options": ["small", "large"],
        "description": "Model used by the provider",
    }


def test_core_schema_wins_over_plugin_collision(monkeypatch):
    from hermes_cli import web_server

    manifest = SimpleNamespace(config_schema={"agent.max_turns": {"type": "string"}})
    manager = SimpleNamespace(
        _plugins={"collision": SimpleNamespace(manifest=manifest)},
        discover_and_load=lambda: None,
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    fields = web_server._plugin_config_schema_fields()

    assert "model" not in fields
