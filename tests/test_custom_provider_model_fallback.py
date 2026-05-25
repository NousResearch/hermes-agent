import sys
from types import SimpleNamespace

from hermes_cli.main import _model_flow_named_custom


class _MenuPickFirst:
    def __init__(self, items, **kwargs):
        self.items = items

    def show(self):
        return 0


class _MenuChooseDefault:
    last_items = None
    last_cursor_index = None

    def __init__(self, items, cursor_index=0, **kwargs):
        self.items = items
        self.cursor_index = cursor_index
        _MenuChooseDefault.last_items = items
        _MenuChooseDefault.last_cursor_index = cursor_index

    def show(self):
        return self.cursor_index


def test_named_custom_provider_uses_declared_models_when_models_endpoint_unavailable(monkeypatch, capsys):
    provider_info = {
        "name": "cpa-codex",
        "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-key",
        "key_env": "",
        "model": "gpt-5.4",
        "models": {
            "gpt-5.4": {},
            "gpt-5.5": {},
            "gpt-5.4-mini": {},
        },
        "api_mode": "chat_completions",
        "provider_key": "",
        "api_key_ref": "",
    }

    cfg = {"model": {}}
    saved_models = []
    saved_custom = []

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda new_cfg: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda model: saved_models.append(model))
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *args, **kwargs: saved_custom.append((args, kwargs)))
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *args, **kwargs: [])
    monkeypatch.setitem(sys.modules, "simple_term_menu", SimpleNamespace(TerminalMenu=_MenuPickFirst))
    monkeypatch.setattr("hermes_cli.curses_ui.flush_stdin", lambda: None)

    _model_flow_named_custom({}, provider_info)

    out = capsys.readouterr().out
    assert "Endpoint /models unavailable; using 3 configured model(s) from config.yaml." in out
    assert "Found 3 model(s):" in out
    assert saved_models == ["gpt-5.4"]
    assert cfg["model"]["provider"] == "custom"
    assert cfg["model"]["base_url"] == "http://127.0.0.1:8317/v1"
    assert cfg["model"]["api_key"] == "test-key"
    assert saved_custom, "expected selected model to be persisted back to custom_providers"


def test_named_custom_provider_preserves_declared_model_metadata_when_models_endpoint_unavailable(monkeypatch, capsys):
    provider_info = {
        "name": "cpa-codex",
        "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-key",
        "key_env": "",
        "model": "gpt-5.5",
        "models": {
            "gpt-5.4": {"context_length": 128000},
            "gpt-5.5": {"context_length": 256000, "supports_vision": True},
            "gpt-5.4-mini": {"context_length": 64000},
        },
        "api_mode": "chat_completions",
        "provider_key": "",
        "api_key_ref": "",
    }

    cfg = {"model": {}}
    saved_models = []
    saved_custom = []

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda new_cfg: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda model: saved_models.append(model))
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *args, **kwargs: saved_custom.append((args, kwargs)))
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *args, **kwargs: [])
    monkeypatch.setitem(sys.modules, "simple_term_menu", SimpleNamespace(TerminalMenu=_MenuChooseDefault))
    monkeypatch.setattr("hermes_cli.curses_ui.flush_stdin", lambda: None)

    _model_flow_named_custom({}, provider_info)

    out = capsys.readouterr().out
    assert "Endpoint /models unavailable; using 3 configured model(s) from config.yaml." in out
    assert saved_models == ["gpt-5.5"]
    assert _MenuChooseDefault.last_cursor_index == 1
    assert any("gpt-5.5 (current)" in item for item in _MenuChooseDefault.last_items)
    assert saved_custom == [
        (("http://127.0.0.1:8317/v1", "test-key", "gpt-5.5"), {"api_mode": "chat_completions"})
    ]
