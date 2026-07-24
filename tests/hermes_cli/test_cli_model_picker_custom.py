"""Tests for prompt_toolkit /model picker custom model entry."""

from types import SimpleNamespace

from cli import HermesCLI


class _Buffer:
    def __init__(self, text=""):
        self.text = text
        self.cursor_position = len(text)
        self.reset_count = 0

    def reset(self):
        self.text = ""
        self.cursor_position = 0
        self.reset_count += 1


def _cli_with_picker_state(state, *, typed=""):
    cli = HermesCLI.__new__(HermesCLI)
    cli._model_picker_state = state
    cli._app = SimpleNamespace(current_buffer=_Buffer(typed))
    cli.model = "old-model"
    cli.provider = "openrouter"
    cli.base_url = "https://openrouter.ai/api/v1"
    cli.api_key = "key"
    cli._invalidate = lambda min_interval=0.0: None
    return cli


def test_model_picker_model_stage_can_enter_custom_model():
    state = {
        "stage": "model",
        "selected": 2,
        "provider_data": {"slug": "openrouter", "name": "OpenRouter"},
        "model_list": ["anthropic/claude-sonnet-4", "openai/gpt-5"],
    }
    cli = _cli_with_picker_state(state, typed="stale")

    cli._handle_model_picker_selection()

    assert state["stage"] == "custom_model"
    assert state["selected"] == 0
    assert state["custom_error"] == ""
    assert cli._app.current_buffer.text == ""
    assert cli._app.current_buffer.reset_count == 1


def test_model_picker_custom_stage_switches_session_model(monkeypatch):
    state = {
        "stage": "custom_model",
        "selected": 0,
        "persist_global": False,
        "provider_data": {"slug": "openrouter", "name": "OpenRouter"},
        "user_provs": {"openrouter": {}},
        "custom_provs": [],
    }
    cli = _cli_with_picker_state(state, typed="acme/new-model")
    applied = {}
    result = SimpleNamespace(success=True, new_model="acme/new-model")

    def fake_switch_model(**kwargs):
        applied["switch_kwargs"] = kwargs
        return result

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)
    cli._close_model_picker = lambda: applied.setdefault("closed", True)
    cli._apply_model_switch_result = lambda res, persist: applied.update(
        result=res,
        persist_global=persist,
    )

    cli._handle_model_picker_selection()

    assert applied["closed"] is True
    assert applied["result"] is result
    assert applied["persist_global"] is False
    assert applied["switch_kwargs"]["raw_input"] == "acme/new-model"
    assert applied["switch_kwargs"]["explicit_provider"] == "openrouter"
    assert applied["switch_kwargs"]["is_global"] is False


def test_model_picker_custom_stage_requires_model_name():
    state = {
        "stage": "custom_model",
        "selected": 0,
        "provider_data": {"slug": "openrouter", "name": "OpenRouter"},
    }
    cli = _cli_with_picker_state(state, typed="  ")

    cli._handle_model_picker_selection()

    assert state["stage"] == "custom_model"
    assert "Enter a model name" in state["custom_error"]
