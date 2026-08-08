"""Regression tests for interactive `/model --global` picker persistence."""

from __future__ import annotations

from unittest.mock import patch


class _StubCLI:
    provider = "openai"
    model = "gpt-5.5"
    base_url = ""
    api_key = ""

    def __init__(self) -> None:
        self._model_picker_state = None
        self.closed = False
        self.applied = None

    def _capture_modal_input_snapshot(self) -> None:
        pass

    def _restore_modal_input_snapshot(self) -> None:
        pass

    def _invalidate(self, min_interval: float = 0.0) -> None:
        _ = min_interval

    def _close_model_picker(self) -> None:
        self.closed = True
        self._model_picker_state = None

    def _apply_model_switch_result(self, result, persist_global: bool) -> None:
        self.applied = (result, persist_global)


def test_open_model_picker_stores_persist_global_flag():
    import cli as cli_mod

    stub = _StubCLI()
    cli_mod.HermesCLI._open_model_picker(
        stub,
        [{"slug": "openai", "models": ["gpt-5.5"], "is_current": True}],
        "gpt-5.5",
        "OpenAI",
        persist_global=True,
    )

    assert stub._model_picker_state is not None
    assert stub._model_picker_state["persist_global"] is True


def test_picker_selection_uses_stored_persist_global_flag(monkeypatch):
    import cli as cli_mod

    stub = _StubCLI()
    stub._model_picker_state = {
        "stage": "model",
        "selected": 0,
        "persist_global": True,
        "provider_data": {"slug": "openai"},
        "model_list": ["gpt-5.5"],
        "user_provs": None,
        "custom_provs": None,
    }

    calls = {}

    def fake_switch_model(**kwargs):
        calls.update(kwargs)
        return object()

    with patch("hermes_cli.model_switch.switch_model", side_effect=fake_switch_model):
        cli_mod.HermesCLI._handle_model_picker_selection(stub)

    assert calls["is_global"] is True
    assert stub.closed is True
    assert stub.applied is not None
    assert stub.applied[1] is True
