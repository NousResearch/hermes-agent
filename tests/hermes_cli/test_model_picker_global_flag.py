from __future__ import annotations

from hermes_cli.model_switch import ModelSwitchResult


class _PickerStub:
    model = "old-model"
    provider = "old-provider"
    base_url = ""
    api_key = ""
    _model_picker_state = None

    def __init__(self):
        self.invalidated = 0
        self.applied: list[bool] = []

    def _capture_modal_input_snapshot(self):
        pass

    def _restore_modal_input_snapshot(self):
        pass

    def _invalidate(self, *args, **kwargs):
        self.invalidated += 1

    def _close_model_picker(self):
        import cli as cli_mod

        cli_mod.HermesCLI._close_model_picker(self)

    def _apply_model_switch_result(self, result, persist_global: bool):
        assert result.success is True
        self.applied.append(persist_global)


def test_model_picker_preserves_global_flag_for_selection(monkeypatch):
    import cli as cli_mod

    switch_calls: list[dict] = []

    def _fake_switch_model(**kwargs):
        switch_calls.append(kwargs)
        return ModelSwitchResult(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider=kwargs["explicit_provider"],
            provider_changed=True,
            is_global=kwargs["is_global"],
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch_model)

    stub = _PickerStub()
    providers = [
        {"slug": "openai", "name": "OpenAI", "models": ["gpt-5"], "is_current": False}
    ]
    cli_mod.HermesCLI._open_model_picker(
        stub,
        providers,
        current_model="old-model",
        current_provider="old-provider",
        persist_global=True,
    )
    stub._model_picker_state.update(
        {
            "stage": "model",
            "provider_data": providers[0],
            "model_list": ["gpt-5"],
            "selected": 0,
        }
    )

    cli_mod.HermesCLI._handle_model_picker_selection(stub)

    assert switch_calls and switch_calls[0]["is_global"] is True
    assert stub.applied == [True]
