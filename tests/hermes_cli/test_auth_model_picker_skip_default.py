"""Portal login picker must not treat bare Enter as the paid flagship (#102943)."""

from hermes_cli.auth_model_picker import _CUSTOM_LABEL, _SKIP_LABEL, _prompt_model_selection


def test_curses_picker_defaults_to_skip_when_current_model_is_absent(monkeypatch):
    captured = {}

    def fake_radio(title, items, selected=0, **kwargs):
        captured["selected"] = selected
        captured["items"] = items
        return selected

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radio)
    monkeypatch.setattr(
        "hermes_cli.auth_model_picker._confirm_selection_guards",
        lambda *a, **k: True,
    )

    models = ["anthropic/claude-fable-5.1", "xiaomi/mimo-v2-pro"]
    result = _prompt_model_selection(models)

    assert captured["items"][-1] == _SKIP_LABEL
    assert captured["items"][-2] == _CUSTOM_LABEL
    assert captured["selected"] == len(models) + 1
    assert result is None


def test_curses_picker_defaults_to_current_model_when_listed(monkeypatch):
    captured = {}

    def fake_radio(title, items, selected=0, **kwargs):
        captured["selected"] = selected
        return 0

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radio)
    monkeypatch.setattr(
        "hermes_cli.auth_model_picker._confirm_selection_guards",
        lambda *a, **k: True,
    )

    result = _prompt_model_selection(
        ["anthropic/claude-fable-5.1", "xiaomi/mimo-v2-pro"],
        current_model="xiaomi/mimo-v2-pro",
    )

    assert captured["selected"] == 0
    assert result == "xiaomi/mimo-v2-pro"
