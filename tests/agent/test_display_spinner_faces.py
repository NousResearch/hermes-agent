from agent import display
from agent.display import KawaiiSpinner


def _set_indicator(monkeypatch, value):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"display": {"tui_status_indicator": value}},
    )


def test_spinner_faces_follow_emoji_indicator(monkeypatch):
    _set_indicator(monkeypatch, "emoji")

    assert KawaiiSpinner.get_waiting_faces() == KawaiiSpinner.STYLE_WAITING["emoji"]
    assert KawaiiSpinner.get_thinking_faces() == KawaiiSpinner.STYLE_THINKING["emoji"]
    assert not any("(" in face or "◕" in face for face in KawaiiSpinner.get_waiting_faces())


def test_spinner_faces_follow_ascii_indicator(monkeypatch):
    _set_indicator(monkeypatch, "ascii")

    assert KawaiiSpinner.get_waiting_faces() == KawaiiSpinner.STYLE_WAITING["ascii"]
    assert KawaiiSpinner.get_thinking_faces() == KawaiiSpinner.STYLE_THINKING["ascii"]


def test_spinner_faces_use_kaomoji_only_when_explicit(monkeypatch):
    _set_indicator(monkeypatch, "kaomoji")
    monkeypatch.setattr(display, "_get_skin", lambda: None)

    assert KawaiiSpinner.get_waiting_faces() == KawaiiSpinner.KAWAII_WAITING
    assert KawaiiSpinner.get_thinking_faces() == KawaiiSpinner.KAWAII_THINKING


def test_spinner_faces_use_default_config_when_unset(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "DEFAULT_CONFIG",
        {"display": {"tui_status_indicator": "emoji"}},
    )
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {"display": {}})

    assert KawaiiSpinner.get_waiting_faces() == KawaiiSpinner.STYLE_WAITING["emoji"]
    assert KawaiiSpinner.get_thinking_faces() == KawaiiSpinner.STYLE_THINKING["emoji"]


def test_spinner_faces_fall_back_to_default_config_when_config_unavailable(monkeypatch):
    import hermes_cli.config as config_mod

    def raise_config_error():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        config_mod,
        "DEFAULT_CONFIG",
        {"display": {"tui_status_indicator": "unicode"}},
    )
    monkeypatch.setattr(config_mod, "load_config_readonly", raise_config_error)

    assert KawaiiSpinner.get_waiting_faces() == KawaiiSpinner.STYLE_WAITING["unicode"]
    assert KawaiiSpinner.get_thinking_faces() == KawaiiSpinner.STYLE_THINKING["unicode"]
