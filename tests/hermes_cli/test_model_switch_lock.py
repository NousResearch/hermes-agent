"""``model.lock_switching`` turns off model switching on every surface that goes through switch_model()."""

import hermes_yaml as yaml

from hermes_cli.model_switch import switch_model
from hermes_cli.model_switch_lock import DEFAULT_SWITCH_LOCK_MESSAGE, model_switch_lock_message


def _home(tmp_path, monkeypatch, model_cfg):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"model": model_cfg}), encoding="utf-8")
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _switch(target="other-model"):
    return switch_model(raw_input=target, current_provider="custom", current_model="locked-model",
                        current_base_url="https://gateway.example.test/v1")


def test_a_locked_install_refuses_every_switch_with_the_operators_message(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, {"default": "locked-model", "provider": "custom", "lock_switching": True,
                                  "lock_switching_message": "Routing policy picks the model. Start a new session."})
    result = _switch()
    assert result.success is False
    assert result.error_message == "Routing policy picks the model. Start a new session."


def test_the_lock_has_a_default_message():
    assert model_switch_lock_message({"model": {"lock_switching": True}}) == DEFAULT_SWITCH_LOCK_MESSAGE
    assert model_switch_lock_message({"model": {"lock_switching": True, "lock_switching_message": "  "}}) == (
        DEFAULT_SWITCH_LOCK_MESSAGE)


def test_without_the_lock_nothing_changes(tmp_path, monkeypatch):
    assert model_switch_lock_message({"model": {"default": "m"}}) == ""
    assert model_switch_lock_message({"model": "flat-string-model"}) == ""
    assert model_switch_lock_message({}) == ""
    _home(tmp_path, monkeypatch, {"default": "locked-model", "provider": "custom"})
    result = _switch()
    assert result.error_message != DEFAULT_SWITCH_LOCK_MESSAGE
