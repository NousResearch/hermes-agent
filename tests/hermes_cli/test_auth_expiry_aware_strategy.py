"""Strategy selection persists only a valid, explicitly selected option."""

import pytest

from hermes_cli import auth_commands, config
from agent.credential_pool import get_pool_strategy


@pytest.mark.parametrize("raw", ["expiry", "0", "-1", "999", ""])
def test_invalid_strategy_does_not_write(monkeypatch, raw):
    monkeypatch.setattr(auth_commands, "_pick_provider", lambda *_: "openai-codex")
    monkeypatch.setattr(auth_commands, "_ask", lambda *_: raw)
    writes = []
    monkeypatch.setattr(config, "save_config", writes.append)
    auth_commands._interactive_strategy()
    assert writes == []


def test_expiry_aware_menu_persists_and_loads_real_config(monkeypatch, capsys):
    monkeypatch.setattr(auth_commands, "_pick_provider", lambda *_: "openai-codex")
    monkeypatch.setattr(auth_commands, "_ask", lambda *_: "5")
    auth_commands._interactive_strategy()
    assert get_pool_strategy("openai-codex") == "expiry_aware"
    assert config.load_config()["credential_pool_strategies"]["openai-codex"] == "expiry_aware"
    assert "expiry_aware" in capsys.readouterr().out
