"""``reply_to_mode: off`` reaches the adapter as ``off`` however it was written.

YAML 1.1 reads a bare ``off`` as ``False`` and ``hermes config set ... off`` writes ``false``. The
single-profile env bridge maps that back to ``off``, but a multiplexed secondary profile skips the
bridge and reads ``PlatformConfig.reply_to_mode`` directly, so ``False`` fell to the adapters'
``or 'first'`` default and every reply kept quoting the user's message.
"""

from __future__ import annotations

import pytest

from agent.secret_scope import set_multiplex_active
from hermes_agent.gateway.config import Platform, PlatformConfig, load_gateway_config
from hermes_cli import config as cli_config
from hermes_cli.plugins import discover_plugins


@pytest.fixture
def secondary(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    home = launch / "profiles" / "b2"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    discover_plugins()
    yield home
    set_multiplex_active(False)


def _load_in_scope(home):
    from hermes_agent.gateway.run import _profile_runtime_scope
    set_multiplex_active(True)
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        return load_gateway_config()


def test_config_set_off_disables_reply_threading_for_a_secondary_profile(secondary, monkeypatch):
    (secondary / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(secondary))
    cli_config.set_config_value("platforms.telegram.reply_to_mode", "off")
    from plugins.platforms.telegram.adapter import TelegramAdapter

    cfg = _load_in_scope(secondary)

    pc = cfg.platforms[Platform.TELEGRAM]
    assert pc.reply_to_mode == "off"
    adapter = TelegramAdapter(pc)
    assert adapter._should_thread_reply("42", 0) is False


@pytest.mark.parametrize(("raw", "expected"), [
    (False, "off"), ("off", "off"), (" ALL ", "all"), ("first", "first"), (None, "first"),
    ("sometimes", "first"), (True, "first"),
])
def test_reply_to_mode_is_always_a_known_mode(raw, expected):
    data = {"enabled": True} if raw is None else {"enabled": True, "reply_to_mode": raw}
    assert PlatformConfig.from_dict(data).reply_to_mode == expected
