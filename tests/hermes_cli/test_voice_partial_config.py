"""voice.partial.* is registered and readable through the normal config path.

The on/off switch for live transcription is ``stt.streaming`` (see #133853). ``voice.partial`` is
tuning for the local re-decode loop only — a reader in tools.voice_partial that must have
registered defaults, or a fresh profile crashes on it and a user cannot tune it via
``hermes config set``.
"""

from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG, load_config, set_config_value
from tools.voice_partial import _DEFAULTS, _FLOORS, _partial_cfg


def test_voice_partial_defaults_are_registered():
    voice = DEFAULT_CONFIG.get("voice")
    assert isinstance(voice, dict), "voice section missing from DEFAULT_CONFIG"
    partial = voice.get("partial")
    assert isinstance(partial, dict), "voice.partial missing from DEFAULT_CONFIG"
    # Tuning only: no second on/off switch fighting stt.streaming.
    assert "enabled" not in partial
    assert partial == _DEFAULTS


def test_voice_partial_keys_are_settable_dotted(tmp_path, monkeypatch):
    """`hermes config set voice.partial.tail_seconds 8` must land where the engine reads it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    set_config_value("voice.partial.tail_seconds", "8")
    set_config_value("voice.partial.interval_seconds", "3")
    cfg = load_config()
    assert cfg["voice"]["partial"]["tail_seconds"] == 8
    assert cfg["voice"]["partial"]["interval_seconds"] == 3


def test_partial_cfg_reads_the_profile_config(tmp_path, monkeypatch):
    """The engine reads the same keys a user sets — not a private source of truth."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import config as hermes_config

    hermes_config.save_config({
        "tts": {"provider": "edge"},
        "voice": {"partial": {"tail_seconds": 6.0, "interval_seconds": 4.0, "min_seconds": 2.0}},
    })
    cfg = _partial_cfg()
    assert cfg["tail_seconds"] == 6.0
    assert cfg["interval_seconds"] == 4.0
    assert cfg["min_seconds"] == 2.0

    # A profile that never mentions voice.partial still gets the registered defaults.
    hermes_config.save_config({"tts": {"provider": "edge"}})
    assert _partial_cfg() == _DEFAULTS


def test_floors_are_below_defaults():
    """Sanity on the clamp: the registered defaults must never be clamped away themselves."""
    for key, default in _DEFAULTS.items():
        assert _FLOORS[key] < default
