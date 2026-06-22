import importlib.util
import os
import sys
import types


_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)


def test_voice_mixer_subclasses_discord_audio_source_when_discord_available(monkeypatch):
    fake_discord = types.ModuleType("discord")

    class _AudioSource:
        pass

    fake_discord.AudioSource = _AudioSource
    monkeypatch.setitem(sys.modules, "discord", fake_discord)

    spec = importlib.util.spec_from_file_location(
        "voice_mixer_with_discord",
        os.path.join(_DISCORD_DIR, "voice_mixer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert issubclass(module.VoiceMixer, _AudioSource)
    assert isinstance(module.VoiceMixer(), _AudioSource)
