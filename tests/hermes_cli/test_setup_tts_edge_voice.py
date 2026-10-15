"""Voice selection must reach persisted runtime config without resetting other languages."""
import asyncio
from types import SimpleNamespace
from pathlib import Path

from hermes_cli import setup_tts
from hermes_cli.config import load_config, save_config
from hermes_cli.config_effective import load_user_config_effective


def test_selected_voice_reaches_tts_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_config({"tts": {"provider": "edge"}})

    def choose(question, choices, default):
        return next(i for i, label in enumerate(choices)
                    if label.startswith("Chinese (Simplified)" if "voice" in question else "Edge TTS"))

    monkeypatch.setattr(setup_tts._setup, "prompt_choice", choose)
    setup_tts.setup_tts(load_config())
    config = load_user_config_effective()
    seen = []

    class Communicate:
        def __init__(self, text, **kwargs):
            seen.append(kwargs["voice"])

        async def save(self, output):
            Path(output).write_bytes(b"fixture audio")

    monkeypatch.setattr("tools.tts_tool._import_edge_tts", lambda: SimpleNamespace(Communicate=Communicate))
    from tools.tts_tool_providers import _generate_edge_tts
    target = str(tmp_path / "speech.mp3")
    assert asyncio.run(_generate_edge_tts("你好", target, config["tts"])) == target
    assert seen == ["zh-CN-XiaoxiaoNeural"]
    assert Path(target).read_bytes() == b"fixture audio"


def test_existing_and_custom_voices_remain_profile_local(tmp_path, monkeypatch):
    homes = [tmp_path / "a", tmp_path / "b"]
    voices = ["fr-FR-DeniseNeural", "ja-JP-NanamiNeural"]
    for home, voice in zip(homes, voices):
        monkeypatch.setenv("HERMES_HOME", str(home))
        save_config({"tts": {"provider": "edge", "edge": {"voice": voice}}})

    custom = [False]
    def choose(question, choices, default):
        if "provider" in question:
            return next(i for i, label in enumerate(choices) if label.startswith("Edge TTS"))
        return len(choices) - 1 if custom[0] else default

    monkeypatch.setattr(setup_tts._setup, "prompt_choice", choose)
    monkeypatch.setattr(setup_tts._setup, "prompt", lambda *args, **kwargs: "de-DE-KatjaNeural")
    for index in (0, 1, 0):
        monkeypatch.setenv("HERMES_HOME", str(homes[index]))
        custom[0] = index == 1
        setup_tts.setup_tts(load_config())
        expected = "de-DE-KatjaNeural" if custom[0] else voices[0]
        assert load_user_config_effective()["tts"]["edge"]["voice"] == expected
