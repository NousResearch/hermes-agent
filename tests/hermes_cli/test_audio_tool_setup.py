"""Audio is opt-in and setup reaches the shared auxiliary model picker."""

from hermes_cli import main_provider_setup, tools_config, tools_config_providers
from toolsets import resolve_toolset


def test_audio_enablement_requires_setup_and_routes_to_audio_picker(monkeypatch):
    from tools import audio_tools
    monkeypatch.setattr(audio_tools, "check_audio_requirements", lambda: False)
    assert "audio" not in tools_config._get_platform_tools({}, "cli")
    assert "audio_analyze" not in resolve_toolset("hermes-cli")
    assert "audio_analyze" in resolve_toolset("audio")
    assert tools_config._toolsets_needing_setup({"audio"}, {}) == ["audio"]
    selected = []
    monkeypatch.setattr(main_provider_setup, "_aux_select_for_task", selected.append)
    tools_config_providers._configure_simple_requirements("audio")
    assert selected == ["audio"]
