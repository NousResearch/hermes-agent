"""Completion colors remain paired after base-style and skin composition.

Additional integration coverage for MoonMao42's fix in PR #107175.
"""

import pytest


@pytest.mark.parametrize("skin_overrides", [True, False])
def test_initialized_completion_styles_preserve_painted_pairs(monkeypatch, skin_overrides):
    import cli
    from hermes_cli import skin_engine
    from prompt_toolkit.styles import Style

    monkeypatch.setattr(skin_engine, "_active_skin", skin_engine.load_skin("default"))
    if not skin_overrides:
        monkeypatch.setattr(skin_engine, "get_prompt_toolkit_style_overrides", dict)

    instance = cli.HermesCLI.__new__(cli.HermesCLI)
    instance._tui_set_base_style()
    rendered = []
    for light_mode in (False, True):
        monkeypatch.setattr(cli, "_LIGHT_MODE_CACHE", light_mode)
        rendered.append(Style.from_dict(instance._build_tui_style_dict()))

    for name in (
        "completion-menu",
        "completion-menu.completion",
        "completion-menu.completion.current",
        "completion-menu.meta.completion",
        "completion-menu.meta.completion.current",
    ):
        dark, light = [style.get_attrs_for_style_str(f"class:{name}") for style in rendered]
        assert (light.color, light.bgcolor) == (dark.color, dark.bgcolor), name
