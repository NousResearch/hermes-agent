"""Tool setup must retain its category after curses clears prior console output."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import curses_ui, tools_config, tools_config_providers


class Screen:
    def __init__(self):
        self.keys = iter([10])
        self.frames = []
        self.rows = {}

    def clear(self):
        self.rows = {}

    def getmaxyx(self):
        return 24, 100

    def addnstr(self, y, x, text, n, attr):
        assert 0 <= y < 24
        self.rows[y] = self.rows.get(y, "") + text[:n]

    def refresh(self):
        self.frames.append(dict(self.rows))

    def getch(self):
        return next(self.keys)


@pytest.fixture
def menu_screen(monkeypatch):
    screen = Screen()
    curses = SimpleNamespace(
        error=type("CursesError", (Exception,), {}),
        A_BOLD=1, A_DIM=2, A_NORMAL=0,
        KEY_UP=259, KEY_DOWN=258, KEY_LEFT=260, KEY_ENTER=343,
        has_colors=lambda: False, curs_set=lambda value: None,
        wrapper=lambda draw: draw(screen),
    )
    monkeypatch.setitem(sys.modules, "curses", curses)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(curses_ui, "flush_stdin", lambda: None)
    return screen, curses


@pytest.mark.parametrize("ts_key", tools_config.TOOL_CATEGORIES)
@pytest.mark.parametrize("reconfigure", [False, True])
def test_provider_picker_keeps_tool_identity_on_screen(monkeypatch, menu_screen, ts_key, reconfigure):
    screen, _ = menu_screen
    category = tools_config.TOOL_CATEGORIES[ts_key]
    providers = [{"name": name, "env_vars": []} for name in ("Provider A", "Provider B")]
    monkeypatch.setattr(tools_config_providers, "_visible_providers", lambda *a, **kw: providers)
    monkeypatch.setattr(tools_config_providers, "_detect_active_provider_index", lambda *a, **kw: 0)
    monkeypatch.setattr(tools_config, "_provider_env_ready", lambda provider: False)
    monkeypatch.setattr(tools_config, "get_nous_subscription_features",
                        lambda *a, **kw: SimpleNamespace(nous_auth_present=False))
    configure_provider = Mock()
    monkeypatch.setattr(tools_config_providers, "_configure_provider", configure_provider)

    config = {}
    tools_config._configure_tool_category(ts_key, category, config, reconfigure=reconfigure)

    # Console headings printed before curses are not part of this fresh frame.
    heading = screen.frames[0][0]
    assert category["name"] in heading
    action = "Select provider" if reconfigure else category.get("setup_title", "Choose a provider")
    assert action in heading
    assert any("Provider A" in row for row in screen.frames[0].values())
    configure_provider.assert_called_once_with(providers[0], config, force_fresh=True, reconfigure=reconfigure)


@pytest.mark.parametrize("fallback", [False, True])
def test_tool_checklist_explains_batch_confirmation_without_changing_selection(
    monkeypatch, menu_screen, capsys, fallback,
):
    screen, curses = menu_screen
    toolsets = [("vision", "Vision / Image Analysis", "vision_analyze"),
                ("image_gen", "Image Generation", "image_generate")]
    monkeypatch.setattr(tools_config, "_get_effective_configurable_toolsets", lambda: toolsets)
    monkeypatch.setattr(tools_config, "_estimate_tool_tokens", dict)
    monkeypatch.setattr(tools_config, "_toolset_has_keys", lambda *a, **kw: True)
    screen.keys = iter([ord("j"), ord("k"), 10])
    if fallback:
        def unavailable(draw):
            raise curses.error("curses unavailable")
        curses.wrapper = unavailable
        monkeypatch.setattr("builtins.input", lambda prompt: "")

    selected = tools_config._prompt_toolset_checklist("All platforms", {"vision", "image_gen"})

    assert selected == {"vision", "image_gen"}
    if fallback:
        assert "all selections" in capsys.readouterr().out
    else:
        assert "ENTER apply all selections" in screen.frames[-1][1]
        assert any("→" in row and "Vision / Image Analysis" in row for row in screen.frames[-1].values())
