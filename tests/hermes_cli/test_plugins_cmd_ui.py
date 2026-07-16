"""Tests for the interactive ``hermes plugins`` composite menu."""

from types import SimpleNamespace

from hermes_cli import plugins_cmd


class _FakeScreen:
    def __init__(self, keys, height=8, width=100):
        self.keys = list(keys)
        self.height = height
        self.width = width
        self.frames = []

    def clear(self):
        self.frames.append([])

    def getmaxyx(self):
        return self.height, self.width

    def addnstr(self, row, column, text, width, attr=0):
        self.frames[-1].append((row, str(text)[:width]))

    def refresh(self):
        pass

    def getch(self):
        return self.keys.pop(0)

    def keypad(self, enabled):
        pass


class _FakeCurses:
    A_NORMAL = 0
    A_BOLD = 1
    A_DIM = 2
    COLOR_GREEN = 2
    COLOR_YELLOW = 3
    COLOR_CYAN = 6
    KEY_UP = 1001
    KEY_DOWN = 1002
    KEY_PPAGE = 1003
    KEY_NPAGE = 1004
    KEY_HOME = 1005
    KEY_END = 1006
    KEY_ENTER = 1007

    class error(Exception):
        pass

    def __init__(self, screen):
        self.screen = screen

    def wrapper(self, draw):
        draw(self.screen)

    def has_colors(self):
        return False

    def curs_set(self, visible):
        pass


def test_build_composite_rows_uses_one_visual_coordinate_space():
    categories = [
        ("Memory Provider", "built-in", lambda: False),
        ("Context Engine", "compressor", lambda: False),
    ]

    rows = plugins_cmd._build_composite_rows(
        plugin_labels=["plugin-a", "plugin-b"],
        categories=categories,
    )

    assert [row["kind"] for row in rows] == [
        "section",
        "plugin",
        "plugin",
        "spacer",
        "section",
        "category",
        "category",
    ]
    assert plugins_cmd._composite_navigation_rows(rows) == [1, 2, 5, 6]
    assert rows[5]["nav"] == ("category", 0)


def test_page_navigation_lands_on_selectable_visual_row():
    navigation_rows = [1, 2, 3, 7, 8]

    assert plugins_cmd._page_composite_cursor(navigation_rows, 0, 4, 1) == 3
    assert plugins_cmd._page_composite_cursor(navigation_rows, 4, 4, -1) == 2


def test_constrained_viewport_renders_selected_provider_category(monkeypatch):
    screen = _FakeScreen(keys=[_FakeCurses.KEY_END, ord("q")], height=8)
    curses = _FakeCurses(screen)
    plugin_keys = [f"group/plugin-{index}" for index in range(6)]
    categories = [
        ("Memory Provider", "built-in", lambda: False),
        ("Context Engine", "compressor", lambda: False),
    ]

    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(plugin_keys))
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda value: None)
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda value: None)

    plugins_cmd._run_composite_ui(
        curses,
        plugin_keys,
        [f"plugin-{index}" for index in range(6)],
        set(range(6)),
        set(),
        categories,
        SimpleNamespace(print=lambda *args, **kwargs: None),
    )

    assert len(screen.frames) == 2
    rendered = [text for _row, text in screen.frames[-1]]
    assert any("→   Context Engine" in text for text in rendered)
    assert any("↑ more" in text for text in rendered)


def test_composite_ui_persists_canonical_nested_plugin_key(monkeypatch):
    screen = _FakeScreen(keys=[ord(" "), ord("q")])
    curses = _FakeCurses(screen)
    saved = {}

    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(
        plugins_cmd, "_save_enabled_set", lambda value: saved.update(enabled=value)
    )
    monkeypatch.setattr(
        plugins_cmd, "_save_disabled_set", lambda value: saved.update(disabled=value)
    )

    plugins_cmd._run_composite_ui(
        curses,
        ["web/firecrawl"],
        ["web-firecrawl — bundled search"],
        set(),
        {"firecrawl"},
        [],
        SimpleNamespace(print=lambda *args, **kwargs: None),
    )

    assert saved["enabled"] == {"web/firecrawl"}
    assert saved["disabled"] == set()
