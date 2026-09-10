import logging

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_dispatch import _normalize_chrome_fragments


def _ctx(manager):
    return PluginContext(PluginManifest(name="fake-chrome", version="1"), manager)


def test_chrome_renderer_registration_is_tracked_and_unregistered():
    manager = PluginManager(scope_key="/tmp/chrome-test")
    ctx = _ctx(manager)
    handle = ctx.register_chrome_renderer(lambda surface, width, info: [("class:a", "x" * width)])
    assert manager.render_chrome("input_rule_top", 4, {"session_id": None, "skin": "default"})
    handle.dispose()
    assert manager.render_chrome("input_rule_top", 4, {}) is None


def test_none_renderer_preserves_default_fallback():
    manager = PluginManager(scope_key="/tmp/chrome-test")
    _ctx(manager).register_chrome_renderer(lambda surface, width, info: None)
    assert manager.render_chrome("input_rule_bot", 3, {}) is None


def test_renderer_exception_falls_back_and_disables(caplog):
    calls = []
    manager = PluginManager(scope_key="/tmp/chrome-test")

    def broken(surface, width, info):
        calls.append(1)
        raise RuntimeError("boom")

    _ctx(manager).register_chrome_renderer(broken)
    with caplog.at_level(logging.WARNING):
        assert manager.render_chrome("input_rule_top", 3, {}) is None
        assert manager.render_chrome("input_rule_top", 3, {}) is None
    assert calls == [1]
    assert "disabling it" in caplog.text


def test_prompt_toolkit_rule_control_renders_multiple_styles():
    from prompt_toolkit.layout.controls import FormattedTextControl

    manager = PluginManager(scope_key="/tmp/chrome-test")
    _ctx(manager).register_chrome_renderer(
        lambda surface, width, info: [("class:red", "x"), ("class:blue", "x"), ("class:green", " " * (width - 2))]
    )
    control = FormattedTextControl(lambda: manager.render_chrome("input_rule_top", 8, {}))
    line = control.create_content(8, 1).get_line(0)
    assert {style for style, _ in line} >= {"class:red", "class:blue"}


def test_renderer_fragments_are_normalized_to_cell_width():
    from prompt_toolkit.utils import get_cwidth

    result = _normalize_chrome_fragments([("a", "１２３"), ("b", "tail")], 4)
    assert sum(get_cwidth(text) for _, text in result) == 4
    assert result == [("a", "１２")]
