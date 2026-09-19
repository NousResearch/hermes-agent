"""Behaviour contracts for Rich-markup input rules."""

import yaml
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth


def _skin_yaml(home, name, **fields):
    skins = home / "skins"
    skins.mkdir(exist_ok=True)
    (skins / f"{name}.yaml").write_text(
        yaml.safe_dump({"name": name, **fields}), encoding="utf-8")


def test_skin_loads_input_rule_art_and_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skin_yaml(tmp_path, "art", input_rule_art="[red]━[/red]")
    _skin_yaml(tmp_path, "plain")

    from hermes_cli.skin_engine import load_skin

    assert load_skin("art").input_rule_art
    assert load_skin("plain").input_rule_art == ""


def test_input_rule_fragments_tile_pad_and_clip_rich_styles():
    from hermes_cli.cli_input_rule import input_rule_fragments

    plain = input_rule_fragments("", 5)
    narrow = input_rule_fragments("[red]ab[/red][blue]c[/blue]", 5)
    wide = input_rule_fragments("[red]abcdef[/red]", 4)

    assert plain == [("class:input-rule", "─────")]
    assert sum(len(text) for _style, text in narrow) == 5
    assert len({style for style, _text in narrow}) > 1
    assert sum(len(text) for _style, text in wide) == 4


def test_malformed_input_rule_markup_falls_back_to_plain_rule(caplog):
    from hermes_cli.cli_input_rule import input_rule_fragments

    with caplog.at_level("DEBUG"):
        result = input_rule_fragments("[/oops]", 6)

    assert result == [("class:input-rule", "──────")]


def test_malformed_input_rule_markup_logs_once_after_parse_cache_eviction(caplog):
    from hermes_cli.cli_input_rule import _parsed_input_rule_art, input_rule_fragments

    _parsed_input_rule_art.cache_clear()
    with caplog.at_level("DEBUG"):
        input_rule_fragments("[/once]", 6)
        _parsed_input_rule_art.cache_clear()
        input_rule_fragments("[/once]", 6)

    records = [record for record in caplog.records if record.message.startswith("Malformed input_rule_art")]
    assert len(records) == 1


def test_input_rule_styles_are_prompt_toolkit_parseable():
    from hermes_cli.cli_input_rule import input_rule_fragments

    fragments = input_rule_fragments("[#20f6e8 on #101426 bold italic]X[/]", 1)
    for style, _text in fragments:
        style_without_class = style.removeprefix("class:input-rule ")
        attrs = Style.from_dict({"x": style_without_class}).get_attrs_for_style_str("class:x")
        assert attrs.bgcolor == "101426"
        assert attrs.color == "20f6e8"
        assert attrs.bold and attrs.italic


def test_input_rule_fragments_measure_columns_for_wide_glyphs():
    from hermes_cli.cli_input_rule import input_rule_fragments

    fragments = input_rule_fragments("[#fff]漢[/]", 9)
    assert sum(get_cwidth(char) for _style, text in fragments for char in text) == 9
    assert "─" in "".join(text for _style, text in fragments)


def test_escaped_rich_bracket_is_valid_input_rule_art():
    from hermes_cli.cli_input_rule import input_rule_fragments

    assert "[" in "".join(text for _style, text in input_rule_fragments(r"\[", 1))


def test_input_rule_strike_style_survives_conversion():
    from hermes_cli.cli_input_rule import input_rule_fragments

    style, text = input_rule_fragments("[strike]X[/]", 1)[0]
    assert text == "X"
    assert "strike" in style


def test_mixin_non_string_input_rule_art_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skin_yaml(tmp_path, "list-art", input_rule_art=["a", "b"])

    from hermes_cli import skin_engine
    from hermes_cli.cli_tui_mixin import CLITuiMixin

    skin_engine.set_active_skin("list-art")
    tui = object.__new__(CLITuiMixin)
    assert tui._tui_input_rule_fragments(5) == [("class:input-rule", "─────")]


def test_non_string_input_rule_art_falls_back_to_plain_rule():
    from hermes_cli.cli_input_rule import input_rule_fragments

    expected = [("class:input-rule", "─────")]
    for art in (["a", "b"], {"a": "b"}, 42, None):
        assert input_rule_fragments(art, 5) == expected


def test_multiline_input_rule_art_ignores_markup_after_first_line():
    from hermes_cli.cli_input_rule import input_rule_fragments

    assert input_rule_fragments("[red]x[/]\n[/bad]", 1) == [("class:input-rule fg:#800000", "x")]


def test_whitespace_only_input_rule_art_falls_back_to_plain_rule():
    from hermes_cli.cli_input_rule import input_rule_fragments

    assert input_rule_fragments("  \t", 4) == [("class:input-rule", "────")]


def test_non_integer_width_falls_back_to_empty_rule():
    from hermes_cli.cli_input_rule import input_rule_fragments

    for width in (True, 3.9, float("inf"), None, "3"):
        assert input_rule_fragments("[red]x[/]", width) == []  # a list prompt_toolkit can render


def test_input_rule_fragments_cache_result_is_unpoisonable():
    """A fresh list every call (prompt_toolkit renders only lists), and mutating
    a returned list must never corrupt the cache for later draws."""
    from hermes_cli.cli_input_rule import input_rule_fragments

    result = input_rule_fragments("[red]x[/red]", 3)
    assert isinstance(result, list)
    result.append(("evil", "Z"))
    assert input_rule_fragments("[red]x[/red]", 3) == [("class:input-rule fg:#800000", "xxx")]


def test_fragments_render_through_prompt_toolkit():
    """Construction is not rendering: FormattedTextControl accepted the old
    tuple, then the first render raised ValueError (Chef, 2026-09-10)."""
    from prompt_toolkit.formatted_text import to_formatted_text

    from hermes_cli.cli_input_rule import input_rule_fragments

    for art in ("[#968CFF on #0B0C10]━[/][#F238FF]─[/]", "", "   ", "[/bad]"):
        fragments = input_rule_fragments(art, 8)
        assert isinstance(fragments, list)
        rendered = to_formatted_text(fragments)
        assert sum(len(text) for _style, text in rendered) == 8

def test_input_rule_reads_active_skin_on_each_render(tmp_path, monkeypatch):
    """The active skin is reread for every draw."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skin_yaml(tmp_path, "red", input_rule_art="[red]R[/red]")
    _skin_yaml(tmp_path, "blue", input_rule_art="[blue]B[/blue]")

    from hermes_cli import skin_engine
    from hermes_cli.cli_tui_mixin import CLITuiMixin

    skin_engine.set_active_skin("red")
    tui = object.__new__(CLITuiMixin)
    first = tui._tui_input_rule_fragments(3)
    skin_engine.set_active_skin("blue")
    second = tui._tui_input_rule_fragments(3)

    assert first != second
    assert "R" in "".join(text for _style, text in first)
    assert "B" in "".join(text for _style, text in second)
