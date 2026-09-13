"""Final Rich output and streamed blocks share Markdown semantics at token edges."""
from io import StringIO

import pytest
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from cli import HermesCLI, _render_final_assistant_content
from hermes_cli.cli_markdown_stream import assistant_label, render_markdown


CASES = [
    'Pipe prose: alpha | beta.\n\nNext paragraph.\n',
    'Name | Value\n--- | ---\nalpha | beta\n\nAfter table.\n',
    '~~~python\nprint("alpha")\n~~~\n\nAfter code.\n',
    '````text\n```literal```\n````\n\nAfter code.\n',
    'Setext heading\n==============\n\nBody.\n',
    '> quoted\nlazy continuation\n> > nested quote\n\nAfter quote.\n',
    'Read [the docs][docs].\n\nAnother paragraph.\n\n[docs]: https://example.com\n',
    'Exactly one trailing newline.\n',
]


@pytest.mark.parametrize('source', CASES)
@pytest.mark.parametrize('width', [20, 80])
def test_stream_and_final_renderable_preserve_same_content(monkeypatch, source, width):
    import cli as facade
    monkeypatch.setenv('NO_COLOR', '1')
    monkeypatch.setattr(facade, '_terminal_columns', lambda: width)
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = 'render'
    cli._reset_stream_state()
    for character in source:
        cli._stream_delta(character)
    cli._flush_stream()
    streamed = Text.from_ansi('\n'.join(emitted)).plain
    buf = StringIO()
    Console(file=buf, width=width, height=25, color_system=None).print(
        _render_final_assistant_content(source, width=width, terminal_wrap=True), crop=False)
    label = Text.from_ansi(assistant_label(width, color=False)).plain.split()
    assert streamed.split() == label + buf.getvalue().split()
    assert label + render_markdown(source, width, color=False, terminal_wrap=True).split() == streamed.split()
    before = list(emitted)
    cli._flush_stream()
    assert emitted == before


def test_final_assistant_content_uses_markdown_renderable():
    renderable = _render_final_assistant_content("# Title\n\n- one\n- two")

    assert isinstance(renderable, Markdown)
    output = _render_to_text(renderable)
    assert all(value in output for value in ("Title", "one", "two"))


def test_final_assistant_content_keeps_non_path_markdown_escapes():
    output = _render_to_text(_render_final_assistant_content(r"1\. Not an ordered list"))
    assert "1. Not an ordered list" in output
    assert r"1\." not in output


def test_strip_mode_preserves_lists_and_blockquotes():
    output = _render_to_text(_render_final_assistant_content(
        "**Formatting**\n- Ran prettier\n- Files changed\n\n> quoted text",
        mode="strip"))
    assert "- Ran prettier" in output
    assert "- Files changed" in output
    assert "> quoted text" in output
    assert "**" not in output


def test_strip_mode_preserves_cron_asterisks_and_identifiers():
    output = _render_to_text(_render_final_assistant_content(
        "* * * * *\n\nLet test_case stay\n\n* * *", mode="strip"))
    assert "* * * * *" in output
    assert "test_case" in output
    assert "* * *" not in output.splitlines()


def test_strip_mode_still_strips_boundary_underscore_emphasis():
    output = _render_to_text(_render_final_assistant_content(
        "say _hi_ and __bold__ now", mode="strip"))
    assert "say hi and bold now" in output


def _render_to_text(renderable) -> str:
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()
