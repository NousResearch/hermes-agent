"""Final answers must retain list content beyond the terminal's right edge."""
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console


def _render(monkeypatch, text, width, mode, modern=True):
    import cli

    output = StringIO()
    console = Console(file=output, width=width, color_system=None)
    monkeypatch.setattr(cli, "ChatConsole", lambda: console)
    monkeypatch.setattr(cli, "_cprint", lambda *_: None)
    monkeypatch.setattr(cli, "_terminal_columns", lambda: width)
    stub = SimpleNamespace(
        final_response_markdown=mode, _last_turn_interrupted=False,
        _streamed_text_this_turn="", _stream_started=False,
        _stream_box_opened=False, _modern_layout=modern,
        _scrollback_box_width=lambda: width,
    )
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False, result={})
    cli.HermesCLI._chat_print_response_panel(stub, turn, text)
    return output.getvalue()


@pytest.mark.parametrize("width", [40, 80, 124, 252])
@pytest.mark.parametrize("modern", [True, False])
def test_markdown_lists_preserve_every_word_and_url(monkeypatch, width, modern):
    words = " ".join(f"word{i}" for i in range(90))
    url = "https://example.com/" + "long-path-segment/" * 24 + "final-endpoint"
    text = f"- {words}\n- Source: {url}\n\n1. {words}\n\n> {words}"
    rendered = _render(monkeypatch, text, width, "render", modern)
    compact = "".join(rendered.replace("▌", "").split())
    assert compact.count("".join(words.split())) == 3
    assert url in compact
    assert all(len(line) <= width for line in rendered.splitlines())


@pytest.mark.parametrize("mode", ["raw", "strip", " RAW "])
def test_plain_answers_keep_terminal_reflow(monkeypatch, mode):
    text = "plain words " * 80 + "ENDMARKER"
    narrow = _render(monkeypatch, text, 40, mode)
    wide = _render(monkeypatch, text, 124, mode)
    assert narrow == wide
    assert text in narrow
