from rich.markdown import Markdown


def test_chat_console_preserves_rich_markdown_osc8_links(monkeypatch):
    import cli

    raw_lines = []
    plain_lines = []
    monkeypatch.setattr(cli, "_cprint_raw_terminal", raw_lines.append, raising=False)
    monkeypatch.setattr(cli, "_cprint", plain_lines.append)

    cli.ChatConsole().print(Markdown("[Miil Máslo 82% — 250 g](https://www.rohlik.cz/1462148-miil-maslo-82)"))

    assert raw_lines, "Markdown link output should be routed through raw terminal printing"
    rendered = "\n".join(raw_lines)
    assert "\x1b]8;" in rendered
    assert "https://www.rohlik.cz/1462148-miil-maslo-82" in rendered
    assert "Miil Máslo 82%" in rendered
    assert plain_lines == []
