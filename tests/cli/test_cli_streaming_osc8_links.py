def test_streaming_markdown_links_emit_osc8_raw_terminal(monkeypatch):
    import cli

    raw_lines = []
    plain_lines = []
    monkeypatch.setattr(cli, "_cprint_raw_terminal", raw_lines.append)
    monkeypatch.setattr(cli, "_cprint", plain_lines.append)
    monkeypatch.setattr(cli.HermesCLI, "_scrollback_box_width", lambda self: 80)

    app = cli.HermesCLI.__new__(cli.HermesCLI)
    app.show_reasoning = False
    app.show_timestamps = False
    app.final_response_markdown = "render"
    app._reset_stream_state()

    app._emit_stream_text("[Miil Máslo 82% — 250 g](https://www.rohlik.cz/1462148-miil-maslo-82)\n")

    rendered = "\n".join(raw_lines)
    assert "\x1b]8;" in rendered
    assert "https://www.rohlik.cz/1462148-miil-maslo-82" in rendered
    assert "Miil Máslo 82% — 250 g" in rendered
    assert not any("[Miil Máslo" in line and "](https://" in line for line in raw_lines + plain_lines)
