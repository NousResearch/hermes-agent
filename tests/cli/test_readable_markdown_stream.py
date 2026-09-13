"""The real CLI callback keeps unfinished Markdown visible and commits content once."""
import asyncio
import pytest
from rich.text import Text

from cli import HermesCLI
from hermes_cli.cli_markdown_stream import preview_window, render_markdown


def make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = 'render'
    cli._reset_stream_state()
    return cli


def test_incremental_markdown_preserves_blocks_and_flushes_once(monkeypatch):
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    cli = make_cli()
    source = '# Heading\n\n**Readable** prose.\n\n- item\n  - nested\n\n```python\ndef greet():\n    return 42\n```\n\n| Name | Value |\n| --- | --- |\n| Test | 42 |\n'
    for char in source:
        cli._stream_delta(char)
    stream = cli._markdown_stream
    assert 'Test' in '\n'.join(stream.preview(40))
    cli._flush_stream()
    output = '\n'.join(emitted)
    for word in ['Heading', 'Readable', 'nested', 'def greet():', 'return 42', 'Test']:
        assert output.count(word) == 1
    assert '\n    return 42' in output
    assert '**Readable**' not in output
    before = list(emitted)
    cli._flush_stream()
    assert emitted == before
    assert not stream.preview(40)
    assert "https://example.com" in render_markdown("[Example](https://example.com)", 80)
    paragraph = "Copy this paragraph without inserted line breaks. " * 8
    assert "\n" not in render_markdown(paragraph, 20, color=False, terminal_wrap=True)
    for width in [20, 40, 80]:
        lines = render_markdown(source, width, color=False).splitlines()
        assert all(Text.from_ansi(line).cell_len <= width for line in lines)


def test_live_worker_preview_input_resize_and_interruption(monkeypatch, live_tui):
    import cli as facade
    committed = []
    monkeypatch.setattr(facade, '_pt_print_ansi', committed.append)

    async def exercise():
        cli = make_cli()
        async with live_tui(cli, preview_window=preview_window) as tui:
            await tui.run_worker(cli._stream_delta, '**Visible before newline**')
            assert 'Visible before newline' in '\n'.join(cli._markdown_stream.preview(40))
            assert not committed
            tui.pipe.send_text('draft stays here')
            for _ in range(100):
                if tui.editor.text == 'draft stays here':
                    break
                await asyncio.sleep(.01)
            assert tui.editor.text == 'draft stays here'
            await tui.run_worker(cli._stream_delta, '\n\n```python\n' + 'print(42)\n' * 30)
            for width in [20, 90, 40]:
                content = tui.app.layout.container.children[0].content.create_content(width, 8)
                assert content.line_count > 8
                assert content.cursor_position.y == content.line_count - 1
                assert all(Text.from_ansi(line).cell_len <= width
                           for line in cli._markdown_stream.preview(width))
            cli._approval_state = {"command": "example"}
            assert tui.app.layout.container.children[0].height().max == 0
            cli._approval_state = None
            # The same flush used by interruption/tools must retain an unfinished fence.
            await tui.run_worker(cli._flush_stream)
            assert not cli._markdown_stream.pending
            assert '\n'.join(committed).count('Visible before newline') == 1
            assert '\n'.join(committed).count('print') == 30
            assert tui.editor.text == 'draft stays here'
    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


def test_large_pending_block_has_bounded_preview_and_complete_flush(monkeypatch):
    import hermes_cli.cli_markdown_stream as renderer
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, "_cprint", emitted.append)
    cli = make_cli()
    source = "```python\n" + "print(42)\n" * 1200
    cli._stream_delta(source)
    original = renderer.render_markdown
    def bounded(source, *args, **kwargs):
        assert len(source) <= 8192
        return original(source, *args, **kwargs)
    monkeypatch.setattr(renderer, "render_markdown", bounded)
    for width in (20, 80):
        preview = cli._markdown_stream.preview(width)
        assert "print(42)" in "\n".join(preview)
        assert all(Text.from_ansi(line).cell_len <= width for line in preview)
    monkeypatch.setattr(renderer, "render_markdown", original)
    cli._flush_stream()
    assert "\n".join(emitted).count("print(42)") == 1200


@pytest.mark.parametrize('source', [
    '[docs]: https://example.com\n\nIntro.\n\nRead [the docs][docs].\n',
    'Read [the docs][docs].\n\nAnother paragraph.\n\n[docs]: https://example.com\n',
    'Read [docs].\n\nAnother paragraph.\n\n[docs]: https://example.com\n',
])
def test_stream_reference_links_keep_document_context(monkeypatch, source):
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    cli = make_cli()
    for character in source:
        cli._stream_delta(character)
    cli._flush_stream()
    output = '\n'.join(emitted)
    assert output.count('https://example.com') == 1
    assert '[docs]' not in output
    assert output.count('Read') == 1
    cli._flush_stream()
    assert '\n'.join(emitted) == output


def test_narrow_table_preserves_values_in_preview_final_and_replay(monkeypatch):
    import cli as facade
    cells = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta']
    source = ('| A | B | C | D | E | F | G | H |\n'
              '| --- | --- | --- | --- | --- | --- | --- | --- |\n'
              '| ' + ' | '.join(cells) + ' |\n')
    emitted, history = [], []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    monkeypatch.setattr(facade, '_record_output_history_entry', history.append)
    monkeypatch.setattr(facade, '_terminal_columns', lambda: 20)
    cli = make_cli()
    cli._stream_delta(source)
    preview = '\n'.join(cli._markdown_stream.preview(20))
    cli._flush_stream()
    for output in [preview, '\n'.join(emitted), '\n'.join(history[0]()),
                   render_markdown(source, 20, color=False)]:
        assert all(value in output for value in cells)
