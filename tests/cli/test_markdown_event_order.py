"""Worker, Markdown, and Codex tool events preserve display order across output boundaries."""
import asyncio

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

from cli import HermesCLI
from hermes_cli.cli_markdown_stream import preview_window


def test_worker_markdown_boundary_and_tool_output_keep_event_order(monkeypatch, live_tui, codex_bridge):
    import cli as facade

    emitted = []
    # Capture only the terminal sink: _cprint and run_in_terminal retain their real
    # scheduling, so a queued tool line must compete with the next Markdown commit.
    monkeypatch.setattr(facade, '_pt_print', lambda value, **kwargs: emitted.append(
        fragment_list_to_text(to_formatted_text(value))))

    async def exercise():
        cli = HermesCLI.__new__(HermesCLI)
        cli.show_reasoning = False
        cli.final_response_markdown = 'render'
        cli.tool_progress_mode = 'all'
        cli._pending_tool_info = {}
        cli._last_scrollback_tool = None
        cli._turn_summary_record = lambda *args: None
        cli._reset_stream_state()
        async with live_tui(cli, preview_window=preview_window) as tui:
            cli._invalidate = lambda *args, **kwargs: tui.app.invalidate()
            bridge = codex_bridge(cli)

            def delta(text):
                bridge({'method': 'item/agentMessage/delta', 'params': {'delta': text}})

            def stream():
                # No sleeps between deltas/events: the callback's own handoff
                # must preserve order across stable-block and tool boundaries.
                for chunk in ['**First', ' block.**', '\n', '\n', 'Before ', 'tool.', '\n']:
                    delta(chunk)
                item = {'id': 'order-tool', 'type': 'commandExecution',
                        'command': 'cat event-order-note.md', 'aggregatedOutput': 'ok',
                        'exitCode': 0, 'durationMs': 1}
                for method in ['item/started', 'item/completed']:
                    bridge({'method': method, 'params': {'item': item}})
                for chunk in ['After ', 'tool.', '\n', '\n', 'Last ', 'block.', '\n']:
                    delta(chunk)
                cli._flush_stream()
                cli._flush_stream()

            await tui.run_worker(stream)
            tui.pipe.send_text('draft typed during streaming')
            for _ in range(100):
                if tui.editor.text == 'draft typed during streaming':
                    break
                await asyncio.sleep(.01)
            output = '\n'.join(emitted)
            markers = ['First block.', 'Before tool.', 'exec_comm', 'After tool.', 'Last block.']
            positions = [output.index(marker) for marker in markers]
            assert positions == sorted(positions)
            assert all(output.count(marker) == 1 for marker in markers)
            assert not cli._markdown_stream.pending
            assert tui.editor.text == 'draft typed during streaming'


@pytest.mark.parametrize('mode', ['render', 'raw', 'strip'])
@pytest.mark.parametrize('completed_only', [False, True])
def test_codex_tools_keep_assistant_messages_in_event_order(monkeypatch, completed_only, mode,
                                                           codex_bridge):
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = mode
    cli.show_timestamps = False
    cli.tool_progress_mode = 'all'
    cli._pending_tool_info = {}
    cli._last_scrollback_tool = None
    cli._turn_summary_record = lambda *args: None
    cli._invalidate = lambda *args, **kwargs: None
    cli._reset_stream_state()
    bridge = codex_bridge(cli)
    emitted.append('USER: Read my note')
    bridge({'method': 'item/agentMessage/delta', 'params': {'delta': "I'll find your note."}})
    item = {'id': 'tool-one', 'type': 'commandExecution', 'command': 'cat note.md',
            'aggregatedOutput': 'Today I wrote a regression test.', 'exitCode': 0, 'durationMs': 10}
    if not completed_only:
        bridge({'method': 'item/started', 'params': {'item': item}})
        assert "I'll find your note." in '\n'.join(emitted)
        assert cli._markdown_stream is None
    bridge({'method': 'item/completed', 'params': {'item': item}})
    bridge({'method': 'item/agentMessage/delta', 'params': {'delta': 'Here is your note.'}})
    cli._flush_stream()
    output = '\n'.join(emitted)
    assert output.index('USER:') < output.index("I'll find") < output.index('exec_comm') < output.index('Here is')
    assert output.count("I'll find your note.") == output.count('Here is your note.') == 1
    assert not any("I'll find" in entry and 'Here is' in entry for entry in emitted)
