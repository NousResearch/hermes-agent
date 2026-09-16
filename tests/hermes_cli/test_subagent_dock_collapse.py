"""Collapsing the passive dock never takes space or focus from the editor."""
import asyncio
from types import SimpleNamespace


def test_collapsed_dock_reserves_one_shaded_row_without_changing_editor():
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import TextArea
    from hermes_cli import cli_subagent_monitor as dock
    from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides

    cli = SimpleNamespace(agent=None)
    dock.install_dock(cli)
    monitor = cli._subagent_monitor
    monitor.entries = [dict(subagent_id=str(i), goal='Worker 界 ' * 20,
        elapsed=12, last_tool='terminal', status='running') for i in range(6)]
    submitted = []
    editor = TextArea(height=1, multiline=False,
                      accept_handler=lambda buffer: submitted.append(buffer.text))
    output = DummyOutput()

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(HSplit([cli._subagent_dock_widget, editor]),
                focused_element=editor), input=pipe, output=output,
                style=Style.from_dict(get_prompt_toolkit_style_overrides()))
            cli._invalidate = app.invalidate
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            output.get_size = lambda: Size(rows=30, columns=100)
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                def height():
                    return app.renderer._last_screen.visible_windows_to_write_positions[editor.window].ypos

                preview_height = height()
                assert preview_height > 1
                editor.buffer.document = Document('keep this draft', 5)
                dock.toggle_dock(cli)
                for columns, rows in [(100, 30), (80, 20), (40, 14)]:
                    output.get_size = lambda: Size(rows=rows, columns=columns)
                    painted.clear()
                    app.invalidate()
                    await asyncio.wait_for(painted.wait(), 3)
                    assert height() == 1
                    assert app.layout.has_focus(editor)
                    assert (editor.text, editor.buffer.cursor_position) == ('keep this draft', 5)
                    screen = app.renderer._last_screen
                    for x in (0, columns // 2, columns - 1):
                        attrs = app._merged_style.get_attrs_for_style_str(screen.data_buffer[0][x].style)
                        assert attrs.bgcolor and attrs.color != attrs.bgcolor
                painted.clear()
                pipe.send_text('X\r')
                await asyncio.wait_for(painted.wait(), 3)
                assert submitted == ['keep Xthis draft']
                assert height() == 1
                output.get_size = lambda: Size(rows=30, columns=100)
                painted.clear()
                dock.toggle_dock(cli)
                await asyncio.wait_for(painted.wait(), 3)
                assert height() == preview_height
            finally:
                app.exit()
                await task
    asyncio.run(run())


def test_collapsed_summary_prioritizes_live_count_and_controls_at_small_widths():
    from prompt_toolkit.utils import get_cwidth
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    monitor = SubagentMonitor(SimpleNamespace())
    monitor.entries = [dict(goal='Inspect 界 ' * 40, elapsed=12,
        last_tool='terminal', status='running') for _ in range(6)]
    monitor.collapsed = True
    for width in (100, 80, 40, 24, 10, 1):
        text = monitor.dock_text(columns=width, rows=20)
        assert len(text.splitlines()) == 1
        assert get_cwidth(text) <= width
        if width >= 24:
            assert '6 live' in text and 'Ctrl+T' in text and 'F7' in text
        if width >= 80:
            assert 'last: terminal' in text and 'Ctrl+T expand' in text and 'F7 restore' in text
    monitor.entries.clear()
    assert monitor.dock_text(columns=80, rows=20) == ''
