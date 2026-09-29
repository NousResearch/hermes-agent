"""A passive dock paints a bounded surface without acquiring the editor."""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('columns,rows,skin', [(100, 30, 'default'), (80, 20, 'daylight')])
def test_passive_dock_fills_rows_but_keeps_input_live(monkeypatch, columns, rows, skin):
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import TextArea
    from hermes_cli.cli_subagent_monitor import install_dock
    from hermes_cli import skin_engine

    monkeypatch.setattr(skin_engine, '_active_skin', skin_engine.load_skin(skin))
    cli = SimpleNamespace(agent=None)
    install_dock(cli)
    cli._subagent_monitor.entries = [dict(goal=f'Worker {i}', elapsed=12,
        last_tool='terminal', status='running') for i in range(6)]
    submitted = []
    editor = TextArea(height=1, multiline=False,
                      accept_handler=lambda buffer: submitted.append(buffer.text))
    output = DummyOutput()
    output.get_size = lambda: Size(rows=rows, columns=columns)

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(HSplit([cli._subagent_dock_widget, editor]),
                focused_element=editor), input=pipe, output=output,
                style=Style.from_dict(skin_engine.get_prompt_toolkit_style_overrides()))
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                screen = app.renderer._last_screen
                dock_height = screen.visible_windows_to_write_positions[editor.window].ypos
                assert 2 <= dock_height <= 6
                for y in range(dock_height):
                    assert screen.data_buffer[y][0].char == ' '
                    for x in (0, columns // 2, columns - 1):
                        attrs = app._merged_style.get_attrs_for_style_str(screen.data_buffer[y][x].style)
                        assert attrs.bgcolor, 'the entire dock row needs a background'
                        assert attrs.color != attrs.bgcolor
                painted.clear()
                pipe.send_text('Follow up while workers run\r')
                await asyncio.wait_for(painted.wait(), 3)
                assert submitted == ['Follow up while workers run']
                assert app.layout.has_focus(editor)
            finally:
                app.exit()
                await task
    asyncio.run(run())


def test_expanded_roster_reserves_activity_before_long_task_names():
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.utils import get_cwidth
    from hermes_cli.cli_subagent_monitor import SubagentMonitor, build_monitor_application

    monitor = SubagentMonitor(SimpleNamespace())
    monitor.entries = [dict(subagent_id=str(i), goal='Inspect 界 ' * 30,
        elapsed=24, status='running', last_tool='terminal') for i in range(6)]
    monitor.selected_id = '0'
    output = DummyOutput()
    with create_pipe_input() as pipe:
        for columns, rows in [(100, 30), (80, 20)]:
            output.get_size = lambda: Size(rows=rows, columns=columns)
            app = build_monitor_application(monitor, input=pipe, output=output)
            fragments = app.layout.current_control.text()
            assert len(fragments) == len(monitor.entries)
            for _, text in fragments:
                assert '24s' in text and 'running' in text and 'last: terminal' in text
                assert get_cwidth(text.rstrip('\n')) == columns
