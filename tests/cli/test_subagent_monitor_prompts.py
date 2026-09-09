"""Blocking prompts reclaim input from the nested monitor without a keypress."""
import asyncio
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('paint_source', ['modal', 'tick'])
def test_prompt_paint_yields_monitor_but_ordinary_paint_does_not(monkeypatch, paint_source):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor, build_monitor_application, install_dock
    from hermes_cli.cli_terminal_mixin import CLITerminalMixin
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    async def run():
        for name in ('_approval_state', '_clarify_state', '_secret_state', '_sudo_state',
                     '_slash_confirm_state'):
            dock_cli = SimpleNamespace(agent=None)
            install_dock(dock_cli)
            dock_cli._subagent_monitor.entries = [{}]
            assert dock_cli._subagent_dock_widget.filter()
            setattr(dock_cli, name, {'pending': True})
            assert not dock_cli._subagent_dock_widget.filter(), 'dock crowds the blocking prompt'
            cli = SimpleNamespace(_app=None)
            monitor = SubagentMonitor(cli)
            cli._subagent_monitor = monitor
            with create_pipe_input() as pipe:
                app = build_monitor_application(monitor, input=pipe, output=DummyOutput())
                monitor.app = app
                rendered = asyncio.Event()
                app.after_render += lambda app: rendered.set()
                task = asyncio.create_task(app.run_async())
                await asyncio.wait_for(rendered.wait(), 3)
                try:
                    rendered.clear()
                    await asyncio.to_thread(CLITerminalMixin._paint_now, cli)
                    await asyncio.wait_for(rendered.wait(), 3)
                    assert not task.done(), 'ordinary paints must not dismiss the monitor'
                    loop = asyncio.get_running_loop()
                    ui_thread = threading.get_ident()
                    stopped = threading.Event()
                    task.add_done_callback(lambda _: stopped.set())
                    schedule = loop.call_soon_threadsafe

                    def schedule_with_teardown(callback, *args, **kwargs):
                        handle = schedule(callback, *args, **kwargs)
                        if (callback == app.on_invalidate.fire
                                and threading.get_ident() != ui_thread):
                            # Let an already-queued frame see the modal and finish
                            # teardown while the worker is still in invalidate().
                            schedule(app._redraw)
                            assert stopped.wait(3), 'monitor did not finish teardown'
                        return handle

                    setattr(cli, name, {'pending': True})
                    with monkeypatch.context() as patch:
                        patch.setattr(loop, 'call_soon_threadsafe', schedule_with_teardown)
                        if paint_source == 'tick':
                            patch.setattr(monitor, 'refresh', lambda: True)
                            await asyncio.to_thread(monitor.tick)
                        else:
                            await asyncio.to_thread(CLITerminalMixin._paint_now, cli)
                    done, _ = await asyncio.wait({task}, timeout=1)
                    assert task in done, f'{name} remained hidden behind the monitor'
                    await task
                finally:
                    if not task.done():
                        app.exit()
                        await task
    asyncio.run(run())


def test_secret_callback_yields_monitor_and_restores_composer():
    from hermes_cli.cli_subagent_monitor import SubagentMonitor, build_monitor_application
    from hermes_cli.cli_terminal_mixin import CLITerminalMixin
    from hermes_cli.cli_modal_mixin import CLIModalMixin
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    class CLI(CLIModalMixin, CLITerminalMixin):
        def _ring_bell(self, **kwargs):
            pass

    async def run():
        cli = CLI()
        draft = Document('keep my draft', 4)
        buffer = Buffer(document=draft)
        cli._app = SimpleNamespace(current_buffer=buffer, invalidate=lambda: None)
        cli._modal_input_snapshot = None
        monitor = cli._subagent_monitor = SubagentMonitor(cli)
        with create_pipe_input() as pipe:
            monitor.app = build_monitor_application(monitor, input=pipe, output=DummyOutput())
            rendered = asyncio.Event()
            monitor.app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(monitor.app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)
            callback = asyncio.create_task(asyncio.to_thread(
                cli._secret_capture_callback, 'FIXTURE_SECRET', 'Owned fixture'))
            try:
                done, _ = await asyncio.wait({task}, timeout=1)
                assert task in done, 'secret arrival did not immediately reclaim input'
                assert buffer.text == '', 'draft must not be submitted as a secret'
                cli._submit_secret_response('')
                result = await asyncio.wait_for(callback, 3)
                assert result['skipped']
                assert buffer.document == draft
            finally:
                if not task.done():
                    monitor.app.exit()
                await task
                if not callback.done():
                    cli._submit_secret_response('')
                    await callback
    asyncio.run(run())
