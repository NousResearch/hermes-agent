"""Master output while the Ctrl+T monitor is open must wait for it to close.

The agent prints from a background thread through ``_cprint``, which paints through the app
prompt_toolkit reports as current: the monitor while it is open. Painting there erases the
monitor's alternate screen, prints the text on the main screen and redraws the monitor, so the
monitor flickers for every streamed chunk.
"""
import asyncio
import threading
from types import SimpleNamespace


def test_master_output_waits_for_the_monitor_to_close(monkeypatch):
    import cli as cli_module
    from hermes_cli import cli_render
    from hermes_cli import cli_subagent_monitor as monitor_module
    from prompt_toolkit import renderer as pt_renderer
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.output import DummyOutput
    from tests.waiting import async_wait_until

    painted = []
    monkeypatch.setattr(cli_module, "_pt_print", lambda text: painted.append(str(text)))
    monkeypatch.setattr(cli_module, "_pt_print_ansi", lambda text, raw="": painted.append(str(text)))
    monkeypatch.setattr(pt_renderer, "print_formatted_text",
                        lambda output, fragments, style: painted.append("".join(t for _, t, *_ in fragments)))
    monkeypatch.setattr(cli_render, "_HELD_PAINTS", None)

    cli = SimpleNamespace(agent=None, session_id=None, _session_db=None, _invalidate=lambda: None)
    monitor = monitor_module.SubagentMonitor(cli)
    cli._subagent_monitor = monitor
    erased = []

    async def run():
        with create_pipe_input() as main_input, create_pipe_input() as monitor_input:
            build = monitor_module.build_monitor_application

            def build_with_pipes(target):
                app = build(target, input=monitor_input, output=DummyOutput())
                real_erase = app.renderer.erase
                app.renderer.erase = lambda *a, **k: (erased.append(True), real_erase(*a, **k))
                return app

            monkeypatch.setattr(monitor_module, "build_monitor_application", build_with_pipes)
            main = Application(layout=Layout(Window()), input=main_input, output=DummyOutput())
            main_task = asyncio.create_task(main.run_async())
            await async_wait_until(lambda: main.is_running, timeout=3)

            monitor_module.open_monitor(cli)
            await async_wait_until(lambda: monitor.app is not None and monitor.app.is_running, timeout=3)

            for line in ("first line from the master", "second line from the master"):
                worker = threading.Thread(target=cli_render._cprint, args=(line,))
                worker.start()
                worker.join()
            # Each paint is scheduled onto the loop: wait until both were either held or painted.
            await async_wait_until(lambda: len(cli_render._HELD_PAINTS or []) + len(painted) >= 2, timeout=3)

            assert painted == []
            assert erased == []

            monitor.app.exit()
            await async_wait_until(lambda: not monitor.opening, timeout=3)
            await async_wait_until(lambda: len(painted) >= 2, timeout=3)
            main.exit()
            await main_task

    asyncio.run(run())

    assert [p.strip() for p in painted if "from the master" in p] == ["first line from the master",
                                                                      "second line from the master"]
    assert cli_render._HELD_PAINTS is None
