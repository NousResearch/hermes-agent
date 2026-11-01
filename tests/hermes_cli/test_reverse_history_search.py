"""Regression coverage for native reverse history search (issue #4065)."""

import asyncio


def test_ctrl_r_searches_the_composer_history(monkeypatch):
    from cli import HermesCLI
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import SearchToolbar

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = HermesCLI(
        model="fixture",
        provider="openai-compat",
        api_key="fixture",
        base_url="http://127.0.0.1:1/v1",
    )
    cli._tui_init_run_state()
    search_toolbar = SearchToolbar(ignore_case=True)
    editor = cli._tui_build_input_area(search_field=search_toolbar)
    editor.buffer.history.append_string("first command")
    editor.buffer.history.append_string("find this needle")

    async def run():
        with create_pipe_input() as pipe:
            app = Application(
                layout=Layout(HSplit([editor, search_toolbar])),
                key_bindings=cli._tui_build_key_bindings(),
                input=pipe,
                output=DummyOutput(),
            )
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                pipe.send_text("\x12needle\r")
                for _ in range(100):
                    if editor.text == "find this needle":
                        break
                    await asyncio.sleep(0.01)
                assert editor.text == "find this needle"
            finally:
                app.exit()
                await task

    asyncio.run(run())
