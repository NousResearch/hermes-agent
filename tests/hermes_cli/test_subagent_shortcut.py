"""The shared TUI shortcut routes without editing the classic composer."""
import asyncio


def test_monitor_shortcuts_preserve_draft_and_respect_modal_prompts(monkeypatch):
    from cli import HermesCLI
    from hermes_cli import cli_subagent_monitor as monitor
    from prompt_toolkit.application import Application
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setenv('HERMES_DEFER_AGENT_STARTUP', '1')
    cli = HermesCLI(model='fixture', provider='openai-compat', api_key='fixture',
                    base_url='http://127.0.0.1:1/v1')
    cli._tui_init_run_state()
    editor = cli._tui_build_input_area()
    cli._input_area = editor
    opened = []
    monkeypatch.setattr(monitor, 'open_monitor', lambda target: opened.append(target))

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
                              input=pipe, output=DummyOutput())
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                for key in ('\x14', '\x1b[17~'):
                    editor.buffer.document = Document('keep my draft', 5)
                    painted.clear()
                    pipe.send_text(key)
                    await asyncio.wait_for(painted.wait(), 3)
                    assert opened.pop() is cli
                    assert (editor.text, editor.buffer.cursor_position) == ('keep my draft', 5)
                cli._approval_state = {'options': []}
                bindings = app.key_bindings
                for key in ('c-t', 'f6'):
                    assert not any(b.filter() for b in bindings.get_bindings_for_keys((key,)))
            finally:
                cli._approval_state = None
                app.exit()
                await task
    asyncio.run(run())
