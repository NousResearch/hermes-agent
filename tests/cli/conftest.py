"""Shared fixtures for CLI tests.

prompt_toolkit / capsys isolation
---------------------------------
``cli._cprint`` renders through ``prompt_toolkit.print_formatted_text``,
which — when called with no explicit ``output=`` — lazily creates an
``Output`` from ``sys.stdout`` **and caches it on the process-global default
``AppSession``** (``prompt_toolkit.application.current._current_app_session``,
a ``ContextVar`` with a module-level default). The cache is keyed to nothing
and never re-reads ``sys.stdout``.

Under pytest, ``capsys`` swaps ``sys.stdout`` for a fresh buffer per test.
So the first CLI test that emits through ``_cprint`` (e.g. one exercising
``/queue``, which prints a "Queued: …" line) locks prompt_toolkit's cached
output onto *its* captured stdout. Every later ``capsys`` test that asserts
on ``_cprint`` output then reads an empty buffer, because the render went to
the first test's now-dead capture target. That is the mechanism behind the
order-dependent ``test_resume_quiet_stderr`` failure: it passes in isolation
and in its own file, but fails in a full ``tests/cli`` run.

Reset the cached output before every CLI test so each one re-creates a fresh
prompt_toolkit ``Output`` bound to its own ``sys.stdout`` on first use. This
is a no-op when prompt_toolkit isn't importable and cheap otherwise (the
property re-creates lazily).
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Thread
from types import SimpleNamespace

import pytest


@dataclass
class LiveTui:
    """Reusable real prompt-toolkit application harness for worker-stream tests."""

    app: object
    pipe: object
    editor: object

    async def run_worker(self, callback, *args) -> None:
        errors = []

        def run():
            try:
                callback(*args)
            except BaseException as error:  # propagate worker failures to the test loop
                errors.append(error)

        worker = Thread(target=run, daemon=True)
        worker.start()
        while worker.is_alive():
            await asyncio.sleep(.01)
        if errors:
            raise errors[0]


@pytest.fixture
def live_tui():
    @asynccontextmanager
    async def start(cli, *, preview_window):
        from prompt_toolkit.application import Application
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.layout import HSplit, Layout
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.widgets import TextArea

        with create_pipe_input() as pipe:
            editor = TextArea(prompt='> ')
            app = Application(layout=Layout(HSplit([preview_window(cli), editor]),
                                             focused_element=editor),
                              input=pipe, output=DummyOutput())
            cli._app = app
            task = asyncio.create_task(app.run_async(set_exception_handler=False))
            try:
                while not app.is_running:
                    if task.done():
                        await task
                    await asyncio.sleep(0)
                yield LiveTui(app, pipe, editor)
            finally:
                if app.is_running:
                    app.exit()
                await task

    return start


@pytest.fixture
def codex_bridge():
    """Build the Codex event adapter against a test CLI's real callbacks."""
    from agent.codex_runtime import make_codex_app_server_event_bridge

    def build(cli):
        return make_codex_app_server_event_bridge(SimpleNamespace(
            _fire_stream_delta=cli._stream_delta,
            tool_progress_callback=cli._on_tool_progress,
        ))

    return build


@pytest.fixture(autouse=True)
def _reset_prompt_toolkit_output_cache():
    """Clear prompt_toolkit's cached AppSession output around each CLI test.

    See the module docstring for the capsys/prompt_toolkit interaction this
    guards against.
    """

    def _clear() -> None:
        try:
            from prompt_toolkit.application.current import get_app_session

            get_app_session()._output = None
        except Exception:
            # prompt_toolkit not importable / internal shape changed — the
            # tests that rely on this simply keep their prior behavior.
            pass

    _clear()
    yield
    _clear()
