"""`/new` after `hermes -w` opens the new session in the worktree.

`hermes -w` selects the worktree through TERMINAL_CWD without a chdir, and /new creates its gateway
session from that folder (``gateway_session_store._launch_cwd``). Answering /new's confirmation must
leave the selection alone. The command runs on a worker thread, as the REPL's process_loop dispatches
it, against a real prompt_toolkit Application whose keys arrive through a pipe; only the gateway
session rotation is replaced, by a recorder of the folder it would send.
"""

import asyncio
import os
import sys

import pytest

from tests.waiting import async_wait_until


@pytest.fixture
def cli(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    return cli


def _drive(cli, command, keys):
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    editor = cli._tui_build_input_area()
    cli._input_area = editor

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
                              input=pipe, output=DummyOutput())
            cli._app = app
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            worker = asyncio.create_task(asyncio.to_thread(cli.process_command, command))
            try:
                await async_wait_until(lambda: cli._slash_confirm_state is not None or worker.done(),
                                       timeout=5, interval=0.01)
                pipe.send_text(keys)
                await asyncio.wait_for(worker, 5)
            finally:
                app.exit()
                await task
    asyncio.run(run())


def test_new_after_worktree_launch_creates_the_session_in_the_worktree(cli, tmp_path, monkeypatch):
    import cli as cli_module
    from hermes_cli.gateway_session_store import _launch_cwd

    checkout, worktree = tmp_path / "checkout", tmp_path / "checkout-wt"
    checkout.mkdir()
    worktree.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    # What `hermes -w` publishes (cli._start_worktree_setup) in a CLI that has not run a turn yet.
    monkeypatch.setattr(cli_module, "_active_worktree", {"path": str(worktree)})
    monkeypatch.setenv("TERMINAL_CWD", str(worktree))
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)
    created_in = []
    monkeypatch.setattr(cli, "new_session", lambda **kw: created_in.append(_launch_cwd()))

    _drive(cli, "/new", "1")  # Approve once

    assert created_in == [os.path.realpath(worktree)]
    assert os.environ["TERMINAL_CWD"] == str(worktree)
