"""Classic CLI ``/btw`` side exploration (hermes_cli/cli_side_thread_mixin.py).

The CLI forks its own live agent for ``/btw --explore``, prints the thread's steps while the main
turn keeps running, and only ever reaches the main conversation by queueing a promotion as the
next normal user turn. A plain ``/btw <question>`` keeps its tool-less snapshot path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import side_thread


@pytest.fixture(autouse=True)
def _empty_registry():
    side_thread._registry.clear()
    yield
    side_thread._registry.clear()


class _Worker:
    def __init__(self, produce, kwargs):
        self.produce, self.kwargs = produce, kwargs

    def start(self):
        self.body = self.produce()


def _cli(agent=object()):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = agent
    cli.session_id = "sess-cli"
    cli.conversation_history = [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "ok"}]
    cli.workers, cli.queued, cli.printed = [], [], []

    def side_worker(produce, **kwargs):
        cli.workers.append(_Worker(produce, kwargs))
        return cli.workers[-1]

    cli._side_worker = side_worker
    cli._queue_prompt_turn = lambda msg, command, ack="": cli.queued.append((msg, command, ack))
    return cli


def _run(cli, command):
    with patch("hermes_cli.gateway_side.connection_for", return_value=None), \
         patch("hermes_cli.cli_side_thread_mixin._cp", side_effect=lambda *lines: cli.printed.extend(lines)):
        cli._handle_btw_command(command)
        for worker in cli.workers:
            if not hasattr(worker, "body"):
                worker.start()


def _fake_run(report="retry lives in net.py:40", steps=("terminal · rg -n retry",)):
    def run(thread, parent, history, *, on_progress=None):
        for step in steps:
            thread.tool_calls.append(step.split(" ")[0])
            on_progress(step)
        thread._finish("done", answer=report)
        return report
    return run


def test_explore_forks_the_live_agent_and_streams_its_steps():
    cli = _cli()
    seen = {}

    def run(thread, parent, history, *, on_progress=None):
        seen.update(parent=parent, history=history)
        return _fake_run()(thread, parent, history, on_progress=on_progress)

    with patch("agent.side_thread.run_side_exploration", side_effect=run), \
         patch("agent.side_thread.explore_refusal", return_value=None):
        _run(cli, "/btw --explore find the retry logic")

    (thread,) = side_thread.list_threads("sess-cli")
    assert seen["parent"] is cli.agent
    assert seen["history"] == cli.conversation_history and seen["history"] is not cli.conversation_history
    assert any("rg -n retry" in line and thread.thread_id in line for line in cli.printed)
    assert "retry lives in net.py:40" in cli.workers[0].body
    assert cli.queued == []  # nothing reaches the main conversation by itself


def test_promote_queues_the_report_as_the_next_turn():
    cli = _cli()
    with patch("agent.side_thread.run_side_exploration", side_effect=_fake_run()), \
         patch("agent.side_thread.explore_refusal", return_value=None):
        _run(cli, "/btw --explore find the retry logic")
    (thread,) = side_thread.list_threads("sess-cli")
    _run(cli, "/btw --promote now fix it")
    ((prompt, command, ack),) = cli.queued
    assert prompt == side_thread.build_promotion_prompt(thread, "now fix it")
    assert command == "/btw --promote" and thread.thread_id in ack


def test_explore_without_a_live_agent_refuses_at_once():
    cli = _cli(agent=None)
    _run(cli, "/btw --explore look around")
    assert side_thread.list_threads("sess-cli") == []
    assert cli.workers == []
    assert any("send a message first" in line.lower() for line in cli.printed)


def test_plain_btw_still_answers_without_tools():
    cli = _cli()
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda q: {"model": "m", "runtime": {}}
    with patch("agent.side_question.answer_side_question", return_value="foo.py") as ask, \
         patch("hermes_cli.cli_side_thread_mixin._gt", return_value="started"), \
         patch("agent.side_thread.run_side_exploration") as explore:
        _run(cli, "/btw which file was it?")
    explore.assert_not_called()
    assert ask.call_args.args[0] == "which file was it?"
    assert cli.workers[0].body == "foo.py"


def test_open_shows_the_desktop_only_line_in_the_cli():
    cli = _cli()
    thread = side_thread.register_thread("sess-cli", "t")
    thread._finish("done", answer="report")
    _run(cli, "/btw --open")
    assert cli.queued == []
    assert any("Desktop" in line for line in cli.printed)
