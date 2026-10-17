"""A CLI exit and the run's own boundary end an interrupted run exactly once between them.

The interactive CLI's agent thread is a daemon: an exit cannot wait for it, so the exit fires
``on_session_end`` for a run still in flight. Whichever of the two claims the run first fires it;
the other stays silent (``agent.turn_finalizer.claim_session_end``).
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent import test_session_end_boundary as _boundary

# Reuse the boundary tests' fixtures by name (an import would shadow-redefine them — F811).
agent, ready, session_ends = _boundary.agent, _boundary.ready, _boundary.session_ends
_mock_plugin_discovery = _boundary._mock_plugin_discovery


@pytest.fixture()
def cli_mod():
    import cli as cli_mod
    cli_mod._cleanup_done = False
    cli_mod._handed_off_session_ids.clear()
    yield cli_mod
    cli_mod._active_agent_ref = None
    cli_mod._cleanup_done = False


def _shutdown_mid_turn(cli_mod, agent):
    cli = cli_mod.HermesCLI()
    cli.agent, cli._agent_running, cli._session_db = agent, True, None
    with (
        patch.object(cli_mod, "_run_cleanup"),
        patch.object(cli, "_print_exit_summary"),
        patch.object(cli, "_release_active_session"),
    ):
        cli._tui_shutdown()


def test_exit_ends_a_run_stuck_in_its_provider_call_and_the_run_stays_silent(ready, session_ends, cli_mod):
    in_call, release = threading.Event(), threading.Event()

    def _stuck_call(_api_kwargs):  # a provider call that ignores the interrupt until released
        in_call.set()
        assert release.wait(30)
        raise InterruptedError("provider call abandoned")

    ready._interruptible_api_call = _stuck_call
    run = threading.Thread(target=ready.run_conversation, args=("hello",), daemon=True)
    run.start()
    assert in_call.wait(30)

    _shutdown_mid_turn(cli_mod, ready)
    release.set()
    run.join(30)

    assert not run.is_alive()
    [end] = session_ends
    assert (end["completed"], end["interrupted"], end["reason"]) == (False, True, "shutdown")


def test_exit_after_the_run_ended_itself_fires_nothing_more(ready, session_ends, cli_mod):
    ready.client.chat.completions.create.side_effect = InterruptedError("interrupted")
    ready.run_conversation("hello", task_id="task-1")
    # The process loop has not yet cleared ``_agent_running`` when the exit handler runs.
    _shutdown_mid_turn(cli_mod, ready)

    [end] = session_ends
    assert end["task_id"] == "task-1"
    assert "reason" not in end


def test_one_shot_ctrl_c_ends_the_run_once(ready, session_ends, cli_mod):
    """``hermes -z`` catches the KeyboardInterrupt the run raised; the run already ended itself."""

    def _ctrl_c(_api_kwargs):
        raise KeyboardInterrupt

    ready._interruptible_api_call = _ctrl_c
    with pytest.raises(KeyboardInterrupt):
        ready.run_conversation("hello", task_id="task-ctrl-c")
    cli_mod._interrupt_single_query_agent(SimpleNamespace(agent=ready, session_id=ready.session_id))

    [end] = session_ends
    assert end["task_id"] == "task-ctrl-c"
    assert end["turn_id"] == ready._current_turn_id
    assert (end["completed"], end["failed"], end["interrupted"]) == (False, False, True)


def test_exit_without_an_open_run_fires_nothing(session_ends, cli_mod):
    idle = MagicMock(spec=["session_id", "model", "platform"], session_id="s", model="m", platform="cli")
    cli_mod._invoke_interrupted_session_end(idle, "s", "shutdown")
    assert session_ends == []
