"""An interrupted turn that outlives chat()'s post-interrupt wait must stop before the next turn starts.

Context compression ignores ordinary interrupts, so a message typed during a long summary call
leaves the worker alive after the wait and chat() stops watching it. The re-queued message used to
start a second ``run_conversation`` on the same agent right away. That turn waited on the session
turn lease, its lease wait aborted on the old turn's interrupt flag and cleared it, and the old turn
then ran its whole task while the user's message waited behind the lease.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from unittest.mock import patch

from tests.hermes_cli.test_cli_interrupt_ack_race import _make_cli

COMPRESSED = [{"role": "user", "content": "[summary of 80 earlier messages]"},
              {"role": "assistant", "content": "read notes.txt"}]


class _CompressingAgent:
    """First turn sits in interrupt-shielded compression until ``compression_done`` is set, then
    stops if the interrupt flag is still set (the loop's next check) or keeps working if it is not."""

    def __init__(self, session_id, rotated_session_id=None):
        self.session_id = session_id
        self.rotated_session_id = rotated_session_id  # compression continues in a child session
        self._interrupt_requested = False
        self._interrupt_message = None
        self._active_children = []
        self.max_iterations = 90
        self.model = "test/model"
        self.platform = "cli"
        self.compression_done = threading.Event()
        self.running = 0
        self.max_running = 0
        self.turns = []
        self._lock = threading.Lock()

    def run_conversation(self, **kwargs):
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            index = len(self.turns)
            self.turns.append({"user": kwargs["user_message"], "history": list(kwargs["conversation_history"])})
        try:
            if index == 0:
                self.compression_done.wait(15)
                if self.rotated_session_id:
                    self.session_id = self.rotated_session_id
                stopped = self._interrupt_requested
                self.turns[0]["stopped_by_interrupt"] = stopped
                if stopped:
                    return {"final_response": "", "messages": list(COMPRESSED), "api_calls": 1,
                            "completed": False, "interrupted": True,
                            "interrupt_message": self._interrupt_message, "partial": True,
                            "response_previewed": True}
            return {"final_response": "done", "messages": [*kwargs["conversation_history"],
                                                           {"role": "user", "content": kwargs["user_message"]},
                                                           {"role": "assistant", "content": "done"}],
                    "api_calls": 1, "completed": True, "partial": True, "response_previewed": True}
        finally:
            with self._lock:
                self.running -= 1

    def interrupt(self, message=None):
        self._interrupt_requested = True
        self._interrupt_message = message

    def clear_interrupt(self):
        self._interrupt_requested = False
        self._interrupt_message = None


def _turn_patches(cli):
    return (
        patch.object(cli, "_ensure_runtime_credentials", return_value=True),
        patch.object(cli, "_resolve_turn_agent_config", return_value={
            "signature": cli._active_agent_route_signature,
            "model": None, "runtime": None, "request_overrides": None,
        }),
        patch.object(cli, "_init_agent", return_value=True),
    )


def _setup(rotated_session_id=None):
    cli = _make_cli()
    agent = _CompressingAgent(cli.session_id, rotated_session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._POST_INTERRUPT_WAIT_S = 0.3
    cli.conversation_history = [{"role": "user", "content": f"old {i}"} for i in range(80)]
    cli._interrupt_queue.put("operator note: stop and read notes.txt")
    return cli, agent


def test_requeued_message_waits_for_the_interrupted_turn_to_stop():
    cli, agent = _setup()
    creds, route, init = _turn_patches(cli)
    with creds, route, init:
        cli.chat("read notes.txt until it is checked")
        # chat() stopped watching a worker that is still inside compression.
        assert agent.running == 1
        threading.Timer(0.5, agent.compression_done.set).start()
        pending = cli._pending_input.get_nowait()
        cli.chat(pending)

    assert agent.max_running == 1, "the next turn started while the interrupted turn was still running"
    assert agent.turns[0]["stopped_by_interrupt"] is True
    assert agent.turns[1]["user"] == "operator note: stop and read notes.txt"
    assert agent.turns[1]["history"] == COMPRESSED
    assert agent._interrupt_requested is False
    assert cli._abandoned_turn is None


def test_wedged_turn_never_shares_its_agent_with_the_next_turn():
    """Past the bound the message waits at the head of the queue; no second turn starts on the agent."""
    cli, agent = _setup()
    creds, route, init = _turn_patches(cli)
    try:
        with creds, route, init, patch("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0.3):
            cli.chat("read notes.txt until it is checked")
            abandoned = cli._abandoned_turn
            pending = cli._pending_input.get_nowait()
            cli._pending_input.put("typed later")
            started = time.monotonic()
            assert cli.chat(pending) is None
            assert time.monotonic() - started < 5
            # No second run_conversation, and the old turn keeps its owner and its interrupt flag.
            assert len(agent.turns) == 1
            assert agent._interrupt_requested is True
            assert cli._abandoned_turn is abandoned
            assert cli._pending_input_items() == [pending, "typed later"]

            # Once the old turn stops, the deferred message runs from its final transcript.
            agent.compression_done.set()
            cli.chat(cli._pending_input.get_nowait())
        assert agent.max_running == 1
        assert agent.turns[0]["stopped_by_interrupt"] is True
        assert agent.turns[1]["user"] == pending
        assert agent.turns[1]["history"] == COMPRESSED
        assert cli._abandoned_turn is None
    finally:
        agent.compression_done.set()


def test_deferred_message_keeps_its_images():
    cli, agent = _setup()
    creds, route, init = _turn_patches(cli)
    try:
        with creds, route, init, patch("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0.3):
            cli.chat("read notes.txt until it is checked")
            cli._pending_input.get_nowait()
            images = [Path("/tmp/shot.png")]
            assert cli.chat("look at this", images=images) is None
        assert cli._pending_input_items() == [("look at this", images)]
    finally:
        agent.compression_done.set()


def test_adopting_the_old_turn_reads_its_own_agent_not_a_rebuilt_one():
    """A model or route change can replace cli.agent while the old worker finishes."""
    cli, agent = _setup(rotated_session_id="compressed-child")
    creds, route, init = _turn_patches(cli)
    with creds, route, init:
        cli.chat("read notes.txt until it is checked")
    # A rebuilt agent with state of its own that the old turn must not touch.
    rebuilt = _CompressingAgent("rebuilt-own-session")
    rebuilt._interrupt_requested = True
    rebuilt._interrupt_message = "rebuilt agent's own interrupt"
    cli.agent = rebuilt
    threading.Timer(0.5, agent.compression_done.set).start()

    assert cli._chat_wait_for_abandoned_turn() is True

    assert agent.turns[0]["stopped_by_interrupt"] is True
    assert agent._interrupt_requested is False
    assert rebuilt._interrupt_requested is True
    assert rebuilt._interrupt_message == "rebuilt agent's own interrupt"
    assert rebuilt.session_id == "rebuilt-own-session"
    assert cli.session_id == "compressed-child"
    assert cli.conversation_history == COMPRESSED
    assert cli._abandoned_turn is None


def test_wedged_turn_on_a_retired_agent_does_not_block_a_rebuilt_one():
    cli, agent = _setup()
    creds, route, init = _turn_patches(cli)
    rebuilt = _CompressingAgent(cli.session_id)
    rebuilt.compression_done.set()
    try:
        with creds, route, init, patch("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0.3):
            cli.chat("read notes.txt until it is checked")
            pending = cli._pending_input.get_nowait()
            cli.agent = rebuilt
            cli.chat(pending)
        assert len(agent.turns) == 1
        assert agent._interrupt_requested is True  # the old worker still gets its stop
        assert [t["user"] for t in rebuilt.turns] == [pending]
        assert cli._abandoned_turn is None
    finally:
        agent.compression_done.set()


def test_session_switch_while_waiting_keeps_the_new_transcript():
    cli, agent = _setup()
    creds, route, init = _turn_patches(cli)
    with creds, route, init:
        cli.chat("read notes.txt until it is checked")
        pending = cli._pending_input.get_nowait()
        # A /new between the two turns: fresh session id and an empty transcript.
        cli.session_id = "fresh-session"
        cli.conversation_history = []
        threading.Timer(0.5, agent.compression_done.set).start()
        cli.chat(pending)

    assert agent.max_running == 1
    assert agent.turns[1]["history"] == []
