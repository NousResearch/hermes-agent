"""Handler-level tests for CLI ``/learn`` skill_manage gating."""

import io
import queue
from contextlib import redirect_stdout
from unittest.mock import MagicMock

from agent.learn_prompt import LEARN_UNAVAILABLE_MESSAGE, build_learn_prompt
from hermes_cli.cli_commands_mixin import CLICommandsMixin

_READ_ONLY = {"memory", "skills_list", "skill_view"}
_WRITABLE = _READ_ONLY | {"skill_manage"}


class _Stub(CLICommandsMixin):
    def __init__(self, agent=None):
        self.agent = agent
        self._pending_input = queue.Queue()


def _run(stub, command):
    buf = io.StringIO()
    with redirect_stdout(buf):
        stub._handle_learn_command(command)
    return buf.getvalue(), stub._pending_input


def test_cli_learn_read_only_prints_unavailable_and_does_not_enqueue():
    stub = _Stub(agent=MagicMock(valid_tool_names=_READ_ONLY))
    out, pending = _run(stub, "/learn auth flow in acme-sdk")
    assert LEARN_UNAVAILABLE_MESSAGE in out
    assert pending.empty()


def test_cli_learn_writable_enqueues_prompt_without_unavailable_message():
    stub = _Stub(agent=MagicMock(valid_tool_names=_WRITABLE))
    out, pending = _run(stub, "/learn auth flow in acme-sdk")
    assert LEARN_UNAVAILABLE_MESSAGE not in out
    msg = pending.get_nowait()
    assert msg == build_learn_prompt("auth flow in acme-sdk")
    assert "[/learn]" in msg
