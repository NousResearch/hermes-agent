"""An interrupted turn recovers the terminal before it prints its own output (regression for #157627).

The post-interrupt recovery erases the viewport and replays the output history. The interrupt
marker is deliberately never recorded (#60920), so a recovery that ran after it would erase it.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

from agent.i18n import t
from tests.hermes_cli.test_cli_interrupt_ack_race import _make_cli, _StubAgent


class _InterruptedByNewMessageAgent(_StubAgent):
    """A turn the user interrupted by typing a new message: it shows the interrupt marker."""

    def run_conversation(self, **kwargs):
        return {
            "final_response": "partial answer",
            "messages": [{"role": "user", "content": "original"},
                         {"role": "assistant", "content": "partial answer"}],
            "api_calls": 1,
            "interrupted": True,
            "interrupt_message": "new message",
            "response_previewed": True,  # skip the Rich panel (crashes under the prompt_toolkit stubs)
        }


def test_interrupted_turn_recovers_the_terminal_before_printing_the_marker():
    cli = _make_cli()
    cli.agent = _InterruptedByNewMessageAgent(cli.session_id, turn_seconds=0)
    cli._interrupt_queue, cli._pending_input = queue.Queue(), queue.Queue()
    cli._app = MagicMock()  # the interactive CLI is running
    painted = []

    import cli as cli_mod

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True), \
         patch.object(cli, "_recover_terminal_after_interrupt", side_effect=lambda: painted.append("<recovery>")), \
         patch.object(cli_mod, "_cprint", side_effect=lambda text, **kw: painted.append(text)):
        cli.chat("original")

    marker = t("cli.chat.interrupted_marker")
    marker_at = next(i for i, text in enumerate(painted) if marker in text)
    assert painted.count("<recovery>") == 1 and painted.index("<recovery>") < marker_at, painted
