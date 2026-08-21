"""Dispatch-wiring tests for /upskill on the classic CLI.

/upskill is wired the same way as /learn: an explicit ``elif canonical ==
"upskill"`` branch in ``HermesCLI.process_command`` calls
``self._handle_upskill_command``. The registry entry alone does NOT route —
the canonical name would otherwise fall through to "Unknown command". These
tests drive the real dispatcher to lock the wiring in (per AGENTS.md: E2E
validation, not just unit mocks).
"""

from queue import Queue
from unittest.mock import patch

from tests.cli.test_cli_init import _make_cli


class TestUpskillDispatchWiring:
    def test_upskill_routes_to_handler_with_scope(self):
        """/upskill <hint> must call _handle_upskill_command with the scope."""
        cli = _make_cli()
        with patch.object(cli, "_handle_upskill_command") as mock_h:
            cli.process_command("/upskill focus on the WiNG console")

        mock_h.assert_called_once_with("/upskill focus on the WiNG console")

    def test_upskill_bare_is_dispatchable(self):
        """A bare /upskill must NOT fall through to 'Unknown command'."""
        cli = _make_cli()
        with patch.object(cli, "_handle_upskill_command") as mock_h:
            cli.process_command("/upskill")

        mock_h.assert_called_once_with("/upskill")


class TestUpskillHandlerInjection:
    def test_handler_injects_sweep_prompt_onto_pending_input(self):
        """/upskill must push the standards-guided sweep prompt to the queue.

        The load-bearing behaviour contract (mirrors /learn): the handler
        builds build_upskill_prompt(...) and puts it on ``_pending_input`` so
        the live agent runs it as a normal turn. We patch the prompt builder
        to a sentinel and assert exactly that message lands on the queue.
        """
        cli = _make_cli()
        cli._pending_input = Queue()
        sentinel = "SWEEP-PROMPT-SENTINEL"

        from agent import upskill_prompt

        with patch.object(upskill_prompt, "build_upskill_prompt", return_value=sentinel):
            cli._handle_upskill_command("/upskill")

        assert cli._pending_input.get_nowait() == sentinel
