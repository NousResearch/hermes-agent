"""Type-ahead queue-drain proof for /compress (issue #61042, PR #68284).

PR #68284 made the classic prompt_toolkit CLI keep the composer editable
while ``/compress`` runs (``_busy_command(..., blocks_input=False)``), with
the docstring claim that "the queued input is processed against the
compacted history after the command completes."

These tests pin that claim end-to-end for the classic CLI:

1. ``test_type_ahead_queued_during_compress_becomes_next_prompt`` — runs
   ``_manual_compress`` on a worker thread (matching the real topology:
   slash commands execute on ``process_loop``, which is blocked inside
   ``process_command`` for the duration), submits a type-ahead payload into
   ``_pending_input`` mid-compression (what ``handle_enter``'s idle branch
   does — ``_agent_running`` is False while a slash command runs), and
   asserts that after compression commits the compacted history the queued
   text is still the next item ``process_loop`` will drain. Nothing inside
   the compression path may consume or drop it.

2. ``test_handle_enter_never_gates_on_command_running`` — structural
   invariant: ``handle_enter`` must not consult ``_command_running`` /
   ``_command_blocks_input``. The composer's read-only state is enforced
   solely by the TextArea's ``read_only=Condition(_command_blocks_input)``;
   if a future edit added an early-return in ``handle_enter`` while a
   command runs, type-ahead submissions would be silently dropped and the
   drain contract above would break without any other test noticing.

The Ink TUI (ui-tui) needs no equivalent fix: its ``/compress`` is an async
``session.compress`` RPC and the composer is never made read-only while it
runs; the gateway already resolves the concurrent-mutation race via the
``history_version`` guard in ``_compress_session_history``.
"""

from __future__ import annotations

import ast
import queue as queue_mod
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_type_ahead_queued_during_compress_becomes_next_prompt():
    """Text queued while /compress runs survives compaction and is the next prompt."""
    shell = _make_cli()
    history = _make_history()
    compressed = [
        {"role": "user", "content": "[summary]"},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id

    compress_started = threading.Event()
    release_compress = threading.Event()
    mid_compress = {}

    def _compress(*_args, **_kwargs):
        compress_started.set()
        # Hold the "compression in flight" window open until the test has
        # simulated the user's type-ahead submission.
        assert release_compress.wait(timeout=10), "test deadlock: never released"
        return (list(compressed), "")

    shell.agent._compress_context.side_effect = _compress

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        worker = threading.Thread(target=shell._manual_compress, daemon=True)
        worker.start()
        assert compress_started.wait(timeout=10), "compression never started"

        # Mid-compression: composer stays editable (spinner up, input open).
        mid_compress["running"] = shell._command_running
        mid_compress["blocks_input"] = shell._command_blocks_input

        # The user types a follow-up and presses Enter. handle_enter's normal
        # routing puts it on _pending_input (agent idle — slash commands run
        # on process_loop, not the agent). process_loop is blocked inside
        # process_command → _manual_compress, so the payload must sit queued.
        shell._pending_input.put("follow-up prompt drafted during compaction")

        release_compress.set()
        worker.join(timeout=10)

    assert not worker.is_alive(), "_manual_compress did not finish"
    assert mid_compress == {"running": True, "blocks_input": False}

    # Compaction committed the compressed transcript...
    assert shell.conversation_history == compressed
    # ...and busy state was fully unwound so the next turn renders normally.
    assert shell._command_running is False
    assert shell._command_blocks_input is False

    # The queued type-ahead is exactly the next item process_loop drains —
    # i.e. it becomes the next prompt, processed against the compacted
    # history. Compression must not consume, reorder, or drop it.
    assert (
        shell._pending_input.get_nowait()
        == "follow-up prompt drafted during compaction"
    )
    with pytest.raises(queue_mod.Empty):
        shell._pending_input.get_nowait()


def test_handle_enter_never_gates_on_command_running():
    """handle_enter must not consult the busy-command flags (drop-proof routing).

    Enter-key routing while a slash command runs must keep flowing into
    ``_pending_input``; read-only enforcement belongs exclusively to the
    TextArea's ``read_only=Condition(...)``. A ``_command_running`` /
    ``_command_blocks_input`` check inside ``handle_enter`` would let a
    future edit silently drop type-ahead submissions during /compress.
    """
    cli_path = Path(__file__).resolve().parents[2] / "cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "handle_enter":
            target = node
            break
    assert target is not None, "handle_enter closure not found in cli.py"

    offenders = [
        node.attr
        for node in ast.walk(target)
        if isinstance(node, ast.Attribute)
        and node.attr in {"_command_running", "_command_blocks_input"}
    ]
    assert not offenders, (
        "handle_enter references busy-command state — Enter routing while a "
        f"slash command runs risks dropping type-ahead input: {offenders}"
    )
