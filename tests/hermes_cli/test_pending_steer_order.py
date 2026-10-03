"""Post-turn /steer must precede interrupt follow-ups (regression for #30323)."""

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("images", [False, True])
def test_leftover_steer_precedes_interrupt_batch(images):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    cli._voice_tts = False
    cli.agent = SimpleNamespace(max_iterations=500)
    cli._chat_print_reasoning_box = Mock()
    cli._chat_print_response_panel = Mock()
    cli._ring_bell = Mock()
    cli._emit_focus_recovery_line = Mock()
    first = ("first", ["first.png"]) if images else "first"
    second = ("second", ["second.png"]) if images else "second"
    cli._chat_resolve_interrupt = Mock(return_value=(first, False))
    cli._interrupt_queue.put(second)
    turn = SimpleNamespace(result={"final_response": "answer", "completed": True,
                                  "pending_steer": "use the safe approach"})

    assert cli._chat_render_turn(turn, None, None) == "answer"
    assert cli._pending_input.get_nowait() == "use the safe approach"
    expected = ("first\nsecond", ["first.png", "second.png"]) if images else "first\nsecond"
    assert cli._pending_input.get_nowait() == expected
    assert cli._pending_input.empty()
    assert cli._interrupt_queue.empty()


@pytest.mark.parametrize("pending,result,expected", [
    (None, None, []),
    (None, {"pending_steer": "guidance"}, ["guidance"]),
    ("follow-up", {}, ["follow-up"]),
    ("follow-up", {"pending_steer": ""}, ["follow-up"]),
])
def test_post_run_queue_preserves_single_inputs(pending, result, expected):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    cli._pending_input.put("previously queued")
    cli._enqueue_post_run_followups(pending, result)
    assert cli._pending_input.get_nowait() == "previously queued"
    assert [cli._pending_input.get_nowait() for _ in expected] == expected
    assert cli._pending_input.empty()
