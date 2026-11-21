"""``/btw --open``: a finished side thread as the seed of a side chat (agent/side_thread.py).

The seed is what Desktop creates a child session from: the conversation the thread forked from
(visible text only, roles alternating), then the side task and its report.
"""

from __future__ import annotations

import pytest

from agent import side_thread
from agent.side_thread import parse_btw_args, register_thread, run_thread_command, side_chat_seed


@pytest.fixture(autouse=True)
def _empty_registry():
    side_thread._registry.clear()
    yield
    side_thread._registry.clear()


def _finished(context):
    thread = register_thread("s", "find the retry logic")
    thread.context = context
    thread.tool_calls = ["search_files", "read_file", "read_file"]
    thread._finish("done", answer="retry lives in net.py:40")
    return thread


def test_seed_keeps_visible_text_alternating_and_ends_with_the_report():
    thread = _finished([
        {"role": "system", "content": "SYSTEM — never seeded"},
        {"role": "user", "content": "fix the flaky test"},
        {"role": "assistant", "content": "Looking.", "tool_calls": [{"function": {"name": "read_file"}}]},
        {"role": "tool", "content": "file body — never seeded"},
        {"role": "assistant", "content": "Fixed."},
    ])
    seed = side_chat_seed(thread)
    roles = [row["role"] for row in seed["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert seed["messages"][1]["content"] == "Looking.\n\nFixed."
    assert seed["messages"][2]["content"] == "/btw --explore find the retry logic"
    assert seed["messages"][3]["content"].startswith("retry lives in net.py:40")
    assert "read_file ×2" in seed["messages"][3]["content"]
    assert "SYSTEM" not in str(seed) and "file body" not in str(seed)
    assert seed["title"] == "btw: find the retry logic"


def test_open_replies_with_the_seed_and_a_fallback_line():
    thread = _finished([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    reply = run_thread_command("s", parse_btw_args(f"--open {thread.thread_id}")).reply()
    assert reply["side_chat"] == side_chat_seed(thread)
    assert thread.thread_id in reply["text"] and thread.thread_id in reply["fallback_text"]
    assert "prompt" not in reply


def test_open_refuses_a_running_thread():
    running = register_thread("s", "still going")
    reply = run_thread_command("s", parse_btw_args(f"--open {running.thread_id}")).reply()
    assert "side_chat" not in reply


def test_parser_accepts_open():
    assert parse_btw_args("--open btw_ab12cd") == side_thread.BtwRequest("open", "", "btw_ab12cd")
    assert parse_btw_args("-o").action == "open"
