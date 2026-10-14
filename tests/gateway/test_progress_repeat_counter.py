"""The tool-progress repeat counter must never land on a closing code fence (#138148).

``_progress_emit``/``_progress_absorb`` append ``(×N)`` to the WHOLE progress message; when that
message is a fenced tool preview an inline suffix used to sit after the closing ````` — Telegram's
anchored fence parser then saw no fence at all and escaped the block into literal backticks. The
counter now starts its own line instead.
"""

import queue
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner
from plugins.platforms.telegram.adapter import TelegramAdapter

FENCED = "💻 terminal\n```\nssh -F /opt/data/home/.ssh/config fed...\n```"


def test_absorbed_repeat_counter_starts_its_own_line():
    turn = TurnRunner(SimpleNamespace(), SimpleNamespace())
    st = SimpleNamespace(progress_lines=[FENCED])
    line = turn._progress_absorb(st, ("__dedup__", FENCED, 3))
    assert line == f"{FENCED}\n(×4)"
    assert "``` (×" not in line


def test_absorbed_repeat_keeps_the_telegram_fence_renderable():
    turn = TurnRunner(SimpleNamespace(), SimpleNamespace())
    st = SimpleNamespace(progress_lines=[FENCED])
    line = turn._progress_absorb(st, ("__dedup__", FENCED, 3))
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    rendered = TelegramAdapter.format_message(adapter, line)
    # A recognized fence is restored verbatim (code is never escaped); pre-fix the whole
    # closing fence fell through the escaper as literal backtick runs.
    assert "```" in rendered
    assert "\\`" not in rendered


def test_native_progress_repeat_moves_off_the_fence():
    seen = []
    sc = SimpleNamespace(accepts_tool_progress=True, on_tool_progress=seen.append)
    ctx = SimpleNamespace(
        last_progress_msg=[FENCED],
        repeat_count=[0],
        progress_queue=queue.Queue(),
        stream_consumer_holder=[sc],
    )
    turn = TurnRunner(SimpleNamespace(), ctx)
    turn._progress_emit(FENCED)  # duplicate → counter fires
    assert seen == [f"{FENCED}\n(×2)"]
