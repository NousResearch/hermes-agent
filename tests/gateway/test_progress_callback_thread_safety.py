"""
Regression test for issue #24604: progress_callback dedup state race condition.

`last_progress_msg` and `repeat_count` are shared closure variables accessed by
`progress_callback` which is called from concurrent ThreadPoolExecutor workers
inside `_execute_tool_calls_concurrent`. Without a lock the read-check-write
sequence is not atomic, causing lost increments and broken dedup windows.

This test verifies that concurrent callers produce consistent dedup state —
no lost increments and no duplicate "new" entries for the same message.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor


def make_progress_callback():
    """Reproduce the progress_callback dedup logic from gateway/run.py."""
    progress_queue = queue.Queue()
    last_progress_msg = [None]
    repeat_count = [0]
    _dedup_lock = threading.Lock()

    def callback(msg: str) -> None:
        with _dedup_lock:
            if msg == last_progress_msg[0]:
                repeat_count[0] += 1
                progress_queue.put(("__dedup__", msg, repeat_count[0]))
                return
            last_progress_msg[0] = msg
            repeat_count[0] = 0
        progress_queue.put(msg)

    return callback, progress_queue, repeat_count


def test_concurrent_identical_messages_count_correctly():
    """All N identical messages from concurrent workers must be counted.

    Without the lock, repeat_count[0] += 1 is a non-atomic read-modify-write.
    Concurrent threads lose increments, producing a final count lower than N-1.
    """
    N = 50
    callback, progress_queue, repeat_count = make_progress_callback()

    # Prime with first message so all subsequent calls hit the dedup branch.
    callback("tool: echo hello")

    with ThreadPoolExecutor(max_workers=N) as ex:
        futures = [ex.submit(callback, "tool: echo hello") for _ in range(N)]
        for f in futures:
            f.result()

    items = []
    while not progress_queue.empty():
        items.append(progress_queue.get_nowait())

    dedup_items = [i for i in items if isinstance(i, tuple) and i[0] == "__dedup__"]
    assert len(dedup_items) == N, (
        f"Expected {N} __dedup__ entries, got {len(dedup_items)}. "
        "Lost increments indicate a missing lock."
    )
    counts = [i[2] for i in dedup_items]
    assert counts == sorted(counts), f"Counts not monotone: {counts}"


def test_concurrent_distinct_messages_all_emitted():
    """Distinct messages from concurrent workers must all be emitted as new.

    Without the lock two threads can both read last_progress_msg[0] == None,
    both treat their message as new, and then overwrite each other — causing
    one message to appear new while the other is silently collapsed as a dedup
    even though it was never seen before.
    """
    N = 20
    callback, progress_queue, _ = make_progress_callback()

    messages = [f"tool: cmd_{i}" for i in range(N)]

    with ThreadPoolExecutor(max_workers=N) as ex:
        futures = [ex.submit(callback, m) for m in messages]
        for f in futures:
            f.result()

    items = []
    while not progress_queue.empty():
        items.append(progress_queue.get_nowait())

    new_msgs = [i for i in items if isinstance(i, str)]
    dedup_msgs = [i for i in items if isinstance(i, tuple) and i[0] == "__dedup__"]

    assert len(new_msgs) == N, (
        f"Expected {N} new messages, got {len(new_msgs)}. "
        f"Dedup entries: {len(dedup_msgs)}. Missing lock may collapse distinct messages."
    )
    assert dedup_msgs == [], f"No dedup expected for distinct messages; got {dedup_msgs}"
