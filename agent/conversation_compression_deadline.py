"""The compression fence's deadline, shared live with the summary stream (sibling of
``conversation_compression``).

Compression waits a total time limit for its summary. That limit stops a summary that trickles
out a token at a time; on a laptop GPU, reading a 150K-token conversation alone can take longer
than the whole limit. Time llama-server spends reading the prompt therefore pushes the fence
deadline back, and hosts read the deadline from the fence instead of adding the configured limit
to the time they started waiting.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

_extend_lock = threading.Lock()


def extend_fence_deadline(fence: Any, seconds: float) -> bool:
    """Push *fence*'s deadline back by *seconds*, unless the attempt is over. A cancelled fence or
    a deadline already passed stays final, so a late progress chunk can't revive an attempt the
    host has given up on."""
    with _extend_lock:
        deadline, now = fence.deadline_monotonic, time.monotonic()
        if seconds <= 0 or deadline is None or deadline <= now or fence.is_cancelled:
            return False
        fence.set_total_ceiling_seconds(deadline + seconds - now)
        return True


def effective_ceiling(fence: Any, ceiling: float, wait_started: float) -> float:
    """The total limit for a host that started waiting at *wait_started*: the configured
    *ceiling*, or longer once prompt reading pushed the fence deadline back."""
    deadline = getattr(fence, "deadline_monotonic", None)
    return ceiling if deadline is None else max(ceiling, deadline - wait_started)


@contextlib.contextmanager
def compression_deadline_scope(fence: Any):
    """Share *fence*'s deadline with the summary stream, live, and push it back while the
    server reads the prompt. A stream checks the deadline as it reads, so a later summary
    call in the same attempt sees the extended deadline too."""
    from agent.auxiliary_client import aux_stream_deadline
    from agent.local_prompt_progress import prompt_reading_hook

    if fence is None:
        yield
        return
    with aux_stream_deadline(lambda: fence.deadline_monotonic), \
            prompt_reading_hook(lambda seconds: extend_fence_deadline(fence, seconds)):
        yield
