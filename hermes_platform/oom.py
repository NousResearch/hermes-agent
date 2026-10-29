"""Make heavy children the Linux OOM killer's first pick instead of the Hermes process.

When a memory ceiling is crossed the kernel kills the process with the highest ``oom_score``
(its share of memory plus ``oom_score_adj``). Left alone, that is often the long-lived gateway,
and every session on it dies with it. Raising each heavy child's ``oom_score_adj`` (terminal
commands, browsers) makes the child the victim: the turn that spawned it sees a killed command,
the gateway and every other session keep running.

The children are raised rather than the parent lowered because an unprivileged process can
raise ``oom_score_adj`` freely but cannot go below the floor a privileged writer set (normally
0). The value is written right after spawn; a grandchild forked in the microseconds before the
write keeps the parent's value, which only means it is not preferred, never less safe.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

# Added to the value the child inherited from its spawner: relative, so a gateway a service
# manager already runs at 200 still ends up 500 points safer than its children.
CHILD_OOM_SCORE_BOOST = 500
_OOM_SCORE_ADJ_MAX = 1000


def prefer_as_oom_victim(pid: int) -> int:
    """Raise ``pid``'s ``oom_score_adj`` by :data:`CHILD_OOM_SCORE_BOOST` (capped at 1000; Linux
    only). Call once, right after spawn. Returns ``pid`` so call sites can wrap the assignment."""
    if sys.platform != "linux" or not pid:
        return pid
    path = Path(f"/proc/{pid}/oom_score_adj")
    # Already exited, /proc mounted read-only, or another uid's process: nothing to prefer.
    with contextlib.suppress(OSError, ValueError):
        inherited = int(path.read_text(encoding="ascii").strip())
        path.write_text(str(min(_OOM_SCORE_ADJ_MAX, inherited + CHILD_OOM_SCORE_BOOST)), encoding="ascii")
    return pid
