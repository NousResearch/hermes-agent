"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory
(design #19214/#19242: `terminal.cwd` is bridged once to `TERMINAL_CWD` at
gateway/cron startup). The local-CLI backend deliberately leaves it unset and
relies on the launch dir. Reading it in one place keeps the system prompt, the
tool surfaces, and context-file discovery agreeing on where the agent lives.

The #29531 per-session extension point is this function: a future PR adds a
contextvar arm inside `resolve_agent_cwd` and `.set()`s it at the
`set_session_vars` seam — by design, not a reopening hazard.
"""

import os
from pathlib import Path


def resolve_agent_cwd() -> Path:
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
    return Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    # None is load-bearing: it signals "skip context-file discovery" so the
    # gateway install dir's AGENTS.md isn't slurped (see system_prompt.py).
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    return Path(raw).expanduser() if raw else None
