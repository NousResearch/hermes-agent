"""
Core types for the session-orchestration subsystem.

Defines:
- ``SessionLifecycle`` — the lifecycle state enum for an agent session.
- ``SessionHandle`` — a typed dataclass returned by ``AgentAdapter.launch()``.
- ``Capabilities`` — a descriptor dataclass declaring what an adapter supports.

Design notes
------------
- ``idle_indicator_regex=None`` means the adapter cannot declare a
  pattern that unambiguously signals active work; hang detection falls
  back to pane-hash-only comparison in the watcher.
- ``dialog_handlers`` is a mapping from a recognisable dialog-title or
  regex key to a callable that drives the tmux pane through the dialog.
  An empty dict means the adapter handles no dialogs automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Lifecycle enum
# ---------------------------------------------------------------------------


class SessionLifecycle(str, Enum):
    """Observable lifecycle state of an external agent session.

    Values
    ------
    RUNNING
        The agent is actively processing; no user input required.
    WAITING_USER
        The agent has produced output and is waiting for the user to
        reply (e.g. Claude Code showing the ``❯`` prompt).
    PAUSED_HANDOFF
        The session reached a ``/clear``-then-handoff checkpoint; the
        watcher may trigger a deterministic ``/clear``+resume.
    STALLED
        The session is alive but has not produced new output for a
        watcher-configured threshold; the watcher will nudge once, then
        escalate to the user.
    DONE
        The session completed its task and has exited or returned to an
        idle state from which no further output is expected.
    ERROR
        The session exited with an error or the adapter detected an
        unrecoverable failure.
    """

    RUNNING = "RUNNING"
    WAITING_USER = "WAITING_USER"
    PAUSED_HANDOFF = "PAUSED_HANDOFF"
    STALLED = "STALLED"
    DONE = "DONE"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# SessionHandle
# ---------------------------------------------------------------------------


@dataclass
class SessionHandle:
    """Opaque handle returned by ``AgentAdapter.launch()``.

    Fields
    ------
    session_id : str
        A unique identifier for this managed session (UUID or similar).
        Used as the primary key in the registry.
    tmux_session : str
        The tmux session name (e.g. ``"hermes-claude-abc123"``).
    pane : str
        The tmux pane target (e.g. ``"hermes-claude-abc123:0.0"``).
    launch_ts : datetime
        UTC timestamp recorded at the moment ``launch()`` returned the
        handle. Used to compute elapsed time and to detect stale handles.
    marker_file : Optional[str]
        Absolute path to the JSONL marker file injected into the tmux
        session environment as ``HERMES_MARKER_FILE``.  ``None`` until
        ``launch()`` sets it (legacy handles or handles not yet launched).
    """

    session_id: str
    tmux_session: str
    pane: str
    launch_ts: datetime
    marker_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass
class Capabilities:
    """Descriptor that an ``AgentAdapter`` declares about itself.

    The watcher asserts declared capabilities against observed behaviour
    at startup; a mismatch hard-fails that adapter with a logged error
    (does not crash the watcher).

    Fields
    ------
    supports_print_mode : bool
        True if the agent CLI supports a non-interactive ``--print``
        (one-shot / headless) mode that writes structured output to stdout.
    has_hooks : bool
        True if the agent supports a ``--hook`` flag (e.g. ``omp --hook``)
        that fires lifecycle callbacks; the watcher can use this as a
        positive-liveness accelerant.
    rpc_mode : bool
        True if the agent supports a structured RPC/JSON transport mode
        (e.g. ``omp --mode=rpc`` or ``omp --mode=json``) that avoids
        terminal scraping.
    json_mode : bool
        True if the agent can emit machine-readable JSON output when asked
        (may overlap with ``rpc_mode`` for some agents).
    idle_indicator_regex : Optional[re.Pattern]
        A compiled regular expression that matches a line in the pane
        capture that unambiguously signals the agent is *active* (e.g.
        a running-tool indicator).  ``None`` means the adapter cannot
        declare such a pattern; hang detection falls back to pane-hash
        comparison only.
    dialog_handlers : dict[str, Callable[[str], None]]
        A mapping from dialog identifiers (strings or regex patterns used
        as keys) to callables that drive the tmux pane through that
        dialog.  The callable receives the tmux pane target as its sole
        argument.  An empty dict means the adapter handles no dialogs
        automatically.
    """

    supports_print_mode: bool = False
    has_hooks: bool = False
    rpc_mode: bool = False
    json_mode: bool = False
    idle_indicator_regex: Optional[re.Pattern] = None  # type: ignore[type-arg]
    dialog_handlers: dict[str, Callable[[str], None]] = field(default_factory=dict)
