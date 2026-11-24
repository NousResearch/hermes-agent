"""Epoch file for a bot-loop reset, stored under gateway-state.

Storage helpers only read and write the epoch. ``note_human_reset`` writes it
when a human turn lands in a session titled exactly Bot Chat and the feature
flag is on. A disk error is logged and never fails the turn.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import tempfile
import time
from pathlib import Path

from hermes_cli.config import bot_loop_reset_on_human_enabled
from hermes_constants import get_default_hermes_root

logger = logging.getLogger(__name__)

EPOCH_NAME = "bot-loop-reset-epoch"


def epoch_path(root: Path) -> Path:
    """``<root>/gateway-state/<EPOCH_NAME>``."""
    return Path(root) / "gateway-state" / EPOCH_NAME


def read_epoch(root: Path) -> float | None:
    """The stored epoch, or None when the file is missing or unreadable."""
    try:
        value = float(epoch_path(root).read_text(encoding="utf-8").strip())
    except (OSError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def write_epoch(root: Path, when: float | None = None) -> float:
    """Store ``when`` (or now) via a same-directory temp file and ``os.replace``."""
    stamp = time.time() if when is None else when
    path = epoch_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{stamp}\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return stamp


def _session_title(agent) -> str:
    """Same two-step lookup as ``agent.conversation_loop._bot_chat_prompt_stale``."""
    title = str(getattr(agent, "_session_title_hint", "") or "").strip()
    if not title and agent._session_db and agent.session_id:
        try:
            from agent.agent_init_session import agent_interface
            store = agent_interface(agent)
            title = str((store.sessions.get_title(agent.session_id) if store is not None
                         else agent._session_db.get_session_title(agent.session_id)) or "").strip()
        except Exception:
            title = ""
    return title


def note_human_reset(agent) -> None:
    """Write the reset epoch for a human turn in Bot Chat when the flag is on.

    Every exception is logged and swallowed. A disk error must not fail the turn.
    """
    try:
        if bot_loop_reset_on_human_enabled() is not True:
            return
        from tools.bot_mode_probe import BOT_CHAT_TITLE
        if _session_title(agent) != BOT_CHAT_TITLE:
            return
        # Install root, not the process HERMES_HOME: a profile's home is the
        # profile directory, and the gateway reads the epoch from the install root.
        write_epoch(get_default_hermes_root())
    except Exception:
        logger.warning("bot-loop human reset failed", exc_info=True)
