"""Epoch file for a bot-loop reset, a sibling of the install-root config.yaml.

Storage helpers only read and write the epoch. ``note_human_reset`` writes it
when a human turn lands in a session titled exactly Bot Chat and
``gateway.bot_loop_guard.reset_all_on_human`` is on. A disk error is logged
and never fails the turn.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EPOCH_NAME = "bot-loop-reset-epoch"


def get_hermes_root() -> Path:
    """Install root. Re-exported so tests can patch ``gateway.bot_loop_reset.get_hermes_root``."""
    from hermes_constants import get_hermes_root as root

    return root()


def epoch_path(root: Path) -> Path:
    """``<root>/<EPOCH_NAME>``, a sibling of config.yaml."""
    return Path(root) / EPOCH_NAME


def read_epoch(root: Path) -> float:
    """The stored epoch. Missing, garbage, non-finite, or ``<= 0`` is ``0.0``."""
    try:
        value = float(epoch_path(root).read_text(encoding="utf-8-sig").strip())
    except (OSError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return value


def write_epoch(root: Path, now: float) -> None:
    """Store ``now`` via a same-directory temp file and ``os.replace``.

    Text is ``f"{now:.6f}\\n"``. Creates ``root`` if needed. Does not return the stamp.
    """
    path = epoch_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{now:.6f}\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _session_title(agent) -> str:
    """Same two-step lookup as ``agent.conversation_loop._bot_chat_prompt_stale``.

    A lookup failure propagates. ``note_human_reset`` logs it and does not write.
    """
    title = str(getattr(agent, "_session_title_hint", "") or "").strip()
    if not title and agent._session_db and agent.session_id:
        from agent.agent_init_session import agent_interface
        store = agent_interface(agent)
        title = str((store.sessions.get_title(agent.session_id) if store is not None
                     else agent._session_db.get_session_title(agent.session_id)) or "").strip()
    return title


def reset_all_on_human(root: Path | None = None) -> bool:
    """True when install-root ``gateway.bot_loop_guard.reset_all_on_human`` is on.

    Reads ``<root or get_hermes_root()>/config.yaml`` exactly as written. A profile
    config.yaml is not consulted. A missing file is False.
    """
    from gateway.bot_loop_guard import _as_bool
    from hermes_cli.config import read_user_config_raw

    install = Path(root) if root is not None else get_hermes_root()
    data = read_user_config_raw(install / "config.yaml")
    gateway = data.get("gateway") if isinstance(data, dict) else None
    block = gateway.get("bot_loop_guard") if isinstance(gateway, dict) else None
    raw = block.get("reset_all_on_human") if isinstance(block, dict) else None
    return _as_bool(raw, False)


def note_human_reset(agent) -> None:
    """Write the reset epoch for a human turn in Bot Chat when the flag is on.

    Every exception is logged and swallowed. A disk error must not fail the turn.
    """
    try:
        if reset_all_on_human() is not True:
            return
        from tools.bot_mode_probe import BOT_CHAT_TITLE
        if _session_title(agent) != BOT_CHAT_TITLE:
            return
        write_epoch(get_hermes_root(), time.time())
    except Exception:
        logger.warning("bot-loop human reset failed", exc_info=True)
