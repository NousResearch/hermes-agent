"""Global emergency stop (ESTOP) — a resumable pause for NEW work only.

``hermes pause`` writes a sentinel at ``$HERMES_HOME/ESTOP``; ``hermes resume``
removes it. While it exists the cron scheduler, kanban dispatcher and new gateway
turns skip work; in-flight work is never killed. The check is one or two uncached
``os.stat`` calls (process home + fleet root when they differ). The body is optional
JSON ``{"reason", "engaged_at"}``; a corrupt/empty file still counts as engaged
(fail safe, e.g. ``touch ~/.hermes/ESTOP``). Ported from gastownhall/gastown estop.go (MIT).

Only an operator may engage or lift the stop. ``engage``/``disengage`` raise
:class:`EstopRefused` when called from an agent or automation context (a gateway or
cron turn, a kanban worker, ``hermes chat -q``, or any process an agent started), so
an agent cannot halt every profile, nor silently lift a halt an operator engaged.
The gateway's ``/pause`` passes ``from_chat=True`` and is governed by
``security.estop_chat_control`` instead. The markers are environment variables an
agent with a shell can scrub, and the sentinel is a plain file it can write: this
stops an agent that runs ``hermes pause``/``hermes resume``; it is not a sandbox
(SECURITY.md §2.4).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import suppress
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Optional

# Same profile-aware / fleet-root resolvers the file-safety guards use (fail-open to ~/.hermes).
from agent.file_safety import _hermes_home_path as _hermes_home, _hermes_root_path as _canonical_root

SENTINEL_NAME = "ESTOP"

logger = logging.getLogger(__name__)


class EstopRefused(PermissionError):
    """An agent/automation context tried to engage or lift the emergency stop."""


# Per-component "logged already for this engagement" flags: log once per engagement, not per tick.
_log_lock = threading.Lock()
_logged_components: set[str] = set()


def sentinel_path() -> Path:
    """Path of the ESTOP sentinel this process would write on `hermes pause`."""
    return _hermes_home() / SENTINEL_NAME


def _candidate_sentinel_paths() -> list:
    """Profile home first, then the fleet root if it is a different directory: a profile
    gateway (HERMES_HOME=~/.hermes/profiles/<n>) must still honor an operator's ~/.hermes/ESTOP."""
    primary = sentinel_path()
    try:
        root = _canonical_root() / SENTINEL_NAME
    except Exception:
        return [primary]
    try:
        distinct = root.resolve() != primary.resolve()
    except Exception:
        # Non-Path test doubles fail .resolve(); plain equality still dedupes.
        distinct = root != primary
    return [primary, root] if distinct else [primary]


def _agent_context() -> Optional[str]:
    """Describe the agent/automation context this call runs in, or None for an operator.

    Session markers are read through the approval predicates (contextvar first, env
    fallback). ``HERMES_AGENT`` is exported into every terminal-tool shell (remote
    backends included), the execute_code sandbox and stdio MCP servers, so finding it
    means an agent started this process, unless this process is a ``hermes`` that set
    it itself (``hermes_constants.inherited_agent_marker``). Fails CLOSED: a context
    that cannot be classified (e.g. a broken import) is treated as an agent.
    """
    try:
        import hermes_constants
        from tools import approval_context as ctx
        from utils import env_var_enabled

        if ctx._is_cron_approval_context():
            return "a cron job"
        if os.environ.get("HERMES_KANBAN_TASK"):
            return "a kanban worker"
        if ctx._is_single_query_approval_context():
            return "a single-query (-q) run"
        if platform := ctx._get_session_platform():
            return f"a {platform} session"
        if ctx._session_env("HERMES_SESSION_ID") or env_var_enabled("HERMES_GATEWAY_SESSION"):
            return "an agent session"
        started_by_agent = hermes_constants.inherited_agent_marker
        if started_by_agent is None:  # not a `hermes` entry point: any HERMES_AGENT was inherited
            started_by_agent = os.environ.get("HERMES_AGENT", "") != ""
        if started_by_agent:
            return "a process an agent started"
    except Exception as exc:  # fail closed: an unclassifiable caller is refused
        logger.warning("ESTOP: cannot classify the caller; treating it as an agent", exc_info=True)
        return f"an unclassifiable context ({type(exc).__name__})"
    return None


def _chat_control_enabled() -> Optional[bool]:
    """``security.estop_chat_control`` (default True) of the active profile AND the fleet root:
    lifting removes the root sentinel too, so one profile cannot opt in past the root's
    ``false``. None when either config cannot be read."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        from utils import is_truthy_value, load_yaml_file_readonly

        values = [cfg_get(load_config_readonly(), "security", "estop_chat_control", default=True)]
        root_config = _canonical_root() / "config.yaml"
        if root_config.resolve() != (_hermes_home() / "config.yaml").resolve() and root_config.exists():
            root = load_yaml_file_readonly(root_config) or {}
            values.append(cfg_get(root, "security", "estop_chat_control", default=True))
        return all(is_truthy_value(value, default=True) for value in values)
    except Exception:  # undecidable: _require_operator lets the brake engage but refuses a lift
        logger.warning("ESTOP: cannot read security.estop_chat_control", exc_info=True)
        return None


_CHAT_COMMAND = {"pause": "/pause", "resume": "/pause off"}


def _require_operator(action: str, *, from_chat: bool) -> None:
    """Raise :class:`EstopRefused` unless an operator is asking to *action* the stop."""
    if from_chat:
        enabled = _chat_control_enabled()
        if enabled or (enabled is None and action == "pause"):
            return
        why = ("chat control of the emergency stop is disabled (security.estop_chat_control)" if enabled is False
               else "security.estop_chat_control cannot be read")
        hint = f"Run `hermes {action}` in a terminal on the Hermes host."
    else:
        context = _agent_context()
        if context is None:
            return
        why = f"called from {context}"
        chat = f" or send {_CHAT_COMMAND[action]} in chat" if _chat_control_enabled() else ""
        hint = f"Ask an operator to run `hermes {action}` in their own terminal{chat}."
    logger.warning("ESTOP %s refused: %s", action, why)
    raise EstopRefused(f"refusing to {action} Hermes: {why}. Only an operator can {action} Hermes. {hint}")


def is_engaged() -> bool:
    """True if ANY candidate sentinel exists; fail SAFE (True) on stat errors."""
    saw_stat_error = False
    for path in _candidate_sentinel_paths():
        try:
            if path.exists():
                return True
        except OSError:
            saw_stat_error = True
    return saw_stat_error


def engage(reason: Optional[str] = None, *, from_chat: bool = False) -> Path:
    """Create the ESTOP sentinel. Idempotent; re-engaging updates the file.

    Raises :class:`EstopRefused` for an agent caller; ``from_chat`` marks the gateway's
    ``/pause`` command, which ``security.estop_chat_control`` governs instead."""
    _require_operator("pause", from_chat=from_chat)
    path = sentinel_path()
    payload = {"engaged_at": datetime.now(UTC).isoformat(), "reason": reason or None}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        with suppress(OSError):  # Best effort: an empty/partial sentinel still pauses (fail safe).
            path.touch(exist_ok=True)
    return path


def disengage(*, from_chat: bool = False) -> bool:
    """Remove every visible sentinel (process-local and fleet-root).

    Guarded like :func:`engage`: this lifts the fleet-root sentinel too, so an
    unguarded caller could undo an operator's fleet-wide halt."""
    _require_operator("resume", from_chat=from_chat)
    lifted = False
    for path in _candidate_sentinel_paths():
        try:
            path.unlink()
            lifted = True
        except (OSError, AttributeError):
            continue
    return lifted


def get_state() -> Optional[dict]:
    """Return ``{"reason", "engaged_at"}`` or None when not engaged; an unreadable/corrupt
    body still reports engaged with both fields None."""
    if not is_engaged():
        return None
    state = {"reason": None, "engaged_at": None}
    found = False
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            return state
        except AttributeError:
            continue
        found = True
        with suppress(OSError, ValueError, AttributeError):
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                state = {"reason": raw.get("reason") or None, "engaged_at": raw.get("engaged_at") or None}
                break
    return state if found else None


def paused_reply() -> Optional[str]:
    """Short user-facing notice for new gateway turns, or None if not paused."""
    state = get_state()
    if state is None:
        return None
    tag = f" ({state['reason']})" if state.get("reason") else ""
    return f"⏸️ Hermes is paused{tag}. New work is on hold; run `hermes resume` to pick things back up."


def check_paused(component: str, logger: logging.Logger) -> bool:
    """Return True when engaged, logging once per engagement per component (re-armed after a resume)."""
    if not is_engaged():
        with _log_lock:
            _logged_components.discard(component)
        return False
    with _log_lock:
        first = component not in _logged_components
        _logged_components.add(component)
    if first:
        reason = (get_state() or {}).get("reason")
        suffix = f" (reason: {reason})" if reason else ""
        logger.info(
            "%s dispatch paused by global emergency stop%s — remove with `hermes resume` (%s)",
            component, suffix, sentinel_path(),
        )
    return True
