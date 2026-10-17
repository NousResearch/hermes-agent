"""Editor session metadata on the gateway path, projected from the gateway's existing reads.

Main's in-process server read these off its live agent (``server.py`` ``_session_response_fields``,
``_build_usage_update``, ``_send_session_info_update``); here every figure comes from a snapshot's
``info`` (``session.info``), ``session.stats``, a ``message.complete`` ``usage`` or ``state.db``.
"""
from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3

from acp.schema import (
    AvailableCommand, AvailableCommandsUpdate, ModelInfo, SessionInfoUpdate, SessionMode, SessionModeState,
    SessionModelState, UnstructuredCommandInput, Usage, UsageUpdate,
)

logger = logging.getLogger(__name__)

# The gateway honours one editor edit policy, ``ask`` (``gateway/session_local_editor.py``), and has
# no verb that changes it on a live session, so main's Accept Edits / Don't Ask are not offered.
DEFAULT_MODE = "default"
DEFAULT_EDIT_APPROVAL_POLICY = "ask"
EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"

# Advertised name -> (description, input hint). ``prompt`` serves each through a gateway verb
# (``acp_adapter/gateway_commands.py``); ``/reset`` is the registry's alias of ``/new``.
COMMANDS: dict[str, tuple[str, str | None]] = {
    "help": ("List available commands", None),
    "model": ("Show current model and provider, or switch models", "model name to switch to"),
    "tools": ("List available tools", None),
    "context": ("Show conversation context usage", None),
    "reset": ("Clear conversation history", None),
    "compress": ("Compress conversation context", None),
    "steer": ("Inject guidance into the currently running agent turn", "guidance for the active turn"),
    "queue": ("Queue a prompt to run after the current turn finishes", "prompt to run next"),
    "usage": ("Show token usage and rate limits", None),
    "status": ("Show session, model, token, and context info", None),
    "version": ("Show Hermes version", None),
    "title": ("Show or set the session title", "new title"),
    "undo": ("Remove the last N user turns and their replies", "number of turns"),
    "retry": ("Resend the last user message", None),
}


def available_commands_update() -> AvailableCommandsUpdate:
    return AvailableCommandsUpdate(session_update="available_commands_update", available_commands=[
        AvailableCommand(name=name, description=description,
                         input=UnstructuredCommandInput(hint=hint) if hint else None)
        for name, (description, hint) in COMMANDS.items()])


def help_text() -> str:
    lines = ["Available commands:", ""]
    lines.extend(f"  /{name:10s}  {description}" for name, (description, _hint) in COMMANDS.items())
    lines.extend(["", "Other Hermes commands run on the gateway, which answers them or says why not.",
                  "Unrecognized /commands are sent to the model as normal messages."])
    return "\n".join(lines)


def session_modes() -> SessionModeState:
    return SessionModeState(current_mode_id=DEFAULT_MODE, available_modes=[
        SessionMode(id=DEFAULT_MODE, name="Default", description="Ask before edits.")])


def model_state(info: dict) -> SessionModelState | None:
    """The picker for the session's model (``session.info``), from the shared inventory; a single
    current-model row when nothing is listable, as main's ``_build_model_state``. Blocking."""
    from acp_adapter.auth import detect_provider
    from acp_adapter.model_catalog import build_model_state, encode_model_choice
    model = str(info.get("model") or "").strip()
    provider = info.get("provider") or detect_provider() or "openrouter"
    try:
        picker = build_model_state(model, provider, "")
        if picker is not None:
            return picker
    except Exception:
        # Boundary: a broken inventory must not fail session/new; the current model still shows.
        logger.debug("Could not build ACP model state", exc_info=True)
    if not model:
        return None
    choice = encode_model_choice(provider, model)
    return SessionModelState(available_models=[ModelInfo(model_id=choice, name=model)], current_model_id=choice)


def provenance_meta(db_path: Path, acp_session_id: str, session_id: str) -> dict | None:
    """``_meta.hermes.sessionProvenance`` for the editor's id, the logical session's current
    compression tip and its lineage (``acp_adapter/provenance.py``). Blocking."""
    from acp_adapter.provenance import session_provenance_meta
    from hermes_state import SessionDB
    if not Path(db_path).exists():
        return None
    try:
        with SessionDB(db_path=db_path, read_only=True) as db:
            return session_provenance_meta(db, acp_session_id, db.get_compression_tip(session_id) or session_id)
    except sqlite3.Error:
        # Best-effort, as main's ``_provenance_meta``: an unreadable store must not fail the attach.
        logger.debug("Could not build ACP session provenance for %s", session_id, exc_info=True)
        return None


# ``stats`` below is the owner's usage dict (``gateway/session_usage.py::usage_stats``): the
# ``session.usage`` event published after every API call and at turn end, or ``session.stats``.

def usage_update(stats: dict) -> UsageUpdate | None:
    """Zed's context indicator: ``size`` the context window, ``used`` its current occupancy."""
    size = stats.get("context_length")
    if type(size) is not int or size <= 0:
        return None
    return UsageUpdate(session_update="usage_update", size=size, used=max(int(stats.get("context_tokens") or 0), 0))


def turn_usage(stats) -> Usage | None:
    """``PromptResponse.usage``: the agent's session counters, the figures main's turn result carried."""
    if not isinstance(stats, dict):
        return None
    return Usage(input_tokens=int(stats.get("input_tokens") or 0), output_tokens=int(stats.get("output_tokens") or 0),
                 total_tokens=int(stats.get("total_tokens") or 0),
                 thought_tokens=stats.get("reasoning_tokens") or None,
                 cached_read_tokens=stats.get("cache_read_tokens") or None)


def title_update(title: str) -> SessionInfoUpdate:
    # ``sessions`` has no ``updated_at``; "now" is right since this fires when the title changed.
    return SessionInfoUpdate(session_update="session_info_update", title=title,
                             updated_at=datetime.now(UTC).isoformat())
