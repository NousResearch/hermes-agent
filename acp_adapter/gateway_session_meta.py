"""Editor session metadata on the gateway path, projected from the gateway's existing reads.

Main's in-process server read these off its live agent (``server.py`` ``_session_response_fields``,
``_build_usage_update``, ``_send_session_info_update``); here every figure comes from a snapshot's
``info`` (``session.info``), ``session.stats``, a ``message.complete`` ``usage`` or ``state.db``.
"""
from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path

from acp.schema import (
    AvailableCommand, AvailableCommandsUpdate, SessionConfigOptionSelect, SessionConfigSelectOption,
    SessionInfoUpdate, SessionMode, SessionModeState, SessionModelState, UnstructuredCommandInput, Usage, UsageUpdate,
)

from acp_adapter.modes import (
    DEFAULT_EDIT_APPROVAL_POLICY, DEFAULT_MODE, EDIT_APPROVAL_POLICY_CONFIG_ID, EDIT_APPROVAL_POLICY_TO_MODE, MODES,
)

logger = logging.getLogger(__name__)


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


def _skill_command_entries() -> dict[str, str]:
    """Installed skills as {name: description} for /<skill-name> autocomplete.
    Local patch: CLI/gateway expose installed skills as dynamic /<name> commands;
    the gateway editor path historically advertised only COMMANDS. Empty on any
    API mismatch so the gateway path never breaks."""
    try:
        from agent.skill_commands import get_interactive_skill_commands
        commands = get_interactive_skill_commands()
    except Exception:
        logger.debug("Could not enumerate skill commands for gateway ACP", exc_info=True)
        return {}
    taken = set(COMMANDS)  # built-ins win, same rule as CLI/gateway
    entries: dict[str, str] = {}
    for key, meta in (commands or {}).items():
        name = (key or "").lstrip("/").strip()
        if not name or name in taken or name in entries:
            continue
        desc = " ".join(str((meta or {}).get("description") or f"Invoke the {name} skill").split())
        entries[name] = desc[:117] + "..." if len(desc) > 120 else desc
    return entries


def available_commands_update() -> AvailableCommandsUpdate:
    commands = [
        AvailableCommand(name=name, description=description,
                         input=UnstructuredCommandInput(hint=hint) if hint else None)
        for name, (description, hint) in COMMANDS.items()]
    commands.extend(AvailableCommand(name=name, description=desc)
                    for name, desc in _skill_command_entries().items())
    return AvailableCommandsUpdate(session_update="available_commands_update", available_commands=commands)


def help_text() -> str:
    lines = ["Available commands:", ""]
    lines.extend(f"  /{name:10s}  {description}" for name, (description, _hint) in COMMANDS.items())
    lines.extend(["", "Other Hermes commands run on the gateway, which answers them or says why not.",
                  "Unrecognized /commands are sent to the model as normal messages."])
    return "\n".join(lines)


def current_mode(info: dict) -> str:
    """The mode of the session's edit approval policy (``session.info``); an owner without the field asks."""
    return EDIT_APPROVAL_POLICY_TO_MODE.get(info.get("edit_approval_policy") or DEFAULT_EDIT_APPROVAL_POLICY,
                                            DEFAULT_MODE)


def session_modes(info: dict | None = None) -> SessionModeState:
    return SessionModeState(current_mode_id=current_mode(info or {}), available_modes=[
        SessionMode(id=mode, name=name, description=description) for mode, (_p, name, description) in MODES.items()])


REASONING_CONFIG_ID = "reasoning_effort"
SERVICE_TIER_CONFIG_ID = "service_tier"
SERVICE_TIER_LABELS = {"normal": "Normal", "fast": "Fast", "auto": "Auto", "cold": "Cold", "ultrafast": "Ultrafast"}


def reasoning_choices() -> tuple[str, ...]:
    """The owner's ``reasoning`` levels; ``default`` hands the level back to the profile."""
    from hermes_constants import VALID_REASONING_EFFORTS
    return ("default", "none", *VALID_REASONING_EFFORTS)


def config_options(models: SessionModelState | None, info: dict, mode_id: str) -> list:
    """Zed renders its composer pickers only from ``configOptions``: the edit mode, the model
    (``set_session_model``'s picker ids) and the reasoning effort (``session.info``'s level)."""
    def select(config_id, name, category, current, choices):
        return SessionConfigOptionSelect(id=config_id, name=name, category=category, type="select", current_value=current,
                                         options=[SessionConfigSelectOption(value=v, name=n) for v, n in choices])
    options = [select(EDIT_APPROVAL_POLICY_CONFIG_ID, "Edits", "mode", MODES[mode_id][0],
                      [(policy, name) for policy, name, _ in MODES.values()])]
    if models is not None:
        options.append(select("model", "Model", "model", models.current_model_id,
                              [(m.model_id, m.name) for m in models.available_models]))
    # ``session.info`` reports the level in force, never ``default`` (a reset session reads the profile's), so
    # ``default`` is offered only while no level is set: a pick never shows as another value than the one chosen.
    levels = reasoning_choices()[1:]
    effort = info.get("reasoning_effort") if info.get("reasoning_effort") in levels else "default"
    options.append(select(REASONING_CONFIG_ID, "Reasoning effort", "thought_level", effort,
                          [(v, v) for v in (levels if effort in levels else ("default", *levels))]))
    return options


async def session_config_options(client, session_id, models, info, mode_id):
    """Project the owner's eligible speed choices alongside the existing ACP pickers."""
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.models import model_supports_fast_mode
    options = config_options(models, info, mode_id)
    if not model_supports_fast_mode(info.get('model'), provider=info.get('provider'), base_url=info.get('base_url')):
        return options
    try:
        speed = await client.rpc('config.get', session_id=session_id, key='service_tier')
    except GatewayRPCError as exc:
        if str(exc) != 'invalid_params':
            raise
        return options  # Older owners do not serve the speed-choice read.
    choices = speed.get('options') or []
    if choices:
        current = speed['value']
        rows = [SessionConfigSelectOption(value=value, name=SERVICE_TIER_LABELS[value]) for value in choices]
        if current not in choices:
            rows.append(SessionConfigSelectOption(value=current,
                        name=f"{SERVICE_TIER_LABELS.get(current, current)} (unavailable on this route)"))
        options.append(SessionConfigOptionSelect(id=SERVICE_TIER_CONFIG_ID, name='Service tier', category='other',
                                                type='select', current_value=current, options=rows))
    return options


def model_state(info: dict) -> SessionModelState | None:
    """The picker for ``session.info`` from shared inventory, with a current-model fallback. Blocking."""
    from acp_adapter.auth import detect_provider
    from acp_adapter.model_catalog import build_model_state, fallback_model_state
    model = str(info.get("model") or "").strip()
    provider = info.get("provider") or detect_provider() or "openrouter"
    try:
        picker = build_model_state(model, provider, "")
        if picker is not None:
            return picker
    except Exception:
        # Boundary: a broken inventory must not fail session/new; the current model still shows.
        logger.debug("Could not build ACP model state", exc_info=True)
    return fallback_model_state(model, provider)


def provenance_meta(home: Path, acp_session_id: str, session_id: str) -> dict | None:
    """``_meta.hermes.sessionProvenance`` for the editor's id, the logical session's current
    compression tip and its lineage (``acp_adapter/provenance.py``), read from the store of the
    profile in ``home``. Blocking."""
    from acp_adapter.provenance import session_provenance_meta
    from persistence.core.errors import STORE_FAILURES
    from persistence_assembly import open_persistence_reader, profile_scope, store_exists
    scope = profile_scope(home=Path(home))
    if not store_exists(scope):
        return None
    try:
        with open_persistence_reader(scope.profile_dir) as persistence:
            return session_provenance_meta(persistence, acp_session_id, persistence.lineage.tip(session_id))
    except STORE_FAILURES:
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
