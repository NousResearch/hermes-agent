"""Surface switch without a prompt rebuild (#104414).

A surface switch (desktop <-> TUI, a session resumed under a different host) changes only which
interface renders the reply, but the surface guidance and the ``Platform:`` trailer are embedded
in the persisted system prompt.  Rebuilding for it diverged the prompt within its first blocks,
so the ENTIRE request behind it re-prefilled — a 220K-token session came back at a 1% cache hit.
The stored bytes are therefore kept and the CURRENT surface's guidance is delivered on the
per-turn user-message channel instead: that lands after the cached prefix and is stamped into
the byte-stable ``api_content`` sidecar so later turns replay it unchanged.  The prompt itself
converges at the next rebuild boundary (compaction).
"""
from __future__ import annotations

import logging
from typing import Any, List

from agent.message_content import flatten_message_text
from agent.prompt_builder import RUNTIME_ENVIRONMENT_END, RUNTIME_ENVIRONMENT_HEADING

logger = logging.getLogger("run_agent")

_SURFACE_SWITCH_NOTE_PREFIX = "[System: This conversation is now being answered on a different interface: "
# Closes the surface name in the note; platform names are free-form for plugin platforms, so the
# terminator (not ".") delimits the parse.
_SURFACE_NAME_END = " — any earlier interface guidance"
# Only the newest note matters, and the note is re-stamped on the switch turn, so a bounded tail
# scan is enough; without a bound every turn of a never-switched session walks the whole transcript.
_NOTE_SCAN_TAIL = 200


def split_runtime_boundary(prompt: str) -> tuple:
    """``(identity, runtime_marker, runtime)`` of a persisted prompt.  Legacy prose may quote
    the runtime heading, but only the new renderer ENDS in the boundary; when the marker is
    empty the whole prompt is identity."""
    identity, runtime_marker, runtime = prompt.rpartition(f"\n\n{RUNTIME_ENVIRONMENT_HEADING}\n\n")
    return (identity, runtime_marker, runtime) if prompt.endswith(RUNTIME_ENVIRONMENT_END) else (prompt, "", "")


def identity_line_value(prompt: str, label: str) -> str:
    """Last ``Label: value`` line in the identity portion (the final runtime block is embedder
    prose, never identity).  Last match wins — safe only for the volatile-tier trailer fields."""
    prefix = f"{label}:"
    matches = [line[len(prefix):].strip() for line in split_runtime_boundary(prompt)[0].splitlines()
               if line.startswith(prefix)]
    return matches[-1] if matches else ""


def _last_announced_surface(conversation_history: Any) -> str:
    """The surface named by the NEWEST switch note in the transcript ("" when none).

    Once a switch has been announced, that note — not the stored prompt's ``Platform:`` trailer —
    is the last thing the model was told it runs on.  Reading it back is also what keeps a fresh
    AIAgent per turn (the gateway shape) from stacking one copy of the note per turn."""
    for msg in reversed((conversation_history or [])[-_NOTE_SCAN_TAIL:]):
        # The note only ever lands on a user row: in its api_content sidecar, or as a text part
        # when the content is a multimodal list (which cannot take the string sidecar).
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        sidecar = msg.get("api_content")
        text = (sidecar if isinstance(sidecar, str) else "") + "\n" + flatten_message_text(msg.get("content"))
        if _SURFACE_SWITCH_NOTE_PREFIX in text:
            tail = text.rsplit(_SURFACE_SWITCH_NOTE_PREFIX, 1)[1]
            return tail.split(_SURFACE_NAME_END, 1)[0].strip()
    return ""


def note_inert_pinned_tools(agent: Any, built_for_this_surface: List[str]) -> None:
    """Name, at the end of the staged note, the pinned tools THIS surface did not build.

    The freeze keeps a previous surface's tools on the wire deliberately — removing them is the one
    thing that would still re-prefill the request behind the preserved prompt — so the model has
    to be TOLD they are inert here, or it plans around a ``focus_pane`` a terminal turn can only
    answer with ``tool_error("desktop only")``."""
    from tools.mcp_tool_agent import agent_tool_names
    surface_names = set(built_for_this_surface)
    inert = [name for name in agent_tool_names(agent) if name not in surface_names]
    note = getattr(agent, "_surface_switch_note", "") or ""
    if not inert or not note:
        return
    agent._surface_switch_note = note + (
        "\n[System: These tools stay listed for this conversation (dropping them would discard "
        "the cached request prefix) but were not loaded for this interface — expect a call to "
        f"one of them to fail: {', '.join(inert)}.]"
    )


def stage_surface_switch_note(agent: Any, prompt: str, conversation_history: Any) -> bool:
    """Stage a one-shot correction when the request would otherwise misdescribe the surface.

    Compares the runtime surface against what the model was last told — the newest switch note
    if one exists, else ``prompt``'s own ``Platform:`` trailer.  Consulting the note matters in
    both directions: switching BACK to the prompt's surface (desktop -> tui -> desktop) leaves the
    trailer agreeing with the runtime while a stale note still says otherwise, and a rebuild for
    an unrelated reason (a model switch) refreshes the prompt but not that note.  The full surface
    guidance is attached only when the prompt itself is out of date; otherwise the note just
    retires the stale one.  Returns whether it staged.

    MoA and codex_app_server turns never stamp the ``api_content`` sidecar, so the note could not
    be read back and would be re-sent every turn; those modes keep the stored prompt and skip the
    note entirely."""
    if getattr(agent, "provider", None) == "moa" or getattr(agent, "api_mode", None) == "codex_app_server":
        return False
    current = str(getattr(agent, "platform", "") or "").strip()
    if not current:
        return False
    described = identity_line_value(prompt, "Platform")
    told = _last_announced_surface(conversation_history) or described
    if not told or told == current:
        return False
    from agent.system_prompt import platform_hint
    hint = platform_hint(agent) if described != current else ""
    where = "the guidance below" if hint else "the interface section in the system prompt above"
    note = (
        f"{_SURFACE_SWITCH_NOTE_PREFIX}{current}{_SURFACE_NAME_END} in this conversation is "
        f"superseded — follow {where} for formatting, file delivery and any interface-specific "
        "capability.]"
    )
    agent._surface_switch_note = f"{note}\n{hint}" if hint else note
    logger.info(
        "Session %s switched surface %s -> %s; keeping the stored system prompt and delivering "
        "the new surface guidance as a turn note (prefix cache preserved).",
        agent.session_id, told, current,
    )
    return True
