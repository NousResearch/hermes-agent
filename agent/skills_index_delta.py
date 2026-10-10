"""Skills that change mid-conversation reach the model without a prompt rebuild.

The ``<available_skills>`` index lives in the system prompt, and a conversation keeps those bytes
for its whole life so the provider prefix cache stays warm (#104414). A skill installed, created or
removed after the prompt was built (hub install, ``skill_manage``, the curator, a plugin going
live, a change made by another process) therefore never reached the index the model routes on
until compaction rebuilt the prompt. Every turn compares the index the prompt carries with the
index the session would build now and delivers the difference on the per-turn user-message note
channel, behind the cached prefix (ported from cloudflare/cloudflare-os#267).

Each note states the FULL difference against the prompt, so the prompt plus the newest note is the
complete record of what the model was told. A fresh agent (the gateway builds one per turn) reads that record back
from the transcript once; a long-lived agent (CLI, TUI, Desktop) keeps it on itself. A note is sent
only when the difference changes.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent.message_content import flatten_message_text

logger = logging.getLogger("run_agent")

_NOTE_PREFIX = "[System: The skills available in this conversation changed after its system prompt was built."
_ADDED_HEAD = "Now available (load with skill_view(name) when relevant, like the skills listed in the system prompt):"
_REMOVED_HEAD = "No longer available:"
_ACCURATE_AGAIN = " The skills listed in the system prompt are accurate again; earlier skill-change notes are superseded."
_ROW = "    - "

_BLOCK_RE = re.compile(r"<available_skills>\n?(.*?)</available_skills>", re.DOTALL)
_NAMES_ONLY_RE = re.compile(r"\[names only\]:\s*(.+)$")

_Delta = tuple[frozenset, frozenset]


def _row_name(line: str) -> str:
    """Skill name of an index row (``    - name: description`` / ``    - name``). Split on ``": "``:
    plugin skills are qualified (``plugin:skill``), so a bare ``":"`` is part of the name."""
    return line.strip()[2:].split(": ", 1)[0].strip()


def index_entries(prompt: Any) -> dict[str, str]:
    """``{name: index row}`` for every skill in ``prompt``'s ``<available_skills>`` block; names-only
    category rows (focus mode) map to ``""``. ``{}`` without a block."""
    match = _BLOCK_RE.search(prompt) if isinstance(prompt, str) else None
    entries: dict[str, str] = {}
    for line in match.group(1).splitlines() if match else ():
        if line.strip().startswith("- ") and (name := _row_name(line)):
            entries.setdefault(name, line.rstrip())
        elif names_only := _NAMES_ONLY_RE.search(line):
            for name in filter(None, (n.strip() for n in names_only.group(1).split(","))):
                entries.setdefault(name, "")
    return entries


def _render(added: dict[str, str], removed: set[str]) -> str:
    if not added and not removed:
        return f"{_NOTE_PREFIX}{_ACCURATE_AGAIN}]"
    parts = [_NOTE_PREFIX]
    if added:
        parts += [_ADDED_HEAD, *(added[name] or f"{_ROW}{name}" for name in sorted(added))]
    if removed:
        parts.append(f"{_REMOVED_HEAD} {', '.join(sorted(removed))}")
    return "\n".join(parts) + "]"


def _parse(note: str) -> _Delta:
    """Inverse of :func:`_render` for the text that follows ``_NOTE_PREFIX``."""
    lines = note.split("\n")
    added: set[str] = set()
    if _ADDED_HEAD in lines[1:2]:
        for line in lines[2:]:
            if not line.startswith(_ROW):
                break
            added.add(_row_name(line.removesuffix("]")))
    removed: set[str] = set()
    for line in lines[1:]:
        if line.startswith(_REMOVED_HEAD):
            tail = line[len(_REMOVED_HEAD):].removesuffix("]")
            removed = {n.strip() for n in tail.split(",") if n.strip()}
            break
    return frozenset(added), frozenset(removed)


def _announced(conversation_history: Any) -> _Delta:
    """The difference named by the newest skills note anywhere in the transcript: a note far back
    still describes the current state when nothing changed since it was sent."""
    for msg in reversed(conversation_history or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        sidecar = msg.get("api_content")
        text = (sidecar if isinstance(sidecar, str) else "") + "\n" + flatten_message_text(msg.get("content"))
        if _NOTE_PREFIX in text:
            return _parse(text.rsplit(_NOTE_PREFIX, 1)[1])
    return frozenset(), frozenset()


def stage_skills_index_note(agent: Any, prompt: Any, conversation_history: Any) -> bool:
    """Stage a one-shot note when the skills this session can load differ from ``prompt``'s index
    by something the model has not been told yet. Returns whether it staged.

    A prompt without an index is never compared: it has no baseline (skills tools off, a seeded or
    embedder-supplied prompt), and announcing the whole catalog there would only add noise. MoA and
    codex_app_server turns never stamp the ``api_content`` sidecar, so the note could not be read
    back from the transcript; those modes skip it, like the surface-switch note."""
    if getattr(agent, "provider", None) == "moa" or getattr(agent, "api_mode", None) == "codex_app_server":
        return False
    stored = index_entries(prompt)
    if not stored:
        return False
    try:
        from agent.system_prompt import _skills_prompt
        current = index_entries(_skills_prompt(agent))
    except Exception:
        logger.debug("skills index delta: current index unavailable", exc_info=True)
        return False
    told = getattr(agent, "_skills_index_told", None)
    key = hash(prompt)
    told_added, told_removed = told[1] if isinstance(told, tuple) and told[0] == key else _announced(conversation_history)
    # What the model believes it can load: the prompt's index as amended by the newest note. A rebuilt
    # prompt (compaction) that now lists what an earlier note announced needs no further note.
    if (set(stored) | told_added) - told_removed == set(current):
        agent._skills_index_told = (key, (told_added, told_removed))
        return False
    added = {name: row for name, row in current.items() if name not in stored}
    removed = set(stored) - set(current)
    agent._skills_index_told = (key, (frozenset(added), frozenset(removed)))
    agent._skills_index_note = _render(added, removed)
    logger.info(
        "Session %s: skills changed (+%d/-%d) since the system prompt was built; delivering the "
        "change as a turn note (prefix cache preserved).",
        getattr(agent, "session_id", None), len(added), len(removed),
    )
    return True
