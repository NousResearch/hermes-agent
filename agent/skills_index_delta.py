"""Skills index drift without a prompt rebuild.

The ``<available_skills>`` routing index is embedded in the persisted system prompt, and a
continuing session reuses those bytes verbatim so the provider prefix cache stays warm (#104414).
A skill that appears or disappears mid-conversation (a hub install, ``skill_manage`` from another
session, an org sync, the curator) therefore never reaches the index the model actually routes on
until compaction rebuilds the prompt; ``skills_list`` is live, but the prompt tells the model to
scan the index, not to call it.  The delta is delivered on the per-turn user-message channel
instead: it lands behind the cached prefix, is stamped into the byte-stable ``api_content``
sidecar, and is re-staged only when the delta itself changes (ported from
cloudflare/cloudflare-os#267, whose chat cached its skill catalog once and never saw a new skill).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Set, Tuple

from agent.message_content import flatten_message_text

logger = logging.getLogger("run_agent")

_SKILLS_INDEX_NOTE_PREFIX = "[System: The skills index in the system prompt is out of date for this conversation."
_ADDED_HEAD = "Skills added since it was built (load with skill_view(name) when relevant, like the index above):"
_REMOVED_HEAD = "Skills no longer available:"
_NOTE_END = "]"
_NOTE_SCAN_TAIL = 200

_BLOCK_RE = re.compile(r"<available_skills>\n?(.*?)</available_skills>", re.DOTALL)
_ENTRY_RE = re.compile(r"^\s*- ([^:\n]+?)(?::|$)", re.MULTILINE)
_NAMES_ONLY_RE = re.compile(r"\[names only\]:\s*(.+)$", re.MULTILINE)


def index_entries(prompt: str) -> Dict[str, str]:
    """``{name: index line}`` for every skill the ``<available_skills>`` block of ``prompt`` lists.

    Names-only category lines (focus mode) contribute the name with an empty line; ``{}`` when
    the prompt has no block at all.
    """
    m = _BLOCK_RE.search(prompt or "")
    if not m:
        return {}
    body = m.group(1)
    entries: Dict[str, str] = {}
    for line in body.splitlines():
        em = _ENTRY_RE.match(line)
        if em:
            entries.setdefault(em.group(1).strip(), line.rstrip())
            continue
        nm = _NAMES_ONLY_RE.search(line)
        if nm:
            for name in nm.group(1).split(","):
                if name.strip():
                    entries.setdefault(name.strip(), "")
    return entries


def _last_announced_delta(conversation_history: Any) -> Tuple[Set[str], Set[str]]:
    """``(added, removed)`` named by the NEWEST skills-index note in the transcript.

    Every note carries the FULL delta against the stored prompt, so the newest one is the
    complete statement of what the model has been told; reading it back is what keeps a fresh
    agent per turn (the gateway shape) from re-sending the same note every turn.
    """
    for msg in reversed((conversation_history or [])[-_NOTE_SCAN_TAIL:]):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        sidecar = msg.get("api_content")
        text = (sidecar if isinstance(sidecar, str) else "") + "\n" + flatten_message_text(msg.get("content"))
        if _SKILLS_INDEX_NOTE_PREFIX not in text:
            continue
        note = text.rsplit(_SKILLS_INDEX_NOTE_PREFIX, 1)[1]
        added: Set[str] = set()
        removed: Set[str] = set()
        if _ADDED_HEAD in note:
            added_part = note.split(_ADDED_HEAD, 1)[1].split(_REMOVED_HEAD, 1)[0]
            added = {m.group(1).strip() for m in _ENTRY_RE.finditer(added_part)}
        if _REMOVED_HEAD in note:
            removed_part = note.split(_REMOVED_HEAD, 1)[1].split(_NOTE_END, 1)[0]
            removed = {n.strip() for n in removed_part.split(",") if n.strip()}
        return added, removed
    return set(), set()


def _render_note(added: Dict[str, str], removed: Set[str]) -> str:
    parts = [_SKILLS_INDEX_NOTE_PREFIX]
    if added:
        parts.append(_ADDED_HEAD)
        parts.extend(added[name] or f"    - {name}" for name in sorted(added))
    if removed:
        parts.append(f"{_REMOVED_HEAD} {', '.join(sorted(removed))}")
    return "\n".join(parts) + _NOTE_END


def stage_skills_index_note(agent: Any, stored_prompt: str, conversation_history: Any) -> bool:
    """Stage a one-shot note when the stored prompt's skills index no longer matches the skills
    the session can load right now.  Returns whether it staged.

    Only a prompt that HAS an index is compared (no baseline otherwise), and only against the
    index the current build would emit — the same cached builder the prompt came from, so a turn
    with no skill change costs one LRU hit.  MoA and codex_app_server turns never stamp the
    ``api_content`` sidecar, so the note could not be read back; those modes skip it.
    """
    if getattr(agent, "provider", None) == "moa" or getattr(agent, "api_mode", None) == "codex_app_server":
        return False
    stored = index_entries(stored_prompt)
    if not stored:
        return False
    try:
        from agent.system_prompt import _skills_prompt
        current = index_entries(_skills_prompt(agent))
    except Exception:
        logger.debug("skills index delta: current index unavailable", exc_info=True)
        return False
    added = {name: line for name, line in current.items() if name not in stored}
    removed = set(stored) - set(current)
    if (set(added), removed) == _last_announced_delta(conversation_history):
        return False
    if not added and not removed:
        # Told about a delta that has since been undone (the skill came back / went away again);
        # the stored index is accurate again and the stale note has to be retired.
        agent._skills_index_note = f"{_SKILLS_INDEX_NOTE_PREFIX} It is accurate again: earlier skills-index notes in this conversation are superseded.{_NOTE_END}"
        return True
    agent._skills_index_note = _render_note(added, removed)
    logger.info(
        "Session %s: skills index drifted (+%d/-%d) since the stored prompt was built; keeping the "
        "prompt and delivering the delta as a turn note (prefix cache preserved).",
        getattr(agent, "session_id", None), len(added), len(removed),
    )
    return True
