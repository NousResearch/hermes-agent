"""Skill-prune / ghost-skill defense helpers extracted from context_compressor.

Ghost-skill defense (#32106): when compaction reduces an old ``skill_view``
result to a 1-line metadata summary, the model still believes the skill is
loaded even though its instructions are gone. This module owns the canonical
prune marker, extraction, re-injection, and protected-name collection.

Part of #78645 + #78647.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from agent.context_compressor_text_utils import (
    _content_text_for_contains,
    _redact_compaction_text,
)



# Ghost-skill defense (#32106): when compaction reduces an old ``skill_view``
# result to a 1-line metadata summary, the model still believes the skill is
# loaded even though its instructions are gone. The marker below is the ONE
# canonical prune signal — ``_skill_pruned_marker()`` builds it and every
# presence check matches against the same string, so the emit side and the
# check side can never drift apart (the original PR #44166 emitted
# ``[SKILL_PRUNED:`` but presence-checked ``[SKILL_PRUNED]``, making
# re-injection fire even when the marker had survived).
SKILL_PRUNED_MARKER_PREFIX = "[SKILL_PRUNED:"
# skill_view results at or below this size stay verbatim in pruned
# summaries — small skills are cheap to keep and their loss is unlikely to
# ghost the model. Shared by the emit site and the summarizer-input scan.
_SKILL_VIEW_PRUNE_MIN_CHARS = 5000
# Cap for the deterministic marker re-injection list — keeps a very long
# session from growing an unbounded "## Pruned Skills" block in every
# iterative summary update. Newest-referenced skills win.
_MAX_PRUNED_SKILL_MARKERS = 20


def _skill_pruned_marker(skill_name: str) -> str:
    """Return the canonical prune marker for *skill_name*.

    Used verbatim by BOTH the emit sites (tool-result summarization,
    summary re-injection) and the survival check in
    ``_reinject_pruned_skill_markers`` — one string, no drift.
    """
    return (
        f"{SKILL_PRUNED_MARKER_PREFIX} content lost in compression; "
        f"reload with skill_view(name='{skill_name}')]"
    )


# Matches the canonical marker and captures the skill name. Anchored on the
# shared prefix constant so a wording change to the marker body updates the
# emit helper and this extractor together.
_SKILL_PRUNED_MARKER_RE = re.compile(
    re.escape(SKILL_PRUNED_MARKER_PREFIX)
    + r"[^\]]*?reload with skill_view\(name='([^']+)'\)"
)


def _extract_pruned_skill_names(text: str) -> list[str]:
    """Return skill names referenced by prune markers in *text*, in order."""
    names: list[str] = []
    for match in _SKILL_PRUNED_MARKER_RE.finditer(text or ""):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def _collect_ghosted_skill_names(turns: List[Dict[str, Any]]) -> list[str]:
    """Skill names whose instructions are about to be lost in compaction.

    Covers BOTH shapes a compacted middle window can carry:

    - a ``skill_view`` result already demoted by Phase-1 pruning — the
      canonical ``[SKILL_PRUNED: ...]`` marker is in the row content;
    - a RAW ``skill_view`` body that was never demoted (it sat inside the
      protected tail of an earlier prune, then aged into the compression
      window). The summarizer will paraphrase the instructions away, which
      is exactly the ghost-skill failure — so it needs a marker too.
    """
    names: list[str] = []

    def _add(name: str) -> None:
        if name and name not in names:
            names.append(name)

    call_id_to_skill: dict[str, str] = {}
    for idx, skill in _skill_view_call_sites(turns):
        msg = turns[idx]
        for tc in msg.get("tool_calls") or []:
            tc_fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            tc_name = tc_fn.get("name", "") if isinstance(tc_fn, dict) else getattr(tc_fn, "name", "")
            if tc_name != "skill_view":
                continue
            cid = tc.get("id", "") if isinstance(tc, dict) else (getattr(tc, "id", "") or "")
            if cid:
                call_id_to_skill[cid] = skill
    for msg in turns:
        content = msg.get("content")
        text = content if isinstance(content, str) else _content_text_for_contains(content)
        for name in _extract_pruned_skill_names(text):
            _add(name)
        if (
            msg.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > _SKILL_VIEW_PRUNE_MIN_CHARS
        ):
            skill = call_id_to_skill.get(str(msg.get("tool_call_id") or ""))
            if skill:
                _add(skill)
    return names


_PRUNED_SKILLS_SECTION_HEADING = "## Pruned Skills"


def _reinject_pruned_skill_markers(summary: str, skill_names: list[str]) -> str:
    """Deterministically restore prune markers the summarizer dropped.

    ``skill_names`` was extracted from the summarizer INPUT before the LLM
    call. For every skill whose canonical marker (``_skill_pruned_marker``)
    is absent from the model's output, append it under a ``## Pruned
    Skills`` section. Presence is checked against the SAME canonical string
    the emit sites produce — a paraphrased or renamed marker counts as
    dropped and is restored (the original PR checked the literal
    ``[SKILL_PRUNED]``, which never matches the emitted ``[SKILL_PRUNED:``
    form, so it duplicated markers that HAD survived).

    The appended block is plain body text: it never carries a handoff
    prefix, the merged-summary delimiter, or a start-of-content scaffolding
    marker, so ``classify_summary_content`` / todo-snapshot flag handling
    are unaffected. The block is routed through ``_redact_compaction_text``
    like every other compaction-boundary text.
    """
    if not skill_names:
        return summary
    missing = [
        name for name in skill_names
        if _skill_pruned_marker(name) not in summary
    ]
    if not missing:
        return summary
    lines = [_skill_pruned_marker(name) for name in missing]
    block = (
        "\n\n" + _PRUNED_SKILLS_SECTION_HEADING + "\n"
        + "\n".join(lines)
        + "\n(The listed skills' instructions were pruned during context "
        "compression. Reload with the skill_view call in each marker before "
        "relying on that skill; one reload per skill is enough — ignore any "
        "older markers for the same skill.)"
    )
    return summary + _redact_compaction_text(block)


# A skill_view call within this many trailing messages counts as "just
# loaded": its full instruction body must survive the Phase-1 prune even when
# the token-budget boundary would otherwise demote it (#32106). Distinct from
# the protected-tail boundary, which is token-based and can land immediately
# after a bulky just-loaded skill body.
_SKILL_PRUNE_RECENT_WINDOW = 10


def _skill_view_call_sites(
    messages: List[Dict[str, Any]],
) -> list[tuple[int, str]]:
    """Yield ``(message_index, skill_name)`` for every skill_view tool call."""
    sites: list[tuple[int, str]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                args_str = fn.get("arguments", "") if isinstance(fn, dict) else ""
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn else ""
                args_str = getattr(fn, "arguments", "") if fn else ""
            if name != "skill_view" or not isinstance(args_str, str) or not args_str:
                continue
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(args, dict):
                skill = args.get("name", "")
                if isinstance(skill, str) and skill:
                    sites.append((i, skill))
    return sites


def _collect_protected_skill_names(
    messages: List[Dict[str, Any]], prune_boundary: int,
) -> set[str]:
    """Skill names whose skill_view bodies must survive Phase-1 demotion.

    A skill is protected (lower-cased set) when any of these hold:

    - its most recent ``skill_view`` call sits within the last
      ``_SKILL_PRUNE_RECENT_WINDOW`` messages (just loaded / just reloaded);
    - its most recent ``skill_view`` call sits inside the protected tail
      (at or after *prune_boundary*);
    - its name is mentioned in a user message inside the protected tail
      (the user is actively steering work that depends on it).

    Protection applies to the ordinary Phase-1/2 prune only. The Pass-4
    pressure demotion deliberately ignores it: when the protected region
    itself exceeds the soft budget, exempting skill bodies would recreate
    the #61932 dead-end shape.
    """
    total = len(messages)
    if not total:
        return set()
    recent_start = max(0, total - _SKILL_PRUNE_RECENT_WINDOW)
    tail_start = max(0, prune_boundary)
    tail_user_texts: list[str] = []
    for msg in messages[tail_start:]:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            tail_user_texts.append(content.lower())
    protected: set[str] = set()
    for idx, skill in _skill_view_call_sites(messages):
        key = skill.lower()
        if idx >= recent_start or idx >= tail_start:
            protected.add(key)
        elif any(key in text for text in tail_user_texts):
            protected.add(key)
    return protected
