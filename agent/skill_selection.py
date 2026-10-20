"""Per-turn skill candidate ranking.

The system-prompt catalog stays byte-stable for the life of a conversation.
When ``skills.selection`` is ``shortlist``, this module annotates only the
current user message (API copy). Durable persistence keeps the original text
via the existing ``persist_user_message`` seam.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SELECTION_CATALOG = "catalog"
SELECTION_SHORTLIST = "shortlist"
SELECTION_OFF = "off"
_VALID_MODES = frozenset({SELECTION_CATALOG, SELECTION_SHORTLIST, SELECTION_OFF})
DEFAULT_SELECTION_MODE = SELECTION_CATALOG
DEFAULT_SELECTION_LIMIT = 12
MAX_SELECTION_LIMIT = 40
CANDIDATE_OPEN = "<skill_candidates>"
CANDIDATE_CLOSE = "</skill_candidates>"

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "you", "your", "this", "that", "with", "from",
    "have", "been", "then", "than", "they", "them", "was", "were", "will",
    "just", "into", "also", "only", "over", "such", "when", "what", "which",
    "while", "about", "into", "onto", "each", "very", "more", "some", "able",
})


def _skills_section() -> Dict[str, Any]:
    from agent.skill_utils import _skills_cfg
    cfg = _skills_cfg()
    return cfg if isinstance(cfg, dict) else {}


def resolve_selection_mode(skills_cfg: Optional[Dict[str, Any]]) -> str:
    raw = ""
    if isinstance(skills_cfg, dict):
        raw = str(skills_cfg.get("selection") or "").strip().lower()
    return raw if raw in _VALID_MODES else DEFAULT_SELECTION_MODE


def resolve_selection_limit(skills_cfg: Optional[Dict[str, Any]]) -> int:
    raw = DEFAULT_SELECTION_LIMIT
    if isinstance(skills_cfg, dict) and skills_cfg.get("selection_limit") is not None:
        try:
            raw = int(skills_cfg.get("selection_limit"))
        except (TypeError, ValueError):
            raw = DEFAULT_SELECTION_LIMIT
    return max(1, min(MAX_SELECTION_LIMIT, raw))


def extract_skill_triggers(frontmatter: Dict[str, Any]) -> List[str]:
    raw = frontmatter.get("triggers") if isinstance(frontmatter, dict) else None
    if raw is None and isinstance(frontmatter, dict):
        metadata = frontmatter.get("metadata")
        hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
        raw = hermes.get("triggers") if isinstance(hermes, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _flatten_user_text(user_message: Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        parts: List[str] = []
        for block in user_message:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _stems(token: str) -> set[str]:
    token = token.lower()
    out = {token}
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        out.add(stem)
        if not stem.endswith("e"):
            out.add(stem + "e")
    if token.endswith("ed") and len(token) > 4:
        out.add(token[:-2])
        out.add(token[:-1])
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        out.add(token[:-1])
    return out


def tokenize(text: str, *, min_len: int = 3) -> set[str]:
    stems: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        token = raw.lower()
        if len(token) < min_len or token in _STOPWORDS:
            continue
        stems.update(_stems(token))
    return stems


def _trigger_hit(query_lower: str, triggers: Sequence[str]) -> bool:
    for trigger in triggers:
        phrase = str(trigger).strip().lower()
        if phrase and phrase in query_lower:
            return True
    return False


def score_skill_entry(query_tokens: set[str], query_lower: str, entry: Dict[str, Any]) -> Tuple[int, bool]:
    name = str(entry.get("name") or "").strip()
    if not name:
        return 0, False
    description = str(entry.get("description") or "")
    triggers = entry.get("triggers") or []
    via_trigger = _trigger_hit(query_lower, triggers if isinstance(triggers, (list, tuple)) else [])
    name_tokens = tokenize(name.replace("-", " "))
    desc_tokens = tokenize(description, min_len=4)
    score = 0
    if name.lower() in query_lower:
        score += 8
    score += 3 * len(query_tokens & name_tokens)
    score += len(query_tokens & desc_tokens)
    if via_trigger:
        score += 5
    return score, via_trigger


def rank_skill_candidates(
    query: str,
    entries: Iterable[Dict[str, Any]],
    limit: int = DEFAULT_SELECTION_LIMIT,
) -> List[Dict[str, Any]]:
    query_lower = (query or "").lower()
    query_tokens = tokenize(query)
    ranked: List[Tuple[int, str, Dict[str, Any]]] = []
    for entry in entries:
        score, via_trigger = score_skill_entry(query_tokens, query_lower, entry)
        if score <= 0:
            continue
        name = str(entry.get("name") or "").strip()
        ranked.append((score, name, {
            "name": name,
            "description": str(entry.get("description") or "").strip(),
            "via_trigger": via_trigger,
            "score": score,
        }))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    cap = max(1, min(MAX_SELECTION_LIMIT, int(limit or DEFAULT_SELECTION_LIMIT)))
    return [row[2] for row in ranked[:cap]]


def format_candidate_overlay(candidates: Sequence[Dict[str, Any]]) -> str:
    if not candidates:
        return ""
    lines = [
        CANDIDATE_OPEN,
        "These skills look relevant to this turn. Load each that applies with skill_view(name) before proceeding:",
    ]
    for row in candidates:
        name = row.get("name") or ""
        desc = row.get("description") or ""
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    lines.append(CANDIDATE_CLOSE)
    return "\n".join(lines)


def _append_overlay(user_message: Any, overlay: str) -> Any:
    if isinstance(user_message, str):
        return f"{user_message.rstrip()}\n\n{overlay}"
    if isinstance(user_message, list):
        return [*(user_message), {"type": "text", "text": overlay}]
    return user_message


def collect_skill_index_entries() -> List[Dict[str, Any]]:
    """Visible skill name/description/trigger tuples for the active profile."""
    from agent.skill_utils import (
        extract_skill_description,
        get_all_skills_dirs,
        get_disabled_skill_names,
        get_project_skills_dirs,
        iter_skill_index_files,
        parse_frontmatter,
        skill_matches_environment,
        skill_matches_platform,
    )

    disabled = get_disabled_skill_names()
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    roots = list(get_all_skills_dirs())
    try:
        roots.extend(get_project_skills_dirs())
    except Exception:
        pass
    for root in roots:
        if not root.is_dir():
            continue
        for skill_file in iter_skill_index_files(root, "SKILL.md"):
            try:
                frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not skill_matches_platform(frontmatter) or not skill_matches_environment(frontmatter):
                continue
            name = str(frontmatter.get("name") or skill_file.parent.name).strip()
            if not name or name in disabled or name in seen:
                continue
            seen.add(name)
            entries.append({
                "name": name,
                "description": extract_skill_description(frontmatter),
                "triggers": extract_skill_triggers(frontmatter),
            })
    return entries


def apply_turn_skill_selection(
    user_message: Any,
    persist_user_message: Any,
    *,
    skills_cfg: Optional[Dict[str, Any]] = None,
    catalog: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """Return ``(api_user_message, persist_user_message, candidates)``.

    Catalog/off modes and empty shortlists are no-ops: persist stays untouched
    so existing transcript semantics do not change. When a shortlist is applied
    and persist was omitted, persist becomes the original user content so the
    durable transcript and ``original_user_message`` stay clean.
    """
    if CANDIDATE_OPEN in _flatten_user_text(user_message):
        return user_message, persist_user_message, []
    cfg = skills_cfg if skills_cfg is not None else _skills_section()
    if resolve_selection_mode(cfg) != SELECTION_SHORTLIST:
        return user_message, persist_user_message, []
    query = _flatten_user_text(user_message)
    if not query.strip():
        return user_message, persist_user_message, []
    try:
        entries = list(catalog) if catalog is not None else collect_skill_index_entries()
        candidates = rank_skill_candidates(query, entries, resolve_selection_limit(cfg))
    except Exception:
        logger.debug("skill selection: catalog ranking failed", exc_info=True)
        return user_message, persist_user_message, []
    overlay = format_candidate_overlay(candidates)
    if not overlay:
        return user_message, persist_user_message, []
    clean = persist_user_message if persist_user_message is not None else user_message
    return _append_overlay(user_message, overlay), clean, candidates


def _message_tool_names(messages: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name")
            if name:
                names.add(str(name))
    return names


def log_unused_skill_shortlist(agent: Any, messages: Sequence[Any]) -> None:
    candidates = list(getattr(agent, "_skill_selection_candidates", None) or [])
    if not candidates:
        return
    if "skill_view" in _message_tool_names(messages):
        return
    names = ", ".join(str(row.get("name") or "") for row in candidates if row.get("name"))
    if names:
        logger.info("skill selection: turn completed without skill_view; unused shortlist: %s", names)
