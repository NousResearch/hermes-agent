"""Edge-mode working memory scratchpad and context-flush helpers.

Gated by ``agent.edge_mode`` in config.  Keeps a dense markdown scratchpad
in agent runtime state and integrates with context compression (focus topic +
post-compaction deltas) without mutating the parent's cached system prompt
(unchanged for the root agent). Subagents inherit working memory via copied
runtime state.

API-time user-message injection uses a **turn-frozen snapshot** of the
scratchpad (``begin_edge_turn_injection``) so mid-turn mutations (absorb,
fault lines, compaction deltas) do not rewrite an already-sent user prefix
and break prompt-cache stability across tool rounds.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from agent.context_compressor import LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX

logger = logging.getLogger(__name__)

# Dense header (legacy ### CRITICAL STATE still accepted for absorb / merges)
SCRATCHPAD_HEADER = "### CRIT"
SCRATCHPAD_MARKERS = ("### CRIT", "### CRITICAL STATE")


class EdgeCompressGuardError(RuntimeError):
    """Raised when edge-mode compaction would violate scratchpad/user anchors."""


def persist_edge_scratchpad_now(agent) -> None:
    """Write scratchpad to SessionDB immediately after any programmatic mutation."""
    if not getattr(agent, "edge_mode", False):
        return
    db = getattr(agent, "_session_db", None)
    sid = getattr(agent, "session_id", None) or ""
    if not db or not sid:
        return
    try:
        db.update_edge_working_memory(sid, getattr(agent, "_edge_scratchpad", "") or "")
    except Exception as exc:
        logger.debug("persist_edge_scratchpad_now failed: %s", exc)


def extract_primary_goal_from_scratchpad(scratchpad: str) -> str:
    for pat in (
        r"^\s*-\s*\*\*Goal:\*\*\s*(.+)$",
        r"^\s*-\s*\*\*Primary Goal:\*\*\s*(.+)$",
    ):
        m = re.search(pat, scratchpad or "", re.I | re.MULTILINE)
        if m:
            return (m.group(1) or "").strip()
    return ""


def _normalize_guard_text(s: str) -> str:
    """Collapse whitespace for fuzzy goal / user anchoring (SLM paraphrase)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_edge_compress_guard(agent, messages: List[Any]) -> None:
    """Anti-lobotomy: ensure scratchpad goal still anchors to a user turn."""
    if not getattr(agent, "edge_mode", False):
        return
    sp = getattr(agent, "_edge_scratchpad", "") or ""
    goal = extract_primary_goal_from_scratchpad(sp)
    if not goal:
        raise EdgeCompressGuardError("scratchpad primary goal missing or empty")
    first_user = ""
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            first_user = _content_text(m.get("content"))
            break
    if not first_user.strip():
        raise EdgeCompressGuardError("no user message to anchor primary goal")
    g = goal.strip()
    fu = first_user.strip()
    if g[:80] in fu or fu[:200] in g:
        return
    if len(g) >= 40 and any(g[i : i + 40] in fu for i in range(0, len(g) - 39, 15)):
        return
    g_n = _normalize_guard_text(g)
    fu_n = _normalize_guard_text(fu)
    if g_n[:80] in fu_n or fu_n[:200] in g_n:
        return
    if len(g_n) >= 40 and any(g_n[i : i + 40] in fu_n for i in range(0, len(g_n) - 39, 15)):
        return
    # Word-overlap fallback: tolerates light local-SLM rephrasing of the goal line.
    g_words = {w for w in re.findall(r"[a-z0-9]{4,}", g_n)}
    fu_words = set(re.findall(r"[a-z0-9]{4,}", fu_n))
    if len(g_words) >= 2:
        overlap = g_words & fu_words
        if len(overlap) >= max(2, min(len(g_words), 1 + len(g_words) // 3)):
            return
    raise EdgeCompressGuardError("primary goal/scratchpad diverged from first user message")


def maybe_edge_flush_mid_turn(
    agent,
    messages: list,
    system_message: str,
    current_turn_user_idx: Optional[int],
    effective_task_id: str,
    active_system_prompt: str,
) -> tuple:
    """Optional edge-only mid-turn compaction (assistant rounds / soft token cap)."""
    ar_cap = int(getattr(agent, "_edge_flush_assistant_rounds", 0) or 0)
    tok_cap = int(getattr(agent, "_edge_flush_token_soft_limit", 0) or 0)
    if ar_cap <= 0 and tok_cap <= 0:
        return messages, active_system_prompt
    from agent.model_metadata import estimate_request_tokens_rough

    approx = estimate_request_tokens_rough(
        messages,
        system_prompt=active_system_prompt or "",
        tools=agent.tools or None,
    )
    tail = messages[(current_turn_user_idx or 0) + 1 :] if current_turn_user_idx is not None else messages
    rounds = sum(1 for m in tail if isinstance(m, dict) and m.get("role") == "assistant")
    fire = (ar_cap > 0 and rounds >= ar_cap) or (tok_cap > 0 and approx >= tok_cap)
    if not fire:
        return messages, active_system_prompt
    try:
        return agent._compress_context(
            messages, system_message, approx_tokens=approx, task_id=effective_task_id,
        )
    except Exception as exc:
        logger.debug("maybe_edge_flush_mid_turn: %s", exc)
        return messages, active_system_prompt


def default_scratchpad(primary_goal: str) -> str:
    """Dense template — keyword-first, minimal filler."""
    goal = (primary_goal or "").strip() or "?"
    if len(goal) > 2000:
        goal = goal[:1997] + "..."
    return (
        f"{SCRATCHPAD_HEADER}\n"
        f"- **Goal:** {goal}\n"
        f"- **Phase:** explore\n\n"
        "### KNOW\n"
        "- **Facts:** —\n"
        "- **Faults:** —\n\n"
        "### NEXT\n"
        "- [ ] —\n"
        "- [ ] —\n"
    )


def ensure_scratchpad_initialized(agent, primary_goal: str) -> None:
    """Create scratchpad once per session when edge mode is on."""
    if not getattr(agent, "edge_mode", False):
        return
    existing = (getattr(agent, "_edge_scratchpad", None) or "").strip()
    if existing:
        return
    db = getattr(agent, "_session_db", None)
    sid = getattr(agent, "session_id", None) or ""
    if db and sid:
        try:
            _stored = db.get_edge_working_memory(sid)
        except Exception:
            _stored = None
        if _stored and str(_stored).strip():
            agent._edge_scratchpad = str(_stored)
            try:
                from agent.edge_fault_damper import resync_edge_failed_signatures

                resync_edge_failed_signatures(agent)
            except Exception:
                pass
            return
    agent._edge_scratchpad = default_scratchpad(primary_goal)
    persist_edge_scratchpad_now(agent)


def begin_edge_turn_injection(agent) -> None:
    """Freeze the scratchpad text used for API user-message injection this turn.

    Live ``_edge_scratchpad`` may still mutate mid-turn (assistant absorb, fault
    damper lines, compaction deltas). The frozen copy keeps the injected user
    prefix byte-stable across tool rounds so prompt caching stays valid.
    """
    if not getattr(agent, "edge_mode", False):
        agent._edge_scratchpad_turn_injection = ""
        return
    agent._edge_scratchpad_turn_injection = getattr(agent, "_edge_scratchpad", "") or ""


def edge_scratchpad_for_injection(agent) -> str:
    """Return the turn-frozen scratchpad for API injection (cache-safe)."""
    if not getattr(agent, "edge_mode", False):
        return ""
    frozen = getattr(agent, "_edge_scratchpad_turn_injection", None)
    if isinstance(frozen, str):
        return frozen
    return getattr(agent, "_edge_scratchpad", "") or ""


def _facts_needle_order(scratchpad: str) -> str:
    for n in ("**Facts:**", "**Validated Facts:**"):
        if n in (scratchpad or ""):
            return n
    return "**Facts:**"


def _faults_needle_order(scratchpad: str) -> str:
    for n in ("**Faults:**", "**Faults & Blockers:**"):
        if n in (scratchpad or ""):
            return n
    return "**Faults:**"


def append_auto_fault_blocker(
    scratchpad: str,
    tool_name: str,
    sig: str,
    err_preview: str,
) -> str:
    """Insert an auto fault line under **Faults:** (legacy **Faults & Blockers:**)."""
    base = scratchpad or ""
    preview = (err_preview or "").replace("\n", " ").strip()
    if len(preview) > 240:
        preview = preview[:237] + "..."
    bullet = f"\n- [auto] `{tool_name}` [sig:{sig}] err:{preview or '?'}"
    needle = _faults_needle_order(base)
    pos = base.find(needle)
    if pos != -1:
        insert_at = pos + len(needle)
        return base[:insert_at] + bullet + base[insert_at:]
    if base.strip():
        return base.rstrip() + "\n\n" + needle + bullet
    return default_scratchpad("?") + "\n" + needle + bullet


def format_edge_working_memory_injection(scratchpad: str) -> str:
    """Ephemeral user-message prefix for API calls (not persisted on raw user row)."""
    body = (scratchpad or "").strip()
    if not body:
        body = default_scratchpad("(user msg)")
    return (
        "<!-- edge_wm -->\n"
        "### EDGE_WM\n"
        f"Sync: echo full block from `{SCRATCHPAD_HEADER}` in assistant when edited.\n\n"
        f"{body}\n"
    )


def merge_focus_topic_with_scratchpad(
    focus_topic: Optional[str],
    scratchpad: str,
) -> Optional[str]:
    """Blend manual /compress focus with scratchpad for guided compaction."""
    sp = (scratchpad or "").strip()
    if not sp:
        return focus_topic
    if focus_topic and str(focus_topic).strip():
        return f"{focus_topic.strip()}\n\n---\nEDGE_WM:\n{sp}"
    return f"Prioritize EDGE_WM facts:\n{sp}"


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n".join(parts)
    return ""


def extract_compression_summary_text(messages: list) -> str:
    """Find the newest compaction summary body in compressed message lists."""
    best = ""
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        text = _content_text(msg.get("content"))
        if not text:
            continue
        if SUMMARY_PREFIX in text:
            idx = text.find(SUMMARY_PREFIX) + len(SUMMARY_PREFIX)
            candidate = text[idx:].strip()
        elif text.strip().startswith(LEGACY_SUMMARY_PREFIX):
            candidate = text.strip()[len(LEGACY_SUMMARY_PREFIX) :].strip()
        else:
            continue
        # Drop the standard trailing instruction tail when present
        end = candidate.find("--- END OF CONTEXT SUMMARY")
        if end != -1:
            candidate = candidate[:end].strip()
        if len(candidate) > len(best):
            best = candidate
    return best


def append_compaction_delta_to_scratchpad(scratchpad: str, summary: str) -> str:
    """Append one dense compaction bullet under **Facts:** / legacy Validated Facts."""
    base = scratchpad or ""
    snippet = (summary or "").strip().replace("\n", " ")
    if len(snippet) > 1200:
        snippet = snippet[:1197] + "..."
    if not snippet:
        return base
    bullet = f"\n- [cmpct] {snippet}"
    needle = _facts_needle_order(base)
    pos = base.find(needle)
    if pos != -1:
        insert_at = pos + len(needle)
        return base[:insert_at] + bullet + base[insert_at:]
    if base.strip():
        return base.rstrip() + "\n\n" + needle + bullet
    return default_scratchpad("?") + "\n" + needle + bullet


def absorb_assistant_scratchpad_update(agent, assistant_text: str) -> None:
    """If the model echoed a full scratchpad block, replace runtime state."""
    if not getattr(agent, "edge_mode", False):
        return
    # Normalize newlines so CRLF / lone CR from local SLMs match stop-sequence logic.
    text = (assistant_text or "").replace("\r\n", "\n").replace("\r", "\n")
    best_i = -1
    marker = ""
    for mk in SCRATCHPAD_MARKERS:
        p = text.find(mk)
        if p != -1 and (best_i < 0 or p < best_i):
            best_i, marker = p, mk
    if best_i < 0:
        return
    extracted = text[best_i:].strip()
    for stop in (
        "\n\n---\n",
        "\n---\n",
        "\n## ",
        "\n# ",
    ):
        j = extracted.find(stop, len(marker) + 5)
        if j != -1:
            extracted = extracted[:j].strip()
            break
    if len(extracted) < len(marker) + 10:
        return
    if len(extracted) > 16_000:
        extracted = extracted[:16_000] + "\n\n…(truncated)"
    agent._edge_scratchpad = extracted
    try:
        from agent.edge_fault_damper import resync_edge_failed_signatures

        resync_edge_failed_signatures(agent)
    except Exception:
        pass
    persist_edge_scratchpad_now(agent)


def effective_compression_trigger_tokens(compressor) -> int:
    """First compression trigger threshold (may be lowered in edge mode)."""
    th = int(getattr(compressor, "threshold_tokens", 0) or 0)
    scale = getattr(compressor, "_compression_threshold_scale", 1.0)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 1.0
    if th <= 0:
        return 1
    if scale <= 0 or scale >= 1.0:
        return max(th, 1)
    return max(int(th * scale), 1)
