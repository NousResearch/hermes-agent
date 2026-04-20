"""Adaptive Memory Curator — LLM-driven pruning, deduplication, and ranking.

Runs as a post-session task (triggered by gateway session:end hook or CLI
exit) to consolidate MEMORY.md and USER.md entries. Uses the auxiliary
LLM client (cheapest available provider) to:

1. Identify semantically duplicate entries → merge into one
2. Score each entry's long-term value (0-10)
3. Remove low-value entries when store is near capacity
4. Promote valuable discoveries to learned_knowledge in the knowledge DB

Inspired by agno's LearningMachine curator pattern.
"""

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from hermes_state import SessionDB
    from tools.memory_tool import MemoryStore

logger = logging.getLogger(__name__)

# Only curate when store is above this % of char limit
CURATION_THRESHOLD_PCT = 0.65
# Entries scored below this are candidates for removal
LOW_VALUE_THRESHOLD = 3
# Maximum entries to process per curation run (avoid huge LLM calls)
MAX_ENTRIES_PER_RUN = 30

_CURATION_PROMPT = """\
You are a memory curator for an AI agent. Analyze the following memory entries and return a JSON object.

ENTRIES (each separated by ---):
{entries_block}

TASKS:
1. DUPLICATES: Find entries that are semantically redundant (same fact, different wording). Group them by content similarity. For each group, pick the best-worded one and list the others as duplicates to remove.

2. SCORING: Rate each entry 0-10 for long-term value:
   - 9-10: Critical user preferences, environment facts that prevent errors, lessons from failures
   - 6-8: Useful conventions, API quirks, workflow patterns
   - 3-5: Somewhat useful but easily re-discoverable
   - 0-2: Outdated, trivial, or task-specific (should have been session state)

3. PROMOTIONS: Entries scoring 8+ that describe a reusable insight or pattern should be flagged for promotion to the knowledge base.

Return ONLY valid JSON (no markdown fencing):
{{
  "duplicates_to_remove": ["exact text of entry to remove", ...],
  "low_value_entries": ["exact text of entry scoring 0-{low_threshold}", ...],
  "promotions": [
    {{"entry": "exact text", "title": "short title for knowledge note", "reason": "why this is valuable"}}
  ],
  "summary": "one-line description of what you did"
}}
"""


def _build_entries_block(entries: List[str]) -> str:
    """Format entries for the LLM prompt."""
    return "\n---\n".join(entries[:MAX_ENTRIES_PER_RUN])


def curate_memory(
    memory_store: "MemoryStore",
    auxiliary_client: Any = None,
    auxiliary_model: Optional[str] = None,
    session_db: Optional["SessionDB"] = None,
    session_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a curation pass on MEMORY.md and USER.md.

    Args:
        memory_store: MemoryStore instance (must be loaded from disk).
        auxiliary_client: OpenAI-compatible client for LLM calls.
        auxiliary_model: Model slug for the auxiliary client.
        session_db: SessionDB instance for promoting to knowledge.
        session_id: Current session ID (for knowledge attribution).
        dry_run: If True, return analysis without modifying anything.

    Returns:
        Dict with curation results and actions taken.
    """
    if auxiliary_client is None or auxiliary_model is None:
        return {"skipped": True, "reason": "No auxiliary LLM available for curation."}

    results = {"targets_curated": [], "total_removed": 0, "total_promoted": 0}

    for target in ("memory", "user"):
        entries = list(memory_store._entries_for(target))
        char_limit = memory_store._char_limit(target)
        current_chars = memory_store._char_count(target)

        # Skip if below threshold
        if char_limit > 0 and (current_chars / char_limit) < CURATION_THRESHOLD_PCT:
            logger.debug("Skipping %s curation: %d/%d chars (below %.0f%% threshold)",
                         target, current_chars, char_limit, CURATION_THRESHOLD_PCT * 100)
            continue

        if len(entries) < 3:
            continue

        # Build and send the curation prompt
        entries_block = _build_entries_block(entries)
        prompt = _CURATION_PROMPT.format(
            entries_block=entries_block,
            low_threshold=LOW_VALUE_THRESHOLD,
        )

        try:
            response = auxiliary_client.chat.completions.create(
                model=auxiliary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            raw_content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                raw_content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            analysis = json.loads(raw_content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Memory curation LLM call failed for %s: %s", target, e)
            continue

        duplicates_to_remove = analysis.get("duplicates_to_remove", [])
        low_value_entries = analysis.get("low_value_entries", [])
        promotions = analysis.get("promotions", [])
        summary = analysis.get("summary", "")

        target_result = {
            "target": target,
            "entries_before": len(entries),
            "duplicates_found": len(duplicates_to_remove),
            "low_value_found": len(low_value_entries),
            "promotions": len(promotions),
            "summary": summary,
        }

        if dry_run:
            target_result["dry_run"] = True
            target_result["would_remove"] = duplicates_to_remove + low_value_entries
            target_result["would_promote"] = promotions
            results["targets_curated"].append(target_result)
            continue

        # Remove duplicates and low-value entries
        to_remove = set(duplicates_to_remove + low_value_entries)
        removed_count = 0
        for entry_text in to_remove:
            # Use substring matching (same as memory tool's remove action)
            result = memory_store.remove(target, entry_text)
            if result.get("success"):
                removed_count += 1

        target_result["actually_removed"] = removed_count
        results["total_removed"] += removed_count

        # Promote high-value entries to knowledge DB + wiki
        if session_db and promotions:
            for promo in promotions:
                try:
                    _promote_to_knowledge(
                        session_db=session_db,
                        title=promo.get("title", "Curated insight"),
                        content=promo.get("entry", ""),
                        reason=promo.get("reason", ""),
                        session_id=session_id,
                    )
                    results["total_promoted"] += 1
                except Exception as e:
                    logger.debug("Failed to promote entry to knowledge: %s", e)
                # Also file into wiki KB (best-effort)
                try:
                    _promote_to_wiki(
                        title=promo.get("title", "Curated insight"),
                        content=promo.get("entry", ""),
                        reason=promo.get("reason", ""),
                    )
                except Exception as e:
                    logger.debug("Failed to promote entry to wiki: %s", e)

        target_result["entries_after"] = len(memory_store._entries_for(target))
        results["targets_curated"].append(target_result)

    return results


def _promote_to_knowledge(
    session_db: "SessionDB",
    title: str,
    content: str,
    reason: str,
    session_id: Optional[str] = None,
) -> None:
    """Add a curated memory entry to the knowledge_notes table."""
    now = time.time()
    full_content = f"{title}\n\n{content}\n\n[Curator reason: {reason}]"

    def _do(conn):
        conn.execute(
            """INSERT INTO knowledge_notes (content, source, session_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (full_content, "memory_curator", session_id, now, now),
        )
    session_db._execute_write(_do)


def _promote_to_wiki(title: str, content: str, reason: str) -> None:
    """File a curated memory entry into the wiki KB as a concept note.

    Best-effort: silently skips if the wiki directory doesn't exist or
    the kb_tool isn't importable (e.g. wiki not initialized yet).
    Checks for existing related pages to avoid wiki sprawl.
    """
    try:
        from tools.kb_tool import kb_tool, check_kb_requirements
    except ImportError:
        return
    if not check_kb_requirements():
        return

    # Dedup check: skip if a related page already exists
    import json as _json
    search_result = kb_tool(action="search", query=title, max_results=3)
    try:
        parsed = _json.loads(search_result)
        if parsed.get("matches", 0) > 0:
            logger.debug("Wiki already has related pages for '%s', skipping promotion", title)
            return
    except (ValueError, TypeError):
        pass

    full_content = f"{content}\n\n*Promoted from memory curator: {reason}*"
    kb_tool(
        action="file",
        title=title,
        content=full_content,
        page_type="concept",
        tags="curated, memory-promotion",
    )
    logger.debug("Promoted to wiki: %s", title)


def run_post_session_curation(
    memory_store: "MemoryStore",
    session_db: Optional["SessionDB"] = None,
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convenience wrapper: resolve auxiliary client and run curation.

    Called by gateway session:end hook or CLI exit handler.
    Fails silently if no auxiliary client is available.
    """
    try:
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("auto")
        if client is None:
            logger.debug("No auxiliary client available for memory curation.")
            return None

        result = curate_memory(
            memory_store=memory_store,
            auxiliary_client=client,
            auxiliary_model=model,
            session_db=session_db,
            session_id=session_id,
        )

        if result.get("total_removed", 0) > 0 or result.get("total_promoted", 0) > 0:
            logger.info(
                "Memory curation: removed %d entries, promoted %d to knowledge",
                result["total_removed"],
                result["total_promoted"],
            )
        return result
    except Exception as e:
        logger.debug("Memory curation failed (non-fatal): %s", e)
        return None
