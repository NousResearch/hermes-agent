"""
Memory Consolidation — periodic background review and maintenance.

Cannibalized from:
- Claude Code (services/autoDream/): 5-gate scheduling, 4-phase consolidation prompt
- HiveMind (memory.rs): tier promotion, archival, strength-based lifecycle

Designed to run as a cron job via Hermes' existing cron infrastructure,
or called directly from the agent after long sessions.

Gate system (cheapest checks first, from Claude Code):
1. Feature enabled? (config check)
2. Time: hours since last consolidation >= threshold
3. Sessions: enough sessions since last run
4. Lock: prevent concurrent consolidation
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consolidation prompt — adapted from Claude Code autoDream + HiveMind
# ---------------------------------------------------------------------------

_CONSOLIDATION_PROMPT = """\
You are a memory maintenance system for an AI agent. Review the current memories \
and perform consolidation to keep the memory store healthy.

CURRENT ACTIVE MEMORIES:
{memories}

MEMORY STATISTICS:
{stats}

BUDGET: memory target max {max_memory}, user target max {max_user}. \
If either target is over budget, you MUST archive low-value entries to get under budget.

TASKS — perform ALL that apply, in priority order:

1. MERGE: If two or more memories cover the same topic or entity, merge them into one \
concise entry that preserves all unique facts. \
Output: {{"action": "merge", "remove_ids": ["id1", "id2"], "new_content": "merged text", "target": "memory"|"user", "type": "preference"|"correction"|"project"|"reference"|"general"}}

2. UPDATE: If any memory has stale relative dates ("yesterday", "last week"), \
outdated facts, or information contradicted by newer memories, update it. \
Output: {{"action": "update", "id": "id", "new_content": "updated text"}}

3. ARCHIVE: If any memory is low-value, too specific to a completed past task, \
no longer relevant, or redundant with another memory, mark for archival. \
Prefer archiving: low access_count, low strength, old age, general type over correction/preference. \
Output: {{"action": "archive", "id": "id", "reason": "why"}}

4. NOTHING: If all memories are clean, current, and within budget, output: NONE

PROTECT corrections and preferences — these are the most valuable. Archive general \
and project memories first when reducing count.

Output ONE JSON object per line. No other text.
IMPORTANT: Use the 8-char IDs shown in brackets, not full UUIDs."""


def check_consolidation_gates(engine, config: dict) -> Optional[str]:
    """Check whether consolidation should run. Returns reason to skip, or None to proceed.

    Gate order follows Claude Code's autoDream pattern (cheapest first).
    """
    # Gate 1: Feature enabled?
    if not config.get("consolidation_enabled", True):
        return "consolidation_enabled is false"

    # Gate 2: Time since last consolidation
    last_str = engine._get_meta("last_consolidation") or ""
    if last_str:
        try:
            last = datetime.fromisoformat(last_str)
            hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            threshold = config.get("consolidation_interval_hours", 12)
            if hours < threshold:
                return f"only {hours:.1f}h since last consolidation (threshold: {threshold}h)"
        except (ValueError, TypeError):
            pass

    # Gate 3: Enough sessions since last consolidation
    count_str = engine._get_meta("consolidation_session_count") or "0"
    min_sessions = config.get("consolidation_min_sessions", 3)
    try:
        if int(count_str) < min_sessions:
            return f"only {count_str} sessions since last consolidation (threshold: {min_sessions})"
    except (ValueError, TypeError):
        pass

    return None  # All gates passed


def consolidate_memories(
    engine,
    auxiliary_client=None,
    model: str = None,
    config: dict = None,
) -> dict:
    """Run memory consolidation. Returns summary of actions taken.

    Can be called directly or via cron job.
    """
    config = config or {}
    start = time.monotonic()

    # Check gates
    skip_reason = check_consolidation_gates(engine, config)
    if skip_reason:
        return {"consolidated": False, "reason": skip_reason}

    if auxiliary_client is None:
        return {"consolidated": False, "reason": "no auxiliary_client available"}

    # Step 1: Run lifecycle maintenance (archival, promotion)
    archived = engine.archive_stale()

    # Step 2: Build memory inventory for LLM
    all_memories = []
    for target in ("memory", "user"):
        for mem in engine.get_active_memories(target):
            all_memories.append(mem)

    if not all_memories:
        _update_consolidation_meta(engine)
        return {"consolidated": True, "actions": 0, "archived": archived, "reason": "no active memories"}

    # Format for prompt
    mem_lines = []
    for m in all_memories:
        age_str = ""
        try:
            created = datetime.fromisoformat(m["created_at"])
            days = (datetime.now(timezone.utc) - created).days
            age_str = f" ({days}d old)"
        except (ValueError, TypeError, KeyError):
            pass

        mem_lines.append(
            f"[{m['id'][:8]}|{m.get('type','gen')}|{m['target']}|strength={m.get('strength',1.0):.1f}|"
            f"accessed={m.get('access_count',0)}x{age_str}] {m['content']}"
        )

    stats = engine.stats()
    from .memory_engine import MAX_ACTIVE_MEMORY, MAX_ACTIVE_USER
    prompt = _CONSOLIDATION_PROMPT.format(
        memories="\n".join(mem_lines),
        stats=json.dumps(stats, indent=2),
        max_memory=MAX_ACTIVE_MEMORY,
        max_user=MAX_ACTIVE_USER,
    )

    # Step 3: Call LLM for consolidation decisions
    try:
        response = auxiliary_client.call_llm(
            prompt=prompt,
            system_message="You are a memory maintenance system. Output JSON lines only.",
            model=model,
            max_tokens=2048,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("Consolidation LLM call failed: %s", e)
        return {"consolidated": False, "reason": f"LLM call failed: {e}", "archived": archived}

    if not response or response.strip().upper() == "NONE":
        _update_consolidation_meta(engine)
        return {"consolidated": True, "actions": 0, "archived": archived}

    # Step 4: Execute consolidation actions
    actions_taken = 0
    id_map = {m["id"][:8]: m["id"] for m in all_memories}

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE" or line.startswith("```"):
            continue

        try:
            action = json.loads(line)
        except json.JSONDecodeError:
            continue

        act_type = action.get("action")

        if act_type == "merge":
            remove_ids = action.get("remove_ids", [])
            new_content = action.get("new_content", "").strip()
            target = action.get("target", "memory")
            mem_type = action.get("type", "general")

            if new_content and remove_ids:
                # Add merged memory
                result = engine.add(
                    content=new_content,
                    target=target,
                    type=mem_type,
                    source="consolidation",
                )
                if result.get("success"):
                    new_id = result["id"]
                    # Supersede old memories
                    for short_id in remove_ids:
                        full_id = id_map.get(short_id)
                        if full_id:
                            engine.supersede(full_id, new_id)
                    actions_taken += 1
                    logger.info("Consolidated %d memories into one: %s", len(remove_ids), new_content[:60])

        elif act_type == "update":
            mem_id = id_map.get(action.get("id", ""))
            new_content = action.get("new_content", "").strip()
            if mem_id and new_content:
                result = engine.replace(mem_id, new_content)
                if result.get("success"):
                    actions_taken += 1
                    logger.info("Updated memory %s: %s", action.get("id", "")[:8], new_content[:60])

        elif act_type == "archive":
            mem_id = id_map.get(action.get("id", ""))
            if mem_id:
                engine._get_conn().execute(
                    "UPDATE memories SET tier = 'archived', updated_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), mem_id),
                )
                engine._get_conn().commit()
                actions_taken += 1
                logger.info("Archived memory %s: %s", action.get("id", "")[:8], action.get("reason", ""))

    _update_consolidation_meta(engine)

    elapsed = time.monotonic() - start
    logger.info("Consolidation complete: %d actions, %d archived, %.1fs", actions_taken, archived, elapsed)

    return {
        "consolidated": True,
        "actions": actions_taken,
        "archived": archived,
        "elapsed_seconds": round(elapsed, 1),
    }


def increment_session_count(engine):
    """Increment the session counter for consolidation gating.

    Call this at session end to track how many sessions have passed
    since the last consolidation.
    """
    try:
        current = int(engine._get_meta("consolidation_session_count") or "0")
        engine._set_meta("consolidation_session_count", str(current + 1))
    except (ValueError, TypeError):
        engine._set_meta("consolidation_session_count", "1")


def _update_consolidation_meta(engine):
    """Update consolidation metadata after a successful run."""
    engine._set_meta("last_consolidation", datetime.now(timezone.utc).isoformat())
    engine._set_meta("consolidation_session_count", "0")
