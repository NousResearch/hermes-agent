"""Compress pipeline — LCM-style hierarchical episode compression.

Implements 3-level escalation:
  D0: Detailed narrative summary (per-session)
  D1: Bullet-point key facts (daily)
  D2: One-line topic digest (weekly)

Each level creates DAG nodes that link back to source material,
enabling drill-down retrieval (LCM pattern: summaries always link
back to originals).

Called:
  - D0: At session end (after merge)
  - D1: Daily via cron (compresses D0 nodes)
  - D2: Weekly via cron (compresses D1 nodes)
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from memory.config import COMPRESS_TIMEOUT, get_memory_model_settings
from memory.episodic_store import EpisodicStore

logger = logging.getLogger(__name__)

# ── D0: Session → Narrative Summary ─────────────────────────────────────

D0_SYSTEM_PROMPT = """You are an episode summarizer. Given raw conversation turns from a single session, produce a structured episode summary.

Output JSON with these fields:
- topic: One-line topic label (e.g. "Setting up Telegram bot integration")
- summary: 2-4 sentence narrative capturing what happened
- key_decisions: Array of decisions made (empty array if none)
- unresolved: Array of open questions or pending items (empty array if none)
- participants: Array of entity names involved

Rules:
- Be concise but preserve specific details (names, numbers, dates, file paths)
- Decisions are things the user explicitly chose or agreed to
- Unresolved items are things left pending or mentioned for later
- Output ONLY valid JSON, no markdown fences"""

D0_USER_TEMPLATE = """SESSION TURNS:
{turns_text}

Produce a structured episode summary for this session."""


def _format_turns_for_compress(turns: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Format turns for compression, with length limits."""
    lines = []
    total = 0
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "")
        tool_name = t.get("tool_name", "")
        if len(content) > 500:
            content = content[:500] + "..."
        if role == "tool" and tool_name:
            line = f"[Tool: {tool_name}] {content}"
        else:
            line = f"[{role}] {content}"
        if total + len(line) > max_chars:
            lines.append("...[truncated]")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def compress_session_to_d0(
    store: EpisodicStore,
    session_id: str,
    turns: List[Dict[str, Any]],
) -> Optional[str]:
    """Compress a session's turns into a D0 narrative summary.

    Creates a DAG node at depth 0 and updates the session's episodes.

    Args:
        store: EpisodicStore instance.
        session_id: Session to compress.
        turns: Raw turn list for this session.

    Returns:
        DAG node ID if successful, None on failure.
    """
    if not turns:
        return None

    turns_text = _format_turns_for_compress(turns)
    user_msg = D0_USER_TEMPLATE.format(turns_text=turns_text)

    provider, model = get_memory_model_settings("compress")

    try:
        response = call_llm(
            provider=provider,
            model=model,
            messages=[
                {"role": "system", "content": D0_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=COMPRESS_TIMEOUT,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            content = "\n".join(lines)

        summary = json.loads(content)

        # Create episode
        turn_ids = [t.get("id") for t in turns if t.get("id")]
        episode_id = store.create_episode(
            session_id=session_id,
            topic=summary.get("topic", "untitled"),
            summary=summary.get("summary", ""),
            key_decisions=json.dumps(summary.get("key_decisions", [])),
            unresolved=json.dumps(summary.get("unresolved", [])),
            participants=json.dumps(summary.get("participants", [])),
            source_turns=turn_ids if turn_ids else None,
            episode_type="substantive",
        )

        # Create DAG node
        node_id = f"d0-{session_id}-{episode_id}"
        store.create_dag_node(
            node_id=node_id,
            parent_ids=[],  # D0 nodes have no parents
            depth=0,
            content=json.dumps(summary, ensure_ascii=False),
            source_range={"session_id": session_id, "turn_ids": turn_ids[:50]},
        )

        logger.info(
            "D0 compress complete: session=%s episode=%d node=%s",
            session_id, episode_id, node_id,
        )
        return node_id

    except json.JSONDecodeError as e:
        logger.error("D0 compress returned invalid JSON: %s", e)
        return None
    except Exception as e:
        logger.error("D0 compress failed for session %s: %s", session_id, e)
        return None


# ── D1: Daily → Bullet-Point Key Facts ─────────────────────────────────

D1_SYSTEM_PROMPT = """You are a knowledge compressor. Given multiple D0 episode summaries from one day, produce a consolidated D1 summary.

Output JSON with:
- date: The date string
- key_facts: Array of the most important facts from all episodes (max 10)
- decisions: Array of decisions made across sessions
- active_entities: Array of entity names that were active
- summary: 1-2 paragraph narrative of the day's activity

Be ruthless about cutting noise. Only keep facts that will matter in a week."""

D1_USER_TEMPLATE = """D0 EPISODE SUMMARIES (date: {date}):
{summaries_text}

Produce a consolidated daily summary."""


def compress_d0_to_d1(
    store: EpisodicStore,
    date_str: str,
    d0_node_ids: List[str],
) -> Optional[str]:
    """Compress a day's D0 nodes into a single D1 summary.

    Args:
        store: EpisodicStore instance.
        date_str: Date label (e.g. "2026-04-21").
        d0_node_ids: List of D0 node IDs from this day.

    Returns:
        D1 DAG node ID if successful, None on failure.
    """
    if not d0_node_ids:
        return None

    # Gather D0 content
    summaries = []
    for nid in d0_node_ids:
        node = store.get_dag_node(nid)
        if node:
            summaries.append(node["content"])

    if not summaries:
        return None

    summaries_text = "\n---\n".join(summaries)
    user_msg = D1_USER_TEMPLATE.format(date=date_str, summaries_text=summaries_text)

    provider, model = get_memory_model_settings("compress")

    try:
        response = call_llm(
            provider=provider,
            model=model,
            messages=[
                {"role": "system", "content": D1_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1500,
            timeout=COMPRESS_TIMEOUT,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            content = "\n".join(lines)

        summary = json.loads(content)

        node_id = f"d1-{date_str}"
        store.create_dag_node(
            node_id=node_id,
            parent_ids=d0_node_ids,
            depth=1,
            content=json.dumps(summary, ensure_ascii=False),
            source_range={"date": date_str, "d0_nodes": d0_node_ids},
        )

        logger.info(
            "D1 compress complete: date=%s node=%s parents=%d",
            date_str, node_id, len(d0_node_ids),
        )
        return node_id

    except Exception as e:
        logger.error("D1 compress failed for %s: %s", date_str, e)
        return None


# ── D2: Weekly → One-Line Topic Digest ─────────────────────────────────

D2_SYSTEM_PROMPT = """You are a weekly summarizer. Given D1 daily summaries from one week, produce a concise D2 digest.

Output JSON with:
- week: Week label (e.g. "2026-W17")
- topics: Array of {topic, one_line_summary, entities} — max 5 most important
- summary: One paragraph overview of the week

Only include topics that represent meaningful progress or decisions."""

D2_USER_TEMPLATE = """D1 DAILY SUMMARIES (week of {week}):
{summaries_text}

Produce a weekly digest."""


def compress_d1_to_d2(
    store: EpisodicStore,
    week_label: str,
    d1_node_ids: List[str],
) -> Optional[str]:
    """Compress a week's D1 nodes into a D2 digest.

    Args:
        store: EpisodicStore instance.
        week_label: Week label (e.g. "2026-W17").
        d1_node_ids: List of D1 node IDs from this week.

    Returns:
        D2 DAG node ID if successful, None on failure.
    """
    if not d1_node_ids:
        return None

    summaries = []
    for nid in d1_node_ids:
        node = store.get_dag_node(nid)
        if node:
            summaries.append(node["content"])

    if not summaries:
        return None

    summaries_text = "\n---\n".join(summaries)
    user_msg = D2_USER_TEMPLATE.format(week=week_label, summaries_text=summaries_text)

    provider, model = get_memory_model_settings("compress")

    try:
        response = call_llm(
            provider=provider,
            model=model,
            messages=[
                {"role": "system", "content": D2_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=800,
            timeout=COMPRESS_TIMEOUT,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            content = "\n".join(lines)

        summary = json.loads(content)

        node_id = f"d2-{week_label}"
        store.create_dag_node(
            node_id=node_id,
            parent_ids=d1_node_ids,
            depth=2,
            content=json.dumps(summary, ensure_ascii=False),
            source_range={"week": week_label, "d1_nodes": d1_node_ids},
        )

        logger.info(
            "D2 compress complete: week=%s node=%s parents=%d",
            week_label, node_id, len(d1_node_ids),
        )
        return node_id

    except Exception as e:
        logger.error("D2 compress failed for %s: %s", week_label, e)
        return None


def get_d0_nodes_for_date(store: EpisodicStore, date_str: str) -> List[str]:
    """Get all D0 DAG node IDs created on a specific date."""
    all_d0 = store.get_dag_nodes_at_depth(0)
    matching = []
    for node in all_d0:
        node_id = node["id"]
        # D0 node IDs are: d0-{session_id}-{episode_id}
        # We check via created_at timestamp
        created = node.get("created_at", 0)
        from datetime import datetime, timezone
        node_date = datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d")
        if node_date == date_str:
            matching.append(node_id)
    return matching


def get_d1_nodes_for_week(store: EpisodicStore, week_label: str) -> List[str]:
    """Get all D1 DAG node IDs for a specific week."""
    all_d1 = store.get_dag_nodes_at_depth(1)
    matching = []
    for node in all_d1:
        node_id = node["id"]
        # D1 node IDs are: d1-{date_str}
        # Extract date from ID and check week
        if node_id.startswith("d1-"):
            date_part = node_id[3:]  # "2026-04-21"
            try:
                from datetime import datetime, timezone
                dt = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                iso_week = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
                if iso_week == week_label:
                    matching.append(node_id)
            except ValueError:
                pass
    return matching
