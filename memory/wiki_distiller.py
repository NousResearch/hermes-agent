"""Wiki distillation — produce high-quality knowledge pages from episodic memory.

Runs as a daily cron job (2AM) to synthesize episodes and entities into
readable wiki pages at ~/wiki/session-memory/.

Unlike raw episode summaries, wiki pages are editorial-quality:
  - Resolve contradictions
  - Cross-reference related entities
  - Note staleness
  - Maintain temporal ordering

Includes fallback framework:
  - Exponential backoff retry on 429/529/503
  - Provider fallback chain from config fallback_providers
  - Inter-request delay to avoid rate spikes
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from memory.config import (
    WIKI_OUTPUT_DIR,
    WIKI_TIMEOUT,
    get_memory_model_settings,
)
from memory.episodic_store import EpisodicStore

logger = logging.getLogger(__name__)

# ── Fallback / retry defaults ──────────────────────────────────────────────
_RETRY_MAX = 3
_RETRY_BASE_DELAY = 2.0
_INTER_REQUEST_DELAY = 0.5


def _is_retryable_error(err: Exception) -> bool:
    text = str(err).lower()
    return any(kw in text for kw in ("429", "rate limit", "529", "overloaded", "503", "service unavailable"))


def _get_fallback_providers() -> List[Dict[str, str]]:
    """Read fallback provider chain from config.yaml."""
    try:
        from memory.config import load_config
        cfg = load_config() or {}
    except Exception:
        return []
    chain = cfg.get("fallback_providers") or cfg.get("fallback_model") or []
    if isinstance(chain, dict):
        chain = [chain]
    if not isinstance(chain, list):
        return []
    result = []
    for entry in chain:
        if isinstance(entry, dict) and entry.get("provider"):
            result.append({
                "provider": str(entry["provider"]),
                "model": str(entry.get("model") or ""),
            })
    return result


def _call_llm_with_fallback(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float = WIKI_TIMEOUT,
) -> Optional[str]:
    """Call the LLM with retry + fallback provider chain.

    Returns the model's text content, or None if all attempts fail.
    """
    providers_chain = [{"provider": provider, "model": model}]
    providers_chain.extend(_get_fallback_providers())

    last_err: Optional[Exception] = None
    for pidx, prov in enumerate(providers_chain):
        attempt_provider = prov["provider"]
        attempt_model = prov["model"] or model

        for attempt in range(_RETRY_MAX):
            try:
                if attempt > 0 or pidx > 0:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1)) + _INTER_REQUEST_DELAY
                    time.sleep(delay)

                response = call_llm(
                    provider=attempt_provider,
                    model=attempt_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    return content
                last_err = RuntimeError(f"Empty response from {attempt_provider}/{attempt_model}")
            except Exception as e:
                last_err = e
                if not _is_retryable_error(e):
                    # Non-retryable (auth, bad request, etc.) — skip retries on this provider
                    logger.warning(
                        "Non-retryable error from %s/%s: %s — moving to next provider",
                        attempt_provider, attempt_model, e,
                    )
                    break
                logger.debug(
                    "Retryable error from %s/%s (attempt %d/%d): %s",
                    attempt_provider, attempt_model, attempt + 1, _RETRY_MAX, e,
                )

        # Small delay between providers
        time.sleep(_INTER_REQUEST_DELAY)

    if last_err:
        logger.error("All provider attempts failed: %s", last_err)
    return None


def _get_entity_episodes(
    store: EpisodicStore,
    entity_name: str,
    limit: int = 20,
    profile: Optional[Dict[str, Any]] = None,
    entity_type: str = "",
) -> List[dict]:
    """Get episodes related to an entity via FTS5 search."""
    queries = []
    if entity_name:
        queries.append(entity_name)
    profile = profile or {}
    canonical_name = _canonical_entity_name(entity_type or "unknown", entity_name, profile)
    if canonical_name and canonical_name not in queries:
        queries.append(canonical_name)
    for key in ("title", "full_name", "preferred_name"):
        value = profile.get(key)
        if value and value not in queries:
            queries.append(str(value))

    results = []
    seen_ids = set()
    for query in queries:
        try:
            matches = store.search_episodes(query, limit=limit)
        except Exception:
            continue
        for match in matches:
            if match.get("id") in seen_ids:
                continue
            seen_ids.add(match.get("id"))
            results.append(match)
            if len(results) >= limit:
                return results
    return results



WIKI_ENTITY_PROMPT = """You are a knowledge editor. Given accumulated conversation episodes and an entity profile, produce a high-quality wiki-style knowledge document.

ENTITY PROFILE:
{entity_json}

RELATED EPISODES:
{episode_summaries}

Produce a wiki entry with:
1. Overview: 2-3 sentence current-state description
2. History: Timeline of key changes and events (from episodes)
3. Current Attributes: Known attributes with values
4. Relationships: How this entity relates to other known entities
5. Open Items: Pending decisions, unresolved questions
6. Last Updated: {date}

Quality standards:
- Resolve contradictions by preferring more recent data
- Note when information may be stale
- Write in encyclopedic, neutral tone
- Be specific — use names, dates, file paths
- Output clean markdown (no frontmatter needed)"""

WIKI_SESSION_PROMPT = """You are a session summarizer. Given episode summaries from a session, produce a clean wiki page.

SESSION: {session_id}
DATE: {date}

EPISODES:
{episode_summaries}

Produce a session wiki page with:
1. Title: Descriptive title for this session
2. Summary: What was accomplished
3. Key Decisions: Bullet list of decisions made
4. Work Done: Specific tasks completed
5. Open Items: Anything left pending
6. Related Entities: People, projects, tools involved

Write in clear, concise markdown."""


def _format_episodes_for_wiki(episodes: List[dict]) -> str:
    parts = []
    for ep in episodes:
        parts.append(
            f"### {ep.get('topic', 'untitled')} ({ep.get('session_id', '?')[:8]}...)\n"
            f"{ep.get('summary', '')}\n"
        )
    return "\n".join(parts)


def _jsonish_has_items(value: Any) -> bool:
    if not value:
        return False
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            parsed = json.loads(text)
        except Exception:
            return bool(text)
        return bool(parsed)
    return True


def _episode_signal_score(ep: Dict[str, Any]) -> int:
    summary = str(ep.get("summary", "") or "")
    topic = str(ep.get("topic", "") or "")
    score = 0
    if len(summary) >= 120:
        score += 3
    elif len(summary) >= 60:
        score += 2
    elif len(summary) >= 25:
        score += 1
    if _jsonish_has_items(ep.get("key_decisions")):
        score += 3
    if _jsonish_has_items(ep.get("unresolved")):
        score += 2
    if _jsonish_has_items(ep.get("source_turns_json")):
        score += 2
    if len(topic) >= 40:
        score += 1
    return score


def filter_session_episodes(episodes: List[dict]) -> List[dict]:
    """Prefer high-signal episodes when a session also contains micro-chatter shards."""
    if len(episodes) <= 1:
        return episodes

    rich = [ep for ep in episodes if _episode_signal_score(ep) >= 4]
    if rich:
        return rich

    best_score = max(_episode_signal_score(ep) for ep in episodes)
    return [ep for ep in episodes if _episode_signal_score(ep) == best_score]


def mark_banter_episodes(store: EpisodicStore) -> int:
    """Mark raw episodes that look like banter/chitchat.

    Called at distill time. Updates episode_type in-place for low-signal
    episodes so they get properly tagged even if the classification heuristic
    at creation time missed them.
    """
    from memory.merge import classify_episode_type

    episodes = store.get_recent_episodes(limit=200)
    count = 0
    for ep in episodes:
        if ep.get("episode_type") != "raw":
            continue
        classified = classify_episode_type(
            ep.get("topic", ""),
            ep.get("summary", ""),
            key_decisions=ep.get("key_decisions"),
            source_turns_json=ep.get("source_turns_json"),
        )
        if classified != "raw":
            store.update_episode_type(ep["id"], classified)
            count += 1
    return count


def _parse_profile(profile: Any) -> Dict[str, Any]:
    if isinstance(profile, dict):
        return dict(profile)
    if isinstance(profile, str):
        try:
            parsed = json.loads(profile)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}



def _strip_internal_metadata(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in profile.items()
        if not str(key).startswith("_")
    }



def _looks_like_url(value: str) -> bool:
    text = (value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")



def _slugify(text: str) -> str:
    slug = "".join(
        c if c.isalnum() or c in {"-", "_"} else "-"
        for c in (text or "").lower().strip().replace(" ", "-")
    )
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "unknown"



def _canonical_entity_type(entity_type: str, entity_name: str, profile: Dict[str, Any]) -> str:
    role = str(profile.get("role", "") or "").lower()
    if _looks_like_url(entity_name) or "url" in role:
        return "resource"
    if entity_type == "tool" and entity_name.lower().endswith(".md"):
        return "artifact"
    return entity_type or "unknown"



def _canonical_entity_name(entity_type: str, entity_name: str, profile: Dict[str, Any]) -> str:
    canonical_type = _canonical_entity_type(entity_type, entity_name, profile)
    if canonical_type == "person":
        if profile.get("full_name"):
            return str(profile["full_name"])
        if "@" in entity_name and entity_name.strip():
            local = entity_name.split("@", 1)[0]
            local = re.sub(r"[._-]+", " ", local).strip()
            if local:
                return local.title()
    if canonical_type == "resource":
        for key in ("title", "document_title", "name"):
            if profile.get(key):
                return str(profile[key])
    return entity_name or "unknown"



def _entity_stub_signal(profile: Dict[str, Any]) -> int:
    score = 0
    aliases = profile.get("aliases")
    if isinstance(aliases, list) and aliases:
        score += 1
    for key in ("full_name", "preferred_name", "title", "document_id", "platform", "access"):
        if profile.get(key):
            score += 1
    values_text = " ".join(str(v) for v in profile.values() if not isinstance(v, (dict, list)))
    if len(values_text) >= 30:
        score += 1
    return score



def _should_skip_entity(entity: Dict[str, Any], episodes: List[dict], profile: Dict[str, Any]) -> bool:
    clean_profile = _strip_internal_metadata(profile)
    if not clean_profile:
        return True
    signal = _entity_stub_signal(clean_profile)
    if signal == 0 and len(episodes) < 2:
        return True
    if not episodes and signal == 0:
        return True
    nonempty_keys = [k for k, v in clean_profile.items() if v not in (None, "", [], {})]
    if len(nonempty_keys) == 1 and nonempty_keys[0] in {"role", "status", "confidence"}:
        return True
    return False



def _build_stub_entity_page(entity_type: str, entity_name: str, profile: Dict[str, Any], date: str) -> str:
    lines = [f"# {entity_name}", "", "## Overview", ""]
    canonical_type = entity_type.capitalize()
    attrs = []
    for key, value in profile.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        attrs.append((str(key), str(value)))

    if attrs:
        summary_bits = []
        for key, value in attrs[:3]:
            summary_bits.append(f"{key.replace('_', ' ')}: {value}")
        lines.append(
            f"This {canonical_type.lower()} entity has limited episode context, but the extracted profile provides a useful durable stub ({'; '.join(summary_bits)})."
        )
    else:
        lines.append(f"This {canonical_type.lower()} entity currently has only minimal extracted profile data.")

    lines.extend(["", "## Current Attributes", ""])
    for key, value in attrs:
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    lines.extend(["", "## Last Updated", "", date])
    return "\n".join(lines)



def _safe_entity_filename(entity_type: str, entity_name: str, profile: Optional[Dict[str, Any]] = None) -> str:
    profile = profile or {}
    canonical_type = _canonical_entity_type(entity_type, entity_name, profile)
    canonical_name = _canonical_entity_name(entity_type, entity_name, profile)
    slug = _slugify(canonical_name)
    return f"{canonical_type}-{slug}.md"


def distill_entity_wiki(
    store: EpisodicStore,
    entity: Dict[str, Any],
) -> Optional[str]:
    """Produce a wiki page for a single entity.

    Args:
        store: EpisodicStore instance.
        entity: Entity dict with id, type, name, profile_json.

    Returns:
        Markdown string if successful, None on failure.
    """
    entity_name = entity.get("name", "")
    if not entity_name:
        return None

    raw_type = str(entity.get("type", "unknown"))
    profile = _parse_profile(entity.get("profile_json", {}))
    clean_profile = _strip_internal_metadata(profile)
    episodes = _get_entity_episodes(store, entity_name, profile=clean_profile, entity_type=raw_type)

    if _should_skip_entity(entity, episodes, profile):
        return None

    canonical_type = _canonical_entity_type(raw_type, entity_name, clean_profile)
    canonical_name = _canonical_entity_name(raw_type, entity_name, clean_profile)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not episodes:
        return _build_stub_entity_page(canonical_type, canonical_name, clean_profile, today)

    episode_text = _format_episodes_for_wiki(episodes)
    entity_json = json.dumps(
        {
            "id": entity.get("id"),
            "name": canonical_name,
            "canonical_name": canonical_name,
            "type": canonical_type,
            "aliases": [] if canonical_type == "resource" else ([entity_name] if canonical_name != entity_name else []),
            "profile": clean_profile,
        },
        indent=2,
        ensure_ascii=False,
    )

    user_msg = WIKI_ENTITY_PROMPT.format(
        entity_json=entity_json,
        episode_summaries=episode_text[:6000],
        date=today,
    )

    provider, model = get_memory_model_settings("wiki")

    content = _call_llm_with_fallback(
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": "You produce clean, factual markdown wiki pages. No frontmatter."},
            {"role": "user", "content": user_msg},
        ],
    )
    return content


def distill_session_wiki(
    session_id: str,
    episodes: List[dict],
) -> Optional[str]:
    """Produce a wiki page for a session.

    Args:
        session_id: Session identifier.
        episodes: Episode dicts from this session.

    Returns:
        Markdown string if successful, None on failure.
    """
    if not episodes:
        return None

    episode_text = _format_episodes_for_wiki(episodes)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    user_msg = WIKI_SESSION_PROMPT.format(
        session_id=session_id,
        date=today,
        episode_summaries=episode_text[:6000],
    )

    provider, model = get_memory_model_settings("wiki")

    content = _call_llm_with_fallback(
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": "You produce clean, factual markdown wiki pages. No frontmatter."},
            {"role": "user", "content": user_msg},
        ],
    )
    return content


def run_daily_distill(db_path: Path = None) -> Dict[str, Any]:
    """Run the daily wiki distillation.

    Entry point for the 2AM cron job. Distills entities and recent
    sessions into wiki pages.

    Args:
        db_path: Optional path to index.db (uses default if None).

    Returns:
        Stats dict with pages generated and errors.
    """
    stats = {"entities_distilled": 0, "sessions_distilled": 0, "errors": 0}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        if db_path:
            store = EpisodicStore(db_path=db_path)
        else:
            store = EpisodicStore()
    except Exception as e:
        logger.error("Failed to open episodic store for wiki distill: %s", e)
        return {"error": str(e)}

    # Ensure output directory exists
    entities_dir = WIKI_OUTPUT_DIR / "entities"
    sessions_dir = WIKI_OUTPUT_DIR / "sessions"
    entities_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # 0. Tag banter episodes before distillation
    try:
        tagged = mark_banter_episodes(store)
        if tagged:
            logger.info("Tagged %d episodes as banter/chitchat", tagged)
    except Exception as e:
        logger.error("Banter tagging failed: %s", e)
        stats["errors"] += 1

    # 1. Distill entities (all recently updated)
    try:
        entities = store.get_recent_entities(limit=10)
        for entity in entities:
            profile = _parse_profile(entity.get("profile_json", {}))
            wiki_content = distill_entity_wiki(store, entity)
            if wiki_content:
                entity_type = _canonical_entity_type(entity.get("type", "unknown"), entity.get("name", "unknown"), profile)
                entity_name = _canonical_entity_name(entity.get("type", "unknown"), entity.get("name", "unknown"), profile)
                filename = _safe_entity_filename(entity.get("type", "unknown"), entity.get("name", "unknown"), profile)
                filepath = entities_dir / filename

                # Add frontmatter
                full_content = (
                    f"---\n"
                    f"type: {entity_type}\n"
                    f"name: {entity_name}\n"
                    f"updated: {today}\n"
                    f"---\n\n"
                    f"{wiki_content}\n"
                )
                filepath.write_text(full_content, encoding="utf-8")
                stats["entities_distilled"] += 1
                logger.info("Distilled entity wiki: %s", filename)
            else:
                logger.info("Skipped low-signal entity wiki: %s", entity.get("name", "unknown"))
    except Exception as e:
        logger.error("Entity wiki distillation failed: %s", e)
        stats["errors"] += 1

    # 2. Distill recent sessions (last 24h of episodes)
    try:
        recent_episodes = store.get_recent_episodes(limit=50)
        # Group by session_id
        by_session: Dict[str, List[dict]] = {}
        for ep in recent_episodes:
            sid = ep.get("session_id", "unknown")
            if sid not in by_session:
                by_session[sid] = []
            by_session[sid].append(ep)

        for sid, episodes in list(by_session.items())[:5]:  # Max 5 sessions per run
            selected_episodes = filter_session_episodes(episodes)
            wiki_content = distill_session_wiki(sid, selected_episodes)
            if wiki_content:
                safe_sid = sid.replace("/", "-").replace(":", "-")[:32]
                filename = f"{safe_sid}.md"
                filepath = sessions_dir / filename

                # Remove stale date-prefixed copies of this same session
                for old_file in sessions_dir.glob(f"*_{safe_sid}.md"):
                    old_file.unlink(missing_ok=True)

                full_content = (
                    f"---\n"
                    f"session: {sid}\n"
                    f"date: {today}\n"
                    f"episodes: {len(selected_episodes)}\n"
                    f"---\n\n"
                    f"{wiki_content}\n"
                )
                filepath.write_text(full_content, encoding="utf-8")
                stats["sessions_distilled"] += 1
                logger.info("Distilled session wiki: %s", filename)
            else:
                stats["errors"] += 1
                logger.warning("Session wiki distillation returned no content for %s", sid)
    except Exception as e:
        logger.error("Session wiki distillation failed: %s", e)
        stats["errors"] += 1

    # Write index
    try:
        index_path = WIKI_OUTPUT_DIR / "INDEX.md"
        entity_files = sorted(entities_dir.glob("*.md")) if entities_dir.exists() else []
        session_files = sorted(sessions_dir.glob("*.md")) if sessions_dir.exists() else []

        index_content = f"""# Session Memory Wiki

Last updated: {today}

## Entities ({len(entity_files)})
"""
        for f in entity_files:
            index_content += f"- [{f.stem}](entities/{f.name})\n"

        index_content += f"\n## Sessions ({len(session_files)})\n"
        for f in session_files:
            index_content += f"- [{f.stem}](sessions/{f.name})\n"

        index_path.write_text(index_content, encoding="utf-8")
    except Exception as e:
        logger.error("Wiki index generation failed: %s", e)

    store.close()
    logger.info("Daily wiki distill complete: %s", json.dumps(stats))
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_daily_distill()
    print(json.dumps(result, indent=2))
    return 1 if result.get("errors") else 0


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(main())
