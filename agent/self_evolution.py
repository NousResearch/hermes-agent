# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/memory/self-evolution.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Self-evolution ledger: track actionable policy / procedure /
constraint / failure-lesson memories the agent learns about itself.

The module is a *ledger*, not a mutator — it never rewrites code,
never changes permissions, never issues LLM calls. It watches a
stream of memories, filters for the actionable kinds, keeps a
bounded recent set, and emits an event so the surrounding system
can decide what (if anything) to do with the update.

.. warning::
   **Prompt caching hazard.** BaiLongma's original code renders the
   recent-events list into the system prompt every turn
   (``formatSelfEvolutionForPrompt``). In Hermes that would break
   per-conversation prompt caching — the system prompt has to stay
   byte-stable for the life of a conversation (see AGENTS.md).

   The rendered text is exposed here as a stand-alone function
   (:func:`format_self_evolution_for_prompt`) so callers can decide
   where to place it. Recommended uses:

   * Inject it as a **user message** at conversation start (once), or
   * Include it in a per-turn context block that is documented to
     invalidate the cache when it changes, or
   * Render it into a session-scoped surface (e.g. status panel,
     dashboard) rather than the model's system prompt.

   **Do not** wire this string into the system prompt on a schedule
   or on every recent-set change — you will invalidate the cache and
   multiply per-turn cost.

Ported from BaiLongma's ``self-evolution.js`` (MIT).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence


logger = logging.getLogger(__name__)


STATE_KEY = "self_evolution_state_v1"
STATE_VERSION = 1
MAX_RECENT = 24
PROMPT_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000  # 7 days

ACTIONABLE_TAGS: frozenset[str] = frozenset(
    {
        "kind:procedure",
        "kind:constraint",
        "kind:failure_lesson",
        "kind:policy",
    }
)

ACTIONABLE_EVENT_TYPES: frozenset[str] = frozenset({"self_constraint"})

_ACTIONABLE_MEM_ID_RE = re.compile(
    r"^(procedure|constraint|policy|lesson|rule)_", re.IGNORECASE
)


# ── Store protocol ─────────────────────────────────────────────────


class ConfigStore(Protocol):
    """Contract for a config-backed persistence layer.

    Two operations are enough:

    * ``get_config(key)`` — return the raw stored value (string or
      already-decoded object; the caller tolerates both).
    * ``set_config(key, value)`` — persist the JSON-encoded ledger
      state under ``key``.
    """

    def get_config(self, key: str) -> Any:
        ...

    def set_config(self, key: str, value: str) -> None:
        ...


class MemoryLookup(Protocol):
    """Optional lookup for "give me the full memory by ``mem_id``"
    used to enrich thin references from callers. If the caller only
    passes fully-formed memories, this can be omitted (``None``).
    """

    def get_memory_by_mem_id(self, mem_id: str) -> Optional[Mapping[str, Any]]:
        ...


# ── Data model ─────────────────────────────────────────────────────


@dataclass
class SelfEvolutionEntry:
    mem_id: str
    kind: str
    action: str
    title: str
    content: str
    salience: int
    tags: list[str]
    learned_at: str
    total_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mem_id": self.mem_id,
            "kind": self.kind,
            "action": self.action,
            "title": self.title,
            "content": self.content,
            "salience": self.salience,
            "tags": list(self.tags),
            "learned_at": self.learned_at,
            "total_events": self.total_events,
        }


@dataclass
class SelfEvolutionState:
    version: int = STATE_VERSION
    enabled: bool = True
    total_events: int = 0
    learned_count: int = 0
    last_at: Optional[str] = None
    recent: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "total_events": self.total_events,
            "learned_count": self.learned_count,
            "last_at": self.last_at,
            "recent": [dict(e) for e in self.recent],
        }


# ── Serialisation helpers ──────────────────────────────────────────


def _safe_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _safe_json_object(value: Any) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, Mapping) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize_state(raw: Any) -> SelfEvolutionState:
    parsed = _safe_json_object(raw) or {}
    recent_raw = parsed.get("recent") if isinstance(parsed, Mapping) else None
    recent = recent_raw if isinstance(recent_raw, list) else []
    filtered_recent = [
        dict(entry)
        for entry in recent
        if isinstance(entry, Mapping) and entry.get("mem_id")
    ][:MAX_RECENT]

    def _positive_int(key: str) -> int:
        try:
            return max(0, int(parsed.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    return SelfEvolutionState(
        version=STATE_VERSION,
        enabled=parsed.get("enabled") is not False,
        total_events=_positive_int("total_events"),
        learned_count=_positive_int("learned_count"),
        last_at=parsed.get("last_at") if parsed.get("last_at") else None,
        recent=filtered_recent,
    )


def _save_state(
    store: ConfigStore, state: SelfEvolutionState
) -> SelfEvolutionState:
    normalized = _normalize_state(state.to_dict())
    normalized.recent = normalized.recent[:MAX_RECENT]
    store.set_config(STATE_KEY, json.dumps(normalized.to_dict(), ensure_ascii=False))
    return normalized


_WHITESPACE_RE = re.compile(r"\s+")


def _truncate(text: Any, max_len: int = 220) -> str:
    value = _WHITESPACE_RE.sub(" ", str(text or "")).strip()
    if len(value) > max_len:
        return value[: max_len - 1] + "..."
    return value


def _tag_kind(tags: Iterable[Any]) -> str:
    for tag in tags:
        text = str(tag)
        if text.startswith("kind:"):
            return text[len("kind:") :]
    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _memory_to_entry(
    memory: Mapping[str, Any], source: Mapping[str, Any] | None = None
) -> SelfEvolutionEntry:
    source = source or {}
    tags = [str(t) for t in _safe_json_array(memory.get("tags"))]
    mem_id = str(
        memory.get("mem_id") or source.get("mem_id") or f"row:{memory.get('id')}"
    )
    kind = _tag_kind(tags)
    if not kind:
        event_type = memory.get("event_type")
        if event_type == "self_constraint":
            kind = "constraint"
        else:
            match = _ACTIONABLE_MEM_ID_RE.match(mem_id)
            kind = (match.group(1) if match else "policy").lower()

    try:
        salience = int(memory.get("salience", source.get("salience", 3)) or 3)
    except (TypeError, ValueError):
        salience = 3

    return SelfEvolutionEntry(
        mem_id=mem_id,
        kind=kind,
        action=str(source.get("action") or "observed"),
        title=_truncate(
            memory.get("title")
            or memory.get("content")
            or source.get("title")
            or "Self-evolution update",
            96,
        ),
        content=_truncate(
            memory.get("content") or source.get("content") or "", 240
        ),
        salience=salience,
        tags=tags,
        learned_at=_now_iso(),
    )


# ── Public API ─────────────────────────────────────────────────────


def get_self_evolution_state(store: ConfigStore) -> SelfEvolutionState:
    """Load and normalize the persisted ledger state."""
    return _normalize_state(store.get_config(STATE_KEY))


def get_self_evolution_snapshot(
    store: ConfigStore, *, max_recent: int = MAX_RECENT
) -> dict[str, Any]:
    """Read-only view suitable for a dashboard / debug panel."""
    state = get_self_evolution_state(store)
    try:
        limit = max(0, min(int(max_recent), MAX_RECENT))
    except (TypeError, ValueError):
        limit = MAX_RECENT
    return {
        "enabled": state.enabled,
        "version": state.version,
        "total_events": state.total_events,
        "learned_count": state.learned_count,
        "last_at": state.last_at,
        "recent": [dict(entry) for entry in state.recent[:limit]],
    }


def reset_self_evolution_state(store: ConfigStore) -> SelfEvolutionState:
    """Wipe the ledger back to defaults. Preserves ``enabled=True``."""
    return _save_state(store, SelfEvolutionState())


def is_self_evolution_memory(memory: Mapping[str, Any]) -> bool:
    """True if this memory row is one the ledger should track."""
    if not isinstance(memory, Mapping):
        return False
    tags = [str(t) for t in _safe_json_array(memory.get("tags"))]
    if any(t in ACTIONABLE_TAGS for t in tags):
        return True
    event_type = memory.get("event_type") or memory.get("type")
    if event_type in ACTIONABLE_EVENT_TYPES:
        return True
    return bool(_ACTIONABLE_MEM_ID_RE.match(str(memory.get("mem_id") or "")))


def record_self_evolution_from_memories(
    store: ConfigStore,
    memories: Sequence[Mapping[str, Any]],
    *,
    memory_lookup: Optional[MemoryLookup] = None,
    emit_event: Optional[Callable[[str, dict], None]] = None,
) -> list[dict[str, Any]]:
    """Ingest a batch of memory rows, keep only the actionable ones,
    update the ledger and (optionally) emit a ``self_evolution`` event.

    Duplicates within the batch are collapsed by ``mem_id``. Existing
    ledger entries are preserved unless a fresh entry with the same
    ``mem_id`` supersedes them. The recent list is capped at
    :data:`MAX_RECENT` after a newest-first sort.

    Returns the list of newly-learned entries (as plain dicts, each
    tagged with the updated ``total_events`` counter). An empty list
    means nothing was ingested — either no actionable rows or the
    ledger is disabled.
    """
    if not memories:
        return []

    state = get_self_evolution_state(store)
    if not state.enabled:
        return []

    learned: list[SelfEvolutionEntry] = []
    seen: set[str] = set()

    for item in memories:
        if not isinstance(item, Mapping):
            continue
        mem_id = str(item.get("mem_id") or item.get("id") or "")
        if not mem_id or mem_id in seen:
            continue
        seen.add(mem_id)

        full: Optional[Mapping[str, Any]] = None
        if memory_lookup is not None:
            try:
                full = memory_lookup.get_memory_by_mem_id(mem_id)
            except Exception as err:  # noqa: BLE001 — lookup is
                # advisory; a broken lookup must not lose the update.
                logger.debug(
                    "[self-evolution] memory lookup for %s raised %s",
                    mem_id,
                    err,
                )
                full = None

        memory = full or item
        if not is_self_evolution_memory(memory):
            continue
        learned.append(_memory_to_entry(memory, item))

    if not learned:
        return []

    by_id: dict[str, dict[str, Any]] = {
        entry.mem_id: entry.to_dict() for entry in learned
    }
    for entry in state.recent:
        entry_id = str(entry.get("mem_id") or "")
        if entry_id and entry_id not in by_id:
            by_id[entry_id] = dict(entry)

    next_recent = sorted(
        by_id.values(),
        key=lambda e: str(e.get("learned_at") or ""),
        reverse=True,
    )[:MAX_RECENT]

    new_state = SelfEvolutionState(
        version=STATE_VERSION,
        enabled=state.enabled,
        total_events=state.total_events + len(learned),
        learned_count=len(next_recent),
        last_at=_now_iso(),
        recent=next_recent,
    )
    persisted = _save_state(store, new_state)

    if callable(emit_event):
        try:
            emit_event(
                "self_evolution",
                {
                    "count": len(learned),
                    "entries": [e.to_dict() for e in learned],
                    "summary": get_self_evolution_snapshot(
                        store, max_recent=5
                    ),
                },
            )
        except Exception as err:  # noqa: BLE001 — event emission must
            # not block ledger progress.
            logger.warning(
                "[self-evolution] emit_event hook raised %s", err
            )

    return [
        {**entry.to_dict(), "total_events": persisted.total_events}
        for entry in learned
    ]


def format_self_evolution_for_prompt(
    store: ConfigStore,
    *,
    max_recent: int = 3,
    max_age_ms: int = PROMPT_MAX_AGE_MS,
    now_ms: Optional[float] = None,
) -> str:
    """Render the recent ledger entries as a prompt-ready string.

    .. warning::
       Do **not** splice this into the model's system prompt on a
       schedule. In Hermes the system prompt is byte-stable for the
       life of a conversation to preserve prompt caching (see the
       module docstring for the mitigation menu).

    Returns an empty string when the ledger is disabled or has no
    entries newer than ``max_age_ms``. Bounds ``max_recent`` to
    ``[1, 8]`` to keep the injected payload small.
    """
    state = get_self_evolution_state(store)
    if not state.enabled or not state.recent:
        return ""

    if now_ms is None:
        now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000.0
    cutoff = now_ms - max_age_ms

    def _within_age(entry: Mapping[str, Any]) -> bool:
        learned_at = entry.get("learned_at")
        if not learned_at:
            return True
        try:
            ts = datetime.fromisoformat(
                str(learned_at).replace("Z", "+00:00")
            ).timestamp() * 1000.0
        except (TypeError, ValueError):
            return True
        return ts >= cutoff

    filtered = [entry for entry in state.recent if _within_age(entry)]
    try:
        capped = max(1, min(int(max_recent), 8))
    except (TypeError, ValueError):
        capped = 3
    recent = filtered[:capped]
    if not recent:
        return ""

    lines: list[str] = []
    for entry in recent:
        title = str(entry.get("title") or "")
        title_part = f"{title}: " if title else ""
        lines.append(
            f"- [{entry.get('kind') or 'policy'}] {entry.get('mem_id')}: "
            f"{title_part}{entry.get('content') or ''}"
        )

    return "\n".join(
        [
            (
                "Self-evolution loop is active. It stores reusable "
                "procedures, constraints, and failure lessons as "
                "long-term policy memories. It does not rewrite source "
                "code or change permissions by itself."
            ),
            "Recent behavior updates:",
            *lines,
            (
                "Use this as provenance. Turn-specific guidance still "
                "comes from <active-policies> when a learned policy "
                "matches the current situation."
            ),
        ]
    )


__all__ = [
    "ACTIONABLE_EVENT_TYPES",
    "ACTIONABLE_TAGS",
    "ConfigStore",
    "MAX_RECENT",
    "MemoryLookup",
    "PROMPT_MAX_AGE_MS",
    "STATE_KEY",
    "STATE_VERSION",
    "SelfEvolutionEntry",
    "SelfEvolutionState",
    "format_self_evolution_for_prompt",
    "get_self_evolution_snapshot",
    "get_self_evolution_state",
    "is_self_evolution_memory",
    "record_self_evolution_from_memories",
    "reset_self_evolution_state",
]
