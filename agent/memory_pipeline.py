"""MemoryPipeline -- organic memory infrastructure inside MemoryManager.

NOT a MemoryProvider.  No name, no tools, no system_prompt_block.
Pure interceptor: executes organic logic before/after MemoryManager lifecycle methods.

All methods are best-effort: exceptions are caught and logged at debug level,
never blocking upstream providers.

Design philosophy (浑然一体):
    Memory's organic properties (salience gating, silent engrams, consolidation,
    reconsolidation, predictive feedback, spreading activation) are infrastructure
    of the entire cognitive system -- not features of a specific storage backend.
    Just as synaptic plasticity is a universal property of neural circuits, not
    a "plugin" for the hippocampus, the MemoryPipeline lives inside MemoryManager
    and operates on ALL memory pathways regardless of which provider is active.

Architecture:
    MemoryManager
        └── MemoryPipeline (interceptor layer, NOT a provider)
            ├── SalienceScorer      (Layer 1: sensory gate)
            ├── SilentEngramEngine  (Layer 2: availability continuum)
            ├── ConsolidationEngine (Layer 3: sleep-like consolidation)
            ├── ReconsolidationEngine (Layer 4: prediction-error updates)
            ├── FeedbackCoordinator (Layer 5: predictive processing + learning)
            └── ActivationGraph     (Layer 6: spreading activation)
        └── providers[] (builtin + one external)

5 Architectural Invariants Preserved:
    1. MemoryProvider ABC contract unchanged
    2. Single external provider limit unchanged
    3. Tool registry unchanged (pipeline exposes no tools)
    4. ContextEngine orthogonality preserved
    5. run_agent.py integration points unchanged
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer 1: SalienceScorer (sensory gate)
# ===========================================================================

_EMOTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"[!！]{2,}"), 0.6),
    (re.compile(r"\b(urgent|critical|emergency|broken|crash|bug|fail)\b", re.I), 0.5),
    (re.compile(r"\b(down|outage|corrupt|overload|timeout|deadlock)\b", re.I), 0.45),
    (re.compile(r"\b(love|hate|amazing|terrible|awesome|awful)\b", re.I), 0.3),
    (re.compile(r"\b(worried|excited|frustrated|angry|happy|sad)\b", re.I), 0.35),
    (re.compile(r"\b(important|crucial|vital|essential|key)\b", re.I), 0.4),
]

_EMOTION_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"[!！]{2,}"), 0.6),
    (re.compile(r"(紧急|严重|崩溃|故障|坏了|挂了)"), 0.5),
    (re.compile(r"(喜欢|讨厌|太好了|太差了|棒极了|糟透了)"), 0.3),
    (re.compile(r"(担心|兴奋|沮丧|生气|开心|难过)"), 0.35),
    (re.compile(r"(重要|关键|必须|一定要|千万|别忘了)"), 0.4),
]

_IMPORTANCE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(decided|decision|agreed|confirmed|final)\b", re.I), 0.7),
    (re.compile(r"\b(requirement|spec|specification|constraint)\b", re.I), 0.6),
    (re.compile(r"\b(deploy|release|production|launch)\b", re.I), 0.6),
    (re.compile(r"\b(architecture|design|refactor|migrat)\b", re.I), 0.5),
    (re.compile(r"\b(remember|note|important|don't forget)\b", re.I), 0.8),
    (re.compile(r"\b(prefer|always|never|usually)\b", re.I), 0.5),
    (re.compile(r"\b(bug|issue|error|problem)\b", re.I), 0.4),
]


_IMPORTANCE_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(决定|确认|最终|确定)"), 0.7),
    (re.compile(r"(需求|规格|约束|限制)"), 0.6),
    (re.compile(r"(部署|发布|上线|投产)"), 0.6),
    (re.compile(r"(记住|笔记|重要|别忘)"), 0.8),
    (re.compile(r"(喜欢|总是|从不|通常)"), 0.5),
]

_TRIVIAL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"^(hi|hello|hey|thanks|ok|yes|no|sure)\s*[.!?]?\s*$", re.I), 0.9),
    (re.compile(r"^(good morning|good night|bye|see you)", re.I), 0.8),
    (re.compile(r"^(what time|what date|weather)", re.I), 0.5),
]

_TRIVIAL_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"^(你好|嗨|谢谢|好的|是|不是|嗯)\s*[。！？]?\s*$"), 0.9),
    (re.compile(r"^(早上好|晚安|再见|拜拜)"), 0.8),
    (re.compile(r"^(几点|什么时间|天气)"), 0.5),
]

# Temporal recency patterns -- expressions indicating a recent or
# time-sensitive event.  Matches boost the novelty dimension of salience.
_RECENCY_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(just now|right now|moments? ago|earlier today|this morning|this afternoon|this evening)", re.I), 0.6),
    (re.compile(r"(today|yesterday|last night)", re.I), 0.5),
    (re.compile(r"(this week|last week|recently|lately|just happened)", re.I), 0.4),
    (re.compile(r"(now|currently|at the moment|as we speak)", re.I), 0.45),
    (re.compile(r"(breaking|just in|update[:\s])", re.I), 0.55),
    (re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}"), 0.3),   # ISO date in text
]

_RECENCY_PATTERNS_ZH: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(刚才|刚刚|此刻|现在|今天早上|今天下午|今天晚上)"), 0.6),
    (re.compile(r"(今天|昨天|前天|昨晚)"), 0.5),
    (re.compile(r"(本周|上周|最近|近期|近日)"), 0.4),
    (re.compile(r"(目前|当前|眼下|此时此刻)"), 0.45),
    (re.compile(r"(最新消息|突发|更新[：:])"), 0.55),
    (re.compile(r"\d{4}[-年/]\d{2}[-月/]\d{2}"), 0.3),   # CJK date in text
]

_ZH_STOPWORDS: set[str] = {
    '的', '了', '是', '在', '我', '你', '他', '她', '它', '们',
    '这', '那', '有', '和', '与', '及', '或', '但', '而', '就',
    '都', '要', '会', '能', '可以', '不', '没', '也', '还', '把',
    '被', '让', '给', '从', '到', '对',
}


@dataclass
class SalienceResult:
    """Multi-dimensional salience score for a message."""
    overall: float = 0.0
    emotion: float = 0.0
    novelty: float = 0.5
    importance: float = 0.0
    repetition_penalty: float = 1.0
    temporal_recency_boost: float = 0.0
    is_trivial: bool = False


@dataclass
class _RepetitionDetector:
    """Detects topic repetition using content hashing (F3 power-law penalty)."""
    window_size: int = 50
    _recent: deque = field(default_factory=lambda: deque(maxlen=50))
    _topic_counts: dict[str, int] = field(default_factory=dict)

    def _fuzzy_bucket(self, text: str) -> str:
        words = [w for w in re.sub(r"[^\w\s]", "", text.lower()).split() if len(w) > 2]
        return " ".join(words[:5])

    def observe(self, text: str) -> float:
        bucket = self._fuzzy_bucket(text)
        if not bucket:
            return 1.0
        self._topic_counts[bucket] = self._topic_counts.get(bucket, 0) + 1
        self._recent.append(bucket)
        if len(self._recent) == self._recent.maxlen or len(self._topic_counts) > self._recent.maxlen * 2:
            window_counts: dict[str, int] = {}
            for b in self._recent:
                window_counts[b] = window_counts.get(b, 0) + 1
            for topic in list(self._topic_counts):
                if topic not in window_counts:
                    del self._topic_counts[topic]
                else:
                    self._topic_counts[topic] = window_counts[topic]
        n = self._topic_counts.get(bucket, 1)
        return max(0.1, 1.0 / math.sqrt(n))

    def reset(self) -> None:
        self._recent.clear()
        self._topic_counts.clear()


class SalienceScorer:
    """Multi-dimensional salience scorer — the sensory gate.

    Pure rule-based — no LLM calls, O(message_length) time.
    Scientific basis: F4 (CREB/excitability allocation).
    Thread-safe: all mutable state protected by _lock.
    """

    def __init__(self, novelty_window: int = 50) -> None:
        self._rep = _RepetitionDetector(window_size=novelty_window)
        self._lock = threading.Lock()

    def score(self, message: str) -> SalienceResult:
        if not message or not message.strip():
            return SalienceResult(overall=0.0, is_trivial=True)
        text = message.strip()
        with self._lock:
            trivial_penalty = 1.0
            for pattern, weight in _TRIVIAL_PATTERNS + _TRIVIAL_PATTERNS_ZH:
                if pattern.search(text):
                    trivial_penalty = min(trivial_penalty, 1.0 - weight)
            is_trivial = trivial_penalty < 0.3
            emotion = 0.0
            for pattern, weight in _EMOTION_PATTERNS + _EMOTION_PATTERNS_ZH:
                if pattern.search(text):
                    emotion = max(emotion, weight)
            if len(text) < 20:
                emotion *= 0.5
            importance = 0.0
            for pattern, weight in _IMPORTANCE_PATTERNS + _IMPORTANCE_PATTERNS_ZH:
                if pattern.search(text):
                    importance = max(importance, weight)
            if len(text) > 200:
                importance = min(1.0, importance + 0.1)
            freshness = self._rep.observe(text)
            novelty = freshness
            rep_factor = freshness
            # Bitemporal boost: recent-event expressions increase novelty
            recency_boost = 0.0
            for pattern, weight in _RECENCY_PATTERNS + _RECENCY_PATTERNS_ZH:
                if pattern.search(text):
                    recency_boost = max(recency_boost, weight)
            novelty = min(1.0, novelty + recency_boost)
            raw = (0.25 * emotion + 0.30 * novelty + 0.30 * importance
                   + 0.15 * min(1.0, len(text) / 200))
            adjusted = raw * rep_factor * (1.0 - (1.0 - trivial_penalty) * 0.8)
            overall = max(0.0, min(1.0, adjusted))
            return SalienceResult(
                overall=overall, emotion=emotion, novelty=novelty,
                importance=importance, repetition_penalty=rep_factor,
                temporal_recency_boost=recency_boost,
                is_trivial=is_trivial,
            )

    def reset(self) -> None:
        self._rep = _RepetitionDetector(window_size=self._rep.window_size)


# ===========================================================================
# Layer 2: SilentEngramEngine (availability continuum)
# ===========================================================================

class SilentEngramEngine:
    """Manages memory strength decay and recovery.

    Memories decay via power-law but NEVER reach zero.  Forgotten facts
    become "silent engrams" that can be recovered via context similarity.
    Scientific basis: F5 (Ryan et al. 2015 Science -- forgetting != erasure).

    When emotion_modulated_decay_enabled is True, emotionally arousing
    memories decay more slowly: adjusted_half_life = half_life *
    (1 + emotion_decay_multiplier * |valence|).  With default multiplier
    of 2.0, high-emotion memories (valence ~0.6) decay up to 2.2x slower.
    Scientific basis: McGaugh 2004 -- amygdala modulates emotionally
    arousing memory consolidation.

    Thresholds:
        active:      strength > 0.5
        semi_active:  0.2 < strength <= 0.5
        silent:       0.05 < strength <= 0.2
        buried:       strength <= 0.05
    """

    ACTIVE = 0.5
    SEMI_ACTIVE = 0.2
    SILENT = 0.05

    def __init__(self, half_life_hours: float = 720.0,
                 emotion_modulated_decay_enabled: bool = False,
                 emotion_decay_multiplier: float = 2.0) -> None:
        self._half_life = half_life_hours
        self._emotion_modulated = emotion_modulated_decay_enabled
        self._emotion_multiplier = emotion_decay_multiplier

    def apply_decay(self, state: 'PipelineState', hours_elapsed: float = 1.0,
                    emotional_valence: float | None = None) -> int:
        """Apply power-law decay to all engram strengths. Returns affected rows.

        When emotional_valence is provided and emotion_modulated_decay is enabled,
        the half-life is adjusted: adjusted = half_life * (1 + multiplier * |valence|).
        High emotion means slower decay (up to 3x with default multiplier).
        Scientific basis: McGaugh 2004 -- amygdala modulates emotionally arousing
        memory consolidation.
        """
        if not state:
            return 0
        try:
            effective_half_life = self._half_life
            if (emotional_valence is not None
                    and self._emotion_modulated
                    and abs(emotional_valence) > 0.0):
                effective_half_life = self._half_life * (
                    1.0 + self._emotion_multiplier * abs(emotional_valence))
            decay_factor = 0.5 ** (hours_elapsed / effective_half_life)
            with state._lock:
                cursor = state._conn.execute(
                    "UPDATE engram_strengths SET "
                    "strength = MAX(0.001, strength * ?), "
                    "last_accessed = CURRENT_TIMESTAMP "
                    "WHERE strength > 0.001",
                    (decay_factor,),
                )
                state._conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug("Engram decay failed: %s", e)
            return 0

    def apply_decay_with_emotion(
        self, state: 'PipelineState', hours_elapsed: float,
        emotional_valences: dict[str, float],
    ) -> int:
        """Apply per-memory emotion-modulated decay.

        Each memory_ref in emotional_valences gets its own adjusted half-life
        based on its emotional valence.  Memories not in the dict use the
        base half-life.

        Args:
            state: PipelineState with engram_strengths table.
            hours_elapsed: Fractional hours since last decay.
            emotional_valences: {memory_ref: emotional_valence} mapping.
                Valence is in [0, 1] range (typically 0.0-0.6 from SalienceScorer).

        Returns:
            Total number of affected rows.
        """
        if not state:
            return 0
        total_affected = 0
        try:
            with state._lock:
                # Update half_life_hours for targeted memories
                for ref, valence in emotional_valences.items():
                    if self._emotion_modulated and abs(valence) > 0.0:
                        adjusted = self._half_life * (
                            1.0 + self._emotion_multiplier * abs(valence))
                    else:
                        adjusted = self._half_life
                    state._conn.execute(
                        "UPDATE engram_strengths SET decay_half_life_hours = ? "
                        "WHERE memory_ref = ?",
                        (adjusted, ref),
                    )

                # Apply decay using per-row half_life_hours
                cursor = state._conn.execute(
                    "UPDATE engram_strengths SET "
                    "strength = MAX(0.001, strength * "
                    "  POWER(0.5, ? / decay_half_life_hours)), "
                    "last_accessed = CURRENT_TIMESTAMP "
                    "WHERE strength > 0.001",
                    (hours_elapsed,),
                )
                total_affected = cursor.rowcount

                # Reset half_life_hours for non-targeted memories back to base
                refs = list(emotional_valences.keys())
                if refs:
                    placeholders = ",".join("?" for _ in refs)
                    state._conn.execute(
                        f"UPDATE engram_strengths SET decay_half_life_hours = ? "
                        f"WHERE memory_ref NOT IN ({placeholders}) "
                        f"AND decay_half_life_hours != ?",
                        [self._half_life] + refs + [self._half_life],
                    )
                else:
                    state._conn.execute(
                        "UPDATE engram_strengths SET decay_half_life_hours = ? "
                        "WHERE decay_half_life_hours != ?",
                        (self._half_life, self._half_life),
                    )

                state._conn.commit()
        except Exception as e:
            logger.debug("Engram emotion-modulated decay failed: %s", e)
        return total_affected

    def strengthen(self, state: 'PipelineState', memory_ref: str,
                   delta: float = 0.03) -> float:
        """Strengthen an engram on retrieval (spacing effect). Returns new strength."""
        if not state:
            return 0.0
        try:
            with state._lock:
                row = state._conn.execute(
                    "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
                    (memory_ref,),
                ).fetchone()
                if row:
                    new_str = min(1.0, row["strength"] + delta)
                    state._conn.execute(
                        "UPDATE engram_strengths SET strength = ?, "
                        "last_accessed = CURRENT_TIMESTAMP, "
                        "access_count = access_count + 1 "
                        "WHERE memory_ref = ?",
                        (new_str, memory_ref),
                    )
                else:
                    new_str = min(1.0, 1.0 + delta)
                    state._conn.execute(
                        "INSERT INTO engram_strengths "
                        "(memory_ref, provider, strength) VALUES (?, 'unknown', ?)",
                        (memory_ref, new_str),
                    )
                state._conn.commit()
                return new_str
        except Exception as e:
            logger.debug("Engram strengthen failed: %s", e)
            return 0.0

    def classify(self, strength: float) -> str:
        """Classify strength into accessibility level."""
        if strength > self.ACTIVE:
            return "active"
        elif strength > self.SEMI_ACTIVE:
            return "semi_active"
        elif strength > self.SILENT:
            return "silent"
        return "buried"


# ===========================================================================
# Layer 3: ConsolidationEngine (sleep-like consolidation)
# ===========================================================================

class ConsolidationEngine:
    """Consolidates episodic memories into semantic schemas.

    Three-phase process mimicking sleep consolidation:
    1. Select: pick salient unconsolidated facts
    2. Transfer: group by entity/category, create schema candidates
    3. Integrate: merge with existing schemas or create new ones
    Scientific basis: F6 (Diekelmann & Born 2019 Nature Reviews Neuroscience).
    """

    def __init__(self, min_facts: int = 5) -> None:
        self._min_facts = min_facts

    def consolidate(self, state: 'PipelineState',
                    facts: list[dict] | None = None) -> dict:
        """Run consolidation. Returns summary dict.

        In Phase 1-2, this operates on pipeline_state.db schemas.
        In Phase 3+, it will pull facts from providers.

        Bitemporal consideration: facts are sorted by their effective
        timestamp (event_time > ingestion_time > created_at) so that
        temporally proximate facts are grouped together during schema
        creation.  Temporal proximity also boosts the initial confidence
        of a new schema -- facts within the same time window are more
        likely to describe the same underlying situation.
        """
        if not state:
            return {"schemas_created": 0, "schemas_updated": 0}
        created, updated = 0, 0
        try:
            with state._lock:
                # Consolidation runs whenever we have enough new facts
                if facts and len(facts) >= self._min_facts:
                    # Sort facts by temporal proximity: prefer event_time,
                    # fall back to ingestion_time, then created_at.
                    def _sort_key(f: dict) -> str:
                        return (f.get("event_time")
                                or f.get("ingestion_time")
                                or f.get("created_at")
                                or "")
                    facts_sorted = sorted(facts, key=_sort_key)

                    # Fix 2: Prefer facts with LOWER engram strength
                    # (they need consolidation most -- like sleep
                    # prioritizes fragile memories for replay)
                    # Batch-fetch all strengths in one query
                    refs_to_strength: dict[str, float] = {}
                    refs = [sha256(f.get('content', '').encode()
                                   ).hexdigest()[:16]
                            for f in facts_sorted]
                    unique_refs = list(set(refs))
                    if unique_refs:
                        placeholders = ",".join(
                            "?" for _ in unique_refs)
                        rows = state._conn.execute(
                            f"SELECT memory_ref, strength "
                            f"FROM engram_strengths "
                            f"WHERE memory_ref IN ({placeholders})",
                            unique_refs,
                        ).fetchall()
                        refs_to_strength = {
                            r["memory_ref"]: r["strength"]
                            for r in rows
                        }
                    def _engram_strength_key(fact_d: dict) -> float:
                        ref = sha256(
                            fact_d.get('content', '').encode()
                        ).hexdigest()[:16]
                        return refs_to_strength.get(ref, 1.0)
                    facts_sorted.sort(key=_engram_strength_key)

                    # Get existing schema contents for dedup
                    existing = state._conn.execute(
                        "SELECT content FROM schemas ORDER BY updated_at DESC LIMIT 20"
                    ).fetchall()
                    existing_contents = {r["content"][:50] for r in existing}

                    for fact in facts_sorted[:10]:
                        content = fact.get("content", "")
                        domain = fact.get("domain", "general")
                        if not content or len(content) < 10:
                            continue
                        if content[:50] in existing_contents:
                            updated += 1
                            continue
                        base_conf = self._temporal_confidence_boost(fact)
                        state._conn.execute(
                            "INSERT INTO schemas (content, domain, confidence) "
                            "VALUES (?, ?, ?)",
                            (content, domain, base_conf),
                        )
                        schema_id = state._conn.execute(
                            "SELECT last_insert_rowid()"
                        ).fetchone()[0]
                        mem_ref = sha256(
                            content.encode()
                        ).hexdigest()[:16]
                        state._conn.execute(
                            "INSERT OR IGNORE INTO "
                            "schema_sources "
                            "(schema_id, memory_ref, "
                            " provider) "
                            "VALUES (?, ?, ?)",
                            (schema_id, mem_ref,
                             "pipeline"),
                        )
                        created += 1

                    # OPT 6: Write cross-domain links for new schemas
                    _new_schemas = []
                    for _fact in facts_sorted[:10]:
                        _c = _fact.get('content', '')
                        _d = _fact.get('domain', 'general')
                        if (_c and len(_c) >= 10
                                and _c[:50] not in existing_contents):
                            _new_schemas.append((_c, _d))

                    if _new_schemas:
                        _entity_domains: dict[str, set[str]] = {}
                        for _c, _d in _new_schemas:
                            _ents = set(
                                re.findall(r'[A-Z][a-z]{2,}', _c))
                            _ents.update(
                                e for e in re.findall(
                                    r'[一-鿿]{2,6}', _c)
                                if e not in _ZH_STOPWORDS)
                            for _e in _ents:
                                _entity_domains.setdefault(
                                    _e, set()).add(_d)

                        for _e, _doms in _entity_domains.items():
                            _ph = ','.join('?' for _ in _doms)
                            _rows = state._conn.execute(
                                'SELECT DISTINCT domain FROM schemas '
                                f'WHERE domain NOT IN ({_ph}) '
                                'AND content LIKE ?',
                                list(_doms) + [f'%{_e}%'],
                            ).fetchall()
                            _all_doms = _doms | {
                                r['domain'] for r in _rows}
                            if len(_all_doms) >= 2:
                                _sorted = sorted(_all_doms)
                                for _i in range(len(_sorted)):
                                    for _j in range(
                                            _i + 1, len(_sorted)):
                                        state._conn.execute(
                                            'INSERT INTO '
                                            'cross_domain_links '
                                            '(entity, domain_a, '
                                            'domain_b, strength) '
                                            'VALUES (?, ?, ?, 0.5)',
                                            (_e, _sorted[_i],
                                             _sorted[_j]),
                                        )

                # Log the run (single commit for atomicity)
                state._conn.execute(
                    "INSERT INTO consolidation_runs "
                    "(session_id, memories_processed, schemas_created, schemas_updated) "
                    "VALUES (?, ?, ?, ?)",
                    (getattr(self, '_session_id', '') or "", len(facts or []), created, updated),
                )
                state._conn.commit()
        except Exception as e:
            logger.debug("Consolidation failed: %s", e)
        return {"schemas_created": created, "schemas_updated": updated}

    @staticmethod
    def _temporal_confidence_boost(fact: dict) -> float:
        """Compute initial confidence with temporal proximity boost.

        Facts with a recent event_time get higher confidence because
        they describe a temporally grounded event.
        """
        base_conf = 0.5
        event_ts = fact.get("event_time") or fact.get("ingestion_time")
        if event_ts:
            try:
                from datetime import datetime, timezone
                evt = datetime.fromisoformat(str(event_ts))
                if evt.tzinfo is None:
                    evt = evt.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - evt).total_seconds() / 3600.0
                if age_hours < 1.0:
                    base_conf = 0.65
                elif age_hours < 24.0:
                    base_conf = 0.55
            except Exception:
                pass
        return base_conf

    def extract_insights(self, messages: list) -> str:
        """Extract key facts from messages about to be discarded by compression."""
        insights = []
        for msg in messages[-5:]:  # last 5 messages
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if len(content) > 50:
                # Extract first sentence as insight
                first_sentence = content.split(".")[0][:200]
                if first_sentence.strip():
                    insights.append(f"- {first_sentence.strip()}")
        return "\n".join(insights) if insights else ""




# ===========================================================================
# Layer 3b: DeepConsolidationEngine (LLM-assisted abstraction)
# ===========================================================================


class DeepConsolidationEngine(ConsolidationEngine):
    """LLM-assisted deep consolidation for abstract schema generation.


    Extends the base ConsolidationEngine with a second pass that uses an
    LLM to synthesise higher-level abstract schemas from the concrete
    schemas produced by the base consolidation.
    """


    def __init__(self, llm_client=None, min_facts: int = 3) -> None:
        super().__init__(min_facts=min_facts)
        self._llm = llm_client


    def consolidate(self, state, facts=None) -> dict:
        """Run base consolidation, then LLM-assisted abstraction."""
        result = super().consolidate(state, facts)
        if not self._llm or not state:
            return result
        try:
            abstract_created = self._generate_abstract_schemas(state)
            result["abstract_schemas_created"] = abstract_created
        except Exception as e:
            logger.debug("Deep consolidation abstraction failed: %s", e)
        return result


    def _generate_abstract_schemas(self, state) -> int:
        """Use LLM to produce abstract schemas from existing schemas."""
        created = 0
        with state._lock:
            rows = state._conn.execute(
                "SELECT schema_id, content, domain, confidence "
                "FROM schemas ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
        if len(rows) < 2:
            return 0
        schema_desc = []
        for r in rows:
            schema_desc.append(
                f"[{r['domain']}] (conf={r['confidence']:.2f}) "
                f"{r['content'][:200]}"
            )
        _nl = chr(10)
        prompt = (
            "You are a memory consolidation assistant. Given these "
            "memory schemas, generate 1-3 higher-level abstract "
            "schemas that capture key patterns or themes."
            + _nl + _nl
            + "SCHEMAS:" + _nl + _nl.join(schema_desc) + _nl + _nl
            + "Format each abstract schema as: DOMAIN|STATEMENT"
            + _nl + "One per line."
        )
        try:
            response = self._llm.complete(prompt)
        except Exception as e:
            logger.debug("LLM abstract schema call failed: %s", e)
            return 0
        if not response:
            return 0
        existing_prefixes = set()
        with state._lock:
            for r in state._conn.execute(
                    "SELECT content FROM schemas").fetchall():
                existing_prefixes.add(r["content"][:50])
        for line in response.strip().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            domain = parts[0].strip().lower()
            content_val = parts[1].strip()
            if not content_val or len(content_val) < 10:
                continue
            if content_val[:50] in existing_prefixes:
                continue
            with state._lock:
                state._conn.execute(
                    "INSERT INTO schemas (content, domain, confidence) "
                    "VALUES (?, ?, ?)",
                    (content_val, domain, 0.70),
                )
                state._conn.commit()
            existing_prefixes.add(content_val[:50])
            created += 1
        return created

# ===========================================================================
# Layer 4: ReconsolidationEngine (prediction-error updates)
# ===========================================================================

class ReconsolidationEngine:
    """Prediction-error driven memory updates.

    When new information contradicts existing memories, the system enters
    a "reconsolidation" mode: evaluating the conflict and updating.
    Scientific basis: F8 (Sinclair & Barense 2019 Trends in Neurosciences).
    """

    def __init__(self, error_threshold: float = 0.3,
                 semantic_conflict_enabled: bool = False,
                 semantic_conflict_threshold: float = 0.7) -> None:
        self._threshold = error_threshold
        self._semantic_enabled = semantic_conflict_enabled
        self._semantic_threshold = semantic_conflict_threshold

    def check_retrieval(self, state: 'PipelineState',
                        query: str, result: str,
                        engrams: 'SilentEngramEngine | None' = None) -> None:
        """Record retrieval event for potential reconsolidation."""
        if not state:
            return
        try:
            ref = sha256(query.encode()).hexdigest()[:16]
            if engrams:
                engrams.strengthen(state, ref)
            else:
                SilentEngramEngine().strengthen(state, ref)
        except Exception as e:
            logger.debug("Reconsolidation check failed: %s", e)

    def detect_conflict(self, new_content: str,
                        existing_contents: list[str],
                        embed_fn: 'Any | None' = None,
                        llm_client: 'Any | None' = None) -> 'float | tuple[float, str]':
        """Detect prediction error between new and existing content.

        If semantic conflict detection is enabled and embed_fn is provided,
        delegates to detect_semantic_conflict for deeper analysis.

        Returns:
            float (legacy): error score [0, 1], high = high conflict
            tuple[float, str] (semantic): (error_score, action)
                action is one of: "update", "keep_both", "supersede", "no_conflict"
        """
        # Route to semantic detection when enabled and embeddings available
        if self._semantic_enabled and embed_fn is not None:
            return self.detect_semantic_conflict(
                new_content, existing_contents,
                embed_fn=embed_fn, llm_client=llm_client)

        # Legacy token-overlap heuristic
        if not existing_contents:
            return 0.0
        new_tokens = set(new_content.lower().split())
        max_overlap = 0.0
        for existing in existing_contents:
            existing_tokens = set(existing.lower().split())
            if not new_tokens or not existing_tokens:
                continue
            overlap = len(new_tokens & existing_tokens) / max(
                1, len(new_tokens | existing_tokens))
            max_overlap = max(max_overlap, overlap)
        # High overlap = low conflict, low overlap = high conflict
        return 1.0 - max_overlap

    def detect_semantic_conflict(
        self,
        new_content: str,
        existing_contents: list[str],
        embed_fn: 'Any | None' = None,
        llm_client: 'Any | None' = None,
    ) -> tuple[float, str]:
        """Two-stage semantic conflict detection.

        Stage 1 (Fast Filter): Compute embedding similarity between the new
        content and each existing content.  Only candidates with cosine
        similarity exceeding semantic_conflict_threshold proceed to Stage 2.

        Stage 2 (LLM Judgment): When llm_client is available, ask the LLM
        whether the new content truly contradicts the high-similarity
        candidates.

        Args:
            new_content: The incoming information to evaluate.
            existing_contents: List of existing memory contents to check against.
            embed_fn: Optional callable(text) -> list[float] that returns
                      an embedding vector for a given text.
            llm_client: Optional object with a .complete(prompt: str) -> str
                        method for LLM-based judgment.

        Returns:
            (error_score, action) where action is one of:
            "update", "keep_both", "supersede", "no_conflict".
        """
        if not existing_contents:
            return (0.0, "no_conflict")

        # --- Stage 1: Fast embedding similarity filter ---
        if embed_fn is None:
            score = self._token_overlap_conflict(new_content, existing_contents)
            action = "update" if score > 0.7 else "no_conflict"
            return (score, action)

        try:
            new_vec = embed_fn(new_content)
        except Exception as e:
            logger.debug("Embedding computation failed for new_content: %s", e)
            score = self._token_overlap_conflict(new_content, existing_contents)
            action = "update" if score > 0.7 else "no_conflict"
            return (score, action)

        if not new_vec:
            return (0.0, "no_conflict")

        candidates: list[tuple[str, float]] = []  # (content, similarity)
        for existing in existing_contents:
            try:
                existing_vec = embed_fn(existing)
            except Exception:
                continue
            if not existing_vec:
                continue
            sim = self._cosine_similarity(new_vec, existing_vec)
            if sim > self._semantic_threshold:
                candidates.append((existing, sim))

        if not candidates:
            return (0.0, "no_conflict")

        # --- Stage 2: LLM judgment ---
        if llm_client is not None:
            try:
                prompt = self._build_conflict_prompt(new_content, candidates)
                response = llm_client.complete(prompt)
                error_score, action = self._parse_llm_conflict_response(
                    response, candidates)
                return (error_score, action)
            except Exception as e:
                logger.debug("LLM conflict judgment failed: %s", e)

        # --- Heuristic fallback (no LLM) ---
        max_sim = max(s for _, s in candidates)
        error_score = 1.0 - max_sim
        if max_sim > 0.9:
            action = "update"
        elif max_sim > self._semantic_threshold + 0.1:
            action = "keep_both"
        else:
            action = "no_conflict"
        return (error_score, action)

    # -- internal helpers for semantic conflict detection --

    def _token_overlap_conflict(self, new_content: str,
                                existing_contents: list[str]) -> float:
        """Legacy token-overlap conflict score."""
        new_tokens = set(new_content.lower().split())
        max_overlap = 0.0
        for existing in existing_contents:
            existing_tokens = set(existing.lower().split())
            if not new_tokens or not existing_tokens:
                continue
            overlap = len(new_tokens & existing_tokens) / max(
                1, len(new_tokens | existing_tokens))
            max_overlap = max(max_overlap, overlap)
        return 1.0 - max_overlap

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for zero-length or mismatched vectors.
        """
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_conflict_prompt(
        new_content: str,
        candidates: list[tuple[str, float]],
    ) -> str:
        """Build LLM prompt for semantic conflict judgment.

        The prompt instructs the LLM to analyse whether the new content
        genuinely contradicts, supersedes, or complements each candidate.

        Args:
            new_content: The incoming information.
            candidates: List of (existing_content, similarity_score) tuples.

        Returns:
            A prompt string ready to send to the LLM.
        """
        parts: list[str] = []
        parts.append(
            "You are a memory conflict analyst.  Determine whether the "
            "NEW CONTENT genuinely contradicts any of the EXISTING KNOWLEDGE "
            "entries below, or whether it simply updates / extends / is "
            "independent of them."
        )
        parts.append("")
        parts.append("NEW CONTENT:")
        parts.append(new_content)
        parts.append("")
        parts.append("EXISTING KNOWLEDGE:")
        for idx, (content, sim) in enumerate(candidates, 1):
            parts.append(f"  [{idx}] (similarity={sim:.2f}) {content[:500]}")
        parts.append("")
        parts.append(
            "Respond with EXACTLY one of these labels on its own line:")
        parts.append(
            "  update      - new content directly corrects/overwrites an existing entry")
        parts.append(
            "  keep_both   - new content and existing entries are complementary; keep both")
        parts.append(
            "  supersede   - new content is a newer version that should replace existing")
        parts.append(
            "  no_conflict - no meaningful conflict detected")
        parts.append("")
        parts.append(
            "Then on a second line, provide a conflict severity score from 0.0 "
            "(no conflict) to 1.0 (severe conflict).")
        parts.append("Example response:")
        parts.append("update")
        parts.append("0.85")
        return "\n".join(parts)

    @staticmethod
    def _parse_llm_conflict_response(
        response: str,
        candidates: list[tuple[str, float]],
    ) -> tuple[float, str]:
        """Parse the LLM conflict judgment response.

        Args:
            response: Raw LLM output string.
            candidates: The candidates list (used for fallback heuristic).

        Returns:
            (error_score, action) tuple.
        """
        if not response:
            max_sim = max((s for _, s in candidates), default=0.0)
            return (1.0 - max_sim, "no_conflict")

        resp_lines = response.strip().splitlines()
        valid_actions = {"update", "keep_both", "supersede", "no_conflict"}
        action = "no_conflict"
        error_score = 0.0

        for line in resp_lines:
            stripped = line.strip().lower()
            if stripped in valid_actions:
                action = stripped
                break

        # Try to extract numeric score from any line
        for line in resp_lines:
            stripped = line.strip()
            match = re.search(r"(0\.\d+|1\.0+)", stripped)
            if match:
                try:
                    error_score = float(match.group(1))
                    error_score = max(0.0, min(1.0, error_score))
                    break
                except ValueError:
                    pass

        # Fallback: derive score from action if no numeric found
        if error_score == 0.0 and action != "no_conflict":
            max_sim = max((s for _, s in candidates), default=0.0)
            if action == "update":
                error_score = max(0.7, 1.0 - max_sim)
            elif action == "supersede":
                error_score = max(0.5, 1.0 - max_sim)
            elif action == "keep_both":
                error_score = max(0.2, 1.0 - max_sim)

        return (error_score, action)


# Layer 5: FeedbackCoordinator (predictive processing + learning)
# ===========================================================================

class FeedbackCoordinator:
    """Three interconnected feedback loops.

    1. SalienceLearner: learns which signals predict useful memories
    2. PredictiveModel: generates expectations from schemas
    3. CrossDomainBridge: discovers unexpected connections
    Scientific basis: Predictive coding (Friston 2010).
    Thread-safe: _pending_predictions protected by _lock.
    """

    def __init__(self) -> None:
        self._pending_predictions: list[str] = []
        self._reconsolidation: 'ReconsolidationEngine | None' = None
        self._lock = threading.Lock()

    def predict(self, state: 'PipelineState', context: str) -> list[str]:
        """Generate predictions from existing schemas.

        Bitemporal support: predictions are drawn from both high-confidence
        schemas AND recently-updated schemas (temporal query).  Recently
        updated schemas are tagged with a recency marker so downstream
        consumers know they describe near-current situations.
        """
        if not state:
            return []
        try:
            with state._lock:
                # High-confidence schemas (original logic)
                conf_rows = state._conn.execute(
                    "SELECT content, confidence, updated_at FROM schemas "
                    "WHERE confidence > 0.3 ORDER BY confidence DESC LIMIT 3"
                ).fetchall()
                # Recently updated schemas (temporal query -- last 24h)
                recent_rows = state._conn.execute(
                    "SELECT content, confidence, updated_at FROM schemas "
                    "WHERE updated_at >= datetime('now', '-1 day') "
                    "ORDER BY updated_at DESC LIMIT 3"
                ).fetchall()
            # Merge, deduplicating by content prefix
            seen_prefixes: set[str] = set()
            predictions: list[str] = []
            for row in conf_rows:
                prefix = row["content"][:50]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                predictions.append(
                    f"Expected pattern (conf={row['confidence']:.2f}): "
                    f"{row['content'][:100]}"
                )
            for row in recent_rows:
                prefix = row["content"][:50]
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                predictions.append(
                    f"Recent pattern (updated={row['updated_at']}, "
                    f"conf={row['confidence']:.2f}): {row['content'][:100]}"
                )
            # Fix 3: Include recently consolidated schemas as predictions
            try:
                last_run = state._conn.execute(
                    "SELECT timestamp FROM consolidation_runs "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if last_run:
                    recent_schemas = state._conn.execute(
                        "SELECT content, confidence FROM schemas "
                        "WHERE created_at >= ? OR updated_at >= ? "
                        "ORDER BY confidence DESC LIMIT 3",
                        (last_run["timestamp"], last_run["timestamp"]),
                    ).fetchall()
                    for srow in recent_schemas:
                        sprefix = srow["content"][:50]
                        if sprefix not in seen_prefixes:
                            seen_prefixes.add(sprefix)
                            predictions.append(
                                f"Consolidated schema "
                                f"(conf={srow['confidence']:.2f}): "
                                f"{srow['content'][:100]}"
                            )
            except Exception as e:
                logger.debug(
                    "Consolidated schema prediction failed: %s", e)
            # Insert predictions into predictions table
            try:
                for pred_text in predictions:
                    row = state._conn.execute(
                        "SELECT schema_id FROM schemas "
                        "WHERE ? LIKE '%' || "
                        "substr(content, 1, 50) || '%' "
                        "ORDER BY confidence DESC "
                        "LIMIT 1",
                        (pred_text,),
                    ).fetchone()
                    sid = (row["schema_id"]
                           if row else None)
                    state._conn.execute(
                        "INSERT INTO predictions "
                        "(schema_id, prediction, "
                        " context) "
                        "VALUES (?, ?, ?)",
                        (sid, pred_text, context),
                    )
                state._conn.commit()
            except Exception as e:
                logger.debug(
                    "Prediction insert failed: %s", e)
            with self._lock:
                self._pending_predictions = predictions
            return predictions
        except Exception as e:
            logger.debug("Prediction failed: %s", e)
            return []

    def observe_outcome(self, state: 'PipelineState',
                        actual: str) -> float:
        """Compare predictions against actual outcome. Returns error score."""
        with self._lock:
            pending = list(self._pending_predictions)
        if not pending or not state:
            return 0.0
        try:
            actual_tokens = set(actual.lower().split())
            max_error = 0.0
            for pred in pending:
                pred_tokens = set(pred.lower().split())
                if not pred_tokens or not actual_tokens:
                    continue
                overlap = len(pred_tokens & actual_tokens) / max(
                    1, len(pred_tokens | actual_tokens))
                error = 1.0 - overlap
                max_error = max(max_error, error)

            # Update schema confidence based on prediction error
            if max_error > 0.5:
                # High error: schema was wrong, decrease confidence
                with state._lock:
                    state._conn.execute(
                        "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05) "
                        "WHERE confidence > 0.3"
                    )
                    state._conn.commit()
                # Fix 4: High prediction error triggers reconsolidation
                if self._reconsolidation:
                    try:
                        self._reconsolidation.check_retrieval(
                            state,
                            pending[0][:200] if pending else "",
                            actual[:200],
                        )
                    except Exception as e:
                        logger.debug(
                            "Prediction-error reconsolidation "
                            "failed: %s", e)
            elif max_error < 0.2:
                # Low error: schema was right, increase confidence
                with state._lock:
                    state._conn.execute(
                        "UPDATE schemas SET confidence = MIN(1.0, confidence + 0.03) "
                        "WHERE confidence > 0.3"
                    )
                    state._conn.commit()

            with self._lock:
                self._pending_predictions = []
            return max_error
        except Exception as e:
            logger.debug("Observe outcome failed: %s", e)
            return 0.0

    def discover_bridges(self, state: 'PipelineState') -> int:
        """Discover cross-domain connections between entities."""
        if not state:
            return 0
        try:
            with state._lock:
                # Find entities that appear in multiple domains
                rows = state._conn.execute(
                    "SELECT entity, COUNT(DISTINCT domain_a) as domain_count "
                    "FROM cross_domain_links GROUP BY entity "
                    "HAVING domain_count >= 2"
                ).fetchall()
                return len(rows)
        except Exception as e:
            logger.debug("Bridge discovery failed: %s", e)
            return 0


# ===========================================================================
# Layer 6: ActivationGraph (spreading activation)
# ===========================================================================

class ActivationGraph:
    """Hebbian co-activation graph for spreading activation.

    When entities are co-retrieved, their connection strengthens.
    Activation spreads through the graph to pre-activate related memories.
    Scientific basis: Collins & Loftus (1975) spreading activation.
    """

    def __init__(self, edge_decay_hours: float = 168.0,
                 pagerank_damping: float = 0.85,
                 pagerank_max_iter: int = 20,
                 pagerank_enabled: bool = False) -> None:
        self._decay_hours = edge_decay_hours
        self._pr_damping = pagerank_damping
        self._pr_max_iter = pagerank_max_iter
        self._pr_enabled = pagerank_enabled

    def record_co_activation(self, state: 'PipelineState',
                             entities: list[str], delta: float = 0.1) -> None:
        """Strengthen edges between co-activated entities (Hebbian learning)."""
        if not state or len(entities) < 2:
            return
        try:
            with state._lock:
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        a, b = sorted([entities[i], entities[j]])
                        state._conn.execute(
                            "INSERT INTO activation_edges "
                            "(source_entity, target_entity, strength, co_activation_count) "
                            "VALUES (?, ?, ?, 1) "
                            "ON CONFLICT(source_entity, target_entity) DO UPDATE SET "
                            "strength = MIN(1.0, strength + ?), "
                            "co_activation_count = co_activation_count + 1, "
                            "last_activated = CURRENT_TIMESTAMP",
                            (a, b, delta, delta),
                        )
                state._conn.commit()
        except Exception as e:
            logger.debug("Co-activation recording failed: %s", e)

    def get_neighbors(self, state: 'PipelineState',
                      entity: str, min_strength: float = 0.3,
                      limit: int = 5) -> list[dict]:
        """Get strongly connected neighbors of an entity."""
        if not state:
            return []
        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT target_entity AS neighbor, strength FROM activation_edges "
                    "WHERE source_entity = ? AND strength >= ? "
                    "UNION ALL "
                    "SELECT source_entity AS neighbor, strength FROM activation_edges "
                    "WHERE target_entity = ? AND strength >= ? "
                    "ORDER BY strength DESC LIMIT ?",
                    (entity, min_strength, entity, min_strength, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("Get neighbors failed: %s", e)
            return []

    def expand_query(self, state: 'PipelineState',
                     query: str, limit: int = 3) -> list[str]:
        """Expand a query using spreading activation.

        Extracts entities from query, finds their neighbors, returns
        additional context strings.
        """
        if not state:
            return []
        try:
            # Simple entity extraction: capitalized words + Chinese entities
            entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
            zh_entities = [e for e in re.findall(r'[一-鿿]{2,6}', query)
                           if e not in _ZH_STOPWORDS]
            entities.extend(zh_entities)
            expansions = []
            for entity in entities[:3]:
                neighbors = self.get_neighbors(state, entity, limit=limit)
                for n in neighbors:
                    expansions.append(
                        f"[co-activated: {entity} → {n['neighbor']} "
                        f"(strength={n['strength']:.2f})]"
                    )
            return expansions
        except Exception as e:
            logger.debug("Query expansion failed: %s", e)
            return []

    def spread_activation_pagerank(
        self, state: 'PipelineState',
        seed_entities: list[str],
        damping: float | None = None,
        max_iter: int | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank spreading activation over the co-activation graph.

        Builds a NetworkX graph from activation_edges, runs personalized
        PageRank with *seed_entities* as the personalisation vector, and
        returns an entity->score mapping (seeds excluded).

        Requires ``networkx``; returns an empty dict when the library is
        absent so callers degrade gracefully.
        """
        if not state or not seed_entities:
            return {}
        try:
            import networkx as nx
        except ImportError:
            logger.debug(
                "networkx not installed -- PageRank spreading activation "
                "unavailable; falling back to direct neighbors.")
            return {}

        d = damping if damping is not None else self._pr_damping
        iters = max_iter if max_iter is not None else self._pr_max_iter
        seed_set = set(seed_entities)

        # --- Build graph from DB ---
        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT source_entity, target_entity, strength "
                    "FROM activation_edges WHERE strength > 0.01",
                ).fetchall()
        except Exception as e:
            logger.debug("PageRank graph load failed: %s", e)
            return {}

        if not rows:
            return {}

        G: nx.Graph = nx.Graph()
        for row in rows:
            G.add_edge(
                row["source_entity"], row["target_entity"],
                weight=float(row["strength"]),
            )

        # Ensure seeds present (even if isolated)
        for s in seed_entities:
            if s not in G:
                G.add_node(s)

        # Personalisation vector: equal weight on seeds, 0 elsewhere
        personalization = {n: (1.0 if n in seed_set else 0.0) for n in G}

        try:
            scores = nx.pagerank(
                G, alpha=damping, max_iter=iters,
                personalization=personalization, weight="weight",
            )
        except Exception as e:
            logger.debug("PageRank computation failed: %s", e)
            return {}

        # Exclude seeds, sort descending
        return {
            ent: score
            for ent, score in sorted(
                scores.items(), key=lambda kv: kv[1], reverse=True
            )
            if ent not in seed_set
        }

    def expand_query_deep(self, state: 'PipelineState',
                          query: str, limit: int = 5) -> list[str]:
        """Expand query using Personalized PageRank instead of direct neighbours.

        Extracts entities from the query, runs PageRank spreading
        activation, and returns formatted expansion strings for the
        top-*limit* scored entities.

        Falls back to ``expand_query`` when PageRank is disabled or
        ``networkx`` is unavailable.
        """
        if not state:
            return []
        if not self._pr_enabled:
            return self.expand_query(state, query, limit=limit)
        try:
            entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
            zh_entities = [e for e in re.findall(r'[一-鿿]{2,6}', query)
                           if e not in _ZH_STOPWORDS]
            entities.extend(zh_entities)
            if not entities:
                return []

            scores = self.spread_activation_pagerank(
                state, entities[:5])
            if not scores:
                return self.expand_query(state, query, limit=limit)

            expansions: list[str] = []
            for ent, score in list(scores.items())[:limit]:
                expansions.append(
                    f"[pagerank: {ent} (score={score:.4f})]"
                )
            return expansions
        except Exception as e:
            logger.debug("Deep query expansion failed: %s", e)
            return self.expand_query(state, query, limit=limit)

    def find_bridge_entities(self, state: 'PipelineState',
                             entity_a: str,
                             entity_b: str) -> list[str]:
        """Discover bridge entities on the shortest path between two nodes.

        Returns the intermediate entities (excluding *entity_a* and
        *entity_b*) along the shortest weighted path in the
        co-activation graph.  Requires ``networkx``.

        Returns an empty list when no path exists or the library is
        missing.
        """
        if not state or not entity_a or not entity_b:
            return []
        try:
            import networkx as nx
        except ImportError:
            logger.debug(
                "networkx not installed -- bridge entity discovery "
                "unavailable.")
            return []

        try:
            with state._lock:
                rows = state._conn.execute(
                    "SELECT source_entity, target_entity, strength "
                    "FROM activation_edges WHERE strength > 0.01",
                ).fetchall()
        except Exception as e:
            logger.debug("Bridge entity graph load failed: %s", e)
            return []

        if not rows:
            return []

        G: nx.Graph = nx.Graph()
        for row in rows:
            G.add_edge(
                row["source_entity"], row["target_entity"],
                weight=float(row["strength"]),
            )

        if entity_a not in G or entity_b not in G:
            return []

        try:
            # Use weight=1/strength so stronger edges are shorter
            path = nx.shortest_path(
                G, source=entity_a, target=entity_b, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        except Exception as e:
            logger.debug("Shortest-path computation failed: %s", e)
            return []

        # Exclude endpoints
        return [node for node in path if node not in (entity_a, entity_b)]

    def decay_edges(self, state: 'PipelineState',
                    hours_elapsed: float = 1.0) -> int:
        """Decay all edge strengths. Returns affected rows."""
        if not state:
            return 0
        try:
            decay_factor = 0.5 ** (hours_elapsed / self._decay_hours)
            with state._lock:
                cursor = state._conn.execute(
                    "UPDATE activation_edges SET "
                    "strength = MAX(0.01, strength * ?) "
                    "WHERE strength > 0.01",
                    (decay_factor,),
                )
                state._conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.debug("Edge decay failed: %s", e)
            return 0


# ===========================================================================

# ===========================================================================
# SleepScheduler (automatic sleep-driven consolidation)
# ===========================================================================


class SleepScheduler:
    """Automatic sleep scheduler for memory consolidation.

    Monitors per-message salience and idle gaps.  When the system has been
    quiet for ``idle_threshold_minutes`` AND the accumulated salience since
    the last sleep cycle exceeds ``salience_threshold``, triggers a two-phase
    sleep cycle:

    Phase 1 (SWS):  Run ConsolidationEngine to transfer episodic facts into
                     semantic schemas.
    Phase 2 (REM):  Run DreamEngine for structured selective replay.

    Thread-safe: all mutable state protected by ``_lock``.
    """

    def __init__(self, idle_threshold_minutes: float = 5.0,
                 salience_threshold: float = 10.0) -> None:
        self._idle_threshold_s: float = idle_threshold_minutes * 60.0
        self._salience_threshold: float = salience_threshold
        self._accumulated_salience: float = 0.0
        self._last_activity: float = time.time()
        self._sleeping: bool = False
        self._sleep_count: int = 0
        self._last_sleep_duration_s: float = 0.0
        self._lock = threading.Lock()
        # Injected by MemoryPipeline during initialize()
        self._state: PipelineState | None = None
        self._session_id: str = ""

    def on_message(self, salience_score: float) -> None:
        """Accumulate salience and update last-activity timestamp."""
        with self._lock:
            self._accumulated_salience += max(0.0, salience_score)
            self._last_activity = time.time()

    def should_sleep(self) -> bool:
        """True when idle AND accumulated salience >= threshold."""
        with self._lock:
            if self._sleeping:
                return False
            idle_seconds = time.time() - self._last_activity
            return (idle_seconds >= self._idle_threshold_s
                    and self._accumulated_salience >= self._salience_threshold)

    def sleep_cycle(self, consolidation_engine, dream_engine) -> dict:
        """Run Phase 1 (SWS) consolidation, Phase 2 (REM) dreaming, then
        reset accumulated salience.

        Returns a summary dict with keys ``sws``, ``rem``,
        ``salience_reset``, ``duration_s``.
        """
        with self._lock:
            if self._sleeping:
                return {"skipped": True, "reason": "already sleeping"}
            self._sleeping = True

        start = time.time()
        result: dict = {"sws": {}, "rem": {}}
        try:
            # --- Phase 1: SWS -- consolidation ---
            if consolidation_engine and self._state:
                try:
                    result["sws"] = consolidation_engine.consolidate(
                        self._state, facts=None)
                except Exception as e:
                    logger.debug("Sleep SWS consolidation failed: %s", e)
                    result["sws"] = {"error": str(e)}

            # --- Phase 2: REM -- dreaming ---
            if dream_engine:
                try:
                    dream_result = dream_engine.dream_cycle(
                        session_id=self._session_id)
                    result["rem"] = {
                        "mode": getattr(dream_result, "mode", "?"),
                        "facts_replayed": getattr(
                            dream_result, "facts_replayed", 0),
                    }
                except Exception as e:
                    logger.debug("Sleep REM dreaming failed: %s", e)
                    result["rem"] = {"error": str(e)}

            # --- Reset salience after full cycle ---
            duration = time.time() - start
            with self._lock:
                result["salience_reset"] = round(
                    self._accumulated_salience, 4)
                self._accumulated_salience = 0.0
                self._sleep_count += 1
                self._last_sleep_duration_s = duration
                self._sleeping = False

            result["duration_s"] = round(duration, 3)
            logger.debug(
                "Sleep cycle completed in %.1fs: sws=%s, rem=%s",
                duration, result["sws"], result["rem"])
        except Exception as e:
            logger.debug("Sleep cycle failed: %s", e)
            with self._lock:
                self._sleeping = False
            result["error"] = str(e)

        return result

    def get_status(self) -> dict:
        """Current state dict for health dashboard."""
        with self._lock:
            idle_seconds = time.time() - self._last_activity
            return {
                "sleeping": self._sleeping,
                "accumulated_salience": round(
                    self._accumulated_salience, 2),
                "salience_threshold": self._salience_threshold,
                "idle_seconds": round(idle_seconds, 1),
                "idle_threshold_minutes": round(
                    self._idle_threshold_s / 60.0, 1),
                "should_sleep": (
                    not self._sleeping
                    and idle_seconds >= self._idle_threshold_s
                    and self._accumulated_salience >= self._salience_threshold
                ),
                "sleep_count": self._sleep_count,
                "last_sleep_duration_s": round(
                    self._last_sleep_duration_s, 2),
            }

    def reset(self) -> None:
        """Reset accumulated salience and activity timer.

        Called during session switches so that salience does not bleed
        across session boundaries.
        """
        with self._lock:
            self._accumulated_salience = 0.0
            self._last_activity = time.time()


# ===========================================================================
# Helper: salience-to-engram-strength mapping
# ===========================================================================

def _salience_to_engram_strength(salience: float) -> float:
    """Map a salience score to an initial engram strength.

    High-salience memories start at full strength, moderate ones at 0.7,
    and low-salience ones at 0.4.  This mirrors the allocation logic
    in ``MemoryPipeline._score_and_record_salience``.
    """
    if salience > 0.5:
        return 1.0
    elif salience > 0.2:
        return 0.7
    return 0.4


# Pipeline Schema (database provisioning)
# ===========================================================================

_PIPELINE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS salience_weights (
    signal_type TEXT PRIMARY KEY, weight REAL NOT NULL,
    sample_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS salience_encoding_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
    fact_ref TEXT, emotion_score REAL, novelty_score REAL,
    importance_score REAL, overall_score REAL, was_helpful INTEGER DEFAULT -1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS engram_strengths (
    memory_ref TEXT PRIMARY KEY, provider TEXT NOT NULL,
    strength REAL DEFAULT 1.0, last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_half_life_hours REAL DEFAULT 720.0, access_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS schemas (
    schema_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,
    domain TEXT DEFAULT 'general', confidence REAL DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, hrr_vector BLOB
);
CREATE TABLE IF NOT EXISTS schema_sources (
    schema_id INTEGER REFERENCES schemas(schema_id),
    memory_ref TEXT NOT NULL, provider TEXT NOT NULL,
    contribution REAL DEFAULT 1.0, PRIMARY KEY (schema_id, memory_ref)
);
CREATE TABLE IF NOT EXISTS reconsolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_ref TEXT,
    old_content TEXT, new_content TEXT, prediction_error REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS consolidation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
    memories_processed INTEGER DEFAULT 0, schemas_created INTEGER DEFAULT 0,
    schemas_updated INTEGER DEFAULT 0, conflicts_found INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, schema_id INTEGER,
    prediction TEXT NOT NULL, context TEXT DEFAULT '', outcome TEXT DEFAULT '',
    error_score REAL DEFAULT 0.0, resolved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, resolved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS salience_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_ref TEXT,
    signal_type TEXT, signal_value REAL, was_helpful INTEGER DEFAULT 0,
    was_retrieved INTEGER DEFAULT 0, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS activation_edges (
    source_entity TEXT NOT NULL, target_entity TEXT NOT NULL,
    strength REAL DEFAULT 0.1, co_activation_count INTEGER DEFAULT 1,
    last_activated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_entity, target_entity)
);
CREATE TABLE IF NOT EXISTS cross_domain_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL,
    domain_a TEXT NOT NULL, domain_b TEXT NOT NULL,
    fact_refs_a TEXT DEFAULT '', fact_refs_b TEXT DEFAULT '',
    strength REAL DEFAULT 0.5, discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_engram_strength ON engram_strengths(strength DESC);
CREATE INDEX IF NOT EXISTS idx_engram_provider ON engram_strengths(provider);
CREATE INDEX IF NOT EXISTS idx_schemas_domain ON schemas(domain);
CREATE INDEX IF NOT EXISTS idx_schemas_confidence ON schemas(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_activation_source ON activation_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_activation_target ON activation_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_cross_links_entity ON cross_domain_links(entity);
CREATE INDEX IF NOT EXISTS idx_salience_feedback_ref ON salience_feedback(memory_ref, was_retrieved);
CREATE INDEX IF NOT EXISTS idx_schemas_updated_at ON schemas(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_consolidation_runs_ts ON consolidation_runs(timestamp DESC);
"""


# ===========================================================================
# PipelineState -- persistent storage
# ===========================================================================

class PipelineState:
    """Persistent state for the memory pipeline (pipeline_state.db).

    Design: single connection + threading.RLock, WAL mode, independent
    from any provider's database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "pipeline_state.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="pipeline_state.db")
        with self._lock:
            self._conn.executescript(_PIPELINE_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception as e:
            logger.debug("PipelineState close failed: %s", e)


# ===========================================================================
# MemoryPipeline -- the interceptor layer
# ===========================================================================

class MemoryPipeline:
    """Organic memory pipeline -- internal infrastructure of MemoryManager.

    NOT a MemoryProvider.  Pure interceptor wrapping MemoryManager lifecycle.
    All methods best-effort: exceptions caught at debug level, never blocking.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._state: PipelineState | None = None
        self._enabled: bool = self._config.get("enabled", True)
        self._session_id: str = ""
        # All 6 layers + episodic + dreaming
        self._salience: SalienceScorer | None = None
        self._engrams: SilentEngramEngine | None = None
        self._consolidation: ConsolidationEngine | None = None
        self._reconsolidation: ReconsolidationEngine | None = None
        self._feedback: FeedbackCoordinator | None = None
        self._activation: ActivationGraph | None = None
        self._episodic = None   # EpisodicTimeline (from holographic plugin)
        self._dreaming = None   # DreamEngine (from holographic plugin)
        self._evolution = None  # SelfEvolution (from holographic plugin)
        self._scheduler: SleepScheduler | None = None
        self._llm_client = None
        self._deep_consolidation: DeepConsolidationEngine | None = None

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize pipeline state and all organic modules."""
        if not self._enabled:
            return
        self._session_id = session_id
        db_path = self._config.get("db_path") or None
        self._state = PipelineState(db_path=db_path)

        # LLM client for deep consolidation (from kwargs or config)
        self._llm_client = kwargs.get("llm_client") or self._config.get("llm_client")

        self._init_core_layers()
        self._init_plugin_layers()

        logger.debug("MemoryPipeline initialized (session=%s, layers=%d)",
                      session_id, sum(1 for x in [self._salience, self._engrams,
                      self._consolidation, self._reconsolidation,
                      self._feedback, self._activation,
                      self._episodic, self._dreaming,
                      self._scheduler] if x))

    def _init_core_layers(self) -> None:
        """Initialize layers 1-6 from config (salience through activation)."""
        # Layer 1: SalienceScorer
        sal_cfg = self._config.get("salience", {})
        if sal_cfg.get("enabled", True):
            self._salience = SalienceScorer(
                novelty_window=sal_cfg.get("novelty_window", 50))

        # Layer 2: SilentEngramEngine
        eng_cfg = self._config.get("silent_engram", {})
        if eng_cfg.get("enabled", True):
            self._engrams = SilentEngramEngine(
                half_life_hours=eng_cfg.get("half_life_hours", 720.0),
                emotion_modulated_decay_enabled=eng_cfg.get(
                    "emotion_modulated_decay_enabled", False),
                emotion_decay_multiplier=eng_cfg.get(
                    "emotion_decay_multiplier", 2.0))

        # Layer 3: ConsolidationEngine (base + optional deep)
        con_cfg = self._config.get("consolidation", {})
        if con_cfg.get("enabled", True):
            self._consolidation = ConsolidationEngine(
                min_facts=con_cfg.get("min_facts_for_consolidation", 5))
            if con_cfg.get("deep_consolidation_enabled", False):
                self._deep_consolidation = DeepConsolidationEngine(
                    llm_client=self._llm_client,
                    min_facts=con_cfg.get("min_facts_for_consolidation", 5))

        # Layer 4: ReconsolidationEngine
        rec_cfg = self._config.get("reconsolidation", {})
        if rec_cfg.get("enabled", True):
            self._reconsolidation = ReconsolidationEngine(
                error_threshold=rec_cfg.get("prediction_error_threshold", 0.3),
                semantic_conflict_enabled=rec_cfg.get(
                    "semantic_conflict_enabled", False),
                semantic_conflict_threshold=rec_cfg.get(
                    "semantic_conflict_threshold", 0.7))

        # Layer 5: FeedbackCoordinator
        if self._config.get("feedback", {}).get("enabled", True):
            self._feedback = FeedbackCoordinator()
            if self._reconsolidation:
                self._feedback._reconsolidation = self._reconsolidation

        # Layer 6: ActivationGraph
        act_cfg = self._config.get("activation", {})
        if act_cfg.get("enabled", True):
            self._activation = ActivationGraph(
                edge_decay_hours=act_cfg.get("edge_decay_hours", 168.0),
                pagerank_damping=act_cfg.get("pagerank_damping", 0.85),
                pagerank_max_iter=act_cfg.get("pagerank_max_iter", 20),
                pagerank_enabled=act_cfg.get("pagerank_enabled", False),
            )

    def _init_plugin_layers(self) -> None:
        """Initialize layers 7-9 from config (episodic, dreaming, hippocampal, sleep)."""
        # Layer 7: EpisodicTimeline (what-where-when binding)
        epi_cfg = self._config.get("episodic", {})
        if epi_cfg.get("enabled", False):
            _mod = self._load_holographic_plugin(
                "holographic_episodic", "episodic.py")
            if _mod:
                self._episodic = _mod.EpisodicTimeline(
                    self._state._conn, self._state._lock)
                self._episodic.init_tables()

        # Layer 8: DreamEngine (structured selective replay)
        dream_cfg = self._config.get("dreaming", {})
        if dream_cfg.get("enabled", False):
            _mod = self._load_holographic_plugin(
                "holographic_dreaming", "dreaming.py")
            if _mod:
                self._dreaming = _mod.DreamEngine(
                    self._state._conn, self._state._lock,
                    cooldown_hours=dream_cfg.get("cooldown_hours", 1.0),
                    mode1_top_k=dream_cfg.get("mode1_top_k", 10),
                    mode2_top_k=dream_cfg.get("mode2_top_k", 5),
                    mode3_idle_hours=dream_cfg.get("mode3_idle_hours", 24.0),
                    mode3_min_schema_conf=dream_cfg.get(
                        "mode3_min_schema_conf", 0.7),
                )
                self._dreaming.init_tables()

        # Layer 8b: SelfEvolution (homeostatic self-regulation)
        evo_cfg = self._config.get('self_evolution', {})
        if evo_cfg.get('self_evolution_enabled', False):
            _mod = self._load_holographic_plugin(
                'self_evolution', 'self_evolution.py')
            if _mod:
                self._evolution = _mod.SelfEvolution(
                    self._state._conn, self._state._lock,
                    evo_cfg,
                    pipeline_conn=self._state._conn,
                )
                self._evolution.init_tables()

        # Layer 9a: HippocampalIndex (sparse index for pattern completion)
        hippo_cfg = self._config.get("hippocampal_index", {})
        self._hippocampal = None
        if hippo_cfg.get("enabled", False):
            _mod = self._load_holographic_plugin(
                "hippocampal_index", "hippocampal_index.py")
            if _mod:
                self._hippocampal = _mod.HippocampalIndex(
                    self._state._conn, self._state._lock)
                self._hippocampal.init_tables()

        # Layer 9: SleepScheduler (automatic sleep-driven consolidation)
        sleep_cfg = self._config.get("sleep", {})
        if sleep_cfg.get("enabled", False):
            self._scheduler = SleepScheduler(
                idle_threshold_minutes=sleep_cfg.get(
                    "idle_minutes", 5.0),
                salience_threshold=sleep_cfg.get(
                    "salience_threshold", 10.0),
            )
            self._scheduler._state = self._state
            self._scheduler._session_id = self._session_id

    def set_llm_client(self, llm_client) -> None:
        """Set or replace the LLM client for deep consolidation."""
        self._llm_client = llm_client

    def shutdown(self) -> None:
        """Flush and close pipeline state."""
        if self._state is not None:
            self._state.close()
            self._state = None
        logger.debug("MemoryPipeline shut down")

    # -- Shared helpers --

    def _load_holographic_plugin(self, module_name: str, file_name: str) -> 'Any | None':
        """Dynamically load a module from the holographic plugin directory.

        Args:
            module_name: Name to register the module as.
            file_name: Python file name (e.g. "episodic.py").

        Returns:
            The loaded module, or None on failure.
        """
        try:
            import importlib.util
            plugin_dir = (Path(__file__).resolve().parent.parent
                          / "plugins" / "memory" / "holographic")
            _spec = importlib.util.spec_from_file_location(
                module_name, str(plugin_dir / file_name))
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            return _mod
        except Exception as e:
            logger.debug("Plugin %s load failed: %s", file_name, e)
            return None

    def _score_and_record_salience(
        self, content: str, provider_tag: str = "salience_init",
    ) -> 'SalienceResult | None':
        """Score content for salience and record engram strength.

        Computes the salience score, then initializes or upgrades the
        engram_strengths row for the content's memory_ref so that
        high-salience content starts strong and low-salience content
        starts modestly.

        Args:
            content: Text to score.
            provider_tag: Value for the ``provider`` column in
                engram_strengths (used for provenance tracking).

        Returns:
            The SalienceResult, or None if scoring is unavailable.
        """
        if not self._salience or not self._state:
            return None
        try:
            result = self._salience.score(content)
            if self._engrams:
                init_str = _salience_to_engram_strength(result.overall)
                ref = sha256(content.encode()).hexdigest()[:16]
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO engram_strengths "
                        "(memory_ref, provider, strength) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(memory_ref) DO UPDATE SET "
                        "strength = MAX(strength, excluded.strength), "
                        "last_accessed = CURRENT_TIMESTAMP",
                        (ref, provider_tag, init_str),
                    )
                    self._state._conn.commit()
            return result
        except Exception as e:
            logger.debug("_score_and_record_salience failed: %s", e)
            return None

    def _record_salience_log(self, result: 'SalienceResult', source: str = "builtin") -> None:
        """Persist a salience score to the encoding log table."""
        if not self._state:
            return
        try:
            with self._state._lock:
                self._state._conn.execute(
                    "INSERT INTO salience_encoding_log "
                    "(source, emotion_score, novelty_score, "
                    "importance_score, overall_score) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (source, result.emotion, result.novelty,
                     result.importance, result.overall),
                )
                self._state._conn.commit()
        except Exception as e:
            logger.debug("Salience log insert failed: %s", e)

    def _update_salience_weights(self, error_score: float) -> None:
        """Adjust salience signal weights after a conflict or feedback event.

        High error nudges weights down; low error nudges them up.
        """
        if not self._state:
            return
        try:
            adj = -0.02 if error_score > 0.5 else 0.01
            for sig in ("emotion", "novelty", "importance"):
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO salience_weights "
                        "(signal_type, weight, "
                        "sample_count, success_count) "
                        "VALUES (?, 0.5, 1, ?) "
                        "ON CONFLICT(signal_type) "
                        "DO UPDATE SET "
                        "weight = MAX(0.1, "
                        "  MIN(1.0, weight + ?)), "
                        "sample_count = sample_count + 1, "
                        "success_count = success_count + ?, "
                        "updated_at = CURRENT_TIMESTAMP",
                        (sig,
                         1 if adj > 0 else 0,
                         adj,
                         1 if adj > 0 else 0),
                    )
                    self._state._conn.commit()
        except Exception as e:
            logger.debug("Salience weight update failed: %s", e)

    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract capitalized and Chinese entities from text."""
        entities = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        zh_entities = [
            e for e in re.findall(r'[一-鿿]{2,6}', text)
            if e not in _ZH_STOPWORDS
        ]
        entities.extend(zh_entities)
        return entities

    # -- Pre-interceptors --

    def pre_turn_start(self, turn: int, message: str) -> None:
        """Reset salience novelty window periodically, decay activation edges."""
        if self._salience and turn > 0 and turn % 100 == 0:
            try:
                self._salience.reset()
            except Exception as e:
                logger.debug("SalienceScorer reset failed: %s", e)
        if self._activation and self._state:
            try:
                self._activation.decay_edges(self._state, hours_elapsed=0.1)
            except Exception as e:
                logger.debug("Activation decay failed: %s", e)

    def pre_sync(self, user: str, asst: str,
                 embed_fn: 'Any | None' = None,
                 llm_client: 'Any | None' = None) -> dict | None:
        """Score user content for salience, persist signals.

        When semantic conflict detection is enabled in the reconsolidation
        config, this method also runs semantic conflict detection against
        existing schemas before the sync proceeds.

        Args:
            user: The user message content.
            asst: The assistant message content.
            embed_fn: Optional callable(text) -> list[float] for embeddings.
            llm_client: Optional LLM client with .complete(prompt) method.

        Returns:
            Metadata dict including salience scores and, when applicable,
            semantic conflict results, or None on failure.
        """
        if not self._salience:
            return None
        # Capture llm_client for deep consolidation
        if llm_client is not None:
            self._llm_client = llm_client
            if self._deep_consolidation:
                self._deep_consolidation._llm = llm_client
        try:
            # Score salience and initialize engram strength (consolidated)
            result = self._score_and_record_salience(user, "salience_init")
            if result is None:
                return None
            meta: dict = {
                "salience_overall": result.overall,
                "salience_emotion": result.emotion,
                "salience_novelty": result.novelty,
                "salience_importance": result.importance,
                "salience_is_trivial": result.is_trivial,
                "salience_temporal_recency_boost": result.temporal_recency_boost,
            }
            # --- Sleep scheduler: accumulate salience, maybe trigger sleep ---
            if self._scheduler:
                try:
                    self._scheduler.on_message(result.overall)
                    if self._scheduler.should_sleep():
                        import threading as _slp_thread
                        _slp_thread.Thread(
                            target=self._scheduler.sleep_cycle,
                            args=(self._consolidation, self._dreaming),
                            daemon=True,
                        ).start()
                except Exception as e:
                    logger.debug("SleepScheduler failed: %s", e)

            # Emotion-modulated engram decay
            if self._engrams and self._state:
                try:
                    decay_affected = self._engrams.apply_decay(
                        self._state,
                        hours_elapsed=1.0,
                        emotional_valence=result.emotion,
                    )
                    meta["decay_affected"] = decay_affected
                except Exception as e:
                    logger.debug("Emotion-modulated engram decay failed: %s", e)

            # Activation expansion
            if self._activation and self._state:
                try:
                    expansions = self._activation.expand_query(
                        self._state, user)
                    if expansions:
                        meta["activation_expansions"] = expansions
                except Exception as e:
                    logger.debug("Activation query expansion failed: %s", e)

            self._record_salience_log(result)

            # --- Semantic conflict detection ---
            self._detect_and_log_conflict(user, meta, embed_fn, llm_client)

            return meta
        except Exception as e:
            logger.debug("SalienceScorer.score failed: %s", e)
            return None

    def _detect_and_log_conflict(
        self, user: str, meta: dict,
        embed_fn: 'Any | None', llm_client: 'Any | None',
    ) -> None:
        """Run semantic conflict detection and log any detected conflict."""
        if not (self._reconsolidation
                and self._reconsolidation._semantic_enabled
                and self._state):
            return
        try:
            with self._state._lock:
                rows = self._state._conn.execute(
                    "SELECT content FROM schemas "
                    "ORDER BY confidence DESC LIMIT 20"
                ).fetchall()
            existing_contents = [r["content"] for r in rows if r["content"]]
            if not existing_contents:
                return

            error_score, action = self._reconsolidation.detect_semantic_conflict(
                user, existing_contents,
                embed_fn=embed_fn, llm_client=llm_client)
            meta["semantic_conflict_score"] = error_score
            meta["semantic_conflict_action"] = action

            if action != "no_conflict" and error_score > 0.2:
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO reconsolidation_log "
                        "(memory_ref, old_content, new_content, prediction_error) "
                        "VALUES (?, ?, ?, ?)",
                        ("pre_sync",
                         existing_contents[0][:500],
                         user[:500],
                         error_score),
                    )
                    self._state._conn.commit()
                logger.debug(
                    "Semantic conflict detected: action=%s, score=%.2f",
                    action, error_score)

            self._update_salience_weights(error_score)
        except Exception as e:
            logger.debug("Semantic conflict detection failed: %s", e)

    def pre_memory_write(
        self, action: str, target: str, content: str, metadata: dict
    ) -> dict | None:
        """Salience gate — score content, attach metadata."""
        if not self._salience or action not in ("add", "replace"):
            return None
        try:
            result = self._salience.score(content)
            if self._state:
                with self._state._lock:
                    self._state._conn.execute(
                        "INSERT INTO salience_encoding_log "
                        "(source, fact_ref, emotion_score, novelty_score, "
                        "importance_score, overall_score) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (target, content[:100], result.emotion, result.novelty,
                         result.importance, result.overall),
                    )
                    self._state._conn.commit()
            return {
                **metadata,
                "pipeline_salience": result.overall,
                "pipeline_emotion": result.emotion,
                "pipeline_novelty": result.novelty,
                "pipeline_importance": result.importance,
                "pipeline_temporal_recency_boost": result.temporal_recency_boost,
            }
        except Exception as e:
            logger.debug("SalienceScorer pre_memory_write failed: %s", e)
            return None

    def pre_compress(self, messages: list) -> str:
        """Extract key insights before context compression."""
        if not self._consolidation:
            return ""
        try:
            return self._consolidation.extract_insights(messages)
        except Exception as e:
            logger.debug("Consolidation extract_insights failed: %s", e)
            return ""

    # -- Post-interceptors --

    def post_prefetch(self, query: str, provider_results: list[str]) -> str:
        """Augment prefetch with predictions and spreading activation."""
        parts = []
        try:
            # Layer 5: predictions from schemas
            if self._feedback and self._state:
                predictions = self._feedback.predict(self._state, query)
                for pred in predictions:
                    parts.append(pred)

            # Layer 6: spreading activation
            if self._activation and self._state:
                expansions = self._activation.expand_query(self._state, query)
                parts.extend(expansions)
        except Exception as e:
            logger.debug("Pipeline post_prefetch failed: %s", e)
        return "\n".join(parts)

    def post_tool_call(self, name: str, args: dict, result: str) -> None:
        """Record retrieval for reconsolidation, co-activation."""
        if not self._state:
            return
        try:
            # Layer 4: reconsolidation check
            if self._reconsolidation and name == "fact_store":
                action = args.get("action", "")
                if action in ("search", "probe"):
                    self._reconsolidation.check_retrieval(
                        self._state, args.get("query", ""), result,
                        engrams=self._engrams)

            # Layer 5: observe outcome for prediction error feedback
            if self._feedback and name == "fact_store":
                action = args.get("action", "")
                if action in ("search", "probe"):
                    try:
                        self._feedback.observe_outcome(self._state, result[:500])
                    except Exception as e:
                        logger.debug('post_tool_call observe_outcome failed: %s', e)

            # Layer 6: record co-activation from search results
            if self._activation and name == "fact_store":
                query = args.get("query", "")
                entities = self._extract_entities_from_text(query)
                if len(entities) >= 2:
                    self._activation.record_co_activation(self._state, entities)

            # GAP 1: record co-activation from result content entities
            if self._activation and name == "fact_store":
                try:
                    result_entities = self._extract_entities_from_text(result)
                    if len(result_entities) >= 2:
                        self._activation.record_co_activation(
                            self._state, result_entities, delta=0.05)
                except Exception as e:
                    logger.debug(
                        "post_tool_call result co-activation "
                        "failed: %s", e)
            # GAP 3: 'add' action
            if name == "fact_store" and args.get("action") == "add":
                self._handle_fact_add(args, result)

            # GAP 5: fact_feedback
            if name == "fact_feedback":
                self._handle_fact_feedback(args)
        except Exception as e:
            logger.debug("Pipeline post_tool_call failed: %s", e)

    def _handle_fact_add(self, args: dict, result: str) -> None:
        """Process fact_store add action: salience, engram, co-activation, episodic."""
        content = args.get('content', '')
        if not content:
            return
        # 1. Score salience and init engram (consolidated)
        self._score_and_record_salience(content, 'add_action')

        # 2. Record co-activation of entities from content
        if self._activation:
            try:
                ents = self._extract_entities_from_text(content)
                if len(ents) >= 2:
                    self._activation.record_co_activation(
                        self._state, ents)
            except Exception as e:
                logger.debug('post_tool_call add co-activation failed: %s', e)

        # 3. Parse fact_id from result
        fact_id = self._parse_fact_id(result)
        if fact_id is None:
            return

        # 3b. Register engram with fact_id key so retrieval can find it
        if self._engrams and self._state:
            try:
                ref = f"fact:{fact_id}"
                self._engrams.strengthen(self._state, ref, delta=0.0)
            except Exception as e:
                logger.debug('post_tool_call engram fact_id register failed: %s', e)

        # 4. Append to current episode
        if self._episodic:
            try:
                self._episodic.append_fact(fact_id)
            except Exception as e:
                logger.debug('post_tool_call add episodic append failed: %s', e)

        # 5. Hippocampal index
        if self._hippocampal:
            try:
                ents = self._extract_entities_from_text(content)
                self._hippocampal.index_memory(
                    str(fact_id), content, entities=ents)
            except Exception as e:
                logger.debug('post_tool_call add hippocampal index failed: %s', e)

    def _handle_fact_feedback(self, args: dict) -> None:
        """Process fact_feedback: record feedback and update salience weights."""
        action_val = args.get('action', '')
        was_helpful = 1 if action_val == 'helpful' else 0
        fact_id = args.get('fact_id', 0)
        memory_ref = str(fact_id)

        if not self._state:
            return
        with self._state._lock:
            self._state._conn.execute(
                "INSERT INTO salience_feedback "
                "(memory_ref, signal_type, "
                " signal_value, was_helpful, "
                " was_retrieved) "
                "VALUES (?, 'fact_feedback', 1.0, ?, 1)",
                (memory_ref, was_helpful),
            )
            self._state._conn.commit()

        adj = 0.02 if was_helpful else -0.02
        self._update_salience_weights(-adj)

    @staticmethod
    def _parse_fact_id(result: str) -> 'int | None':
        """Extract fact_id from a JSON tool result string."""
        try:
            parsed = json.loads(result)
            return parsed.get('fact_id')
        except (json.JSONDecodeError, TypeError):
            return None

    def post_session_end(self, messages: list) -> None:
        """Consolidation, engram decay, bridge discovery, dreaming."""
        if not self._state:
            return
        try:
            # Layer 2: apply engram decay (1 hour worth)
            if self._engrams:
                self._engrams.apply_decay(self._state, hours_elapsed=1.0)

            # Layer 3: run consolidation (deep if available, else base)
            _engine = (
                self._deep_consolidation
                if (self._deep_consolidation
                    and self._deep_consolidation._llm)
                else self._consolidation)
            if _engine:
                facts = []
                for msg in messages[-10:]:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if content and len(content) > 20:
                        facts.append({"content": content, "domain": "general"})
                _engine.consolidate(self._state, facts)

            # Layer 5: discover cross-domain bridges
            if self._feedback:
                self._feedback.discover_bridges(self._state)

            # Layer 6: decay activation edges
            if self._activation:
                self._activation.decay_edges(self._state, hours_elapsed=1.0)

            # Layer 7: close episodic episode
            if self._episodic:
                try:
                    summary = f"Session {self._session_id}: {len(messages)} messages"
                    self._episodic.close_episode(summary=summary)
                    # Fix 7: Episodic to Consolidation: mini-consolidation
                    # on episode facts when episode closes
                    if self._consolidation:
                        try:
                            epi_facts = []
                            for msg in messages[-5:]:
                                c = (msg.get("content", "")
                                     if isinstance(msg, dict)
                                     else str(msg))
                                if c and len(c) > 15:
                                    epi_facts.append(
                                        {"content": c,
                                         "domain": "episode"})
                            min_req = max(
                                2,
                                self._consolidation._min_facts // 2)
                            if len(epi_facts) >= min_req:
                                self._consolidation.consolidate(
                                    self._state, facts=epi_facts)
                        except Exception as e:
                            logger.debug(
                                "Episodic mini-consolidation "
                                "failed: %s", e)
                except Exception as e:
                    logger.debug("Episodic close_episode failed: %s", e)

            # Layer 8: run dream cycle if conditions met
            if self._dreaming:
                try:
                    if self._dreaming.should_dream():
                        import threading as _t
                        _t.Thread(
                            target=self._run_dream_postprocessing,
                            daemon=True,
                        ).start()
                except Exception as e:
                    logger.debug("Dream cycle failed: %s", e)

            # Layer 8b: run self-evolution cycle if conditions met
            if self._evolution:
                import threading as _t2
                _t2.Thread(
                    target=self._evolution.run_evolution_cycle,
                    daemon=True,
                ).start()
        except Exception as e:
            logger.debug("Pipeline post_session_end failed: %s", e)

    def _run_dream_postprocessing(self) -> None:
        """Run dream cycle with post-processing (schema boost + predictions).

        Extracted from post_session_end for readability.  Runs in a
        daemon thread: boosts schema confidences after replay and adds
        dream hypotheses as pending predictions.
        """
        try:
            dr = self._dreaming.dream_cycle(self._session_id)
            if self._state:
                with self._state._lock:
                    self._state._conn.execute(
                        "UPDATE schemas SET "
                        "confidence = MIN(1.0, confidence + 0.02) "
                        "WHERE confidence > 0.5"
                    )
                    self._state._conn.commit()
            if self._feedback and dr and getattr(dr, "hypotheses", 0) > 0:
                try:
                    hyps = self._dreaming.get_hypotheses(limit=3)
                    if hyps:
                        with self._feedback._lock:
                            self._feedback._pending_predictions.extend(
                                [f"Dream: {h['content']}" for h in hyps])
                except Exception as e:
                    logger.debug("Dream prediction extension failed: %s", e)
        except Exception as e:
            logger.debug("Dream post-processing failed: %s", e)

    def post_session_switch(self, new_id: str, **kwargs) -> None:
        """Propagate session switch to pipeline internals.

        Updates the cached session_id, closes the current episodic episode
        and opens a new one, and resets the sleep scheduler so accumulated
        salience does not bleed across sessions.
        """
        old_id = self._session_id
        self._session_id = new_id

        # Close old episode, start new one
        if self._episodic:
            try:
                self._episodic.close_episode(
                    summary=f"Session {old_id} ended (switch)")
                self._episodic.start_episode(new_id)
            except Exception as e:
                logger.debug("Episode switch failed: %s", e)

        # Update sleep scheduler so salience accumulators reset
        if self._scheduler:
            try:
                self._scheduler._session_id = new_id
                self._scheduler.reset()
            except Exception as e:
                logger.debug("Scheduler reset failed: %s", e)

    def post_delegation(self, task: str, result: str, **kwargs) -> None:
        """No-op for now. Phase 2+: score subagent result."""
        pass

    def augment_system_prompt(self) -> str:
        """Inject organic memory status into system prompt."""
        if not self._state:
            return ""
        try:
            with self._state._lock:
                engram_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM engram_strengths"
                ).fetchone()[0]
                schema_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM schemas"
                ).fetchone()[0]
                edge_count = self._state._conn.execute(
                    "SELECT COUNT(*) FROM activation_edges"
                ).fetchone()[0]
            if engram_count == 0 and schema_count == 0:
                return ""
            return (
                f"[Organic Memory: {engram_count} engrams, "
                f"{schema_count} schemas, {edge_count} activation edges]"
            )
        except Exception as e:
            logger.debug("augment_system_prompt failed: %s", e)
            return ""


# ===========================================================================
# Config loader
# ===========================================================================

def _load_pipeline_config() -> dict:
    """Load memory.pipeline config from $HERMES_HOME/config.yaml."""
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        return cfg_get(config, "memory", "pipeline", default={}) or {}
    except Exception as e:
        logger.debug("Failed to load pipeline config: %s", e)
        return {}
