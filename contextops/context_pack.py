"""Deterministic context pack builder for the ContextOps Epistemic State Engine.

A context pack is a *phase-restoration packet*, not a transcript window. This
builder turns seed cognitive state (events / threads / tensions) plus the latest
user message into a minimal :class:`~contextops.models.ContextPack` with explicit
``restore`` and ``avoid`` sections.

Scoring is intentionally deterministic and privileges *cognitive pressure* over
shallow signals:

* open tensions on a thread weigh most heavily,
* heat is read from its *pressure* components (unresolvedness, contradiction
  density, ...) and never from recency alone,
* explicit evidence / anchor references named in the message are strong signals,
* cognitive-pressure words in the message rank above topic-label overlap.

A thread that is merely the most recent, or a bare topic-label match, must not
out-rank a thread carrying live unresolved pressure.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from contextops.models import ContextPack, Event, Tension, Thread

# Cognitive-pressure vocabulary. A message hit here signals an unresolved line
# of thinking rather than a topic. Kept deliberately small and inspectable.
PRESSURE_WORDS: frozenset[str] = frozenset(
    {
        "tension",
        "tensions",
        "unresolved",
        "unresolvedness",
        "contradiction",
        "contradictions",
        "anomaly",
        "anomalies",
        "coupling",
        "coupled",
        "pressure",
        "stuck",
        "still",
        "conflict",
        "drift",
        "tradeoff",
        "ambiguous",
        "uncertain",
        "reactivate",
        "recurring",
    }
)

# Heat components that reflect cognitive pressure. ``recency`` is excluded on
# purpose: high recency must never masquerade as high heat.
PRESSURE_HEAT_COMPONENTS: tuple[str, ...] = (
    "recurrence",
    "unresolvedness",
    "emotional_salience",
    "contradiction_density",
    "cross_thread_connectivity",
    "explicit_reactivation",
)

# Scoring weights, ordered by how much they reflect cognitive pressure.
_W_TENSION = 4.0
_W_PRESSURE_HEAT = 3.0
_W_EVIDENCE = 2.0
_W_PRESSURE_MSG = 0.5
_W_TOPIC = 0.25

# Threads scoring below this fraction of the top score are dropped from the
# pack: a phase packet stays minimal instead of restoring everything.
_SELECTION_FRACTION = 0.5

# The five contamination collapses the lane forbids (see epistemic-state-engine.md).
_AVOID_GUARDS: tuple[str, ...] = (
    "Do not treat a thread as a topic label; it is a persistent cognitive line.",
    "Do not treat heat as recency; pressure components, not recent mention, drive heat.",
    "Do not treat compaction as a summary; preserve the unresolved core.",
    "Do not treat the context pack as a transcript; it is a phase-restoration packet.",
    "Do not treat a StateDelta as note-taking; record only response-changing deltas.",
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _load_seed(seed: dict[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    """Return a plain seed dict from either an in-memory dict or a YAML path."""

    if isinstance(seed, dict):
        return seed
    if isinstance(seed, (str, os.PathLike)):
        data = yaml.safe_load(Path(seed).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("seed YAML must decode to a mapping")
        return data
    raise TypeError("seed must be a mapping or a path to a YAML file")


def _message_words(message: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(message)}


def _pressure_heat(thread: Thread) -> float:
    """Heat derived from pressure components only; falls back to raw heat."""

    components = thread.metadata.get("heat_components")
    if isinstance(components, dict):
        values = [
            float(components[name])
            for name in PRESSURE_HEAT_COMPONENTS
            if name in components
        ]
        if values:
            return sum(values) / len(values)
    return thread.heat


def _evidence_pool(thread: Thread, events: dict[str, Event], tensions: list[Tension]) -> set[str]:
    """All explicit references that anchor a thread: ids + event refs + tension refs."""

    pool: set[str] = set(thread.anchor_event_ids)
    for event_id in thread.anchor_event_ids:
        event = events.get(event_id)
        if event is not None:
            pool.update(event.refs)
    for tension in tensions:
        pool.update(tension.evidence_refs)
    return pool


def _score_thread(
    thread: Thread,
    *,
    open_tensions: list[Tension],
    message_words: set[str],
    message_lower: str,
    evidence_pool: set[str],
) -> float:
    open_tension_count = len(open_tensions)

    pressure_hits = len(message_words & PRESSURE_WORDS)
    evidence_overlap = sum(
        1 for ref in evidence_pool if ref.lower() in message_lower
    )
    topic_labels = thread.metadata.get("topic_labels", [])
    topic_hits = sum(
        1
        for label in topic_labels
        if isinstance(label, str) and label.lower() in message_lower
    )

    # Pressure words only meaningfully restore a thread that actually carries an
    # open tension; otherwise they are heavily discounted.
    pressure_msg_factor = 1.0 if open_tension_count else 0.3

    score = (
        _W_TENSION * open_tension_count
        + _W_PRESSURE_HEAT * _pressure_heat(thread)
        + _W_EVIDENCE * evidence_overlap
        + _W_PRESSURE_MSG * pressure_hits * pressure_msg_factor
        + _W_TOPIC * topic_hits
    )
    return round(score, 6)


def build_context_pack(
    seed: dict[str, Any] | str | os.PathLike[str],
    message: str,
    *,
    pack_id: str = "pack-contextops",
) -> ContextPack:
    """Build a deterministic :class:`ContextPack` from seed state and a message.

    ``seed`` is a mapping (or path to a YAML file) with ``events``, ``threads``
    and ``tensions`` lists. ``message`` is the latest user message that the pack
    should restore the correct cognitive phase for.
    """

    data = _load_seed(seed)

    events = {
        event.id: event
        for event in (Event.model_validate(row) for row in data.get("events", []))
    }
    threads = [Thread.model_validate(row) for row in data.get("threads", [])]
    tensions = [Tension.model_validate(row) for row in data.get("tensions", [])]
    if not threads:
        raise ValueError("seed must contain at least one thread")

    tensions_by_thread: dict[str, list[Tension]] = {}
    for tension in tensions:
        tensions_by_thread.setdefault(tension.thread_id, []).append(tension)

    message_words = _message_words(message)
    message_lower = message.lower()

    scores: dict[str, float] = {}
    open_by_thread: dict[str, list[Tension]] = {}
    evidence_by_thread: dict[str, set[str]] = {}
    for thread in threads:
        thread_tensions = tensions_by_thread.get(thread.id, [])
        open_tensions = sorted(
            (t for t in thread_tensions if t.status == "open"), key=lambda t: t.id
        )
        evidence_pool = _evidence_pool(thread, events, thread_tensions)
        open_by_thread[thread.id] = open_tensions
        evidence_by_thread[thread.id] = evidence_pool
        scores[thread.id] = _score_thread(
            thread,
            open_tensions=open_tensions,
            message_words=message_words,
            message_lower=message_lower,
            evidence_pool=evidence_pool,
        )

    # Rank by score desc, breaking ties on thread id for determinism.
    ranked = sorted(threads, key=lambda t: (-scores[t.id], t.id))
    top_score = scores[ranked[0].id]
    threshold = top_score * _SELECTION_FRACTION
    selected = [
        thread
        for thread in ranked
        if thread is ranked[0] or scores[thread.id] >= threshold
    ]

    restore: list[str] = []
    avoid: list[str] = list(_AVOID_GUARDS)
    selected_event_ids: set[str] = set()
    selected_tension_ids: list[str] = []
    evidence_refs: set[str] = set()

    for thread in selected:
        restore.append(f"Restore stance: {thread.stance}")
        for tension in open_by_thread[thread.id]:
            restore.append(f"Restore unresolved tension: {tension.description}")
            selected_tension_ids.append(tension.id)
        selected_event_ids.update(thread.anchor_event_ids)
        for tension in open_by_thread[thread.id]:
            selected_event_ids.update(tension.evidence_refs)
        evidence_refs.update(evidence_by_thread[thread.id])

    # Name de-selected recency-only threads in `avoid` so a recent mention cannot
    # contaminate the restored phase.
    for thread in ranked:
        if thread in selected:
            continue
        components = thread.metadata.get("heat_components", {})
        recency = components.get("recency", 0.0) if isinstance(components, dict) else 0.0
        if recency >= 0.5 and recency > _pressure_heat(thread):
            avoid.append(
                f"Do not restore {thread.id}: it ranked on recency, not cognitive pressure."
            )

    metadata: dict[str, Any] = {
        "message": message,
        "scores": scores,
        "selected_thread_ids": [thread.id for thread in selected],
        "evidence_refs": sorted(evidence_refs),
        "pressure_words_matched": sorted(message_words & PRESSURE_WORDS),
    }

    return ContextPack(
        id=pack_id,
        thread_ids=[thread.id for thread in selected],
        restore=restore,
        avoid=avoid,
        event_ids=sorted(selected_event_ids),
        tension_ids=sorted(selected_tension_ids),
        metadata=metadata,
    )
