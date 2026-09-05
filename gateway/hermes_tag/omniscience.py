"""Provenance-bearing fact storage and scope-safe context assembly."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from .errors import IncompleteScope, StorageError
from .ledger import HermesTagLedger
from .model import (
    ContextBundle,
    Fact,
    ScopeRef,
    Sensitivity,
    canonical_json,
    new_id,
    parse_utc,
    utc_now,
    utc_text,
)


def _scope_from_mapping(raw: dict[str, Any]) -> ScopeRef:
    return ScopeRef(
        profile=raw["profile"],
        platform=raw["platform"],
        scope_id=raw["scope_id"],
        chat_id=raw["chat_id"],
        principal_id=raw["principal_id"],
        thread_id=raw.get("thread_id"),
        project_id=raw.get("project_id"),
        continuity_id=raw.get("continuity_id"),
    )


def _fact_from_row(row: sqlite3.Row) -> Fact:
    return Fact(
        fact_id=row["fact_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        value=json.loads(row["value_json"]),
        scope=_scope_from_mapping(json.loads(row["scope_json"])),
        source_type=row["source_type"],
        source_id=row["source_id"],
        source_revision=row["source_revision"],
        confidence=float(row["confidence"]),
        authority=int(row["authority"]),
        sensitivity=Sensitivity(int(row["sensitivity"])),
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        supersedes=row["supersedes"],
        tags=tuple(json.loads(row["tags_json"])),
    )


def _fact_visible_in(candidate: ScopeRef, requested: ScopeRef) -> bool:
    """Admit a fact only when every identity boundary is compatible.

    `*` is an explicit operator-authored wildcard for workspace- or
    channel-level facts. Optional dimensions on a fact are broader ancestors;
    they may flow down into a narrower request, never across another explicit
    value.
    """
    for name in ("profile", "platform", "scope_id", "principal_id"):
        value = getattr(candidate, name)
        if value not in {"*", getattr(requested, name)}:
            return False
    if candidate.chat_id not in {"*", requested.chat_id}:
        return False
    for name in ("thread_id", "project_id", "continuity_id"):
        value = getattr(candidate, name)
        if value is not None and value not in {"*", getattr(requested, name)}:
            return False
    return True


class FactStore:
    """Durable fact authority with explicit provenance and conflicts."""

    def __init__(self, ledger: HermesTagLedger) -> None:
        self.ledger = ledger

    def observe(self, fact: Fact) -> Fact:
        """Insert a fact idempotently and optionally supersede one predecessor."""
        with self.ledger.transaction() as connection:
            duplicate = connection.execute(
                "SELECT * FROM hermes_tag_facts WHERE content_hash=?",
                (fact.content_hash,),
            ).fetchone()
            if duplicate is not None:
                existing = _fact_from_row(duplicate)
                metadata_matches = (
                    existing.confidence == fact.confidence
                    and existing.authority == fact.authority
                    and existing.sensitivity == fact.sensitivity
                    and existing.valid_from == fact.valid_from
                    and existing.valid_until == fact.valid_until
                    and existing.supersedes == fact.supersedes
                    and existing.tags == fact.tags
                )
                if not metadata_matches:
                    raise StorageError(
                        "duplicate fact content carries conflicting metadata"
                    )
                return existing

            if fact.supersedes is not None:
                prior = connection.execute(
                    "SELECT * FROM hermes_tag_facts WHERE fact_id=?",
                    (fact.supersedes,),
                ).fetchone()
                if prior is None:
                    raise StorageError("superseded fact does not exist")
                prior_fact = _fact_from_row(prior)
                if (
                    prior_fact.subject != fact.subject
                    or prior_fact.predicate != fact.predicate
                    or prior_fact.scope.digest != fact.scope.digest
                ):
                    raise IncompleteScope(
                        "a fact may supersede only the same predicate in the same scope"
                    )
                connection.execute(
                    "UPDATE hermes_tag_facts SET active=0 WHERE fact_id=?",
                    (fact.supersedes,),
                )

            connection.execute(
                """
                INSERT INTO hermes_tag_facts(
                    fact_id, subject, predicate, value_json, scope_json,
                    source_type, source_id, source_revision, confidence,
                    authority, sensitivity, valid_from, valid_until,
                    supersedes, tags_json, content_hash, active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    fact.fact_id,
                    fact.subject,
                    fact.predicate,
                    canonical_json(fact.value),
                    canonical_json(fact.scope),
                    fact.source_type,
                    fact.source_id,
                    fact.source_revision,
                    float(fact.confidence),
                    int(fact.authority),
                    int(fact.sensitivity),
                    fact.valid_from,
                    fact.valid_until,
                    fact.supersedes,
                    canonical_json(fact.tags),
                    fact.content_hash,
                    fact.valid_from,
                ),
            )
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="fact.observed",
            payload={
                "fact_id": fact.fact_id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "content_hash": fact.content_hash,
                "source_type": fact.source_type,
                "source_id": fact.source_id,
                "source_revision": fact.source_revision,
                "scope_digest": fact.scope.digest,
                "authority": fact.authority,
                "sensitivity": int(fact.sensitivity),
                "supersedes": fact.supersedes,
            },
        )
        return fact

    def get(self, fact_id: str) -> Fact:
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                "SELECT * FROM hermes_tag_facts WHERE fact_id=?", (fact_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise StorageError("unknown fact")
        return _fact_from_row(row)

    def query(
        self,
        scope: ScopeRef,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        tags: Iterable[str] = (),
        sensitivity_ceiling: Sensitivity = Sensitivity.INTERNAL,
        max_facts: int = 64,
        max_chars: int = 12000,
        now: datetime | None = None,
    ) -> ContextBundle:
        if max_facts < 1 or max_chars < 256:
            raise ValueError("context bounds are too small")
        ceiling = Sensitivity.coerce(sensitivity_ceiling)
        current = now or utc_now()
        clauses = ["active=1", "valid_from<=?"]
        params: list[Any] = [utc_text(current)]
        if subject is not None:
            clauses.append("subject=?")
            params.append(subject)
        if predicate is not None:
            clauses.append("predicate=?")
            params.append(predicate)
        connection = self.ledger.connection()
        try:
            rows = connection.execute(
                f"""
                SELECT * FROM hermes_tag_facts
                WHERE {' AND '.join(clauses)}
                ORDER BY authority DESC, confidence DESC, valid_from DESC, fact_id
                """,
                params,
            ).fetchall()
        finally:
            connection.close()

        requested_tags = set(tags)
        candidates: list[Fact] = []
        omitted = 0
        for row in rows:
            fact = _fact_from_row(row)
            # Cross-scope rows are invisible, including through omission counts.
            if not _fact_visible_in(fact.scope, scope):
                continue
            if fact.valid_until is not None and parse_utc(fact.valid_until) <= current:
                omitted += 1
                continue
            if fact.sensitivity > ceiling:
                omitted += 1
                continue
            if requested_tags and not requested_tags.issubset(set(fact.tags)):
                omitted += 1
                continue
            candidates.append(fact)

        groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        for fact in candidates:
            groups[(fact.subject, fact.predicate)].append(fact)

        selected: list[Fact] = []
        conflicts: list[tuple[str, ...]] = []
        for key in sorted(groups):
            group = groups[key]
            top_authority = max(item.authority for item in group)
            top = [item for item in group if item.authority == top_authority]
            value_shapes = {canonical_json(item.value) for item in top}
            if len(value_shapes) > 1:
                conflicts.append(tuple(sorted(item.fact_id for item in top)))
                selected.extend(sorted(top, key=lambda item: item.fact_id))
            else:
                selected.append(
                    max(top, key=lambda item: (item.confidence, item.valid_from, item.fact_id))
                )

        selected.sort(
            key=lambda item: (
                -item.authority,
                -item.confidence,
                item.subject,
                item.predicate,
                item.fact_id,
            )
        )
        if len(selected) > max_facts:
            omitted += len(selected) - max_facts
            selected = selected[:max_facts]

        lines: list[str] = []
        rendered_facts: list[Fact] = []
        used_chars = 0
        for fact in selected:
            line = canonical_json(
                {
                    "fact_id": fact.fact_id,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": fact.value,
                    "source": {
                        "type": fact.source_type,
                        "id": fact.source_id,
                        "revision": fact.source_revision,
                    },
                    "confidence": fact.confidence,
                    "authority": fact.authority,
                    "sensitivity": fact.sensitivity.name.lower(),
                }
            )
            added = len(line) + (1 if lines else 0)
            if used_chars + added > max_chars:
                omitted += 1
                continue
            lines.append(line)
            rendered_facts.append(fact)
            used_chars += added

        return ContextBundle(
            facts=tuple(rendered_facts),
            conflicts=tuple(conflicts),
            omitted_count=omitted,
            rendered_text="\n".join(lines),
        )
