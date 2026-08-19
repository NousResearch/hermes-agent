"""Isolated X manager contract — approval-only, fail-closed, text-first.

Three lanes, all approval-gated:

1. **On-demand user-material transforms** — turn a piece of Sahil's own
   material (a note, a screenshot caption, a build log) into a draft X post.
2. **Quote-tweet candidate scan** — take a set of candidate source tweets and
   emit 3-5 substantive draft quote tweets.
3. **Morning finished X Article drafts** — take the day's chosen signals and
   emit a finished, text-first X Article draft.

Every artifact MUST carry an ``ArgumentPack`` — ``claim``, ``evidence``,
``mechanism``, ``position`` — plus lane-appropriate ``context``. An artifact
with an incomplete pack is rejected (fail closed) and never staged.

The manager has **no publish path**. It only ever stages artifacts as
``pending`` for explicit human approval. It deliberately does not import the
legacy ``x_scout`` / ``engagement_suggester`` / ``engagement_x_poster``
auto-post pathways, so the generic scout/engagement posting surface cannot be
reached through this contract. No automatic image generation: articles are
text-first unless a future approved evidence artifact is supplied.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import DB_PATH

# ── Lanes ─────────────────────────────────────────────────────────────────

LANE_TRANSFORM = "user_material_transform"
LANE_QUOTE_SCAN = "quote_tweet_scan"
LANE_ARTICLE = "morning_article"

LANES = (LANE_TRANSFORM, LANE_QUOTE_SCAN, LANE_ARTICLE)

# Required argument-pack fields. Every artifact must carry all four, non-empty.
REQUIRED_PACK_FIELDS = ("claim", "evidence", "mechanism", "position")

# Quote-tweet scan must emit between 3 and 10 substantive drafts.
QUOTE_SCAN_MIN = 3
QUOTE_SCAN_MAX = 10

# Dedicated approval channel. It is deliberately injectable at formatting time
# so tests and future Discord adapters cannot silently route to #content.
X_MANAGER_CHANNEL_ID = "1539435276160729148"

# Approval status vocabulary. There is no "published" state reachable from
# this module — publishing is a separate, human-driven step outside the
# manager's surface.
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
_STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)


class XManagerError(Exception):
    """Raised when an artifact fails the contract (incomplete pack, bad lane)."""


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class ArgumentPack:
    """Evidence/argument pack required on every X manager artifact.

    ``claim``      — the assertion the post makes.
    ``evidence``   — the supporting evidence (source, data, quote, link).
    ``mechanism``  — how/why it holds (the causal or explanatory step).
    ``position``   — the stance/angle Sahil takes on it.
    ``context``    — lane-appropriate context (source material ref, source
                     tweet, or the signals that grounded an article).
    """

    claim: str
    evidence: str
    mechanism: str
    position: str
    context: dict = field(default_factory=dict)

    def missing_fields(self) -> list[str]:
        return [
            f for f in REQUIRED_PACK_FIELDS
            if not (getattr(self, f) or "").strip()
        ]

    def is_complete(self) -> bool:
        return not self.missing_fields()


@dataclass
class XArtifact:
    """An approval-only X manager artifact.

    ``status`` starts at ``pending`` and can only move to ``approved`` or
    ``rejected`` via an explicit decision. There is no auto-publish.
    """

    id: str
    lane: str
    brand: str
    body: str
    pack: ArgumentPack
    status: str = STATUS_PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DeliveryCard:
    """Pure approval-card payload for a Discord adapter to deliver later.

    Keeping this as data, not an HTTP call, means manager code cannot publish
    or talk to any social platform as a side effect.
    """

    artifact_id: str
    channel_id: str
    body: str


# ── Persistence (approval-only, fail closed) ──────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS x_manager_artifacts (
    id          TEXT PRIMARY KEY,
    lane        TEXT NOT NULL,
    brand       TEXT NOT NULL,
    body        TEXT NOT NULL,
    claim       TEXT NOT NULL,
    evidence    TEXT NOT NULL,
    mechanism   TEXT NOT NULL,
    position    TEXT NOT NULL,
    context     TEXT NOT NULL,   -- JSON blob
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    decided_by  TEXT
);

CREATE INDEX IF NOT EXISTS idx_x_manager_status ON x_manager_artifacts(status);
CREATE INDEX IF NOT EXISTS idx_x_manager_lane ON x_manager_artifacts(lane);
"""


def init_db() -> None:
    """Idempotently create the X manager artifact table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _validate_artifact(artifact: XArtifact) -> None:
    """Fail closed: reject any artifact that violates the contract."""
    if artifact.lane not in LANES:
        raise XManagerError(f"unknown lane: {artifact.lane!r}")
    if artifact.status not in _STATUSES:
        raise XManagerError(f"unknown status: {artifact.status!r}")
    missing = artifact.pack.missing_fields()
    if missing:
        raise XManagerError(
            f"incomplete argument pack for {artifact.id}: missing {missing}"
        )
    if not (artifact.body or "").strip():
        raise XManagerError(f"artifact {artifact.id} has empty body")


def stage_for_approval(artifact: XArtifact) -> str:
    """Persist an artifact as ``pending``. Returns the artifact id.

    This is the only write path in the manager, and it always writes
    ``status='pending'`` — there is no code path that publishes. Raises
    ``XManagerError`` (and writes nothing) if the artifact is incomplete.
    """
    _validate_artifact(artifact)
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            INSERT INTO x_manager_artifacts
              (id, lane, brand, body, claim, evidence, mechanism, position,
               context, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.lane,
                artifact.brand,
                artifact.body,
                artifact.pack.claim,
                artifact.pack.evidence,
                artifact.pack.mechanism,
                artifact.pack.position,
                json.dumps(artifact.pack.context, ensure_ascii=False),
                STATUS_PENDING,
                artifact.created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return artifact.id


def format_approval_card(
    artifact: XArtifact,
    *,
    channel_id: str = X_MANAGER_CHANNEL_ID,
) -> DeliveryCard:
    """Return a paste-ready pending-approval card without sending it anywhere."""
    _validate_artifact(artifact)
    return DeliveryCard(
        artifact_id=artifact.id,
        channel_id=channel_id,
        body=(
            f"**X manager · {artifact.lane} · PENDING APPROVAL**\n"
            f"ID: `{artifact.id}`\n\n"
            f"{artifact.body}\n\n"
            f"**Claim:** {artifact.pack.claim}\n"
            f"**Evidence:** {artifact.pack.evidence}\n"
            f"**Mechanism:** {artifact.pack.mechanism}\n"
            f"**Position:** {artifact.pack.position}\n\n"
            "Approve or reject explicitly. This manager cannot publish."
        ),
    )


def stage_and_format_card(
    artifact: XArtifact,
    *,
    channel_id: str = X_MANAGER_CHANNEL_ID,
) -> DeliveryCard:
    """Stage an artifact then return its pending approval card, with no network I/O."""
    stage_for_approval(artifact)
    return format_approval_card(artifact, channel_id=channel_id)


def decide(artifact_id: str, action: str, decided_by: str = "") -> bool:
    """Apply an explicit approval decision. ``action`` in approve|reject.

    Returns True iff the row was found and updated. There is no "publish"
    action — approval only flips the row to ``approved``; the actual post is
    a separate human step outside this contract.
    """
    if action not in (STATUS_APPROVED, STATUS_REJECTED):
        raise XManagerError(f"unknown decision action: {action!r}")
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.execute(
            """
            UPDATE x_manager_artifacts
               SET status = ?, decided_at = ?, decided_by = ?
             WHERE id = ?
            """,
            (action, datetime.now(timezone.utc).isoformat(), decided_by, artifact_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_artifacts(status: Optional[str] = None, lane: Optional[str] = None) -> list[dict]:
    """List artifacts, optionally filtered by status and/or lane."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if lane is not None:
            clauses.append("lane = ?")
            params.append(lane)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM x_manager_artifacts{where} ORDER BY created_at ASC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Lane builders ──────────────────────────────────────────────────────────


def _new_id(lane: str) -> str:
    return f"xm_{lane[:4]}_{uuid.uuid4().hex[:10]}"


def transform_user_material(
    material: dict,
    *,
    brand: str = "sahil_twitter",
    pack: Optional[ArgumentPack] = None,
) -> XArtifact:
    """Lane 1: transform a piece of user material into a draft X post.

    ``material`` must carry the source text (``text``) and, optionally, a
    ``source`` label. The caller supplies the argument pack; if omitted, a
    pack is derived from the material but still validated (fail closed).
    """
    text = (material.get("text") or "").strip()
    source = material.get("source") or "user-material"
    if not text:
        raise XManagerError("user material has no text")

    if pack is None:
        pack = ArgumentPack(
            claim=text,
            evidence=source,
            mechanism="",
            position="",
            context={"source": source},
        )

    artifact = XArtifact(
        id=_new_id(LANE_TRANSFORM),
        lane=LANE_TRANSFORM,
        brand=brand,
        body=text,
        pack=pack,
    )
    _validate_artifact(artifact)
    return artifact


def scan_quote_tweet_candidates(
    candidates: list[dict],
    *,
    brand: str = "sahil_twitter",
) -> list[XArtifact]:
    """Lane 2: emit 3-5 substantive draft quote tweets from candidates.

    Each candidate must supply ``tweet_id``, ``author``, ``text`` and a
    ``pack`` (ArgumentPack). Candidates with an incomplete pack are dropped
    (fail closed). The result is clamped to [QUOTE_SCAN_MIN, QUOTE_SCAN_MAX];
    if fewer than QUOTE_SCAN_MIN candidates survive, an empty list is
    returned rather than padding with low-quality drafts.
    """
    artifacts: list[XArtifact] = []
    for c in candidates:
        tweet_id = c.get("tweet_id") or c.get("id") or ""
        author = c.get("author") or ""
        source_text = (c.get("text") or "").strip()
        quote_draft = (c.get("quote_draft") or "").strip()
        pack = c.get("pack")
        if not tweet_id or not source_text or not quote_draft or not isinstance(pack, ArgumentPack):
            continue
        # Re-staging the source tweet is not a quote tweet. It is a copy, so
        # reject it rather than manufacturing a weak candidate.
        if quote_draft.casefold() == source_text.casefold():
            continue
        if not pack.is_complete():
            continue
        context = dict(pack.context or {})
        context.setdefault("tweet_id", tweet_id)
        context.setdefault("author", author)
        context.setdefault("source_url", f"https://x.com/i/web/status/{tweet_id}")
        artifact = XArtifact(
            id=_new_id(LANE_QUOTE_SCAN),
            lane=LANE_QUOTE_SCAN,
            brand=brand,
            body=quote_draft,
            pack=ArgumentPack(
                claim=pack.claim,
                evidence=pack.evidence,
                mechanism=pack.mechanism,
                position=pack.position,
                context=context,
            ),
        )
        _validate_artifact(artifact)
        artifacts.append(artifact)

    if len(artifacts) < QUOTE_SCAN_MIN:
        return []
    return artifacts[:QUOTE_SCAN_MAX]


def morning_article_drafts(
    signals: list[dict],
    *,
    brand: str = "sahil_twitter",
    pack: Optional[ArgumentPack] = None,
    body: str = "",
) -> XArtifact:
    """Lane 3: emit a finished, text-first X Article draft from signals.

    ``signals`` are the day's chosen signals (each with at least a
    ``summary``). The article is text-first: no image generation is performed
    here. The caller supplies the finished ``body`` and ``pack``; both are
    validated (fail closed) before the artifact is returned.
    """
    if not signals:
        raise XManagerError("no signals supplied for article")
    summaries = [s.get("summary") or s.get("title") or "" for s in signals]
    summaries = [s for s in summaries if s.strip()]
    if not summaries:
        raise XManagerError("signals have no summary text")

    if pack is None:
        pack = ArgumentPack(
            claim=summaries[0],
            evidence="; ".join(summaries),
            mechanism="",
            position="",
            context={"signals": summaries},
        )

    artifact = XArtifact(
        id=_new_id(LANE_ARTICLE),
        lane=LANE_ARTICLE,
        brand=brand,
        body=body,
        pack=pack,
    )
    _validate_artifact(artifact)
    return artifact


def stage_morning_article_package(
    drafts: list[dict],
    *,
    channel_id: str = X_MANAGER_CHANNEL_ID,
) -> list[DeliveryCard]:
    """Stage at most two already-written, evidence-backed morning articles."""
    cards: list[DeliveryCard] = []
    for draft in drafts[:2]:
        artifact = morning_article_drafts(
            draft.get("signals") or [],
            brand=draft.get("brand", "sahil_twitter"),
            pack=draft.get("pack"),
            body=draft.get("body", ""),
        )
        cards.append(stage_and_format_card(artifact, channel_id=channel_id))
    return cards
