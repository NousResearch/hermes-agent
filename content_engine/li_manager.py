"""Isolated LinkedIn manager contract - approval-only, fail-closed, text-first.

Separate from the X manager and from SahilBlog. Two lanes, both approval-gated:

1. **Daily package** - exactly (1) one short, opinion-led PM Insight summary
   carrying a direct link to the full post, plus (2) two independent,
   evidence-backed AI/PM posts. The cardinality is a hard contract: the
   package is rejected (fail closed) unless it is exactly 1 insight + 2 posts.

2. **On-demand user material** - turn a piece of Sahil's own shared material
   (a note, a link, a snippet) into a single draft LinkedIn post.

Every artifact MUST carry an ``ArgumentPack`` - ``claim``, ``evidence``,
``mechanism``, ``position`` - plus lane-appropriate ``context`` and, for the
PM insight, a non-empty ``source_url`` (direct link to the full post). An
artifact with an incomplete pack is rejected and never staged.

The manager has **no publish path**. It only ever stages artifacts as
``pending`` for explicit human approval. It deliberately does not import the
legacy social generators, Postiz bridge, blog pipeline, or the X manager, so
those surfaces cannot be reached through this contract. No automatic image
generation, no Postiz enqueue, no Blog/X coupling.

Isolation from the legacy personal-brand backlog: this manager persists to its
own ``li_manager_artifacts`` table and never reads the legacy ``drafts`` table
(which still holds the old rejected sahil_linkedin rows). Those rows are
dead to this contract and can never be resurfaced through it.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from config import DB_PATH

# ── Lanes ─────────────────────────────────────────────────────────────────

LANE_DAILY_PACKAGE = "daily_package"
LANE_ON_DEMAND = "on_demand_material"

LANES = (LANE_DAILY_PACKAGE, LANE_ON_DEMAND)

# ── Artifact kinds ────────────────────────────────────────────────────────

KIND_PM_INSIGHT = "pm_insight"
KIND_AI_PM_POST = "ai_pm_post"
KIND_USER_MATERIAL = "user_material"

KINDS = (KIND_PM_INSIGHT, KIND_AI_PM_POST, KIND_USER_MATERIAL)

# Required argument-pack fields. Every artifact must carry all four, non-empty.
REQUIRED_PACK_FIELDS = ("claim", "evidence", "mechanism", "position")

# Daily package cardinality is a hard, deterministic contract.
DAILY_INSIGHT_COUNT = 1
DAILY_POST_COUNT = 2

# Dedicated approval channel. Deliberately injectable at formatting time so
# tests and future Discord adapters cannot silently route to #content or to the
# X manager channel. The concrete value is filled in at cron/channel wiring
# time - the contract only guarantees a stable, dedicated default.
LI_MANAGER_CHANNEL_ID = "1539633360664657920"

# Approval status vocabulary. There is no "published" state reachable from
# this module - publishing is a separate, human-driven step outside the
# manager's surface.
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
_STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)


class LiManagerError(Exception):
    """Raised when an artifact violates the contract (incomplete pack, bad lane,
    bad kind, or a daily package with the wrong cardinality)."""


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class ArgumentPack:
    """Evidence/argument pack required on every LinkedIn manager artifact.

    ``claim``      - the assertion the post makes.
    ``evidence``   - the supporting evidence (source, data, quote, link).
    ``mechanism``  - how/why it holds (the causal or explanatory step).
    ``position``   - the stance/angle Sahil takes on it.
    ``context``    - lane-appropriate context (source material ref, or the
                     signals/source that grounded the post).
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
class LiArtifact:
    """An approval-only LinkedIn manager artifact.

    ``kind`` distinguishes the PM insight summary from an AI/PM post from an
    on-demand user-material post. ``source_url`` is required (non-empty) for
    the PM insight - it is the direct link to the full post - and optional but
    encouraged for posts (the original source/evidence).

    ``status`` starts at ``pending`` and can only move to ``approved`` or
    ``rejected`` via an explicit decision. There is no auto-publish.
    """

    id: str
    lane: str
    kind: str
    body: str
    pack: ArgumentPack
    brand: str = "sahil_linkedin"
    source_url: str = ""
    status: str = STATUS_PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DailyPackage:
    """A validated daily package: exactly one PM insight + exactly two posts.

    ``build_daily_package`` is the only way to construct one, so the
    1-insight + 2-post invariant is enforced at the type boundary.
    """

    insight: LiArtifact
    posts: list[LiArtifact]


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
CREATE TABLE IF NOT EXISTS li_manager_artifacts (
    id          TEXT PRIMARY KEY,
    lane        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    brand       TEXT NOT NULL,
    body        TEXT NOT NULL,
    source_url  TEXT NOT NULL DEFAULT '',
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

CREATE INDEX IF NOT EXISTS idx_li_manager_status ON li_manager_artifacts(status);
CREATE INDEX IF NOT EXISTS idx_li_manager_lane ON li_manager_artifacts(lane);
CREATE INDEX IF NOT EXISTS idx_li_manager_kind ON li_manager_artifacts(kind);
"""


def init_db() -> None:
    """Idempotently create the LinkedIn manager artifact table.

    This table is independent of the X manager's artifact table and of the
    legacy ``drafts`` table, so LinkedIn state can never couple to X state and
    the old rejected personal-brand backlog can never leak in.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _validate_artifact(artifact: LiArtifact) -> None:
    """Fail closed: reject any artifact that violates the contract."""
    if artifact.lane not in LANES:
        raise LiManagerError(f"unknown lane: {artifact.lane!r}")
    if artifact.kind not in KINDS:
        raise LiManagerError(f"unknown kind: {artifact.kind!r}")
    if artifact.status not in _STATUSES:
        raise LiManagerError(f"unknown status: {artifact.status!r}")
    if artifact.brand != "sahil_linkedin":
        raise LiManagerError(f"unexpected brand: {artifact.brand!r}")
    missing = artifact.pack.missing_fields()
    if missing:
        raise LiManagerError(
            f"incomplete argument pack for {artifact.id}: missing {missing}"
        )
    if not (artifact.body or "").strip():
        raise LiManagerError(f"artifact {artifact.id} has empty body")
    if artifact.kind == KIND_PM_INSIGHT and not (artifact.source_url or "").strip():
        raise LiManagerError(
            f"PM insight {artifact.id} is missing its direct link to the full post"
        )
    if artifact.source_url:
        parsed = urlparse(artifact.source_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise LiManagerError(f"unsafe source URL for {artifact.id}")
    words = len(artifact.body.split())
    if artifact.kind == KIND_PM_INSIGHT and not (20 <= words <= 120):
        raise LiManagerError("PM insight summary must be 20-120 words")
    if artifact.kind == KIND_AI_PM_POST and not (80 <= words <= 250):
        raise LiManagerError("AI/PM post must be 80-250 words")


def stage_for_approval(artifact: LiArtifact) -> str:
    """Persist an artifact as ``pending``. Returns the artifact id.

    This is the only write path in the manager, and it always writes
    ``status='pending'`` - there is no code path that publishes. Raises
    ``LiManagerError`` (and writes nothing) if the artifact is incomplete.
    """
    _validate_artifact(artifact)
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            INSERT INTO li_manager_artifacts
              (id, lane, kind, brand, body, source_url, claim, evidence,
               mechanism, position, context, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.lane,
                artifact.kind,
                artifact.brand,
                artifact.body,
                artifact.source_url or "",
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
    artifact: LiArtifact,
    *,
    channel_id: str = LI_MANAGER_CHANNEL_ID,
) -> DeliveryCard:
    """Return a paste-ready pending-approval card without sending it anywhere."""
    _validate_artifact(artifact)
    source_line = (
        f"**Full post:** {artifact.source_url}\n" if artifact.source_url else ""
    )
    return DeliveryCard(
        artifact_id=artifact.id,
        channel_id=channel_id,
        body=(
            f"**LinkedIn manager · {artifact.kind} · PENDING APPROVAL**\n"
            f"ID: `{artifact.id}`\n\n"
            f"{artifact.body}\n\n"
            f"{source_line}"
            f"**Claim:** {artifact.pack.claim}\n"
            f"**Evidence:** {artifact.pack.evidence}\n"
            f"**Mechanism:** {artifact.pack.mechanism}\n"
            f"**Position:** {artifact.pack.position}\n\n"
            "Approve or reject explicitly. This manager cannot publish."
        ),
    )


def stage_and_format_card(
    artifact: LiArtifact,
    *,
    channel_id: str = LI_MANAGER_CHANNEL_ID,
) -> DeliveryCard:
    """Stage an artifact then return its pending approval card, with no network I/O."""
    stage_for_approval(artifact)
    return format_approval_card(artifact, channel_id=channel_id)


def decide(artifact_id: str, action: str, decided_by: str = "") -> bool:
    """Apply an explicit approval decision. ``action`` in approve|reject.

    Returns True iff the row was found and updated. There is no "publish"
    action - approval only flips the row to ``approved``; the actual post is
    a separate human step outside this contract.
    """
    if action not in (STATUS_APPROVED, STATUS_REJECTED):
        raise LiManagerError(f"unknown decision action: {action!r}")
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.execute(
            """
            UPDATE li_manager_artifacts
               SET status = ?, decided_at = ?, decided_by = ?
             WHERE id = ?
            """,
            (action, datetime.now(timezone.utc).isoformat(), decided_by, artifact_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_artifacts(status: Optional[str] = None, lane: Optional[str] = None,
                   kind: Optional[str] = None) -> list[dict]:
    """List artifacts, optionally filtered by status, lane and/or kind."""
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
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM li_manager_artifacts{where} ORDER BY created_at ASC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Lane builders ──────────────────────────────────────────────────────────


def _new_id(prefix: str) -> str:
    return f"li_{prefix[:8]}_{uuid.uuid4().hex[:10]}"


def _make_artifact(
    *,
    lane: str,
    kind: str,
    body: str,
    pack: ArgumentPack,
    brand: str = "sahil_linkedin",
    source_url: str = "",
) -> LiArtifact:
    artifact = LiArtifact(
        id=_new_id(kind),
        lane=lane,
        kind=kind,
        body=body,
        pack=pack,
        brand=brand,
        source_url=source_url,
    )
    _validate_artifact(artifact)
    return artifact


def build_daily_package(
    insight: LiArtifact,
    posts: list[LiArtifact],
) -> DailyPackage:
    """Assemble and validate a daily package: exactly 1 insight + 2 posts.

    This is the deterministic seam. It fails closed if the insight is not a
    PM insight (with a direct link to the full post), if there are not exactly
    two AI/PM posts, or if any artifact violates the contract. Callers cannot
    accidentally ship an empty or padded package.
    """
    _validate_artifact(insight)
    if insight.lane != LANE_DAILY_PACKAGE:
        raise LiManagerError(f"insight {insight.id} is not on the daily package lane")
    if insight.kind != KIND_PM_INSIGHT:
        raise LiManagerError(f"daily package insight must be kind {KIND_PM_INSIGHT!r}")

    if not isinstance(posts, list) or len(posts) != DAILY_POST_COUNT:
        raise LiManagerError(
            f"daily package requires exactly {DAILY_POST_COUNT} posts, got "
            f"{len(posts) if isinstance(posts, list) else 'non-list'}"
        )
    for post in posts:
        _validate_artifact(post)
        if post.lane != LANE_DAILY_PACKAGE:
            raise LiManagerError(f"post {post.id} is not on the daily package lane")
        if post.kind != KIND_AI_PM_POST:
            raise LiManagerError(f"daily package posts must be kind {KIND_AI_PM_POST!r}")

    ids = {insight.id, *(p.id for p in posts)}
    if len(ids) != 1 + len(posts):
        raise LiManagerError("daily package contains duplicate artifact ids")

    def fingerprint(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    bodies = {fingerprint(p.body) for p in posts}
    claims = {fingerprint(p.pack.claim) for p in posts}
    evidence = {fingerprint(p.pack.evidence) for p in posts}
    if len(bodies) != len(posts) or len(claims) != len(posts):
        raise LiManagerError("daily package posts are not independent")
    if len(evidence) != len(posts):
        raise LiManagerError("daily package posts must use independent evidence")

    return DailyPackage(insight=insight, posts=list(posts))


def stage_daily_package(
    insight: LiArtifact,
    posts: list[LiArtifact],
    *,
    channel_id: str = LI_MANAGER_CHANNEL_ID,
) -> list[DeliveryCard]:
    """Validate the 1+2 package, stage every artifact as pending, and return
    one approval card per artifact (insight first, then posts)."""
    package = build_daily_package(insight, posts)
    ordered = [package.insight, *package.posts]
    return [stage_and_format_card(art, channel_id=channel_id) for art in ordered]


def transform_user_material(
    material: dict,
    *,
    pack: Optional[ArgumentPack] = None,
    source_url: str = "",
    kind: str = KIND_USER_MATERIAL,
) -> LiArtifact:
    """On-demand lane: transform a piece of user-shared material into a draft.

    ``material`` must carry the source text (``text``) and, optionally, a
    ``source`` label. The caller supplies the argument pack; if omitted, a
    pack is derived from the material but still validated (fail closed).
    """
    text = (material.get("text") or "").strip()
    source = material.get("source") or "user-material"
    if not text:
        raise LiManagerError("user material has no text")

    if pack is None:
        pack = ArgumentPack(
            claim=text,
            evidence=source,
            mechanism="",
            position="",
            context={"source": source},
        )

    return _make_artifact(
        lane=LANE_ON_DEMAND,
        kind=kind,
        body=text,
        pack=pack,
        source_url=source_url or "",
    )
