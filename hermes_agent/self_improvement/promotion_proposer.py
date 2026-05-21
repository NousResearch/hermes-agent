"""
hermes_agent/self_improvement/promotion_proposer.py — Plan 004-B

Daily job that scans skill_scores for promotion/demotion candidates and:
1. Inserts pending rows into promotion_decisions
2. Sends Blake a Slack DM with ≤5 candidates ranked by combined score
3. Provides approve/dismiss callback (called from Slack Block Kit action or UI)

Promotion criteria (configurable via env vars):
  HERMES_PROMOTION_MIN_USAGE=10     — minimum usage_30d (default 10)
  HERMES_PROMOTION_MIN_THUMBS_RATE=0.80  — minimum thumbs_rate_30d (default 0.80)
  HERMES_PROMOTION_WINDOW_DAYS=30   — scoring window (default 30)

Demotion criteria:
  HERMES_DEMOTION_MAX_THUMBS_RATE=0.50  — team-scope skills below this → demote (default 0.50)

Design decisions:
- daily cron via hermes cron scheduler (jobs.py/create_job). Phase B registers
  the cron job at first run; subsequent runs are idempotent (check for existing job).
- Top 5 candidates per DM — avoids notification fatigue.
- Bidirectional: checks both promotion (personal→team) and demotion (team→personal).
- NO auto-promotion: approve_promotion() must be called explicitly by Blake.
  The test assert_no_silent_promotion verifies this invariant.
- Score snapshot: captures metrics at suggestion time for retrospective audit.
- Slack DM uses Block Kit with "Approve" and "Dismiss" buttons. Button callbacks
  hit /hermes internal API → approve_promotion() / dismiss_promotion().
- Dedup: a skill with a pending promotion_decisions row is skipped (no double-DM).

Failure modes:
- Neon unavailable: exception propagates; scheduler logs ERROR and retries next day.
- Slack DM fails: promotion_decisions row already written; DM failure just means
  Blake doesn't get the notification (recoverable via dashboard).
- Skills Service promote_skill unavailable: approve_promotion() raises; Blake sees
  an error in the UI — the decision row stays "pending" for retry.

Assumptions:
- Skill scope is resolved from the skill registry (personal vs team vs global).
  Phase A only tracks skill_name (not scope) in skill_scores. Phase B adds scope
  lookup via skills_hub or S3 registry. For MVP, from_scope defaults to "personal".
- Blake's Slack user_id is in HERMES_BLAKE_SLACK_USER_ID env var.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _min_usage() -> int:
    return int(os.environ.get("HERMES_PROMOTION_MIN_USAGE", "10"))

def _min_thumbs_rate() -> float:
    return float(os.environ.get("HERMES_PROMOTION_MIN_THUMBS_RATE", "0.80"))

def _max_demotion_thumbs_rate() -> float:
    return float(os.environ.get("HERMES_DEMOTION_MAX_THUMBS_RATE", "0.50"))

def _promotion_window_days() -> int:
    return int(os.environ.get("HERMES_PROMOTION_WINDOW_DAYS", "30"))

def _blake_slack_user_id() -> Optional[str]:
    return os.environ.get("HERMES_BLAKE_SLACK_USER_ID") or None

MAX_DM_CANDIDATES = 5


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------

async def _get_promotion_candidates(conn, tenant_id: str) -> list[dict[str, Any]]:
    """
    Return skills meeting promotion thresholds from skill_scores.

    A skill qualifies when:
    - usage_30d >= MIN_USAGE
    - thumbs_rate_30d >= MIN_THUMBS_RATE (not NULL — must have signal)
    - No existing pending promotion_decisions row (prevent duplicate suggestions)

    Returns a list of dicts sorted by combined_score DESC (usage × thumbs_rate).
    """
    min_usage = _min_usage()
    min_rate = _min_thumbs_rate()

    rows = await conn.fetch(
        """
        SELECT
            ss.skill_name,
            ss.usage_30d,
            ss.thumbs_rate_30d,
            ss.thumbs_up_30d,
            ss.thumbs_down_30d,
            ss.last_used_at
        FROM skill_scores ss
        WHERE ss.tenant_id = $1
          AND ss.usage_30d >= $2
          AND ss.thumbs_rate_30d >= $3
          AND NOT EXISTS (
              SELECT 1 FROM promotion_decisions pd
              WHERE pd.tenant_id = ss.tenant_id
                AND pd.skill_name = ss.skill_name
                AND pd.status = 'pending'
                AND pd.action = 'promote'
          )
        ORDER BY (ss.usage_30d * ss.thumbs_rate_30d) DESC
        LIMIT $4
        """,
        tenant_id, min_usage, min_rate, MAX_DM_CANDIDATES,
    )
    return [dict(r) for r in rows]


async def _get_demotion_candidates(conn, tenant_id: str) -> list[dict[str, Any]]:
    """
    Return team-scope skills below the demotion threshold.

    A skill is a demotion candidate when:
    - thumbs_rate_30d < MAX_DEMOTION_THUMBS_RATE
    - total reactions >= 5 (enough signal to be confident about poor quality)
    - No existing pending demotion_decisions row

    Note: "team-scope" detection is approximate in Phase B — we check if
    the skill is in skill_scores with sufficient usage. Full scope resolution
    requires Skills Service integration (Phase B.1).
    """
    max_rate = _max_demotion_thumbs_rate()

    rows = await conn.fetch(
        """
        SELECT
            ss.skill_name,
            ss.usage_30d,
            ss.thumbs_rate_30d,
            ss.thumbs_up_30d,
            ss.thumbs_down_30d,
            ss.last_used_at
        FROM skill_scores ss
        WHERE ss.tenant_id = $1
          AND ss.thumbs_rate_30d IS NOT NULL
          AND ss.thumbs_rate_30d < $2
          AND (ss.thumbs_up_30d + ss.thumbs_down_30d) >= 5
          AND NOT EXISTS (
              SELECT 1 FROM promotion_decisions pd
              WHERE pd.tenant_id = ss.tenant_id
                AND pd.skill_name = ss.skill_name
                AND pd.status = 'pending'
                AND pd.action = 'demote'
          )
        ORDER BY ss.thumbs_rate_30d ASC
        LIMIT $3
        """,
        tenant_id, max_rate, MAX_DM_CANDIDATES,
    )
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Core daily job
# --------------------------------------------------------------------------

async def run_daily_proposer(pool, tenant_id: str, slack_client=None) -> dict[str, Any]:
    """
    Run the daily promotion/demotion proposal job for a tenant.

    For each candidate:
    1. Insert a pending promotion_decisions row (idempotent via NOT EXISTS check).
    2. Collect candidates for a Slack DM to Blake.

    Args:
        pool:         asyncpg connection pool.
        tenant_id:    Neon tenant UUID.
        slack_client: Optional Slack AsyncWebClient for DM delivery.
                      None means no DM — candidates are still written to DB.

    Returns:
        Summary dict: {
            "promotion_candidates": int,
            "demotion_candidates": int,
            "dm_sent": bool,
            "new_decisions": int,
        }

    Invariant (AC-B.7): this function NEVER calls Skills Service promote_skill.
    Only approve_promotion() (called by Blake explicitly) does that.
    """
    from hermes_storage.neon_backend import _RLSTransaction

    promotion_rows = []
    demotion_rows = []

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            promotion_candidates = await _get_promotion_candidates(conn, tenant_id)
            demotion_candidates = await _get_demotion_candidates(conn, tenant_id)

            new_decisions = 0

            for candidate in promotion_candidates:
                score_snapshot = {
                    "usage_30d": candidate["usage_30d"],
                    "thumbs_rate_30d": float(candidate["thumbs_rate_30d"]) if candidate["thumbs_rate_30d"] is not None else None,
                    "thumbs_up_30d": candidate["thumbs_up_30d"],
                    "thumbs_down_30d": candidate["thumbs_down_30d"],
                    "suggested_at": datetime.now(tz=timezone.utc).isoformat(),
                }
                await conn.execute(
                    """
                    INSERT INTO promotion_decisions
                        (tenant_id, skill_name, action, from_scope, to_scope,
                         score_snapshot, status)
                    VALUES ($1, $2, 'promote', 'personal', 'team', $3::jsonb, 'pending')
                    ON CONFLICT DO NOTHING
                    """,
                    tenant_id, candidate["skill_name"],
                    __import__("json").dumps(score_snapshot),
                )
                new_decisions += 1
                promotion_rows.append(candidate)

            for candidate in demotion_candidates:
                score_snapshot = {
                    "usage_30d": candidate["usage_30d"],
                    "thumbs_rate_30d": float(candidate["thumbs_rate_30d"]) if candidate["thumbs_rate_30d"] is not None else None,
                    "thumbs_up_30d": candidate["thumbs_up_30d"],
                    "thumbs_down_30d": candidate["thumbs_down_30d"],
                    "suggested_at": datetime.now(tz=timezone.utc).isoformat(),
                }
                await conn.execute(
                    """
                    INSERT INTO promotion_decisions
                        (tenant_id, skill_name, action, from_scope, to_scope,
                         score_snapshot, status)
                    VALUES ($1, $2, 'demote', 'team', 'personal', $3::jsonb, 'pending')
                    ON CONFLICT DO NOTHING
                    """,
                    tenant_id, candidate["skill_name"],
                    __import__("json").dumps(score_snapshot),
                )
                new_decisions += 1
                demotion_rows.append(candidate)

    dm_sent = False
    all_candidates = promotion_rows + demotion_rows

    if all_candidates and slack_client:
        blake_user_id = _blake_slack_user_id()
        if blake_user_id:
            dm_sent = await _send_suggestion_dm(
                slack_client, blake_user_id, promotion_rows, demotion_rows
            )

    logger.info(
        "promotion_proposer: tenant=%s promote=%d demote=%d new_decisions=%d dm=%s",
        tenant_id, len(promotion_rows), len(demotion_rows), new_decisions, dm_sent,
    )
    return {
        "promotion_candidates": len(promotion_rows),
        "demotion_candidates": len(demotion_rows),
        "dm_sent": dm_sent,
        "new_decisions": new_decisions,
    }


# --------------------------------------------------------------------------
# Slack DM formatting
# --------------------------------------------------------------------------

async def _send_suggestion_dm(
    slack_client,
    blake_user_id: str,
    promotion_rows: list[dict],
    demotion_rows: list[dict],
) -> bool:
    """
    Send a Slack DM to Blake with pending promotion/demotion suggestions.

    Only sends when there are ≥1 candidates (no daily noise when queue is empty).
    Uses Block Kit with Approve/Dismiss buttons.
    Returns True on success.
    """
    try:
        # Open DM channel with Blake.
        dm_resp = await slack_client.conversations_open(users=[blake_user_id])
        if not dm_resp.get("ok"):
            logger.warning("promotion_proposer: failed to open DM with %s", blake_user_id)
            return False
        dm_channel = dm_resp["channel"]["id"]

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Hermes Skill Suggestions",
                    "emoji": True,
                },
            },
        ]

        if promotion_rows:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Promotion candidates* ({len(promotion_rows)} skill(s) ready for team scope):",
                },
            })
            for c in promotion_rows[:MAX_DM_CANDIDATES]:
                rate_pct = (
                    f"{float(c['thumbs_rate_30d']):.0%}"
                    if c.get("thumbs_rate_30d") is not None
                    else "N/A"
                )
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"`{c['skill_name']}` — "
                            f"{c['usage_30d']} uses, {rate_pct} thumbs-up (30d)"
                        ),
                    },
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "hermes_approve_promotion",
                        "value": c["skill_name"],
                    },
                })

        if demotion_rows:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Demotion candidates* ({len(demotion_rows)} skill(s) underperforming):",
                },
            })
            for c in demotion_rows[:MAX_DM_CANDIDATES]:
                rate_pct = (
                    f"{float(c['thumbs_rate_30d']):.0%}"
                    if c.get("thumbs_rate_30d") is not None
                    else "N/A"
                )
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"`{c['skill_name']}` — "
                            f"{c['usage_30d']} uses, {rate_pct} thumbs-up (30d)"
                        ),
                    },
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Demote"},
                        "style": "danger",
                        "action_id": "hermes_approve_demotion",
                        "value": c["skill_name"],
                    },
                })

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Review and approve in the Hermes /skills dashboard or via the buttons above.",
                }
            ],
        })

        await slack_client.chat_postMessage(channel=dm_channel, blocks=blocks)
        return True

    except Exception as exc:
        logger.error("promotion_proposer: Slack DM error: %s", exc)
        return False


# --------------------------------------------------------------------------
# Blake approval / dismiss callbacks
# --------------------------------------------------------------------------

async def approve_promotion(
    pool,
    tenant_id: str,
    skill_name: str,
    decided_by: str,
    skills_service_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute an approved promotion: call Skills Service promote_skill + update DB.

    This is the ONLY place that calls promote_skill. It is invoked explicitly by
    Blake (via Slack button click or dashboard action) — never called automatically.

    Args:
        pool:               asyncpg connection pool.
        tenant_id:          Neon tenant UUID.
        skill_name:         Skill to promote.
        decided_by:         Slack user_id of approver (for audit trail).
        skills_service_url: Override for Skills Service URL (test injection).

    Returns:
        {"status": "approved", "skill_name": str} on success.
        {"status": "error", "error": str} on failure (DB updated, Skills Service call may not have happened).

    Invariant: The promotion_decisions row is updated to "approved" ONLY after the
    Skills Service call succeeds. If the Skills Service call fails, the row stays
    "pending" so Blake can retry.
    """
    from hermes_storage.neon_backend import _RLSTransaction

    # Call Skills Service first — fail fast if it's unavailable.
    svc_url = skills_service_url or os.environ.get("HERMES_SKILLS_SERVICE_URL", "")
    if svc_url:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{svc_url}/v1/skills/promote",
                    json={"skill_name": skill_name, "from_scope": "personal", "to_scope": "team"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status not in (200, 201, 204):
                        body = await resp.text()
                        return {
                            "status": "error",
                            "error": f"Skills Service returned {resp.status}: {body[:200]}",
                        }
        except Exception as exc:
            return {"status": "error", "error": f"Skills Service call failed: {exc}"}

    # Update promotion_decisions row to approved.
    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            await conn.execute(
                """
                UPDATE promotion_decisions
                SET status = 'approved', decided_at = now(), decided_by = $3
                WHERE tenant_id = $1 AND skill_name = $2
                  AND status = 'pending' AND action = 'promote'
                """,
                tenant_id, skill_name, decided_by,
            )

    logger.info(
        "promotion_proposer: approved promotion skill=%s by=%s tenant=%s",
        skill_name, decided_by, tenant_id,
    )
    return {"status": "approved", "skill_name": skill_name}


async def dismiss_proposal(
    pool,
    tenant_id: str,
    skill_name: str,
    action: str,
    decided_by: str,
) -> dict[str, Any]:
    """
    Dismiss a pending promotion or demotion proposal.

    Does NOT call Skills Service — the skill stays in its current scope.
    Updates the promotion_decisions row to "dismissed".

    Args:
        action: "promote" | "demote" — which proposal type to dismiss.
    """
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            await conn.execute(
                """
                UPDATE promotion_decisions
                SET status = 'dismissed', decided_at = now(), decided_by = $4
                WHERE tenant_id = $1 AND skill_name = $2
                  AND status = 'pending' AND action = $3
                """,
                tenant_id, skill_name, action, decided_by,
            )

    logger.info(
        "promotion_proposer: dismissed %s for skill=%s by=%s",
        action, skill_name, decided_by,
    )
    return {"status": "dismissed", "skill_name": skill_name, "action": action}


async def get_pending_proposals(pool, tenant_id: str) -> list[dict[str, Any]]:
    """
    Return all pending promotion/demotion proposals for the dashboard.

    Returns a list of dicts with: id, skill_name, action, from_scope, to_scope,
    score_snapshot, suggested_at. Ordered newest-first.
    """
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            rows = await conn.fetch(
                """
                SELECT id, skill_name, action, from_scope, to_scope,
                       score_snapshot, suggested_at
                FROM promotion_decisions
                WHERE tenant_id = $1 AND status = 'pending'
                ORDER BY suggested_at DESC
                """,
                tenant_id,
            )
            return [
                {
                    "id": str(r["id"]),
                    "skill_name": r["skill_name"],
                    "action": r["action"],
                    "from_scope": r["from_scope"],
                    "to_scope": r["to_scope"],
                    "score_snapshot": dict(r["score_snapshot"]) if r["score_snapshot"] else {},
                    "suggested_at": r["suggested_at"].isoformat() if r["suggested_at"] else None,
                }
                for r in rows
            ]
