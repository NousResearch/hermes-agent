"""
hermes_agent/self_improvement/drift_detector.py — Plan 004-C

Daily regression detector that watches per-skill thumbs_rate_30d and alerts
when a previously-good skill drops below the regression threshold.

Alert criteria (all configurable via env vars):
  HERMES_DRIFT_BASELINE_RATE=0.80     — "was good at" threshold (default 0.80)
  HERMES_DRIFT_ALERT_THRESHOLD=0.50   — "dropped to below" threshold (default 0.50)
  HERMES_DRIFT_WINDOW_DAYS=14         — detection window (default 14 days)
  HERMES_DRIFT_MIN_SIGNAL=5           — minimum reactions in window (default 5)
  HERMES_DRIFT_RESOLUTION_RATE=0.65   — auto-resolve when rate climbs above this

Algorithm:
1. For each skill in skill_scores:
   - If total_30d_reactions >= MIN_SIGNAL AND thumbs_rate_30d < ALERT_THRESHOLD:
     - Check if a prior "good" baseline exists (previous scores from skill_scores
       or a synthetic baseline of BASELINE_RATE)
     - If no existing 'alerted' row: INSERT into skill_drift_alerts (dedup)
     - If existing 'alerted' row: skip (already alerted, no re-notify)
   - If an 'alerted' row exists AND thumbs_rate_30d >= RESOLUTION_RATE:
     - Mark resolved → send "resolved" DM

Design decisions:
- "Was good" baseline is simplified in Phase C: we use BASELINE_RATE as the
  implicit prior (a skill was assumed good before it started getting reactions).
  Phase D could extend this to track historical rate snapshots for more precise
  "was at 87%, dropped to 42%" reporting.
- Dedup is DB-enforced via UNIQUE (tenant_id, skill_name) on skill_drift_alerts.
  INSERT ... ON CONFLICT DO NOTHING prevents duplicate alerts.
- Slack DM only on FIRST detection. The dedup prevents re-DM on subsequent runs.
- Auto-resolution checks run every day alongside alert detection.

Failure modes:
- Pool unavailable: exception propagates to scheduler (logged as ERROR).
- Slack DM failure: alert row already written; Blake sees it in dashboard.
- skill_scores not yet populated (Phase A not yet run): no rows → no alerts.

Assumptions:
- skill_scores is populated by skill_scorer.run_for_all_tenants() (Phase A).
  If the scorer hasn't run recently, alert detection has stale data.
- A skill with NULL thumbs_rate (insufficient signal) is skipped — can't detect
  drift without signal.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _baseline_rate() -> float:
    return float(os.environ.get("HERMES_DRIFT_BASELINE_RATE", "0.80"))

def _alert_threshold() -> float:
    return float(os.environ.get("HERMES_DRIFT_ALERT_THRESHOLD", "0.50"))

def _resolution_rate() -> float:
    return float(os.environ.get("HERMES_DRIFT_RESOLUTION_RATE", "0.65"))

def _min_signal() -> int:
    return int(os.environ.get("HERMES_DRIFT_MIN_SIGNAL", "5"))

def _blake_slack_user_id() -> Optional[str]:
    return os.environ.get("HERMES_BLAKE_SLACK_USER_ID") or None


# --------------------------------------------------------------------------
# Core daily detection job
# --------------------------------------------------------------------------

async def run_drift_detection(
    pool,
    tenant_id: str,
    slack_client=None,
) -> dict[str, Any]:
    """
    Run drift detection for a single tenant.

    Two passes:
    1. Auto-resolution: alerted skills that have recovered → mark resolved.
    2. New alerts: skills below threshold with enough signal → insert alert row.

    Args:
        pool:         asyncpg connection pool.
        tenant_id:    Neon tenant UUID.
        slack_client: Optional Slack AsyncWebClient. If provided, sends DMs.

    Returns:
        Summary dict: {
            "new_alerts": int,
            "auto_resolved": int,
            "total_skills_checked": int,
        }
    """
    from hermes_storage.neon_backend import _RLSTransaction

    alert_threshold = _alert_threshold()
    resolution_rate = _resolution_rate()
    min_signal = _min_signal()

    new_alerts = 0
    auto_resolved = 0
    total_checked = 0

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            # --- Pass 1: Auto-resolution ---
            # Find skills with active alerts that have recovered.
            alerted_skills = await conn.fetch(
                """
                SELECT sda.id, sda.skill_name, ss.thumbs_rate_30d,
                       (ss.thumbs_up_30d + ss.thumbs_down_30d) AS total_reactions
                FROM skill_drift_alerts sda
                JOIN skill_scores ss
                  ON ss.tenant_id = sda.tenant_id AND ss.skill_name = sda.skill_name
                WHERE sda.tenant_id = $1 AND sda.status = 'alerted'
                """,
                tenant_id,
            )

            for row in alerted_skills:
                rate = float(row["thumbs_rate_30d"]) if row["thumbs_rate_30d"] is not None else None
                if rate is not None and rate >= resolution_rate:
                    # Auto-resolve.
                    await conn.execute(
                        """
                        UPDATE skill_drift_alerts
                        SET status = 'resolved',
                            resolved_at = now(),
                            resolved_rate = $2,
                            actioned_by = 'system'
                        WHERE id = $1
                        """,
                        row["id"], rate,
                    )
                    auto_resolved += 1
                    logger.info(
                        "drift_detector: auto-resolved skill=%s rate=%.2f tenant=%s",
                        row["skill_name"], rate, tenant_id,
                    )
                    if slack_client:
                        await _send_resolved_dm(
                            slack_client, row["skill_name"], rate
                        )

            # --- Pass 2: New alert detection ---
            # Find skills below threshold with sufficient signal and NO active alert.
            at_risk = await conn.fetch(
                """
                SELECT
                    ss.skill_name,
                    ss.thumbs_rate_30d,
                    ss.thumbs_up_30d,
                    ss.thumbs_down_30d,
                    (ss.thumbs_up_30d + ss.thumbs_down_30d) AS total_reactions
                FROM skill_scores ss
                WHERE ss.tenant_id = $1
                  AND ss.thumbs_rate_30d IS NOT NULL
                  AND ss.thumbs_rate_30d < $2
                  AND (ss.thumbs_up_30d + ss.thumbs_down_30d) >= $3
                  AND NOT EXISTS (
                      SELECT 1 FROM skill_drift_alerts sda
                      WHERE sda.tenant_id = ss.tenant_id
                        AND sda.skill_name = ss.skill_name
                        AND sda.status = 'alerted'
                  )
                """,
                tenant_id, alert_threshold, min_signal,
            )

            total_checked = len(alerted_skills) + len(at_risk)

            for row in at_risk:
                rate = float(row["thumbs_rate_30d"])
                baseline = _baseline_rate()

                # Insert new alert row. ON CONFLICT DO NOTHING handles the edge case
                # where a row was inserted between our NOT EXISTS check and this INSERT
                # (concurrent runs are unlikely but possible in multi-instance setups).
                await conn.execute(
                    """
                    INSERT INTO skill_drift_alerts
                        (tenant_id, skill_name, status, baseline_rate, alert_rate)
                    VALUES ($1, $2, 'alerted', $3, $4)
                    ON CONFLICT (tenant_id, skill_name) DO NOTHING
                    """,
                    tenant_id, row["skill_name"], baseline, rate,
                )
                new_alerts += 1
                logger.warning(
                    "drift_detector: NEW ALERT skill=%s rate=%.2f (below threshold %.2f) tenant=%s",
                    row["skill_name"], rate, alert_threshold, tenant_id,
                )

                if slack_client:
                    await _send_alert_dm(
                        slack_client, row["skill_name"], rate, baseline
                    )

    logger.info(
        "drift_detector: tenant=%s new_alerts=%d auto_resolved=%d checked=%d",
        tenant_id, new_alerts, auto_resolved, total_checked,
    )
    return {
        "new_alerts": new_alerts,
        "auto_resolved": auto_resolved,
        "total_skills_checked": total_checked,
    }


async def run_for_all_tenants(pool, slack_client=None) -> list[dict[str, Any]]:
    """Run drift detection for every tenant. Entry point for the daily cron job."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id FROM tenants")
        tenant_ids = [str(r["id"]) for r in rows]

    results = []
    for tid in tenant_ids:
        try:
            summary = await run_drift_detection(pool, tid, slack_client=slack_client)
            summary["tenant_id"] = tid
            results.append(summary)
        except Exception as exc:
            logger.error("drift_detector: tenant=%s failed: %s", tid, exc)
            results.append({"tenant_id": tid, "error": str(exc)})
    return results


# --------------------------------------------------------------------------
# Manual actions
# --------------------------------------------------------------------------

async def dismiss_alert(
    pool,
    tenant_id: str,
    skill_name: str,
    actioned_by: str,
) -> dict[str, Any]:
    """
    Mark an active drift alert as 'dismissed' (acknowledged, not fixing now).

    Does not affect the skill itself — it stays in its current scope. The alert
    will NOT re-trigger unless the skill is first resolved (manually or by recovery).
    """
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            await conn.execute(
                """
                UPDATE skill_drift_alerts
                SET status = 'dismissed', resolved_at = now(), actioned_by = $3
                WHERE tenant_id = $1 AND skill_name = $2 AND status = 'alerted'
                """,
                tenant_id, skill_name, actioned_by,
            )
    logger.info(
        "drift_detector: dismissed alert skill=%s by=%s", skill_name, actioned_by
    )
    return {"status": "dismissed", "skill_name": skill_name}


async def mark_iterate(
    pool,
    tenant_id: str,
    skill_name: str,
    actioned_by: str,
) -> dict[str, Any]:
    """
    Mark an active drift alert as 'iterate' (skill author should improve it).

    Like dismiss but signals intent to improve the skill, not just ignore the alert.
    """
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            await conn.execute(
                """
                UPDATE skill_drift_alerts
                SET status = 'iterate', resolved_at = now(), actioned_by = $3
                WHERE tenant_id = $1 AND skill_name = $2 AND status = 'alerted'
                """,
                tenant_id, skill_name, actioned_by,
            )
    logger.info(
        "drift_detector: marked iterate skill=%s by=%s", skill_name, actioned_by
    )
    return {"status": "iterate", "skill_name": skill_name}


async def get_active_alerts(pool, tenant_id: str) -> list[dict[str, Any]]:
    """Return all 'alerted' drift alerts for the dashboard."""
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            rows = await conn.fetch(
                """
                SELECT id, skill_name, status, baseline_rate, alert_rate,
                       alerted_at, notes
                FROM skill_drift_alerts
                WHERE tenant_id = $1 AND status = 'alerted'
                ORDER BY alerted_at DESC
                """,
                tenant_id,
            )
            return [
                {
                    "id": str(r["id"]),
                    "skill_name": r["skill_name"],
                    "status": r["status"],
                    "baseline_rate": float(r["baseline_rate"]) if r["baseline_rate"] is not None else None,
                    "alert_rate": float(r["alert_rate"]) if r["alert_rate"] is not None else None,
                    "alerted_at": r["alerted_at"].isoformat() if r["alerted_at"] else None,
                    "notes": r["notes"],
                }
                for r in rows
            ]


# --------------------------------------------------------------------------
# Slack DM helpers
# --------------------------------------------------------------------------

async def _send_alert_dm(
    slack_client,
    skill_name: str,
    current_rate: float,
    baseline_rate: float,
) -> bool:
    """Send a first-detection drift alert DM to Blake."""
    blake_user_id = _blake_slack_user_id()
    if not blake_user_id:
        return False
    try:
        dm_resp = await slack_client.conversations_open(users=[blake_user_id])
        if not dm_resp.get("ok"):
            return False
        dm_channel = dm_resp["channel"]["id"]

        text = (
            f":warning: *Skill regression detected*: `{skill_name}`\n"
            f"Thumbs rate dropped from ~{baseline_rate:.0%} → {current_rate:.0%} (30d rolling)\n"
            f"Review in the Hermes /skills dashboard."
        )
        await slack_client.chat_postMessage(channel=dm_channel, text=text)
        return True
    except Exception as exc:
        logger.warning("drift_detector: DM alert error: %s", exc)
        return False


async def _send_resolved_dm(
    slack_client,
    skill_name: str,
    resolved_rate: float,
) -> bool:
    """Send an auto-resolution DM to Blake."""
    blake_user_id = _blake_slack_user_id()
    if not blake_user_id:
        return False
    try:
        dm_resp = await slack_client.conversations_open(users=[blake_user_id])
        if not dm_resp.get("ok"):
            return False
        dm_channel = dm_resp["channel"]["id"]
        text = (
            f":white_check_mark: *Skill regression resolved*: `{skill_name}`\n"
            f"Thumbs rate recovered to {resolved_rate:.0%} (30d rolling)"
        )
        await slack_client.chat_postMessage(channel=dm_channel, text=text)
        return True
    except Exception as exc:
        logger.warning("drift_detector: DM resolve error: %s", exc)
        return False
