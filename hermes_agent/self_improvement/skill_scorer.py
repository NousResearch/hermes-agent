"""
hermes_agent/self_improvement/skill_scorer.py — Plan 004-A

Aggregates per-skill feedback into rolling-window scores and upserts them
into the `skill_scores` Neon table.

Score model:
  - usage_Nd  = count of distinct slack_ts values in skill_output_map within N days
  - thumbs_up_Nd / thumbs_down_Nd = reaction counts from skill_feedback within N days
  - thumbs_rate_Nd = thumbs_up / (thumbs_up + thumbs_down) — NULL when total < 3

Windows: 7d, 30d, all-time.

Design decisions:
- Runs as a periodic job (daily, via the hermes cron scheduler or an async task).
  Does NOT run inline with reaction events — keeps the reaction path < 5ms.
- Tenant-scoped: updates skill_scores rows for a single tenant per call.
  run_for_all_tenants() iterates tenants and calls score_tenant() for each.
- RLS: all reads/writes use _RLSTransaction per Plan 001-D pattern.
- Insufficient-signal rule: thumbs_rate is set to NULL when total reactions < 3.
  This prevents misleading 100% scores from 1-reaction skills.
- UPSERT strategy: INSERT ... ON CONFLICT DO UPDATE for skill_scores rows.
  This is idempotent: re-running the scorer produces the same result.

Failure modes:
- Pool not initialised: RuntimeError propagates to the caller (scheduler).
- Neon transient error: exception propagates; scheduler should retry.
- Zero skills in output_map: produces zero rows in skill_scores (correct).

Assumptions:
- tenants table has no RLS (root isolation boundary); iterating tenants
  for run_for_all_tenants() is an unguarded SELECT.
- skill_feedback and skill_output_map are populated by feedback_capture.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimum total reactions (thumbs_up + thumbs_down) before computing thumbs_rate.
# Below this threshold, thumbs_rate is NULL (insufficient signal).
MIN_SIGNAL = 3

# Rolling windows in days.
WINDOW_7D = 7
WINDOW_30D = 30


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _window_cutoff(days: int) -> datetime:
    """Return UTC datetime N days ago."""
    return _utc_now() - timedelta(days=days)


def _compute_thumbs_rate(up: int, down: int) -> Optional[float]:
    """
    Compute thumbs_rate from raw counts.

    Returns:
        float in [0.0, 1.0] when total >= MIN_SIGNAL, else None.
    """
    total = up + down
    if total < MIN_SIGNAL:
        return None
    return round(up / total, 4)


async def score_tenant(pool, tenant_id: str) -> dict[str, Any]:
    """
    Aggregate per-skill feedback for a single tenant and upsert skill_scores.

    Args:
        pool:      asyncpg connection pool (NeonBackend._pool).
        tenant_id: Neon UUID of the tenant to score.

    Returns:
        Summary dict: {
            "tenant_id": str,
            "skills_scored": int,
            "skills_zero_feedback": int,
            "elapsed_ms": float,
        }

    This function runs inside a single RLS transaction so all reads and
    the final UPSERT batch are consistent within one snapshot.
    """
    import time
    t0 = time.monotonic()

    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            cutoff_7d = _window_cutoff(WINDOW_7D)
            cutoff_30d = _window_cutoff(WINDOW_30D)

            # All distinct skill_names this tenant has produced outputs for.
            skill_rows = await conn.fetch(
                """
                SELECT DISTINCT skill_name FROM skill_output_map
                WHERE tenant_id = $1
                """,
                tenant_id,
            )
            skill_names = [r["skill_name"] for r in skill_rows]

            if not skill_names:
                logger.debug("skill_scorer: tenant=%s has no skill outputs yet", tenant_id)
                return {
                    "tenant_id": tenant_id,
                    "skills_scored": 0,
                    "skills_zero_feedback": 0,
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                }

            skills_scored = 0
            skills_zero_feedback = 0

            for skill_name in skill_names:
                # Usage = distinct messages in output_map within window.
                usage_7d_row = await conn.fetchrow(
                    """
                    SELECT COUNT(DISTINCT slack_ts) AS cnt FROM skill_output_map
                    WHERE tenant_id = $1 AND skill_name = $2
                      AND registered_at >= $3
                    """,
                    tenant_id, skill_name, cutoff_7d,
                )
                usage_30d_row = await conn.fetchrow(
                    """
                    SELECT COUNT(DISTINCT slack_ts) AS cnt FROM skill_output_map
                    WHERE tenant_id = $1 AND skill_name = $2
                      AND registered_at >= $3
                    """,
                    tenant_id, skill_name, cutoff_30d,
                )
                usage_all_row = await conn.fetchrow(
                    """
                    SELECT COUNT(DISTINCT slack_ts) AS cnt FROM skill_output_map
                    WHERE tenant_id = $1 AND skill_name = $2
                    """,
                    tenant_id, skill_name,
                )
                last_used_row = await conn.fetchrow(
                    """
                    SELECT MAX(registered_at) AS last_used FROM skill_output_map
                    WHERE tenant_id = $1 AND skill_name = $2
                    """,
                    tenant_id, skill_name,
                )

                usage_7d = int(usage_7d_row["cnt"] or 0)
                usage_30d = int(usage_30d_row["cnt"] or 0)
                usage_all = int(usage_all_row["cnt"] or 0)
                last_used_at = last_used_row["last_used"]

                # Feedback aggregation within windows.
                def _agg_sql(cutoff_param: int) -> str:
                    return """
                        SELECT
                            COUNT(*) FILTER (WHERE reaction = 'thumbs_up') AS up,
                            COUNT(*) FILTER (WHERE reaction = 'thumbs_down') AS down
                        FROM skill_feedback
                        WHERE tenant_id = $1 AND skill_name = $2
                          AND reacted_at >= $3
                    """

                fb_7d = await conn.fetchrow(
                    _agg_sql(3), tenant_id, skill_name, cutoff_7d
                )
                fb_30d = await conn.fetchrow(
                    _agg_sql(3), tenant_id, skill_name, cutoff_30d
                )

                up_7d = int(fb_7d["up"] or 0)
                down_7d = int(fb_7d["down"] or 0)
                up_30d = int(fb_30d["up"] or 0)
                down_30d = int(fb_30d["down"] or 0)

                rate_7d = _compute_thumbs_rate(up_7d, down_7d)
                rate_30d = _compute_thumbs_rate(up_30d, down_30d)

                if (up_7d + down_7d + up_30d + down_30d) == 0:
                    skills_zero_feedback += 1

                # UPSERT skill_scores row.
                await conn.execute(
                    """
                    INSERT INTO skill_scores
                        (tenant_id, skill_name,
                         usage_7d, usage_30d, usage_all,
                         thumbs_up_7d, thumbs_down_7d,
                         thumbs_up_30d, thumbs_down_30d,
                         thumbs_rate_7d, thumbs_rate_30d,
                         last_used_at, last_scored_at)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now())
                    ON CONFLICT (tenant_id, skill_name) DO UPDATE SET
                        usage_7d        = EXCLUDED.usage_7d,
                        usage_30d       = EXCLUDED.usage_30d,
                        usage_all       = EXCLUDED.usage_all,
                        thumbs_up_7d    = EXCLUDED.thumbs_up_7d,
                        thumbs_down_7d  = EXCLUDED.thumbs_down_7d,
                        thumbs_up_30d   = EXCLUDED.thumbs_up_30d,
                        thumbs_down_30d = EXCLUDED.thumbs_down_30d,
                        thumbs_rate_7d  = EXCLUDED.thumbs_rate_7d,
                        thumbs_rate_30d = EXCLUDED.thumbs_rate_30d,
                        last_used_at    = EXCLUDED.last_used_at,
                        last_scored_at  = now()
                    """,
                    tenant_id, skill_name,
                    usage_7d, usage_30d, usage_all,
                    up_7d, down_7d,
                    up_30d, down_30d,
                    rate_7d, rate_30d,
                    last_used_at,
                )
                skills_scored += 1
                logger.debug(
                    "skill_scorer: tenant=%s skill=%s usage_30d=%d rate_30d=%s",
                    tenant_id, skill_name, usage_30d,
                    f"{rate_30d:.2%}" if rate_30d is not None else "insufficient_signal",
                )

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        "skill_scorer: tenant=%s scored %d skills (%d zero feedback) in %.1fms",
        tenant_id, skills_scored, skills_zero_feedback, elapsed_ms,
    )
    return {
        "tenant_id": tenant_id,
        "skills_scored": skills_scored,
        "skills_zero_feedback": skills_zero_feedback,
        "elapsed_ms": elapsed_ms,
    }


async def run_for_all_tenants(pool) -> list[dict[str, Any]]:
    """
    Run score_tenant() for every tenant in the tenants table.

    Iterates tenant rows (no RLS — tenants table is the root isolation boundary)
    and calls score_tenant() for each. Returns a list of per-tenant summary dicts.

    This is the entry point called by the daily cron job (Phase 004-B registers
    the cron; Phase 004-A sets it up for manual / scheduled runs).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id FROM tenants")
        tenant_ids = [str(r["id"]) for r in rows]

    if not tenant_ids:
        logger.info("skill_scorer.run_for_all_tenants: no tenants found")
        return []

    logger.info("skill_scorer.run_for_all_tenants: scoring %d tenant(s)", len(tenant_ids))
    results = []
    for tid in tenant_ids:
        try:
            summary = await score_tenant(pool, tid)
            results.append(summary)
        except Exception as exc:
            logger.error("skill_scorer: tenant=%s scoring failed: %s", tid, exc)
            results.append({"tenant_id": tid, "error": str(exc)})

    return results


async def get_skill_scores(pool, tenant_id: str) -> list[dict[str, Any]]:
    """
    Return all skill_scores rows for a tenant, sorted by thumbs_rate_30d DESC.

    Used by the /skills dashboard API endpoint to populate the table.
    Reads from the pre-aggregated skill_scores table (no on-the-fly aggregation).

    Returns:
        List of dicts with keys: skill_name, usage_7d, usage_30d, usage_all,
        thumbs_up_30d, thumbs_down_30d, thumbs_rate_30d, last_used_at, last_scored_at.
        thumbs_rate_30d is None when total reactions < MIN_SIGNAL (insufficient signal).
    """
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            rows = await conn.fetch(
                """
                SELECT
                    skill_name,
                    usage_7d, usage_30d, usage_all,
                    thumbs_up_7d, thumbs_down_7d,
                    thumbs_up_30d, thumbs_down_30d,
                    thumbs_rate_7d, thumbs_rate_30d,
                    last_used_at, last_scored_at
                FROM skill_scores
                WHERE tenant_id = $1
                ORDER BY thumbs_rate_30d DESC NULLS LAST, usage_30d DESC
                """,
                tenant_id,
            )
            return [
                {
                    "skill_name": r["skill_name"],
                    "usage_7d": r["usage_7d"],
                    "usage_30d": r["usage_30d"],
                    "usage_all": r["usage_all"],
                    "thumbs_up_7d": r["thumbs_up_7d"],
                    "thumbs_down_7d": r["thumbs_down_7d"],
                    "thumbs_up_30d": r["thumbs_up_30d"],
                    "thumbs_down_30d": r["thumbs_down_30d"],
                    # Convert Decimal to float for JSON serialization.
                    "thumbs_rate_7d": float(r["thumbs_rate_7d"]) if r["thumbs_rate_7d"] is not None else None,
                    "thumbs_rate_30d": float(r["thumbs_rate_30d"]) if r["thumbs_rate_30d"] is not None else None,
                    "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
                    "last_scored_at": r["last_scored_at"].isoformat() if r["last_scored_at"] else None,
                }
                for r in rows
            ]
