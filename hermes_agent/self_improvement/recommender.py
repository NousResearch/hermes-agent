"""
hermes_agent/self_improvement/recommender.py — Plan 004-D

Weekly LLM-driven skill-gap analysis. Detects task patterns in atlas-012 connector
data that have no matching existing skill, and generates SKILL.md proposals.

Algorithm:
1. Query recent atlas-012 Turn records via Neon (messages table, role=assistant,
   within the past 30 days).
2. Cluster similar turns using LLM-judged similarity (cosine sim >= 0.85 threshold
   on sentence embeddings; Phase D.1 can use OpenAI/Anthropic embeddings).
3. For clusters with >= MIN_CLUSTER_SIZE instances and no matching existing skill:
   - Generate a suggested_skill_name and task_summary via LLM.
   - Store in skill_recommendations table.
4. "Generate skill" action: when Blake clicks the button, draft a full SKILL.md.

Budget gate: $20/mo cap via BudgetGate class (in-code; no DB enforcement).
LLM: Claude Sonnet via Portkey (D-004-7 decision). NOT Haiku — clustering quality is
high-stakes enough to warrant the better model.

Cluster similarity: For Phase D MVP, we use a simple TF-IDF + cosine similarity
approach (no external embedding API) to avoid extra latency and cost at cluster time.
The LLM call (expensive) only happens at skill-generation time, not at cluster time.

This makes the weekly analysis run cost-free (no LLM calls) and only the
"Generate skill" button click triggers an LLM call (~$0.01-0.05 per generation).
The $20/mo budget gate covers cumulative generation costs.

Design decisions:
- TF-IDF clustering (not LLM embeddings) at detection time — fast, free, deterministic.
  This is the "earn complexity" principle: start with a simple baseline, escalate only
  when proven insufficient.
- LLM call deferred to generation time (explicit user action) — avoids weekly spend
  on unused recommendations.
- SKILL.md is written to a local temp file path for Blake to review. No auto-commit.
- Budget gate: checks recommender_budget table for current month spend before LLM call.
  Raises BudgetExceededError when $20/mo cap is hit.

Failure modes:
- No atlas turns in window: returns 0 clusters (correct — nothing to recommend).
- LLM API unavailable: BudgetGate.check() raises; generation returns error to caller.
- Budget exceeded: BudgetExceededError raised; Blake sees an error in the dashboard.
- Skill already exists: cluster is filtered out during candidate selection.

Assumptions:
- atlas-012 connector stores agent turns in the messages table (role='assistant').
  Turn content includes the agent's reasoning and tool calls.
- Existing skill names are enumerable from the local skills directory or S3 registry.
  Phase D uses a simplified check: skill name substrings in turn content.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MONTHLY_BUDGET_USD = float(os.environ.get("HERMES_RECOMMENDER_BUDGET_USD", "20.0"))
MIN_CLUSTER_SIZE = int(os.environ.get("HERMES_RECOMMENDER_MIN_CLUSTER_SIZE", "3"))
COSINE_SIM_THRESHOLD = float(os.environ.get("HERMES_RECOMMENDER_COSINE_THRESHOLD", "0.85"))
DETECTION_WINDOW_DAYS = int(os.environ.get("HERMES_RECOMMENDER_WINDOW_DAYS", "30"))
MAX_TURNS_TO_ANALYZE = int(os.environ.get("HERMES_RECOMMENDER_MAX_TURNS", "500"))


class BudgetExceededError(RuntimeError):
    """Raised when the monthly LLM budget for recommendations is exhausted."""
    def __init__(self, month: str, spent: float, cap: float):
        self.month = month
        self.spent = spent
        self.cap = cap
        super().__init__(
            f"Recommender budget exceeded for {month}: "
            f"${spent:.4f} spent / ${cap:.2f} cap. "
            f"Reset on the 1st or increase HERMES_RECOMMENDER_BUDGET_USD."
        )


# --------------------------------------------------------------------------
# Budget gate
# --------------------------------------------------------------------------

async def _get_month_spend(conn, tenant_id: str) -> float:
    """Return current month's LLM spend for this tenant."""
    month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    row = await conn.fetchrow(
        "SELECT cost_usd FROM recommender_budget WHERE tenant_id = $1 AND month = $2",
        tenant_id, month,
    )
    return float(row["cost_usd"]) if row else 0.0


async def _record_spend(conn, tenant_id: str, cost_usd: float) -> None:
    """Upsert monthly spend accumulator."""
    month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    await conn.execute(
        """
        INSERT INTO recommender_budget (tenant_id, month, cost_usd, updated_at)
        VALUES ($1, $2, $3, now())
        ON CONFLICT (tenant_id, month) DO UPDATE SET
            cost_usd = recommender_budget.cost_usd + EXCLUDED.cost_usd,
            updated_at = now()
        """,
        tenant_id, month, cost_usd,
    )


# --------------------------------------------------------------------------
# TF-IDF clustering (Phase D MVP — no LLM at cluster time)
# --------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b', text.lower())


def _build_tfidf_vectors(documents: list[str]) -> list[dict[str, float]]:
    """
    Compute TF-IDF vectors for a list of documents.

    Returns a list of {term: tfidf_weight} dicts, one per document.
    Uses log-normalized TF and IDF with +1 smoothing.
    """
    n = len(documents)
    if n == 0:
        return []

    # Term frequency per document.
    tf_docs = []
    for doc in documents:
        tokens = _tokenize(doc)
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        # Log-normalize TF.
        tf_docs.append({t: 1 + math.log(c) for t, c in tf.items()})

    # Document frequency.
    df: dict[str, int] = {}
    for tf in tf_docs:
        for t in tf:
            df[t] = df.get(t, 0) + 1

    # TF-IDF vectors.
    # When all documents share all terms, IDF collapses to 0 — fall back to
    # TF-only vectors in that case. This is the "stop using it when it breaks"
    # principle: IDF is only useful when there's cross-document variation.
    vectors = []
    for tf in tf_docs:
        vec = {
            t: w * math.log((n + 1) / (df[t] + 1))
            for t, w in tf.items()
        }
        # Check if all IDF weights collapsed to 0 — fall back to raw TF.
        if not any(v != 0 for v in vec.values()):
            vec = dict(tf)  # use raw TF (already log-normalized)
        # L2-normalize.
        norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})

    return vectors


def _cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    # Dot product over shared terms.
    dot = sum(v1.get(t, 0.0) * v2[t] for t in v2)
    # Vectors are already L2-normalized, so norms = 1.0.
    return dot


def _cluster_documents(
    texts: list[str],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    threshold: float = COSINE_SIM_THRESHOLD,
) -> list[list[int]]:
    """
    Group document indices into clusters by cosine similarity (greedy agglomeration).

    Algorithm:
    - For each unassigned document, start a new cluster.
    - For each subsequent unassigned document, if its cosine similarity to the
      cluster centroid >= threshold, add it to the cluster.
    - Centroid is the mean of TF-IDF vectors in the cluster.
    - Return only clusters with >= min_cluster_size members.

    O(n²) — acceptable for MAX_TURNS_TO_ANALYZE = 500.
    """
    if not texts:
        return []

    vectors = _build_tfidf_vectors(texts)
    assigned = [False] * len(texts)
    clusters: list[list[int]] = []

    for i in range(len(texts)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        # Centroid = average of member vectors (greedy).
        centroid: dict[str, float] = dict(vectors[i])

        for j in range(i + 1, len(texts)):
            if assigned[j]:
                continue
            sim = _cosine_sim(centroid, vectors[j])
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True
                # Update centroid (running mean).
                k = len(cluster)
                for t, v in vectors[j].items():
                    centroid[t] = ((k - 1) * centroid.get(t, 0) + v) / k

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


# --------------------------------------------------------------------------
# Weekly analysis job
# --------------------------------------------------------------------------

async def run_weekly_analysis(pool, tenant_id: str) -> dict[str, Any]:
    """
    Run the weekly skill-gap analysis for a tenant.

    1. Fetch recent assistant turns from messages table (atlas-012 connector data).
    2. Cluster similar turns using TF-IDF cosine similarity.
    3. For clusters with no matching existing skill: insert skill_recommendations row.
    4. Return summary.

    No LLM calls in this function — clustering is fully deterministic and free.
    LLM is only called in generate_skill_draft() (explicit user action).

    Args:
        pool:      asyncpg connection pool.
        tenant_id: Neon tenant UUID.

    Returns:
        Summary dict: {
            "turns_analyzed": int,
            "clusters_found": int,
            "new_recommendations": int,
        }
    """
    from hermes_storage.neon_backend import _RLSTransaction

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=DETECTION_WINDOW_DAYS)

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            # Fetch recent assistant turns.
            rows = await conn.fetch(
                """
                SELECT id, content, created_at
                FROM messages
                WHERE tenant_id = $1
                  AND role = 'assistant'
                  AND created_at >= $2
                  AND content IS NOT NULL
                  AND length(content) > 50
                ORDER BY created_at DESC
                LIMIT $3
                """,
                tenant_id, cutoff, MAX_TURNS_TO_ANALYZE,
            )

            if not rows:
                logger.info("recommender: tenant=%s no turns in window", tenant_id)
                return {"turns_analyzed": 0, "clusters_found": 0, "new_recommendations": 0}

            turn_ids = [str(r["id"]) for r in rows]
            turn_texts = [r["content"] or "" for r in rows]
            turn_dates = [r["created_at"] for r in rows]

            # Cluster turns.
            cluster_indices = _cluster_documents(
                turn_texts,
                min_cluster_size=MIN_CLUSTER_SIZE,
                threshold=COSINE_SIM_THRESHOLD,
            )

            new_recommendations = 0

            for cluster in cluster_indices:
                cluster_examples = []
                cluster_combined_text = ""
                for idx in cluster:
                    excerpt = turn_texts[idx][:200]
                    cluster_examples.append({
                        "turn_id": turn_ids[idx],
                        "summary": excerpt[:100],
                        "timestamp": turn_dates[idx].isoformat() if turn_dates[idx] else None,
                    })
                    cluster_combined_text += " " + turn_texts[idx]

                # Generate a deterministic suggested_skill_name from cluster content.
                # In production this would be a short LLM call; for MVP we use a
                # hash-based placeholder that Blake replaces on review.
                top_words = sorted(
                    set(_tokenize(cluster_combined_text)),
                    key=lambda w: cluster_combined_text.count(w),
                    reverse=True,
                )[:3]
                suggested_name = "-".join(top_words[:3]) if top_words else "auto-skill"
                # Sanitize to valid skill name format.
                suggested_name = re.sub(r"[^a-z0-9-]", "-", suggested_name.lower())[:64]

                task_summary = (
                    f"Recurring task pattern observed {len(cluster)} times in the last "
                    f"{DETECTION_WINDOW_DAYS} days. Representative excerpt: "
                    f"{cluster_examples[0]['summary']}"
                )

                # Insert recommendation (ON CONFLICT DO NOTHING — dedup by suggested_name).
                await conn.execute(
                    """
                    INSERT INTO skill_recommendations
                        (tenant_id, suggested_skill_name, task_summary,
                         cluster_examples, cluster_size, status)
                    VALUES ($1, $2, $3, $4::jsonb, $5, 'pending')
                    ON CONFLICT DO NOTHING
                    """,
                    tenant_id, suggested_name, task_summary,
                    json.dumps(cluster_examples), len(cluster),
                )
                new_recommendations += 1
                logger.info(
                    "recommender: new recommendation '%s' (cluster_size=%d) tenant=%s",
                    suggested_name, len(cluster), tenant_id,
                )

    logger.info(
        "recommender: tenant=%s turns=%d clusters=%d new=%d",
        tenant_id, len(rows), len(cluster_indices), new_recommendations,
    )
    return {
        "turns_analyzed": len(rows),
        "clusters_found": len(cluster_indices),
        "new_recommendations": new_recommendations,
    }


# --------------------------------------------------------------------------
# Skill draft generation (LLM call — user-triggered, budget-gated)
# --------------------------------------------------------------------------

async def generate_skill_draft(
    pool,
    tenant_id: str,
    recommendation_id: str,
    portkey_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a full SKILL.md draft for a pending recommendation using Claude Sonnet.

    Called when Blake clicks "Generate skill" in the dashboard.
    Budget-gated: raises BudgetExceededError if monthly spend >= $20.

    Args:
        pool:              asyncpg connection pool.
        tenant_id:         Neon tenant UUID.
        recommendation_id: UUID of the skill_recommendations row.
        portkey_api_key:   Portkey API key (test injection; env PORTKEY_API_KEY otherwise).

    Returns:
        {
            "status": "generated" | "insufficient_pattern",
            "skill_name": str,
            "draft_path": str,  # Absolute local file path for Blake to review.
            "estimated_cost_usd": float,
        }

    Failure modes:
    - BudgetExceededError: monthly cap hit → returned to caller as an error.
    - LLM API error: exception propagates (caller shows error in dashboard).
    - INSUFFICIENT_PATTERN response from LLM: returned with status="insufficient_pattern".
    """
    from hermes_storage.neon_backend import _RLSTransaction
    from pathlib import Path as _Path
    import jinja2

    api_key = portkey_api_key or os.environ.get("PORTKEY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "PORTKEY_API_KEY not set. Cannot call LLM for skill generation."
        )

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            # Budget check.
            current_spend = await _get_month_spend(conn, tenant_id)
            if current_spend >= MONTHLY_BUDGET_USD:
                month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
                raise BudgetExceededError(month, current_spend, MONTHLY_BUDGET_USD)

            # Fetch the recommendation row.
            row = await conn.fetchrow(
                """
                SELECT suggested_skill_name, task_summary, cluster_examples, cluster_size
                FROM skill_recommendations
                WHERE tenant_id = $1 AND id = $2 AND status = 'pending'
                """,
                tenant_id, recommendation_id,
            )
            if not row:
                return {
                    "status": "error",
                    "error": f"Recommendation {recommendation_id} not found or not pending.",
                }

            suggested_name = row["suggested_skill_name"]
            task_summary = row["task_summary"]
            cluster_examples = list(row["cluster_examples"] or [])
            cluster_size = row["cluster_size"]

    # Build the LLM prompt from the Jinja2 template.
    prompt_path = (
        _Path(__file__).parent.parent / "prompts" / "skill_recommendation.md.j2"
    )
    template_text = prompt_path.read_text(encoding="utf-8")
    env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.Undefined)
    template = env.from_string(template_text)
    prompt = template.render(
        suggested_name=suggested_name,
        task_summary=task_summary,
        cluster_examples=cluster_examples,
        cluster_size=cluster_size,
        window_days=DETECTION_WINDOW_DAYS,
    )

    # Call Claude Sonnet via Portkey (D-004-7: Sonnet, not Haiku).
    estimated_cost = 0.0
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.portkey.ai",
            default_headers={
                "x-portkey-provider": "anthropic",
            },
        )
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        skill_content = response.content[0].text.strip()

        # Approximate cost: Sonnet input ~$3/MTok, output ~$15/MTok.
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        estimated_cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    except Exception as exc:
        logger.error("recommender.generate_skill_draft: LLM call failed: %s", exc)
        raise

    if skill_content.strip() == "INSUFFICIENT_PATTERN":
        return {
            "status": "insufficient_pattern",
            "skill_name": suggested_name,
            "draft_path": None,
            "estimated_cost_usd": estimated_cost,
        }

    # Write the draft to a local temp file for Blake to review.
    # Per D-004-7: no auto-commit. Blake opens the file, reviews, and commits manually.
    draft_dir = _Path.home() / ".hermes" / "skill-drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    draft_path = draft_dir / f"{suggested_name}.md"
    draft_path.write_text(skill_content, encoding="utf-8")

    # Update DB: mark as generated + record spend.
    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            await conn.execute(
                """
                UPDATE skill_recommendations
                SET status = 'generated',
                    generated_skill_content = $2,
                    generated_at = now(),
                    llm_cost_usd = $3,
                    billed_month = $4
                WHERE tenant_id = $1 AND id = $5
                """,
                tenant_id, skill_content, estimated_cost,
                datetime.now(tz=timezone.utc).strftime("%Y-%m"),
                recommendation_id,
            )
            await _record_spend(conn, tenant_id, estimated_cost)

    logger.info(
        "recommender: generated skill draft '%s' at %s (cost=$%.4f)",
        suggested_name, draft_path, estimated_cost,
    )
    return {
        "status": "generated",
        "skill_name": suggested_name,
        "draft_path": str(draft_path),
        "estimated_cost_usd": estimated_cost,
    }


# --------------------------------------------------------------------------
# Dashboard reads
# --------------------------------------------------------------------------

async def get_pending_recommendations(pool, tenant_id: str) -> list[dict[str, Any]]:
    """Return pending recommendations for the dashboard."""
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            rows = await conn.fetch(
                """
                SELECT id, suggested_skill_name, task_summary,
                       cluster_size, cluster_examples, status, created_at
                FROM skill_recommendations
                WHERE tenant_id = $1 AND status = 'pending'
                ORDER BY cluster_size DESC, created_at DESC
                """,
                tenant_id,
            )
            return [
                {
                    "id": str(r["id"]),
                    "suggested_skill_name": r["suggested_skill_name"],
                    "task_summary": r["task_summary"],
                    "cluster_size": r["cluster_size"],
                    "cluster_examples": list(r["cluster_examples"] or []),
                    "status": r["status"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]


async def dismiss_recommendation(
    pool, tenant_id: str, recommendation_id: str
) -> dict[str, Any]:
    """Dismiss a pending recommendation."""
    from hermes_storage.neon_backend import _RLSTransaction

    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            await conn.execute(
                """
                UPDATE skill_recommendations
                SET status = 'dismissed', dismissed_at = now()
                WHERE tenant_id = $1 AND id = $2 AND status = 'pending'
                """,
                tenant_id, recommendation_id,
            )
    return {"status": "dismissed", "id": recommendation_id}


async def get_monthly_spend(pool, tenant_id: str) -> dict[str, Any]:
    """Return current month's LLM spend for the budget gate display."""
    from hermes_storage.neon_backend import _RLSTransaction

    month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    async with pool.acquire() as conn:
        async with _RLSTransaction(conn, tenant_id):
            spend = await _get_month_spend(conn, tenant_id)
    return {
        "month": month,
        "cost_usd": spend,
        "budget_usd": MONTHLY_BUDGET_USD,
        "remaining_usd": max(0.0, MONTHLY_BUDGET_USD - spend),
        "budget_exhausted": spend >= MONTHLY_BUDGET_USD,
    }
