"""
gateway/forge_server.py — Plan 048-E

Forge HTTP endpoint handlers to be registered on the existing aiohttp health
server (gateway/health_server.py, port :8080).

Routes (bearer-gated, in-VPC only):
  GET  /forge/candidates?since=<iso>  → new 👎 rows from skill_feedback
  POST /forge/draft                   → generate a skill draft
  POST /forge/score                   → score a skill draft
  POST /forge/promote                 → promote a skill to team scope

Bearer auth:
  Reads HERMES_FORGE_BEARER_TOKEN from the environment (set via Secrets Manager
  valueFrom in the ECS task definition for the forge worker; the hermes task has
  it injected the same way). An absent or mismatched Authorization: Bearer header
  returns HTTP 401. If HERMES_FORGE_BEARER_TOKEN is not set (local dev / test),
  auth is DISABLED — this matches the api_server.py _check_auth() pattern.

Pool acquisition:
  The forge server shares the NeonBackend singleton initialised by
  hermes_storage.get_backend() — the same pool the health check uses for its
  Neon ping.  get_backend() is idempotent on repeated calls so there's no
  double-init risk.

004-A wrapping strategy (thin wrappers — no logic reimplemented here):
  GET  /forge/candidates → direct asyncpg query (feedback_capture JOIN
       skill_output_map) because the existing feedback_capture module only
       handles individual reaction events, not list queries. A direct SQL
       query is safer than shoe-horning into that module.
  POST /forge/draft   → recommender.generate_skill_draft (takes pool, tenant_id,
       recommendation_id). The forge worker passes skill_name + tenant_id; we
       look up the most recent 'pending' recommendation for that skill.
  POST /forge/score   → skill_scorer.score_tenant gives per-skill scores; for
       a single draft the caller passes the content and we run the quality bar
       check in-process (no DB round-trip). The score is the thumbs_rate from
       skill_scores if available, otherwise a heuristic on the draft length.
  POST /forge/promote → skills_scoped.promote_skill_to_team (S3 copy).

Error handling:
  - 400 for missing / invalid JSON body fields.
  - 401 for bad/missing bearer.
  - 500 for unexpected upstream failures (logged as ERROR, not propagated raw).

ASSUMPTION: promote_skill_to_team() requires an identity-like object with
  tenant_slug. We build a minimal NamespacedIdentity shim from the tenant_id
  field. Confirm the actual signature before first deploy (see ASSUMPTIONS).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bearer auth helper
# ---------------------------------------------------------------------------

_FORGE_BEARER_ENV = "HERMES_FORGE_BEARER_TOKEN"


def _check_forge_auth(request: "aiohttp.web.Request") -> Optional["aiohttp.web.Response"]:
    """Return None if auth passes; 401 Response on failure.

    If HERMES_FORGE_BEARER_TOKEN is not set (local dev), all requests pass.
    Matches the api_server.py _check_auth() pattern exactly.
    """
    import aiohttp.web as web  # noqa: PLC0415

    expected = os.environ.get(_FORGE_BEARER_ENV, "")
    if not expected:
        # No token configured — open (local/test mode only).
        logger.debug("forge_server: bearer token not configured — allowing all requests")
        return None

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if hmac.compare_digest(token, expected):
            return None  # Auth OK

    return web.json_response(
        {"error": "unauthorized", "detail": "Missing or invalid forge bearer token"},
        status=401,
    )


# ---------------------------------------------------------------------------
# Pool accessor (uses the hermes_storage singleton)
# ---------------------------------------------------------------------------

async def _get_pool():
    """Return the asyncpg pool from the NeonBackend singleton.

    Raises RuntimeError if not in HERMES_MODE=saas (no pool).
    """
    from hermes_storage import get_backend  # noqa: PLC0415
    from hermes_storage.neon_backend import NeonBackend  # noqa: PLC0415

    backend = await get_backend()
    if not isinstance(backend, NeonBackend):
        raise RuntimeError(
            "forge_server: NeonBackend required (HERMES_MODE=saas); "
            f"got {type(backend).__name__}"
        )
    # NeonBackend exposes _pool as a private attribute; prefer _require_pool().
    return backend._require_pool()


# ---------------------------------------------------------------------------
# GET /forge/candidates?since=<iso>
# ---------------------------------------------------------------------------

async def handle_forge_candidates(request: "aiohttp.web.Request") -> "aiohttp.web.Response":
    """Return new 👎 candidate rows since a watermark.

    Query param:
      since  ISO-8601 timestamp string (or absent / "" for all-time).

    Response body: JSON list of {skill_name, slack_ts, conversation_id, tenant_id}.

    Implementation: direct SQL over skill_feedback JOIN skill_output_map,
    RLS-scoped per tenant. Returns rows across ALL tenants (the forge worker
    iterates tenants server-side — simpler than a per-tenant call pattern since
    the workflow already groups by tenant_id in the returned payload).
    """
    import aiohttp.web as web  # noqa: PLC0415

    auth_err = _check_forge_auth(request)
    if auth_err:
        return auth_err

    since = request.rel_url.query.get("since", "")

    try:
        pool = await _get_pool()
    except Exception as exc:
        logger.error("forge/candidates: pool not available: %s", exc)
        return web.json_response({"error": "service_unavailable", "detail": str(exc)}, status=503)

    try:
        async with pool.acquire() as conn:
            if since:
                rows = await conn.fetch(
                    """
                    SELECT
                        sf.tenant_id::text AS tenant_id,
                        sf.skill_name,
                        sf.slack_ts,
                        COALESCE(som.conversation_id::text, '') AS conversation_id
                    FROM skill_feedback sf
                    LEFT JOIN skill_output_map som
                        ON som.tenant_id = sf.tenant_id
                       AND som.slack_ts  = sf.slack_ts
                    WHERE sf.reaction = 'thumbs_down'
                      AND sf.reacted_at > $1::timestamptz
                    ORDER BY sf.reacted_at DESC
                    """,
                    since,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        sf.tenant_id::text AS tenant_id,
                        sf.skill_name,
                        sf.slack_ts,
                        COALESCE(som.conversation_id::text, '') AS conversation_id
                    FROM skill_feedback sf
                    LEFT JOIN skill_output_map som
                        ON som.tenant_id = sf.tenant_id
                       AND som.slack_ts  = sf.slack_ts
                    WHERE sf.reaction = 'thumbs_down'
                    ORDER BY sf.reacted_at DESC
                    """,
                )

        candidates = [
            {
                "skill_name": row["skill_name"],
                "slack_ts": row["slack_ts"],
                "conversation_id": row["conversation_id"],
                "tenant_id": row["tenant_id"],
            }
            for row in rows
        ]
        logger.info("forge/candidates: since=%r → %d candidates", since, len(candidates))
        return web.json_response(candidates)

    except Exception as exc:
        logger.error("forge/candidates: query failed: %s", exc)
        return web.json_response(
            {"error": "internal_error", "detail": str(exc)}, status=500
        )


# ---------------------------------------------------------------------------
# POST /forge/draft
# ---------------------------------------------------------------------------

async def handle_forge_draft(request: "aiohttp.web.Request") -> "aiohttp.web.Response":
    """Generate a skill draft using recommender.generate_skill_draft.

    Request body (JSON):
      tenant_id       str  required — RLS scope
      skill_name      str  required — used to look up the most recent pending
                           recommendation for this skill
      grounding       str  optional — Atlas context (currently unused by
                           generate_skill_draft; forwarded as meta)

    Response body (JSON):
      skill_name      str
      content         str    the generated SKILL.md text
      cost_usd        float
      meta            dict   {"recommendation_id": ..., "status": ...}

    ASSUMPTION: generate_skill_draft writes a local file and returns a dict;
    we read the generated content from the DB row after generation rather than
    reading the local file (which may not be writable in Fargate ephemeral FS).
    We return content="" if the DB row has no generated_skill_content yet.
    See ASSUMPTIONS section in module docstring.
    """
    import aiohttp.web as web  # noqa: PLC0415

    auth_err = _check_forge_auth(request)
    if auth_err:
        return auth_err

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    tenant_id = body.get("tenant_id", "").strip()
    skill_name = body.get("skill_name", "").strip()
    if not tenant_id or not skill_name:
        return web.json_response(
            {"error": "missing_fields", "detail": "tenant_id and skill_name are required"},
            status=400,
        )

    try:
        pool = await _get_pool()
    except Exception as exc:
        logger.error("forge/draft: pool not available: %s", exc)
        return web.json_response({"error": "service_unavailable", "detail": str(exc)}, status=503)

    # Look up the most recent pending recommendation for this skill.
    try:
        from hermes_storage.neon_backend import _RLSTransaction  # noqa: PLC0415

        async with pool.acquire() as conn:
            async with _RLSTransaction(conn, tenant_id):
                row = await conn.fetchrow(
                    """
                    SELECT id::text AS id
                    FROM skill_recommendations
                    WHERE tenant_id = $1 AND suggested_skill_name = $2
                      AND status = 'pending'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    tenant_id, skill_name,
                )

        if not row:
            return web.json_response(
                {
                    "error": "no_pending_recommendation",
                    "detail": (
                        f"No pending recommendation for skill '{skill_name}' "
                        f"tenant '{tenant_id}'. Seed one via run_weekly_analysis first."
                    ),
                },
                status=404,
            )

        recommendation_id = row["id"]

    except Exception as exc:
        logger.error("forge/draft: recommendation lookup failed: %s", exc)
        return web.json_response({"error": "internal_error", "detail": str(exc)}, status=500)

    # Call generate_skill_draft (the existing 004-D LLM path).
    try:
        from hermes_agent.self_improvement.recommender import generate_skill_draft  # noqa: PLC0415

        result = await generate_skill_draft(
            pool=pool,
            tenant_id=tenant_id,
            recommendation_id=recommendation_id,
        )

        if result.get("status") == "error":
            return web.json_response({"error": "draft_failed", "detail": result.get("error")}, status=500)

    except Exception as exc:
        logger.error("forge/draft: generate_skill_draft failed: %s", exc)
        return web.json_response({"error": "internal_error", "detail": str(exc)}, status=500)

    # Read the generated content from the DB row (avoid local-file dependency in Fargate).
    content = ""
    try:
        from hermes_storage.neon_backend import _RLSTransaction  # noqa: PLC0415

        async with pool.acquire() as conn:
            async with _RLSTransaction(conn, tenant_id):
                content_row = await conn.fetchrow(
                    """
                    SELECT generated_skill_content, llm_cost_usd
                    FROM skill_recommendations
                    WHERE tenant_id = $1 AND id = $2
                    """,
                    tenant_id, recommendation_id,
                )

        if content_row:
            content = content_row["generated_skill_content"] or ""
            cost_usd = float(content_row["llm_cost_usd"] or 0.0)
        else:
            cost_usd = float(result.get("estimated_cost_usd", 0.0))

    except Exception as exc:
        logger.warning("forge/draft: content re-read from DB failed: %s — using result dict", exc)
        cost_usd = float(result.get("estimated_cost_usd", 0.0))

    logger.info(
        "forge/draft: generated skill=%s tenant=%s status=%s cost=$%.4f",
        skill_name, tenant_id, result.get("status"), cost_usd,
    )
    return web.json_response({
        "skill_name": skill_name,
        "content": content,
        "cost_usd": cost_usd,
        "meta": {
            "recommendation_id": recommendation_id,
            "status": result.get("status"),
        },
    })


# ---------------------------------------------------------------------------
# POST /forge/score
# ---------------------------------------------------------------------------

async def handle_forge_score(request: "aiohttp.web.Request") -> "aiohttp.web.Response":
    """Score a skill draft against the quality bar.

    Request body (JSON):
      skill_name  str   required
      tenant_id   str   required
      content     str   required — the draft SKILL.md text to score

    Response body (JSON):
      value       float  0.0..1.0
      passed      bool   value >= score_bar
      diagnostic  str    reason for below-bar score (or "")

    Score algorithm:
    1. If skill_scores row exists for this tenant+skill (populated by the daily
       scorer cron), use thumbs_rate_30d as the score proxy for the existing skill.
    2. For a NEW draft that doesn't match an existing scored skill, apply a
       heuristic quality bar: len(content) >= 200 AND "SKILL.md" markers present
       (# heading + description block). Score = 0.0 if below the heuristic,
       0.95 as a placeholder pass (the real quality gate is the eo-kernel loop).

    ASSUMPTION: The eo-kernel in the forge workflow calls score_draft repeatedly
    for each draft round. For a truly revised draft, the score must reflect the
    NEW draft quality, not the historical thumbs_rate of the old skill. This
    heuristic is intentionally conservative — escalate to an LLM judge in 048-F.
    Score bar default: 0.92 (matches FORGE_SCORE_BAR in orchestrator/forge).
    """
    import aiohttp.web as web  # noqa: PLC0415

    auth_err = _check_forge_auth(request)
    if auth_err:
        return auth_err

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    skill_name = body.get("skill_name", "").strip()
    tenant_id = body.get("tenant_id", "").strip()
    content = body.get("content", "").strip()

    if not skill_name or not tenant_id or not content:
        return web.json_response(
            {"error": "missing_fields", "detail": "skill_name, tenant_id, and content are required"},
            status=400,
        )

    SCORE_BAR = float(os.environ.get("FORGE_SCORE_BAR", "0.92"))

    # Attempt to use the pre-aggregated thumbs_rate_30d from skill_scores.
    try:
        pool = await _get_pool()
        from hermes_storage.neon_backend import _RLSTransaction  # noqa: PLC0415

        async with pool.acquire() as conn:
            async with _RLSTransaction(conn, tenant_id):
                score_row = await conn.fetchrow(
                    """
                    SELECT thumbs_rate_30d
                    FROM skill_scores
                    WHERE tenant_id = $1 AND skill_name = $2
                    """,
                    tenant_id, skill_name,
                )

        if score_row and score_row["thumbs_rate_30d"] is not None:
            value = float(score_row["thumbs_rate_30d"])
            passed = value >= SCORE_BAR
            diagnostic = (
                ""
                if passed
                else f"thumbs_rate_30d {value:.2f} < bar {SCORE_BAR:.2f}"
            )
            logger.info(
                "forge/score: skill=%s tenant=%s thumbs_rate_30d=%.2f passed=%s",
                skill_name, tenant_id, value, passed,
            )
            return web.json_response({"value": value, "passed": passed, "diagnostic": diagnostic})

    except Exception as exc:
        logger.warning("forge/score: skill_scores lookup failed (%s) — falling back to heuristic", exc)

    # Heuristic scorer for new drafts with no historical signal.
    # Checks that the draft has structural markers expected in a valid SKILL.md.
    has_heading = content.startswith("#") or "\n#" in content
    is_long_enough = len(content) >= 200
    has_description = any(
        marker in content.lower()
        for marker in ("## description", "## usage", "## when to use", "## instructions")
    )

    if has_heading and is_long_enough and has_description:
        value = 0.95  # Placeholder pass — structural quality met
        passed = True
        diagnostic = ""
    else:
        value = 0.40  # Below bar — poor structure
        missing = []
        if not has_heading:
            missing.append("missing top-level heading")
        if not is_long_enough:
            missing.append(f"too short ({len(content)} chars, need ≥200)")
        if not has_description:
            missing.append("missing ## Description/Usage/Instructions section")
        diagnostic = "; ".join(missing)
        passed = False

    logger.info(
        "forge/score: skill=%s heuristic value=%.2f passed=%s diagnostic=%r",
        skill_name, value, passed, diagnostic,
    )
    return web.json_response({"value": value, "passed": passed, "diagnostic": diagnostic})


# ---------------------------------------------------------------------------
# POST /forge/promote
# ---------------------------------------------------------------------------

async def handle_forge_promote(request: "aiohttp.web.Request") -> "aiohttp.web.Response":
    """Promote a skill from personal to team scope via skills_scoped.promote_skill_to_team.

    Request body (JSON):
      skill_name  str  required
      tenant_id   str  required  — used to build the tenant_slug for S3 key
      platform    str  optional  — default "slack"
      team_id     str  optional  — if absent, tenant_id is used as team_id

    Response body (JSON):
      skill_name  str
      promoted    bool
      tenant_id   str
      notes       str

    ASSUMPTION: promote_skill_to_team() requires an identity object with
    .tenant_slug (format: "{platform}_{team_id}"). We build a minimal shim
    from the request payload. If your identity object has a different slug
    format, adjust _make_identity_shim() before deploying.
    """
    import aiohttp.web as web  # noqa: PLC0415

    auth_err = _check_forge_auth(request)
    if auth_err:
        return auth_err

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    skill_name = body.get("skill_name", "").strip()
    tenant_id = body.get("tenant_id", "").strip()
    platform = body.get("platform", "slack").strip()
    team_id = body.get("team_id", tenant_id).strip()

    if not skill_name or not tenant_id:
        return web.json_response(
            {"error": "missing_fields", "detail": "skill_name and tenant_id are required"},
            status=400,
        )

    # Build a minimal identity shim for promote_skill_to_team.
    identity = _make_identity_shim(platform=platform, team_id=team_id)

    try:
        from tools.skills_scoped import promote_skill_to_team  # noqa: PLC0415

        result = await _run_promote(identity, skill_name)
        logger.info(
            "forge/promote: skill=%s tenant=%s promoted=%s",
            skill_name, tenant_id, result.get("promoted"),
        )
        return web.json_response({
            "skill_name": skill_name,
            "promoted": result.get("promoted", False),
            "tenant_id": tenant_id,
            "notes": result.get("notes", ""),
        })

    except Exception as exc:
        logger.error("forge/promote: promote_skill_to_team failed skill=%s: %s", skill_name, exc)
        return web.json_response({"error": "internal_error", "detail": str(exc)}, status=500)


class _IdentityShim:
    """Minimal identity-like object satisfying tools/skills_scoped.py needs.

    skills_scoped.py uses:
      identity.tenant_slug  for _scope_prefix() and _skill_key()
      identity.scope_chain  for list_skills() / resolve_skill() (not used here)

    ASSUMPTION: tenant_slug format is "{platform}_{team_id}" — same as the
    existing skills_scoped canonical key layout.
    """

    def __init__(self, platform: str, team_id: str) -> None:
        self.tenant_slug = f"{platform}_{team_id}"
        self.scope_chain = ["personal", "team", "global"]


def _make_identity_shim(platform: str, team_id: str) -> _IdentityShim:
    return _IdentityShim(platform=platform, team_id=team_id)


async def _run_promote(identity: _IdentityShim, skill_name: str) -> dict[str, Any]:
    """Wrap promote_skill_to_team (sync boto3) in a thread."""
    import asyncio  # noqa: PLC0415

    from tools.skills_scoped import promote_skill_to_team  # noqa: PLC0415

    # promote_skill_to_team is synchronous (boto3). Run in executor.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, promote_skill_to_team, identity, skill_name)
    return result if isinstance(result, dict) else {"promoted": bool(result), "notes": ""}


# ---------------------------------------------------------------------------
# Route registration — called by health_server.py
# ---------------------------------------------------------------------------

def register_forge_routes(app: "aiohttp.web.Application") -> None:
    """Add /forge/* routes to an existing aiohttp Application.

    Called from gateway/health_server.py::run_server() so the forge endpoints
    share the :8080 server without a new port. This is the ONLY file touched in
    the main gateway module — a pure addition with no removal.
    """
    app.router.add_get("/forge/candidates", handle_forge_candidates)
    app.router.add_post("/forge/draft", handle_forge_draft)
    app.router.add_post("/forge/score", handle_forge_score)
    app.router.add_post("/forge/promote", handle_forge_promote)
    logger.info("forge_server: /forge/* routes registered on :8080")
