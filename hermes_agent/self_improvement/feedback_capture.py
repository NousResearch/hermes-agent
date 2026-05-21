"""
hermes_agent/self_improvement/feedback_capture.py — Plan 004-A

Captures explicit user feedback (👍/👎 Slack reactions) on Hermes skill outputs
and persists them to the Neon `skill_feedback` table.

Data flow:
  Slack reaction_added event → handle_reaction_added() → lookup skill_output_map
                                                        → write skill_feedback row

  Slack reaction_removed event → handle_reaction_removed() → delete skill_feedback row

  Hermes sends a message → register_output() → write skill_output_map row

Design decisions:
- Correlation via slack_ts (Slack message timestamp). Hermes calls register_output()
  at send time to store (slack_ts, channel_id) → skill_name mapping. Reaction events
  reference (channel, ts) — we JOIN skill_output_map to resolve skill_name.
- RLS: all writes set app.tenant_id via NeonBackend._RLSTransaction pattern.
  tenant_id is resolved from identity.team_id + platform (same as NeonBackend).
- Emoji mapping: "thumbs_up" (👍, :thumbsup:, :+1:) and "thumbs_down" (👎).
  Unrecognised emoji are ignored — not every reaction is a quality signal.
- Atomicity: reaction_added → INSERT, reaction_removed → DELETE. The UNIQUE
  constraint on skill_feedback prevents duplicates from Socket Mode re-delivery.
- register_output() is a best-effort write — if it fails (pool cold-start, network),
  the message is still sent. A failed registration means reactions on that message
  are silently skipped (no skill_name lookup succeeds). This is logged as WARNING.

Failure modes:
- Pool not initialised: NeonBackend raises RuntimeError. Caller should ensure
  initialize() was called during gateway startup.
- Skill not found in skill_output_map: reaction is silently discarded with DEBUG log.
  This is correct for reactions on non-Hermes messages in the same channel.
- DB write fails (network, transient): logged as ERROR; reaction is lost.
  Volume is low enough that no retry queue is needed in Phase A.

Assumptions:
- tenant_id is resolved from (platform, team_id) using the same tenants table
  as NeonBackend._get_or_create_tenant(). We do NOT re-create tenants here;
  if a reaction arrives for an unknown tenant, we skip it (no message was ever
  registered for an unknown tenant).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Emoji → internal reaction name mapping.
# Slack sends the emoji name without colons (e.g. "+1", "thumbsup").
_THUMBS_UP_EMOJIS = frozenset({"+1", "thumbsup", "thumbs_up", "thumbsup2"})
_THUMBS_DOWN_EMOJIS = frozenset({"-1", "thumbsdown", "thumbs_down"})


def _classify_emoji(emoji_name: str) -> Optional[str]:
    """
    Return "thumbs_up" | "thumbs_down" | None for the given Slack emoji name.

    Slack sends emoji names without colons, e.g. "+1" not ":+1:".
    Returns None for all other emoji (not a quality signal we track).
    """
    name = emoji_name.strip().lower()
    if name in _THUMBS_UP_EMOJIS:
        return "thumbs_up"
    if name in _THUMBS_DOWN_EMOJIS:
        return "thumbs_down"
    return None


async def _resolve_tenant_id(conn, platform: str, team_id: str) -> Optional[str]:
    """
    Resolve the Neon tenant UUID from (platform, team_id).

    Returns None if the tenant doesn't exist yet (a reaction arrived before
    any Hermes output was registered for this workspace — safe to skip).
    The tenants table has no RLS (it's the root isolation boundary), so
    this lookup runs outside an RLS transaction.
    """
    row = await conn.fetchrow(
        "SELECT id FROM tenants WHERE platform = $1 AND external_id = $2",
        platform, team_id,
    )
    return str(row["id"]) if row else None


async def register_output(
    pool,
    platform: str,
    team_id: str,
    channel_id: str,
    slack_ts: str,
    skill_name: str,
) -> bool:
    """
    Register a Hermes output message in skill_output_map at send time.

    Called by the Slack gateway's send() path immediately after a message is
    posted. Stores (tenant_id, channel_id, slack_ts) → skill_name so that
    subsequent reaction events can resolve the skill that produced the output.

    Args:
        pool:       asyncpg connection pool (from NeonBackend._pool).
        platform:   Platform name, e.g. "slack".
        team_id:    Workspace/team identifier (Slack team_id, T-prefixed).
        channel_id: Channel where the message was sent.
        slack_ts:   Message timestamp returned by Slack API (e.g. "1234567890.123456").
        skill_name: Name of the skill that produced this output. If no skill was
                    active, callers should pass an empty string ("") or skip calling
                    this function — feedback on non-skill outputs is not tracked.

    Returns:
        True on successful write, False if registration was skipped or failed.

    Failure mode: DB errors are logged as WARNING but not raised — the message
    was already sent; registration failure means future reactions on this message
    won't be correlated to a skill, which is acceptable data loss for Phase A.
    """
    if not skill_name:
        return False

    try:
        async with pool.acquire() as conn:
            tenant_id = await _resolve_tenant_id(conn, platform, team_id)
            if not tenant_id:
                logger.warning(
                    "feedback_capture.register_output: no tenant for platform=%s team=%s; skipping",
                    platform, team_id,
                )
                return False

            from hermes_storage.neon_backend import _RLSTransaction
            async with _RLSTransaction(conn, tenant_id):
                await conn.execute(
                    """
                    INSERT INTO skill_output_map
                        (tenant_id, slack_ts, channel_id, skill_name)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (tenant_id, channel_id, slack_ts) DO NOTHING
                    """,
                    tenant_id, slack_ts, channel_id, skill_name,
                )
            logger.debug(
                "feedback_capture: registered output skill=%s ts=%s channel=%s tenant=%s",
                skill_name, slack_ts, channel_id, tenant_id,
            )
            return True

    except Exception as exc:
        logger.warning(
            "feedback_capture.register_output: write failed (skill=%s ts=%s): %s",
            skill_name, slack_ts, exc,
        )
        return False


async def handle_reaction_added(
    pool,
    platform: str,
    team_id: str,
    reactor_user_id: str,
    channel_id: str,
    item_ts: str,
    emoji_name: str,
) -> bool:
    """
    Handle a Slack reaction_added event.

    Looks up the skill that produced the message at item_ts in channel_id,
    then writes a row to skill_feedback.

    Args:
        pool:           asyncpg connection pool.
        platform:       "slack".
        team_id:        Slack workspace team_id.
        reactor_user_id: Slack user_id of the person who reacted.
        channel_id:     Channel containing the message (item.channel).
        item_ts:        Timestamp of the reacted-to message (item.ts).
        emoji_name:     Slack emoji name without colons, e.g. "+1".

    Returns:
        True if a feedback row was written, False if skipped or failed.

    Failure modes:
    - Unrecognised emoji: returns False immediately (not a quality signal).
    - Unknown slack_ts (not a Hermes output): returns False with DEBUG log.
    - DB error: logged as ERROR, returns False.
    """
    reaction = _classify_emoji(emoji_name)
    if reaction is None:
        logger.debug(
            "feedback_capture: ignoring non-quality emoji :%s: on ts=%s",
            emoji_name, item_ts,
        )
        return False

    try:
        async with pool.acquire() as conn:
            tenant_id = await _resolve_tenant_id(conn, platform, team_id)
            if not tenant_id:
                logger.debug(
                    "feedback_capture.handle_reaction_added: unknown tenant platform=%s team=%s",
                    platform, team_id,
                )
                return False

            from hermes_storage.neon_backend import _RLSTransaction
            async with _RLSTransaction(conn, tenant_id):
                # Resolve skill_name from the output map.
                map_row = await conn.fetchrow(
                    """
                    SELECT skill_name FROM skill_output_map
                    WHERE tenant_id = $1 AND channel_id = $2 AND slack_ts = $3
                    """,
                    tenant_id, channel_id, item_ts,
                )
                if not map_row:
                    logger.debug(
                        "feedback_capture: no skill mapping for ts=%s channel=%s (not a Hermes output)",
                        item_ts, channel_id,
                    )
                    return False

                skill_name = map_row["skill_name"]

                # Write feedback row; ON CONFLICT means Socket Mode re-delivery is idempotent.
                await conn.execute(
                    """
                    INSERT INTO skill_feedback
                        (tenant_id, skill_name, slack_ts, channel_id, reaction, reactor_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, skill_name, slack_ts, reactor_id) DO NOTHING
                    """,
                    tenant_id, skill_name, item_ts, channel_id, reaction, reactor_user_id,
                )
                logger.info(
                    "feedback_capture: %s on skill=%s ts=%s by user=%s tenant=%s",
                    reaction, skill_name, item_ts, reactor_user_id, tenant_id,
                )
                return True

    except Exception as exc:
        logger.error(
            "feedback_capture.handle_reaction_added: DB error (emoji=%s ts=%s): %s",
            emoji_name, item_ts, exc,
        )
        return False


async def handle_reaction_removed(
    pool,
    platform: str,
    team_id: str,
    reactor_user_id: str,
    channel_id: str,
    item_ts: str,
    emoji_name: str,
) -> bool:
    """
    Handle a Slack reaction_removed event.

    Deletes the corresponding skill_feedback row (if it exists).
    Idempotent — deleting a non-existent row is not an error.

    Args:
        pool:           asyncpg connection pool.
        platform:       "slack".
        team_id:        Slack workspace team_id.
        reactor_user_id: Slack user_id of the person who removed the reaction.
        channel_id:     Channel containing the message (item.channel).
        item_ts:        Timestamp of the reacted-to message (item.ts).
        emoji_name:     Slack emoji name (same as the original reaction_added event).

    Returns:
        True if a row was deleted, False if not found or not a tracked emoji.
    """
    reaction = _classify_emoji(emoji_name)
    if reaction is None:
        return False

    try:
        async with pool.acquire() as conn:
            tenant_id = await _resolve_tenant_id(conn, platform, team_id)
            if not tenant_id:
                return False

            from hermes_storage.neon_backend import _RLSTransaction
            async with _RLSTransaction(conn, tenant_id):
                # Resolve skill_name so we can match the exact row.
                map_row = await conn.fetchrow(
                    """
                    SELECT skill_name FROM skill_output_map
                    WHERE tenant_id = $1 AND channel_id = $2 AND slack_ts = $3
                    """,
                    tenant_id, channel_id, item_ts,
                )
                if not map_row:
                    return False

                skill_name = map_row["skill_name"]
                result = await conn.execute(
                    """
                    DELETE FROM skill_feedback
                    WHERE tenant_id = $1
                      AND skill_name = $2
                      AND slack_ts   = $3
                      AND reactor_id = $4
                    """,
                    tenant_id, skill_name, item_ts, reactor_user_id,
                )
                deleted = result != "DELETE 0"
                if deleted:
                    logger.info(
                        "feedback_capture: removed %s on skill=%s ts=%s by user=%s",
                        reaction, skill_name, item_ts, reactor_user_id,
                    )
                return deleted

    except Exception as exc:
        logger.error(
            "feedback_capture.handle_reaction_removed: DB error (ts=%s): %s",
            item_ts, exc,
        )
        return False
