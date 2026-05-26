-- =====================================================================
-- Migration 007: raw_events dedup index correction
-- =====================================================================
-- Discovered during Plan 007-B live integration testing:
--   the original dedup index from migration 006
--     (tenant_id, conversation_id, event_kind, platform_message_id)
--   never fires when conversation_id IS NULL (Postgres treats NULLs as
--   non-equal for uniqueness purposes). This breaks Slack-redelivery
--   idempotency for inbound events that arrive before a conversation
--   is established.
--
-- Fix: drop conversation_id from the dedup key. The natural Slack
--      dedup tuple is (tenant_id, event_kind, platform_message_id).
--      conversation_id is derived metadata and not part of the identity.
--
-- Idempotent. Safe to re-run.
-- =====================================================================

DROP INDEX IF EXISTS raw_events_dedup_key;

CREATE UNIQUE INDEX IF NOT EXISTS raw_events_dedup_key
    ON raw_events (tenant_id, event_kind, platform_message_id)
    WHERE platform_message_id IS NOT NULL;

COMMENT ON INDEX raw_events_dedup_key IS
    'Idempotency key for Slack redeliveries. (tenant, kind, msg_id) is the '
    'natural identity; conversation_id is derived and was removed in mig 007 '
    'because NULL conversation_ids broke the dedup invariant.';
