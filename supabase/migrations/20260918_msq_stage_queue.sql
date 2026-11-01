-- ============================================================================
-- 20260918_msq_stage_queue.sql  (Message Staging Queue v1 Phase 1, task 805c8e51)
--
-- GATE-7 HOUSE PATTERN — ORACLE-FIRST RECONCILIATION (2026-09-18):
-- A table public.message_stage_queue was found LIVE on DEV with NO ledger
-- stamp, NO repo file, and NO engine code anywhere (full-box search + Archie
-- logs + state.db transcripts). Provenance: orphaned Phase-1 start by the
-- original 9/17 claimant of this card (bus dispatch ace7faff "Phase 1 build
-- starting", SLA-stale-reset 2026-09-17T23:30Z before engine work began).
-- The applied table is E-RLS-compliant (RLS forced, no policies, service_role
-- + postgres grants only, 0 rows) and its design is adopted: full event
-- snapshot in `event jsonb` (crash-safe replay) + bot_name/sender_name.
--
-- This file is byte-faithful to the live oracle (pg_constraintdef +
-- information_schema, captured 2026-09-18 ~11:4xZ) modulo:
--   - CREATE ... IF NOT EXISTS / ADD IF NOT EXISTS guards (idempotent re-run)
--   - additive-only hardening: drain index, staged-depth index, crash dedupe
--     unique, `done` terminal state (append-only completion — statuses were
--     staged/processing/dead with NO success terminal, which would have forced
--     row DELETE on completion, violating the append-only law).
-- The identity column, status vocab base, and event-snapshot shape are the
-- original author's; the hardening riders are builder02's (this claim).
--
-- E-RLS PRODUCT-DB TABLE LAW: the live table already ships ENABLE+FORCE RLS
-- with default grants revoked and no policies (deny-by-default, service-role
-- writers). The guards below re-assert it on re-run.
-- ============================================================================

create extension if not exists pgcrypto;

-- ----------------------------------------------------------------------------
-- 1. message_stage_queue — staging FIFO per chat session (oracle-faithful)
-- ----------------------------------------------------------------------------
create table if not exists public.message_stage_queue (
  id           uuid primary key default gen_random_uuid(),
  session_key  text not null,
  platform     text not null,
  chat_id      text not null,
  bot_name     text,
  message_id   text,
  sender_id    text,
  sender_name  text,
  seq          bigint generated always as identity,
  status       text not null default 'staged'
               check (status in ('staged','processing','dead')),
  attempts     integer not null default 0,
  event        jsonb not null,
  created_at   timestamptz not null default now(),
  released_at  timestamptz,
  updated_at   timestamptz not null default now()
);

-- ----------------------------------------------------------------------------
-- 2. Additive hardening riders (idempotent, guarded)
-- ----------------------------------------------------------------------------

-- 2a. Append-only completion: `done` terminal state (was missing — completion
--     would have required row DELETE, violating the fleet append-only law).
--     Existing check is dropped and re-created WITH the extended vocab; drop
--     is safe (check constraints have no dependents) and re-run-safe because
--     the new CREATE at the end of this block guards on the new name.
alter table public.message_stage_queue
  drop constraint if exists message_stage_queue_status_check;
alter table public.message_stage_queue
  add constraint message_stage_queue_status_check
  check (status in ('staged','processing','done','dead'));

-- 2b. FIFO drain path: next staged row(s) per session.
create index if not exists msq_session_status_seq_idx
  on public.message_stage_queue (session_key, status, seq);

-- 2b'. Processing-recovery path: an in-flight row whose gateway died mid-turn.
create index if not exists msq_session_processing_idx
  on public.message_stage_queue (session_key, seq)
  where status = 'processing';

-- 2c. Depth/alarm path: oldest staged per session.
create index if not exists msq_staged_created_idx
  on public.message_stage_queue (created_at)
  where status = 'staged';

-- 2d. Crash-restart dedupe: re-staging the same platform message converges on
--     one row (partial — commands/media may lack a platform message id).
create unique index if not exists msq_platform_msg_dedupe
  on public.message_stage_queue (platform, chat_id, message_id)
  where message_id is not null and status in ('staged','processing');

-- 2e. updated_at trigger — bump on every UPDATE (was missing; attempts/retry
--     and state transitions need it for observability + alarm thresholds).
create or replace function public.msq_touch_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists msq_touch_updated_at on public.message_stage_queue;
create trigger msq_touch_updated_at
  before update on public.message_stage_queue
  for each row execute function public.msq_touch_updated_at();

-- ----------------------------------------------------------------------------
-- 3. E-RLS re-assert (idempotent guards; already live from the original apply)
-- ----------------------------------------------------------------------------
alter table public.message_stage_queue enable row level security;
alter table public.message_stage_queue force  row level security;

revoke all on public.message_stage_queue from anon, authenticated, public;
revoke all on sequence public.message_stage_queue_seq_seq from anon, authenticated, public;

-- ----------------------------------------------------------------------------
-- 4. Migration ledger (house pattern), provenance-stamped
-- ----------------------------------------------------------------------------
insert into supabase_migrations.schema_migrations (version, name)
values ('20260918_msq_stage_queue',
        'msq_stage_queue (MSQ Phase 1 805c8e51: message_stage_queue staging FIFO — oracle-first reconciliation of orphaned 9/17 apply (ace7faff claim, no ledger/code) + hardening riders: done-terminal append-only vocab, drain/depth/dedupe indexes, updated_at trigger; E-RLS forced, grants revoked; engine ships separately pending delivery-vehicle ruling)')
on conflict (version) do nothing;
insert into supabase_migrations.schema_migrations (version, name)
values ('20260918_msq_stage_queue_orphan_base',
        'msq_stage_queue orphan base (provenance stamp: the live table pre-exists this migration — created ~2026-09-17 by the original card claimant via inline DDL outside the rail, discovered 2026-09-18 by builder02; this pseudo-version records that fact so the ledger never lies about the true DDL surface)')
on conflict (version) do nothing;

select 'reconciled' as result;
