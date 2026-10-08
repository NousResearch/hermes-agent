-- ============================================================================
-- 20260918_msq_stage_queue_r2.sql  (MSQ Phase 1 rider r2, task 805c8e51)
--
-- Writer rail + queue semantics as SECURITY DEFINER RPC core.
-- Context: table ships E-RLS-forced with ZERO policies (deny-by-default, per
-- E-RLS law). The scoped writer key in service is builder_rw (sb_secret rail,
-- same house pattern as bus_message/build_queue grants). RLS being FORCEd
-- means no grant-less role can touch rows; the RPC surface (DEFINER, owner
-- postgres) is the only writer door. Declared in bus 5a2155ad before landing.
--
-- Additive only: no column drops, no DELETE anywhere (append-only law:
-- completion is status='done', failure terminal is status='dead').
-- ============================================================================

-- r2.0 additive column: last_error for dead-letter diagnostics
alter table public.message_stage_queue
  add column if not exists last_error text;

-- r2.1 msq_stage — stage an inbound platform event (crash-dedupe aware)
create or replace function public.msq_stage(
  p_session_key text,
  p_platform    text,
  p_chat_id     text,
  p_event       jsonb,
  p_bot_name    text default null,
  p_message_id  text default null,
  p_sender_id   text default null,
  p_sender_name text default null
) returns uuid
language plpgsql security definer set search_path = public, pg_temp
as $$
declare
  v_id uuid;
begin
  insert into public.message_stage_queue
    (session_key, platform, chat_id, bot_name, message_id, sender_id, sender_name, event)
  values
    (p_session_key, p_platform, p_chat_id, p_bot_name, p_message_id, p_sender_id, p_sender_name, p_event)
  returning id into v_id;
  return v_id;
exception
  when unique_violation then
    -- partial unique (platform, chat_id, message_id) WHERE staged|processing:
    -- a restart re-staging the same platform message converges on the live row
    select id into v_id from public.message_stage_queue
    where platform = p_platform and chat_id = p_chat_id and message_id = p_message_id
      and status in ('staged','processing')
    limit 1;
    return v_id;
end;
$$;

-- r2.2 msq_claim_next — FIFO claim, one in flight per session
create or replace function public.msq_claim_next(
  p_session_key text
) returns table (
  id uuid, seq bigint, platform text, chat_id text, bot_name text,
  message_id text, sender_id text, sender_name text, event jsonb, attempts integer
)
language plpgsql security definer set search_path = public, pg_temp
as $$
declare
  v_id uuid;
begin
  -- one in flight: skip claim if a processing row exists for this session
  if exists (select 1 from public.message_stage_queue q
             where q.session_key = p_session_key and q.status = 'processing') then
    return;
  end if;

  select q.id into v_id
  from public.message_stage_queue q
  where q.session_key = p_session_key and q.status = 'staged'
  order by q.seq
  limit 1
  for update skip locked;

  if v_id is null then
    return;
  end if;

  return query
  update public.message_stage_queue q
  set status = 'processing', attempts = q.attempts + 1, released_at = now(), updated_at = now()
  where q.id = v_id
  returning q.id, q.seq, q.platform, q.chat_id, q.bot_name,
            q.message_id, q.sender_id, q.sender_name, q.event, q.attempts;
end;
$$;

-- r2.3 msq_complete — processing -> done | dead | back-to-staged (retry)
create or replace function public.msq_complete(
  p_row   uuid,
  p_ok    boolean,
  p_error text default null
) returns text
language plpgsql security definer set search_path = public, pg_temp
as $$
declare
  v_status text;
begin
  update public.message_stage_queue q
  set status = case
        when p_ok then 'done'
        when q.attempts >= 3 then 'dead'
        else 'staged'
      end,
      last_error = p_error,
      updated_at = now()
  where q.id = p_row and q.status = 'processing'
  returning q.status into v_status;
  return v_status;
end;
$$;

-- r2.4 msq_recover — crash recovery: re-queue stale processing rows
create or replace function public.msq_recover(
  p_session_key text default null,
  p_older_than  interval default '10 minutes'
) returns integer
language plpgsql security definer set search_path = public, pg_temp
as $$
declare
  v_n integer;
begin
  update public.message_stage_queue q
  set status = 'staged', updated_at = now(), last_error = 'recovered: stale processing'
  where q.status = 'processing'
    and q.updated_at < now() - p_older_than
    and (p_session_key is null or q.session_key = p_session_key);
  get diagnostics v_n = row_count;
  return v_n;
end;
$$;

-- r2.5 msq_depth — staged depth telemetry (session or fleet-wide)
create or replace function public.msq_depth(
  p_session_key text default null
) returns integer
language plpgsql security definer set search_path = public, pg_temp
as $$
begin
  return (select count(*) from public.message_stage_queue q
          where q.status = 'staged'
            and (p_session_key is null or q.session_key = p_session_key));
end;
$$;

-- r2.6 E-RLS grants: EXECUTE revoked from public/anon/authenticated in THIS migration;
--      writer rail = builder_rw (house pattern) + service_role
revoke all on function
  public.msq_stage(text, text, text, jsonb, text, text, text, text),
  public.msq_claim_next(text),
  public.msq_complete(uuid, boolean, text),
  public.msq_recover(text, interval),
  public.msq_depth(text)
from public, anon, authenticated;

grant execute on function
  public.msq_stage(text, text, text, jsonb, text, text, text, text),
  public.msq_claim_next(text),
  public.msq_complete(uuid, boolean, text),
  public.msq_recover(text, interval),
  public.msq_depth(text)
to builder_rw, service_role;

-- r2.7 ledger
insert into supabase_migrations.schema_migrations (version, name)
values ('20260918_msq_stage_queue_r2',
        'msq rider r2 (805c8e51 P1): SECURITY DEFINER RPC core — msq_stage/claim_next/complete/recover/depth; last_error column; EXECUTE revoked public+anon+auth, granted builder_rw+service_role; writer door for scoped builder_rw key; declared bus 5a2155ad')
on conflict (version) do nothing;
