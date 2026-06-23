-- V1.1 optional migration (status is stored in metrics.status jsonb by default).
-- Run only if you want a dedicated status column for reporting.

alter table public.campaigns
  add column if not exists status text not null default 'live';

create index if not exists campaigns_status_idx
  on public.campaigns (status);
