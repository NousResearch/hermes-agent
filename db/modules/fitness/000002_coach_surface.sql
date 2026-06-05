-- Fitness Coach Surface: token-scoped visual routine/nutrition pages and events.
CREATE TABLE IF NOT EXISTS fitness.coach_workspaces (
  coach_workspace_id text PRIMARY KEY,
  public_token text NOT NULL UNIQUE,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  status text NOT NULL DEFAULT 'active',
  public_url text,
  expires_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.daily_plans (
  daily_plan_id text PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  plan_date date NOT NULL,
  plan_type text NOT NULL DEFAULT 'daily_regimen',
  title text NOT NULL,
  target_calories numeric,
  target_protein_g numeric,
  target_carbs_g numeric,
  target_fat_g numeric,
  target_water_ml numeric,
  meals jsonb NOT NULL DEFAULT '[]'::jsonb,
  routine jsonb NOT NULL DEFAULT '{}'::jsonb,
  recommendations jsonb NOT NULL DEFAULT '[]'::jsonb,
  status text NOT NULL DEFAULT 'generated',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (profile_id, plan_date, plan_type)
);

CREATE TABLE IF NOT EXISTS fitness.coach_events (
  coach_event_id bigserial PRIMARY KEY,
  coach_workspace_id text REFERENCES fitness.coach_workspaces(coach_workspace_id) ON DELETE CASCADE,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  event_type text NOT NULL,
  actor_type text NOT NULL DEFAULT 'user',
  actor_ref text,
  comment text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fitness_coach_workspaces_profile ON fitness.coach_workspaces(profile_id, status);
CREATE INDEX IF NOT EXISTS idx_fitness_daily_plans_profile_date ON fitness.daily_plans(profile_id, plan_date DESC);
CREATE INDEX IF NOT EXISTS idx_fitness_coach_events_profile_time ON fitness.coach_events(profile_id, occurred_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON fitness.coach_workspaces, fitness.daily_plans, fitness.coach_events TO fitness_runtime;
GRANT SELECT ON fitness.coach_workspaces, fitness.daily_plans, fitness.coach_events TO agent_runtime;
GRANT USAGE, SELECT ON SEQUENCE fitness.coach_events_coach_event_id_seq TO fitness_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA fitness GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fitness_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA fitness GRANT SELECT ON TABLES TO agent_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA fitness GRANT USAGE, SELECT ON SEQUENCES TO fitness_runtime;
