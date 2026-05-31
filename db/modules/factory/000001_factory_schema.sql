-- Factory progress schema in the shared Agent Core DB.
CREATE SCHEMA IF NOT EXISTS factory;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('factory', 'Factory progress, lanes, tasks, gates, events, metrics, and artifacts.', 'software-factory', 'factory')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

CREATE TABLE IF NOT EXISTS factory.agents (
  agent_id text PRIMARY KEY,
  profile_name text NOT NULL,
  role text NOT NULL,
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factory.projects (
  project_id text PRIMARY KEY,
  name text NOT NULL,
  repo_path text,
  repo_remote text,
  base_branch text,
  status text NOT NULL DEFAULT 'intake',
  autonomy_level integer,
  methodology text,
  risk_level text,
  human_owner text,
  started_at timestamptz,
  updated_at timestamptz NOT NULL DEFAULT now(),
  summary text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS factory.lanes (
  lane_id text PRIMARY KEY,
  project_id text NOT NULL REFERENCES factory.projects(project_id) ON DELETE CASCADE,
  name text NOT NULL,
  methodology text,
  branch text,
  worktree_path text,
  status text NOT NULL DEFAULT 'planned',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factory.tasks (
  task_id text PRIMARY KEY,
  project_id text NOT NULL REFERENCES factory.projects(project_id) ON DELETE CASCADE,
  lane_id text REFERENCES factory.lanes(lane_id) ON DELETE SET NULL,
  kanban_id text,
  title text NOT NULL,
  description text,
  phase text,
  status text NOT NULL DEFAULT 'todo',
  owner_profile text,
  engine text,
  priority integer DEFAULT 0,
  dependencies jsonb NOT NULL DEFAULT '[]'::jsonb,
  branch text,
  worktree_path text,
  acceptance_criteria jsonb NOT NULL DEFAULT '[]'::jsonb,
  started_at timestamptz,
  finished_at timestamptz,
  result_summary text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factory.events (
  event_id bigserial PRIMARY KEY,
  project_id text REFERENCES factory.projects(project_id) ON DELETE SET NULL,
  task_id text REFERENCES factory.tasks(task_id) ON DELETE SET NULL,
  timestamp timestamptz NOT NULL DEFAULT now(),
  actor text NOT NULL,
  event_type text NOT NULL,
  message text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS factory.gates (
  gate_id bigserial PRIMARY KEY,
  project_id text NOT NULL REFERENCES factory.projects(project_id) ON DELETE CASCADE,
  task_id text REFERENCES factory.tasks(task_id) ON DELETE CASCADE,
  gate_type text NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  reviewer text,
  evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
  notes text,
  timestamp timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factory.artifacts (
  artifact_id bigserial PRIMARY KEY,
  project_id text REFERENCES factory.projects(project_id) ON DELETE CASCADE,
  task_id text REFERENCES factory.tasks(task_id) ON DELETE SET NULL,
  artifact_type text NOT NULL,
  path text NOT NULL,
  checksum text,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS factory.metrics (
  metric_id bigserial PRIMARY KEY,
  project_id text REFERENCES factory.projects(project_id) ON DELETE CASCADE,
  task_id text REFERENCES factory.tasks(task_id) ON DELETE CASCADE,
  engine text,
  duration_seconds numeric,
  files_changed integer,
  lines_added integer,
  lines_removed integer,
  tests_total integer,
  tests_passed integer,
  tests_failed integer,
  review_findings_count integer,
  rework_count integer,
  cost_estimate_usd numeric,
  spec_score numeric,
  quality_score numeric,
  security_score numeric,
  created_at timestamptz NOT NULL DEFAULT now(),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);
