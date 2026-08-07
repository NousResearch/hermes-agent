-- ============================================================
-- 记忆内核 metadata 模块 — SQLite 初始化脚本（v3.2 · 按实例分库）
-- 每个实例独立库文件；表内无 instance_id。
--
-- 用法:
--   sqlite3 data/metadata/tdai_metadata_default/metadata.db < scripts/db/sqlite-init.sql
--
-- 注意：此脚本须与 sqlite-adapter.ts createSchema() 保持同步。
-- ============================================================

PRAGMA foreign_keys = ON;

-- ── meta_users ──
CREATE TABLE IF NOT EXISTS meta_users (
  user_id TEXT PRIMARY KEY,
  password TEXT,
  auth_provider TEXT NOT NULL,
  external_id TEXT NOT NULL,
  username TEXT NOT NULL,
  display_name TEXT,
  email TEXT,
  raw_profile_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'active',
  user_type TEXT NOT NULL DEFAULT 'normal',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_meta_users_system_admin ON meta_users(user_type) WHERE user_type = 'system_admin';
CREATE INDEX IF NOT EXISTS idx_meta_users_auth_username ON meta_users(auth_provider, username);
CREATE INDEX IF NOT EXISTS idx_meta_users_auth_external ON meta_users(auth_provider, external_id);
CREATE INDEX IF NOT EXISTS idx_meta_users_email ON meta_users(email) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_meta_users_created ON meta_users(created_at DESC);

-- ── meta_user_keys ──
CREATE TABLE IF NOT EXISTS meta_user_keys (
  key_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  key_value TEXT NOT NULL UNIQUE,
  name TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  is_default INTEGER NOT NULL DEFAULT 0,
  last_used_at TEXT,
  expires_at TEXT,
  created_at TEXT NOT NULL,
  revoked_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_user_keys_user ON meta_user_keys(user_id, status);
CREATE INDEX IF NOT EXISTS idx_meta_user_keys_user_created ON meta_user_keys(user_id, created_at DESC);

-- ── meta_teams ──
CREATE TABLE IF NOT EXISTS meta_teams (
  team_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  owner_user_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_teams_created ON meta_teams(created_at DESC);

-- ── meta_team_members ──
CREATE TABLE IF NOT EXISTS meta_team_members (
  id TEXT PRIMARY KEY,
  team_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'member',
  joined_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  UNIQUE(team_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_meta_members_user_status ON meta_team_members(user_id, status);
CREATE INDEX IF NOT EXISTS idx_meta_members_team_status_joined ON meta_team_members(team_id, status, joined_at DESC);

-- ── meta_agents ──
CREATE TABLE IF NOT EXISTS meta_agents (
  agent_id TEXT PRIMARY KEY,
  team_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  prompt TEXT,
  visibility TEXT NOT NULL DEFAULT 'team',
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_agents_team_status ON meta_agents(team_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_agents_owner_status_created ON meta_agents(owner_user_id, status, created_at DESC);

-- ── meta_tasks ──
CREATE TABLE IF NOT EXISTS meta_tasks (
  task_id TEXT PRIMARY KEY,
  team_id TEXT NOT NULL,
  creator_user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  source_type TEXT NOT NULL DEFAULT 'manual',
  source_url TEXT,
  status TEXT NOT NULL DEFAULT 'running',
  auto_assign_floating_assets INTEGER NOT NULL DEFAULT 0,
  risk_level TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_tasks_team_status ON meta_tasks(team_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_tasks_creator_status_created ON meta_tasks(creator_user_id, status, created_at DESC);

-- ── meta_task_agents ──
CREATE TABLE IF NOT EXISTS meta_task_agents (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  role_in_task TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  UNIQUE(task_id, agent_id)
);
CREATE INDEX IF NOT EXISTS idx_meta_task_agents_task_status_created ON meta_task_agents(task_id, status, created_at DESC);

-- ── meta_participation_logs ──
CREATE TABLE IF NOT EXISTS meta_participation_logs (
  id            TEXT PRIMARY KEY,
  team_id       TEXT NOT NULL,
  task_id       TEXT NOT NULL,
  agent_id      TEXT NOT NULL,
  user_id       TEXT NOT NULL,
  source        TEXT NOT NULL DEFAULT 'unknown',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meta_pl_team_created
  ON meta_participation_logs(team_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_pl_team_task_agent_created
  ON meta_participation_logs(team_id, task_id, agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_pl_team_user_created
  ON meta_participation_logs(team_id, user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_pl_team_dims_created
  ON meta_participation_logs(team_id, task_id, agent_id, user_id, created_at DESC);

-- ── meta_assets ──
CREATE TABLE IF NOT EXISTS meta_assets (
  asset_id TEXT PRIMARY KEY,
  team_id TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  owner_user_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_ref TEXT,
  version INTEGER NOT NULL DEFAULT 1,
  visibility TEXT NOT NULL DEFAULT 'team',
  status TEXT NOT NULL DEFAULT 'draft',
  confidence REAL,
  expires_at TEXT,
  last_used_at TEXT,
  usage_count INTEGER NOT NULL DEFAULT 0,
  content_ref TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_meta_assets_team_status ON meta_assets(team_id, status, created_at DESC);

-- ── meta_agent_fixed_assets ──
CREATE TABLE IF NOT EXISTS meta_agent_fixed_assets (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  asset_id TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  injection_mode TEXT NOT NULL DEFAULT 'summary',
  priority INTEGER NOT NULL DEFAULT 50,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(agent_id, asset_id)
);
CREATE INDEX IF NOT EXISTS idx_meta_fixed_agent_prio_created ON meta_agent_fixed_assets(agent_id, priority DESC, created_at DESC);

-- ── meta_asset_acl ──
CREATE TABLE IF NOT EXISTS meta_asset_acl (
  id TEXT PRIMARY KEY,
  asset_id TEXT NOT NULL,
  subject_type TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  permission TEXT NOT NULL,
  effect TEXT NOT NULL DEFAULT 'allow',
  granted_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(asset_id, subject_type, subject_id, permission)
);
CREATE INDEX IF NOT EXISTS idx_meta_acl_asset_created ON meta_asset_acl(asset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_acl_subject_created ON meta_asset_acl(subject_type, subject_id, created_at DESC);

-- ── meta_config_params ──
CREATE TABLE IF NOT EXISTS meta_config_params (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope TEXT NOT NULL CHECK (scope IN ('global', 'user')),
  user_id TEXT,
  module TEXT NOT NULL,
  param_name TEXT NOT NULL,
  param_value TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK (
    (scope = 'global' AND user_id IS NULL) OR
    (scope = 'user' AND user_id IS NOT NULL)
  )
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_meta_config_params_global
  ON meta_config_params(module, param_name) WHERE scope = 'global';
CREATE UNIQUE INDEX IF NOT EXISTS ux_meta_config_params_user
  ON meta_config_params(user_id, module, param_name) WHERE scope = 'user';
CREATE INDEX IF NOT EXISTS idx_meta_config_params_module
  ON meta_config_params(module);
