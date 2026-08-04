/**
 * 记忆内核 metadata 模块 — MongoDB 索引初始化（v3.2 · 按实例分库）
 * 每个实例独立 database：tdai_metadata_{service-id}
 *
 * 用法:
 *   mongosh "$URI" --eval 'const dbName="tdai_metadata_default"' scripts/db/mongodb-init.js
 */

// eslint-disable-next-line no-undef
const database = db.getSiblingDB(typeof dbName !== "undefined" ? dbName : "tdai_metadata_default");

// ── meta_users ──
database.meta_users.createIndex({ user_id: 1 }, { unique: true });
database.meta_users.createIndex(
  { user_type: 1 },
  { unique: true, partialFilterExpression: { user_type: "system_admin" } },
);
database.meta_users.createIndex({ auth_provider: 1, username: 1 });
database.meta_users.createIndex(
  { auth_provider: 1, external_id: 1 },
  { sparse: true },
);
database.meta_users.createIndex({ email: 1 }, { sparse: true });
database.meta_users.createIndex({ created_at: -1 });

// ── meta_user_keys ──
database.meta_user_keys.createIndex({ key_id: 1 }, { unique: true });
database.meta_user_keys.createIndex({ key_value: 1 }, { unique: true });
database.meta_user_keys.createIndex({ user_id: 1, status: 1 });
database.meta_user_keys.createIndex({ user_id: 1, created_at: -1 });

// ── meta_teams ──
database.meta_teams.createIndex({ team_id: 1 }, { unique: true });
database.meta_teams.createIndex({ created_at: -1 });

// ── meta_team_members ──
database.meta_team_members.createIndex({ team_id: 1, user_id: 1 }, { unique: true });
database.meta_team_members.createIndex({ team_id: 1, status: 1, joined_at: -1 });
database.meta_team_members.createIndex({ user_id: 1, status: 1 });

// ── meta_agents ──
database.meta_agents.createIndex({ agent_id: 1 }, { unique: true });
database.meta_agents.createIndex({ team_id: 1, status: 1, created_at: -1 });
database.meta_agents.createIndex({ owner_user_id: 1, status: 1, created_at: -1 });

// ── meta_tasks ──
database.meta_tasks.createIndex({ task_id: 1 }, { unique: true });
database.meta_tasks.createIndex({ team_id: 1, status: 1, created_at: -1 });
database.meta_tasks.createIndex({ creator_user_id: 1, status: 1, created_at: -1 });

// ── meta_task_agents ──
database.meta_task_agents.createIndex({ task_id: 1, agent_id: 1 }, { unique: true });
database.meta_task_agents.createIndex({ task_id: 1, status: 1, created_at: -1 });

// ── meta_participation_logs ──
database.meta_participation_logs.createIndex(
  { team_id: 1, created_at: -1 },
  { name: "ix_pl_team_created" },
);
database.meta_participation_logs.createIndex(
  { team_id: 1, task_id: 1, agent_id: 1, created_at: -1 },
  { name: "ix_pl_team_task_agent_created" },
);
database.meta_participation_logs.createIndex(
  { team_id: 1, user_id: 1, created_at: -1 },
  { name: "ix_pl_team_user_created" },
);
database.meta_participation_logs.createIndex(
  { team_id: 1, task_id: 1, agent_id: 1, user_id: 1, created_at: -1 },
  { name: "ix_pl_team_dims_created" },
);

// ── meta_assets ──
database.meta_assets.createIndex({ asset_id: 1 }, { unique: true });
database.meta_assets.createIndex({ team_id: 1, status: 1, created_at: -1 });

// ── meta_agent_fixed_assets ──
database.meta_agent_fixed_assets.createIndex({ agent_id: 1, asset_id: 1 }, { unique: true });
database.meta_agent_fixed_assets.createIndex({ agent_id: 1, priority: -1, created_at: -1 });

// ── meta_asset_acl ──
database.meta_asset_acl.createIndex(
  { asset_id: 1, subject_type: 1, subject_id: 1, permission: 1 },
  { unique: true },
);
database.meta_asset_acl.createIndex({ id: 1 }, { unique: true });
database.meta_asset_acl.createIndex({ asset_id: 1, created_at: -1 });
database.meta_asset_acl.createIndex({ subject_type: 1, subject_id: 1, created_at: -1 });

// ── meta_config_params ──
database.meta_config_params.createIndex(
  { scope: 1, user_id: 1, module: 1, param_name: 1 },
  { unique: true },
);
database.meta_config_params.createIndex({ module: 1 });
database.meta_config_params.createIndex(
  { user_id: 1, module: 1 },
  { partialFilterExpression: { scope: "user" } },
);

print(`indexes ensured on ${database.getName()}`);
