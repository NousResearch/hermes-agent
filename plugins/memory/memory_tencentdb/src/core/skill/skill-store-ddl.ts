/**
 * DDL 常量 — Skill 数据层 v2 重构
 *
 * 详见 `docs/design/2026-06-17-skill-redesign-v2.md` §2.1 / §2.2。
 *
 * 三张本模块所有的物理对象：
 *   - skills      — 主表，每行 = (skill_id, version) 一个不可变快照
 *   - skill_fts   — fts5 虚拟表（基于 head 行的 name/description/content）
 *   - skill_vec   — vec0 虚拟表（仅 dimensions>0 时创建）
 *
 * 故意不在此 DDL 中创建：
 *   - skill_bindings / task_skill_drafts / task_floating_skills / task_fixed_skills（绑定/草稿/浮动概念已下沉到管控面）
 *   - skill_resources（manifest 收敛到 skills.manifest_json 列）
 *   - assets / task_asset_bindings（全局资产体系不在数据面落库）
 */

// ═════════════════════════════════════════════════════════════════════
//  skills 主表 — 单表多行多版本
// ═════════════════════════════════════════════════════════════════════

export const SKILLS_DDL = `
  CREATE TABLE IF NOT EXISTS skills (
    row_id          TEXT PRIMARY KEY,
    skill_id        TEXT NOT NULL,
    version         INTEGER NOT NULL,
    is_head         INTEGER NOT NULL DEFAULT 1,

    user_id         TEXT NOT NULL,
    owner_agent_id  TEXT NOT NULL,
    team_id         TEXT NOT NULL,
    task_id         TEXT NOT NULL DEFAULT '',

    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    manifest_json   TEXT NOT NULL DEFAULT '[]',
    storage_dir     TEXT NOT NULL,

    status          TEXT NOT NULL DEFAULT 'active',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL,

    UNIQUE(skill_id, version)
  );

  CREATE UNIQUE INDEX IF NOT EXISTS uniq_skills_team_agent_name_head
    ON skills(team_id, owner_agent_id, name) WHERE is_head=1 AND status='active';

  CREATE INDEX IF NOT EXISTS idx_skills_team_head
    ON skills(team_id, is_head, status);

  CREATE INDEX IF NOT EXISTS idx_skills_owner_head
    ON skills(owner_agent_id, is_head, status);

  CREATE INDEX IF NOT EXISTS idx_skills_user
    ON skills(user_id, is_head);

  CREATE INDEX IF NOT EXISTS idx_skills_skill_version
    ON skills(skill_id, version DESC);

  CREATE INDEX IF NOT EXISTS idx_skills_task_audit
    ON skills(task_id, created_at_ms DESC);
`;

// ═════════════════════════════════════════════════════════════════════
//  skill_fts — FTS5 虚拟表（仅索引 head 行）
// ═════════════════════════════════════════════════════════════════════

export const SKILL_FTS_DDL = `
  CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
    name,
    description,
    content,
    skill_id UNINDEXED,
    team_id UNINDEXED,
    owner_agent_id UNINDEXED,
    task_id UNINDEXED,
    user_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 1'
  );
`;

// ═════════════════════════════════════════════════════════════════════
//  skill_vec — vec0 虚拟表（dimensions>0 时调用方负责 exec）
// ═════════════════════════════════════════════════════════════════════

/**
 * `__DIM__` 在 init 时被替换为实际维度（如 1536）。
 */
export const SKILL_VEC_DDL_TEMPLATE = `
  CREATE VIRTUAL TABLE IF NOT EXISTS skill_vec USING vec0(
    skill_id TEXT PRIMARY KEY,
    embedding float[__DIM__] distance_metric=cosine
  );
`;

// ═════════════════════════════════════════════════════════════════════
//  常量
// ═════════════════════════════════════════════════════════════════════

/** FTS 索引中 content 的最大字符数（避免巨大 SKILL.md 撑爆 fts5）。 */
export const FTS_CONTENT_MAX = 4000;
