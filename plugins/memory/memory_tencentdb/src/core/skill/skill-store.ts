/**
 * SqliteSkillStore — 新版 skill 数据访问层。
 *
 * 设计目标：
 *   - 单表多行多版本：每条 (skill_id, version) 一行，is_head=1 标记当前版本
 *   - 五元组身份硬字段：user_id / owner_agent_id / team_id / task_id / skill_id
 *   - 不感知绑定 / 浮动 / 草稿 / 冲突
 *   - 写入路径：旧 head 改 0 → INSERT 新行 → fts5 同步（事务原子）
 *   - 读取路径：所有查询强制 team_id 过滤
 *
 * 详见 docs/design/2026-06-17-skill-redesign-v2.md §2 / §5.
 *
 * 注：此文件是 Phase 2 新增。Phase 10 清理时旧 `skill-store.ts` 删除，
 * 本文件改名为 `skill-store.ts`。
 */

import type { DatabaseSync } from "node:sqlite";

import { randomBase62 } from "../../utils/short-id.js";
import { SKILLS_DDL, SKILL_FTS_DDL, SKILL_VEC_DDL_TEMPLATE, FTS_CONTENT_MAX } from "./skill-store-ddl.js";
import { buildFtsQuery, tokenizeForFts } from "../store/sqlite.js";
import type { ISkillStore, ExpiredVersionMeta, SkillStoreCapabilities, SkillSearchResult } from "./skill-store.interface.js";
import type {
  AppendVersionInput,
  ListSkillsOptions,
  SearchSkillsOptions,
  SkillManifestEntry,
  SkillStatus,
  Skill,
} from "./types.js";

// ═══════════════════════════════════════════════════════════════════════
//  错误类型
// ═══════════════════════════════════════════════════════════════════════

export type SkillErrorCode =
  | "SKILL_NAME_DUPLICATE"
  | "SKILL_NOT_FOUND";

export class SkillStoreError extends Error {
  constructor(public readonly code: SkillErrorCode, message?: string) {
    super(message ? `${code}: ${message}` : code);
    this.name = "SkillStoreError";
  }
}

/**
 * 当 appendVersion 的 content_hash 与当前 head 完全相同时抛出。
 * 由调用方决定如何处理（一般做幂等返回 head）。store 层不静默吞掉。
 */
export class IdempotentNoOpError extends Error {
  constructor(public readonly head: Skill) {
    super("IDEMPOTENT_NO_OP: content_hash unchanged");
    this.name = "IdempotentNoOpError";
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  Logger 接口
// ═══════════════════════════════════════════════════════════════════════

export interface StoreLogger {
  debug?(msg: string): void;
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}

// ═══════════════════════════════════════════════════════════════════════
//  Options
// ═══════════════════════════════════════════════════════════════════════

export interface SqliteSkillStoreOptions {
  db: DatabaseSync;
  /** Embedding 维度。0 = 不创建 skill_vec 虚拟表。 */
  dimensions: number;
  logger?: StoreLogger;
  /** 注入的 now（毫秒）。默认 Date.now。便于测试。 */
  now?: () => number;
  /** 注入的 ULID 生成器。默认见 mkUlid()。便于测试。 */
  ulid?: () => string;
}

// ═══════════════════════════════════════════════════════════════════════
//  内部辅助
// ═══════════════════════════════════════════════════════════════════════

/**
 * row_id 生成器 —— 每行的物理主键（TEXT PRIMARY KEY）。
 * 与 skill_id 分离：skill_id 在版本间共享，row_id 每一行唯一。
 * base62 12 字符（~71 bit CSPRNG 真熵）；无 base36 的伪随机隐患。
 */
function defaultUlid(): string {
  return randomBase62(12);
}

interface SkillRowRaw {
  row_id: string;
  skill_id: string;
  version: number;
  is_head: number;

  user_id: string;
  owner_agent_id: string;
  team_id: string;
  task_id: string;

  name: string;
  description: string;
  content: string;
  content_hash: string;
  manifest_json: string;
  storage_dir: string;

  status: string;
  metadata_json: string;
  created_at_ms: number;
  updated_at_ms: number;
}

function toSkill(raw: SkillRowRaw): Skill {
  let manifest: SkillManifestEntry[];
  try {
    manifest = JSON.parse(raw.manifest_json);
    if (!Array.isArray(manifest)) manifest = [];
  } catch {
    manifest = [];
  }
  return {
    row_id: raw.row_id,
    skill_id: raw.skill_id,
    version: raw.version,
    is_head: raw.is_head === 1,
    user_id: raw.user_id,
    owner_agent_id: raw.owner_agent_id,
    team_id: raw.team_id,
    task_id: raw.task_id,
    name: raw.name,
    description: raw.description,
    content: raw.content,
    content_hash: raw.content_hash,
    manifest,
    storage_dir: raw.storage_dir,
    status: raw.status as SkillStatus,
    metadata_json: raw.metadata_json,
    created_at_ms: raw.created_at_ms,
    updated_at_ms: raw.updated_at_ms,
  };
}

// ═══════════════════════════════════════════════════════════════════════
//  Store 实现
// ═══════════════════════════════════════════════════════════════════════

export class SqliteSkillStore implements ISkillStore {
  private readonly db: DatabaseSync;
  private readonly dimensions: number;
  private readonly logger?: StoreLogger;
  private readonly now: () => number;
  private readonly ulid: () => string;
  private vecAvailable = false;
  private degraded = false;

  constructor(opts: SqliteSkillStoreOptions) {
    this.db = opts.db;
    this.dimensions = Math.max(0, Math.floor(opts.dimensions ?? 0));
    this.logger = opts.logger;
    this.now = opts.now ?? (() => Date.now());
    this.ulid = opts.ulid ?? defaultUlid;
  }

  /** 创建表与索引。幂等。同时迁移旧索引与 FTS schema。 */
  init(): void {
    this.db.exec(SKILLS_DDL);
    // 迁移：删除旧版 (team_id, name) 唯一索引（v2 重构后改为 team_id + owner_agent_id + name）
    this.db.exec("DROP INDEX IF EXISTS uniq_skills_team_name_head");
    this.db.exec(SKILL_FTS_DDL);
    // 迁移：检测 skill_fts 是否缺少 owner_agent_id 列（旧 schema 只有 5 列）
    this.migrateFtsSchema();
    if (this.dimensions > 0) {
      try {
        const ddl = SKILL_VEC_DDL_TEMPLATE.replace(/__DIM__/g, String(this.dimensions));
        this.db.exec(ddl);
        this.vecAvailable = true;
      } catch (e) {
        this.logger?.warn(`[skill-store] vec0 init failed: ${(e as Error).message}; downgrade to bm25-only`);
        this.vecAvailable = false;
      }
    }
  }

  /**
   * 检测并迁移 skill_fts 表 schema。
   *
   * 问题背景：旧版 FTS DDL 只有 5 列（name, description, content, skill_id, team_id），
   * 新版增加了 owner_agent_id / task_id / user_id 三列。但 CREATE VIRTUAL TABLE IF NOT EXISTS
   * 不会修改已存在的表，导致新代码中按 8 列查询/写入时出现 "no such column" 错误。
   *
   * 迁移策略：检测缺少 owner_agent_id → DROP 旧 FTS 表 → 重建 → 从 skills 主表回填 head 行。
   * 这比 ALTER 安全（FTS5 不支持 ALTER），且不丢失任何数据（主表是唯一数据源）。
   */
  private migrateFtsSchema(): void {
    // 检测：如果 skill_fts 表不存在则跳过（首次启动会由 SKILL_FTS_DDL 自动创建）
    const tableCheck = this.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='skill_fts'")
      .get() as { name: string } | undefined;
    if (!tableCheck) return;

    // 检测现有 skill_fts 是否包含 owner_agent_id 列
    const cols = this.db
      .prepare("PRAGMA table_info('skill_fts')")
      .all() as Array<{ cid: number; name: string; type: string; notnull: number; dflt_value: unknown; pk: number }>;
    const colNames = new Set(cols.map((c) => c.name));

    // 新版需要 owner_agent_id / task_id / user_id，旧版只有 5 列缺这 3 个
    if (colNames.has("owner_agent_id")) return; // 已是新版 schema，无需迁移

    this.logger?.info("[skill-store] migrating skill_fts schema: old columns missing owner_agent_id/task_id/user_id");

    // Step 1: 从主表读取所有 head 行（回填用）
    interface HeadRow {
      skill_id: string;
      name: string;
      description: string;
      content: string;
      team_id: string;
      owner_agent_id: string;
      task_id: string;
      user_id: string;
    }
    const headRows = this.db
      .prepare("SELECT skill_id, name, description, content, team_id, owner_agent_id, task_id, user_id FROM skills WHERE is_head=1 AND status='active'")
      .all() as unknown as HeadRow[];

    // Step 2: DROP 旧表 → 重建新 schema（事务内原子操作）
    this.db.exec("BEGIN");
    try {
      this.db.exec("DROP TABLE IF EXISTS skill_fts");
      this.db.exec(SKILL_FTS_DDL);

      // Step 3: 回填所有 head 行到 FTS
      const insertStmt = this.db.prepare(
        "INSERT INTO skill_fts (name, description, content, skill_id, team_id, owner_agent_id, task_id, user_id) VALUES (?,?,?,?,?,?,?,?)",
      );
      for (const row of headRows) {
        const ftsContent = row.content.length > FTS_CONTENT_MAX
          ? row.content.slice(0, FTS_CONTENT_MAX)
          : row.content;
        insertStmt.run(
          tokenizeForFts(row.name),
          tokenizeForFts(row.description),
          tokenizeForFts(ftsContent),
          row.skill_id,
          row.team_id,
          row.owner_agent_id,
          row.task_id,
          row.user_id,
        );
      }

      this.db.exec("COMMIT");
      this.logger?.info(`[skill-store] fts migration done: rebuilt skill_fts with ${headRows.length} head rows`);
    } catch (e) {
      this.db.exec("ROLLBACK");
      this.logger?.error(`[skill-store] fts migration failed: ${(e as Error).message}`);
      // 即使迁移失败也不抛异常——gateway 仍可启动，只是 skill search 可能降级
      // 下次重启会重试（因为 skill_fts 被 drop 后 SKILL_FTS_DDL 会重建空表）
    }
  }

  /** 是否处于降级模式（SQLite 连接异常等） */
  isDegraded(): boolean {
    return this.degraded;
  }

  /** 获取 store 能力声明 */
  getCapabilities(): SkillStoreCapabilities {
    return {
      vectorSearch: this.vecAvailable,
      ftsSearch: true,
      nativeHybridSearch: false,
      sparseVectors: false,
    };
  }

  /** 关闭 store（SQLite 模式下为 no-op，连接由外部管理） */
  close(): void {
    this.degraded = true;
  }

  // ────────────────────────────────────────────────────────────────────
  //  appendVersion
  //
  //  注意：store 不再做 content_hash 的幂等检查 —— 这一层只管"写一行新版本"。
  //  幂等语义（content + manifest 都没变 → 不写新版本）由 skill-versioning 在
  //  上层判断，因为只有它同时知道 content_hash 与新 manifest_json。
  // ────────────────────────────────────────────────────────────────────
  async appendVersion(input: AppendVersionInput): Promise<Skill> {
    const tid = input.team_id ?? "default";
    const head = await this.getHead(input.skill_id, tid);

    // [4] 同 team 同 agent 同 name 已 active head（且不是同一 skill_id 的更新） → 重名
    if (!head) {
      const oid = input.owner_agent_id ?? "default";
      const dupRaw = this.db
        .prepare(
          "SELECT * FROM skills WHERE team_id=? AND owner_agent_id=? AND name=? AND is_head=1 AND status='active' LIMIT 1",
        )
        .get(tid, oid, input.name) as SkillRowRaw | undefined;
      if (dupRaw) {
        throw new SkillStoreError("SKILL_NAME_DUPLICATE", `name '${input.name}' already exists for agent in team`);
      }
    } else {
      // 已有同 skill_id 的历史 → name 不可变
      if (head.name !== input.name) {
        throw new SkillStoreError("SKILL_NAME_DUPLICATE", "name change is not allowed across versions");
      }
    }

    const newVersion = head ? head.version + 1 : 1;
    const ownerForRow = head ? head.owner_agent_id : (input.owner_agent_id ?? "default");
    // user_id 记录的是本次操作者，而非首次创建者。后续版本取 input.user_id。
    const userIdForRow = input.user_id ?? "default";
    const ts = this.now();
    const newRowId = this.ulid();

    // 事务：旧 head 翻 0 → INSERT 新行 → 同步 fts5
    // node:sqlite 暂无 db.transaction() helper，使用 BEGIN/COMMIT 手写。
    // 必须保持 synchronous（不要在中间 await），与 sqlite-transaction-guard 同思路。
    this.db.exec("BEGIN IMMEDIATE");
    try {
      if (head) {
        this.db
          .prepare("UPDATE skills SET is_head=0 WHERE skill_id=? AND version=?")
          .run(head.skill_id, head.version);
      }

      this.db
        .prepare(
          `INSERT INTO skills (
            row_id, skill_id, version, is_head,
            user_id, owner_agent_id, team_id, task_id,
            name, description, content, content_hash, manifest_json, storage_dir,
            status, metadata_json, created_at_ms, updated_at_ms
          ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?,?,?, ?,?,?,?)`,
        )
        .run(
          newRowId,
          input.skill_id,
          newVersion,
          1,
          userIdForRow,
          ownerForRow,
          tid,
          input.task_id ?? "default",
          input.name,
          input.description,
          input.content,
          input.content_hash,
          JSON.stringify(input.manifest ?? []),
          input.storage_dir,
          "active",
          input.metadata_json ?? "{}",
          ts,
          ts,
        );

      // FTS 同步：删除该 skill_id 的所有旧索引行 → 仅插入新 head
      this.db.prepare("DELETE FROM skill_fts WHERE skill_id=?").run(input.skill_id);
      const ftsContent = input.content.length > FTS_CONTENT_MAX
        ? input.content.slice(0, FTS_CONTENT_MAX)
        : input.content;
      // 使用 jieba 预分词（与 L0/L1 FTS 一致的方案），让 unicode61 tokenizer 能按空格切中文。
      // 如果 jieba 不可用，tokenizeForFts 会直接返回原文。
      const ftsName = tokenizeForFts(input.name);
      const ftsDescription = tokenizeForFts(input.description);
      const ftsContentTokenized = tokenizeForFts(ftsContent);
      this.db
        .prepare(
          "INSERT INTO skill_fts (name, description, content, skill_id, team_id, owner_agent_id, task_id, user_id) VALUES (?,?,?,?,?,?,?,?)",
        )
        .run(ftsName, ftsDescription, ftsContentTokenized, input.skill_id, tid, ownerForRow, input.task_id ?? "default", userIdForRow);

      this.db.exec("COMMIT");
    } catch (e) {
      try { this.db.exec("ROLLBACK"); } catch { /* ignore */ }
      throw e;
    }

    // 取回新行（事务外读取，节省持锁时间）
    const inserted = this.db
      .prepare("SELECT * FROM skills WHERE row_id=?")
      .get(newRowId) as SkillRowRaw;
    return toSkill(inserted);
  }

  // ────────────────────────────────────────────────────────────────────
  //  archiveHead
  // ────────────────────────────────────────────────────────────────────
  async archiveHead(skillId: string, teamId?: string): Promise<{ archived: boolean }> {
    const where = teamId ? "skill_id=? AND team_id=? AND is_head=1" : "skill_id=? AND is_head=1";
    const args: unknown[] = teamId ? [this.now(), skillId, teamId] : [this.now(), skillId];
    const r = this.db
      .prepare(`UPDATE skills SET status='archived', updated_at_ms=? WHERE ${where}`)
      .run(...args);

    // archived 后从 fts 中移除（不再可搜索到）
    if ((r.changes ?? 0) > 0) {
      this.db.prepare("DELETE FROM skill_fts WHERE skill_id=?").run(skillId);
      return { archived: true };
    }

    // 检查是否之前已 archived（仍算成功 / 幂等）
    const checkWhere = teamId
      ? "skill_id=? AND team_id=? AND is_head=1 AND status='archived'"
      : "skill_id=? AND is_head=1 AND status='archived'";
    const checkArgs = teamId ? [skillId, teamId] : [skillId];
    const exists = this.db.prepare(`SELECT 1 FROM skills WHERE ${checkWhere} LIMIT 1`).get(...checkArgs);
    return { archived: !!exists };
  }

  // ────────────────────────────────────────────────────────────────────
  //  getHead / getByVersion / listVersions
  // ────────────────────────────────────────────────────────────────────
  /**
   * 获取当前 head，且强制 `status='active'`。archived skill 视同不存在。
   *
   * 与 TCVDB 侧 `_getHeadAsync` 语义对齐（那边 filter 一直带 `status="active"`）。
   * 想拿到 archived 行请用 {@link getHeadIncludingArchived}。
   */
  async getHead(skillId: string, teamId?: string): Promise<Skill | null> {
    if (teamId) {
      const raw = this.db
        .prepare("SELECT * FROM skills WHERE skill_id=? AND team_id=? AND is_head=1 AND status='active' LIMIT 1")
        .get(skillId, teamId) as SkillRowRaw | undefined;
      return raw ? toSkill(raw) : null;
    }
    const raw = this.db
      .prepare("SELECT * FROM skills WHERE skill_id=? AND is_head=1 AND status='active' LIMIT 1")
      .get(skillId) as SkillRowRaw | undefined;
    return raw ? toSkill(raw) : null;
  }

  /**
   * 获取当前 head，不管 status。仅供 delete 幂等回读 / 补偿任务 / 管控台使用。
   * 普通读写路径 **不应** 调用本方法。
   */
  async getHeadIncludingArchived(skillId: string, teamId?: string): Promise<Skill | null> {
    if (teamId) {
      const raw = this.db
        .prepare("SELECT * FROM skills WHERE skill_id=? AND team_id=? AND is_head=1 LIMIT 1")
        .get(skillId, teamId) as SkillRowRaw | undefined;
      return raw ? toSkill(raw) : null;
    }
    const raw = this.db
      .prepare("SELECT * FROM skills WHERE skill_id=? AND is_head=1 LIMIT 1")
      .get(skillId) as SkillRowRaw | undefined;
    return raw ? toSkill(raw) : null;
  }

  async getByVersion(skillId: string, version: number, teamId?: string): Promise<Skill | null> {
    if (teamId) {
      const raw = this.db
        .prepare("SELECT * FROM skills WHERE skill_id=? AND version=? AND team_id=? LIMIT 1")
        .get(skillId, version, teamId) as SkillRowRaw | undefined;
      return raw ? toSkill(raw) : null;
    }
    const raw = this.db
      .prepare("SELECT * FROM skills WHERE skill_id=? AND version=? LIMIT 1")
      .get(skillId, version) as SkillRowRaw | undefined;
    return raw ? toSkill(raw) : null;
  }

  async listVersions(
    skillId: string,
    teamId?: string,
    opts: { limit?: number; offset?: number } = {},
  ): Promise<Skill[]> {
    const limit = Math.min(Math.max(opts.limit ?? 50, 1), 1000);
    const offset = Math.max(opts.offset ?? 0, 0);
    if (teamId) {
      const rows = this.db
        .prepare("SELECT * FROM skills WHERE skill_id=? AND team_id=? ORDER BY version DESC LIMIT ? OFFSET ?")
        .all(skillId, teamId, limit, offset) as SkillRowRaw[];
      return rows.map(toSkill);
    }
    const rows = this.db
      .prepare("SELECT * FROM skills WHERE skill_id=? ORDER BY version DESC LIMIT ? OFFSET ?")
      .all(skillId, limit, offset) as SkillRowRaw[];
    return rows.map(toSkill);
  }

  /** 该 skill_id 下的版本总数（team_id 可选过滤）。 */
  async countVersions(skillId: string, teamId?: string): Promise<number> {
    if (teamId) {
      const row = this.db
        .prepare("SELECT COUNT(*) AS c FROM skills WHERE skill_id=? AND team_id=?")
        .get(skillId, teamId) as { c: number };
      return row.c;
    }
    const row = this.db
      .prepare("SELECT COUNT(*) AS c FROM skills WHERE skill_id=?")
      .get(skillId) as { c: number };
    return row.c;
  }

  // ────────────────────────────────────────────────────────────────────
  //  listSkills
  // ────────────────────────────────────────────────────────────────────
  async listSkills(opts: ListSkillsOptions): Promise<{ items: Skill[]; total: number }> {
    const status = opts.status?.length ? opts.status : (["active"] as SkillStatus[]);
    const limit = Math.min(Math.max(opts.limit ?? 50, 1), 1000);
    const offset = Math.max(opts.offset ?? 0, 0);

    const where: string[] = ["is_head=1"];
    const args: unknown[] = [];

    // 四个 ID：传了就过滤，不传不限制
    if (opts.team_id) {
      where.push("team_id=?");
      args.push(opts.team_id);
    }
    if (opts.owner_agent_id) {
      where.push("owner_agent_id=?");
      args.push(opts.owner_agent_id);
    }
    if (opts.user_id) {
      where.push("user_id=?");
      args.push(opts.user_id);
    }
    if (opts.task_id) {
      where.push("task_id=?");
      args.push(opts.task_id);
    }

    where.push(`status IN (${status.map(() => "?").join(",")})`);
    args.push(...status);

    if (opts.name_prefix) {
      where.push("name LIKE ?");
      args.push(`${opts.name_prefix}%`);
    }

    const whereSql = where.join(" AND ");

    const totalRow = this.db.prepare(`SELECT COUNT(*) AS c FROM skills WHERE ${whereSql}`).get(...args) as { c: number };
    const rows = this.db
      .prepare(`SELECT * FROM skills WHERE ${whereSql} ORDER BY updated_at_ms DESC LIMIT ? OFFSET ?`)
      .all(...args, limit, offset) as SkillRowRaw[];

    return { items: rows.map(toSkill), total: totalRow.c };
  }

  // ────────────────────────────────────────────────────────────────────
  //  searchSkills（仅 BM25 实现；vec 部分由 hybrid 调用方组合）
  // ────────────────────────────────────────────────────────────────────
  async searchSkills(opts: SearchSkillsOptions): Promise<SkillSearchResult[]> {
    const topK = Math.min(Math.max(opts.topK ?? 10, 1), 50);
    const query = (opts.query ?? "").trim();
    if (!query) return [];

    // mode 透传：当前 store 仅实现 BM25 路径。
    // - 'bm25' / 未传  → 直接走 BM25（默认）
    // - 'embedding' / 'hybrid' 但 vec 不可用或未给 queryEmbedding → 降级到 BM25 + 一条 warn
    // 真实 hybrid (RRF) / 纯 vec 路径是后置项；契约层面 mode 不会被静默吞掉。
    const requestedMode = opts.mode ?? "bm25";
    const wantsVec = requestedMode === "embedding" || requestedMode === "hybrid";
    if (wantsVec && (!this.vecAvailable || !opts.queryEmbedding)) {
      this.logger?.warn(
        `[skill-store] search mode='${requestedMode}' downgraded to 'bm25' ` +
          `(vec_available=${this.vecAvailable}, has_embedding=${!!opts.queryEmbedding})`,
      );
    }
    // pure embedding 路径暂未实现 → 仍回 BM25；hybrid 同样回 BM25（后续 RRF 融合）。

    // FTS5 查询：使用 buildFtsQuery（与 L0/L1 一致的 jieba 分词 + 引号包裹 + OR 连接）。
    // jieba cutForSearch 能正确处理中文分词；fallback 到 Unicode 正则切分。
    // 每个 token 用双引号包裹，避免 FTS5 保留词（AND/OR/NOT/NEAR）被误解析为布尔运算符。
    const ftsQuery = buildFtsQuery(query);
    if (!ftsQuery) return [];

    // FTS5 snippet(table, col, prefix, suffix, ellipsis, tokenCount)
    // col=2 → skill_fts.content（DDL 列序：name=0, description=1, content=2, skill_id=3, team_id=4, owner_agent_id=5, task_id=6, user_id=7）。
    // 16 token 片段，带 <mark> 高亮。
    //
    // 四个 ID 在 FTS5 层直接过滤（传了就加），避免先搜全集再回查过滤的低效路径。
    let ftsRows: Array<{ skill_id: string; bm25: number; snippet: string }>;
    try {
      const ftsArgs: Array<string | number> = [ftsQuery];
      let ftsWhere = "skill_fts MATCH ?";
      if (opts.team_id) {
        ftsWhere += " AND team_id=?";
        ftsArgs.push(opts.team_id);
      }
      if (opts.agent_id) {
        ftsWhere += " AND owner_agent_id=?";
        ftsArgs.push(opts.agent_id);
      }
      if (opts.task_id) {
        ftsWhere += " AND task_id=?";
        ftsArgs.push(opts.task_id);
      }
      if (opts.user_id) {
        ftsWhere += " AND user_id=?";
        ftsArgs.push(opts.user_id);
      }
      ftsArgs.push(topK * 2);
      ftsRows = this.db
        .prepare(
          `SELECT skill_id,
                  bm25(skill_fts) AS bm25,
                  snippet(skill_fts, 2, '<mark>', '</mark>', '…', 16) AS snippet
           FROM skill_fts
           WHERE ${ftsWhere}
           ORDER BY bm25 LIMIT ?`,
        )
        .all(...ftsArgs) as Array<{ skill_id: string; bm25: number; snippet: string }>;
    } catch (e) {
      this.logger?.warn(`[skill-store] fts query failed: ${(e as Error).message}`);
      return [];
    }

    // 回查主表：验证 is_head=1 AND status='active'（FTS5 没有这两个字段）
    const hits: Array<{ skill: Skill; score: number; snippet: string }> = [];
    for (const r of ftsRows) {
      const row = this.db
        .prepare(
          `SELECT * FROM skills WHERE skill_id=? AND is_head=1 AND status='active' LIMIT 1`,
        )
        .get(r.skill_id) as SkillRowRaw | undefined;
      if (!row) continue;
      // bm25 越小越相关 → 转为越大越好的 score
      hits.push({
        skill: toSkill(row),
        score: -r.bm25,
        snippet: r.snippet ?? "",
      });
      if (hits.length >= topK) break;
    }
    return hits;
  }

  // ────────────────────────────────────────────────────────────────────
  //  TTL 清理
  // ────────────────────────────────────────────────────────────────────

  /** 查询 created_at_ms < cutoffMs 的过期非 head 版本。 */
  async findExpiredVersions(cutoffMs: number): Promise<ExpiredVersionMeta[]> {
    const rows = this.db
      .prepare(
        `SELECT skill_id, version, is_head, status, storage_dir, created_at_ms
         FROM skills WHERE is_head=0 AND status='active' AND created_at_ms < ?
         ORDER BY skill_id ASC, version ASC`,
      )
      .all(cutoffMs) as Array<{
        skill_id: string;
        version: number;
        is_head: number;
        status: string;
        storage_dir: string;
        created_at_ms: number;
      }>;
    return rows.map((r) => ({
      skill_id: r.skill_id,
      version: r.version,
      is_head: r.is_head === 1,
      status: r.status as SkillStatus,
      storage_dir: r.storage_dir,
      created_at_ms: r.created_at_ms,
    }));
  }

  /** 物理删除指定版本行（仅 is_head=0 安全锁）。 */
  async deleteVersion(skillId: string, version: number): Promise<boolean> {
    const r = this.db
      .prepare("DELETE FROM skills WHERE skill_id=? AND version=? AND is_head=0")
      .run(skillId, version);
    return (r.changes ?? 0) > 0;
  }

  /**
   * 物理删除同 skill_id 下的所有版本行（含 head + archived）+ 清 fts / vec。
   * 返回实际删除的行数。`SkillCore.delete` 走此路径，权限校验由调用方负责。
   */
  async deleteAllVersions(skillId: string, teamId?: string): Promise<number> {
    const where = teamId ? "skill_id=? AND team_id=?" : "skill_id=?";
    const args: unknown[] = teamId ? [skillId, teamId] : [skillId];
    const r = this.db.prepare(`DELETE FROM skills WHERE ${where}`).run(...args);
    const changes = r.changes ?? 0;
    // 仅当主表真的删掉了行时才 DELETE 附属表 —— 避免跨 team 校验失败时误清 fts
    if (changes > 0) {
      try {
        this.db.prepare("DELETE FROM skill_fts WHERE skill_id=?").run(skillId);
      } catch {
        // fts 可能已被 archiveHead 清过，二次删除幂等
      }
      if (this.vecAvailable) {
        try {
          this.db.prepare("DELETE FROM skill_vec WHERE skill_id=?").run(skillId);
        } catch {
          /* non-fatal */
        }
      }
    }
    return changes;
  }

  // ────────────────────────────────────────────────────────────────────
  //  Embedding 维护
  // ────────────────────────────────────────────────────────────────────
  upsertEmbedding(skillId: string, embedding: Float32Array): void {
    if (!this.vecAvailable) return;
    if (embedding.length !== this.dimensions) {
      this.logger?.warn(`[skill-store] embedding dim mismatch: ${embedding.length} vs ${this.dimensions}`);
      return;
    }
    try {
      this.db.prepare("DELETE FROM skill_vec WHERE skill_id=?").run(skillId);
      this.db.prepare("INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)").run(skillId, Buffer.from(embedding.buffer));
    } catch (e) {
      this.logger?.warn(`[skill-store] upsertEmbedding failed: ${(e as Error).message}`);
    }
  }

  deleteEmbedding(skillId: string): void {
    if (!this.vecAvailable) return;
    try {
      this.db.prepare("DELETE FROM skill_vec WHERE skill_id=?").run(skillId);
    } catch (e) {
      this.logger?.warn(`[skill-store] deleteEmbedding failed: ${(e as Error).message}`);
    }
  }
}
