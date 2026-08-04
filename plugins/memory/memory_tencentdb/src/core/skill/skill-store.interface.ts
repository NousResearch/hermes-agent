/**
 * ISkillStore — Skill 存储层抽象接口
 *
 * 用于解耦 SkillCore / SkillVersioning 与具体存储实现（SQLite / TCVDB）。
 * SqliteSkillStore（Standalone 模式）和 TcvdbSkillStore（Service 模式）
 * 各自实现本接口，消费者只依赖接口。
 *
 * 设计文档：
 *   - docs/design/2026-06-25-skill-service-mode-design.md §6.1
 *   - docs/design/2026-06-17-skill-redesign-v2.md
 */

import type {
  AppendVersionInput,
  ListSkillsOptions,
  SearchSkillsOptions,
  Skill,
  SkillStatus,
} from "./types.js";

// ─── Capabilities ──────────────────────────────────────────────────────────

/** Store 能力声明，用于检索降级判断 */
export interface SkillStoreCapabilities {
  /** 密集向量搜索可用 */
  vectorSearch: boolean;
  /** BM25 稀疏向量搜索可用 */
  ftsSearch: boolean;
  /** TCVDB 原生 hybridSearch 可用（本地模式 false） */
  nativeHybridSearch: boolean;
  /** 稀疏向量支持 */
  sparseVectors: boolean;
}

// ─── Search Result ─────────────────────────────────────────────────────────

export interface SkillSearchResult {
  skill: Skill;
  score: number;
  snippet?: string;
}

// ─── TTL Cleanup Meta ──────────────────────────────────────────────────────

/** TTL 清理用的过期版本元信息（轻量，不读 content/manifest）。 */
export interface ExpiredVersionMeta {
  skill_id: string;
  version: number;
  is_head: boolean;
  status: SkillStatus;
  storage_dir: string;
  created_at_ms: number;
}

// ─── Store Interface ───────────────────────────────────────────────────────

export interface ISkillStore {
  // ── 生命周期 ──
  /** 初始化存储（建表/建 Collection 等）。 */
  init(): void;
  /** 是否处于降级模式（不可用） */
  isDegraded(): boolean;
  /** 获取 store 能力声明 */
  getCapabilities(): SkillStoreCapabilities;
  /** 关闭存储（释放连接等） */
  close(): void;

  // ── CRUD ──
  /** 追加一个版本行。store 不负责幂等校验（由上层 SkillVersioning 处理）。 */
  appendVersion(input: AppendVersionInput): Promise<Skill>;
  /**
   * 获取当前 head 版本（is_head=1 且 status='active'）。
   *
   * 语义：`archived` 视同"逻辑删除"，对外任何普通读接口都不可见——本方法一律返回 null。
   * 需要看到 archived head 用 {@link getHeadIncludingArchived}（仅供 `SkillCore.delete`
   * 幂等回读、TTL cleaner、管控台等内部路径使用）。
   */
  getHead(skillId: string, teamId?: string): Promise<Skill | null>;
  /**
   * 获取当前 head 版本，包含 archived。
   *
   * 仅供内部使用：
   *   - `SkillCore.delete` 需要拿到 archived head 才能实现幂等 `{ archived: true }`
   *   - 后台补偿任务扫描 archived skill 与 asset 漂移
   *   - 管控台"回收站"视图
   *
   * 普通读/写路径 **不应** 调用本方法——用 `getHead`。
   */
  getHeadIncludingArchived(skillId: string, teamId?: string): Promise<Skill | null>;
  /** 获取指定版本行 */
  getByVersion(skillId: string, version: number, teamId?: string): Promise<Skill | null>;
  /** 将 head 标记为 archived（软删） */
  archiveHead(skillId: string, teamId?: string): Promise<{ archived: boolean }>;

  // ── 查询 ──
  /** 列出 head 行，支持五元组过滤 + 分页 */
  listSkills(opts: ListSkillsOptions): Promise<{ items: Skill[]; total: number }>;
  /** 搜索 skill（BM25 / embedding / hybrid，由实现决定） */
  searchSkills(opts: SearchSkillsOptions): Promise<SkillSearchResult[]>;
  /** 列出某 skill 的全部版本（DESC） */
  listVersions(skillId: string, teamId?: string, pagination?: { limit?: number; offset?: number }): Promise<Skill[]>;
  /** 某 skill 的版本总数 */
  countVersions(skillId: string, teamId?: string): Promise<number>;

  // ── TTL Cleanup ──
  /** 查询 created_at_ms < cutoffMs 的过期非 head 版本（跨 team 全量扫描）。 */
  findExpiredVersions(cutoffMs: number): Promise<ExpiredVersionMeta[]>;
  /** 物理删除指定版本行（仅 is_head=0）。返回是否实际删除了行。 */
  deleteVersion(skillId: string, version: number): Promise<boolean>;
  /**
   * 物理删除同 skill_id 下的**所有版本行**（含 head + archived）。
   * 返回实际删除的行数（可能为 0：skill 不存在 / team 不匹配）。
   *
   * 用于 `SkillCore.delete` 的真删除路径，与 `deleteVersion` 的 head 保护相反：
   * 这里承担"以 skill 为整体单位一次性清空"的语义，权限校验由调用方
   * （SkillCore）先完成。
   *
   * 语义：
   *   - `teamId` 传入 → WHERE 强制过滤，跨 team 不生效（返回 0）
   *   - `teamId` 省略 → 跨 team 删除（供管控台 / 后台补偿任务使用，业务路径不应调用）
   */
  deleteAllVersions(skillId: string, teamId?: string): Promise<number>;
}
