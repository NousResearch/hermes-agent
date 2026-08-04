/**
 * SkillCore — 6 个 manage action 的编排门面
 *
 * 编排逻辑：
 *   1. 解析 + 校验 SKILL.md（frontmatter）
 *   2. 取 head（如有）
 *   3. assertTeamMatch / assertOwner / assertVersionFresh
 *   4. 调 SkillVersioning.appendNextVersion / createNewSkill
 *
 * 6 个写动作：
 *   - create        新建 skill v1
 *   - update        替换 SKILL.md
 *   - patch         单点串替
 *   - delete        head status=archived
 *   - writeFiles    增/改资源
 *   - removeFiles   删资源
 *
 * 4 个读动作：
 *   - get           返回 detail（默认 head；可指定 version）
 *   - list          按 team_id + filters 返回 head 行
 *   - search        FTS 命中
 *   - listVersions  历史版本元信息
 *   - readFile      读资源字节
 */

import { parseSkillFile, validateSkillFile } from "./skill-format.js";
import { SkillResourceStore, type SkillResourcePayload } from "./skill-resource-store.js";
import type { ISkillStore, SkillSearchResult } from "./skill-store.interface.js";
import { SkillVersioning } from "./skill-versioning.js";
import { randomBase62 } from "../../utils/short-id.js";
import {
  SkillPermissionError,
  assertOwner,
  assertTeamMatch,
  assertVersionFresh,
} from "./skill-permission.js";
import type {
  IdFields,
  ListSkillsOptions,
  SearchSkillsOptions,
  SkillStatus,
  Skill,
  SkillSimilarityResult,
  SkillProposeResult,
} from "./types.js";


const TAG = "[skill-core]";

// ═════════════════════════════════════════════════════════════════════
//  错误类型（用于 gateway 映射 HTTP 错误码）
// ═════════════════════════════════════════════════════════════════════


export type SkillCoreErrorCode =
  | "INVALID_FRONTMATTER"
  | "SKILL_FRONTMATTER_INVALID"
  | "SKILL_PATCH_NOT_UNIQUE"
  | "SKILL_NAME_DUPLICATE"
  | "SKILL_NOT_OWNER"
  | "SKILL_TEAM_MISMATCH"
  | "SKILL_NOT_FOUND"
  | "SKILL_VERSION_STALE"
  | "SKILL_VERSION_EXPIRED"
  | "SKILL_ID_COLLISION"
  | "INVALID_PATH"
  | "RESOURCE_TOO_LARGE"
  | "STORAGE_NOT_FOUND"
  | "LLM_UNAVAILABLE"
  | "SKILL_COS_REQUIRED";

export class SkillCoreError extends Error {
  constructor(public readonly code: SkillCoreErrorCode, message?: string) {
    super(message ? `${code}: ${message}` : code);
    this.name = "SkillCoreError";
  }
}

// 工具：把下层抛的各类错误统一翻译为 SkillCoreError（保留原 message）
function toCoreError(e: unknown): never {
  if (e instanceof SkillCoreError) throw e;
  const code = (e as { code?: string }).code as SkillCoreErrorCode | undefined;
  const msg = (e as Error).message;
  if (code) {
    throw new SkillCoreError(code, msg);
  }
  throw e as Error;
}

// ═════════════════════════════════════════════════════════════════════
//  Options
// ═════════════════════════════════════════════════════════════════════

export interface SkillCoreOptions {
  store: ISkillStore;
  resources: SkillResourceStore;
  versioning: SkillVersioning;
  /**
   * 用于 skill_id 生成。默认 `skl-` + 12 字符 base62（CSPRNG，71 bit 真熵）。
   * 保持与老 sid 相同长度 (16 字符)，仅字符集从 base36 扩到 base62 且用真随机源。
   * 测试可注入固定值。
   */
  ulid?: () => string;
  /** Date.now 的注入。默认 Date.now。 */
  now?: () => number;
  /** 旧版本 TTL 秒数。0 = 关闭。 */
  versionTtlSeconds?: number;
  /**
   * `delete` 成功归档 head 后同步触发。fire-and-forget：钩子抛异常
   * 会被吞掉，不影响 delete 返回值（asset 状态漂移可容忍：skill 已经
   * archived，asset 慢一步同步不影响业务）。
   *
   * 与 `SkillVersioning.onSkillCreated` 成对：一个负责 v1 登记，一个
   * 负责整 skill 归档，二者共同覆盖 asset 生命周期两端。
   */
  onSkillArchived?: (params: { skill_id: string; team_id?: string }) => void;
  /**
   * 读路径自愈补登记钩子。
   *
   * 触发时机：`get` / `readFile` 成功返回单个 skill 之后。
   * 不触发：`list` / `search` / `listing` / `listVersions`（浏览类，一次 N 条，
   * 走 LRU 也偏贵；且这些接口不必然代表"使用"）。
   *
   * 契约：
   *  - fire-and-forget：抛异常吞掉，不影响 read 的返回
   *  - 上层实现须幂等且带 LRU（同一 skill_id 只有首次真正查 store）
   *  - 用途：兜底修复 asset 缺失（历史数据 / 迁移遗漏 / 人工误删），
   *    保证下次前端管控页能看到这个 skill
   */
  onSkillAccessed?: (skill: Skill) => void;
}

// 各 action 入参类型（四个 ID 全部可选）
export interface CreateInput extends IdFields {
  name: string;
  content: string;
  resources?: SkillResourcePayload[];
  metadata?: Record<string, unknown>;
}

export interface UpdateInput extends IdFields {
  skill_id: string;
  expected_version: number;
  content: string;
}

export interface PatchInput extends IdFields {
  skill_id: string;
  expected_version: number;
  old_string: string;
  new_string: string;
  replace_all?: boolean;
}

export interface DeleteInput extends IdFields {
  skill_id: string;
  expected_version: number;
}

export interface GetInput extends IdFields {
  skill_id: string;
  version?: number;
  include_content?: boolean;
  include_manifest?: boolean;
}

export interface WriteFilesInput extends IdFields {
  skill_id: string;
  expected_version: number;
  files: SkillResourcePayload[];
}

export interface RemoveFilesInput extends IdFields {
  skill_id: string;
  expected_version: number;
  paths: string[];
}

export interface ReadFileInput extends IdFields {
  skill_id: string;
  version?: number;
  path: string;
  encoding?: "utf-8" | "base64";
}

export interface ListInput extends IdFields {
  filters?: {
    owner_agent_id?: string;
    name_prefix?: string;
    status?: SkillStatus[];
  };
  pagination?: { limit?: number; offset?: number };
}

export interface SearchInput extends IdFields {
  query: string;
  top_k?: number;
  mode?: "bm25" | "embedding" | "hybrid";
}

export interface ListVersionsInput extends IdFields {
  skill_id: string;
  pagination?: { limit?: number; offset?: number };
}

// ═════════════════════════════════════════════════════════════════════
//  Implementation
// ═════════════════════════════════════════════════════════════════════

export class SkillCore {
  private readonly store: ISkillStore;
  private readonly resources: SkillResourceStore;
  private readonly versioning: SkillVersioning;
  private readonly ulid: () => string;
  private readonly now: () => number;
  private readonly versionTtlSeconds: number;
  private readonly onSkillArchived?: SkillCoreOptions["onSkillArchived"];
  private readonly onSkillAccessed?: SkillCoreOptions["onSkillAccessed"];

  constructor(opts: SkillCoreOptions) {
    this.store = opts.store;
    this.resources = opts.resources;
    this.versioning = opts.versioning;
    // 默认 sid = `skl-` + 12 字符 base62（CSPRNG，~71 bit 真熵）；总长 16。
    // 与旧 `skl-${Math.random().toString(36).slice(2,14)}` 相同长度，
    // 碰撞概率从单实例 100 万条时的 ~1.1e-4 降至 ~1.5e-10（详见 utils/short-id.ts）。
    this.ulid = opts.ulid ?? (() => `skl-${randomBase62(12)}`);
    this.now = opts.now ?? (() => Date.now());
    this.versionTtlSeconds = opts.versionTtlSeconds ?? 0;
    this.onSkillArchived = opts.onSkillArchived;
    this.onSkillAccessed = opts.onSkillAccessed;
  }

  /** 读路径读到具体 skill 后 fire。异常吞掉，不阻塞读。 */
  private notifyAccessed(skill: Skill): void {
    if (!this.onSkillAccessed) return;
    try { this.onSkillAccessed(skill); } catch { /* swallow */ }
  }

  // ───────────────────────────────────────────────────────────────────
  //  WRITE actions
  // ───────────────────────────────────────────────────────────────────

  async create(input: CreateInput): Promise<Skill> {
    // 1) parse + validate
    const file = this.parseAndValidate(input.content);
    if (file.frontmatter.name !== input.name) {
      throw new SkillCoreError("INVALID_FRONTMATTER", `frontmatter.name '${file.frontmatter.name}' != body.name '${input.name}'`);
    }

    // 2) 生成 sid 并做碰撞检测
    //
    // 背景：默认 ulid 走 CSPRNG base62 12 字符（~71 bit 真熵，单实例 100 万
    // skill 时碰撞概率 ~1.5e-10），工程上"永远不会撞"；但仍加一层 preflight
    // 防御，把"撞了静默覆盖"变成"撞了 retry"。为什么不上 DB UNIQUE 约束：
    //   - SQLite skills 表 UNIQUE(skill_id, version) 已存在，v1 碰撞会被物理挡住
    //   - TCVDB 主键是 row_id（每行唯一），无法给 skill_id 加"仅 v1 唯一"约束
    //     （skill 天生多版本，version 2/3 就是同 skill_id 共存）
    // → 应用层 preflight 是唯一可移植到两种 store 的方案。
    //
    // 注：注入的 ulid 工厂可能不带 'skl-' 前缀，这里兜底拼上。
    const MAX_ID_ATTEMPTS = 3;
    let sid = "";
    for (let attempt = 1; attempt <= MAX_ID_ATTEMPTS; attempt++) {
      const u = this.ulid();
      sid = u.startsWith("skl-") ? u : `skl-${u}`;

      // 全 team 范围查（不带 team_id）：只要 skill_id 全局撞了就重试。
      // 用 getHeadIncludingArchived 覆盖 archived 行——归档不代表 sid 空闲，
      // 版本表 UNIQUE(skill_id, version) 仍会挡住写入。
      const existing = await this.store.getHeadIncludingArchived(sid);
      if (!existing) break;

      if (attempt >= MAX_ID_ATTEMPTS) {
        // 连续 3 次撞 —— 只可能是 ulid 注入器坏了（比如测试里固定返回同一值）
        // 或熵源崩了，不是概率事件，直接抛。
        throw new SkillCoreError(
          "SKILL_ID_COLLISION",
          `failed to generate a unique skill_id after ${MAX_ID_ATTEMPTS} attempts`,
        );
      }
    }

    try {
      return await this.versioning.createNewSkill(
        sid,
        input.agent_id ?? "default",
        { user_id: input.user_id, team_id: input.team_id, agent_id: input.agent_id, task_id: input.task_id },
        {
          content: input.content,
          name: input.name,
          description: file.frontmatter.description,
          resourcesToWrite: input.resources,
          metadata_json: input.metadata ? JSON.stringify(input.metadata) : undefined,
        },
      );
    } catch (e) {
      toCoreError(e);
    }
  }

  async update(input: UpdateInput): Promise<Skill> {
    const head = await this.requireHead(input.skill_id, input.team_id);
    if (input.agent_id) assertOwnerWrap(head, input.agent_id, input.team_id);
    assertVersionFreshWrap(head, input.expected_version);

    const file = this.parseAndValidate(input.content);
    if (file.frontmatter.name !== head.name) {
      throw new SkillCoreError("INVALID_FRONTMATTER", "name change is not allowed across versions");
    }

    try {
      const result = await this.versioning.appendNextVersion(head, this.ctxOf(input), {
        content: input.content,
        name: head.name,
        description: file.frontmatter.description,
      });
      void this.versioning.cleanupExpiredVersionsForSkill(
        head.skill_id, this.versionTtlSeconds,
      ).catch(() => { /* fire-and-forget */ });
      return result;
    } catch (e) {
      toCoreError(e);
    }
  }

  async patch(input: PatchInput): Promise<Skill> {
    const head = await this.requireHead(input.skill_id, input.team_id);
    if (input.agent_id) assertOwnerWrap(head, input.agent_id, input.team_id);
    assertVersionFreshWrap(head, input.expected_version);

    // count occurrences
    const occ = countOccurrences(head.content, input.old_string);
    if (occ === 0) {
      throw new SkillCoreError("SKILL_PATCH_NOT_UNIQUE", `old_string not found`);
    }
    if (occ > 1 && !input.replace_all) {
      throw new SkillCoreError("SKILL_PATCH_NOT_UNIQUE", `old_string occurs ${occ} times; pass replace_all=true to replace all`);
    }

    const newContent = input.replace_all
      ? splitJoin(head.content, input.old_string, input.new_string)
      : head.content.replace(input.old_string, input.new_string);

    // re-parse + validate
    const file = this.parseAndValidate(newContent);
    if (file.frontmatter.name !== head.name) {
      throw new SkillCoreError("INVALID_FRONTMATTER", "patch attempted to rename skill");
    }

    try {
      const result = await this.versioning.appendNextVersion(head, this.ctxOf(input), {
        content: newContent,
        name: head.name,
        description: file.frontmatter.description,
      });
      void this.versioning.cleanupExpiredVersionsForSkill(
        head.skill_id, this.versionTtlSeconds,
      ).catch(() => { /* fire-and-forget */ });
      return result;
    } catch (e) {
      toCoreError(e);
    }
  }

  async delete(input: DeleteInput): Promise<{ skill_id: string; archived: boolean }> {
    // 语义：物理真删除（2026-07 变更，原为软删）。
    // - head 不存在（skill 不存在 / 已被删）→ SKILL_NOT_FOUND
    // - 用 getHeadIncludingArchived 兼容历史遗留 archived 行：老数据里可能还有
    //   未被清理的 archived head（旧软删语义留下的），此时 delete 应视为"补物理删"
    //   而不是 404。
    const head = await this.store.getHeadIncludingArchived(input.skill_id, input.team_id);
    if (input.team_id) assertTeamMatchWrap(head, input.team_id);
    if (!head) throw new SkillCoreError("SKILL_NOT_FOUND");
    if (input.agent_id) assertOwnerWrap(head, input.agent_id, input.team_id);
    assertVersionFreshWrap(head, input.expected_version);

    // 物理删除所有版本 + 清 storage + 汇总上报 shark(-N)
    const deleted = await this.versioning.deleteSkill(input.skill_id, input.team_id);

    // fire-and-forget：asset 状态同步失败不回滚 delete
    // deleted > 0 才触发 —— 与 store.deleteAllVersions 语义对齐
    if (deleted > 0 && this.onSkillArchived) {
      try { this.onSkillArchived({ skill_id: input.skill_id, team_id: input.team_id }); }
      catch { /* swallow */ }
    }

    // 返回结构保持 wire 兼容：archived=true 表示"已完成删除"（真删也复用该字段）
    return { skill_id: input.skill_id, archived: deleted > 0 };
  }

  async writeFiles(input: WriteFilesInput): Promise<Skill> {
    const head = await this.requireHead(input.skill_id, input.team_id);
    if (input.agent_id) assertOwnerWrap(head, input.agent_id, input.team_id);
    assertVersionFreshWrap(head, input.expected_version);

    try {
      const result = await this.versioning.appendNextVersion(head, this.ctxOf(input), {
        content: head.content,
        name: head.name,
        description: head.description,
        resourcesToWrite: input.files,
      });
      void this.versioning.cleanupExpiredVersionsForSkill(
        head.skill_id, this.versionTtlSeconds,
      ).catch(() => { /* fire-and-forget */ });
      return result;
    } catch (e) {
      toCoreError(e);
    }
  }

  async removeFiles(input: RemoveFilesInput): Promise<Skill> {
    const head = await this.requireHead(input.skill_id, input.team_id);
    if (input.agent_id) assertOwnerWrap(head, input.agent_id, input.team_id);
    assertVersionFreshWrap(head, input.expected_version);

    // 过滤出真实存在于 head manifest 中的 path（避免无效的资源变更触发空 v+1）
    const manifestPaths = new Set(head.manifest.map((m) => m.path));
    const toRemove = input.paths.filter((p) => manifestPaths.has(p));

    if (toRemove.length === 0) {
      return head; // 幂等
    }

    try {
      const result = await this.versioning.appendNextVersion(head, this.ctxOf(input), {
        content: head.content,
        name: head.name,
        description: head.description,
        resourcesToRemove: toRemove,
      });
      void this.versioning.cleanupExpiredVersionsForSkill(
        head.skill_id, this.versionTtlSeconds,
      ).catch(() => { /* fire-and-forget */ });
      return result;
    } catch (e) {
      toCoreError(e);
    }
  }

  // ───────────────────────────────────────────────────────────────────
  //  READ actions
  // ───────────────────────────────────────────────────────────────────

  async get(input: GetInput): Promise<Skill> {
    if (typeof input.version === "number") {
      // 指定版本查询：先确认 skill 存在（通过 head），再查指定版本
      const head = await this.store.getHead(input.skill_id, input.team_id);
      if (input.team_id) assertTeamMatchWrap(head, input.team_id);
      if (!head) throw new SkillCoreError("SKILL_NOT_FOUND");
      const row = await this.store.getByVersion(input.skill_id, input.version, input.team_id);
      if (!row) {
        throw new SkillCoreError(
          "SKILL_NOT_FOUND",
          `version ${input.version} not found (may have been GC'd); current head is v${head.version}`,
        );
      }
      this.assertVersionNotExpired(row, head.version);
      // 读时自愈：即使拿的是历史版本，也用 head 触发（asset 只认 skill_id）
      this.notifyAccessed(head);
      return row;
    }
    // 不传 version → 返回最新 head
    const row = await this.store.getHead(input.skill_id, input.team_id);
    if (input.team_id) assertTeamMatchWrap(row, input.team_id);
    if (!row) throw new SkillCoreError("SKILL_NOT_FOUND");
    this.notifyAccessed(row);
    return row;
  }

  async list(input: ListInput): Promise<{ items: Skill[]; total: number }> {
    const opts: ListSkillsOptions = {
      team_id: input.team_id,
      owner_agent_id: input.filters?.owner_agent_id ?? input.agent_id,
      // user_id is an audit column (who wrote the skill), NOT an ownership filter.
      // When team_id is present, skills are team-shared — filtering by user_id
      // would exclude skills written by other team members.
      // Only apply user_id filter when there is no team_id (personal scope).
      user_id: input.team_id ? undefined : input.user_id,
      // task_id 与 user_id 同性质，是写审计字段（记 skill 首次落库时的对话上下文），
      // 不参与"哪些 skill 可用"的检索。传了会造成抽取器每次新对话（新 task_id）
      // 都看不到已有 skill → LLM 走 skill_create → 撞 SKILL_NAME_DUPLICATE →
      // 加后缀 (foo-v2, foo-v3, …) 产生大量同族 skill_id。
      // Store 层保留按 task_id 过滤的能力（供审计接口显式使用）。
      task_id: undefined,
      name_prefix: input.filters?.name_prefix,
      status: input.filters?.status,
      limit: input.pagination?.limit,
      offset: input.pagination?.offset,
    };
    return this.store.listSkills(opts);
  }

  async search(input: SearchInput): Promise<SkillSearchResult[]> {
    const opts: SearchSkillsOptions = {
      team_id: input.team_id,
      query: input.query,
      topK: input.top_k,
      mode: input.mode,
      agent_id: input.agent_id,
      // Same rationale as list(): task_id is audit-only, not a read filter.
      // See list() 上的详细注释。
      task_id: undefined,
      // Same rationale as list(): user_id is audit-only, not a read filter
      // when team-scoped. Skills are team-shared assets.
      user_id: input.team_id ? undefined : input.user_id,
    };
    return this.store.searchSkills(opts);
  }

  async listVersions(input: ListVersionsInput): Promise<{
    items: Array<Skill & { is_expired: boolean }>;
    total: number;
  }> {
    const items = await this.store.listVersions(input.skill_id, input.team_id, {
      limit: input.pagination?.limit,
      offset: input.pagination?.offset,
    });
    const total = await this.store.countVersions(input.skill_id, input.team_id);
    return {
      items: items.map((s) => ({ ...s, is_expired: this.isVersionExpired(s) })),
      total,
    };
  }

  async readFile(input: ReadFileInput): Promise<{
    path: string; content: string; encoding: "utf-8" | "base64";
    size_bytes: number; mime_type: string; version: number;
  }> {
    const head = await this.store.getHead(input.skill_id, input.team_id);
    if (input.team_id) assertTeamMatchWrap(head, input.team_id);
    if (!head) throw new SkillCoreError("SKILL_NOT_FOUND");
    const ver = input.version ?? head.version;
    // 校验 path 是否在该版本的 manifest 中（如果是 head 直接看 head.manifest，否则查 by version）
    const target = typeof input.version === "number" && input.version !== head.version
      ? await this.store.getByVersion(input.skill_id, input.version, input.team_id)
      : head;
    if (!target) {
      // head 存在但指定版本查不到 → 版本已被 GC，不是 skill 不存在
      throw new SkillCoreError(
        "STORAGE_NOT_FOUND",
        `version ${input.version} not found (may have been GC'd); current head is v${head.version}`,
      );
    }
    // TTL 检查：指定旧版本资源时拦截
    if (typeof input.version === "number" && input.version !== head.version) {
      this.assertVersionNotExpired(target, head.version);
    }
    const exists = target.manifest.some((m) => m.path === input.path);
    if (!exists) throw new SkillCoreError("SKILL_NOT_FOUND", `path ${input.path} not in manifest of v${ver}`);

    const r = await this.resources.readResource(input.skill_id, ver, input.path, input.encoding ?? "utf-8");
    if (!r) throw new SkillCoreError("STORAGE_NOT_FOUND", "version directory missing (may be GC'd)");
    this.notifyAccessed(head);
    return {
      path: r.path,
      content: r.content,
      encoding: r.encoding,
      size_bytes: r.size_bytes,
      mime_type: r.mime_type,
      version: ver,
    };
  }

  // ───────────────────────────────────────────────────────────────────
  //  helpers
  // ───────────────────────────────────────────────────────────────────

  private parseAndValidate(raw: string) {
    let file;
    try {
      file = parseSkillFile(raw);
      validateSkillFile(file);
    } catch (e) {
      // Parse / 长度 / regex 失败 → 42203（设计 §3.6）
      throw new SkillCoreError("SKILL_FRONTMATTER_INVALID", (e as Error).message);
    }
    return file;
  }

  private async requireHead(skillId: string, teamId?: string): Promise<Skill> {
    const head = await this.store.getHead(skillId, teamId);
    if (teamId) assertTeamMatchWrap(head, teamId);
    if (!head) throw new SkillCoreError("SKILL_NOT_FOUND");
    return head;
  }

  private ctxOf(input: { user_id?: string; team_id?: string; agent_id?: string; task_id?: string }) {
    return {
      user_id: input.user_id,
      team_id: input.team_id,
      agent_id: input.agent_id,
      task_id: input.task_id,
    };
  }

  // ── TTL helpers ──

  /** 判断非 head 版本是否过期。head 永不过期；ttlSeconds=0 关闭。 */
  private isVersionExpired(skill: Skill): boolean {
    if (skill.is_head) return false;
    if (!this.versionTtlSeconds) return false;
    return this.now() - skill.created_at_ms > this.versionTtlSeconds * 1000;
  }

  private assertVersionNotExpired(skill: Skill, headVersion: number): void {
    if (!this.isVersionExpired(skill)) return;
    const ageDays = ((this.now() - skill.created_at_ms) / 86400000).toFixed(1);
    const ttlDays = this.versionTtlSeconds / 86400;
    throw new SkillCoreError(
      "SKILL_VERSION_EXPIRED",
      `Skill '${skill.name}' version v${skill.version} has expired ` +
      `(created ${ageDays} days ago, TTL is ${ttlDays} days). ` +
      `Please use the latest version v${headVersion}.`,
    );
  }
}

// ═════════════════════════════════════════════════════════════════════
//  内部小工具
// ═════════════════════════════════════════════════════════════════════

function assertOwnerWrap(head: Skill, agentId: string, teamId?: string): void {
  try { assertOwner(head, agentId, teamId); }
  catch (e) {
    if (e instanceof SkillPermissionError) throw new SkillCoreError(e.code as SkillCoreErrorCode, e.message);
    throw e;
  }
}

function assertTeamMatchWrap(row: Skill | null, teamId: string): asserts row is Skill {
  try { assertTeamMatch(row, teamId); }
  catch (e) {
    if (e instanceof SkillPermissionError) throw new SkillCoreError(e.code as SkillCoreErrorCode, e.message);
    throw e;
  }
}

function assertVersionFreshWrap(head: Skill, expected: number): void {
  try { assertVersionFresh(head, expected); }
  catch (e) {
    if (e instanceof SkillPermissionError) throw new SkillCoreError(e.code as SkillCoreErrorCode, e.message);
    throw e;
  }
}

function countOccurrences(haystack: string, needle: string): number {
  if (!needle) return 0;
  let count = 0;
  let idx = 0;
  while ((idx = haystack.indexOf(needle, idx)) !== -1) {
    count++;
    idx += needle.length;
  }
  return count;
}

function splitJoin(s: string, find: string, replace: string): string {
  return s.split(find).join(replace);
}
