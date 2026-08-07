/**
 * TcvdbSkillStore — Skill 存储层的 TCVDB (Service 模式) 实现
 *
 * 实现 ISkillStore 接口，提供技能元数据/内容的 VDB 持久化。
 * 对标 SqliteSkillStore 的 12 个方法，参考 TcvdbMemoryStore 的
 * Collection 创建 / upsert / query / search / hybridSearch 模式。
 *
 * Schema: 详见 docs/design/2026-06-29-skill-vdb-schema.md
 * 接口:   src/core/skill/skill-store.interface.ts
 */

import { randomBase62 } from "../../utils/short-id.js";
import { TcvdbClient, TcvdbApiError, type QueryResponse } from "./tcvdb-client.js";
import type { BM25LocalEncoder } from "./bm25-local.js";
import type { SparseVector } from "@tencentdb-agent-memory/tcvdb-text";
import type { StoreLogger } from "./types.js";
import type {
  ISkillStore,
  SkillStoreCapabilities,
  SkillSearchResult,
  ExpiredVersionMeta,
} from "../skill/skill-store.interface.js";
import type {
  AppendVersionInput,
  ListSkillsOptions,
  SearchSkillsOptions,
  Skill,
  SkillManifestEntry,
  SkillStatus,
} from "../skill/types.js";
import { SkillStoreError } from "../skill/skill-store.js";

// ─── Config ─────────────────────────────────────────────────────────────

export interface TcvdbSkillStoreConfig {
  /** VDB 实例 URL */
  url: string;
  /** 账户名 (默认 "root") */
  username: string;
  /** API Key */
  apiKey: string;
  /** Database 名称 */
  database: string;
  /** Embedding 模型名 (与 L1 共用 "bge-large-zh") */
  embeddingModel: string;
  /** 请求超时 ms */
  timeout: number;
  /** CA 证书路径 */
  caPemPath?: string;
  logger?: StoreLogger;
  /** BM25 编码器 (shared instance) */
  bm25Encoder?: BM25LocalEncoder;
  /** 注入 ulid 工厂 */
  ulid?: () => string;
  /** 注入 now */
  now?: () => number;
}

// ─── Constants ──────────────────────────────────────────────────────────

const TAG = "[tcvdb-skill-store]";
const SKILLS_COLLECTION_SUFFIX = "_skills";

/** VDB 密集向量索引 (DISK_FLAT, HNSW fallback) */
const VECTOR_INDEX_DISK_FLAT: Record<string, unknown> = {
  fieldName: "vector",
  fieldType: "vector",
  indexType: "DISK_FLAT",
  dimension: 1024,
  metricType: "COSINE",
  params: { M: 16, efConstruction: 200 },
};
const VECTOR_INDEX_HNSW: Record<string, unknown> = {
  fieldName: "vector",
  fieldType: "vector",
  indexType: "HNSW",
  dimension: 1024,
  metricType: "COSINE",
  params: { M: 16, efConstruction: 200 },
};

/** 查询时返回的字段 (全部, vector/sparse_vector 除外) */
const SKILL_OUTPUT_FIELDS: string[] = [
  "id", "skill_id", "version", "is_head",
  "team_id", "owner_agent_id", "user_id", "task_id",
  "name", "description", "content", "content_hash",
  "manifest_json", "storage_dir", "status", "metadata_json",
  "created_at_ms", "updated_at_ms",
];

/** 向量字段名 (VDB 内部名) */
const DENSE_VECTOR_FIELD = "vector";
const SPARSE_VECTOR_FIELD = "sparse_vector";

// ─── Ulid helpers ───────────────────────────────────────────────────────

// row_id 生成器 —— VDB doc 的物理主键 (`id` primaryKey field)。
// 与 skill_id 分离：skill_id 在版本间共享，row_id 每一行唯一。
// base62 12 字符（~71 bit CSPRNG 真熵）。
function defaultUlid(): string {
  return randomBase62(12);
}

// ─── Error helpers ──────────────────────────────────────────────────────

function isDiskFlatUnsupported(err: unknown): boolean {
  if (!(err instanceof TcvdbApiError)) return false;
  if (err.apiCode === 15113) return true;
  const msg = err.message.toLowerCase();
  return msg.includes("disk_flat") && msg.includes("not support");
}

// ─── Implementation ─────────────────────────────────────────────────────

export class TcvdbSkillStore implements ISkillStore {
  private readonly client: TcvdbClient;
  private readonly skillsCollection: string;
  private readonly embeddingModel: string;
  private readonly logger?: StoreLogger;
  private readonly bm25Encoder?: BM25LocalEncoder;
  private readonly ulid: () => string;
  private readonly now: () => number;

  private degraded = false;
  private initPromise?: Promise<void>;
  private initialized = false;

  constructor(config: TcvdbSkillStoreConfig) {
    this.client = new TcvdbClient({
      url: config.url,
      username: config.username,
      apiKey: config.apiKey,
      database: config.database,
      timeout: config.timeout,
      caPemPath: config.caPemPath,
    });
    this.skillsCollection = `${config.database}${SKILLS_COLLECTION_SUFFIX}`;
    this.embeddingModel = config.embeddingModel;
    this.logger = config.logger;
    this.bm25Encoder = config.bm25Encoder;
    this.ulid = config.ulid ?? defaultUlid;
    this.now = config.now ?? (() => Date.now());
  }

  // ── ISkillStore: 生命周期 ─────────────────────────────────────────────

  init(): void {
    if (this.initialized) return;
    this.initialized = true;
    this.initPromise = this._initAsync().catch((err) => {
      this.logger?.error(`${TAG} Init failed: ${err instanceof Error ? err.message : String(err)}`);
      this.degraded = true;
    });
  }

  isDegraded(): boolean {
    return this.degraded;
  }

  getCapabilities(): SkillStoreCapabilities {
    return {
      vectorSearch: !this.degraded,
      ftsSearch: !!this.bm25Encoder && !this.degraded,
      nativeHybridSearch: !!this.bm25Encoder && !this.degraded,
      sparseVectors: !!this.bm25Encoder,
    };
  }

  close(): void {
    this.degraded = true;
  }

  // ── ISkillStore: CRUD ─────────────────────────────────────────────────

  async appendVersion(input: AppendVersionInput): Promise<Skill> {
    await this._ensureInit();
    if (this.degraded) throw new Error("TcvdbSkillStore degraded");

    const tid = input.team_id ?? "default";
    const sid = input.skill_id;

    // 1. 查旧 head
    const head = await this._getHeadAsync(sid, tid);

    // 2. Name 唯一性校验 (无 head → 新 skill, 检查重名)
    if (!head) {
      await this._assertNameUnique(input.name, tid, input.owner_agent_id ?? "default", sid);
    } else {
      // 已有 history → name 不可变
      if (head.name !== input.name) {
        throw new SkillStoreError("SKILL_NAME_DUPLICATE", "name change is not allowed across versions");
      }
    }

    // 3. 版本唯一性校验
    const newVersion = head ? head.version + 1 : 1;
    const existing = await this._queryOneAsync(
      `skill_id="${this._escape(sid)}" and version=${newVersion} and team_id="${this._escape(tid)}"`,
    );
    if (existing) {
      // 同一版本已存在 → 幂等返回
      return existing;
    }

    const ownerForRow = head ? head.owner_agent_id : (input.owner_agent_id ?? "default");
    const userIdForRow = input.user_id ?? "default";
    const ts = this.now();
    const rowId = this.ulid();
    const storageDir = `skills/${sid}/v${newVersion}`;

    // 4. 构建新行文档
    const doc: Record<string, unknown> = {
      id: rowId,
      skill_id: sid,
      version: newVersion,
      is_head: 1,
      team_id: tid,
      owner_agent_id: ownerForRow,
      user_id: userIdForRow,
      task_id: input.task_id ?? "default",
      name: input.name,
      description: input.description,
      content: input.content,
      content_hash: input.content_hash,
      manifest_json: JSON.stringify(input.manifest ?? []),
      storage_dir: storageDir,
      status: "active",
      metadata_json: input.metadata_json ?? "{}",
      created_at_ms: ts,
      updated_at_ms: ts,
    };

    // 5. BM25 稀疏向量编码
    if (this.bm25Encoder) {
      const sparse = this.bm25Encoder.encodeTexts([input.content]);
      if (sparse.length > 0 && sparse[0] && sparse[0].length > 0) {
        doc[SPARSE_VECTOR_FIELD] = sparse[0];
      }
    }

    // 6. 先 INSERT 新行 (补偿 VDB 无事务：新行先落，旧行后翻)
    await this.client.upsert(this.skillsCollection, [doc]);

    // 7. 再翻旧 head
    if (head) {
      try {
        await this._updateDocAsync(head.row_id, { is_head: 0 } as Record<string, unknown>);
      } catch (err) {
        this.logger?.warn(`${TAG} Failed to flip old head is_head for ${sid} v${head.version}: ${err instanceof Error ? err.message : String(err)}`);
        // 不抛 — 新行已写入，双 head 可通过 version DESC 取最新解决
      }
    }

    return this._docToSkill(doc);
  }

  async getHead(skillId: string, teamId?: string): Promise<Skill | null> {
    await this._ensureInit();
    if (this.degraded) return null;

    return this._getHeadAsync(skillId, teamId);
  }

  /**
   * 内部使用：获取当前 head 但不过滤 status。archived head 也会返回。
   * 与 `SqliteSkillStore.getHeadIncludingArchived` 语义一致。
   * 供 `SkillCore.delete` 幂等回读、asset 补偿任务、管控台使用。
   */
  async getHeadIncludingArchived(skillId: string, teamId?: string): Promise<Skill | null> {
    await this._ensureInit();
    if (this.degraded) return null;

    return this._getHeadAsync(skillId, teamId, { includeArchived: true });
  }

  async getByVersion(skillId: string, version: number, teamId?: string): Promise<Skill | null> {
    await this._ensureInit();
    if (this.degraded) return null;

    const filter = teamId
      ? `skill_id="${this._escape(skillId)}" and version=${version} and team_id="${this._escape(teamId)}"`
      : `skill_id="${this._escape(skillId)}" and version=${version}`;

    return this._queryOneAsync(filter);
  }

  async archiveHead(skillId: string, teamId?: string): Promise<{ archived: boolean }> {
    await this._ensureInit();
    if (this.degraded) return { archived: false };

    const head = await this._getHeadAsync(skillId, teamId);
    if (!head) return { archived: false };

    try {
      await this._updateDocAsync(head.row_id, {
        status: "archived",
        updated_at_ms: this.now(),
      } as Record<string, unknown>);
      return { archived: true };
    } catch (err) {
      this.logger?.warn(`${TAG} archiveHead failed for ${skillId}: ${err instanceof Error ? err.message : String(err)}`);
      return { archived: false };
    }
  }

  // ── ISkillStore: 查询 ─────────────────────────────────────────────────

  async listSkills(opts: ListSkillsOptions): Promise<{ items: Skill[]; total: number }> {
    await this._ensureInit();
    if (this.degraded) return { items: [], total: 0 };

    const conditions: string[] = ["is_head=1"];
    if (opts.team_id) conditions.push(`team_id="${this._escape(opts.team_id)}"`);
    if (opts.owner_agent_id) conditions.push(`owner_agent_id="${this._escape(opts.owner_agent_id)}"`);
    if (opts.user_id) conditions.push(`user_id="${this._escape(opts.user_id)}"`);
    if (opts.task_id) conditions.push(`task_id="${this._escape(opts.task_id)}"`);

    const statuses = opts.status?.length ? opts.status : (["active"] as SkillStatus[]);
    if (statuses.length === 1) {
      conditions.push(`status="${this._escape(statuses[0])}"`);
    } else {
      // VDB 不支持 `IN (...)` 语法，只支持 `OR`；用 `(status="a" or status="b")` 展开。
      // 之前用 IN 会静默失败（VDB 返回 code=14000），导致 status=["active","archived"] 空结果。
      conditions.push(
        `(${statuses.map((s) => `status="${this._escape(s)}"`).join(" or ")})`,
      );
    }

    const filter = conditions.join(" and ");
    const limit = Math.min(Math.max(opts.limit ?? 50, 1), 1000);
    const offset = Math.max(opts.offset ?? 0, 0);

    // name_prefix: VDB filter 语法不支持 LIKE / 字符串前缀匹配，
    // 改为拉取符合其余条件的 head 行后在内存里做前缀过滤再分页。
    // (head 行数量级小，一次性拉取上限 1000 可接受)
    if (opts.name_prefix) {
      const prefix = opts.name_prefix;
      try {
        const resp = await this.client.query(this.skillsCollection, {
          filter,
          limit: 1000,
          outputFields: SKILL_OUTPUT_FIELDS,
          sort: [{ fieldName: "updated_at_ms", direction: "desc" }],
        });
        const all = (resp.documents ?? [])
          .map((d) => this._docToSkill(d))
          .filter((s) => s.name.startsWith(prefix));
        return { items: all.slice(offset, offset + limit), total: all.length };
      } catch (err) {
        this.logger?.warn(`${TAG} listSkills(name_prefix) query failed: ${err instanceof Error ? err.message : String(err)}`);
        return { items: [], total: 0 };
      }
    }

    let total: number;
    try {
      total = await this.client.count(this.skillsCollection, filter);
    } catch {
      total = 0;
    }

    let rows: Skill[] = [];
    try {
      const resp = await this.client.query(this.skillsCollection, {
        filter,
        limit,
        offset,
        outputFields: SKILL_OUTPUT_FIELDS,
        sort: [{ fieldName: "updated_at_ms", direction: "desc" }],
      });
      rows = (resp.documents ?? []).map((d) => this._docToSkill(d));
    } catch (err) {
      this.logger?.warn(`${TAG} listSkills query failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    return { items: rows, total };
  }

  async searchSkills(opts: SearchSkillsOptions): Promise<SkillSearchResult[]> {
    await this._ensureInit();
    if (this.degraded) return [];

    const topK = Math.min(Math.max(opts.topK ?? 10, 1), 50);
    const filter = this._buildSearchFilter(opts);
    const mode = opts.mode ?? "bm25";

    try {
      if (mode === "hybrid" && this.bm25Encoder) {
        return this._searchHybridAsync(opts.query, topK, filter);
      }
      if (mode === "embedding") {
        return this._searchEmbeddingAsync(opts.query, topK, filter);
      }
      // bm25 (default)
      return this._searchBm25Async(opts.query, topK, filter);
    } catch (err) {
      this.logger?.warn(`${TAG} searchSkills failed: ${err instanceof Error ? err.message : String(err)}`);
      return [];
    }
  }

  async listVersions(
    skillId: string,
    teamId?: string,
    pagination?: { limit?: number; offset?: number },
  ): Promise<Skill[]> {
    await this._ensureInit();
    if (this.degraded) return [];

    const filter = teamId
      ? `skill_id="${this._escape(skillId)}" and team_id="${this._escape(teamId)}"`
      : `skill_id="${this._escape(skillId)}"`;

    const limit = Math.min(Math.max(pagination?.limit ?? 50, 1), 1000);
    const offset = Math.max(pagination?.offset ?? 0, 0);

    try {
      const resp = await this.client.query(this.skillsCollection, {
        filter,
        limit,
        offset,
        outputFields: SKILL_OUTPUT_FIELDS,
        sort: [{ fieldName: "version", direction: "desc" }],
      });
      return (resp.documents ?? []).map((d) => this._docToSkill(d));
    } catch (err) {
      this.logger?.warn(`${TAG} listVersions query failed: ${err instanceof Error ? err.message : String(err)}`);
      return [];
    }
  }

  async countVersions(skillId: string, teamId?: string): Promise<number> {
    await this._ensureInit();
    if (this.degraded) return 0;

    const filter = teamId
      ? `skill_id="${this._escape(skillId)}" and team_id="${this._escape(teamId)}"`
      : `skill_id="${this._escape(skillId)}"`;

    try {
      return await this.client.count(this.skillsCollection, filter);
    } catch {
      return 0;
    }
  }

  // ── ISkillStore: TTL Cleanup ─────────────────────────────────────────

  async findExpiredVersions(cutoffMs: number): Promise<ExpiredVersionMeta[]> {
    await this._ensureInit();
    if (this.degraded) return [];

    try {
      const resp = await this.client.query(this.skillsCollection, {
        filter: `is_head=0 and status="active" and created_at_ms<${cutoffMs}`,
        limit: 10000,
        outputFields: ["skill_id", "version", "is_head", "status", "storage_dir", "created_at_ms"],
        // VDB 要求 sort 字段为 uint64；skill_id 是 string 不可排序（code 15143）。
        // 用 created_at_ms 升序（先清最老的），符合 TTL 清理语义。
        sort: [{ fieldName: "created_at_ms", direction: "asc" }],
      });
      return (resp.documents ?? []).map((d) => ({
        skill_id: d.skill_id as string,
        version: d.version as number,
        is_head: (d.is_head as number) === 1,
        status: d.status as SkillStatus,
        storage_dir: d.storage_dir as string,
        created_at_ms: d.created_at_ms as number,
      }));
    } catch (err) {
      this.logger?.warn(`${TAG} findExpiredVersions failed: ${err instanceof Error ? err.message : String(err)}`);
      return [];
    }
  }

  async deleteVersion(skillId: string, version: number): Promise<boolean> {
    await this._ensureInit();
    if (this.degraded) return false;

    try {
      const filter = `skill_id="${this._escape(skillId)}" and version=${version} and is_head=0`;
      const affected = await this.client.deleteDoc(this.skillsCollection, { query: { filter } });
      return affected > 0;
    } catch (err) {
      this.logger?.warn(`${TAG} deleteVersion failed: ${err instanceof Error ? err.message : String(err)}`);
      return false;
    }
  }

  /**
   * 物理删除同 skill_id 下的所有版本行。`SkillCore.delete` 走此路径。
   * 权限校验由调用方 SkillCore 负责；本方法只按 (skill_id, team_id) 一次 deleteDoc。
   */
  async deleteAllVersions(skillId: string, teamId?: string): Promise<number> {
    await this._ensureInit();
    if (this.degraded) return 0;

    try {
      const filter = teamId
        ? `skill_id="${this._escape(skillId)}" and team_id="${this._escape(teamId)}"`
        : `skill_id="${this._escape(skillId)}"`;
      const affected = await this.client.deleteDoc(this.skillsCollection, { query: { filter } });
      return affected;
    } catch (err) {
      this.logger?.warn(`${TAG} deleteAllVersions failed for ${skillId}: ${err instanceof Error ? err.message : String(err)}`);
      return 0;
    }
  }

  // ─── Private: Init ────────────────────────────────────────────────────

  private async _initAsync(): Promise<void> {
    try {
      const dbCreated = await this.client.createDatabase();
      if (dbCreated) {
        this.logger?.debug?.(`${TAG} Database created, waiting 5s...`);
        await new Promise((r) => setTimeout(r, 5_000));
      }
    } catch (err) {
      if (err instanceof TcvdbApiError && err.apiCode === 15201) {
        this.logger?.debug?.(`${TAG} Database already exists (benign)`);
      } else {
        throw err;
      }
    }

    // Create skills collection with DISK_FLAT → HNSW fallback
    await this._createCollectionWithVectorFallback(
      {
        collection: this.skillsCollection,
        shardNum: 1,
        replicaNum: 2,
        description: "Skill 技能存储",
        embedding: {
          status: "enabled",
          field: "content",
          vectorField: DENSE_VECTOR_FIELD,
          model: this.embeddingModel,
        },
      },
      [
        { fieldName: "skill_id",       fieldType: "string", indexType: "filter" },
        { fieldName: "version",        fieldType: "uint64", indexType: "filter" },
        { fieldName: "is_head",        fieldType: "uint64", indexType: "filter" },
        { fieldName: "team_id",        fieldType: "string", indexType: "filter" },
        { fieldName: "owner_agent_id", fieldType: "string", indexType: "filter" },
        { fieldName: "user_id",        fieldType: "string", indexType: "filter" },
        { fieldName: "task_id",        fieldType: "string", indexType: "filter" },
        { fieldName: "name",           fieldType: "string", indexType: "filter" },
        { fieldName: "status",         fieldType: "string", indexType: "filter" },
        { fieldName: "created_at_ms",  fieldType: "uint64", indexType: "filter" },
        { fieldName: "updated_at_ms",  fieldType: "uint64", indexType: "filter" },
      ],
    );

    this.logger?.info(`${TAG} Initialized: collection=${this.skillsCollection}, model=${this.embeddingModel}`);
  }

  private async _createCollectionWithVectorFallback(
    params: Record<string, unknown>,
    filterIndexes: Array<Record<string, unknown>>,
  ): Promise<void> {
    const buildIndexes = (vectorIndex: Record<string, unknown>) => [
      { fieldName: "id", fieldType: "string", indexType: "primaryKey" },
      vectorIndex,
      { fieldName: SPARSE_VECTOR_FIELD, fieldType: "sparseVector", indexType: "inverted", metricType: "IP", diskSwapEnabled: true },
      ...filterIndexes,
    ];

    try {
      await this.client.createCollection({ ...params, indexes: buildIndexes(VECTOR_INDEX_DISK_FLAT) });
    } catch (err) {
      if (isDiskFlatUnsupported(err)) {
        this.logger?.debug?.(`${TAG} DISK_FLAT not supported, falling back to HNSW`);
        await this.client.createCollection({ ...params, indexes: buildIndexes(VECTOR_INDEX_HNSW) });
      } else {
        throw err;
      }
    }
  }

  private async _ensureInit(): Promise<void> {
    if (this.initPromise) {
      try { await this.initPromise; } catch { /* degraded already set */ }
    }
  }

  // ─── Private: Query helpers ───────────────────────────────────────────

  /**
   * 查 head 行。默认强制 `status="active"`；`includeArchived=true` 时不加 status 过滤，
   * 供 `getHeadIncludingArchived` 使用（archived head 的幂等回读 / 补偿任务）。
   */
  private async _getHeadAsync(
    skillId: string,
    teamId?: string,
    opts?: { includeArchived?: boolean },
  ): Promise<Skill | null> {
    const statusClause = opts?.includeArchived ? "" : ' and status="active"';
    const filter = teamId
      ? `skill_id="${this._escape(skillId)}" and team_id="${this._escape(teamId)}" and is_head=1${statusClause}`
      : `skill_id="${this._escape(skillId)}" and is_head=1${statusClause}`;

    return this._queryOneAsync(filter);
  }

  /** 按 filter 取一条 */
  private async _queryOneAsync(filter: string): Promise<Skill | null> {
    try {
      const resp = await this.client.query(this.skillsCollection, {
        filter,
        limit: 1,
        outputFields: SKILL_OUTPUT_FIELDS,
      });
      return resp.documents && resp.documents.length > 0
        ? this._docToSkill(resp.documents[0])
        : null;
    } catch (err) {
      this.logger?.warn(`${TAG} queryOne failed: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  }

  // ─── Private: Write helpers ───────────────────────────────────────────

  /** Upsert 更新文档的部分字段 (保留未传字段) */
  private async _updateDocAsync(rowId: string, partial: Record<string, unknown>): Promise<void> {
    // VDB upsert 需要完整文档或至少 id + 变更字段
    // 先读取现有文档，合并后 upsert
    const existing = await this._queryByIdAsync(rowId);
    if (!existing) return;

    const doc = this._skillToDoc(existing);
    Object.assign(doc, partial);
    await this.client.upsert(this.skillsCollection, [doc]);
  }

  /**
   * 按主键 (id / row_id) 取一条。
   * 注意：id 是 primaryKey 而非 filter 索引，不能用 `filter: id="..."` 查询
   * （VDB 会报 Field Not Found:id 被 catch 成 null）。必须走 documentIds 主键查找，
   * 对齐 memory 生产实现 (tcvdb.ts "Primary key lookup: use documentIds")。
   */
  private async _queryByIdAsync(rowId: string): Promise<Skill | null> {
    try {
      const resp = await this.client.query(this.skillsCollection, {
        documentIds: [rowId],
        limit: 1,
        retrieveVector: false,
        outputFields: SKILL_OUTPUT_FIELDS,
      });
      return resp.documents && resp.documents.length > 0
        ? this._docToSkill(resp.documents[0])
        : null;
    } catch (err) {
      this.logger?.warn(`${TAG} queryById failed: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  }

  /** Name 唯一性校验 (同 team + agent + name 且 is_head=1 且 status=active) */
  private async _assertNameUnique(
    name: string,
    teamId: string,
    ownerAgentId: string,
    excludeSkillId: string,
  ): Promise<void> {
    const filter =
      `team_id="${this._escape(teamId)}" and owner_agent_id="${this._escape(ownerAgentId)}" ` +
      `and name="${this._escape(name)}" and is_head=1 and status="active"`;

    try {
      const resp = await this.client.query(this.skillsCollection, {
        filter,
        limit: 1,
        outputFields: ["skill_id"],
      });
      if (resp.documents && resp.documents.length > 0) {
        const dupId = resp.documents[0].skill_id as string;
        if (dupId !== excludeSkillId) {
          throw new SkillStoreError("SKILL_NAME_DUPLICATE", `name '${name}' already exists for agent in team`);
        }
      }
    } catch (err) {
      if (err instanceof SkillStoreError) throw err;
      this.logger?.warn(`${TAG} name uniqueness check failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // ─── Private: Search ─────────────────────────────────────────────────

  private _buildSearchFilter(opts: SearchSkillsOptions): string {
    const conditions: string[] = ["is_head=1", 'status="active"'];
    if (opts.team_id) conditions.push(`team_id="${this._escape(opts.team_id)}"`);
    if (opts.agent_id) conditions.push(`owner_agent_id="${this._escape(opts.agent_id)}"`);
    if (opts.task_id) conditions.push(`task_id="${this._escape(opts.task_id)}"`);
    if (opts.user_id) conditions.push(`user_id="${this._escape(opts.user_id)}"`);
    return conditions.join(" and ");
  }

  /**
   * bm25 模式：TCVDB 无纯稀疏检索通道（/document/search 只做 dense；
   * hybridSearch 的 ann 为必填）。skill collection 的服务端 embedding 恒开启，
   * 故 bm25 模式在 Service 模式下降级为 hybrid（dense + sparse），
   * 语义等价且检索质量不弱于纯 BM25。standalone(SQLite) 才是真正的纯 BM25。
   */
  private async _searchBm25Async(
    queryText: string,
    topK: number,
    filter: string,
  ): Promise<SkillSearchResult[]> {
    this.logger?.debug?.(
      `${TAG} bm25 mode on TCVDB → degrade to hybrid (server-side embedding always enabled, no pure-sparse channel)`,
    );
    return this._searchHybridAsync(queryText, topK, filter);
  }

  /**
   * embedding 模式：dense-only。走 /document/search + embeddingItems，
   * 由 VDB 服务端对 query 文本做 embedding（collection.embedding.field=content）。
   * 注意：/document/search 不接受 ann/match，服务端 embedding 用 embeddingItems 传原始文本。
   */
  private async _searchEmbeddingAsync(
    queryText: string,
    topK: number,
    filter: string,
  ): Promise<SkillSearchResult[]> {
    const resp = await this.client.search(this.skillsCollection, {
      embeddingItems: [queryText],
      filter,
      limit: topK,
      retrieveVector: false,
      outputFields: SKILL_OUTPUT_FIELDS,
    });

    return this._parseSearchResponse(resp, topK);
  }

  /**
   * hybrid 模式：dense(服务端 embedding) + sparse(BM25) + RRF 融合。
   * 对齐 memory 生产实现 (tcvdb.ts searchL1HybridAsync)：
   *   - ann / match 均为数组
   *   - ann.fieldName = 服务端 embedding 源字段 "content"，data 传原始 query 文本
   *   - query 侧稀疏向量用 encodeQueries（IDF 权重），与写入侧 encodeTexts（TF）区分
   *   - rerank: { method: "rrf", k: 60 }
   * 无 BM25 编码器时退化为 dense-only（embedding）。
   */
  private async _searchHybridAsync(
    queryText: string,
    topK: number,
    filter: string,
  ): Promise<SkillSearchResult[]> {
    const sparse = this.bm25Encoder?.encodeQueries([queryText]) ?? [];
    const sparseVec: SparseVector | undefined =
      sparse.length > 0 && sparse[0] && sparse[0].length > 0 ? sparse[0] : undefined;

    if (!sparseVec) {
      // 无稀疏信号 → dense-only
      return this._searchEmbeddingAsync(queryText, topK, filter);
    }

    const searchParams: Record<string, unknown> = {
      filter,
      limit: topK,
      retrieveVector: false,
      outputFields: SKILL_OUTPUT_FIELDS,
      ann: [{
        fieldName: "content",
        data: [queryText],
        limit: topK * 2,
      }],
      match: [{
        fieldName: SPARSE_VECTOR_FIELD,
        data: [sparseVec],
        limit: topK * 2,
      }],
      rerank: {
        method: "rrf",
        k: 60,
      },
    };

    const resp = await this.client.hybridSearch(this.skillsCollection, searchParams);
    return this._parseSearchResponse(resp, topK);
  }

  private _parseSearchResponse(
    resp: { documents: Array<Array<Record<string, unknown>>> },
    topK: number,
  ): SkillSearchResult[] {
    const results: SkillSearchResult[] = [];
    const docs = resp.documents?.[0] ?? [];

    for (const d of docs) {
      if (results.length >= topK) break;
      const skill = this._docToSkill(d);
      const score = (d.score as number) ?? 0;
      let snippet: string | undefined;
      if (d.text !== undefined && typeof d.text === "string") {
        snippet = d.text.slice(0, 200);
      }
      results.push({ skill, score, snippet });
    }

    return results;
  }

  // ─── Private: Doc ↔ Skill mapping ─────────────────────────────────────

  private _docToSkill(doc: Record<string, unknown>): Skill {
    let manifest: SkillManifestEntry[] = [];
    try {
      const raw = doc.manifest_json as string | undefined;
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) manifest = parsed;
      }
    } catch { /* ignore */ }

    return {
      row_id: (doc.id ?? doc.row_id) as string,
      skill_id: doc.skill_id as string,
      version: (doc.version as number) ?? 0,
      is_head: (doc.is_head as number) === 1,
      user_id: doc.user_id as string,
      owner_agent_id: doc.owner_agent_id as string,
      team_id: doc.team_id as string,
      task_id: doc.task_id as string,
      name: doc.name as string,
      description: doc.description as string,
      content: doc.content as string,
      content_hash: doc.content_hash as string,
      manifest,
      storage_dir: doc.storage_dir as string,
      status: (doc.status as SkillStatus) ?? "active",
      metadata_json: (doc.metadata_json as string) ?? "{}",
      created_at_ms: (doc.created_at_ms as number) ?? 0,
      updated_at_ms: (doc.updated_at_ms as number) ?? 0,
    };
  }

  /** Skill → VDB doc (用于 update 时重写) */
  private _skillToDoc(skill: Skill): Record<string, unknown> {
    return {
      id: skill.row_id,
      skill_id: skill.skill_id,
      version: skill.version,
      is_head: skill.is_head ? 1 : 0,
      team_id: skill.team_id,
      owner_agent_id: skill.owner_agent_id,
      user_id: skill.user_id,
      task_id: skill.task_id,
      name: skill.name,
      description: skill.description,
      content: skill.content,
      content_hash: skill.content_hash,
      manifest_json: JSON.stringify(skill.manifest),
      storage_dir: skill.storage_dir,
      status: skill.status,
      metadata_json: skill.metadata_json,
      created_at_ms: skill.created_at_ms,
      updated_at_ms: skill.updated_at_ms,
    };
  }

  // ─── Private: String escape ───────────────────────────────────────────

  private _escape(s: string): string {
    return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  }
}
