/**
 * Memory Store Abstraction Layer — Core Types & Interfaces.
 *
 * This module defines the storage contracts that all backend implementations
 * (SQLite local, Tencent Cloud VectorDB, etc.) must satisfy.
 *
 * Design principles:
 * 1. **Backend-agnostic**: Upper-layer modules (hooks, tools, pipeline, record)
 *    depend only on these interfaces — never on concrete implementations.
 * 2. **Capability-based**: Features like vector search, FTS, and hybrid search
 *    are expressed as capability flags so callers can gracefully degrade.
 * 3. **Fault-tolerant**: All methods return empty results or `false` on
 *    failure rather than throwing, unless explicitly documented otherwise.
 * 4. **Sync-first**: Matches current SQLite DatabaseSync usage. TCVDB backend
 *    adapts internally without changing these signatures.
 */

import type { MemoryRecord } from "../record/l1-writer.js";
import type { EmbeddingProviderInfo } from "./embedding.js";
import type { Logger } from "../types.js";
import type { IsolationFilter } from "./isolation.js";

// Re-export so consumers can import everything from types.ts
export type { MemoryRecord, EmbeddingProviderInfo };

// Re-export isolation primitives so all store consumers import from here.
export type {
  IsolationContext,
  IsolationFilter,
  IsolationConfig,
} from "./isolation.js";
export {
  DEFAULT_ISOLATION_ID,
  LEGACY_ISOLATION_PLACEHOLDER,
  DEFAULT_ISOLATION_CONFIG,
  assertIsolation,
  buildIsolationWhere,
  rowMatchesIsolation,
  IsolationError,
} from "./isolation.js";

// ============================
// Common Types
// ============================

/** Minimal logger interface accepted by store implementations. */
export type StoreLogger = Logger;

// ============================
// L1 Types (Structured Memories)
// ============================

/** Result from an L1 vector similarity search. */
export interface L1SearchResult {
  record_id: string;
  content: string;
  type: string;
  priority: number;
  scene_name: string;
  /** Similarity score (0–1, higher is better). */
  score: number;
  timestamp_str: string;
  timestamp_start: string;
  timestamp_end: string;
  version: number;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  metadata_json: string;
}

/** Result from an L1 FTS keyword search. */
export interface L1FtsResult {
  record_id: string;
  content: string;
  type: string;
  priority: number;
  scene_name: string;
  /** BM25-derived score (0–1, higher is better). */
  score: number;
  timestamp_str: string;
  timestamp_start: string;
  timestamp_end: string;
  version: number;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  metadata_json: string;
}

/** Filter options for querying L1 records. */
export interface L1QueryFilter {
  /** Query by document primary keys (maps to VDB `documentIds`, max 20). */
  recordIds?: string[];
  sessionKey?: string;
  sessionId?: string;
  taskId?: string;
  /** Isolation dimensions (any subset). */
  teamId?: string;
  userId?: string;
  agentId?: string;
  /** Only return records with updated_time strictly after this ISO 8601 UTC timestamp. */
  updatedAfter?: string;
}

/** Row shape returned by L1 query methods. */
export interface L1RecordRow {
  record_id: string;
  content: string;
  type: string;
  priority: number;
  scene_name: string;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  version: number;
  timestamp_str: string;
  timestamp_start: string;
  timestamp_end: string;
  created_time: string;
  updated_time: string;
  metadata_json: string;
}

// ============================
// L0 Types (Raw Conversations)
// ============================

/** An L0 conversation message record for vector indexing. */
export interface L0Record {
  id: string;
  sessionKey: string;
  sessionId: string;
  /**
   * Three-dim isolation (new in this branch).
   *
   * Mandatory for new writes once gateway-level enforcement is on, but kept
   * optional on the type during the rollout window. SQLite upsert defaults to
   * '' when missing; the migration script backfills existing rows to
   * `__legacy__`.
   */
  teamId?: string;
  userId?: string;
  agentId?: string;
  taskId?: string;
  role: string;
  messageText: string;
  recordedAt: string;
  /** Original message timestamp (epoch ms). */
  timestamp: number;
}

/** Result from an L0 vector similarity search. */
export interface L0SearchResult {
  record_id: string;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  role: string;
  message_text: string;
  /** Similarity score (0–1, higher is better). */
  score: number;
  recorded_at: string;
  timestamp: number;
}

/** Result from an L0 FTS keyword search. */
export interface L0FtsResult {
  record_id: string;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  role: string;
  message_text: string;
  /** BM25-derived score (0–1, higher is better). */
  score: number;
  recorded_at: string;
  timestamp: number;
}

/** Raw L0 row returned by query methods (used by L1 runner). */
export interface L0QueryRow {
  record_id: string;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  role: string;
  message_text: string;
  recorded_at: string;
  timestamp: number;
}

/** L0 messages grouped by session ID (for L1 runner). */
export interface L0SessionGroup {
  sessionId: string;
  teamId?: string;
  userId: string;
  agentId: string;
  taskId?: string;
  messages: Array<{
    id: string;
    role: string;
    content: string;
    timestamp: number;
    /** Epoch ms when this message was recorded into L0 (used by L1 cursor). */
    recordedAtMs: number;
  }>;
}

// ============================
// Store Init Result
// ============================

/** Result of store initialization. */
export interface StoreInitResult {
  /** Whether embeddings need to be regenerated (provider/model change). */
  needsReindex: boolean;
  /** Human-readable reason (for logging). */
  reason?: string;
}

// ============================
// Capability Flags
// ============================

/**
 * Describes what search capabilities a store backend supports.
 * Callers use this to select search strategies and degrade gracefully.
 */
export interface StoreCapabilities {
  /** Whether vector (embedding) search is available. */
  vectorSearch: boolean;
  /** Whether FTS (full-text keyword) search is available. */
  ftsSearch: boolean;
  /** Whether native hybrid search is supported (e.g., TCVDB hybridSearch). */
  nativeHybridSearch: boolean;
  /** Whether the store supports sparse vectors (BM25 encoding). */
  sparseVectors: boolean;
}

// ============================
// L2/L3 Profile Sync Types
// ============================

/** Canonical L2/L3 profile row shared between local cache and remote store. */
export interface ProfileRecord {
  /** Stable ID: `profile:v1:${sha256(scope + "\0" + type + "\0" + filename)}`. */
  id: string;
  type: "l2" | "l3";
  filename: string;
  content: string;
  contentMd5: string;
  /** L2/L3 profile identity is team+agent scoped. userId/sessionId are optional
   *  for backwards compatibility with old in-memory shapes; new profile writes leave them empty. */
  teamId?: string;
  agentId?: string;
  userId?: string;
  sessionId?: string;
  version: number;
  createdAtMs: number;
  updatedAtMs: number;
}

/** Profile upsert payload with optimistic-lock baseline from the last pull. */
export interface ProfileSyncRecord extends ProfileRecord {
  baselineVersion?: number;
}

export interface ProfileCountFilter {
  type?: ProfileRecord["type"];
  teamId?: string;
  userId?: string;
  agentId?: string;
  pathPrefix?: string;
}

// ============================
// v2 API Paginated Query Types
// ============================

/** Filter for v2 L0 paginated query (`/conversation/query`). */
export interface L0CountFilter {
  /** Filter by session. */
  sessionId?: string;
  /** Isolation dimensions (any subset). */
  teamId?: string;
  userId?: string;
  agentId?: string;
  taskId?: string;
  /** Timestamp >= (epoch ms, inclusive). */
  timeStartMs?: number;
  /** Timestamp <= (epoch ms, inclusive). */
  timeEndMs?: number;
}

export interface L0PaginatedFilter extends L0CountFilter {
  /** Page size. */
  limit: number;
  /** Page offset. */
  offset: number;
}

/** Result of v2 L0 paginated query. */
export interface L0PaginatedResult {
  rows: L0QueryRow[];
  /** Total count matching filters (for pagination). */
  total: number;
}

/** Filter for v2 L1 paginated query (`/atomic/query`). */
export interface L1CountFilter {
  /** Filter by memory type (episodic/persona/instruction). */
  type?: string;
  /** Filter by session. */
  sessionId?: string;
  /** Isolation dimensions (any subset). */
  teamId?: string;
  userId?: string;
  agentId?: string;
  taskId?: string;
  /** Filter by updated_time >= (ISO 8601). */
  timeStart?: string;
  /** Filter by updated_time <= (ISO 8601). */
  timeEnd?: string;
}

export interface L1PaginatedFilter extends L1CountFilter {
  /** Page size. */
  limit: number;
  /** Page offset. */
  offset: number;
}

/** Result of v2 L1 paginated query. */
export interface L1PaginatedResult {
  rows: L1RecordRow[];
  /** Total count matching filters (for pagination). */
  total: number;
}

// ============================
// Entity Metadata Types (Team / User / Agent / Task)
// ============================

export type TeamStatus = "active" | "archived";
export type UserStatus = "active" | "inactive";
export type AgentStatus = "active" | "inactive";
export type AgentVisibility = "team" | "restricted";
export type TaskSourceType = "manual" | "github" | "tapd" | "other";

export interface TeamEntity {
  team_id: string;
  name: string;
  description?: string;
  owner_user_id: string;
  status: TeamStatus;
  user_ids?: string[];
  agent_ids?: string[];
  task_ids?: string[];
  created_at: string;
  updated_at: string;
}

export interface UserEntity {
  user_id: string;
  name: string;
  job_description?: string;
  team_ids: string[];
  task_ids: string[];
  owned_agent_ids: string[];
  task_agent_ids?: string[];
  status: UserStatus;
  created_at: string;
  updated_at: string;
}

export interface AgentEntity {
  agent_id: string;
  team_id: string;
  name: string;
  description?: string;
  prompt?: string;
  owner_user_id?: string;
  visibility: AgentVisibility;
  status: AgentStatus;
  task_ids?: string[];
  created_at: string;
  updated_at: string;
}

export interface TaskEntity {
  task_id: string;
  team_id: string;
  creator_user_id: string;
  title?: string;
  description?: string;
  source_type: TaskSourceType;
  source_url?: string;
  agent_ids: string[];
  user_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface BatchDeleteResult {
  deleted_ids: string[];
  failed: Array<{ id: string; reason: string }>;
}

export type KnowledgeType = "wiki" | "code-graph";

export interface KnowledgeEntity {
  knowledge_id: string;
  type: KnowledgeType;
  service_url: string;
  name: string;
  summary: string | null;
  team_id: string;
  /** 预留：agent 绑定维度（当前写 ""，绑定权威在 meta_assets）。 */
  agent_id?: string;
  user_id: string | null;
  repo_url?: string;
  branch?: string;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeListResult {
  items: KnowledgeEntity[];
  total: number;
}

// ============================
// IMemoryStore — The Core Abstraction
// ============================

/**
 * Unified memory store interface.
 *
 * Implementations:
 * - `SqliteMemoryStore` (sqlite.ts) — local SQLite + sqlite-vec + FTS5
 * - `TcvdbMemoryStore` (tcvdb.ts) — Tencent Cloud VectorDB (future)
 *
 * All methods are fault-tolerant: they return empty results or `false` on
 * failure rather than throwing, unless explicitly documented otherwise.
 */
/**
 * Helper type: a value that may be sync or async.
 * Callers should always `await` the result — it's safe for both sync and async values.
 */
export type MaybePromise<T> = T | Promise<T>;

// ============================
// Memory Audit (修改审计)
// ============================

/**
 * 一次记忆修改事件。原始 L0/L1/L2/L3 表完全不动；这里只追加一行审计。
 *
 * 设计要点（per user 决策）：
 *   - 不存历史 content / 旧值，只记"什么时间、由谁、改了哪条"
 *   - team/agent/user/task 来自外部请求 IdFields（不是 record 原值）
 *   - version = 修改后该记录的新版本号（与原表 version 字段一致）
 *   - L0 不进 audit（不可变流水）；L1/L2/L3 update + delete 各记一条
 */
export interface AuditEntry {
  /** 自动生成主键，建议 audit-{uuid}。 */
  audit_id: string;
  /**
   * 被修改的记录主键：
   *   - L1 → MemoryRecord.id (msg-xxx / mem-xxx)
   *   - L2 → 文件路径 (scene_blocks/xxx.md)
   *   - L3 → "core"（全实例只一份）或 path
   */
  record_id: string;
  layer: "L1" | "L2" | "L3";
  action: "update" | "delete";
  /** 外部请求 IdFields 副本，来源是调用方传入的 body / header（resolveIsolation 后）。 */
  team_id?: string;
  agent_id?: string;
  user_id?: string;
  task_id?: string;
  /** 修改后该记录的新版本号。delete 用 0 或上一版本号 + 1。 */
  version: number;
  /** 修改时间（毫秒）。 */
  updated_at_ms: number;
  /** Gateway request_id，便于 trace。 */
  request_id?: string;
}

/** queryAudit 过滤条件，全部可选。 */
export interface AuditQueryFilter {
  record_id?: string;
  layer?: "L1" | "L2" | "L3";
  action?: "update" | "delete";
  team_id?: string;
  agent_id?: string;
  user_id?: string;
  task_id?: string;
  /** 只返 updated_at_ms ≥ since_ms 的事件。 */
  since_ms?: number;
  /** 只返 updated_at_ms ≤ until_ms 的事件。 */
  until_ms?: number;
  limit?: number;   // 默认 100，上限 1000
  offset?: number;
}

export interface IMemoryStore {
  // ── Capabilities ───────────────────────────────────────────

  /**
   * Whether this store supports deferred (background) embedding updates.
   *
   * When `true`, auto-capture writes metadata-only via `upsertL0(record, undefined)`
   * and later calls `updateL0Embedding()` in a fire-and-forget background task.
   * When `false` or absent, embedding is computed inline and passed to `upsertL0()`.
   */
  readonly supportsDeferredEmbedding?: boolean;

  // ── Lifecycle (always sync) ──────────────────────────────

  init(providerInfo?: EmbeddingProviderInfo): MaybePromise<StoreInitResult>;
  isDegraded(): boolean;
  getCapabilities(): StoreCapabilities;
  close(): void;

  // ── L1 Write ─────────────────────────────────────────────

  upsertL1(record: MemoryRecord, embedding?: Float32Array): MaybePromise<boolean>;
  deleteL1(recordId: string, filter?: IsolationFilter): MaybePromise<boolean>;
  deleteL1Batch(recordIds: string[], filter?: IsolationFilter): MaybePromise<boolean>;
  deleteL1Expired(cutoffIso: string): MaybePromise<number>;

  // ── L1 Read ──────────────────────────────────────────────

  countL1(filter?: L1CountFilter): MaybePromise<number>;
  queryL1Records(filter?: L1QueryFilter): MaybePromise<L1RecordRow[]>;
  getAllL1Texts(): MaybePromise<Array<{ record_id: string; content: string; updated_time: string }>>;

  // ── L1 Search ────────────────────────────────────────────

  searchL1Vector(queryEmbedding: Float32Array, topK?: number, queryText?: string, filter?: IsolationFilter): MaybePromise<L1SearchResult[]>;
  searchL1Fts(ftsQuery: string, limit?: number, filter?: IsolationFilter): MaybePromise<L1FtsResult[]>;
  searchL1Hybrid?(params: {
    query?: string;
    queryEmbedding?: Float32Array;
    sparseVector?: Array<[number, number]>;
    topK?: number;
    filter?: IsolationFilter;
  }): MaybePromise<L1SearchResult[]>;

  // ── L0 Write ─────────────────────────────────────────────

  upsertL0(record: L0Record, embedding?: Float32Array): MaybePromise<boolean>;
  /** Update only the vector embedding for an existing L0 record (sqlite background path). */
  updateL0Embedding?(recordId: string, embedding: Float32Array): MaybePromise<boolean>;
  deleteL0(recordId: string, filter?: IsolationFilter): MaybePromise<boolean>;
  deleteL0Expired(cutoffIso: string): MaybePromise<number>;

  // ── L0 Read ──────────────────────────────────────────────

  countL0(filter?: L0CountFilter): MaybePromise<number>;
  queryL0ForL1(sessionKey: string, afterRecordedAtMs?: number, limit?: number): MaybePromise<L0QueryRow[]>;
  queryL0GroupedBySessionId(sessionKey: string, afterRecordedAtMs?: number, limit?: number): MaybePromise<L0SessionGroup[]>;
  getAllL0Texts(): MaybePromise<Array<{ record_id: string; message_text: string; recorded_at: string }>>;

  // ── L0 Search ────────────────────────────────────────────

  searchL0Vector(queryEmbedding: Float32Array, topK?: number, queryText?: string, filter?: IsolationFilter): MaybePromise<L0SearchResult[]>;
  searchL0Fts(ftsQuery: string, limit?: number, filter?: IsolationFilter): MaybePromise<L0FtsResult[]>;
  searchL0Hybrid?(params: {
    query?: string;
    queryEmbedding?: Float32Array;
    sparseVector?: Array<[number, number]>;
    topK?: number;
    filter?: IsolationFilter;
  }): MaybePromise<L0SearchResult[]>;

  pullProfiles?(): Promise<ProfileRecord[]>;
  /**
   * 按 profile stable id 批量查询（轻量，仅查指定 id）。
   * 当 store 支持时，调用方应优先使用此接口而非 pullProfiles() 全量拉取。
   * 不支持的 store 返回 undefined → 调用方 fallback 到 pullProfiles()。
   */
  queryProfilesByIds?(ids: string[]): Promise<ProfileRecord[]>;
  countProfiles?(filter?: ProfileCountFilter): Promise<number>;
  syncProfiles?(records: ProfileSyncRecord[]): Promise<void>;
  deleteProfiles?(recordIds: string[]): Promise<void>;

  // ── Re-index ─────────────────────────────────────────────

  reindexAll(
    embedFn: (text: string) => Promise<Float32Array>,
    onProgress?: (done: number, total: number, layer: "L1" | "L0") => void,
  ): Promise<{ l1Count: number; l0Count: number }>;

  // ── FTS (always sync — cached flag) ──────────────────────

  isFtsAvailable(): boolean;

  // ── v2 API: Paginated queries (optional — added for Gateway v2) ──

  /**
   * L0 paginated query for v2 API `/conversation/query`.
   * Returns rows matching the filter, paginated by limit/offset,
   * plus the total count of matching rows.
   */
  queryL0Paginated?(filter: L0PaginatedFilter): MaybePromise<L0PaginatedResult>;

  /**
   * L1 paginated query for v2 API `/atomic/query`.
   * Returns rows matching the filter, paginated by limit/offset,
   * plus the total count of matching rows.
   */
  queryL1Paginated?(filter: L1PaginatedFilter): MaybePromise<L1PaginatedResult>;

  /**
   * Delete all L0 messages belonging to a session.
   * Returns the actual number of rows deleted.
   * Used by v2 API `/conversation/delete` (session mode).
   */
  deleteL0BySession?(sessionId: string, filter?: IsolationFilter): MaybePromise<number>;

  // ── Entity metadata (Team / User / Agent / Task) ───────────
  createTeam?(input: Omit<TeamEntity, "created_at" | "updated_at" | "status" | "user_ids" | "agent_ids" | "task_ids"> & { team_id?: string; status?: TeamStatus }): MaybePromise<TeamEntity>;
  getTeam?(teamId: string): MaybePromise<TeamEntity | null>;
  updateTeam?(teamId: string, patch: Partial<Pick<TeamEntity, "name" | "description" | "owner_user_id" | "user_ids" | "agent_ids" | "status">>): MaybePromise<TeamEntity | null>;
  deleteTeams?(teamIds: string[]): MaybePromise<BatchDeleteResult>;

  createUser?(input: Pick<UserEntity, "name"> & Partial<Pick<UserEntity, "job_description" | "status">> & { user_id?: string }): MaybePromise<UserEntity>;
  getUser?(userId: string): MaybePromise<UserEntity | null>;
  updateUser?(userId: string, patch: Partial<Pick<UserEntity, "name" | "job_description" | "status">>): MaybePromise<UserEntity | null>;
  deleteUsers?(userIds: string[]): MaybePromise<BatchDeleteResult>;

  createAgent?(input: Omit<AgentEntity, "created_at" | "updated_at" | "status" | "visibility"> & { agent_id?: string; status?: AgentStatus; visibility?: AgentVisibility }): MaybePromise<AgentEntity>;
  getAgent?(agentId: string): MaybePromise<AgentEntity | null>;
  updateAgent?(agentId: string, patch: Partial<Pick<AgentEntity, "name" | "description" | "prompt" | "owner_user_id" | "visibility" | "status">>): MaybePromise<AgentEntity | null>;
  deleteAgents?(agentIds: string[]): MaybePromise<BatchDeleteResult>;

  createTask?(input: Omit<TaskEntity, "created_at" | "updated_at" | "source_type" | "agent_ids" | "user_ids"> & { task_id?: string; source_type?: TaskSourceType; agent_ids?: string[]; user_ids?: string[] }): MaybePromise<TaskEntity>;
  getTask?(taskId: string): MaybePromise<TaskEntity | null>;
  updateTask?(taskId: string, patch: Partial<Pick<TaskEntity, "title" | "description" | "source_type" | "source_url" | "agent_ids" | "user_ids">>): MaybePromise<TaskEntity | null>;
  deleteTasks?(taskIds: string[]): MaybePromise<BatchDeleteResult>;

  // ── Knowledge entity (wiki / code-graph metadata) ───────────
  createKnowledge?(input: Omit<KnowledgeEntity, "created_at" | "updated_at">): MaybePromise<KnowledgeEntity>;
  getKnowledge?(knowledgeId: string): MaybePromise<KnowledgeEntity | null>;
  updateKnowledge?(knowledgeId: string, patch: Partial<Pick<KnowledgeEntity, "name" | "summary" | "service_url" | "repo_url" | "branch">>): MaybePromise<KnowledgeEntity | null>;
  deleteKnowledge?(knowledgeIds: string[], teamId?: string): MaybePromise<BatchDeleteResult>;
  listKnowledge?(input: { team_id: string; type?: KnowledgeType; knowledge_ids?: string[]; limit?: number; offset?: number }): MaybePromise<KnowledgeListResult>;

  // ── Memory Audit（修改审计；optional 让 store 可以选择不实现）──
  appendAudit?(entry: AuditEntry): MaybePromise<void>;
  queryAudit?(filter: AuditQueryFilter): MaybePromise<AuditEntry[]>;
}

// ============================
// IEmbeddingService — re-exported from embedding.ts for convenience
// ============================

/**
 * Re-export EmbeddingService as IEmbeddingService for backward compatibility.
 * The canonical definition lives in `./embedding.ts`. All concrete implementations
 * (LocalEmbeddingService, OpenAIEmbeddingService, NoopEmbeddingService) implement
 * the EmbeddingService interface from embedding.ts.
 */
export type { EmbeddingService as IEmbeddingService } from "./embedding.js";
