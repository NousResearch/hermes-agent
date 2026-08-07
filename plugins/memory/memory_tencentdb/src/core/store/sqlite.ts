/**
 * VectorStore: SQLite-based vector storage using sqlite-vec extension.
 *
 * Manages two layers of vector-indexed data in a single SQLite database:
 *
 * **L1 (structured memories):**
 * 1. `l1_records` — relational metadata table (content, type, priority, scene, timestamps)
 * 2. `l1_vec` — vec0 virtual table for cosine similarity search
 *
 * **L0 (raw conversations):**
 * 3. `l0_conversations` — relational metadata table (session_key, role, message text, timestamps)
 * 4. `l0_vec` — vec0 virtual table for cosine similarity search on individual messages
 *
 * Dependencies: Node.js built-in `node:sqlite` (Node 22+) + `sqlite-vec` (from root workspace).
 *
 * Design:
 * - All operations are synchronous (DatabaseSync API).
 * - Writes use manual BEGIN/COMMIT transactions for atomicity (metadata + vector).
 * - vec0 virtual table does NOT support ON CONFLICT, so upsert = delete + insert.
 * - Thread-safe via WAL mode.
 */

import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import type { DatabaseSync, StatementSync, SQLInputValue } from "node:sqlite";
import type { MemoryRecord } from "../record/l1-writer.js";
import type { EmbeddingProviderInfo } from "./embedding.js";
import type {
  IMemoryStore,
  StoreCapabilities,
  L0Record,
  L1SearchResult,
  L1FtsResult,
  L0SearchResult,
  L0FtsResult,
  L0QueryRow,
  L1RecordRow,
  L1QueryFilter,
  L0CountFilter,
  L0PaginatedFilter,
  L0PaginatedResult,
  L1CountFilter,
  L1PaginatedFilter,
  L1PaginatedResult,
  IsolationFilter,
  TeamEntity,
  UserEntity,
  AgentEntity,
  TaskEntity,
  KnowledgeEntity,
  KnowledgeType,
  KnowledgeListResult,
  BatchDeleteResult,
  AuditEntry,
  AuditQueryFilter,
} from "./types.js";
import { DEFAULT_ISOLATION_ID, rowMatchesIsolation } from "./types.js";
import { SKILLS_DDL, SKILL_FTS_DDL } from "../skill/skill-store-ddl.js";
import type { Logger } from "../types.js";

export type { L1RecordRow } from "./types.js";

// ============================
// Types
// ============================

export interface VectorSearchResult {
  record_id: string;
  content: string;
  type: string;
  priority: number;
  scene_name: string;
  /** Cosine similarity score (1.0 - cosine_distance) */
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
  /** Raw metadata JSON string (e.g., contains activity_start_time / activity_end_time for episodic) */
  metadata_json: string;
}

/** L0 single-message vector search result. */
export interface L0VectorSearchResult {
  record_id: string;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  role: string;
  message_text: string;
  /** Cosine similarity score (1.0 - cosine_distance) */
  score: number;
  recorded_at: string;
  /** Original message timestamp (epoch ms) */
  timestamp: number;
}

export interface L0RecordRow {
  record_id: string;
  session_key: string;
  session_id: string;
  role: string;
  message_text: string;
  recorded_at: string;
  timestamp: number;
}

const TAG = "[memory-tdai][sqlite]";

/** Persisted metadata about the embedding provider used to generate stored vectors. */
interface EmbeddingMeta {
  provider: string;
  model: string;
  dimensions: number;
}

/** Result of VectorStore.init() — indicates whether a re-embed is needed. */
export interface VectorStoreInitResult {
  /**
   * `true` if the embedding provider/model/dimensions changed since
   * the vectors were last written.  Callers should re-embed all texts
   * (via `reindexAll()`) after receiving this flag.
   */
  needsReindex: boolean;
  /** Human-readable reason (for logging). */
  reason?: string;
}

// Use createRequire to load the experimental node:sqlite module
const require = createRequire(import.meta.url);

function requireNodeSqlite(): typeof import("node:sqlite") {
  return require("node:sqlite") as typeof import("node:sqlite");
}

// ============================
// FTS5 helpers (adapted from openclaw core hybrid.ts)
// ============================

// ── Chinese word segmentation (jieba) ──
// Lazy-loaded singleton: initialised on first call to `buildFtsQuery`.
// If @node-rs/jieba is unavailable, falls back to Unicode-regex splitting.

interface JiebaInstance {
  cutForSearch(text: string, hmm: boolean): string[];
}

let _jieba: JiebaInstance | null | undefined; // undefined = not yet tried

function getJieba(): JiebaInstance | null {
  if (_jieba !== undefined) return _jieba;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { Jieba } = require("@node-rs/jieba");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { dict } = require("@node-rs/jieba/dict");
    _jieba = Jieba.withDict(dict) as JiebaInstance;
  } catch {
    _jieba = null; // mark as unavailable — won't retry
  }
  return _jieba;
}

/**
 * Common Chinese stop-words that add noise to FTS5 queries.
 * Kept small on purpose — only high-frequency function words.
 */
const ZH_STOP_WORDS = new Set([
  "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
  "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
  "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那",
  "吗", "吧", "呢", "啊", "呀", "哦", "嗯",
]);

/**
 * Build an FTS5 MATCH query from raw text.
 *
 * When `@node-rs/jieba` is available, uses jieba's search-engine mode
 * (`cutForSearch`) for accurate Chinese word segmentation, producing
 * much better recall than the previous regex-only approach.
 *
 * Falls back to Unicode-regex splitting (`/[\p{L}\p{N}_]+/gu`) if
 * jieba is not installed.
 *
 * Tokens are OR-joined as quoted FTS5 phrase terms so that a document
 * matching *any* token is returned.  BM25 naturally ranks documents that
 * match more tokens higher, so precision is preserved while recall is
 * significantly improved — especially for longer queries and when running
 * in FTS-only fallback mode (no embedding available).
 *
 * Example (with jieba):
 *   "用户喜欢编程和TypeScript" → '"用户" OR "喜欢" OR "编程" OR "TypeScript"'
 * Example (fallback):
 *   "旅行计划 API" → '"旅行计划" OR "API"'
 */
export function buildFtsQuery(raw: string): string | null {
  const jieba = getJieba();

  let tokens: string[];
  if (jieba) {
    // jieba cutForSearch: splits long words further for better recall
    // e.g. "北京烤鸭" → ["北京", "烤鸭", "北京烤鸭"]
    tokens = jieba
      .cutForSearch(raw, true)
      .map((t) => t.trim())
      .filter((t) => {
        if (!t) return false;
        // Remove pure whitespace / punctuation tokens
        if (!/[\p{L}\p{N}]/u.test(t)) return false;
        // Remove common Chinese stop-words to reduce noise
        if (ZH_STOP_WORDS.has(t)) return false;
        return true;
      });
    // Deduplicate (cutForSearch may produce duplicates for sub-words)
    tokens = [...new Set(tokens)];
  } else {
    // Fallback: simple Unicode regex split
    tokens =
      raw
        .match(/[\p{L}\p{N}_]+/gu)
        ?.map((t) => t.trim())
        .filter(Boolean) ?? [];
  }

  if (tokens.length === 0) return null;
  const quoted = tokens.map((t) => `"${t.replaceAll('"', "")}"`);
  return quoted.join(" OR ");
}

/**
 * Tokenize text for FTS5 indexing (write-side).
 *
 * Uses jieba `cutForSearch()` (search-engine mode) to segment Chinese text,
 * then joins tokens with spaces. The resulting string is stored in the FTS5
 * `content` column so that `unicode61` tokenizer can split it into meaningful
 * words — including both full words and their sub-words.
 *
 * Using `cutForSearch` (instead of `cut`) ensures that the index contains
 * the same sub-word tokens that `buildFtsQuery()` produces on the query side.
 * For example, "人工智能" is indexed as "人工 智能 人工智能", so queries for
 * either the full term or sub-words will match.
 *
 * Falls back to the original text if jieba is unavailable.
 *
 * Example (with jieba):
 *   "用户五月去日本旅行" → "用户 五月 去 日本 旅行"
 *   "人工智能的分支"     → "人工 智能 人工智能 的 分支"
 * Example (fallback):
 *   "用户五月去日本旅行" → "用户五月去日本旅行" (unchanged)
 */
export function tokenizeForFts(raw: string): string {
  const jieba = getJieba();
  if (!jieba) return raw;

  // Use `cutForSearch` (search-engine mode) for indexing — it produces both
  // full words AND their sub-word components. This ensures that query-side
  // tokens (also produced by `cutForSearch` in `buildFtsQuery`) will always
  // find a match in the index.
  const tokens = jieba.cutForSearch(raw, true);

  // Join with spaces so `unicode61` tokenizer can split them.
  // Punctuation tokens are kept — unicode61 treats them as separators anyway.
  return tokens.join(" ");
}

/**
 * Reset jieba state so next call to `buildFtsQuery` re-initialises.
 * Exported for testing only.
 * @internal
 */
export function _resetJiebaForTest(): void {
  _jieba = undefined;
}

/**
 * Override jieba instance (or set to `null` to force fallback).
 * Exported for testing only.
 * @internal
 */
export function _setJiebaForTest(instance: JiebaInstance | null): void {
  _jieba = instance;
}

/**
 * Convert a BM25 rank (negative = more relevant) to a 0–1 score.
 * Mirrors the formula in openclaw core `hybrid.ts`.
 */
export function bm25RankToScore(rank: number): number {
  if (!Number.isFinite(rank)) return 1 / (1 + 999);
  if (rank < 0) {
    const relevance = -rank;
    return relevance / (1 + relevance);
  }
  return 1 / (1 + rank);
}

/** FTS5 search result for L1 records. */
export interface FtsSearchResult {
  record_id: string;
  content: string;
  type: string;
  priority: number;
  scene_name: string;
  /** BM25-derived score (0–1, higher is better) */
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

/** FTS5 search result for L0 records. */
export interface L0FtsSearchResult {
  record_id: string;
  session_key: string;
  session_id: string;
  team_id: string;
  task_id: string;
  user_id: string;
  agent_id: string;
  role: string;
  message_text: string;
  /** BM25-derived score (0–1, higher is better) */
  score: number;
  recorded_at: string;
  timestamp: number;
}

// ============================
// VectorStore class
// ============================

export class VectorStore implements IMemoryStore {
  private db: DatabaseSync;
  private readonly dimensions: number;
  private readonly logger?: Logger;

  /** @see IMemoryStore.supportsDeferredEmbedding */
  readonly supportsDeferredEmbedding = true;

  /**
   * When `true`, the store is in a degraded state (e.g. sqlite-vec failed to
   * load, or init() encountered an unrecoverable error).  All public methods
   * become safe no-ops so the plugin never blocks the main OpenClaw flow.
   */
  private degraded = false;

  /** Tracks whether close() has been called to prevent double-close errors. */
  private closed = false;

  /**
   * `true` when vec0 virtual tables (l1_vec / l0_vec) have been created and
   * their prepared statements are ready.  When `dimensions === 0` (i.e.
   * provider="none"), vec0 tables are deferred and this stays `false`.
   */
  private vecTablesReady = false;

  // Prepared statements — L1 (initialized in init())
  private stmtUpsertMeta!: StatementSync;
  private stmtDeleteVec?: StatementSync;   // optional — only set when vecTablesReady
  private stmtInsertVec?: StatementSync;   // optional — only set when vecTablesReady
  private stmtDeleteMeta!: StatementSync;
  private stmtGetMeta!: StatementSync;
  private stmtSearchVec?: StatementSync;   // optional — only set when vecTablesReady
  private stmtQueryBySessionId!: StatementSync;
  private stmtQueryBySessionIdSince!: StatementSync;
  private stmtQueryBySessionKey!: StatementSync;
  private stmtQueryBySessionKeySince!: StatementSync;
  private stmtQueryAll!: StatementSync;
  private stmtQueryAllSince!: StatementSync;

  // Prepared statements — L0 (initialized in init())
  private stmtL0UpsertMeta!: StatementSync;
  private stmtL0DeleteVec?: StatementSync;   // optional — only set when vecTablesReady
  private stmtL0InsertVec?: StatementSync;   // optional — only set when vecTablesReady
  private stmtL0DeleteMeta!: StatementSync;
  private stmtL0GetMeta!: StatementSync;
  private stmtL0SearchVec?: StatementSync;   // optional — only set when vecTablesReady
  /** L0 query for L1 runner: all messages for a session key */
  private stmtL0QueryAll!: StatementSync;
  /** L0 query for L1 runner: messages after a timestamp cursor */
  private stmtL0QueryAfter!: StatementSync;
  /** L1 cursor-based pagination for migration (by PK) */
  private stmtL1QueryMigrationCursor!: StatementSync;
  /** L0 cursor-based pagination for migration (by PK) */
  private stmtL0QueryMigrationCursor!: StatementSync;

  // FTS5 tables availability flag (created best-effort — may be false if fts5 is not compiled in)
  private ftsAvailable = false;

  // Prepared statements — FTS5 L1 (initialized in init())
  private stmtL1FtsInsert!: StatementSync;
  private stmtL1FtsDelete!: StatementSync;
  private stmtL1FtsSearch!: StatementSync;

  // Prepared statements — FTS5 L0 (initialized in init())
  private stmtL0FtsInsert!: StatementSync;
  private stmtL0FtsDelete!: StatementSync;
  private stmtL0FtsSearch!: StatementSync;

  /**
   * Create a VectorStore instance.
   *
   * Note: After construction, you MUST call `init()` to load the sqlite-vec
   * extension and create the schema.
   */
  constructor(dbPath: string, dimensions: number, logger?: Logger) {
    this.dimensions = dimensions;
    this.logger = logger;

    // Ensure parent directory exists (for non-default instance paths)
    const dbDir = path.dirname(dbPath);
    if (!existsSync(dbDir)) {
      mkdirSync(dbDir, { recursive: true });
    }

    // Open database with extension support enabled
    const { DatabaseSync: DbSync } = requireNodeSqlite();
    this.db = new DbSync(dbPath, { allowExtension: true });

    // Set busy timeout so concurrent processes retry instead of failing with SQLITE_BUSY
    this.db.exec("PRAGMA busy_timeout = 5000");

    // Enable WAL mode for better concurrent read performance
    this.db.exec("PRAGMA journal_mode = WAL");

    // Cap page cache at 64 MB
    this.db.exec("PRAGMA cache_size = -65536");

    // Cap memory-mapped I/O at 128 MB to bound RSS growth
    this.db.exec("PRAGMA mmap_size = 134217728");

    // Auto-checkpoint WAL every 1000 pages (~4 MB) to keep WAL file compact
    this.db.exec("PRAGMA wal_autocheckpoint = 1000");
  }

  /**
   * Expose the underlying `DatabaseSync` handle for tightly-coupled co-tenants
   * that need to live in the SAME SQLite connection (skill_meta / skill_fts /
   * skill_vec live alongside l1_records — see SKILL_ENGINEERING_DESIGN §13.3).
   *
   * Intentional escape hatch: do NOT use this for unrelated stores. Sharing
   * the connection is what gives us one sqlite-vec load + one WAL session +
   * cross-table transactions; opening a second connection on `vectors.db`
   * would defeat all three.
   */
  getRawDb(): DatabaseSync {
    return this.db;
  }

  /** Embedding dimension this store was opened with (0 when provider="none"). */
  getEmbeddingDimensions(): number {
    return this.dimensions;
  }

  /**
   * Whether the store is in degraded mode (e.g. sqlite-vec failed to load).
   * When degraded, all write/search operations become safe no-ops.
   */
  isDegraded(): boolean {
    return this.degraded;
  }


  /**
   * Load sqlite-vec extension and initialize database schema.
   * Must be called once after construction.
   *
   * @param providerInfo  Current embedding provider info. When provided,
   *   the store compares it against the persisted metadata. If the provider,
   *   model, or dimensions changed, the vector tables are dropped and
   *   re-created with the new dimensions, and `needsReindex: true` is returned
   *   so the caller can schedule a full re-embed.
   */
  init(providerInfo?: EmbeddingProviderInfo): VectorStoreInitResult {
    // Load sqlite-vec extension only when vector tables are needed.
    // dimensions=0 is a supported metadata/FTS-only mode and must not degrade
    // just because sqlite-vec is unavailable in the local test/runtime build.
    if (this.dimensions > 0) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const sqliteVec = require("sqlite-vec");
        this.db.enableLoadExtension(true);
        sqliteVec.load(this.db);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this.logger?.error(
          `${TAG} Failed to load sqlite-vec extension: ${message}. ` +
          `VectorStore entering degraded mode — all operations will be no-ops.`,
        );
        this.degraded = true;
        return { needsReindex: false, reason: `sqlite-vec load failed: ${message}` };
      }
    }

    // ── Schema creation & prepared statements ──────────────────────────────
    // Wrapped in try-catch: if anything fails during schema init (e.g. the DB
    // is corrupted, disk full, etc.), we degrade gracefully instead of crashing.
    try {
      return this.initSchema(providerInfo);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.logger?.error(
        `${TAG} Schema initialization failed: ${message}. ` +
        `VectorStore entering degraded mode.`,
      );
      this.degraded = true;
      return { needsReindex: false, reason: `schema init failed: ${message}` };
    }
  }

  /**
   * Internal schema initialization — separated from init() so we can
   * catch errors at the top level and degrade gracefully.
   */
  private initSchema(providerInfo?: EmbeddingProviderInfo): VectorStoreInitResult {
    // Tracks which provider/model/dimensions were used to generate vectors.
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS embedding_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    `);

    // Detect whether re-index is needed
    let needsReindex = false;
    let reindexReason: string | undefined;

    const savedMeta = this.readEmbeddingMeta();

    if (providerInfo) {
      if (savedMeta) {
        const providerChanged = savedMeta.provider !== providerInfo.provider;
        const modelChanged = savedMeta.model !== providerInfo.model;
        const dimsChanged = savedMeta.dimensions !== this.dimensions;

        if (providerChanged || modelChanged || dimsChanged) {
          const reasons: string[] = [];
          if (providerChanged) reasons.push(`provider: ${savedMeta.provider} → ${providerInfo.provider}`);
          if (modelChanged) reasons.push(`model: ${savedMeta.model} → ${providerInfo.model}`);
          if (dimsChanged) reasons.push(`dimensions: ${savedMeta.dimensions} → ${this.dimensions}`);
          reindexReason = reasons.join(", ");

          this.logger?.info(
            `${TAG} Embedding config changed (${reindexReason}). ` +
            `Dropping vector tables for rebuild...`,
          );

          // Drop and re-create vector tables with new dimensions
          this.dropVectorTables();
          needsReindex = true;
        }
      } else {
        // No saved meta — first run or legacy DB without meta table.
        // Two cases require dropping vector tables:
        // 1. Existing data created without meta tracking (legacy DB) — need re-embed
        // 2. vec0 tables exist with wrong dimensions (e.g. previously created with
        //    provider="none" placeholder 768D, now switching to a real provider
        //    with different dimensions) — must rebuild even if data tables are empty
        const l1Count = this.tableRowCount("l1_records");
        const l0Count = this.tableRowCount("l0_conversations");
        const existingVecDims = this.getVecTableDimensions();

        if (l1Count > 0 || l0Count > 0) {
          this.logger?.info(
            `${TAG} No embedding_meta found but existing data exists ` +
            `(L1=${l1Count}, L0=${l0Count}). Dropping vector tables for safety...`,
          );
          this.dropVectorTables();
          needsReindex = true;
          reindexReason = "legacy DB without embedding_meta — cannot verify vector compatibility";
        } else if (existingVecDims !== null && existingVecDims !== this.dimensions) {
          // vec0 tables exist (from a previous provider="none" placeholder or
          // different config) but with mismatched dimensions.  Drop them so they
          // get re-created with the correct dimensions below.
          this.logger?.info(
            `${TAG} vec0 table dimension mismatch (existing=${existingVecDims}, ` +
            `required=${this.dimensions}). Dropping vector tables for rebuild...`,
          );
          this.dropVectorTables();
          // No needsReindex — there's no data to re-embed
        }
      }
    }

    // ── L1 schema ──────────────────────────────────

    // Metadata table
    // NOTE: user_id / agent_id added for three-dim tenancy isolation
    //       (see docs/l0l3-tenant-isolation-design.md).
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS l1_records (
        record_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        type TEXT DEFAULT '',
        priority INTEGER DEFAULT 50,
        scene_name TEXT DEFAULT '',
        session_key TEXT DEFAULT '',
        session_id TEXT DEFAULT 'default',
        team_id TEXT DEFAULT 'default',
        task_id TEXT DEFAULT '',
        user_id TEXT NOT NULL DEFAULT 'default',
        agent_id TEXT NOT NULL DEFAULT 'default',
        version INTEGER NOT NULL DEFAULT 0,
        timestamp_str TEXT DEFAULT '',
        timestamp_start TEXT DEFAULT '',
        timestamp_end TEXT DEFAULT '',
        created_time TEXT DEFAULT '',
        updated_time TEXT DEFAULT '',
        metadata_json TEXT DEFAULT '{}'
      )
    `);

    // Online migration: pre-isolation DBs lack user_id/agent_id columns. ALTER ADD is
    // idempotent-safe: a try/catch around each statement is the SQLite-3 standard idiom.
    try { this.db.exec("ALTER TABLE l1_records ADD COLUMN team_id TEXT DEFAULT 'default'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l1_records ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l1_records ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l1_records ADD COLUMN task_id TEXT DEFAULT ''"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l1_records ADD COLUMN version INTEGER NOT NULL DEFAULT 0"); } catch { /* exists */ }
    this.db.prepare("UPDATE l1_records SET team_id = ? WHERE team_id = '' OR team_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.prepare("UPDATE l1_records SET user_id = ? WHERE user_id = '' OR user_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.prepare("UPDATE l1_records SET agent_id = ? WHERE agent_id = '' OR agent_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.prepare("UPDATE l1_records SET session_id = ? WHERE session_id = '' OR session_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.exec("UPDATE l1_records SET version = 0 WHERE version IS NULL OR version < 0");

    // Indexes for common queries
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_type ON l1_records(type)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_session_key ON l1_records(session_key)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_session_id ON l1_records(session_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_scene ON l1_records(scene_name)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_ts_start ON l1_records(timestamp_start)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_ts_end ON l1_records(timestamp_end)");
    // Composite index: session_id exact match + updated_time range scan (for incremental L2 queries)
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_session_updated ON l1_records(session_id, updated_time)");
    // Composite index: session_key exact match + updated_time range scan (for pipeline cursor queries)
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_sessionkey_updated ON l1_records(session_key, updated_time)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_task_updated ON l1_records(task_id, updated_time)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_team_agent_updated ON l1_records(team_id, agent_id, updated_time)");
    // Isolation indexes (three-dim tenancy)
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_user_agent_session ON l1_records(user_id, agent_id, session_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_user_updated  ON l1_records(user_id, updated_time)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l1_agent_updated ON l1_records(agent_id, updated_time)");

    // Vector virtual table (cosine distance) — only created when dimensions > 0.
    // When provider="none", dimensions=0 and vec0 tables are deferred until a
    // real embedding provider is configured.
    if (this.dimensions > 0) {
      this.db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS l1_vec USING vec0(
          record_id TEXT PRIMARY KEY,
          embedding float[${this.dimensions}] distance_metric=cosine,
          updated_time TEXT DEFAULT ''
        )
      `);
    }

    // Prepare statements for reuse
    // NOTE: user_id / agent_id appended at the end of the column list so that
    // existing positional bindings in this file remain in the same relative
    // order; new bindings always come last. See upsertL1() for the call-site.
    this.stmtUpsertMeta = this.db.prepare(`
      INSERT INTO l1_records (
        record_id, content, type, priority, scene_name, session_key, session_id,
        team_id, task_id, version, timestamp_str, timestamp_start, timestamp_end,
        created_time, updated_time, metadata_json,
        user_id, agent_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(record_id) DO UPDATE SET
        content=excluded.content,
        type=excluded.type,
        priority=excluded.priority,
        scene_name=excluded.scene_name,
        team_id=excluded.team_id,
        task_id=excluded.task_id,
        version=excluded.version,
        timestamp_str=excluded.timestamp_str,
        timestamp_start=excluded.timestamp_start,
        timestamp_end=excluded.timestamp_end,
        updated_time=excluded.updated_time,
        metadata_json=excluded.metadata_json,
        user_id=excluded.user_id,
        agent_id=excluded.agent_id
    `);

    if (this.dimensions > 0) {
      this.stmtDeleteVec = this.db.prepare("DELETE FROM l1_vec WHERE record_id = ?");
      this.stmtInsertVec = this.db.prepare("INSERT INTO l1_vec (record_id, embedding, updated_time) VALUES (?, ?, ?)");
    }
    this.stmtDeleteMeta = this.db.prepare("DELETE FROM l1_records WHERE record_id = ?");

    this.stmtGetMeta = this.db.prepare(`
      SELECT content, type, priority, scene_name, session_key, session_id, team_id, task_id, user_id, agent_id,
             version, timestamp_str, timestamp_start, timestamp_end, metadata_json
      FROM l1_records WHERE record_id = ?
    `);

    if (this.dimensions > 0) {
      this.stmtSearchVec = this.db.prepare(`
        SELECT record_id, distance
        FROM l1_vec
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
      `);
    }

    // ── L0 schema ──────────────────────────────────

    // L0 metadata table: stores individual messages for vector search.
    // NOTE: user_id / agent_id added for three-dim tenancy isolation
    //       (see docs/l0l3-tenant-isolation-design.md).
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS l0_conversations (
        record_id TEXT PRIMARY KEY,
        session_key TEXT NOT NULL,
        session_id TEXT DEFAULT 'default',
        team_id TEXT DEFAULT 'default',
        task_id TEXT DEFAULT '',
        user_id TEXT NOT NULL DEFAULT 'default',
        agent_id TEXT NOT NULL DEFAULT 'default',
        role TEXT NOT NULL DEFAULT '',
        message_text TEXT NOT NULL,
        recorded_at TEXT DEFAULT '',
        timestamp INTEGER DEFAULT 0
      )
    `);

    // Online migrations: each ADD COLUMN is wrapped in try/catch so re-running
    // init() on an already-migrated DB is a no-op.
    try {
      this.db.exec("ALTER TABLE l0_conversations ADD COLUMN timestamp INTEGER DEFAULT 0");
      this.logger?.debug?.(`${TAG} Migrated l0_conversations: added timestamp column`);
    } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l0_conversations ADD COLUMN team_id TEXT DEFAULT 'default'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l0_conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l0_conversations ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'default'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE l0_conversations ADD COLUMN task_id TEXT DEFAULT ''"); } catch { /* exists */ }
    this.db.prepare("UPDATE l0_conversations SET team_id = ? WHERE team_id = '' OR team_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.prepare("UPDATE l0_conversations SET user_id = ? WHERE user_id = '' OR user_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.prepare("UPDATE l0_conversations SET agent_id = ? WHERE agent_id = '' OR agent_id IS NULL").run(DEFAULT_ISOLATION_ID);
    this.db.prepare("UPDATE l0_conversations SET session_id = ? WHERE session_id = '' OR session_id IS NULL").run(DEFAULT_ISOLATION_ID);

    // Skill schema belongs to the same vectors.db. Initialize its base tables
    // with the SQLite store so a freshly installed in-process OpenClaw plugin
    // has a complete database even when SkillCore is not otherwise activated.
    // SkillCore may call the same idempotent DDL later and initialize skill_vec.
    try {
      this.db.exec(SKILLS_DDL);
      this.db.exec(SKILL_FTS_DDL);
    } catch (err) {
      this.logger?.warn(
        `${TAG} Skill base schema init failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    // Indexes for L0 queries
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_session ON l0_conversations(session_key)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_session_id ON l0_conversations(session_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_task ON l0_conversations(task_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_team_agent ON l0_conversations(team_id, agent_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_recorded ON l0_conversations(recorded_at)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_timestamp ON l0_conversations(timestamp)");
    // Isolation indexes (three-dim tenancy)
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_user_agent_session ON l0_conversations(user_id, agent_id, session_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_user_recorded  ON l0_conversations(user_id, recorded_at)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_l0_agent_recorded ON l0_conversations(agent_id, recorded_at)");

    // L0 vector virtual table (cosine distance, same dimensions as L1) — deferred when dimensions=0
    if (this.dimensions > 0) {
      this.db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS l0_vec USING vec0(
          record_id TEXT PRIMARY KEY,
          embedding float[${this.dimensions}] distance_metric=cosine,
          recorded_at TEXT DEFAULT ''
        )
      `);
    }

    // L0 prepared statements
    // user_id / agent_id appended at the end of the bind list (see upsertL0()).
    this.stmtL0UpsertMeta = this.db.prepare(`
      INSERT INTO l0_conversations (
        record_id, session_key, session_id, team_id, task_id, role, message_text, recorded_at, timestamp,
        user_id, agent_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(record_id) DO UPDATE SET
        message_text=excluded.message_text,
        recorded_at=excluded.recorded_at,
        timestamp=excluded.timestamp,
        team_id=excluded.team_id,
        task_id=excluded.task_id,
        user_id=excluded.user_id,
        agent_id=excluded.agent_id
    `);

    if (this.dimensions > 0) {
      this.stmtL0DeleteVec = this.db.prepare("DELETE FROM l0_vec WHERE record_id = ?");
      this.stmtL0InsertVec = this.db.prepare("INSERT INTO l0_vec (record_id, embedding, recorded_at) VALUES (?, ?, ?)");
    }
    this.stmtL0DeleteMeta = this.db.prepare("DELETE FROM l0_conversations WHERE record_id = ?");

    this.stmtL0GetMeta = this.db.prepare(`
      SELECT session_key, session_id, team_id, task_id, user_id, agent_id, role, message_text, recorded_at, timestamp
      FROM l0_conversations WHERE record_id = ?
    `);

    if (this.dimensions > 0) {
      this.stmtL0SearchVec = this.db.prepare(`
        SELECT record_id, distance
        FROM l0_vec
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
      `);
    }

    // L0 query statements for L1 runner (oldest-first + LIMIT to bound memory).
    //
    // Why ASC: the L1 runner advances `last_l1_cursor` to max(recorded_at) of
    // the batch it just consumed. If we returned newest-first under a backlog,
    // the cursor would jump to the latest record and silently skip the older
    // ones. Returning the oldest `LIMIT` rows above the cursor is the only way
    // to guarantee progress without data loss when there is a backlog.
    //
    // Sort/filter by recorded_at (write time) instead of timestamp (conversation
    // time) because L1 cursor uses recorded_at semantics. ISO 8601 string
    // comparison preserves time order.
    this.stmtL0QueryAll = this.db.prepare(`
      SELECT record_id, session_key, session_id, team_id, task_id, user_id, agent_id, role, message_text, recorded_at, timestamp
      FROM l0_conversations
      WHERE session_key = ?
      ORDER BY recorded_at ASC
      LIMIT ?
    `);

    this.stmtL0QueryAfter = this.db.prepare(`
      SELECT record_id, session_key, session_id, team_id, task_id, user_id, agent_id, role, message_text, recorded_at, timestamp
      FROM l0_conversations
      WHERE session_key = ? AND recorded_at > ?
      ORDER BY recorded_at ASC
      LIMIT ?
    `);

    this.stmtL0QueryMigrationCursor = this.db.prepare(`
      SELECT record_id, session_key, session_id, role, message_text, recorded_at, timestamp
      FROM l0_conversations
      WHERE record_id > ?
      ORDER BY record_id ASC
      LIMIT ?
    `);

    // ── Entity metadata tables (Team / User / Agent / Task) ───────────────
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS entity_teams (
        team_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        owner_user_id TEXT NOT NULL,
        user_ids_json TEXT NOT NULL DEFAULT '[]',
        agent_ids_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS entity_users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        job_description TEXT DEFAULT '',
        team_ids_json TEXT NOT NULL DEFAULT '[]',
        task_ids_json TEXT NOT NULL DEFAULT '[]',
        owned_agent_ids_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS entity_agents (
        agent_id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        prompt TEXT DEFAULT '',
        owner_user_id TEXT DEFAULT '',
        visibility TEXT NOT NULL DEFAULT 'team',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS entity_tasks (
        task_id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        creator_user_id TEXT NOT NULL,
        title TEXT DEFAULT '',
        description TEXT DEFAULT '',
        source_type TEXT NOT NULL DEFAULT 'manual',
        source_url TEXT DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        auto_assign_floating_assets INTEGER NOT NULL DEFAULT 0,
        risk_level TEXT NOT NULL DEFAULT 'low',
        agent_ids_json TEXT NOT NULL DEFAULT '[]',
        user_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS entity_knowledge (
        knowledge_id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        service_url TEXT NOT NULL,
        name TEXT NOT NULL,
        summary TEXT,
        team_id TEXT NOT NULL,
        agent_id TEXT NOT NULL DEFAULT '',
        user_id TEXT,
        repo_url TEXT,
        branch TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `);
    // 在线迁移：老库补 agent_id 列（预留 binding 维度，绑定权威在 meta_assets）
    try { this.db.exec("ALTER TABLE entity_knowledge ADD COLUMN agent_id TEXT NOT NULL DEFAULT ''"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE entity_teams ADD COLUMN user_ids_json TEXT NOT NULL DEFAULT '[]'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE entity_teams ADD COLUMN agent_ids_json TEXT NOT NULL DEFAULT '[]'"); } catch { /* exists */ }
    try { this.db.exec("ALTER TABLE entity_tasks ADD COLUMN user_ids_json TEXT NOT NULL DEFAULT '[]'"); } catch { /* exists */ }
    this.db.exec("DROP INDEX IF EXISTS idx_entity_teams_name");
    this.db.exec("DROP INDEX IF EXISTS idx_entity_agents_team_name");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_teams_name ON entity_teams(name)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_users_status ON entity_users(status)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_agents_team_name ON entity_agents(team_id, name)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_agents_team_status ON entity_agents(team_id, status)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_tasks_team_status ON entity_tasks(team_id, status)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_knowledge_team ON entity_knowledge(team_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_entity_knowledge_team_type ON entity_knowledge(team_id, type)");

    // ── Memory Audit (修改审计) ──
    // 设计要点（per user 决策）：
    //   - 原始 L0/L1/L2/L3 表完全不动，本表只追加事件
    //   - 不存历史 content / 旧值，只记"什么时间、由谁、改了哪条"
    //   - team/agent/user/task 来自外部请求 IdFields（不是 record 原值）
    //   - L0 不参与（不可变流水）；L1/L2/L3 update + delete 各记一条
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memory_audit (
        audit_id      TEXT PRIMARY KEY,
        record_id     TEXT NOT NULL,
        layer         TEXT NOT NULL CHECK (layer IN ('L1','L2','L3')),
        action        TEXT NOT NULL CHECK (action IN ('update','delete')),
        team_id       TEXT,
        agent_id      TEXT,
        user_id       TEXT,
        task_id       TEXT,
        version       INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        request_id    TEXT
      )
    `);
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_memory_audit_record    ON memory_audit(record_id, updated_at_ms)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_memory_audit_isolation ON memory_audit(team_id, agent_id, user_id, task_id)");
    this.db.exec("CREATE INDEX IF NOT EXISTS idx_memory_audit_time      ON memory_audit(updated_at_ms)");

    // ── FTS5 tables (best-effort — gracefully degrade if fts5 is not compiled in) ──
    // Schema v2: `content` column stores jieba-segmented text (for indexing),
    // `content_original` (UNINDEXED) stores the raw text (for display).
    // If old v1 tables exist (no content_original column), drop + recreate.
    try {
      // ── Migrate old FTS5 tables (v1 → v2) ──
      // v1 tables stored raw text in the `content` column. v2 stores segmented
      // text in `content` and raw text in `content_original` / `message_text_original`.
      // FTS5 virtual tables don't support ALTER TABLE ADD COLUMN, so we must
      // drop and recreate. The data will be repopulated by `rebuildFtsIndex()`.
      const needsFtsRebuild = this.migrateFtsTablesIfNeeded();

      // L1 FTS5 virtual table (v2 schema + isolation columns).
      // user_id / agent_id added as UNINDEXED so they're carried alongside
      // every FTS hit without affecting BM25 ranking.
      this.db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS l1_fts USING fts5(
          content,
          content_original UNINDEXED,
          record_id UNINDEXED,
          type UNINDEXED,
          priority UNINDEXED,
          scene_name UNINDEXED,
          session_key UNINDEXED,
          session_id UNINDEXED,
          team_id UNINDEXED,
          task_id UNINDEXED,
          user_id UNINDEXED,
          agent_id UNINDEXED,
          version UNINDEXED,
          timestamp_str UNINDEXED,
          timestamp_start UNINDEXED,
          timestamp_end UNINDEXED,
          metadata_json UNINDEXED
        )
      `);

      // L0 FTS5 virtual table (v2 schema + isolation columns).
      this.db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS l0_fts USING fts5(
          message_text,
          message_text_original UNINDEXED,
          record_id UNINDEXED,
          session_key UNINDEXED,
          session_id UNINDEXED,
          team_id UNINDEXED,
          task_id UNINDEXED,
          user_id UNINDEXED,
          agent_id UNINDEXED,
          role UNINDEXED,
          recorded_at UNINDEXED,
          timestamp UNINDEXED
        )
      `);

      // L1 FTS prepared statements
      this.stmtL1FtsInsert = this.db.prepare(`
        INSERT INTO l1_fts (content, content_original, record_id, type, priority, scene_name,
          session_key, session_id, team_id, task_id, user_id, agent_id, version,
          timestamp_str, timestamp_start, timestamp_end, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      this.stmtL1FtsDelete = this.db.prepare("DELETE FROM l1_fts WHERE record_id = ?");

      this.stmtL1FtsSearch = this.db.prepare(`
        SELECT record_id, content_original AS content, type, priority, scene_name,
               session_key, session_id, team_id, task_id, user_id, agent_id, version,
               timestamp_str, timestamp_start, timestamp_end,
               metadata_json,
               bm25(l1_fts) AS rank
        FROM l1_fts
        WHERE l1_fts MATCH ?
        ORDER BY rank ASC
        LIMIT ?
      `);

      // L0 FTS prepared statements
      this.stmtL0FtsInsert = this.db.prepare(`
        INSERT INTO l0_fts (message_text, message_text_original, record_id,
          session_key, session_id, team_id, task_id, user_id, agent_id, role, recorded_at, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      this.stmtL0FtsDelete = this.db.prepare("DELETE FROM l0_fts WHERE record_id = ?");

      this.stmtL0FtsSearch = this.db.prepare(`
        SELECT record_id, message_text_original AS message_text,
               session_key, session_id, team_id, task_id, user_id, agent_id, role, recorded_at, timestamp,
               bm25(l0_fts) AS rank
        FROM l0_fts
        WHERE l0_fts MATCH ?
        ORDER BY rank ASC
        LIMIT ?
      `);

      this.ftsAvailable = true;
      this.logger?.debug?.(`${TAG} FTS5 tables initialized (l1_fts, l0_fts) [schema v2 — jieba segmented]`);

      // Rebuild FTS index if migrated from v1 or tables were freshly created
      if (needsFtsRebuild) {
        this.rebuildFtsIndex();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.ftsAvailable = false;
      this.logger?.warn(
        `${TAG} FTS5 tables NOT available (fts5 may not be compiled in): ${message}. ` +
        `FTS-based keyword search will be unavailable; recall will use in-memory scoring if needed.`,
      );
    }

    // Save current embedding meta (write after schema is ready)
    if (providerInfo) {
      this.writeEmbeddingMeta({
        provider: providerInfo.provider,
        model: providerInfo.model,
        dimensions: this.dimensions,
      });
    }

    // Mark vec0 tables as ready only when they were actually created
    this.vecTablesReady = this.dimensions > 0;
    // L1 query statements (for l1-reader)
    // user_id / agent_id surfaced in every L1 read so callers (router /
    // candidate-pool / l1-reader) can enforce isolation downstream.
    const l1QueryCols = `record_id, content, type, priority, scene_name, session_key, session_id,
      team_id, task_id, user_id, agent_id, version,
      timestamp_str, timestamp_start, timestamp_end,
      created_time, updated_time, metadata_json`;

    this.stmtQueryBySessionId = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      WHERE session_id = ?
      ORDER BY updated_time ASC
    `);

    this.stmtQueryBySessionIdSince = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      WHERE session_id = ? AND updated_time > ?
      ORDER BY updated_time ASC
    `);

    this.stmtQueryBySessionKey = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      WHERE session_key = ?
      ORDER BY updated_time ASC
    `);

    this.stmtQueryBySessionKeySince = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      WHERE session_key = ? AND updated_time > ?
      ORDER BY updated_time ASC
    `);

    this.stmtQueryAll = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      ORDER BY updated_time ASC
    `);

    this.stmtQueryAllSince = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      WHERE updated_time > ?
      ORDER BY updated_time ASC
    `);

    this.stmtL1QueryMigrationCursor = this.db.prepare(`
      SELECT ${l1QueryCols} FROM l1_records
      WHERE record_id > ?
      ORDER BY record_id ASC
      LIMIT ?
    `);

    this.logger?.debug?.(`${TAG} Initialized (dimensions=${this.dimensions})`);

    return { needsReindex, reason: reindexReason };
  }

  // ── Embedding meta helpers ──────────────────────────────

  private readEmbeddingMeta(): EmbeddingMeta | null {
    try {
      const row = this.db
        .prepare("SELECT value FROM embedding_meta WHERE key = ?")
        .get("embedding_provider_info") as { value: string } | undefined;
      if (!row) return null;
      return JSON.parse(row.value) as EmbeddingMeta;
    } catch {
      return null;
    }
  }

  private writeEmbeddingMeta(meta: EmbeddingMeta): void {
    this.db.prepare(
      "INSERT INTO embedding_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
    ).run("embedding_provider_info", JSON.stringify(meta));
  }

  /** Allowed table names for row counting (whitelist to prevent SQL injection). */
  private static readonly COUNTABLE_TABLES = new Set(["l1_records", "l0_conversations"]);

  /**
   * Extra rows to retrieve from vec0 KNN search to compensate for legacy
   * zero-vector placeholders that may still linger from older data.
   */
  private static readonly ZERO_VEC_BUFFER = 10;

  /** Default result limit for FTS5 keyword searches. */
  private static readonly FTS_DEFAULT_LIMIT = 20;

  private tableRowCount(table: string): number {
    if (!VectorStore.COUNTABLE_TABLES.has(table)) {
      this.logger?.warn(`${TAG} tableRowCount: rejected unknown table name "${table}"`);
      return 0;
    }
    try {
      const row = this.db
        .prepare(`SELECT COUNT(*) AS cnt FROM ${table}`)
        .get() as { cnt: number } | undefined;
      return row?.cnt ?? 0;
    } catch {
      return 0;
    }
  }

  /**
   * Detect the embedding dimension of an existing vec0 table by inspecting
   * the DDL stored in sqlite_master.  Returns `null` if the table doesn't
   * exist or the dimension cannot be determined.
   *
   * The vec0 DDL looks like:
   *   CREATE VIRTUAL TABLE l1_vec USING vec0(... embedding float[768] ...)
   * We parse the number inside `float[N]`.
   */
  private getVecTableDimensions(): number | null {
    try {
      const row = this.db
        .prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name=?")
        .get("l1_vec") as { sql: string } | undefined;
      if (!row?.sql) return null;
      const match = row.sql.match(/float\[(\d+)\]/);
      return match ? Number(match[1]) : null;
    } catch {
      return null;
    }
  }

  /**
   * Drop both L1 and L0 vector virtual tables.
   * Metadata tables (l1_records, l0_conversations) are preserved — only
   * the vec0 tables need to be rebuilt with the new dimensions.
   */
  private dropVectorTables(): void {
    this.db.exec("DROP TABLE IF EXISTS l1_vec");
    this.db.exec("DROP TABLE IF EXISTS l0_vec");
    this.logger?.info(`${TAG} Dropped vector tables (l1_vec, l0_vec)`);
  }

  /**
   * Write or update a memory record (metadata + vector).
   * Uses a manual transaction for atomicity.
   *
   * If `embedding` is `undefined` or a zero vector (all elements are 0), only
   * the metadata row is written — the vec0 table is left untouched.  This
   * allows callers without an EmbeddingService to still persist metadata + FTS
   * without constructing a throwaway zero-vector, and prevents placeholder
   * zero vectors (from embedding-service failures) from polluting KNN search
   * results with null / NaN distances.
   *
   * **Fault-tolerant**: catches all errors internally so that a vector store
   * failure never propagates to the caller / main OpenClaw flow.
   * Returns `true` on success, `false` on failure (logged as warning).
   */
  upsertL1(record: MemoryRecord, embedding: Float32Array | undefined): boolean {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [L1-upsert] SKIPPED (degraded mode) id=${record.id}`);
      return false;
    }
    try {
      const { id: recordId, timestamps } = record;
      const tsStr = timestamps[0] ?? "";
      const tsStart =
        timestamps.length > 0
          ? timestamps.reduce((a, b) => (a < b ? a : b))
          : tsStr;
      const tsEnd =
        timestamps.length > 0
          ? timestamps.reduce((a, b) => (a > b ? a : b))
          : tsStr;

      const skipVec = !embedding || embedding.every(v => v === 0) || !this.vecTablesReady;

      this.logger?.debug?.(
        `${TAG} [L1-upsert] START id=${recordId}, type=${record.type}, ` +
        `content="${record.content.slice(0, 60)}..."` +
        (embedding
          ? `, embeddingDims=${embedding.length}, ` +
            `embeddingNorm=${Math.sqrt(Array.from(embedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}` +
            `${skipVec ? " (ZERO VECTOR or vec tables not ready — vec write will be skipped)" : ""}`
          : " (no embedding — metadata-only write)"),
      );

      this.db.exec("BEGIN");
      try {
        // Upsert metadata (INSERT OR UPDATE).
        // user_id / agent_id appended at the end to match the prepared statement
        // column order added by the isolation migration. We tolerate undefined
        // for legacy callers (e.g. older tests or pre-isolation seed scripts):
        // empty string preserves the historic "no-isolation" semantics until the
        // migration backfills.
        this.stmtUpsertMeta.run(
          recordId,
          record.content,
          record.type,
          record.priority,
          record.scene_name,
          record.sessionKey,
          record.sessionId || DEFAULT_ISOLATION_ID,
          (record as MemoryRecord & { teamId?: string }).teamId || DEFAULT_ISOLATION_ID,
          record.taskId || "",
          record.version ?? 0,
          tsStr,
          tsStart,
          tsEnd,
          record.createdAt,
          record.updatedAt,
          JSON.stringify(record.metadata),
          (record as MemoryRecord & { userId?: string }).userId || DEFAULT_ISOLATION_ID,
          (record as MemoryRecord & { agentId?: string }).agentId || DEFAULT_ISOLATION_ID,
        );

        if (!skipVec) {
          // vec0 does not support ON CONFLICT → delete then insert
          this.stmtDeleteVec!.run(recordId);
          this.stmtInsertVec!.run(recordId, Buffer.from(embedding!.buffer), record.updatedAt);
        } else {
          this.logger?.debug?.(
            `${TAG} [L1-upsert] Skipping vec write (${embedding ? "zero vector" : "no embedding"}) id=${recordId}`,
          );
        }

        // Sync FTS5 (delete + re-insert to handle updates).
        // user_id / agent_id mirrored into FTS so post-recall isolation
        // filtering doesn't require a join.
        if (this.ftsAvailable) {
          try {
            this.stmtL1FtsDelete.run(recordId);
            this.stmtL1FtsInsert.run(
              tokenizeForFts(record.content), // content — segmented for indexing
              record.content,                 // content_original — raw for display
              recordId,
              record.type,
              record.priority,
              record.scene_name,
              record.sessionKey,
              record.sessionId || DEFAULT_ISOLATION_ID,
              (record as MemoryRecord & { teamId?: string }).teamId || DEFAULT_ISOLATION_ID,
              record.taskId || "",
              (record as MemoryRecord & { userId?: string }).userId || DEFAULT_ISOLATION_ID,
              (record as MemoryRecord & { agentId?: string }).agentId || DEFAULT_ISOLATION_ID,
              record.version ?? 0,
              tsStr,
              tsStart,
              tsEnd,
              JSON.stringify(record.metadata),
            );
          } catch (ftsErr) {
            // FTS write failure is non-fatal — log and continue
            this.logger?.warn(
              `${TAG} [L1-upsert] FTS write failed (non-fatal) id=${recordId}: ${ftsErr instanceof Error ? ftsErr.message : String(ftsErr)}`,
            );
          }
        }

        this.db.exec("COMMIT");
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
      this.logger?.debug?.(`${TAG} [L1-upsert] OK id=${recordId}${skipVec ? " (meta-only)" : ""}`);
      return true;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L1-upsert] FAILED (non-fatal) id=${record.id}: ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * Vector similarity search (cosine distance).
   * Returns top-k results sorted by similarity (highest first).
   *
   * **Fault-tolerant**: returns an empty array on any error (e.g. dimension
   * mismatch, corrupted DB) so callers can fall back to keyword search.
   */
  searchL1Vector(queryEmbedding: Float32Array, topK = 5, _queryText?: string, filter?: IsolationFilter): VectorSearchResult[] {
    if (this.degraded || !this.vecTablesReady) {
      if (this.degraded) this.logger?.warn(`${TAG} [L1-search] SKIPPED (degraded mode)`);
      return [];
    }
    try {
      // Over-retrieve to compensate for legacy zero-vector placeholders that
      // may still exist in the vec0 table.  New zero vectors are no longer
      // inserted (upsert() skips vec write for zero vectors since v3.x), but
      // older data may still contain them — they surface as NULL/NaN distance
      // in KNN results.  A small buffer of 10 is sufficient for remnants.
      // NOTE: "AND distance IS NOT NULL" is NOT usable because vec0 does not
      // support that constraint — it causes an empty result set.
      const ZERO_VEC_BUFFER = 10;
      const retrieveCount = filter ? Math.max(topK * 5, topK + ZERO_VEC_BUFFER) : topK + ZERO_VEC_BUFFER;

      this.logger?.debug?.(
        `${TAG} [L1-search] START topK=${topK}, retrieveCount=${retrieveCount}, ` +
        `queryEmbeddingDims=${queryEmbedding.length}, ` +
        `queryNorm=${Math.sqrt(Array.from(queryEmbedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}`,
      );

      const rows = this.stmtSearchVec!.all(
        Buffer.from(queryEmbedding.buffer),
        retrieveCount,
      ) as Array<{ record_id: string; distance: number }>;

      this.logger?.debug?.(`${TAG} [L1-search] vec0 returned ${rows.length} candidate(s)`);

      if (rows.length === 0) return [];

      const results: VectorSearchResult[] = [];

      for (const { record_id, distance } of rows) {
        // sqlite-vec returns null distance for zero vectors (cosine undefined when ‖v‖=0).
        // Skip these — they are placeholder vectors from embedding-service-unavailable fallback.
        if (distance == null || Number.isNaN(distance)) {
          this.logger?.warn(
            `${TAG} [L1-search] record_id=${record_id} has null/NaN distance (likely zero vector) — skipping`,
          );
          continue;
        }

        const meta = this.stmtGetMeta.get(record_id) as
          | {
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
              metadata_json: string;
            }
          | undefined;

        if (!meta) {
          this.logger?.warn(`${TAG} [L1-search] record_id=${record_id} has vector but NO metadata (orphan)`);
          continue;
        }
        if (!rowMatchesIsolation(meta, filter)) {
          continue;
        }

        const score = 1.0 - distance;
        this.logger?.debug?.(
          `${TAG} [L1-search] HIT id=${record_id}, distance=${distance.toFixed(4)}, score=${score.toFixed(4)}, ` +
          `type=${meta.type}, content="${meta.content.slice(0, 60)}..."`,
        );

        results.push({
          record_id,
          content: meta.content,
          type: meta.type,
          priority: meta.priority,
          scene_name: meta.scene_name,
          score,
          timestamp_str: meta.timestamp_str,
          timestamp_start: meta.timestamp_start,
          timestamp_end: meta.timestamp_end,
          version: meta.version ?? 0,
          session_key: meta.session_key,
          session_id: meta.session_id,
          team_id: meta.team_id ?? "",
          task_id: meta.task_id ?? "",
          user_id: meta.user_id ?? "",
          agent_id: meta.agent_id ?? "",
          metadata_json: meta.metadata_json,
        });
      }

      // Trim back to the caller's requested topK (we over-fetched above).
      const trimmed = results.slice(0, topK);
      this.logger?.info(
        `${TAG} [L1-search] DONE returning ${trimmed.length} result(s) (from ${results.length} valid, ${rows.length} raw)`,
      );
      return trimmed;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L1-search] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * Delete a single record (metadata + vector).
   *
   * **Fault-tolerant**: logs a warning on failure, never throws.
   */
  deleteL1(recordId: string, filter?: IsolationFilter): boolean {
    if (this.degraded) return false;
    try {
      if (filter) {
        const meta = this.stmtGetMeta.get(recordId) as { user_id?: string; agent_id?: string; session_id?: string; session_key?: string } | undefined;
        if (!meta || !rowMatchesIsolation(meta, filter)) return false;
      }
      this.db.exec("BEGIN");
      try {
        const result = this.stmtDeleteMeta.run(recordId);
        const deleted = (result as any)?.changes > 0;
        if (this.vecTablesReady) this.stmtDeleteVec!.run(recordId);
        if (this.ftsAvailable) {
          try { this.stmtL1FtsDelete.run(recordId); } catch { /* non-fatal */ }
        }
        this.db.exec("COMMIT");
        return deleted;
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
    } catch (err) {
      this.logger?.warn(
        `${TAG} delete failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * Delete multiple records (metadata + vector).
   *
   * **Fault-tolerant**: logs a warning on failure, never throws.
   */
  deleteL1Batch(recordIds: string[], filter?: IsolationFilter): boolean {
    if (this.degraded) return false;
    if (recordIds.length === 0) return true;

    try {
      this.db.exec("BEGIN");
      try {
        for (const id of recordIds) {
          if (filter) {
            const meta = this.stmtGetMeta.get(id) as { user_id?: string; agent_id?: string; session_id?: string; session_key?: string } | undefined;
            if (!meta || !rowMatchesIsolation(meta, filter)) continue;
          }
          this.stmtDeleteMeta.run(id);
          if (this.vecTablesReady) this.stmtDeleteVec!.run(id);
          if (this.ftsAvailable) {
            try { this.stmtL1FtsDelete.run(id); } catch { /* non-fatal */ }
          }
        }
        this.db.exec("COMMIT");
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
      return true;
    } catch (err) {
      this.logger?.warn(
        `${TAG} deleteBatch failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * Get the total number of L1 records in the store.
   *
   * **Fault-tolerant**: returns 0 on failure.
   * TTL cleanup by updated_time.
   *
   * Deletes expired rows from l1_records and matching vectors from l1_vec
   * in a single transaction to guarantee consistency.
   */
  deleteL1Expired(cutoffIso: string): number {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [deleteExpired] SKIPPED (degraded mode)`);
      return 0;
    }
    try {
      const row = this.db.prepare(
        "SELECT COUNT(*) AS cnt FROM l1_records WHERE updated_time != '' AND updated_time < ?",
      ).get(cutoffIso) as { cnt: number } | undefined;
      const expiredCount = row?.cnt ?? 0;
      if (expiredCount <= 0) return 0;

      // Ratio protection: refuse to delete > 80% in one pass
      const totalRow = this.db.prepare(
        "SELECT COUNT(*) AS cnt FROM l1_records",
      ).get() as { cnt: number };
      const total = totalRow.cnt;
      const ratio = total > 0 ? expiredCount / total : 0;
      if (ratio > 0.8) {
        this.logger?.warn(
          `${TAG} [L1-deleteExpired] BLOCKED: would delete ${expiredCount}/${total} ` +
          `(${(ratio * 100).toFixed(1)}%) — exceeds 80% safety threshold, cutoff=${cutoffIso}`,
        );
        return 0;
      }

      this.db.exec("BEGIN");
      try {
        if (this.vecTablesReady) {
          this.db.prepare(
            "DELETE FROM l1_vec WHERE updated_time != '' AND updated_time < ?",
          ).run(cutoffIso);
        }
        this.db.prepare(
          "DELETE FROM l1_records WHERE updated_time != '' AND updated_time < ?",
        ).run(cutoffIso);
        this.db.exec("COMMIT");
        this.logger?.info?.(
          `${TAG} [L1-deleteExpired] Deleted ${expiredCount}/${total} records (cutoff=${cutoffIso})`,
        );
        return expiredCount;
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
    } catch (err) {
      this.logger?.warn(
        `${TAG} deleteL1ExpiredByUpdatedTime failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return 0;
    }
  }

  /**
   * Get the total number of L1 records matching optional filters.
   */
  countL1(filter?: L1CountFilter): number {
    if (this.degraded) return 0;
    try {
      const conditions: string[] = [];
      const params: SQLInputValue[] = [];

      if (filter?.type) {
        conditions.push("type = ?");
        params.push(filter.type);
      }
      if (filter?.sessionId) {
        conditions.push("session_id = ?");
        params.push(filter.sessionId);
      }
      if (filter?.teamId !== undefined) {
        conditions.push("team_id = ?");
        params.push(filter.teamId);
      }
      if (filter?.userId !== undefined) {
        conditions.push("user_id = ?");
        params.push(filter.userId);
      }
      if (filter?.agentId !== undefined) {
        conditions.push("agent_id = ?");
        params.push(filter.agentId);
      }
      if (filter?.taskId !== undefined) {
        conditions.push("task_id = ?");
        params.push(filter.taskId);
      }
      if (filter?.timeStart) {
        conditions.push("updated_time >= ?");
        params.push(filter.timeStart);
      }
      if (filter?.timeEnd) {
        conditions.push("updated_time <= ?");
        params.push(filter.timeEnd);
      }

      const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
      const row = this.db
        .prepare(`SELECT COUNT(*) AS cnt FROM l1_records ${where}`)
        .get(...params) as { cnt: number } | undefined;
      const total = row?.cnt ?? 0;
      this.logger?.debug?.(`${TAG} [L1-count] total=${total}`);
      return total;
    } catch (err) {
      this.logger?.warn(
        `${TAG} count failed (non-fatal, returning 0): ${err instanceof Error ? err.message : String(err)}`,
      );
      return 0;
    }
  }

  /**
   * Query L1 records with optional session and time filters.
   *
   * Uses the composite index `idx_l1_session_updated(session_id, updated_time)`
   * for efficient filtering. All timestamps are compared as UTC ISO 8601 strings.
   *
   * **Fault-tolerant**: returns an empty array on any error (degraded mode, DB issues).
   */
  queryL1Records(filter?: L1QueryFilter): L1RecordRow[] {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [L1-query] SKIPPED (degraded mode)`);
      return [];
    }
    try {
      const { sessionKey, sessionId, taskId, updatedAfter } = filter ?? {};

      let raw: Record<string, unknown>[];

      // Priority: sessionId > sessionKey (sessionId is more specific)
      if (sessionId && updatedAfter) {
        raw = this.stmtQueryBySessionIdSince.all(sessionId, updatedAfter) as Record<string, unknown>[];
      } else if (sessionId) {
        raw = this.stmtQueryBySessionId.all(sessionId) as Record<string, unknown>[];
      } else if (sessionKey && updatedAfter) {
        raw = this.stmtQueryBySessionKeySince.all(sessionKey, updatedAfter) as Record<string, unknown>[];
      } else if (sessionKey) {
        raw = this.stmtQueryBySessionKey.all(sessionKey) as Record<string, unknown>[];
      } else if (updatedAfter) {
        raw = this.stmtQueryAllSince.all(updatedAfter) as Record<string, unknown>[];
      } else {
        raw = this.stmtQueryAll.all() as Record<string, unknown>[];
      }

      // Runtime sanity check: verify first row has expected columns (guards against schema drift)
      if (raw.length > 0 && !("record_id" in raw[0] && "content" in raw[0])) {
        this.logger?.warn(
          `${TAG} [L1-query] Schema mismatch: first row missing expected columns. ` +
          `Got keys: [${Object.keys(raw[0]).join(", ")}]`,
        );
        return [];
      }

      let rows = raw as unknown as L1RecordRow[];
      // Prepared statements above optimize the common session/time predicates.
      // Isolation dimensions are optional and can be combined with any query
      // shape (notably L2 profile queries use teamId+agentId+updatedAfter
      // without sessionKey). Apply them in memory to keep the statement matrix
      // bounded and to match queryL1Paginated semantics.
      if (filter?.teamId !== undefined) rows = rows.filter((r) => r.team_id === filter.teamId);
      if (filter?.userId !== undefined) rows = rows.filter((r) => r.user_id === filter.userId);
      if (filter?.agentId !== undefined) rows = rows.filter((r) => r.agent_id === filter.agentId);
      if (taskId !== undefined) rows = rows.filter((r) => r.task_id === taskId);

      this.logger?.info(
        `${TAG} [L1-query] filter={sessionKey=${sessionKey ?? "(all)"}, sessionId=${sessionId ?? "(all)"}, teamId=${filter?.teamId ?? "(all)"}, userId=${filter?.userId ?? "(all)"}, agentId=${filter?.agentId ?? "(all)"}, taskId=${taskId ?? "(all)"}, updatedAfter=${updatedAfter ?? "(none)"}}, ` +
        `returned ${rows.length} record(s)`,
      );
      return rows;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L1-query] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`
      );
      return [];
    }
  }

  // ── L0 operations ──────────────────────────────────

  /**
   * Write or update an L0 single-message record (metadata + vector).
   * Uses a manual transaction for atomicity.
   *
   * If `embedding` is `undefined` or a zero vector (all elements are 0), only
   * the metadata row (`l0_conversations`) is written — the vec0 table
   * (`l0_vec`) is left untouched.  This allows callers without an
   * EmbeddingService to still persist metadata + FTS without constructing a
   * throwaway zero-vector, and prevents placeholder zero vectors (from
   * embedding-service failures) from polluting KNN search results.
   *
   * **Fault-tolerant**: catches all errors internally, never throws.
   * Returns `true` on success, `false` on failure (logged as warning).
   */
  upsertL0(record: L0Record, embedding: Float32Array | undefined): boolean {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [L0-upsert] SKIPPED (degraded mode) id=${record.id}`);
      return false;
    }
    try {
      const skipVec = !embedding || embedding.every(v => v === 0) || !this.vecTablesReady;

      this.logger?.debug?.(
        `${TAG} [L0-upsert] START id=${record.id}, session=${record.sessionKey}, role=${record.role}, ` +
        `text="${record.messageText.slice(0, 60)}..."` +
        (embedding
          ? `, embeddingDims=${embedding.length}, ` +
            `embeddingNorm=${Math.sqrt(Array.from(embedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}` +
            `${skipVec ? " (ZERO VECTOR or vec tables not ready — vec write will be skipped)" : ""}`
          : " (no embedding — metadata-only write)"),
      );

      this.db.exec("BEGIN");
      try {
        // Legacy callers may omit isolation fields; normalize them to the
        // same default identity used by schema defaults and query filters.
        this.stmtL0UpsertMeta.run(
          record.id,
          record.sessionKey,
          record.sessionId || DEFAULT_ISOLATION_ID,
          (record as L0Record & { teamId?: string }).teamId || DEFAULT_ISOLATION_ID,
          record.taskId || "",
          record.role,
          record.messageText,
          record.recordedAt,
          record.timestamp,
          (record as L0Record & { userId?: string }).userId || DEFAULT_ISOLATION_ID,
          (record as L0Record & { agentId?: string }).agentId || DEFAULT_ISOLATION_ID,
        );

        if (!skipVec) {
          // vec0 does not support ON CONFLICT → delete then insert
          this.stmtL0DeleteVec!.run(record.id);
          this.stmtL0InsertVec!.run(record.id, Buffer.from(embedding!.buffer), record.recordedAt);
        } else {
          this.logger?.debug?.(
            `${TAG} [L0-upsert] Skipping vec write (${embedding ? "zero vector" : "no embedding"}) id=${record.id}`,
          );
        }

        // Sync FTS5 (delete + re-insert to handle updates).
        // user_id / agent_id mirrored into FTS so post-recall isolation
        // filtering doesn't require a join.
        if (this.ftsAvailable) {
          try {
            this.stmtL0FtsDelete.run(record.id);
            this.stmtL0FtsInsert.run(
              tokenizeForFts(record.messageText), // message_text — segmented for indexing
              record.messageText,                 // message_text_original — raw for display
              record.id,
              record.sessionKey,
              record.sessionId || DEFAULT_ISOLATION_ID,
              (record as L0Record & { teamId?: string }).teamId || DEFAULT_ISOLATION_ID,
              record.taskId || "",
              (record as L0Record & { userId?: string }).userId || DEFAULT_ISOLATION_ID,
              (record as L0Record & { agentId?: string }).agentId || DEFAULT_ISOLATION_ID,
              record.role,
              record.recordedAt,
              record.timestamp,
            );
          } catch (ftsErr) {
            // FTS write failure is non-fatal — log and continue
            this.logger?.warn(
              `${TAG} [L0-upsert] FTS write failed (non-fatal) id=${record.id}: ${ftsErr instanceof Error ? ftsErr.message : String(ftsErr)}`,
            );
          }
        }

        this.db.exec("COMMIT");
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
      this.logger?.debug?.(`${TAG} [L0-upsert] OK id=${record.id}${skipVec ? " (meta-only)" : ""}`);
      return true;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-upsert] FAILED (non-fatal) id=${record.id}: ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * Update ONLY the vector embedding for an existing L0 record.
   * The metadata row must already exist in l0_conversations (written by upsertL0).
   *
   * This is used by the background embedding task in auto-capture:
   *   1. upsertL0() writes metadata + FTS synchronously (no embedding)
   *   2. Background task calls embedBatch() then updateL0Embedding() for each record
   *
   * **Fault-tolerant**: catches all errors internally, never throws.
   * Returns `true` on success, `false` on failure.
   */
  updateL0Embedding(recordId: string, embedding: Float32Array): boolean {
    if (this.degraded || !this.vecTablesReady) {
      return false;
    }
    if (!embedding || embedding.every(v => v === 0)) {
      this.logger?.debug?.(`${TAG} [L0-update-embedding] Skipping zero vector for ${recordId}`);
      return false;
    }
    try {
      // Look up recorded_at from metadata for the vec0 row
      const meta = this.stmtL0GetMeta.get(recordId) as { recorded_at: string } | undefined;
      if (!meta) {
        this.logger?.warn(`${TAG} [L0-update-embedding] No metadata found for ${recordId}, skipping`);
        return false;
      }

      this.db.exec("BEGIN");
      try {
        this.stmtL0DeleteVec!.run(recordId);
        this.stmtL0InsertVec!.run(recordId, Buffer.from(embedding.buffer), meta.recorded_at);
        this.db.exec("COMMIT");
      } catch (err) {
        try { this.db.exec("ROLLBACK"); } catch { /* ignore */ }
        throw err;
      }
      return true;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-update-embedding] FAILED (non-fatal) id=${recordId}: ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * Vector similarity search on L0 individual messages (cosine distance).
   * Returns top-k results sorted by similarity (highest first).
   *
   * **Fault-tolerant**: returns an empty array on any error.
   */
  searchL0Vector(queryEmbedding: Float32Array, topK = 5, _queryText?: string, filter?: IsolationFilter): L0VectorSearchResult[] {
    if (this.degraded || !this.vecTablesReady) {
      if (this.degraded) this.logger?.warn(`${TAG} [L0-search] SKIPPED (degraded mode)`);
      return [];
    }
    try {
      // Over-retrieve to compensate for legacy zero-vector placeholders that
      // may still exist in the vec0 table.  New zero vectors are no longer
      // inserted (upsertL0() skips vec write for zero vectors since v3.x), but
      // older data may still contain them — they surface as NULL/NaN distance
      // in KNN results.
      // NOTE: "AND distance IS NOT NULL" is NOT usable because vec0 does not
      // support that constraint — it causes an empty result set.
      const retrieveCount = filter ? Math.max(topK * 5, topK + VectorStore.ZERO_VEC_BUFFER) : topK + VectorStore.ZERO_VEC_BUFFER;

      this.logger?.debug?.(
        `${TAG} [L0-search] START topK=${topK}, retrieveCount=${retrieveCount}, ` +
        `queryEmbeddingDims=${queryEmbedding.length}, ` +
        `queryNorm=${Math.sqrt(Array.from(queryEmbedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}`,
      );

      const rows = this.stmtL0SearchVec!.all(
        Buffer.from(queryEmbedding.buffer),
        retrieveCount,
      ) as Array<{ record_id: string; distance: number }>;

      this.logger?.debug?.(`${TAG} [L0-search] vec0 returned ${rows.length} candidate(s)`);

      if (rows.length === 0) return [];

      const results: L0VectorSearchResult[] = [];

      for (const { record_id, distance } of rows) {
        // sqlite-vec returns null distance for zero vectors (cosine undefined when ‖v‖=0).
        // Skip these — they are placeholder vectors from embedding-service-unavailable fallback.
        if (distance == null || Number.isNaN(distance)) {
          this.logger?.warn(
            `${TAG} [L0-search] record_id=${record_id} has null/NaN distance (likely zero vector) — skipping`,
          );
          continue;
        }

        const meta = this.stmtL0GetMeta.get(record_id) as
          | {
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
          | undefined;

        if (!meta) {
          this.logger?.warn(`${TAG} [L0-search] record_id=${record_id} has vector but NO metadata (orphan)`);
          continue;
        }
        if (!rowMatchesIsolation(meta, filter)) {
          continue;
        }

        const score = 1.0 - distance;
        this.logger?.debug?.(
          `${TAG} [L0-search] HIT id=${record_id}, distance=${distance.toFixed(4)}, score=${score.toFixed(4)}, ` +
          `role=${meta.role}, session=${meta.session_key}, text="${meta.message_text.slice(0, 60)}..."`,
        );

        results.push({
          record_id,
          session_key: meta.session_key,
          session_id: meta.session_id,
          team_id: meta.team_id ?? "",
          task_id: meta.task_id ?? "",
          user_id: meta.user_id ?? "",
          agent_id: meta.agent_id ?? "",
          role: meta.role,
          message_text: meta.message_text,
          score,
          recorded_at: meta.recorded_at,
          timestamp: meta.timestamp ?? 0,
        });
      }

      // Trim back to the caller's requested topK (we over-fetched above).
      const trimmed = results.slice(0, topK);
      this.logger?.info(
        `${TAG} [L0-search] DONE returning ${trimmed.length} result(s) (from ${results.length} valid, ${rows.length} raw)`,
      );
      return trimmed;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-search] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * Delete a single L0 record (metadata + vector).
   *
   * **Fault-tolerant**: logs a warning on failure, never throws.
   */
  deleteL0(recordId: string, filter?: IsolationFilter): boolean {
    if (this.degraded) return false;
    try {
      if (filter) {
        const meta = this.stmtL0GetMeta.get(recordId) as { user_id?: string; agent_id?: string; session_id?: string; session_key?: string } | undefined;
        if (!meta || !rowMatchesIsolation(meta, filter)) return false;
      }
      this.db.exec("BEGIN");
      try {
        const result = this.stmtL0DeleteMeta.run(recordId);
        const deleted = (result as any)?.changes > 0;
        if (this.vecTablesReady) this.stmtL0DeleteVec!.run(recordId);
        if (this.ftsAvailable) {
          try { this.stmtL0FtsDelete.run(recordId); } catch { /* non-fatal */ }
        }
        this.db.exec("COMMIT");
        return deleted;
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
    } catch (err) {
      this.logger?.warn(
        `${TAG} deleteL0 failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * TTL cleanup by recorded_at (ISO string) for L0 records.
   *
   * Deletes expired rows from l0_conversations and matching vectors from l0_vec
   * in a single transaction to guarantee consistency.
   */
  deleteL0Expired(cutoffIso: string): number {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [deleteExpiredL0] SKIPPED (degraded mode)`);
      return 0;
    }

    try {
      const row = this.db.prepare(
        "SELECT COUNT(*) AS cnt FROM l0_conversations WHERE recorded_at != '' AND recorded_at < ?",
      ).get(cutoffIso) as { cnt: number } | undefined;
      const expiredCount = row?.cnt ?? 0;
      if (expiredCount <= 0) return 0;

      // Ratio protection: refuse to delete > 80% in one pass
      const totalRow = this.db.prepare(
        "SELECT COUNT(*) AS cnt FROM l0_conversations",
      ).get() as { cnt: number };
      const total = totalRow.cnt;
      const ratio = total > 0 ? expiredCount / total : 0;
      if (ratio > 0.8) {
        this.logger?.warn(
          `${TAG} [L0-deleteExpired] BLOCKED: would delete ${expiredCount}/${total} ` +
          `(${(ratio * 100).toFixed(1)}%) — exceeds 80% safety threshold, cutoff=${cutoffIso}`,
        );
        return 0;
      }

      this.db.exec("BEGIN");
      try {
        if (this.vecTablesReady) {
          this.db.prepare(
            "DELETE FROM l0_vec WHERE recorded_at != '' AND recorded_at < ?",
          ).run(cutoffIso);
        }
        this.db.prepare(
          "DELETE FROM l0_conversations WHERE recorded_at != '' AND recorded_at < ?",
        ).run(cutoffIso);
        this.db.exec("COMMIT");
        this.logger?.info?.(
          `${TAG} [L0-deleteExpired] Deleted ${expiredCount}/${total} records (cutoff=${cutoffIso})`,
        );
        return expiredCount;
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch { /* ignore rollback errors */ }
        throw err;
      }
    } catch (err) {
      this.logger?.warn(
        `${TAG} deleteL0ExpiredByRecordedAt failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return 0;
    }
  }

  /**
   * Get the total number of L0 message records matching optional filters.
   *
   * **Fault-tolerant**: returns 0 on failure.
   */
  countL0(filter?: L0CountFilter): number {
    if (this.degraded) return 0;
    try {
      const conditions: string[] = [];
      const params: SQLInputValue[] = [];

      if (filter?.sessionId) {
        conditions.push("(session_key = ? OR session_id = ?)");
        params.push(filter.sessionId, filter.sessionId);
      }
      if (filter?.teamId !== undefined) {
        conditions.push("team_id = ?");
        params.push(filter.teamId);
      }
      if (filter?.userId !== undefined) {
        conditions.push("user_id = ?");
        params.push(filter.userId);
      }
      if (filter?.agentId !== undefined) {
        conditions.push("agent_id = ?");
        params.push(filter.agentId);
      }
      if (filter?.taskId !== undefined) {
        conditions.push("task_id = ?");
        params.push(filter.taskId);
      }
      if (filter?.timeStartMs !== undefined) {
        conditions.push("timestamp >= ?");
        params.push(filter.timeStartMs);
      }
      if (filter?.timeEndMs !== undefined) {
        conditions.push("timestamp <= ?");
        params.push(filter.timeEndMs);
      }

      const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
      const row = this.db
        .prepare(`SELECT COUNT(*) AS cnt FROM l0_conversations ${where}`)
        .get(...params) as { cnt: number } | undefined;
      const total = row?.cnt ?? 0;
      this.logger?.debug?.(`${TAG} [L0-count] total=${total}`);
      return total;
    } catch (err) {
      this.logger?.warn(
        `${TAG} countL0 failed (non-fatal, returning 0): ${err instanceof Error ? err.message : String(err)}`,
      );
      return 0;
    }
  }

  // ── Re-index operations ──────────────────────────────────

  /**
   * Get all L1 record texts for re-embedding.
   * Returns record_id → content pairs.
   */
  getAllL1Texts(): Array<{ record_id: string; content: string; updated_time: string }> {
    if (this.degraded) return [];
    try {
      return this.db
        .prepare("SELECT record_id, content, updated_time FROM l1_records")
        .all() as Array<{ record_id: string; content: string; updated_time: string }>;
    } catch (err) {
      this.logger?.warn(
        `${TAG} getAllL1Texts failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * Get all L0 message texts for re-embedding.
   * Returns record_id → message_text/recorded_at tuples.
   */
  getAllL0Texts(): Array<{ record_id: string; message_text: string; recorded_at: string }> {
    if (this.degraded) return [];
    try {
      return this.db
        .prepare("SELECT record_id, message_text, recorded_at FROM l0_conversations")
        .all() as Array<{ record_id: string; message_text: string; recorded_at: string }>;
    } catch (err) {
      this.logger?.warn(
        `${TAG} getAllL0Texts failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * Re-embed all existing L1 and L0 texts with a new embedding function.
   *
   * This is called after `init()` returns `needsReindex: true` — the vector
   * tables have already been dropped and re-created with the correct dimensions.
   * This method reads every text from the metadata tables and writes fresh
   * embeddings into the new vector tables.
   *
   * @param embedFn  A function that converts text → Float32Array embedding.
   * @param onProgress  Optional callback for progress reporting.
   */
  async reindexAll(
    embedFn: (text: string) => Promise<Float32Array>,
    onProgress?: (done: number, total: number, layer: "L1" | "L0") => void,
  ): Promise<{ l1Count: number; l0Count: number }> {
    if (this.degraded || !this.vecTablesReady) {
      if (this.degraded) this.logger?.warn(`${TAG} reindexAll skipped: VectorStore is in degraded mode`);
      return { l1Count: 0, l0Count: 0 };
    }

    try {
      // ── Re-embed L1 ──
      const l1Rows = this.getAllL1Texts();
      let l1Done = 0;
      for (const { record_id, content, updated_time } of l1Rows) {
        try {
          const embedding = await embedFn(content);
          // Wrap delete+insert in a transaction to prevent orphan vectors
          this.db.exec("BEGIN");
          try {
            this.stmtDeleteVec!.run(record_id);
            this.stmtInsertVec!.run(record_id, Buffer.from(embedding.buffer), updated_time);
            this.db.exec("COMMIT");
          } catch (txErr) {
            try { this.db.exec("ROLLBACK"); } catch { /* ignore */ }
            throw txErr;
          }
        } catch (err) {
          this.logger?.warn?.(
            `${TAG} reindex L1 skip ${record_id}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        l1Done++;
        onProgress?.(l1Done, l1Rows.length, "L1");
      }

      // ── Re-embed L0 ──
      const l0Rows = this.getAllL0Texts();
      let l0Done = 0;
      for (const { record_id, message_text, recorded_at } of l0Rows) {
        try {
          const embedding = await embedFn(message_text);
          // Wrap delete+insert in a transaction to prevent orphan vectors
          this.db.exec("BEGIN");
          try {
            this.stmtL0DeleteVec!.run(record_id);
            this.stmtL0InsertVec!.run(record_id, Buffer.from(embedding.buffer), recorded_at);
            this.db.exec("COMMIT");
          } catch (txErr) {
            try { this.db.exec("ROLLBACK"); } catch { /* ignore */ }
            throw txErr;
          }
        } catch (err) {
          this.logger?.warn?.(
            `${TAG} reindex L0 skip ${record_id}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        l0Done++;
        onProgress?.(l0Done, l0Rows.length, "L0");
      }

      this.logger?.info(
        `${TAG} Reindex complete: L1=${l1Done}/${l1Rows.length}, L0=${l0Done}/${l0Rows.length}`,
      );

      return { l1Count: l1Done, l0Count: l0Done };
    } catch (err) {
      this.logger?.error(
        `${TAG} reindexAll failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return { l1Count: 0, l0Count: 0 };
    }
  }

  // ── L0 query operations (for L1 runner) ──────────────────────────────────

  /**
   * Query L0 messages for a given session key, optionally filtered by recorded_at cursor.
   * Returns up to `limit` rows with `recorded_at > afterRecordedAtMs`, ordered by
   * recorded_at ASC (chronological write order, **oldest-first**).
   *
   * Used by L1 runner to read L0 data from DB. The runner is responsible for
   * batching/slicing the returned rows (e.g. process N, defer the rest).
   */
  queryL0ForL1(
    sessionKey: string,
    afterRecordedAtMs?: number,
    limit = 50,
  ): L0QueryRow[] {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [L0-query] SKIPPED (degraded mode)`);
      return [];
    }
    try {
      // Query oldest-first (ASC) with LIMIT — preserves backlog ordering so
      // callers that advance a recorded_at cursor never skip older rows.
      let rows: Array<Record<string, unknown>>;
      if (afterRecordedAtMs && afterRecordedAtMs > 0) {
        // Convert epoch ms to ISO string for recorded_at comparison
        const afterRecordedAtIso = new Date(afterRecordedAtMs).toISOString();
        rows = this.stmtL0QueryAfter.all(sessionKey, afterRecordedAtIso, limit) as Array<Record<string, unknown>>;
      } else {
        rows = this.stmtL0QueryAll.all(sessionKey, limit) as Array<Record<string, unknown>>;
      }

      this.logger?.info(
        `${TAG} [L0-query] session=${sessionKey}, afterRecordedAtMs=${afterRecordedAtMs ?? "(all)"}, ` +
        `limit=${limit}, returned ${rows.length} row(s)`,
      );

      return rows.map((r) => ({
        record_id: r.record_id as string,
        session_key: r.session_key as string,
        session_id: (r.session_id as string) || "",
        team_id: (r.team_id as string) || "",
        task_id: (r.task_id as string) || "",
        user_id: (r.user_id as string) || "",
        agent_id: (r.agent_id as string) || "",
        role: r.role as string,
        message_text: r.message_text as string,
        recorded_at: (r.recorded_at as string) || "",
        timestamp: (r.timestamp as number) || 0,
      }));
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-query] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * Query L0 messages for a given session key, grouped by session_id.
   * Each group's messages are in chronological order (recorded_at ASC).
   * Groups are sorted by earliest message timestamp.
   *
   * Used by L1 runner to replace readConversationMessagesGroupedBySessionId().
   */
  queryL0GroupedBySessionId(
    sessionKey: string,
    afterRecordedAtMs?: number,
    limit = 50,
  ): Array<{ sessionId: string; teamId?: string; taskId?: string; userId: string; agentId: string; messages: Array<{ id: string; role: string; content: string; timestamp: number; recordedAtMs: number }> }> {
    if (this.degraded) {
      this.logger?.warn(`${TAG} [L0-query-grouped] SKIPPED (degraded mode)`);
      return [];
    }
    try {
      const rows = this.queryL0ForL1(sessionKey, afterRecordedAtMs, limit);

      // Group by full isolation tuple + session_id to avoid cross-tenant merging.
      const groupMap = new Map<string, {
        sessionId: string;
        teamId?: string;
        taskId?: string;
        userId: string;
        agentId: string;
        messages: Array<{ id: string; role: string; content: string; timestamp: number; recordedAtMs: number }>;
      }>();
      for (const row of rows) {
        const sid = row.session_id || "";
        const teamId = row.team_id || undefined;
        const taskId = row.task_id || undefined;
        const userId = row.user_id || "";
        const agentId = row.agent_id || "";
        const groupKey = `${teamId ?? ""}\u0000${userId}\u0000${agentId}\u0000${sid}\u0000${taskId ?? ""}`;
        let group = groupMap.get(groupKey);
        if (!group) {
          group = { sessionId: sid, teamId, taskId, userId, agentId, messages: [] };
          groupMap.set(groupKey, group);
        }
        group.messages.push({
          id: row.record_id,
          role: row.role,
          content: row.message_text,
          timestamp: row.timestamp,
          recordedAtMs: row.recorded_at ? Date.parse(row.recorded_at) || 0 : 0,
        });
      }

      // Convert to array, sorted by earliest message timestamp
      const groups: Array<{ sessionId: string; teamId?: string; taskId?: string; userId: string; agentId: string; messages: Array<{ id: string; role: string; content: string; timestamp: number; recordedAtMs: number }> }> = [];
      for (const group of groupMap.values()) {
        if (group.messages.length > 0) {
          groups.push(group);
        }
      }
      groups.sort((a, b) => a.messages[0].timestamp - b.messages[0].timestamp);

      this.logger?.info(
        `${TAG} [L0-query-grouped] session=${sessionKey}, afterRecordedAtMs=${afterRecordedAtMs ?? "(all)"}, ` +
        `${rows.length} messages across ${groups.length} group(s)`,
      );

      return groups;
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-query-grouped] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  // ── Cursor-based pagination for migration ──────────────────

  /**
   * Read a page of L1 records using primary key cursor.
   * Returns rows with `record_id > afterId`, ordered by PK, limited to `pageSize`.
   * Pass `""` as `afterId` for the first page.
   */
  queryL1RecordsCursor(afterId: string, pageSize: number): L1RecordRow[] {
    if (this.degraded) return [];
    try {
      return this.stmtL1QueryMigrationCursor.all(afterId, pageSize) as unknown as L1RecordRow[];
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L1-query-cursor] FAILED (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * Read a page of L0 records using primary key cursor.
   * Returns rows with `record_id > afterId`, ordered by PK, limited to `pageSize`.
   * Pass `""` as `afterId` for the first page.
   */
  queryL0RecordsCursor(afterId: string, pageSize: number): L0RecordRow[] {
    if (this.degraded) return [];
    try {
      return this.stmtL0QueryMigrationCursor.all(afterId, pageSize) as unknown as L0RecordRow[];
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-query-cursor] FAILED (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  // ── FTS5 search operations ──────────────────────────────────

  /**
   * Whether FTS5 full-text search is available.
   * When `false`, callers should skip keyword-based recall entirely.
   */
  isFtsAvailable(): boolean {
    return this.ftsAvailable;
  }

  // ── v2 API: Paginated queries ─────────────────────────────

  /**
   * L0 paginated query for v2 `/conversation/query`.
   * Uses SQL WHERE + LIMIT + OFFSET, no full-table scan.
   */
  queryL0Paginated(filter: L0PaginatedFilter): L0PaginatedResult {
    if (this.degraded) return { rows: [], total: 0 };

    try {
      const conditions: string[] = [];
      const params: SQLInputValue[] = [];

      if (filter.sessionId) {
        conditions.push("(session_key = ? OR session_id = ?)");
        params.push(filter.sessionId, filter.sessionId);
      }
      // Isolation dimensions — see docs/l0l3-tenant-isolation-design.md.
      if (filter.teamId !== undefined) {
        conditions.push("team_id = ?");
        params.push(filter.teamId);
      }
      if (filter.userId !== undefined) {
        conditions.push("user_id = ?");
        params.push(filter.userId);
      }
      if (filter.agentId !== undefined) {
        conditions.push("agent_id = ?");
        params.push(filter.agentId);
      }
      if (filter.taskId !== undefined) {
        conditions.push("task_id = ?");
        params.push(filter.taskId);
      }
      if (filter.timeStartMs !== undefined) {
        conditions.push("timestamp >= ?");
        params.push(filter.timeStartMs);
      }
      if (filter.timeEndMs !== undefined) {
        conditions.push("timestamp <= ?");
        params.push(filter.timeEndMs);
      }

      const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

      // Count total
      const countSql = `SELECT COUNT(*) AS cnt FROM l0_conversations ${where}`;
      const countRow = this.db.prepare(countSql).get(...params) as { cnt: number } | undefined;
      const total = countRow?.cnt ?? 0;

      // Fetch page
      const dataSql = `SELECT record_id, session_key, session_id, team_id, task_id, user_id, agent_id, role, message_text, recorded_at, timestamp FROM l0_conversations ${where} ORDER BY timestamp DESC LIMIT ? OFFSET ?`;
      const rows = this.db.prepare(dataSql).all(...params, filter.limit, filter.offset) as unknown as L0QueryRow[];

      return { rows, total };
    } catch (err) {
      this.logger?.warn(`[sqlite] queryL0Paginated failed: ${err instanceof Error ? err.message : String(err)}`);
      return { rows: [], total: 0 };
    }
  }

  /**
   * L1 paginated query for v2 `/atomic/query`.
   * Uses SQL WHERE + LIMIT + OFFSET, no full-table scan.
   */
  queryL1Paginated(filter: L1PaginatedFilter): L1PaginatedResult {
    if (this.degraded) return { rows: [], total: 0 };

    try {
      const conditions: string[] = [];
      const params: SQLInputValue[] = [];

      if (filter.type) {
        conditions.push("type = ?");
        params.push(filter.type);
      }
      if (filter.sessionId) {
        conditions.push("session_id = ?");
        params.push(filter.sessionId);
      }
      // Isolation dimensions.
      if (filter.teamId !== undefined) {
        conditions.push("team_id = ?");
        params.push(filter.teamId);
      }
      if (filter.userId !== undefined) {
        conditions.push("user_id = ?");
        params.push(filter.userId);
      }
      if (filter.agentId !== undefined) {
        conditions.push("agent_id = ?");
        params.push(filter.agentId);
      }
      if (filter.taskId !== undefined) {
        conditions.push("task_id = ?");
        params.push(filter.taskId);
      }
      if (filter.timeStart) {
        conditions.push("updated_time >= ?");
        params.push(filter.timeStart);
      }
      if (filter.timeEnd) {
        conditions.push("updated_time <= ?");
        params.push(filter.timeEnd);
      }

      const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

      // Count total
      const countSql = `SELECT COUNT(*) AS cnt FROM l1_records ${where}`;
      const countRow = this.db.prepare(countSql).get(...params) as { cnt: number } | undefined;
      const total = countRow?.cnt ?? 0;

      // Fetch page — must include user_id / agent_id so callers can enforce
      // isolation in downstream filters / Coordinator candidate pool.
      const dataSql = `SELECT record_id, content, type, priority, scene_name, session_key, session_id, team_id, task_id, user_id, agent_id, version, timestamp_str, timestamp_start, timestamp_end, created_time, updated_time, metadata_json FROM l1_records ${where} ORDER BY updated_time DESC LIMIT ? OFFSET ?`;
      const rows = this.db.prepare(dataSql).all(...params, filter.limit, filter.offset) as unknown as L1RecordRow[];

      return { rows, total };
    } catch (err) {
      this.logger?.warn(`[sqlite] queryL1Paginated failed: ${err instanceof Error ? err.message : String(err)}`);
      return { rows: [], total: 0 };
    }
  }

  /**
   * Delete all L0 messages belonging to a session.
   * Returns the count of actually deleted rows.
   */
  deleteL0BySession(sessionId: string, filter?: IsolationFilter): number {
    if (this.degraded) return 0;

    try {
      // First get all record_ids for the session
      const rows = this.db.prepare(
        "SELECT record_id, session_key, session_id, user_id, agent_id FROM l0_conversations WHERE session_key = ? OR session_id = ?"
      ).all(sessionId, sessionId) as Array<{ record_id: string; session_key: string; session_id: string; user_id: string; agent_id: string }>;

      if (rows.length === 0) return 0;

      this.db.exec("BEGIN");
      try {
        let deletedCount = 0;
        for (const row of rows) {
          if (filter && !rowMatchesIsolation(row, filter)) continue;
          const result = this.db.prepare("DELETE FROM l0_conversations WHERE record_id = ?").run(row.record_id);
          if (((result as any)?.changes ?? 0) <= 0) continue;
          deletedCount++;
          if (this.vecTablesReady) {
            try { this.db.prepare("DELETE FROM l0_vec WHERE record_id = ?").run(row.record_id); } catch { /* vec may not exist */ }
          }
          if (this.ftsAvailable) {
            try { this.db.prepare("DELETE FROM l0_fts WHERE record_id = ?").run(row.record_id); } catch { /* fts may not exist */ }
          }
        }
        this.db.exec("COMMIT");
        return deletedCount;
      } catch (err) {
        this.db.exec("ROLLBACK");
        throw err;
      }
    } catch (err) {
      this.logger?.warn(`[sqlite] deleteL0BySession failed: ${err instanceof Error ? err.message : String(err)}`);
      return 0;
    }
  }

  private entityId(prefix: string): string {
    return `${prefix}_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  }

  private jsonArray(value: unknown): string {
    return JSON.stringify(Array.isArray(value) ? value.filter((v): v is string => typeof v === "string" && v.trim().length > 0) : []);
  }

  private parseArray(value: unknown): string[] {
    if (Array.isArray(value)) return value.filter((v): v is string => typeof v === "string");
    if (typeof value !== "string" || !value) return [];
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === "string") : [];
    } catch {
      return [];
    }
  }

  private teamFromRow(row: any, includeRefs = false): TeamEntity {
    const teamId = String(row.team_id ?? "");
    const ownerUserId = String(row.owner_user_id ?? "");
    const team: TeamEntity = {
      team_id: teamId,
      name: String(row.name ?? ""),
      description: String(row.description ?? "") || undefined,
      owner_user_id: ownerUserId,
      status: (row.status as TeamEntity["status"]) || "active",
      created_at: String(row.created_at ?? ""),
      updated_at: String(row.updated_at ?? ""),
    };
    if (includeRefs) {
      const userIds = new Set<string>(this.parseArray(row.user_ids_json));
      if (ownerUserId) userIds.add(ownerUserId);
      team.user_ids = Array.from(userIds).sort();
      const agentIds = new Set<string>(this.parseArray(row.agent_ids_json));
      const agents = this.db.prepare("SELECT agent_id FROM entity_agents WHERE team_id = ? AND status = 'active' ORDER BY agent_id").all(teamId) as any[];
      for (const agent of agents) agentIds.add(String(agent.agent_id));
      team.agent_ids = Array.from(agentIds).sort();
      team.task_ids = (this.db.prepare("SELECT task_id FROM entity_tasks WHERE team_id = ? ORDER BY task_id").all(teamId) as any[]).map((r) => String(r.task_id));
    }
    return team;
  }

  private userFromRow(row: any, includeDerived = false): UserEntity {
    const userId = String(row.user_id ?? "");
    const user: UserEntity = {
      user_id: userId,
      name: String(row.name ?? ""),
      job_description: String(row.job_description ?? "") || undefined,
      team_ids: [],
      task_ids: [],
      owned_agent_ids: [],
      status: (row.status as UserEntity["status"]) || "active",
      created_at: String(row.created_at ?? ""),
      updated_at: String(row.updated_at ?? ""),
    };
    if (includeDerived) {
      const teams = this.db.prepare("SELECT team_id, owner_user_id, user_ids_json FROM entity_teams WHERE status = 'active'").all() as any[];
      user.team_ids = teams
        .filter((team) => String(team.owner_user_id ?? "") === userId || this.parseArray(team.user_ids_json).includes(userId))
        .map((team) => String(team.team_id))
        .sort();
      const tasks = this.db.prepare("SELECT task_id, creator_user_id, user_ids_json, agent_ids_json FROM entity_tasks ORDER BY task_id").all() as any[];
      const taskIds = new Set<string>();
      const taskAgentIds = new Set<string>();
      for (const task of tasks) {
        const participates = String(task.creator_user_id ?? "") === userId || this.parseArray(task.user_ids_json).includes(userId);
        if (!participates) continue;
        taskIds.add(String(task.task_id));
        for (const agentId of this.parseArray(task.agent_ids_json)) taskAgentIds.add(agentId);
      }
      user.task_ids = Array.from(taskIds).sort();
      user.task_agent_ids = Array.from(taskAgentIds).sort();
      user.owned_agent_ids = (this.db.prepare("SELECT agent_id FROM entity_agents WHERE owner_user_id = ? AND status = 'active' ORDER BY agent_id").all(userId) as any[]).map((r) => String(r.agent_id));
    }
    return user;
  }

  private agentFromRow(row: any, includeDerived = true): AgentEntity {
    const agentId = String(row.agent_id ?? "");
    const agent: AgentEntity = {
      agent_id: agentId,
      team_id: String(row.team_id ?? ""),
      name: String(row.name ?? ""),
      description: String(row.description ?? "") || undefined,
      prompt: String(row.prompt ?? "") || undefined,
      owner_user_id: String(row.owner_user_id ?? "") || undefined,
      visibility: (row.visibility as AgentEntity["visibility"]) || "team",
      status: (row.status as AgentEntity["status"]) || "active",
      created_at: String(row.created_at ?? ""),
      updated_at: String(row.updated_at ?? ""),
    };
    if (includeDerived) {
      const tasks = this.db.prepare("SELECT task_id, agent_ids_json FROM entity_tasks ORDER BY task_id").all() as any[];
      agent.task_ids = tasks.filter((task) => this.parseArray(task.agent_ids_json).includes(agentId)).map((task) => String(task.task_id));
    }
    return agent;
  }

  private taskFromRow(row: any): TaskEntity {
    return {
      task_id: String(row.task_id ?? ""),
      team_id: String(row.team_id ?? ""),
      creator_user_id: String(row.creator_user_id ?? ""),
      title: String(row.title ?? "") || undefined,
      description: String(row.description ?? "") || undefined,
      source_type: (row.source_type as TaskEntity["source_type"]) || "manual",
      source_url: String(row.source_url ?? "") || undefined,
      agent_ids: this.parseArray(row.agent_ids_json),
      user_ids: this.parseArray(row.user_ids_json),
      created_at: String(row.created_at ?? ""),
      updated_at: String(row.updated_at ?? ""),
    };
  }

  createTeam(input: Omit<TeamEntity, "created_at" | "updated_at" | "status" | "user_ids" | "agent_ids" | "task_ids"> & { team_id?: string; status?: TeamEntity["status"] }): TeamEntity {
    const now = new Date().toISOString();
    const id = input.team_id || this.entityId("team");
    this.db.prepare("INSERT INTO entity_teams (team_id, name, description, owner_user_id, user_ids_json, agent_ids_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)").run(id, input.name, input.description ?? "", input.owner_user_id, this.jsonArray([input.owner_user_id]), this.jsonArray([]), input.status ?? "active", now, now);
    return this.getTeam(id) ?? { team_id: id, name: input.name, owner_user_id: input.owner_user_id, status: input.status ?? "active", created_at: now, updated_at: now };
  }

  getTeam(teamId: string): TeamEntity | null {
    const row = this.db.prepare("SELECT * FROM entity_teams WHERE team_id = ?").get(teamId) as any;
    return row ? this.teamFromRow(row, true) : null;
  }

  updateTeam(teamId: string, patch: Partial<Pick<TeamEntity, "name" | "description" | "owner_user_id" | "user_ids" | "agent_ids" | "status">>): TeamEntity | null {
    const row = this.db.prepare("SELECT * FROM entity_teams WHERE team_id = ?").get(teamId) as any;
    if (!row) return null;
    const current = this.teamFromRow(row, true);
    const ownerUserId = patch.owner_user_id ?? current.owner_user_id;
    const userIds = patch.user_ids !== undefined ? Array.from(new Set([...patch.user_ids, ownerUserId])).sort() : this.parseArray(row.user_ids_json);
    if (ownerUserId && !userIds.includes(ownerUserId)) userIds.push(ownerUserId);
    const agentIds = patch.agent_ids !== undefined ? patch.agent_ids : this.parseArray(row.agent_ids_json);
    const next = { ...current, ...patch, owner_user_id: ownerUserId, updated_at: new Date().toISOString() };
    this.db.prepare("UPDATE entity_teams SET name = ?, description = ?, owner_user_id = ?, user_ids_json = ?, agent_ids_json = ?, status = ?, updated_at = ? WHERE team_id = ?").run(next.name, next.description ?? "", ownerUserId, this.jsonArray(userIds), this.jsonArray(agentIds), next.status, next.updated_at, teamId);
    return this.getTeam(teamId);
  }

  deleteTeams(teamIds: string[]): BatchDeleteResult {
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of teamIds) {
      const row = this.getTeam(id);
      if (!row) { result.failed.push({ id, reason: "not_found" }); continue; }
      this.updateTeam(id, { status: "archived" });
      result.deleted_ids.push(id);
    }
    return result;
  }

  createUser(input: Pick<UserEntity, "name"> & Partial<Pick<UserEntity, "job_description" | "status">> & { user_id?: string }): UserEntity {
    const now = new Date().toISOString();
    const id = input.user_id || this.entityId("user");
    this.db.prepare("INSERT INTO entity_users (user_id, name, job_description, team_ids_json, task_ids_json, owned_agent_ids_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)").run(id, input.name, input.job_description ?? "", this.jsonArray([]), this.jsonArray([]), this.jsonArray([]), input.status ?? "active", now, now);
    return this.getUser(id) ?? { user_id: id, name: input.name, team_ids: [], task_ids: [], owned_agent_ids: [], status: input.status ?? "active", created_at: now, updated_at: now };
  }

  getUser(userId: string): UserEntity | null {
    const row = this.db.prepare("SELECT * FROM entity_users WHERE user_id = ?").get(userId) as any;
    return row ? this.userFromRow(row, true) : null;
  }

  updateUser(userId: string, patch: Partial<Pick<UserEntity, "name" | "job_description" | "status">>): UserEntity | null {
    const current = this.getUser(userId);
    if (!current) return null;
    const next = { ...current, ...patch, updated_at: new Date().toISOString() };
    this.db.prepare("UPDATE entity_users SET name = ?, job_description = ?, status = ?, updated_at = ? WHERE user_id = ?").run(next.name, next.job_description ?? "", next.status, next.updated_at, userId);
    return this.getUser(userId);
  }

  deleteUsers(userIds: string[]): BatchDeleteResult {
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of userIds) {
      if (!this.getUser(id)) { result.failed.push({ id, reason: "not_found" }); continue; }
      this.updateUser(id, { status: "inactive" });
      result.deleted_ids.push(id);
    }
    return result;
  }

  createAgent(input: Omit<AgentEntity, "created_at" | "updated_at" | "status" | "visibility" | "task_ids"> & { agent_id?: string; status?: AgentEntity["status"]; visibility?: AgentEntity["visibility"] }): AgentEntity {
    const now = new Date().toISOString();
    const id = input.agent_id || this.entityId("agent");
    this.db.prepare("INSERT INTO entity_agents (agent_id, team_id, name, description, prompt, owner_user_id, visibility, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)").run(id, input.team_id, input.name, input.description ?? "", input.prompt ?? "", input.owner_user_id ?? "", input.visibility ?? "team", input.status ?? "active", now, now);
    const team = this.getTeam(input.team_id);
    if (team && !(team.agent_ids ?? []).includes(id)) this.updateTeam(input.team_id, { agent_ids: [...(team.agent_ids ?? []), id] });
    return this.getAgent(id) ?? { agent_id: id, team_id: input.team_id, name: input.name, visibility: input.visibility ?? "team", status: input.status ?? "active", created_at: now, updated_at: now };
  }

  getAgent(agentId: string): AgentEntity | null {
    const row = this.db.prepare("SELECT * FROM entity_agents WHERE agent_id = ?").get(agentId) as any;
    return row ? this.agentFromRow(row) : null;
  }

  updateAgent(agentId: string, patch: Partial<Pick<AgentEntity, "name" | "description" | "prompt" | "owner_user_id" | "visibility" | "status">>): AgentEntity | null {
    const current = this.getAgent(agentId);
    if (!current) return null;
    const next = { ...current, ...patch, updated_at: new Date().toISOString() };
    this.db.prepare("UPDATE entity_agents SET name = ?, description = ?, prompt = ?, owner_user_id = ?, visibility = ?, status = ?, updated_at = ? WHERE agent_id = ?").run(next.name, next.description ?? "", next.prompt ?? "", next.owner_user_id ?? "", next.visibility, next.status, next.updated_at, agentId);
    return this.getAgent(agentId);
  }

  deleteAgents(agentIds: string[]): BatchDeleteResult {
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of agentIds) {
      if (!this.getAgent(id)) { result.failed.push({ id, reason: "not_found" }); continue; }
      this.updateAgent(id, { status: "inactive" });
      result.deleted_ids.push(id);
    }
    return result;
  }

  createTask(input: Omit<TaskEntity, "created_at" | "updated_at" | "source_type" | "agent_ids" | "user_ids"> & { task_id?: string; source_type?: TaskEntity["source_type"]; agent_ids?: string[]; user_ids?: string[] }): TaskEntity {
    const now = new Date().toISOString();
    const id = input.task_id || this.entityId("task");
    this.db.prepare("INSERT INTO entity_tasks (task_id, team_id, creator_user_id, title, description, source_type, source_url, status, auto_assign_floating_assets, risk_level, agent_ids_json, user_ids_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)").run(id, input.team_id, input.creator_user_id, input.title ?? "", input.description ?? "", input.source_type ?? "manual", input.source_url ?? "", "pending", 0, "low", this.jsonArray(input.agent_ids), this.jsonArray(input.user_ids), now, now);
    return this.getTask(id) ?? { task_id: id, team_id: input.team_id, creator_user_id: input.creator_user_id, source_type: input.source_type ?? "manual", agent_ids: input.agent_ids ?? [], user_ids: input.user_ids ?? [], created_at: now, updated_at: now };
  }

  getTask(taskId: string): TaskEntity | null {
    const row = this.db.prepare("SELECT * FROM entity_tasks WHERE task_id = ?").get(taskId) as any;
    return row ? this.taskFromRow(row) : null;
  }

  updateTask(taskId: string, patch: Partial<Pick<TaskEntity, "title" | "description" | "source_type" | "source_url" | "agent_ids" | "user_ids">>): TaskEntity | null {
    const current = this.getTask(taskId);
    if (!current) return null;
    const next = { ...current, ...patch, updated_at: new Date().toISOString() };
    this.db.prepare("UPDATE entity_tasks SET title = ?, description = ?, source_type = ?, source_url = ?, agent_ids_json = ?, user_ids_json = ?, updated_at = ? WHERE task_id = ?").run(next.title ?? "", next.description ?? "", next.source_type, next.source_url ?? "", this.jsonArray(next.agent_ids), this.jsonArray(next.user_ids), next.updated_at, taskId);
    return this.getTask(taskId);
  }

  deleteTasks(taskIds: string[]): BatchDeleteResult {
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of taskIds) {
      if (!this.getTask(id)) { result.failed.push({ id, reason: "not_found" }); continue; }
      this.db.prepare("DELETE FROM entity_tasks WHERE task_id = ?").run(id);
      result.deleted_ids.push(id);
    }
    return result;
  }

  // ── Knowledge entity ──

  private knowledgeFromRow(row: any): KnowledgeEntity {
    return {
      knowledge_id: String(row.knowledge_id ?? ""),
      type: (row.type as KnowledgeType) ?? "wiki",
      service_url: String(row.service_url ?? ""),
      name: String(row.name ?? ""),
      summary: row.summary ?? null,
      team_id: String(row.team_id ?? ""),
      agent_id: String(row.agent_id ?? ""),
      user_id: row.user_id ?? null,
      repo_url: row.repo_url ?? undefined,
      branch: row.branch ?? undefined,
      created_at: String(row.created_at ?? ""),
      updated_at: String(row.updated_at ?? ""),
    };
  }

  createKnowledge(input: Omit<KnowledgeEntity, "created_at" | "updated_at">): KnowledgeEntity {
    const now = new Date().toISOString();
    // Upsert: if knowledge_id exists, update; else insert
    const existing = this.db.prepare("SELECT created_at FROM entity_knowledge WHERE knowledge_id = ?").get(input.knowledge_id) as any;
    if (existing) {
      this.db.prepare(
        "UPDATE entity_knowledge SET type=?, service_url=?, name=?, summary=?, team_id=?, agent_id=?, user_id=?, repo_url=?, branch=?, updated_at=? WHERE knowledge_id=?"
      ).run(input.type, input.service_url, input.name, input.summary ?? null, input.team_id, input.agent_id ?? "", input.user_id ?? null, input.repo_url ?? null, input.branch ?? null, now, input.knowledge_id);
      return this.getKnowledge(input.knowledge_id)!;
    }
    this.db.prepare(
      "INSERT INTO entity_knowledge (knowledge_id, type, service_url, name, summary, team_id, agent_id, user_id, repo_url, branch, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    ).run(input.knowledge_id, input.type, input.service_url, input.name, input.summary ?? null, input.team_id, input.agent_id ?? "", input.user_id ?? null, input.repo_url ?? null, input.branch ?? null, now, now);
    return this.getKnowledge(input.knowledge_id)!;
  }

  getKnowledge(knowledgeId: string): KnowledgeEntity | null {
    const row = this.db.prepare("SELECT * FROM entity_knowledge WHERE knowledge_id = ?").get(knowledgeId) as any;
    return row ? this.knowledgeFromRow(row) : null;
  }

  updateKnowledge(knowledgeId: string, patch: Partial<Pick<KnowledgeEntity, "name" | "summary" | "service_url" | "repo_url" | "branch">>): KnowledgeEntity | null {
    const current = this.getKnowledge(knowledgeId);
    if (!current) return null;
    const now = new Date().toISOString();
    const sets: string[] = ["updated_at = ?"];
    const args: SQLInputValue[] = [now];
    if (patch.name !== undefined) { sets.push("name = ?"); args.push(patch.name); }
    if (patch.summary !== undefined) { sets.push("summary = ?"); args.push(patch.summary); }
    if (patch.service_url !== undefined) { sets.push("service_url = ?"); args.push(patch.service_url); }
    if (patch.repo_url !== undefined) { sets.push("repo_url = ?"); args.push(patch.repo_url); }
    if (patch.branch !== undefined) { sets.push("branch = ?"); args.push(patch.branch); }
    args.push(knowledgeId);
    this.db.prepare(`UPDATE entity_knowledge SET ${sets.join(", ")} WHERE knowledge_id = ?`).run(...args);
    return this.getKnowledge(knowledgeId);
  }

  deleteKnowledge(knowledgeIds: string[], teamId?: string): BatchDeleteResult {
    const result: BatchDeleteResult = { deleted_ids: [], failed: [] };
    for (const id of knowledgeIds) {
      const row = this.getKnowledge(id);
      if (!row) { result.failed.push({ id, reason: "not_found" }); continue; }
      if (teamId && row.team_id !== teamId) { result.failed.push({ id, reason: "team_mismatch" }); continue; }
      this.db.prepare("DELETE FROM entity_knowledge WHERE knowledge_id = ?").run(id);
      result.deleted_ids.push(id);
    }
    return result;
  }

  listKnowledge(input: { team_id: string; type?: KnowledgeType; knowledge_ids?: string[]; limit?: number; offset?: number }): KnowledgeListResult {
    const limit = Math.min(Math.max(input.limit ?? 20, 1), 1000);
    const offset = Math.max(input.offset ?? 0, 0);
    // knowledge_ids 过滤（Proxy 按 id 批量联查明细）；空数组 → 空结果
    const ids = input.knowledge_ids;
    if (ids && ids.length === 0) return { items: [], total: 0 };
    const idClause = ids && ids.length > 0 ? ` AND knowledge_id IN (${ids.map(() => "?").join(",")})` : "";

    let sql = "SELECT * FROM entity_knowledge WHERE team_id = ?";
    const args: SQLInputValue[] = [input.team_id];
    if (input.type) { sql += " AND type = ?"; args.push(input.type); }
    if (idClause) { sql += idClause; args.push(...ids!); }
    sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?";
    args.push(limit, offset);
    const rows = this.db.prepare(sql).all(...args) as any[];
    const items = rows.map((r) => this.knowledgeFromRow(r));

    let countSql = "SELECT COUNT(*) as total FROM entity_knowledge WHERE team_id = ?";
    const countArgs: SQLInputValue[] = [input.team_id];
    if (input.type) { countSql += " AND type = ?"; countArgs.push(input.type); }
    if (idClause) { countSql += idClause; countArgs.push(...ids!); }
    const totalRow = this.db.prepare(countSql).get(...countArgs) as any;
    return { items, total: totalRow?.total ?? 0 };
  }

  /**
   * FTS5 keyword search on L1 records.
   * Returns top-`limit` results sorted by BM25 relevance (highest first).
   *
   * @param ftsQuery  A pre-built FTS5 MATCH expression (from `buildFtsQuery()`).
   * @param limit     Maximum number of results to return.
   *
   * **Fault-tolerant**: returns an empty array on any error.
   */
  searchL1Fts(ftsQuery: string, limit = 20, filter?: IsolationFilter): FtsSearchResult[] {
    if (this.degraded || !this.ftsAvailable) return [];
    try {
      const retrieveLimit = filter ? Math.max(limit * 5, limit) : limit;
      const rows = this.stmtL1FtsSearch.all(ftsQuery, retrieveLimit) as Array<{
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
        metadata_json: string;
        rank: number;
      }>;

      return rows
        .filter((r) => rowMatchesIsolation(r, filter))
        .slice(0, limit)
        .map((r) => ({
          record_id: r.record_id,
          content: r.content,
          type: r.type,
          priority: r.priority,
          scene_name: r.scene_name,
          score: bm25RankToScore(r.rank),
          timestamp_str: r.timestamp_str,
          timestamp_start: r.timestamp_start,
          timestamp_end: r.timestamp_end,
          version: r.version ?? 0,
          session_key: r.session_key,
          session_id: r.session_id,
          team_id: r.team_id ?? "",
          task_id: r.task_id ?? "",
          user_id: r.user_id ?? "",
          agent_id: r.agent_id ?? "",
          metadata_json: r.metadata_json,
        }));
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L1-fts-search] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  /**
   * FTS5 keyword search on L0 conversation messages.
   * Returns top-`limit` results sorted by BM25 relevance (highest first).
   *
   * @param ftsQuery  A pre-built FTS5 MATCH expression (from `buildFtsQuery()`).
   * @param limit     Maximum number of results to return.
   *
   * **Fault-tolerant**: returns an empty array on any error.
   */
  searchL0Fts(ftsQuery: string, limit = VectorStore.FTS_DEFAULT_LIMIT, filter?: IsolationFilter): L0FtsSearchResult[] {
    if (this.degraded || !this.ftsAvailable) return [];
    try {
      const retrieveLimit = filter ? Math.max(limit * 5, limit) : limit;
      const rows = this.stmtL0FtsSearch.all(ftsQuery, retrieveLimit) as Array<{
        record_id: string;
        message_text: string;
        session_key: string;
        session_id: string;
        team_id: string;
        task_id: string;
        user_id: string;
        agent_id: string;
        role: string;
        recorded_at: string;
        timestamp: number;
        rank: number;
      }>;

      return rows
        .filter((r) => rowMatchesIsolation(r, filter))
        .slice(0, limit)
        .map((r) => ({
          record_id: r.record_id,
          session_key: r.session_key,
          session_id: r.session_id,
          team_id: r.team_id ?? "",
          task_id: r.task_id ?? "",
          user_id: r.user_id ?? "",
          agent_id: r.agent_id ?? "",
          role: r.role,
          message_text: r.message_text,
          score: bm25RankToScore(r.rank),
          recorded_at: r.recorded_at,
          timestamp: r.timestamp ?? 0,
        }));
    } catch (err) {
      this.logger?.warn(
        `${TAG} [L0-fts-search] FAILED (non-fatal, returning empty): ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  // ── FTS5 migration & rebuild ──────────────────────────────────────────────

  /**
   * Detect old FTS5 v1 schema (no `content_original` column) and drop the
   * tables so they can be recreated with the v2 schema.
   *
   * FTS5 virtual tables do NOT support `ALTER TABLE ADD COLUMN`, so the only
   * migration path is DROP + recreate + repopulate.
   *
   * @returns `true` if migration was performed (= FTS index needs rebuilding).
   * @internal
   */
  private migrateFtsTablesIfNeeded(): boolean {
    try {
      // Check if l1_fts exists at all
      const l1Exists = this.db
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='l1_fts'")
        .get();
      if (!l1Exists) {
        // Fresh install — tables will be created with v2 schema.
        // Still need rebuild if there's existing data in l1_records.
        const hasData = this.db.prepare("SELECT 1 FROM l1_records LIMIT 1").get();
        return !!hasData;
      }

      // Check if the v2 column `content_original` exists.
      // FTS5 tables appear in pragma_table_info with their column names.
      const cols = this.db
        .prepare("SELECT name FROM pragma_table_info('l1_fts')")
        .all() as Array<{ name: string }>;
      const hasV2Col = cols.some((c) => c.name === "content_original");
      // v3 marker: isolation columns added (user_id / agent_id).
      const hasV3Col = cols.some((c) => c.name === "user_id")
        && cols.some((c) => c.name === "agent_id");
      const hasV4Col = cols.some((c) => c.name === "version");
      const hasV5Col = cols.some((c) => c.name === "task_id");

      if (hasV2Col && hasV3Col && hasV4Col && hasV5Col) {
        return false; // Already current — no migration needed
      }

      // Migrate forward. FTS5 has no ALTER ADD COLUMN, so any forward step
      // means DROP both FTS tables and rely on rebuildFtsIndex() to repopulate
      // from l0_conversations / l1_records (which now carry user_id/agent_id
      // after the L0/L1 schema migration above).
      if (!hasV2Col) {
        this.logger?.info(`${TAG} Migrating FTS5 tables v1 → v3 (jieba + tenancy isolation)`);
      } else if (!hasV3Col) {
        this.logger?.info(`${TAG} Migrating FTS5 tables v2 → v3 (add user_id / agent_id columns)`);
      } else if (!hasV4Col) {
        this.logger?.info(`${TAG} Migrating FTS5 tables v3 → v4 (add version column)`);
      } else if (!hasV5Col) {
        this.logger?.info(`${TAG} Migrating FTS5 tables v4 → v5 (add task_id column)`);
      }
      this.db.exec("DROP TABLE IF EXISTS l1_fts");
      this.db.exec("DROP TABLE IF EXISTS l0_fts");
      return true;
    } catch (err) {
      this.logger?.warn(
        `${TAG} FTS migration check failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  /**
   * Rebuild the FTS5 index from scratch by reading all records from the
   * metadata tables and re-inserting them with jieba-segmented text.
   *
   * Called automatically after:
   *  - Schema migration from v1 to v2
   *  - Fresh table creation when existing data exists
   *
   * Safe to call multiple times (idempotent — clears FTS tables first).
   */
  rebuildFtsIndex(): void {
    if (!this.ftsAvailable) return;

    try {
      this.logger?.info(`${TAG} Rebuilding FTS5 index with jieba segmentation…`);

      // ── Rebuild L1 FTS ──
      // Clear existing FTS data
      this.db.exec("DELETE FROM l1_fts");

      // Read all L1 records from metadata table.
      // Include user_id / agent_id so the rebuilt FTS rows carry isolation info.
      const l1Rows = this.db
        .prepare(`
          SELECT record_id, content, type, priority, scene_name,
                 session_key, session_id, team_id, task_id, user_id, agent_id, version,
                 timestamp_str, timestamp_start, timestamp_end, metadata_json
          FROM l1_records
        `)
        .all() as Array<{
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
          metadata_json: string;
        }>;

      let l1Count = 0;
      for (const r of l1Rows) {
        try {
          this.stmtL1FtsInsert.run(
            tokenizeForFts(r.content),  // content — segmented
            r.content,                   // content_original — raw
            r.record_id,
            r.type,
            r.priority,
            r.scene_name,
            r.session_key,
            r.session_id || DEFAULT_ISOLATION_ID,
            r.team_id || "",
            r.task_id || "",
            r.user_id || DEFAULT_ISOLATION_ID,
            r.agent_id || DEFAULT_ISOLATION_ID,
            r.version ?? 0,
            r.timestamp_str,
            r.timestamp_start,
            r.timestamp_end,
            r.metadata_json,
          );
          l1Count++;
        } catch (err) {
          this.logger?.warn?.(
            `${TAG} FTS rebuild skip L1 ${r.record_id}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      }

      // ── Rebuild L0 FTS ──
      this.db.exec("DELETE FROM l0_fts");

      const l0Rows = this.db
        .prepare(`
          SELECT record_id, message_text, session_key, session_id, team_id, task_id, user_id, agent_id,
                 role, recorded_at, timestamp
          FROM l0_conversations
        `)
        .all() as Array<{
          record_id: string;
          message_text: string;
          session_key: string;
          session_id: string;
          team_id: string;
          task_id: string;
          user_id: string;
          agent_id: string;
          role: string;
          recorded_at: string;
          timestamp: number;
        }>;

      let l0Count = 0;
      for (const r of l0Rows) {
        try {
          this.stmtL0FtsInsert.run(
            tokenizeForFts(r.message_text),  // message_text — segmented
            r.message_text,                   // message_text_original — raw
            r.record_id,
            r.session_key,
            r.session_id,
            r.team_id ?? "",
            r.task_id ?? "",
            r.user_id ?? "",
            r.agent_id ?? "",
            r.role,
            r.recorded_at,
            r.timestamp,
          );
          l0Count++;
        } catch (err) {
          this.logger?.warn?.(
            `${TAG} FTS rebuild skip L0 ${r.record_id}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      }

      this.logger?.info(
        `${TAG} FTS5 rebuild complete: L1=${l1Count}/${l1Rows.length}, L0=${l0Count}/${l0Rows.length}`,
      );
    } catch (err) {
      this.logger?.warn(
        `${TAG} FTS5 rebuild failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  // ============================
  // IMemoryStore interface implementation
  // ============================

  /** Query the store's search capabilities. */
  getCapabilities(): StoreCapabilities {
    return {
      vectorSearch: this.vecTablesReady,
      ftsSearch: this.ftsAvailable,
      nativeHybridSearch: false,
      sparseVectors: false,
    };
  }

  // ─────────────────────────────────────────────────────────
  // Memory Audit (修改审计)
  // ─────────────────────────────────────────────────────────

  appendAudit(entry: AuditEntry): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO memory_audit
        (audit_id, record_id, layer, action,
         team_id, agent_id, user_id, task_id,
         version, updated_at_ms, request_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      entry.audit_id,
      entry.record_id,
      entry.layer,
      entry.action,
      entry.team_id ?? null,
      entry.agent_id ?? null,
      entry.user_id ?? null,
      entry.task_id ?? null,
      entry.version,
      entry.updated_at_ms,
      entry.request_id ?? null,
    );
  }

  queryAudit(filter: AuditQueryFilter): AuditEntry[] {
    const conds: string[] = [];
    const args: SQLInputValue[] = [];
    if (filter.record_id !== undefined) { conds.push("record_id = ?"); args.push(filter.record_id); }
    if (filter.layer !== undefined)     { conds.push("layer = ?");     args.push(filter.layer); }
    if (filter.action !== undefined)    { conds.push("action = ?");    args.push(filter.action); }
    if (filter.team_id !== undefined)   { conds.push("team_id = ?");   args.push(filter.team_id); }
    if (filter.agent_id !== undefined)  { conds.push("agent_id = ?");  args.push(filter.agent_id); }
    if (filter.user_id !== undefined)   { conds.push("user_id = ?");   args.push(filter.user_id); }
    if (filter.task_id !== undefined)   { conds.push("task_id = ?");   args.push(filter.task_id); }
    if (filter.since_ms !== undefined)  { conds.push("updated_at_ms >= ?"); args.push(filter.since_ms); }
    if (filter.until_ms !== undefined)  { conds.push("updated_at_ms <= ?"); args.push(filter.until_ms); }

    const where = conds.length > 0 ? `WHERE ${conds.join(" AND ")}` : "";
    const limit = Math.min(Math.max(filter.limit ?? 100, 1), 1000);
    const offset = Math.max(filter.offset ?? 0, 0);

    const sql = `
      SELECT audit_id, record_id, layer, action,
             team_id, agent_id, user_id, task_id,
             version, updated_at_ms, request_id
      FROM memory_audit
      ${where}
      ORDER BY updated_at_ms DESC, audit_id DESC
      LIMIT ? OFFSET ?
    `;
    const stmt = this.db.prepare(sql);
    const rows = stmt.all(...args, limit, offset) as Array<{
      audit_id: string;
      record_id: string;
      layer: "L1" | "L2" | "L3";
      action: "update" | "delete";
      team_id: string | null;
      agent_id: string | null;
      user_id: string | null;
      task_id: string | null;
      version: number;
      updated_at_ms: number;
      request_id: string | null;
    }>;
    return rows.map((r) => ({
      audit_id: r.audit_id,
      record_id: r.record_id,
      layer: r.layer,
      action: r.action,
      team_id: r.team_id ?? undefined,
      agent_id: r.agent_id ?? undefined,
      user_id: r.user_id ?? undefined,
      task_id: r.task_id ?? undefined,
      version: r.version,
      updated_at_ms: r.updated_at_ms,
      request_id: r.request_id ?? undefined,
    }));
  }

  /**
   * Close the database connection.
   * Should be called on shutdown. Idempotent — safe to call multiple times.
   */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      this.db.close();
    } catch (err) {
      this.logger?.warn?.(
        `${TAG} Error closing database: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
}
