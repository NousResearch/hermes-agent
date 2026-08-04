/**
 * Pipeline factory: shared infrastructure for creating and wiring
 * MemoryPipelineManager instances with VectorStore, EmbeddingService,
 * L1 runner, L2 runner, L3 runner, and persister.
 *
 * Used by both:
 * - `index.ts` (live plugin runtime)
 * - `seed-runtime.ts` (standalone seed CLI command)
 *
 * This avoids duplicating VectorStore init, L1/L2/L3 extraction logic,
 * persister wiring, and destroy sequences across multiple callers.
 */

import fs from "node:fs";
import path from "node:path";
import type { MemoryTdaiConfig } from "../config.js";
import { MemoryPipelineManager } from "./pipeline-manager.js";
import type { L2Runner, L3Runner } from "./pipeline-manager.js";
import { SessionFilter } from "./session-filter.js";
import { extractL1Memories } from "../core/record/l1-extractor.js";
import { readConversationMessagesGroupedBySessionId } from "../core/conversation/l0-recorder.js";
import type { ConversationMessage } from "../core/conversation/l0-recorder.js";
import { CheckpointManager } from "./checkpoint.js";
import type { PipelineSessionState } from "./checkpoint.js";
import { createStoreBundle } from "../core/store/factory.js";
import type { IMemoryStore } from "../core/store/types.js";
import type { EmbeddingService } from "../core/store/embedding.js";
import {
  readManifest,
  writeManifest,
  buildStoreInfo,
  diffStoreBinding,
  type Manifest,
} from "./manifest.js";
import { SceneExtractor } from "../core/scene/scene-extractor.js";
import { PersonaTrigger } from "../core/persona/persona-trigger.js";
import { PersonaGenerator } from "../core/persona/persona-generator.js";
import {
  DEFAULT_PROFILE_SCOPE,
  buildProfileIsolationScope,
  parseProfileIsolationScope,
  pullProfilesToLocal,
  syncLocalProfilesToStore,
  type ProfileIsolation,
  type ProfileScopeOptions,
} from "../core/profile/profile-sync.js";
import { createScopedStorageAdapter, type StorageAdapter } from "../core/storage/adapter.js";
import type { Logger } from "../core/types.js";

const TAG = "[memory-tdai] [pipeline-factory]";

// ============================
// L1 batch sizing constants
// ============================
//
// Each L1 run consumes at most `L1_BATCH_PROCESS` L0 rows past the cursor.
// The runner over-fetches `L1_BATCH_QUERY` (= 2 * L1_BATCH_PROCESS) rows so
// it can detect backlog from the query result without an extra round-trip:
//   - returned R == L1_BATCH_QUERY → DB very likely has many more rows;
//     pipeline-manager / executor immediately enqueues the next L1 round.
//   - L1_BATCH_PROCESS < R < L1_BATCH_QUERY → small tail; pipeline-manager
//     defers via the existing l1Idle timer (reuses the standard idle path).
//   - R <= L1_BATCH_PROCESS → fully consumed; nothing to do.
//
// Constants live here because both the standalone runner (this file) and
// the service-mode worker (gateway/server.ts) depend on the same
// over-fetch-by-1x semantic to recognize backlog.
// Aligned with l1-extractor's maxMessagesPerExtraction (default 10) so that
// every L0 row in the batch is seen by the LLM as a "new" message.
// Previous value of 20 caused the extractor's slice(-10) to silently
// truncate the first 5 rows per batch. Trade-off: drain rounds double
// under backlog, but zero data loss.
export const L1_BATCH_PROCESS = 10;
export const L1_BATCH_QUERY = L1_BATCH_PROCESS * 2;

function supportsProfileSyncWrite(store?: IMemoryStore): boolean {
  return !!(store?.syncProfiles || store?.deleteProfiles);
}

export const PROFILE_L2_KEY_PREFIX = "profile:";

function buildIsolationScope(ctx?: ProfileIsolation): string {
  return buildProfileIsolationScope(ctx);
}

export function buildProfileL2Key(ctx?: ProfileIsolation): string {
  const sourceSession = ctx?.sessionId ? `|session:${encodeURIComponent(ctx.sessionId)}` : "";
  return `${PROFILE_L2_KEY_PREFIX}${buildIsolationScope(ctx)}${sourceSession}`;
}

function parseProfileL2Key(key: string): ProfileIsolation | undefined {
  if (!key.startsWith(PROFILE_L2_KEY_PREFIX)) return undefined;
  return parseProfileIsolationScope(key.slice(PROFILE_L2_KEY_PREFIX.length));
}

function profileStoragePrefixForScope(scope: string): string {
  return `profiles/${encodeURIComponent(scope)}/`;
}

function scopedStorage(storage: StorageAdapter | undefined, ctx?: ProfileIsolation): StorageAdapter | undefined {
  return storage ? createScopedStorageAdapter(storage, profileStoragePrefixForScope(buildIsolationScope(ctx))) : undefined;
}

function scopedStorageForScope(storage: StorageAdapter | undefined, scope: string): StorageAdapter | undefined {
  if (!storage || scope === DEFAULT_PROFILE_SCOPE) return storage;
  return createScopedStorageAdapter(storage, profileStoragePrefixForScope(scope));
}

function scopedDataDir(dataDir: string, ctx?: ProfileIsolation): string {
  return scopedDataDirForScope(dataDir, buildIsolationScope(ctx));
}

function scopedDataDirForScope(dataDir: string, scope: string): string {
  return scope === DEFAULT_PROFILE_SCOPE ? dataDir : path.join(dataDir, "profiles", encodeURIComponent(scope));
}

function profileOptionsForScope(scope: string): ProfileScopeOptions | undefined {
  if (scope === DEFAULT_PROFILE_SCOPE) return undefined;
  return { scope, isolation: parseProfileIsolationScope(scope) };
}

async function discoverProfileScopes(dataDir: string, storage: StorageAdapter | undefined, logger: Logger): Promise<string[]> {
  const scopes = new Set<string>();
  if (storage) {
    try {
      const result = await storage.getBackend().listObjects("profiles/", { recursive: true, maxKeys: 10000 });
      for (const entry of result.entries) {
        const rest = entry.key.startsWith("profiles/") ? entry.key.slice("profiles/".length) : "";
        const encoded = rest.split("/")[0];
        if (encoded) scopes.add(decodeURIComponent(encoded));
      }
    } catch (err) {
      logger.debug?.(`${TAG} [L3] Failed to discover storage profile scopes: ${err instanceof Error ? err.message : String(err)}`);
    }
  } else {
    try {
      const entries = await fs.promises.readdir(path.join(dataDir, "profiles"), { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory()) scopes.add(decodeURIComponent(entry.name));
      }
    } catch {
      // No scoped profiles yet; legacy root mode below handles existing unscoped scene files.
    }
  }
  return Array.from(scopes).sort();
}

// ============================
// Logger interface
// ============================

/** @deprecated Use `Logger` from `../core/types.js` directly. */
export type PipelineLogger = Logger;

// ============================
// Factory options
// ============================

export interface PipelineFactoryOptions {
  /** Plugin data directory (L0, records, scene_blocks, vectors.db, etc.). */
  pluginDataDir: string;
  /** Parsed memory-tdai config. */
  cfg: MemoryTdaiConfig;
  /** OpenClaw config object (needed for LLM calls in L1). */
  openclawConfig: unknown;
  /** Logger instance. */
  logger: PipelineLogger;
  /** Session filter (optional, defaults to empty). */
  sessionFilter?: SessionFilter;
  /** Host-neutral LLM runner for L1 extraction (text-only, enableTools=false). */
  l1LlmRunner?: import("../core/types.js").LLMRunner;
  /** Host-neutral LLM runner for L2/L3 (tool-call enabled, enableTools=true). */
  l2l3LlmRunner?: import("../core/types.js").LLMRunner;
}

// ============================
// Factory result
// ============================

export interface PipelineInstance {
  /** The pipeline scheduler. */
  scheduler: MemoryPipelineManager;
  /** VectorStore (undefined if init failed or degraded). */
  vectorStore: IMemoryStore | undefined;
  /** EmbeddingService (undefined if not configured or init failed). */
  embeddingService: EmbeddingService | undefined;
  /**
   * Destroy all resources (scheduler, VectorStore, EmbeddingService).
   * Call this on shutdown / cleanup.
   */
  destroy: () => Promise<void>;
}

// ============================
// Data directory init
// ============================

/**
 * Ensure all required data subdirectories exist under `pluginDataDir`.
 * Safe to call multiple times (mkdirSync with `recursive: true`).
 *
 * When a StorageAdapter is provided, local directory creation is skipped
 * because files are stored remotely (COS). The backend handles path creation.
 */
export function initDataDirectories(dataDir: string, storage?: StorageAdapter): void {
  if (storage) return; // COS mode: no local directories needed
  const dirs = ["conversations", "records", "scene_blocks", ".metadata", ".backup"];
  for (const sub of dirs) {
    fs.mkdirSync(path.join(dataDir, sub), { recursive: true });
  }
}

// ============================
// Store init (once-async singleton)
// ============================

export interface StoreInitResult {
  vectorStore: IMemoryStore | undefined;
  embeddingService: EmbeddingService | undefined;
  /** Whether a background re-index is needed (embedding config changed). */
  needsReindex: boolean;
  reindexReason?: string;
}

/**
 * Cached store init promises — keyed by `pluginDataDir` so that different
 * data directories (e.g. live runtime vs. seed output) each get their own
 * store instance, while concurrent callers for the *same* directory share
 * one initialization.
 */
const _storeInitCache = new Map<string, Promise<StoreInitResult>>();

/**
 * Initialize store backend and (optionally) EmbeddingService.
 *
 * **Once-async semantics per dataDir**: the first call for a given
 * `pluginDataDir` creates the store and caches the result; subsequent
 * calls with the same dir return the cached Promise immediately.
 * Call `resetStores()` during shutdown to clear the cache.
 *
 * Supports both SQLite (sync init) and TCVDB (async init) backends.
 */
export function initStores(
  cfg: MemoryTdaiConfig,
  pluginDataDir: string,
  logger: PipelineLogger,
): Promise<StoreInitResult> {
  const key = pluginDataDir;
  if (!_storeInitCache.has(key)) {
    _storeInitCache.set(key, _doInitStores(cfg, pluginDataDir, logger));
  }
  return _storeInitCache.get(key)!;
}

/**
 * Reset the cached store singleton(s).
 *
 * Call this during `gateway_stop` (after closing the actual store/embedding
 * resources) so that a subsequent `register()` on hot-restart can
 * re-initialize fresh instances.
 *
 * @param pluginDataDir  If provided, only clear the cache for that dir.
 *                       If omitted, clear all cached stores.
 */
export function resetStores(pluginDataDir?: string): void {
  if (pluginDataDir) {
    _storeInitCache.delete(pluginDataDir);
  } else {
    _storeInitCache.clear();
  }
}

/**
 * Internal: actual store initialization logic (called once by the cache).
 */
async function _doInitStores(
  cfg: MemoryTdaiConfig,
  pluginDataDir: string,
  logger: PipelineLogger,
): Promise<StoreInitResult> {
  let vectorStore: IMemoryStore | undefined;
  let embeddingService: EmbeddingService | undefined;
  let needsReindex = false;
  let reindexReason: string | undefined;

  try {
    const bundle = createStoreBundle(cfg, {
      dataDir: pluginDataDir,
      logger,
    });
    vectorStore = bundle.store;
    embeddingService = bundle.embedding ?? undefined;

    const providerInfo = embeddingService?.getProviderInfo();
    const initResult = await vectorStore.init(providerInfo);

    if (vectorStore.isDegraded()) {
      throw new Error(`${TAG} VectorStore is in degraded mode — refusing to proceed without functional store`);
    } else {
      logger.debug?.(
        `${TAG} Store initialized: backend=${cfg.storeBackend}, provider=${cfg.embedding.provider}`,
      );
      needsReindex = initResult.needsReindex;
      reindexReason = initResult.reason;

      // ── Manifest: first-write + config-drift detection ──
      try {
        const currentStoreInfo = buildStoreInfo(bundle.storeSnapshot);
        const existing = readManifest(pluginDataDir);

        if (!existing) {
          // First init — write manifest
          const manifest: Manifest = {
            version: 1,
            createdAt: new Date().toISOString(),
            store: currentStoreInfo,
            seed: null,
          };
          writeManifest(pluginDataDir, manifest);
          logger.debug?.(`${TAG} Manifest created: ${JSON.stringify(currentStoreInfo)}`);
        } else {
          // Compare persisted store binding against current config
          const diffs = diffStoreBinding(existing.store, currentStoreInfo);
          if (diffs.length > 0) {
            logger.debug?.(
              `${TAG} Store config differs from initial binding recorded in manifest ` +
              `(${diffs.join("; ")}). ` +
              `This is expected if the storage backend was switched intentionally.`,
            );
          }
        }
      } catch (err) {
        logger.warn(`${TAG} Failed to read/write manifest (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  } catch (err) {
    logger.warn(
      `${TAG} Store init failed; vector/FTS recall and dedup conflict detection will be unavailable: ${err instanceof Error ? err.message : String(err)}`,
    );
    vectorStore = undefined;
    embeddingService = undefined;
  }

  return { vectorStore, embeddingService, needsReindex, reindexReason };
}

// ============================
// L1 Runner factory
// ============================

/**
 * Create the standard L1 runner function.
 *
 * Reads L0 messages (from VectorStore DB or JSONL fallback), groups by sessionId,
 * runs extractL1Memories for each group, and updates the checkpoint cursor.
 */
export function createL1Runner(opts: {
  pluginDataDir: string;
  cfg: MemoryTdaiConfig;
  openclawConfig: unknown;
  vectorStore: IMemoryStore | undefined;
  embeddingService: EmbeddingService | undefined;
  logger: PipelineLogger;
  /**
   * Getter for the plugin instance ID used for metric reporting.
   * Called at runner execution time (not at creation time) so that the ID is
   * available even when the runner is wired before instanceId is resolved.
   * Metrics are skipped when the getter returns undefined.
   */
  getInstanceId?: () => string | undefined;
  /** Host-neutral LLM runner for L1 extraction (standalone/gateway mode). */
  llmRunner?: import("../core/types.js").LLMRunner;
  /** StorageAdapter for file operations (COS/local). */
  storage?: StorageAdapter;
}): (params: { sessionKey: string }) => Promise<{
  processedCount: number;
  storedCount: number;
  /** True iff the over-fetch returned > L1_BATCH_PROCESS rows (i.e. there's residual past the cursor). */
  hasMore: boolean;
  /** True iff the over-fetch returned exactly L1_BATCH_QUERY rows (i.e. likely large backlog). */
  hasFullBacklog: boolean;
  profileScopes: string[];
}> {
  const { pluginDataDir, cfg, openclawConfig, vectorStore, embeddingService, logger, getInstanceId, llmRunner, storage } = opts;
  const config = openclawConfig as Record<string, unknown> | undefined;

  return async ({ sessionKey }) => {
    if (!config && !llmRunner) {
      logger.debug?.(`${TAG} [l1] No OpenClaw config and no LLM runner, skipping L1 extraction`);
      return { processedCount: 0, storedCount: 0, hasMore: false, hasFullBacklog: false, profileScopes: [] };
    }

    const checkpoint = new CheckpointManager(pluginDataDir, logger, storage);
    const cp = await checkpoint.read();
    const runnerState = checkpoint.getRunnerState(cp, sessionKey);

    logger.info(
      `${TAG} [l1] Session ${sessionKey}: l1_cursor=${runnerState.last_l1_cursor || "(start)"}`,
    );

    try {
      // ── Step 1: over-fetch L0 from DB (or JSONL fallback) ──
      //
      // Pull at most L1_BATCH_QUERY (= 2N) rows past the cursor. We then keep
      // the oldest L1_BATCH_PROCESS (= N) for actual processing and use the
      // remaining rows merely as a *signal* to detect backlog. See file-level
      // comment on L1_BATCH_PROCESS / L1_BATCH_QUERY for rationale.
      type FlatMessage = ConversationMessage & { sessionId: string; teamId?: string; taskId?: string; userId: string; agentId: string; recordedAtMs: number };
      let flat: FlatMessage[] = [];
      let queriedCount = 0;

      if (vectorStore && !vectorStore.isDegraded()) {
        const l1Cursor = runnerState.last_l1_cursor > 0
          ? runnerState.last_l1_cursor
          : undefined;
        const dbGroups = await vectorStore.queryL0GroupedBySessionId(sessionKey, l1Cursor, L1_BATCH_QUERY);
        for (const g of dbGroups) {
          for (const m of g.messages) {
            flat.push({
              id: m.id,
              role: m.role as "user" | "assistant",
              content: m.content,
              timestamp: m.timestamp,
              sessionId: g.sessionId,
              teamId: g.teamId,
              taskId: g.taskId,
              userId: g.userId,
              agentId: g.agentId,
              recordedAtMs: m.recordedAtMs,
            });
          }
        }
        queriedCount = flat.length;
        logger.debug?.(`${TAG} [l1] L0 data source: VectorStore DB, fetched ${queriedCount} rows (limit=${L1_BATCH_QUERY})`);
      } else {
        logger.debug?.(`${TAG} [l1] L0 data source: JSONL files (VectorStore unavailable)`);
        const jsonlGroups = await readConversationMessagesGroupedBySessionId(
          sessionKey,
          pluginDataDir,
          runnerState.last_l1_cursor || undefined,
          logger,
          L1_BATCH_QUERY,
        );
        // NOTE: readConversationMessagesGroupedBySessionId's `limit` semantic
        // historically retains the **newest** N rows when truncating. That is
        // wrong for our backlog-progress-by-cursor model. Since the JSONL path
        // is a degraded fallback (only hit when VectorStore is unavailable),
        // we accept this minor inconsistency for now and rely on the DB path
        // being the production code path. Resort to oldest-first by sorting +
        // re-slicing here as a best-effort.
        for (const g of jsonlGroups) {
          for (const m of g.messages) {
            flat.push({
              id: m.id,
              role: m.role as "user" | "assistant",
              content: m.content,
              timestamp: m.timestamp,
              sessionId: g.sessionId,
              teamId: undefined,
              taskId: undefined,
              userId: "",
              agentId: "",
              recordedAtMs: m.recordedAtMs,
            });
          }
        }
        // Force chronological (oldest-first) ordering by recordedAtMs ↑ then timestamp ↑.
        flat.sort((a, b) => (a.recordedAtMs - b.recordedAtMs) || (a.timestamp - b.timestamp));
        queriedCount = flat.length;
      }

      if (queriedCount === 0) {
        logger.debug?.(`${TAG} [l1] No new L0 messages for session ${sessionKey}`);
        return { processedCount: 0, storedCount: 0, hasMore: false, hasFullBacklog: false, profileScopes: [] };
      }

      // Re-sort by recordedAtMs ascending (DB path returns ASC already, but
      // groupBy may have permuted ordering across groups; this is cheap).
      flat.sort((a, b) => (a.recordedAtMs - b.recordedAtMs) || (a.timestamp - b.timestamp));

      // ── Step 2: slice the first L1_BATCH_PROCESS rows + same-ms boundary alignment ──
      //
      // To advance the cursor safely we must NOT split a group of rows that
      // share the same recorded_at_ms. Otherwise the next round's filter
      // `recorded_at_ms > cursor` would skip the trailing siblings of the
      // boundary millisecond. Concretely: if rows 20 and 21 carry the same
      // recordedAtMs, we extend the slice past row 21 (and any further siblings)
      // until we hit a strictly greater recordedAtMs or exhaust the buffer.
      //
      // Cost: at most a handful of extra rows per round (bounded by how many
      // siblings share one millisecond). Benefit: zero data loss across
      // millisecond-collision boundaries (e.g. seed bulk-load, multi-message
      // agent_end where all rows are stamped with one `now`).
      let sliceEnd = Math.min(L1_BATCH_PROCESS, flat.length);
      if (sliceEnd < flat.length) {
        const boundaryMs = flat[sliceEnd - 1].recordedAtMs;
        while (sliceEnd < flat.length && flat[sliceEnd].recordedAtMs === boundaryMs) {
          sliceEnd++;
        }
      }
      const processed = flat.slice(0, sliceEnd);

      // ── Step 3: re-group sliced messages by isolation tuple + sessionId (chronological within each group) ──
      const groupMap = new Map<string, { sessionId: string; teamId?: string; taskId?: string; userId: string; agentId: string; messages: ConversationMessage[] }>();
      let maxRecordedAtMs = 0;
      for (const m of processed) {
        if (m.recordedAtMs > maxRecordedAtMs) maxRecordedAtMs = m.recordedAtMs;
        const groupKey = `${m.userId}\u0000${m.agentId}\u0000${m.sessionId}`;
        let g = groupMap.get(groupKey);
        if (!g) {
          g = { sessionId: m.sessionId, teamId: m.teamId, taskId: m.taskId, userId: m.userId, agentId: m.agentId, messages: [] };
          groupMap.set(groupKey, g);
        }
        g.messages.push({ id: m.id, role: m.role, content: m.content, timestamp: m.timestamp });
      }
      const groups: Array<{ sessionId: string; teamId?: string; taskId?: string; userId: string; agentId: string; messages: ConversationMessage[] }> = [];
      for (const group of groupMap.values()) {
        groups.push(group);
      }
      // Sort groups by earliest timestamp so extractL1Memories sees them in
      // the same order they were captured (matches pre-existing behavior).
      groups.sort((a, b) => a.messages[0].timestamp - b.messages[0].timestamp);

      // ── Step 4: backlog detection ──
      //
      // queriedCount is bounded by LIMIT L1_BATCH_QUERY (= 2N).
      // sliceEnd may exceed L1_BATCH_PROCESS due to boundary alignment but
      // never exceeds queriedCount.
      //
      //   - hasFullBacklog: queriedCount === L1_BATCH_QUERY AND there are
      //     unprocessed rows in this batch (sliceEnd < queriedCount). DB
      //     returned a full page → likely many more rows past the cursor;
      //     pipeline-manager / executor enqueues the next L1 task immediately.
      //   - hasMore: any unprocessed row in this batch (queriedCount > sliceEnd)
      //     that is not also flagged as full backlog → small tail; defer to
      //     the standard l1Idle timer.
      //
      // EDGE CASE: if queriedCount === L1_BATCH_QUERY and ALL 2N rows share a
      // single recordedAtMs, boundary alignment cannot detect siblings beyond
      // the LIMIT and `sliceEnd` will end up at queriedCount (everything
      // processed, no unprocessed rows). The cursor advances to that ms; the
      // next round's `> cursor` filter would skip any further same-ms siblings
      // existing past the LIMIT. This is unreachable under realistic capture
      // patterns (agent_end writes ≤ ~10 rows per `now`; seed assigns a fresh
      // `now` per round). If hit, see TODO below for cursor-tiebreaker fix.
      // TODO(known-issue): switch to (recorded_at, record_id) composite cursor
      //   to defend against ≥2N rows sharing one recorded_at_ms.
      const hasUnprocessedInBatch = queriedCount > sliceEnd;
      const hasFullBacklog = queriedCount === L1_BATCH_QUERY && hasUnprocessedInBatch;
      const hasMore = hasUnprocessedInBatch && !hasFullBacklog;

      const totalMessages = processed.length;
      logger.info(
        `${TAG} [l1] Processing ${totalMessages} L0 messages across ${groups.length} sessionId group(s) ` +
        `for session ${sessionKey} (queried=${queriedCount}, sliceEnd=${sliceEnd}, ` +
        `hasMore=${hasMore}, hasFullBacklog=${hasFullBacklog})`,
      );

      let totalExtracted = 0;
      let totalStored = 0;
      let lastSceneName: string | undefined;
      const profileScopes = new Set<string>();

      for (const group of groups) {
        logger.debug?.(
          `${TAG} [l1] Group sessionId=${group.sessionId || "(empty)"}: ${group.messages.length} messages`,
        );

        const l1Result = await extractL1Memories({
          messages: group.messages,
          sessionKey,
          sessionId: group.sessionId,
          taskId: group.taskId,
          teamId: group.teamId,
          userId: group.userId,
          agentId: group.agentId,
          baseDir: pluginDataDir,
          config,
          options: {
            enableDedup: cfg.extraction.enableDedup,
            maxMemoriesPerSession: cfg.extraction.maxMemoriesPerSession,
            model: cfg.extraction.model,
            promptMode: cfg.extraction.promptMode,
            previousSceneName: lastSceneName ?? (runnerState.last_scene_name || undefined),
            vectorStore,
            embeddingService,
            conflictRecallTopK: cfg.embedding.conflictRecallTopK,
            embeddingTimeoutMs: cfg.embedding.captureTimeoutMs ?? cfg.embedding.timeoutMs,
            llmRunner,
          },
          logger,
          instanceId: getInstanceId?.(),
          storage,
        });

        totalExtracted += l1Result.extractedCount;
        totalStored += l1Result.storedCount;
        if (l1Result.storedCount > 0) {
          // L2/L3 output is team+agent scoped, but each L2 extraction input must
          // stay bounded to the source session that just produced L1. Encode the
          // source session in the L2 task key; buildIsolationScope() will ignore
          // it later when choosing the profile output directory.
          profileScopes.add(buildProfileL2Key({
            teamId: group.teamId,
            userId: group.userId,
            agentId: group.agentId,
            sessionId: group.sessionId,
          }));
        }
        if (l1Result.lastSceneName) {
          lastSceneName = l1Result.lastSceneName;
        }
      }

      // Use maxRecordedAtMs (write time) of the **processed** slice as cursor —
      // always positive, TCVDB-safe. Boundary alignment guarantees we will not
      // skip same-ms siblings on the next round.
      await checkpoint.markL1ExtractionComplete(sessionKey, totalStored, maxRecordedAtMs || undefined, lastSceneName);
      logger.info(
        `${TAG} [l1] L1 complete: extracted=${totalExtracted}, stored=${totalStored} (${groups.length} group(s))`,
      );

      return { processedCount: totalMessages, storedCount: totalStored, hasMore, hasFullBacklog, profileScopes: Array.from(profileScopes) };
    } catch (err) {
      logger.error(`${TAG} [l1] L1 failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
      throw err;
    }
  };
}

// ============================
// Persister factory
// ============================

/**
 * Create the standard pipeline state persister.
 * Saves pipeline session states to the checkpoint file.
 */
export function createPersister(
  pluginDataDir: string,
  logger: PipelineLogger,
  storage?: StorageAdapter,
): (states: Record<string, PipelineSessionState>) => Promise<void> {
  return async (states) => {
    const checkpoint = new CheckpointManager(pluginDataDir, logger, storage);
    await checkpoint.mergePipelineStates(states);
  };
}

// ============================
// L2 Runner factory
// ============================

/**
 * Create the standard L2 runner function (scene extraction).
 *
 * Reads L1 memory records (incremental via VectorStore or JSONL fallback),
 * runs SceneExtractor, and returns the latest cursor for pipeline-manager
 * to track incremental progress.
 *
 * Used by both `index.ts` (live runtime) and `seed-runtime.ts` (seed CLI).
 */
export function createL2Runner(opts: {
  pluginDataDir: string;
  cfg: MemoryTdaiConfig;
  openclawConfig: unknown;
  vectorStore: IMemoryStore | undefined;
  logger: PipelineLogger;
  instanceId?: string;
  /** Host-neutral LLM runner for L2 scene extraction (standalone/gateway mode). Must have enableTools=true. */
  llmRunner?: import("../core/types.js").LLMRunner;
  /** StorageAdapter for file operations (COS/local). */
  storage?: StorageAdapter;
}): L2Runner {
  const { pluginDataDir, cfg, openclawConfig, vectorStore, logger, instanceId, llmRunner, storage } = opts;
  let profileBaseline = new Map<string, { version: number; contentMd5: string; createdAtMs: number }>();

  return async (sessionKey: string, cursor?: string) => {
    const profileFilter = parseProfileL2Key(sessionKey);
    logger.debug?.(
      `${TAG} [L2] session=${sessionKey}, profile=${profileFilter ? buildIsolationScope(profileFilter) : "(legacy-session)"}, updatedAfter=${cursor ?? "(full)"}`,
    );

    if (!openclawConfig && !llmRunner) {
      logger.warn(`${TAG} [L2] No OpenClaw config and no LLM runner, skipping scene extraction`);
      return;
    }

    let records: Array<{ content: string; created_at: string; id: string; updatedAt: string; teamId?: string; userId?: string; agentId?: string; sessionId?: string; taskId?: string }>;

    if (vectorStore && !vectorStore.isDegraded()) {
      const { queryMemoryRecords } = await import("../core/record/l1-reader.js");
      const memRecords = await queryMemoryRecords(vectorStore, profileFilter ? {
        teamId: profileFilter.teamId,
        userId: profileFilter.userId,
        agentId: profileFilter.agentId,
        sessionId: profileFilter.sessionId,
        updatedAfter: cursor,
      } : {
        sessionKey,
        updatedAfter: cursor,
      }, logger);

      if (memRecords.length === 0) {
        logger.debug?.(
          `${TAG} [L2] No new L1 records since cursor (session=${sessionKey}, updatedAfter=${cursor ?? "(full)"}), skipping scene extraction`,
        );
        return { skipped: true };
      }

      logger.debug?.(
        `${TAG} [L2] Incremental query returned ${memRecords.length} record(s) (session=${sessionKey})`,
      );

      records = memRecords.map((r) => ({
        content: r.content,
        created_at: r.createdAt,
        id: r.id,
        updatedAt: r.updatedAt,
        teamId: r.teamId,
        userId: r.userId,
        agentId: r.agentId,
        sessionId: r.sessionId,
        taskId: r.taskId,
      }));
    } else {
      throw new Error(`${TAG} [L2] VectorStore unavailable — cannot read L1 memories for scene extraction (session=${sessionKey})`);
    }

    if (records.length === 0) {
      logger.debug?.(`${TAG} [L2] No new L1 records found (session=${sessionKey}), skipping scene extraction`);
      return;
    }

    const grouped = new Map<string, typeof records>();
    for (const record of records) {
      const key = buildIsolationScope(record);
      const list = grouped.get(key) ?? [];
      list.push(record);
      grouped.set(key, list);
    }

    let processedTotal = 0;
    let anyEmptyExtraction = false;
    for (const groupRecords of grouped.values()) {
      const ctx = groupRecords[0];
      const groupStorage = scopedStorage(storage, ctx);
      const groupDataDir = scopedDataDir(pluginDataDir, ctx);
      const groupScope = buildIsolationScope(ctx);
      const groupProfileOptions = profileOptionsForScope(groupScope);
      const groupBaseline = vectorStore?.pullProfiles && !vectorStore.isDegraded()
        ? await pullProfilesToLocal(groupDataDir, vectorStore, logger, groupStorage, groupProfileOptions)
        : profileBaseline;

      const extractor = new SceneExtractor({
        dataDir: groupDataDir,
        config: openclawConfig!,
        model: cfg.persona.model,
        promptMode: cfg.persona.promptMode,
        maxScenes: cfg.persona.maxScenes,
        sceneBackupCount: cfg.persona.sceneBackupCount,
        logger,
        instanceId,
        llmRunner,
        storage: groupStorage,
        // langfuse: 透传身份四元组，UI 上可按 user/session 列过滤
        traceContext: { teamId: ctx.teamId, userId: ctx.userId, agentId: ctx.agentId, sessionId: ctx.sessionId },
      });

      const memories = groupRecords.map((r) => ({
        content: r.content,
        created_at: r.created_at,
        id: r.id,
      }));

      const preCheckpoint = new CheckpointManager(groupDataDir, logger, groupStorage);
      const preState = await preCheckpoint.read();
      const preScenesProcessed = preState.scenes_processed;
      const preTotalProcessed = preState.total_processed;

      const extractResult = await extractor.extract(memories);
      if (!(extractResult.success && extractResult.memoriesProcessed > 0)) continue;
      if (extractResult.emptyExtraction) {
        anyEmptyExtraction = true;
        logger.warn(`${TAG} [L2] Extraction produced no file changes (empty run), skipping checkpoint increment`);
        continue;
      }

      const checkpoint = new CheckpointManager(groupDataDir, logger, groupStorage);
      const postState = await checkpoint.read();
      if (postState.scenes_processed < preScenesProcessed || postState.total_processed < preTotalProcessed) {
        logger.warn(
          `${TAG} [L2] ⚠️ Checkpoint corruption detected! ` +
          `scenes_processed: ${preScenesProcessed} → ${postState.scenes_processed}, ` +
          `total_processed: ${preTotalProcessed} → ${postState.total_processed}. Repairing...`,
        );
        await checkpoint.write({
          ...postState,
          scenes_processed: Math.max(postState.scenes_processed, preScenesProcessed),
          total_processed: Math.max(postState.total_processed, preTotalProcessed),
        });
        logger.info(`${TAG} [L2] Checkpoint repaired`);
      }

      if (vectorStore && supportsProfileSyncWrite(vectorStore)) {
        await syncLocalProfilesToStore(groupDataDir, vectorStore, groupBaseline, logger, groupStorage, groupProfileOptions);
      }
      await checkpoint.incrementScenesProcessed();
      processedTotal += extractResult.memoriesProcessed;
    }

    if (processedTotal > 0) {
      const latestCursor = records.reduce((latest, r) => r.updatedAt > latest ? r.updatedAt : latest, "");
      logger.debug?.(`${TAG} [L2] Extraction complete: processed=${processedTotal}, latestCursor=${latestCursor}`);
      return { latestCursor: latestCursor || undefined };
    }
    if (anyEmptyExtraction) return { skipped: true };
  };
}

// ============================
// L3 Runner factory
// ============================

/**
 * Create the standard L3 runner function (persona generation).
 *
 * Uses PersonaTrigger to check if generation is needed, then runs
 * PersonaGenerator. Used by both `index.ts` and `seed-runtime.ts`.
 */
export function createL3Runner(opts: {
  pluginDataDir: string;
  cfg: MemoryTdaiConfig;
  openclawConfig: unknown;
  vectorStore?: IMemoryStore;
  logger: PipelineLogger;
  instanceId?: string;
  /** Host-neutral LLM runner for L3 persona generation (standalone/gateway mode). Must have enableTools=true. */
  llmRunner?: import("../core/types.js").LLMRunner;
  /** StorageAdapter for file operations (COS/local). */
  storage?: StorageAdapter;
}): L3Runner {
  const { pluginDataDir, cfg, openclawConfig, vectorStore, logger, instanceId, llmRunner, storage } = opts;

  return async () => {
    const scopes = await discoverProfileScopes(pluginDataDir, storage, logger);
    const executionScopes = scopes.length > 0 ? scopes : [DEFAULT_PROFILE_SCOPE];
    let generatedAny = false;

    for (const scope of executionScopes) {
      const scopedDir = scopedDataDirForScope(pluginDataDir, scope);
      const scopedStore = scopedStorageForScope(storage, scope);
      const profileOptions = profileOptionsForScope(scope);

      const trigger = new PersonaTrigger({
        dataDir: scopedDir,
        interval: cfg.persona.triggerEveryN,
        logger,
        storage: scopedStore,
      });

      const { should, reason } = await trigger.shouldGenerate();
      if (!should) {
        logger.debug?.(`${TAG} [L3] Persona generation not needed (scope=${scope})`);
        continue;
      }

      if (!openclawConfig && !llmRunner) {
        logger.warn(`${TAG} [L3] No OpenClaw config and no LLM runner, skipping persona generation`);
        return;
      }

      // Guard: no scene files → nothing to generate from. Skip without marking
      // checkpoint so cold-start trigger remains available for the next attempt.
      const { readSceneIndex } = await import("../core/scene/scene-index.js");
      const sceneIndex = await readSceneIndex(scopedDir, scopedStore);
      if (sceneIndex.length === 0) {
        logger.info(`${TAG} [L3] No scene files available for scope=${scope}, skipping (checkpoint unchanged)`);
        continue;
      }

      // Pull remote profiles to establish fresh baseline before generation.
      // This ensures syncLocalProfilesToStore() has correct baselineVersion
      // for the optimistic-lock check instead of defaulting to 0.
      let profileBaseline = new Map<string, { version: number; contentMd5: string; createdAtMs: number }>();
      if (vectorStore?.pullProfiles && !vectorStore.isDegraded()) {
        profileBaseline = await pullProfilesToLocal(scopedDir, vectorStore, logger, scopedStore, profileOptions);
      }

      logger.info(`${TAG} [L3] Starting persona generation: ${reason} (scope=${scope})`);
      // 反解 scope 拿回 teamId/userId/agentId/sessionId 给 langfuse trace 用
      const scopeIsolation = parseProfileIsolationScope(scope);
      const generator = new PersonaGenerator({
        dataDir: scopedDir,
        config: openclawConfig,
        model: cfg.persona.model,
        promptMode: cfg.persona.promptMode,
        backupCount: cfg.persona.backupCount,
        logger,
        instanceId,
        llmRunner,
        storage: scopedStore,
        traceContext: scopeIsolation,
      });
      const genResult = await generator.generateLocalPersona(reason);

      const checkpoint = new CheckpointManager(scopedDir, logger, scopedStore);
      const cp = await checkpoint.read();
      const personaMarker = cp.total_processed;

      if (!genResult) {
        logger.info(`${TAG} [L3] Persona generation skipped (no changes, scope=${scope})`);
        await checkpoint.markPersonaGenerated(personaMarker);
        continue;
      }

      if (vectorStore && supportsProfileSyncWrite(vectorStore)) {
        await syncLocalProfilesToStore(scopedDir, vectorStore, profileBaseline, logger, scopedStore, profileOptions);
      }

      await checkpoint.markPersonaGenerated(personaMarker);
      generatedAny = true;
      logger.info(`${TAG} [L3] Persona generation succeeded (scope=${scope})`);
    }

    if (!generatedAny) {
      logger.debug?.(`${TAG} [L3] No scoped persona generated`);
    }
  };
}

// ============================
// Pipeline Manager factory
// ============================

/**
 * Create a MemoryPipelineManager with the standard config mapping.
 */
export function createPipelineManager(
  cfg: MemoryTdaiConfig,
  logger: PipelineLogger,
  sessionFilter?: SessionFilter,
): MemoryPipelineManager {
  return new MemoryPipelineManager(
    {
      everyNConversations: cfg.pipeline.everyNConversations,
      enableWarmup: cfg.pipeline.enableWarmup,
      l1: { idleTimeoutSeconds: cfg.pipeline.l1IdleTimeoutSeconds },
      l2: {
        delayAfterL1Seconds: cfg.pipeline.l2DelayAfterL1Seconds,
        minIntervalSeconds: cfg.pipeline.l2MinIntervalSeconds,
        maxIntervalSeconds: cfg.pipeline.l2MaxIntervalSeconds,
        sessionActiveWindowHours: cfg.pipeline.sessionActiveWindowHours,
      },
    },
    logger,
    sessionFilter ?? new SessionFilter([]),
  );
}

// ============================
// Full pipeline factory
// ============================

/**
 * Create a fully wired pipeline instance: VectorStore + EmbeddingService +
 * MemoryPipelineManager with L1 runner and persister attached.
 *
 * This is the high-level entry point used by both `index.ts` and `seed-runtime.ts`.
 * Callers should attach L2/L3 runners after creation using `createL2Runner()`
 * and `createL3Runner()` from this module.
 */
export async function createPipeline(opts: PipelineFactoryOptions): Promise<PipelineInstance> {
  const { pluginDataDir, cfg, openclawConfig, logger, sessionFilter, l1LlmRunner } = opts;

  // Ensure data directories exist
  initDataDirectories(pluginDataDir);

  // Initialize stores (once-async: reuses cached result if already initialized)
  const stores = await initStores(cfg, pluginDataDir, logger);
  const { vectorStore, embeddingService } = stores;

  // Create pipeline manager
  const scheduler = createPipelineManager(cfg, logger, sessionFilter);

  // Wire L1 runner
  scheduler.setL1Runner(createL1Runner({
    pluginDataDir,
    cfg,
    openclawConfig,
    vectorStore,
    embeddingService,
    logger,
    llmRunner: l1LlmRunner,
  }));

  // Wire persister
  scheduler.setPersister(createPersister(pluginDataDir, logger));

  // Destroy function
  const destroy = async () => {
    logger.info(`${TAG} Destroying pipeline...`);
    await scheduler.destroy();
    if (vectorStore) {
      logger.info(`${TAG} Closing VectorStore`);
      vectorStore.close();
    }
    if (embeddingService?.close) {
      try {
        logger.info(`${TAG} Closing EmbeddingService`);
        await embeddingService.close();
      } catch (err) {
        logger.warn(`${TAG} Error closing EmbeddingService: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    resetStores(pluginDataDir);
    logger.info(`${TAG} Pipeline destroyed`);
  };

  return { scheduler, vectorStore, embeddingService, destroy };
}

// ============================
// V2: StateBackend-based pipeline factory (需求 #8)
// ============================

import type { IStateBackend } from "../core/state/types.js";
import { StatefulPipelineManager } from "./stateful-pipeline-manager.js";

/**
 * Create a StatefulPipelineManager that uses IStateBackend for all state.
 *
 * Drop-in replacement for createPipelineManager() when running with an
 * externalized state backend.
 */
export function createStatefulPipelineManager(
  cfg: MemoryTdaiConfig,
  stateBackend: IStateBackend,
  instanceId: string,
  logger: PipelineLogger,
  sessionFilter?: SessionFilter,
): StatefulPipelineManager {
  return new StatefulPipelineManager(
    {
      everyNConversations: cfg.pipeline.everyNConversations,
      enableWarmup: cfg.pipeline.enableWarmup,
      l1: { idleTimeoutSeconds: cfg.pipeline.l1IdleTimeoutSeconds },
      l2: {
        delayAfterL1Seconds: cfg.pipeline.l2DelayAfterL1Seconds,
        minIntervalSeconds: cfg.pipeline.l2MinIntervalSeconds,
        maxIntervalSeconds: cfg.pipeline.l2MaxIntervalSeconds,
        sessionActiveWindowHours: cfg.pipeline.sessionActiveWindowHours,
      },
    },
    stateBackend,
    instanceId,
    logger,
    sessionFilter ?? new SessionFilter([]),
  );
}
