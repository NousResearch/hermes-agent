/**
 * auto-capture hook (v3): records conversation messages locally (L0),
 * then notifies the MemoryPipelineManager for L1/L2/L3 scheduling.
 *
 * Key design decisions:
 * - Always write L0 locally via l0-recorder.
 * - When VectorStore + EmbeddingService are available, also write L0 vector index.
 * - Notify MemoryPipelineManager for L1/L2/L3 trigger evaluation.
 * - L1 Runner reads from VectorStore DB (primary) or L0 JSONL files (fallback).
 * - Extraction is NOT triggered here. The pipeline manager decides when.
 */

import crypto from "node:crypto";
import type { MemoryTdaiConfig } from "../../config.js";
import { CheckpointManager } from "../../utils/checkpoint.js";
import type { MemoryPipelineManager } from "../../utils/pipeline-manager.js";
import { recordConversation } from "../conversation/l0-recorder.js";
import type { ConversationMessage } from "../conversation/l0-recorder.js";
import { DEFAULT_ISOLATION_ID, type IMemoryStore, type L0Record } from "../store/types.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { StorageAdapter } from "../storage/adapter.js";

import type { Logger } from "../types.js";

const TAG = "[memory-tdai] [capture]";

export interface AutoCaptureResult {
  /** Whether the scheduler was notified (conversation count incremented) */
  schedulerNotified: boolean;
  /** Number of messages recorded to L0 */
  l0RecordedCount: number;
  /** Number of L0 message vectors written */
  l0VectorsWritten: number;
  /** Filtered messages for L1 immediate use */
  filteredMessages: ConversationMessage[];
}

/**
 * Generate a unique L0 record ID for vector indexing.
 * Includes an index to distinguish multiple messages within the same round.
 */
function generateL0RecordId(sessionKey: string, index: number): string {
  return `l0_${sessionKey}_${Date.now()}_${index}_${crypto.randomBytes(3).toString("hex")}`;
}

export async function performAutoCapture(params: {
  messages: unknown[];
  sessionKey: string;
  sessionId?: string;
  cfg: MemoryTdaiConfig;
  pluginDataDir: string;
  logger?: Logger;
  scheduler?: MemoryPipelineManager;
  /** Clean original user prompt from before_prompt_build cache (pre-prependContext). */
  originalUserText?: string;
  /**
   * Number of messages in the session at before_prompt_build time.
   * Used by l0-recorder to locate the exact user message that originalUserText
   * corresponds to: rawMessages[originalUserMessageCount] is the polluted user message.
   */
  originalUserMessageCount?: number;
  /** Epoch ms when the plugin was registered (cold-start time).
   *  Used as fallback cursor when checkpoint has no prior timestamp —
   *  prevents the first agent_end from dumping all session history into L0. */
  pluginStartTimestamp?: number;
  /** VectorStore for L0 vector indexing (optional). */
  vectorStore?: IMemoryStore;
  /** EmbeddingService for L0 vector indexing (optional). */
  embeddingService?: EmbeddingService;
  /**
   * Tracks in-flight fire-and-forget background tasks started by this
   * capture (currently: deferred L0 embedding for SQLite-style stores).
   *
   * When provided, each background task's Promise is added to the set
   * on creation and removed on completion.  This lets the owning
   * ``TdaiCore`` instance await all pending background work before
   * closing ``vectorStore`` / ``embeddingService`` in ``destroy()``,
   * so we never hit an already-closed DB connection with a late
   * ``updateL0Embedding`` call.
   *
   * Optional for backwards compatibility — callers that don't care
   * (tests, short-lived CLI invocations) can omit it and accept the
   * pre-fix behaviour (background task may outlive its owner).
   */
  bgTaskRegistry?: Set<Promise<void>>;
  /** StorageAdapter for file operations (COS/local). Falls back to fs when absent. */
  storage?: StorageAdapter;
}): Promise<AutoCaptureResult> {
  const {
    messages, sessionKey, sessionId, cfg, pluginDataDir, logger, scheduler,
    originalUserText, originalUserMessageCount, pluginStartTimestamp,
    vectorStore, embeddingService, bgTaskRegistry, storage,
  } = params;
  const tCaptureStart = performance.now();

  const checkpoint = new CheckpointManager(pluginDataDir, logger, storage);

  // ============================
  // Step 1 + 2: L0 recording + checkpoint update (ATOMIC)
  // ============================
  // These steps are combined inside captureAtomically() to prevent the race
  // condition where two concurrent agent_end events both read the same stale
  // cursor and produce duplicate L0 records. The file lock is held for the
  // entire read-cursor → recordConversation → advance-cursor sequence.
  const tL0RecordStart = performance.now();
  let filteredMessages: ConversationMessage[] = [];
  try {
    await checkpoint.captureAtomically(
      sessionKey,
      pluginStartTimestamp,
      async (afterTimestamp) => {
        logger?.debug?.(`${TAG} L0 capture cursor (per-session, atomic): afterTimestamp=${afterTimestamp} session=${sessionKey}`);

        if (afterTimestamp === pluginStartTimestamp && pluginStartTimestamp && pluginStartTimestamp > 0) {
          logger?.debug?.(
            `${TAG} No per-session checkpoint cursor found for session=${sessionKey} — ` +
            `using pluginStartTimestamp as floor: ` +
            `${afterTimestamp} (${new Date(afterTimestamp).toISOString()})`,
          );
        }

        filteredMessages = await recordConversation({
          sessionKey,
          sessionId,
          rawMessages: messages,
          baseDir: pluginDataDir,
          logger,
          originalUserText,
          afterTimestamp,
          originalUserMessageCount,
          storage,
        });

        if (filteredMessages.length === 0) {
          return null; // Nothing captured — cursor stays unchanged
        }

        logger?.debug?.(`${TAG} L0 recorded: ${filteredMessages.length} messages for session ${sessionKey}`);
        const maxTs = Math.max(...filteredMessages.map((m) => m.timestamp));
        return { maxTimestamp: maxTs, messageCount: filteredMessages.length };
      },
    );
  } catch (err) {
    logger?.error(`${TAG} L0 recording failed: ${err instanceof Error ? err.message : String(err)}`);
  }
  const tL0RecordEnd = performance.now();

  // ============================
  // Step 1.5: L0 vector indexing
  // ============================
  // Two paths depending on store capabilities:
  //
  // A) Store supports updateL0Embedding (sqlite):
  //    - Write metadata + FTS immediately WITHOUT embedding (~ms)
  //    - Fire-and-forget background task: embedBatch + updateL0Embedding
  //    - PERF: avoids blocking agent_end with 2-3s embedding calls
  //
  // B) Store does NOT support updateL0Embedding (VDB / remote):
  //    - Embed synchronously, then upsertL0 with embedding in one call
  //    - VDB backends handle embedding server-side or need it upfront
  const tL0VecStart = performance.now();
  let l0VectorsWritten = 0;
  let l0EmbedTotalMs = 0;
  let l0UpsertTotalMs = 0;
  logger?.debug?.(
    `${TAG} [L0-vec-index] Check: filteredMessages=${filteredMessages.length}, ` +
    `vectorStore=${vectorStore ? "available" : "UNAVAILABLE"}, ` +
    `embeddingService=${embeddingService ? "available" : "UNAVAILABLE"}`,
  );

  const supportsBgEmbed = vectorStore?.supportsDeferredEmbedding === true;

  if (filteredMessages.length > 0 && vectorStore) {
    const now = new Date().toISOString();
    const bgRecords: Array<{ recordId: string; content: string }> = [];
    logger?.debug?.(
      `${TAG} [L0-vec-index] START indexing ${filteredMessages.length} message(s) for session ${sessionKey} ` +
      `(mode=${supportsBgEmbed ? "async-bg" : "sync"})`,
    );

    for (let i = 0; i < filteredMessages.length; i++) {
      const msg = filteredMessages[i];
      try {
        const l0Record: L0Record = {
          id: generateL0RecordId(sessionKey, i),
          sessionKey,
          sessionId: sessionId || DEFAULT_ISOLATION_ID,
          role: msg.role,
          messageText: msg.content,
          recordedAt: now,
          timestamp: msg.timestamp,
        };

        let embedding: Float32Array | undefined;

        if (!supportsBgEmbed && embeddingService) {
          // Path B (VDB): embed synchronously — needed for upsertL0
          // Skip local embed when using server-side embedding (NoopEmbeddingService, dims=0)
          if (embeddingService.getDimensions() === 0) {
            logger?.debug?.(
              `${TAG} [L0-vec-index] Server-side embedding (dims=0), skipping local embed for message ${i}`,
            );
          } else {
            const tEmbedStart = performance.now();
            try {
              embedding = await embeddingService.embed(msg.content);
              l0EmbedTotalMs += performance.now() - tEmbedStart;
              logger?.debug?.(
                `${TAG} [L0-vec-index] Embedding OK: dims=${embedding.length}, ` +
                `norm=${Math.sqrt(Array.from(embedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}`,
              );
            } catch (embedErr) {
              l0EmbedTotalMs += performance.now() - tEmbedStart;
              logger?.warn(
                `${TAG} [L0-vec-index] Embedding FAILED for message ${i}, ` +
                `will write metadata only: ${embedErr instanceof Error ? embedErr.message : String(embedErr)}`,
              );
            }
          }
        }

        // Path A (sqlite): pass undefined embedding — metadata + FTS only
        // Path B (VDB): pass embedding (may be undefined on failure)
        const tUpsertStart = performance.now();
        const upsertOk = await vectorStore.upsertL0(l0Record, supportsBgEmbed ? undefined : embedding);
        l0UpsertTotalMs += performance.now() - tUpsertStart;

        if (upsertOk) {
          l0VectorsWritten++;
          if (supportsBgEmbed) {
            bgRecords.push({ recordId: l0Record.id, content: msg.content });
          }
        } else {
          logger?.warn(`${TAG} [L0-vec-index] upsertL0 returned false for message ${i}`);
        }
      } catch (err) {
        logger?.warn?.(`${TAG} [L0-vec-index] FAILED for message ${i} (non-blocking): ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    const modeLabel = supportsBgEmbed ? "metadata-only, embed=background" : `embed=${l0EmbedTotalMs.toFixed(0)}ms, upsert=${l0UpsertTotalMs.toFixed(0)}ms`;
    logger?.debug?.(`${TAG} [L0-vec-index] DONE: ${l0VectorsWritten}/${filteredMessages.length} records written (${modeLabel})`);

    // Path A only: fire-and-forget background embedding for sqlite stores
    if (supportsBgEmbed && bgRecords.length > 0 && embeddingService) {
      const bgVectorStore = vectorStore;
      const bgEmbeddingService = embeddingService;
      const bgSnapshot = [...bgRecords];
      const bgLogger = logger;

      // Do NOT await — runs in background after response is sent.
      //
      // Register the task in bgTaskRegistry (if provided) so TdaiCore.destroy()
      // can await it before closing vectorStore / embeddingService.  The
      // ``.finally`` clean-up ensures the entry is removed on both success
      // and failure; without that the set would leak and eventually block
      // shutdown indefinitely.
      const bgPromise: Promise<void> = (async () => {
        const tBgStart = performance.now();
        try {
          const texts = bgSnapshot.map((r) => r.content);
          const embeddings = await bgEmbeddingService.embedBatch(texts);

          let bgUpdated = 0;
          for (let i = 0; i < bgSnapshot.length; i++) {
            try {
              const ok = await bgVectorStore.updateL0Embedding!(bgSnapshot[i].recordId, embeddings[i]);
              if (ok) bgUpdated++;
            } catch (err) {
              bgLogger?.warn?.(
                `${TAG} [L0-vec-index-bg] Failed to update embedding for ${bgSnapshot[i].recordId}: ` +
                `${err instanceof Error ? err.message : String(err)}`,
              );
            }
          }
          const bgMs = performance.now() - tBgStart;
          bgLogger?.debug?.(
            `${TAG} [L0-vec-index-bg] Background embedding complete: ${bgUpdated}/${bgSnapshot.length} vectors updated (${bgMs.toFixed(0)}ms)`,
          );
        } catch (err) {
          const bgMs = performance.now() - tBgStart;
          bgLogger?.warn?.(
            `${TAG} [L0-vec-index-bg] Background embedding failed (${bgMs.toFixed(0)}ms, non-fatal): ` +
            `${err instanceof Error ? err.message : String(err)}`,
          );
        }
      })();

      if (bgTaskRegistry) {
        bgTaskRegistry.add(bgPromise);
        void bgPromise.finally(() => {
          bgTaskRegistry.delete(bgPromise);
        });
      }
    }
  } else if (filteredMessages.length > 0) {
    logger?.warn(`${TAG} [L0-vec-index] SKIPPED: vectorStore not available`);
  }
  const tL0VecEnd = performance.now();

  // ============================
  // Step 3: Notify scheduler of this conversation round
  // ============================
  const tNotifyStart = performance.now();
  // Pass empty array: L1 Runner reads from VectorStore DB (or L0 JSONL fallback), not from in-memory buffers.
  if (scheduler) {
    await scheduler.notifyConversation(sessionKey, []);
    logger?.debug?.(`${TAG} Scheduler notified of conversation round (sessionKey=${sessionKey})`);

    const totalMs = performance.now() - tCaptureStart;
    const vecDetail = supportsBgEmbed
      ? `metadata-only, embed=background, msgs=${filteredMessages.length}`
      : `embed=${l0EmbedTotalMs.toFixed(0)}ms, upsert=${l0UpsertTotalMs.toFixed(0)}ms, msgs=${filteredMessages.length}`;
    logger?.info(
      `${TAG} ⏱ Capture timing: total=${totalMs.toFixed(0)}ms, ` +
      `l0Record+checkpoint=${(tL0RecordEnd - tL0RecordStart).toFixed(0)}ms, ` +
      `l0VecIndex=${(tL0VecEnd - tL0VecStart).toFixed(0)}ms (${vecDetail}), ` +
      `notify=${(performance.now() - tNotifyStart).toFixed(0)}ms`,
    );

    return {
      schedulerNotified: true,
      l0RecordedCount: filteredMessages.length,
      l0VectorsWritten,
      filteredMessages,
    };
  }

  const totalMs = performance.now() - tCaptureStart;
  const vecDetail = supportsBgEmbed
    ? `metadata-only, embed=background, msgs=${filteredMessages.length}`
    : `embed=${l0EmbedTotalMs.toFixed(0)}ms, upsert=${l0UpsertTotalMs.toFixed(0)}ms, msgs=${filteredMessages.length}`;
  logger?.info(
    `${TAG} ⏱ Capture timing: total=${totalMs.toFixed(0)}ms, ` +
    `l0Record+checkpoint=${(tL0RecordEnd - tL0RecordStart).toFixed(0)}ms, ` +
    `l0VecIndex=${(tL0VecEnd - tL0VecStart).toFixed(0)}ms (${vecDetail}), ` +
    `notify=${(performance.now() - tNotifyStart).toFixed(0)}ms`,
  );

  logger?.debug?.(`${TAG} No scheduler provided, skipping notification`);
  return {
    schedulerNotified: false,
    l0RecordedCount: filteredMessages.length,
    l0VectorsWritten,
    filteredMessages,
  };
}
