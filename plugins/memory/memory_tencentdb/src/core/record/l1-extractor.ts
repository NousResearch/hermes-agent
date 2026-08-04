/**
 * L1 Memory Extractor: extracts structured memories from L0 conversation messages
 * using a single LLM call with JSON-mode structured output.
 *
 * v3: Aligned with Kenty's prompt — scene segmentation + memory extraction in one call,
 * followed by batch conflict detection.
 *
 * Pipeline:
 * 1. Read recent messages from L0 (split into background + new)
 * 2. Call LLM to extract scene-segmented memories
 * 3. Batch conflict detection against existing records
 * 4. Write to L1 JSONL files
 */

import type { ConversationMessage } from "../conversation/l0-recorder.js";
import { formatExtractionPrompt, getExtractMemoriesSystemPrompt, type MemoryPromptMode } from "../prompts/l1-extraction.js";
import { batchDedup } from "./l1-dedup.js";
import { writeMemory, generateMemoryId } from "./l1-writer.js";
import type { ExtractedMemory, MemoryRecord, MemoryType, DedupDecision } from "./l1-writer.js";
import { CleanContextRunner } from "../../utils/clean-context-runner.js";
import { sanitizeJsonForParse, shouldExtractL1 } from "../../utils/sanitize.js";
import type { IMemoryStore } from "../store/types.js";
import type { EmbeddingService } from "../store/embedding.js";
import { report } from "../report/reporter.js";
import { metricProducer } from "../report/kafka-metric-producer.js";
import { reportL1LatencyMetrics } from "../report/metric-tracking-l1-latency.js";
import type { LLMRunner, Logger, TraceContext } from "../types.js";
import { buildTraceParams } from "../types.js";
import type { StorageAdapter } from "../storage/adapter.js";

const TAG = "[memory-tdai][l1-extractor]";

// ============================
// Types
// ============================

/** A scene segment with its extracted memories (LLM output) */
interface SceneSegment {
  scene_name: string;
  message_ids: string[];
  memories: Array<{
    content: string;
    type: string;
    priority: number;
    source_message_ids: string[];
    metadata: Record<string, unknown>;
  }>;
}

export interface L1ExtractionResult {
  /** Whether extraction succeeded */
  success: boolean;
  /** Number of memories extracted */
  extractedCount: number;
  /** Number of memories actually stored (after dedup) */
  storedCount: number;
  /** The memory records that were stored */
  records: MemoryRecord[];
  /** Scene names detected during extraction */
  sceneNames: string[];
  /** Last scene name (for continuity in next extraction) */
  lastSceneName?: string;
}

// ============================
// Core function
// ============================

/**
 * Run the full L1 extraction pipeline on conversation messages.
 *
 * @param messages - Filtered conversation messages (from L0 or directly from hook)
 * @param sessionKey - The session key
 * @param baseDir - Base data directory (~/.openclaw/memory-tdai/)
 * @param config - OpenClaw config (for LLM access)
 * @param options - Extraction options
 * @param logger - Optional logger
 */
export async function extractL1Memories(params: {
  messages: ConversationMessage[];
  sessionKey: string;
  sessionId?: string;
  taskId?: string;
  teamId?: string;
  userId?: string;
  agentId?: string;
  baseDir: string;
  config: unknown;
  options?: {
    /** Max new messages to send in one extraction call */
    maxMessagesPerExtraction?: number;
    /** Max background messages for context */
    maxBackgroundMessages?: number;
    /** Enable conflict detection */
    enableDedup?: boolean;
    /** Max memories extracted per call */
    maxMemoriesPerSession?: number;
    /** LLM model override */
    model?: string;
    /** Previous scene name for continuity */
    previousSceneName?: string;
    /** Prompt family for L1 extraction (default: chat). */
    promptMode?: MemoryPromptMode;
    /** Vector store for cosine similarity candidate recall */
    vectorStore?: IMemoryStore;
    /** Embedding service for computing query vectors */
    embeddingService?: EmbeddingService;
    /** Top-K candidates for conflict recall (default: 5) */
    conflictRecallTopK?: number;
    /** Override embedding timeout for capture-path calls (milliseconds) */
    embeddingTimeoutMs?: number;
    /**
     * Host-neutral LLM runner. When provided, used instead of creating
     * a CleanContextRunner (decouples from OpenClaw runtime).
     */
    llmRunner?: LLMRunner;
  };
  logger?: Logger;
  /** Plugin instance ID for metric reporting (optional — metrics skipped if absent) */
  instanceId?: string;
  /**
   * StorageAdapter for L1 JSONL writes.
   * - service mode: must be provided (CosStorageBackend) — JSONL is the source of
   *   truth for backup/recovery; without storage, writes silently fall back to local
   *   pod fs and are lost on pod restart (CR-2 root cause, fixed 2026-05-19).
   * - standalone mode: caller usually provides LocalStorageBackend; if absent,
   *   writeMemory falls back to fs at `{baseDir}/records/{date}.jsonl`.
   */
  storage?: StorageAdapter;
}): Promise<L1ExtractionResult> {
  const { messages, sessionKey, sessionId, taskId, teamId, userId, agentId, baseDir, config, logger, instanceId: metricInstanceId, storage } = params;
  const options = params.options ?? {};
  const maxNewMessages = options.maxMessagesPerExtraction ?? 10;
  const maxBgMessages = options.maxBackgroundMessages ?? 5;
  const enableDedup = options.enableDedup ?? true;
  const maxMemoriesPerSession = options.maxMemoriesPerSession ?? 10;

  if (messages.length === 0) {
    logger?.debug?.(`${TAG} No messages to extract from`);
    return { success: true, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
  }

  const l1StartMs = Date.now();

  // Quality gate: filter messages through L1 extraction rules (length, symbols,
  // prompt injection, etc.) before sending to the LLM. L0 deliberately captures
  // everything; the strict filtering happens here at L1 stage.
  const qualifiedMessages = messages.filter((m) => shouldExtractL1(m.content));
  if (qualifiedMessages.length < messages.length) {
    logger?.debug?.(
      `${TAG} L1 quality filter: ${messages.length} → ${qualifiedMessages.length} messages ` +
      `(${messages.length - qualifiedMessages.length} filtered out)`,
    );
  }

  if (qualifiedMessages.length === 0) {
    logger?.debug?.(`${TAG} All messages filtered out by L1 quality gate`);
    return { success: true, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
  }

  // Split messages into background (older) + new (recent)
  const newMessages = qualifiedMessages.slice(-maxNewMessages);
  const bgEndIdx = qualifiedMessages.length - newMessages.length;
  const backgroundMessages = bgEndIdx > 0
    ? qualifiedMessages.slice(Math.max(0, bgEndIdx - maxBgMessages), bgEndIdx)
    : [];

  logger?.debug?.(`${TAG} Extracting from ${newMessages.length} new messages (+ ${backgroundMessages.length} background) [${qualifiedMessages.length} qualified from ${messages.length} input]`);

  // Step 1: LLM extraction (scene segmentation + memory extraction)
  let scenes: SceneSegment[];
  try {
    scenes = await callLlmExtraction({
      newMessages,
      backgroundMessages,
      previousSceneName: options.previousSceneName,
      config,
      logger,
      model: options.model,
      promptMode: options.promptMode,
      traceContext: { teamId, userId, agentId, sessionId },
      llmRunner: options.llmRunner,
    });
    logger?.debug?.(`${TAG} LLM detected ${scenes.length} scene(s)`);
  } catch (err) {
    logger?.error(`${TAG} LLM extraction failed: ${err instanceof Error ? err.message : String(err)}`);
    return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
  }

  // Flatten all memories across scenes
  const allExtracted: ExtractedMemory[] = [];
  const sceneNames: string[] = [];

  for (const scene of scenes) {
    sceneNames.push(scene.scene_name);
    for (const mem of scene.memories) {
      const memType = normalizeType(mem.type);
      if (!memType) {
        logger?.warn?.(`${TAG} Skipping memory with invalid type "${mem.type}"`);
        continue;
      }
      allExtracted.push({
        content: mem.content,
        type: memType,
        priority: typeof mem.priority === "number" ? mem.priority : 50,
        source_message_ids: Array.isArray(mem.source_message_ids) ? mem.source_message_ids : [],
        metadata: mem.metadata ?? {},
        scene_name: scene.scene_name,
      });
    }
  }

  logger?.debug?.(`${TAG} Total extracted memories: ${allExtracted.length} across ${scenes.length} scene(s)`);

  if (allExtracted.length === 0) {
    // ── 评测指标：L1 提取率（提取为空的情况） ──
    if (metricInstanceId) {
      try {
        const l0Count = messages.length;
        metricProducer.send({ metric: "l0_input_count", instanceId: metricInstanceId, value: l0Count, source: "core" });
        metricProducer.send({ metric: "l1_extracted_count", instanceId: metricInstanceId, value: 0, source: "core" });
        if (l0Count > 0) {
          metricProducer.send({ metric: "l1_extraction_rate", instanceId: metricInstanceId, value: 0, source: "core" });
        }
      } catch {
        // 静默忽略，不影响业务逻辑
      }
    }
    return {
      success: true,
      extractedCount: 0,
      storedCount: 0,
      records: [],
      sceneNames,
      lastSceneName: sceneNames[sceneNames.length - 1],
    };
  }

  // Limit per session
  let extracted = allExtracted;
  if (extracted.length > maxMemoriesPerSession) {
    logger?.debug?.(`${TAG} Limiting from ${extracted.length} to ${maxMemoriesPerSession} memories per session`);
    extracted = extracted.slice(0, maxMemoriesPerSession);
  }

  // Assign temporary IDs to extracted memories (needed for batch dedup)
  const memoriesWithIds = extracted.map((m) => ({
    ...m,
    record_id: generateMemoryId(),
  }));

  // Step 2: Batch Conflict Detection + Write
  let storedRecords: MemoryRecord[];
  let dedupLatencyMs: number | null = null;

  if (enableDedup) {
    try {
      const dedupStartMs = Date.now();
      const decisions = await batchDedup({
        memories: memoriesWithIds,
        config,
        logger,
        model: options.model,
        promptMode: options.promptMode,
        vectorStore: options.vectorStore,
        embeddingService: options.embeddingService,
        conflictRecallTopK: options.conflictRecallTopK,
        embeddingTimeoutMs: options.embeddingTimeoutMs,
        llmRunner: options.llmRunner,
        traceContext: { teamId, userId, agentId, sessionId },
        ...(teamId || userId || agentId || sessionId || taskId ? { filter: { teamId, userId, agentId, sessionId, taskId } } : {}),
      });
      dedupLatencyMs = Date.now() - dedupStartMs;

      // ── 评测指标：去重决策分布 ──
      if (metricInstanceId) {
        try {
          const dedupCounts = { store: 0, update: 0, merge: 0, skip: 0 };
          for (const d of decisions) {
            if (d.action in dedupCounts) {
              dedupCounts[d.action as keyof typeof dedupCounts]++;
            }
          }
          metricProducer.send({ metric: "l1_dedup_store_count", instanceId: metricInstanceId, value: dedupCounts.store, source: "core" });
          metricProducer.send({ metric: "l1_dedup_update_count", instanceId: metricInstanceId, value: dedupCounts.update, source: "core" });
          metricProducer.send({ metric: "l1_dedup_merge_count", instanceId: metricInstanceId, value: dedupCounts.merge, source: "core" });
          metricProducer.send({ metric: "l1_dedup_skip_count", instanceId: metricInstanceId, value: dedupCounts.skip, source: "core" });
        } catch {
          // 静默忽略，不影响业务逻辑
        }
      }

      storedRecords = await applyDecisions({
        memoriesWithIds,
        decisions,
        baseDir,
        sessionKey,
        sessionId,
        taskId,
        teamId,
        userId,
        agentId,
        logger,
        vectorStore: options.vectorStore,
        embeddingService: options.embeddingService,
        storage,
      });

    } catch (err) {
      logger?.warn?.(`${TAG} Batch dedup failed, storing all as new: ${err instanceof Error ? err.message : String(err)}`);
      storedRecords = await storeAllDirectly(memoriesWithIds, baseDir, sessionKey, sessionId, taskId, teamId, userId, agentId, logger, options.vectorStore, options.embeddingService, storage);
    }
  } else {
    storedRecords = await storeAllDirectly(memoriesWithIds, baseDir, sessionKey, sessionId, taskId, teamId, userId, agentId, logger, options.vectorStore, options.embeddingService, storage);
  }

  logger?.info(`${TAG} Extraction complete: extracted=${extracted.length}, stored=${storedRecords.length}`);

  // ── l1_extraction metric ──
  if (metricInstanceId && logger) {
    // Build type distribution of stored memories
    const memoriesByType: Record<string, number> = {};
    for (const r of storedRecords) {
      memoriesByType[r.type] = (memoriesByType[r.type] ?? 0) + 1;
    }
    report("l1_extraction", {
      sessionKey,
      inputMessageCount: messages.length,
      memoriesExtracted: extracted.length,
      memoriesStored: storedRecords.length,
      memoriesStoredContent: storedRecords.map((r) => ({
        content: r.content,
        type: r.type,
        scene: r.scene_name ?? null,
      })),
      memoriesByType,
      totalDurationMs: Date.now() - l1StartMs,
      success: true,
      error: null,
    });
  }

  // ── 评测指标：L1 提取率 ──
  if (metricInstanceId) {
    try {
      const l0Count = messages.length;
      const l1Count = extracted.length;
      metricProducer.send({ metric: "l0_input_count", instanceId: metricInstanceId, value: l0Count, source: "core" });
      metricProducer.send({ metric: "l1_extracted_count", instanceId: metricInstanceId, value: l1Count, source: "core" });
      if (l0Count > 0) {
        metricProducer.send({ metric: "l1_extraction_rate", instanceId: metricInstanceId, value: l1Count / l0Count, source: "core" });
      }
    } catch {
      // 静默忽略，不影响业务逻辑
    }
  }

  // ── 评测指标：L1 延迟 ──
  try {
    reportL1LatencyMetrics({
      instanceId: metricInstanceId ?? "",
      extractionLatencyMs: Date.now() - l1StartMs,
      dedupLatencyMs,
      hasError: false,
    });
  } catch {
    // 静默忽略
  }

  return {
    success: true,
    extractedCount: extracted.length,
    storedCount: storedRecords.length,
    records: storedRecords,
    sceneNames,
    lastSceneName: sceneNames[sceneNames.length - 1],
  };
}

// ============================
// LLM call
// ============================

/**
 * Call LLM to extract scene-segmented memories from conversation messages.
 */
async function callLlmExtraction(params: {
  newMessages: ConversationMessage[];
  backgroundMessages: ConversationMessage[];
  previousSceneName?: string;
  config: unknown;
  logger?: Logger;
  model?: string;
  promptMode?: MemoryPromptMode;
  /** Host-neutral LLM runner — when provided, used instead of CleanContextRunner. */
  llmRunner?: LLMRunner;
  /** langfuse 上报身份四元组（team/user/agent/session）。 */
  traceContext?: TraceContext;
}): Promise<SceneSegment[]> {
  const { newMessages, backgroundMessages, previousSceneName, config, logger, model, promptMode = "chat", llmRunner, traceContext } = params;

  const systemPrompt = getExtractMemoriesSystemPrompt(promptMode);
  const userPrompt = formatExtractionPrompt({
    newMessages,
    backgroundMessages,
    previousSceneName,
  });

  // [l1-debug] ENTRY — what are we about to ask the LLM to extract?
  logger?.debug?.(
    `${TAG} [l1-debug] ENTRY taskId=l1-extraction, promptMode=${promptMode}, newMsgs=${newMessages.length}, bgMsgs=${backgroundMessages.length}, userPromptLen=${userPrompt.length}, sysPromptLen=${systemPrompt.length}, model=${model ?? "(default)"}, previousSceneName=${previousSceneName ? JSON.stringify(previousSceneName) : "(none)"}, runnerKind=${llmRunner ? "llmRunner" : "CleanContextRunner"}`,
  );

  let result: string;

  // langfuse trace 语义：让此次 L1 抽取在 UI 有稳定 name / 顶级 user/session 列
  // / 可筛选 tags。避免所有记忆抽取都显示为 Unnamed trace。
  const traceParams = buildTraceParams("memory.l1-extract", traceContext);

  if (llmRunner) {
    // Use the host-neutral LLMRunner interface
    result = await llmRunner.run({
      prompt: userPrompt,
      systemPrompt,
      taskId: "l1-extraction",
      timeoutMs: 180_000,
      ...traceParams,
    });
  } else {
    // Fallback: create CleanContextRunner (OpenClaw path)
    const runner = new CleanContextRunner({
      config,
      modelRef: model,
      enableTools: false,
      logger,
    });

    result = await runner.run({
      prompt: userPrompt,
      systemPrompt,
      taskId: "l1-extraction",
      timeoutMs: 180_000,
      ...traceParams,
    });
  }

  return parseExtractionResult(result, logger);
}

/**
 * Parse the LLM's JSON response into SceneSegment array.
 * Expected format: [{scene_name, message_ids, memories: [...]}]
 */
function parseExtractionResult(raw: string, logger?: Logger): SceneSegment[] {
  try {
    // Strip markdown code block wrappers if present
    let cleaned = raw.trim();
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
    }

    // Try to extract JSON array
    const arrayMatch = cleaned.match(/\[[\s\S]*\]/);
    if (!arrayMatch) {
      logger?.warn?.(`${TAG} No JSON array found in extraction response`);
      // [l1-debug] NO_JSON — dump the full raw so we can see what the LLM actually said
      const rawPreview = raw.slice(0, 2048);
      logger?.warn?.(
        `${TAG} [l1-debug] NO_JSON taskId=l1-extraction, rawLen=${raw.length}, cleanedLen=${cleaned.length}, rawFull=${JSON.stringify(rawPreview)}${raw.length > 2048 ? `…(+${raw.length - 2048})` : ""}`,
      );
      return [];
    }

    // Sanitize control characters inside JSON string literals that LLM may produce.
    // Some weaker OpenAI-compatible models occasionally emit bare identifiers for
    // numeric fields (e.g. `"priority": sheet`). Repair only known safe fields and
    // retry once so one bad scalar does not drop the whole extraction result.
    const sanitized = sanitizeJsonForParse(arrayMatch[0]);
    let parsed: unknown[];
    try {
      parsed = JSON.parse(sanitized) as unknown[];
    } catch (err) {
      const repaired = repairExtractionJson(sanitized);
      if (repaired === sanitized) throw err;
      parsed = JSON.parse(repaired) as unknown[];
      logger?.warn?.(`${TAG} Repaired non-strict extraction JSON: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (!Array.isArray(parsed)) {
      logger?.warn?.(`${TAG} Extraction response is not an array`);
      return [];
    }

    const scenes: SceneSegment[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const s = item as Record<string, unknown>;

      scenes.push({
        scene_name: typeof s.scene_name === "string" ? s.scene_name : "未知情境",
        message_ids: Array.isArray(s.message_ids) ? s.message_ids.map(String) : [],
        memories: Array.isArray(s.memories)
          ? (s.memories as Array<Record<string, unknown>>)
              .filter((m) => m && typeof m === "object" && typeof m.content === "string" && (m.content as string).length > 0)
              .map((m) => ({
                content: String(m.content),
                type: String(m.type ?? "episodic"),
                priority: typeof m.priority === "number" ? m.priority : 50,
                source_message_ids: Array.isArray(m.source_message_ids) ? m.source_message_ids.map(String) : [],
                metadata: (m.metadata && typeof m.metadata === "object" ? m.metadata : {}) as Record<string, unknown>,
              }))
          : [],
      });
    }

    return scenes;
  } catch (err) {
    logger?.warn?.(`${TAG} Failed to parse extraction result: ${err instanceof Error ? err.message : String(err)}`);
    const rawPreview = raw.slice(0, 2048);
    logger?.warn?.(
      `${TAG} [l1-debug] PARSE_FAIL rawLen=${raw.length}, rawFull=${JSON.stringify(rawPreview)}${raw.length > 2048 ? `…(+${raw.length - 2048})` : ""}`,
    );
    return [];
  }
}

function repairExtractionJson(json: string): string {
  return json
    .replace(
      /("priority"\s*:\s*)(?!-?\d+(?:\.\d+)?\s*[,}]|"[^"\\]*(?:\\.[^"\\]*)*"\s*[,}])([\s\S]*?)(?=,\s*"(?:content|type|priority|source_message_ids|metadata)"\s*:|[}\]])/g,
      (_m, prefix: string) => `${prefix}50`,
    )
    .replace(/,\s*([}\]])/g, "$1");
}

// ============================
// Write helpers
// ============================

/**
 * Apply batch dedup decisions — write memories according to their decisions.
 */
async function applyDecisions(params: {
  memoriesWithIds: Array<ExtractedMemory & { record_id: string }>;
  decisions: DedupDecision[];
  baseDir: string;
  sessionKey: string;
  sessionId?: string;
  taskId?: string;
  teamId?: string;
  userId?: string;
  agentId?: string;
  logger?: Logger;
  vectorStore?: IMemoryStore;
  embeddingService?: EmbeddingService;
  storage?: StorageAdapter;
}): Promise<MemoryRecord[]> {
  const { memoriesWithIds, decisions, baseDir, sessionKey, sessionId, taskId, teamId, userId, agentId, logger, vectorStore, embeddingService, storage } = params;
  const storedRecords: MemoryRecord[] = [];

  // Build a map from record_id → decision
  const decisionMap = new Map<string, DedupDecision>();
  for (const d of decisions) {
    decisionMap.set(d.record_id, d);
  }

  for (const memoryWithId of memoriesWithIds) {
    const decision = decisionMap.get(memoryWithId.record_id) ?? {
      record_id: memoryWithId.record_id,
      action: "store" as const,
      target_ids: [],
    };

    try {
      const record = await writeMemory({
        memory: memoryWithId,
        decision,
        baseDir,
        sessionKey,
        sessionId,
        taskId,
        teamId,
        userId,
        agentId,
        logger,
        vectorStore,
        embeddingService,
        storage,
      });

      if (record) {
        storedRecords.push(record);
      }
    } catch (err) {
      logger?.warn?.(
        `${TAG} Write failed for memory "${memoryWithId.content.slice(0, 50)}...": ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  return storedRecords;
}

/**
 * Store all memories directly (no dedup).
 */
async function storeAllDirectly(
  memoriesWithIds: Array<ExtractedMemory & { record_id: string }>,
  baseDir: string,
  sessionKey: string,
  sessionId: string | undefined,
  taskId: string | undefined,
  teamId?: string,
  userId?: string,
  agentId?: string,
  logger?: Logger,
  vectorStore?: IMemoryStore,
  embeddingService?: EmbeddingService,
  storage?: StorageAdapter,
): Promise<MemoryRecord[]> {
  const storedRecords: MemoryRecord[] = [];

  for (const memoryWithId of memoriesWithIds) {
    try {
      const record = await writeMemory({
        memory: memoryWithId,
        decision: {
          record_id: memoryWithId.record_id,
          action: "store",
          target_ids: [],
        },
        baseDir,
        sessionKey,
        sessionId,
        taskId,
        teamId,
        userId,
        agentId,
        logger,
        vectorStore,
        embeddingService,
        storage,
      });
      if (record) {
        storedRecords.push(record);
      }
    } catch (err) {
      logger?.warn?.(
        `${TAG} Write failed for memory "${memoryWithId.content.slice(0, 50)}...": ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  return storedRecords;
}

// ============================
// Helpers
// ============================

const VALID_TYPES: MemoryType[] = ["persona", "episodic", "instruction", "work_fact", "work_task", "work_method", "work_artifact"];

function normalizeType(raw: string): MemoryType | null {
  const lower = raw.toLowerCase().trim();
  if (VALID_TYPES.includes(lower as MemoryType)) {
    return lower as MemoryType;
  }
  // Handle legacy type names
  if (lower === "episode") return "episodic";
  if (lower === "instruct") return "instruction";
  if (lower === "preference") return "persona"; // fold preference into persona
  return null;
}
