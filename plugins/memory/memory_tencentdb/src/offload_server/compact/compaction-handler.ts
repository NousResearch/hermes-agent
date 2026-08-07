/**
 * L3 Compaction Handler — orchestrates fast-path, MMD injection, and compression.
 * Synchronous API: client awaits the response.
 *
 * Uses compact-state.json (independent from state.json) to avoid lock contention
 * with L1/L1.5/L2 executors that write state.json concurrently.
 */
import type http from "node:http";
import type { StorageAdapter } from "../../core/storage/adapter.js";
import type { OffloadEntry, OffloadState, OffloadExecutorConfig, CompactState } from "../types.js";
import { defaultOffloadState, defaultCompactState } from "../types.js";
import { parseJsonl } from "../parsers/json-utils.js";
import { CompactionRequestSchemaV2 } from "../schemas.js";
import { buildOffloadBasePath } from "../session-utils.js";
import { applyFastPath } from "./fast-path.js";
import { injectActiveMmd, injectHistoryMmds } from "./mmd-injector.js";
import { resolveLevel, mildCompress, aggressiveCompress, emergencyCompress } from "./compressor.js";
import { estimateMessageTokens, extractToolResultId } from "./helpers.js";
import type { Message } from "./helpers.js";
import { traceServerCompaction } from "../opik-tracer.js";

export interface CompactionDeps {
  storage: StorageAdapter;
  config: OffloadExecutorConfig;
  logger: { info: (...args: unknown[]) => void; warn: (...args: unknown[]) => void; error: (...args: unknown[]) => void };
}

export interface CompactionReport {
  resolvedLevel: string;
  originalCount: number;
  compactedCount: number;
  fastPathReplaced: number;
  fastPathDeleted: number;
  mildReplacements: number;
  aggressiveDeleted: number;
  emergencyDeleted: number;
  mmdInjected: number;
}

/**
 * Handle POST /v2/offload/compact.
 */
export async function handleCompaction(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  auth: { serviceId: string },
  deps: CompactionDeps,
  requestId: string,
  parseJsonBody: <T>(req: http.IncomingMessage) => Promise<T>,
  sendJson: (res: http.ServerResponse, status: number, body: unknown) => void,
  successEnvelope: <T>(data: T, requestId: string) => unknown,
  errorEnvelope: (code: number, message: string, requestId: string) => unknown,
): Promise<void> {
  const body = await parseJsonBody(req);
  const parsed = CompactionRequestSchemaV2.safeParse(body);
  if (!parsed.success) {
    sendJson(res, 400, errorEnvelope(400, parsed.error.message, requestId));
    return;
  }

  const { session_id: sessionId, messages, ratio, context_window: contextWindow, message_tokens: messageTokens } = parsed.data;
  let { total_tokens: totalTokens } = parsed.data;
  const { storage, config } = deps;
  const basePath = buildOffloadBasePath(sessionId);
  const originalCount = messages.length;
  const compactionStartMs = Date.now();

  const { preciseMessageTokens } = await import("./compressor.js");
  const originalTotalTokens = totalTokens;

  // Compute fixed overhead: system prompt + tool schemas + message framing.
  // These are included in clientTotalTokens but NOT in messages.
  // overhead = clientTotal - sum(preciseMessageTokens for each message)
  const messagesTokenSum = (messages as any[]).reduce(
    (s: number, msg: any) => s + preciseMessageTokens(msg), 0,
  );
  const fixedOverhead = Math.max(0, totalTokens - messagesTokenSum);

  deps.logger.info(
    `[offload-server] compaction: clientTotal=${totalTokens}, msgsTokens=${messagesTokenSum}, ` +
    `overhead=${fixedOverhead}, msgs=${originalCount}, ratio=${ratio.toFixed(2)}`,
  );

  // Read offload state (L1/L1.5/L2 managed, read-only here) and compact state (L3 owned)
  const state = await readOffloadState(storage, basePath);
  const compactState = await readCompactState(storage, basePath);
  const entriesRaw = await storage.readFile(`${basePath}/entries.jsonl`);
  const entries = entriesRaw
    ? parseJsonl<OffloadEntry>(entriesRaw, (line, err) => {
        deps.logger.warn(`[offload-server] compaction: bad JSONL line: ${line}`, err);
      })
    : [];

  // Merge node-mapping.jsonl into entries (L2 writes node_id to a separate file)
  const nodeMappingRaw = await storage.readFile(`${basePath}/node-mapping.jsonl`);
  if (nodeMappingRaw) {
    const mappings = parseJsonl<{ tool_call_id: string; node_id: string }>(nodeMappingRaw);
    const nodeMap = new Map(mappings.map((m) => [m.tool_call_id, m.node_id]));
    for (const entry of entries) {
      if (!entry.node_id && nodeMap.has(entry.tool_call_id)) {
        entry.node_id = nodeMap.get(entry.tool_call_id)!;
      }
    }
  }

  // Token array: use tiktoken values directly (no calibration against clientTotal).
  // clientTotal includes fixedOverhead which is not in messages, so calibration
  // would inflate per-message tokens incorrectly.
  const tokenArray = buildTokenArray(messages as Message[], messagesTokenSum, messageTokens);

  const report: CompactionReport = {
    resolvedLevel: "fastpath",
    originalCount,
    compactedCount: 0,
    fastPathReplaced: 0,
    fastPathDeleted: 0,
    mildReplacements: 0,
    aggressiveDeleted: 0,
    emergencyDeleted: 0,
    mmdInjected: 0,
  };

  // Step 1: Fast-path re-apply (uses compactState for confirmed/deleted IDs)
  const fp = applyFastPath(messages, entries, compactState);
  report.fastPathReplaced = fp.replacedCount;
  report.fastPathDeleted = fp.deletedCount;

  // Recalculate totalTokens after fast-path:
  // totalTokens = tiktoken(remaining messages) + fixedOverhead
  if (fp.deletedCount > 0) {
    const postFpMsgsTokens = (messages as any[]).reduce(
      (s: number, msg: any) => s + preciseMessageTokens(msg), 0,
    );
    totalTokens = postFpMsgsTokens + fixedOverhead;
    // Rebuild tokenArray for remaining messages
    tokenArray.length = 0;
    tokenArray.push(...buildTokenArray(messages as Message[], postFpMsgsTokens, undefined));
  }
  const effectiveRatio = contextWindow > 0 ? totalTokens / contextWindow : ratio;

  // Step 2: Resolve compression level using post-fast-path ratio
  const level = resolveLevel(effectiveRatio, {
    mildRatio: config.mildOffloadRatio,
    aggressiveRatio: config.aggressiveCompressRatio,
    emergencyRatio: config.emergencyCompressRatio,
  });
  report.resolvedLevel = level;

  deps.logger.info(
    `[offload-server] compaction: level=${level}, ratio=${effectiveRatio.toFixed(2)} (pre-fp=${ratio.toFixed(2)}), msgs=${messages.length}, entries=${entries.length}`,
  );

  // Step 3: Mild compression
  if (level === "mild" || level === "aggressive" || level === "emergency") {
    const mild = mildCompress(messages, entries);
    report.mildReplacements = mild.replacedCount;
    compactState.confirmedOffloadIds.push(...mild.confirmedIds);

    // Sync tokenArray after mild replacements (content changed)
    // Incremental update: only recalculate tokens for messages that were actually replaced
    if (mild.replacedCount > 0) {
      const confirmedSet = new Set(mild.confirmedIds);
      let tokenDelta = 0;
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i] as any;
        // Use extractToolResultId which handles all formats (OpenAI, Anthropic content blocks)
        const tid = extractToolResultId(msg);
        if (!tid || !confirmedSet.has(tid)) continue;
        const newTokens = preciseMessageTokens(msg);
        tokenDelta += newTokens - tokenArray[i];
        tokenArray[i] = newTokens;
      }
      totalTokens += tokenDelta;
    }
  }

  // Step 4: Aggressive compression
  // Target: just below the aggressive trigger threshold (leave ~5% headroom)
  let aggRemainingTokens = totalTokens; // track for emergency
  if (level === "aggressive" || level === "emergency") {
    const aggTargetTokens = Math.floor(contextWindow * (config.aggressiveCompressRatio - 0.05));
    // Skip aggressive if mild already brought tokens below target
    if (totalTokens <= aggTargetTokens) {
      deps.logger.info(
        `[offload-server] aggressive skipped: mild already reduced tokens to ${totalTokens} (target=${aggTargetTokens})`,
      );
    } else {
      const agg = aggressiveCompress(messages, aggTargetTokens, tokenArray, totalTokens);
      report.aggressiveDeleted = agg.deletedCount;
      aggRemainingTokens = agg.remainingTokens;
      compactState.deletedOffloadIds.push(...agg.deletedIds);

      // Inject history MMDs for deleted entries
      if (agg.deletedIds.length > 0) {
        const mmdBudget = Math.floor(contextWindow * 0.1 / 4); // 10% of context, in estimated tokens
        const hist = await injectHistoryMmds(
          messages, agg.deletedIds, entries, state, storage, basePath, mmdBudget,
        );
        report.mmdInjected += hist.injectedCount;
        // Sync tokenArray for any injected MMD messages (calibrate against messages-only tokens)
        if (hist.injectedCount > 0) {
          const postAggMsgsTokens = (messages as any[]).reduce(
            (s: number, msg: any) => s + preciseMessageTokens(msg), 0,
          );
          tokenArray.length = 0;
          tokenArray.push(...buildTokenArray(messages as Message[], postAggMsgsTokens, undefined));
        }
      }
    }
  }

  // Step 6: Emergency compression
  // Target: just below the aggressive threshold (so next turn won't immediately re-trigger)
  if (level === "emergency") {
    const emTargetTokens = Math.floor(contextWindow * (config.aggressiveCompressRatio - 0.10));
    const em = emergencyCompress(messages, emTargetTokens, tokenArray, aggRemainingTokens);
    report.emergencyDeleted = em.deletedCount;
    aggRemainingTokens = em.remainingTokens;
    compactState.deletedOffloadIds.push(...em.deletedIds);
  }

  // Step 7: Inject active MMD (after all compression, so position is correct)
  const mmdResult = await injectActiveMmd(messages, state, storage, basePath);
  report.mmdInjected += mmdResult.injectedCount;

  // Step 8: Write compact-state.json (independent file, no lock needed)
  report.compactedCount = messages.length;
  compactState.lastCompactedAt = new Date().toISOString();
  await writeCompactState(storage, basePath, compactState);

  // Compute remaining tokens (tracked from aggressive/emergency, no full re-scan)
  const remainingTokens = level === "fastpath" || level === "mild"
    ? totalTokens  // no deletion happened
    : aggRemainingTokens;  // tracked through aggressive → emergency chain
  const remainingRatio = contextWindow > 0 ? (remainingTokens / contextWindow).toFixed(2) : "N/A";

  // Opik trace: compaction decision
  traceServerCompaction({
    sessionId,
    level,
    ratio,
    contextWindow,
    totalTokensBefore: originalTotalTokens,
    totalTokensAfter: remainingTokens,
    originalMsgCount: originalCount,
    compactedMsgCount: messages.length,
    report: report as unknown as Record<string, unknown>,
    messages: messages as unknown[],
    durationMs: Date.now() - compactionStartMs,
    logger: deps.logger,
  });

  // Step 8: Return
  sendJson(res, 200, successEnvelope({ messages, report }, requestId));

  deps.logger.info(
    `[offload-server] compaction done: ${originalCount}→${messages.length} msgs, level=${level}, ` +
    `tokens=${originalTotalTokens}→${remainingTokens} (${remainingRatio}), ` +
    `fp=${fp.replacedCount}r/${fp.deletedCount}d, mild=${report.mildReplacements}, ` +
    `agg=${report.aggressiveDeleted}, em=${report.emergencyDeleted}, mmd=${report.mmdInjected}`,
  );
}

// ─── Token Array Builder ─────────────────────────────────────────────────────

/** Calibration threshold: if heuristic estimate drifts >15% from API totalTokens, apply linear scaling. */
const CALIBRATION_THRESHOLD = 0.15;
/** Clamp calibration factor to prevent extreme scaling from noisy estimates. */
const CALIBRATION_FACTOR_MIN = 0.5;
const CALIBRATION_FACTOR_MAX = 3.0;

/**
 * Build a pre-computed token array for all messages, with optional linear calibration.
 * - If messageTokens[i] is available, use it directly (precise value).
 * - Otherwise, use estimateMessageTokens (CJK-aware heuristic).
 * - If totalTokens is provided and drift > 15%, apply a calibration factor to estimated items.
 */
export function buildTokenArray(
  messages: Message[],
  totalTokens: number,
  messageTokens?: number[],
): number[] {
  const raw = messages.map((msg, i) =>
    messageTokens && i < messageTokens.length
      ? messageTokens[i]
      : estimateMessageTokens(msg),
  );

  const rawTotal = raw.reduce((s, v) => s + v, 0);
  if (rawTotal <= 0 || totalTokens <= 0) return raw;

  const drift = Math.abs(totalTokens - rawTotal) / totalTokens;
  if (drift <= CALIBRATION_THRESHOLD) return raw;

  // Linear calibration: only scale estimated items, keep precise items unchanged
  const factor = Math.max(CALIBRATION_FACTOR_MIN, Math.min(CALIBRATION_FACTOR_MAX, totalTokens / rawTotal));
  return raw.map((v, i) =>
    messageTokens && i < messageTokens.length
      ? v
      : Math.max(1, Math.round(v * factor)),
  );
}

// ─── State Helpers ───────────────────────────────────────────────────────────

async function readOffloadState(storage: StorageAdapter, basePath: string): Promise<OffloadState> {
  const raw = await storage.readFile(`${basePath}/state.json`);
  if (!raw) return defaultOffloadState();
  try {
    return { ...defaultOffloadState(), ...JSON.parse(raw) };
  } catch {
    return defaultOffloadState();
  }
}

async function readCompactState(storage: StorageAdapter, basePath: string): Promise<CompactState> {
  const raw = await storage.readFile(`${basePath}/compact-state.json`);
  if (!raw) return defaultCompactState();
  try {
    return { ...defaultCompactState(), ...JSON.parse(raw) };
  } catch {
    return defaultCompactState();
  }
}

async function writeCompactState(storage: StorageAdapter, basePath: string, state: CompactState): Promise<void> {
  await storage.writeFile(`${basePath}/compact-state.json`, JSON.stringify(state));
}
