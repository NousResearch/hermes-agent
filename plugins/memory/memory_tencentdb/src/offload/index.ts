/**
 * Context Offload Module Entry
 *
 * Exports `registerOffload(api, offloadConfig)` for conditional registration
 * from the main plugin index.ts.
 *
 * This module is the merged equivalent of the standalone context-offload-plugin's index.js,
 * adapted to co-exist with the memory-tencentdb plugin.
 */
import { OffloadStateManager } from "./state-manager.js";
import { createAfterToolCallHandler } from "./hooks/after-tool-call.js";
import { createBeforePromptBuildHandler } from "./hooks/before-prompt-build.js";
import { shouldForceL1 } from "./hooks/llm-output.js";
import { handleTaskTransition, normalizeJudgment } from "./hooks/before-agent-start.js";
import { checkL2Trigger, backfillNodeIds } from "./pipelines/l2-mermaid.js";
import { PLUGIN_DEFAULTS } from "./types.js";
import { initOffloadOpikTracer } from "./opik-tracer.js";
import {
  readAllOffloadEntries,
  readOffloadEntries,
  markOffloadStatus,
  DEFAULT_DATA_ROOT,
} from "./storage.js";
import { buildTiktokenContextSnapshot, configureTokenTracker, tiktokenCount, jsonReplacer } from "./context-token-tracker.js";
import { fastEstimateMessages } from "./fast-token-estimate.js";
import {
  normalizeToolCallIdForLookup,
  getOffloadEntry,
  populateOffloadLookupMap,
  isToolResultMessage,
  extractToolCallId,
  isOnlyToolUseAssistant,
  extractAllToolUseIds,
  isAssistantMessageWithToolUse,
  replaceWithSummary,
  replaceAssistantToolUseWithSummary,
  compressNonCurrentToolUseBlocks,
  getCurrentTaskNodeIds,
} from "./l3-helpers.js";
import { createL3TokenCounter } from "./l3-token-counter.js";
import {
  compressByScoreCascade,
  aggressiveCompressUntilBelowThreshold,
  buildHistoryMmdInjection,
  removeExistingMmdInjections,
  emergencyCompress,
  EMERGENCY_MIN_MESSAGES_TO_KEEP,
  isTokenOverflowError,
} from "./hooks/llm-input-l3.js";
import { findHistoryMmdInsertionPoint } from "./mmd-injector.js";
import type { OffloadConfig } from "../config.js";
import type { PluginConfig, PluginLogger } from "./types.js";
import { BackendClient } from "./backend-client.js";
import { LocalLlmClient } from "./local-llm/index.js";
import type { L1Request, L15Request, L2Request } from "./backend-client.js";
import { parseMmdMeta } from "./mmd-meta.js";
import { sanitizeText, writeRefMd } from "./storage.js";
import { listMmds, readMmd, writeMmd, patchMmd } from "./storage.js";
import {
  appendOffloadEntries,
  rewriteAllOffloadEntries,
} from "./storage.js";
import { nowChinaISO } from "./time-utils.js";
import { traceOffloadDecision, traceMessagesSnapshot } from "./opik-tracer.js";
import { SessionRegistry } from "./session-registry.js";
import { reclaimOffloadData } from "./reclaimer.js";
import { buildL3TriggerReport, reportL3Trigger } from "./state-reporter.js";
import { resolveUserId, getUserIdSource } from "./user-id.js";

// ─── Module-level state ──────────────────────────────────────────────────────
// OpenClaw calls registerOffload() multiple times during lifecycle.
// L2 scheduler and L1.5 dispose flag are shared across invocations.
// L2 scheduler state — shared across registerOffload() calls
let _l2Running = false;
let _l2PollHandle: ReturnType<typeof setTimeout> | null = null;
let _l2FirstNotifyAt: number | null = null;

// L1.5 retry loop dispose flag
let _l15Disposed = false;

// Reclaim scheduler timer — module-level so dispose() can clear it
let _reclaimTimer: ReturnType<typeof setTimeout> | null = null;

// Context Engine singleton — survives across registerOffload() calls.
let _sharedEngine: OffloadContextEngine | null = null;
let _contextEngineRegistered = false;
/** Set to true when registerContextEngine returns ok=false or throws — all offload functions disabled. */
let _contextEngineRejected = false;

// SessionRegistry singleton — MUST be shared between engine and hooks.
// OpenClaw calls register() N times; hooks from different calls may coexist.
// If each call creates a new SessionRegistry, the same sessionKey resolves
// to different manager instances in engine vs hooks, breaking L1.5→L2 state.
let _sharedSessions: SessionRegistry | null = null;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function parseCreateSkillCommand(
  prompt: string,
): { mmdName: string | null; skillFocus: string | null } | null {
  if (typeof prompt !== "string") return null;
  const trimmed = prompt.trim();
  const match = trimmed.match(/^\/create-skill(?:\s+(.*))?$/i);
  if (!match) return null;
  const args = (match[1] || "").trim();
  if (!args) return { mmdName: null, skillFocus: null };
  const parts = args.split(/\s+/);
  const mmdName = parts[0] || null;
  const skillFocus = parts.slice(1).join(" ") || null;
  return { mmdName, skillFocus };
}

function simpleHash(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return hash;
}

/** Compute a fingerprint for a message (role + first 200 chars of content). */
function _msgFingerprint(msg: any): number {
  const role = msg.role ?? msg.message?.role ?? msg.type ?? "";
  let content = "";
  const raw = msg.type === "message" ? msg.message?.content : msg.content;
  if (typeof raw === "string") content = raw.slice(0, 200);
  else if (Array.isArray(raw)) content = JSON.stringify(raw).slice(0, 200);
  return simpleHash(`${role}:${content}`);
}


function _extractLatestTurn(_messages: any[], currentPrompt: string | null): string | null {
  const effectivePrompt = _isHeartbeatText(currentPrompt ?? "") ? null : currentPrompt;
  if (!effectivePrompt) return null;
  return `[User]: ${String(effectivePrompt).slice(0, 500)}`;
}

function _extractMsgText(msg: any): string {
  const content = msg.content ?? msg.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.filter((c: any) => c.type === "text" && typeof c.text === "string").map((c: any) => c.text).join(" ");
  return "";
}

function _normalizePromptForCompare(text: string | null): string {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

/**
 * Check if a message text looks like a heartbeat probe.
 * Matches both user heartbeat prompts and assistant HEARTBEAT_OK replies.
 */
function _isHeartbeatText(text: string): boolean {
  return text.includes("HEARTBEAT") || text.includes("heartbeat");
}

/**
 * Extract recent history messages for L1/L2 context, organized as
 * user-assistant pairs: each user message followed by up to
 * `maxAssistantPerUser` assistant replies from that turn.
 *
 * Output format:
 *   [User]: xxx
 *   [Assistant]: aaa
 *   [User]: yyy
 *   [Assistant]: bbb
 *   [Assistant]: ccc
 *
 * Scans messages in forward order, skipping MMD injections, heartbeat
 * probes, and the current prompt (to avoid duplication).
 */
function _extractRecentHistory(messages: any[], currentPrompt: string | null = null, maxAssistantPerUser = 3): string | null {
  const normalizedCurrent = _normalizePromptForCompare(currentPrompt);

  // Collect turns: each turn = { user: string, assistants: string[] }
  const turns: Array<{ user: string; assistants: string[] }> = [];
  let currentTurn: { user: string; assistants: string[] } | null = null;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg._mmdContextMessage || msg._mmdInjection) continue;
    const role = msg.role ?? msg.message?.role ?? msg.type;

    if (role === "user") {
      let text = _extractMsgText(msg);
      if (!text || text.length <= 5) continue;
      // Skip heartbeat probes
      if (_isHeartbeatText(text)) { currentTurn = null; continue; }
      text = text.slice(0, 400);
      // Skip current prompt (already in "current msg" section)
      if (normalizedCurrent) {
        const normalizedText = _normalizePromptForCompare(text);
        if (normalizedText === normalizedCurrent || normalizedText.startsWith(normalizedCurrent) || normalizedCurrent.startsWith(normalizedText)) continue;
      }
      // Start a new turn
      currentTurn = { user: text, assistants: [] };
      turns.push(currentTurn);
    } else if (role === "assistant" && currentTurn) {
      if (currentTurn.assistants.length >= maxAssistantPerUser) continue;
      const directText = _extractMsgText(msg);
      if (!directText || directText.length <= 10) continue;
      // Skip heartbeat replies (e.g. "HEARTBEAT_OK")
      if (_isHeartbeatText(directText)) continue;
      currentTurn.assistants.push(directText.slice(0, 400));
    }
  }

  // Keep only the most recent turns (limit total to avoid oversized context)
  const maxTurns = 5;
  const recentTurns = turns.slice(-maxTurns);

  const parts: string[] = [];
  for (const turn of recentTurns) {
    parts.push(`[User]: ${turn.user}`);
    for (const a of turn.assistants) {
      parts.push(`[Assistant]: ${a}`);
    }
  }

  return parts.length > 0 ? parts.join("\n") : null;
}

function _buildL1RecentContext(stateManager: OffloadStateManager): string {
  // Skip heartbeat prompts in current msg
  const rawPrompt = stateManager.cachedUserPrompt;
  const isHeartbeat = typeof rawPrompt === "string" && _isHeartbeatText(rawPrompt);
  const currentLine = (!isHeartbeat && typeof rawPrompt === "string" && rawPrompt.trim())
    ? `[User]: ${rawPrompt.slice(0, 500)}`
    : (stateManager.cachedLatestTurnMessages || "(none)");
  const historyBlock = stateManager.cachedRecentHistory || "(none)";
  return `## current msg:\n${currentLine}\n\n## history msg:\n${historyBlock}`;
}

/** L1.5-specific format: history as reference first, latest user message as focus last. */
function _buildL15RecentContext(stateManager: OffloadStateManager): string {
  const rawPrompt = stateManager.cachedUserPrompt;
  const isHeartbeat = typeof rawPrompt === "string" && _isHeartbeatText(rawPrompt);
  const currentLine = (!isHeartbeat && typeof rawPrompt === "string" && rawPrompt.trim())
    ? `[User]: ${rawPrompt.slice(0, 500)}`
    : (stateManager.cachedLatestTurnMessages || "(none)");
  const historyBlock = stateManager.cachedRecentHistory || "(none)";
  return `历史消息，可作为参考：\n${historyBlock}\n\n最新user message：\n${currentLine}`;
}

/**
 * Register the offload module with OpenClaw plugin API.
 * Called from main index.ts when offload.enabled = true.
 *
 * NOTE: No idempotency guard here. OpenClaw calls register() multiple
 * times during its lifecycle (plugin scan → gateway start → config reload).
 * Each call provides a different `api` instance; only the last one is the
 * live runtime api. Hooks registered on earlier api instances are discarded.
 * registerContextEngine and api.on/registerHook are safe to call repeatedly.
 */

/**
 * Detect internal memory-pipeline sessions that should NOT run offload.
 * Actual format from framework: `agent:main:explicit:memory-{taskId}-session-{ts}`
 * Raw format from clean-context-runner: `memory-{taskId}-session-{ts}`
 */
const INTERNAL_SESSION_RE = /memory-.*-session-\d+/;

function isInternalMemorySession(sessionKey: string | null | undefined): boolean {
  return typeof sessionKey === "string" && INTERNAL_SESSION_RE.test(sessionKey);
}

export function registerOffload(api: any, offloadConfig: OffloadConfig): void {
  const logger: PluginLogger = api.logger;

  // ── Diagnostic: detect whether api.on / api.registerHook is functional ──
  const regMode = api.registrationMode ?? "(not exposed)";
  const hasRegisterHook = typeof api.registerHook === "function";
  const hasOn = typeof api.on === "function";
  const hasRegisterContextEngine = typeof api.registerContextEngine === "function";
  const onFnName = api.on?.name ?? "(unnamed)";
  const onFnBody = String(api.on).slice(0, 200);
  logger.debug?.(
    `[context-offload] [DIAG] registrationMode=${regMode}, ` +
    `registerHook=${hasRegisterHook}, api.on=${hasOn} name="${onFnName}", ` +
    `registerContextEngine=${hasRegisterContextEngine}, ` +
    `api.on body=${onFnBody}`,
  );

  logger.debug?.("[context-offload] Registering offload module...");
  initOffloadOpikTracer(api.config, logger);

  // Build plugin config from OffloadConfig
  const pCfg: Partial<PluginConfig> = {
    model: offloadConfig.model,
    temperature: offloadConfig.temperature,
    forceTriggerThreshold: offloadConfig.forceTriggerThreshold,
    dataDir: offloadConfig.dataDir,
    defaultContextWindow: offloadConfig.defaultContextWindow,
    maxPairsPerBatch: offloadConfig.maxPairsPerBatch,
    l2NullThreshold: offloadConfig.l2NullThreshold,
    l2TimeoutSeconds: offloadConfig.l2TimeoutSeconds,
    mildOffloadRatio: offloadConfig.mildOffloadRatio,
    aggressiveCompressRatio: offloadConfig.aggressiveCompressRatio,
    mmdMaxTokenRatio: offloadConfig.mmdMaxTokenRatio,
  };

  // Fix 4: Configure token tracker encoding to match plugin config (default: o200k_base)
  const _encoding = pCfg.l3TiktokenEncoding ?? PLUGIN_DEFAULTS.l3TiktokenEncoding;
  configureTokenTracker(pCfg.l3TiktokenEncoding);
  logger.debug?.(`[context-offload] Token tracker encoding: ${_encoding} (configured from ${pCfg.l3TiktokenEncoding ? "pluginConfig" : "default"})`);

  // Session Registry — module-level singleton so engine + hooks always share the same instance
  const dataRoot = offloadConfig.dataDir ?? DEFAULT_DATA_ROOT;
  if (!_sharedSessions) {
    _sharedSessions = new SessionRegistry(dataRoot);
  }
  const sessions = _sharedSessions;

  // Resolve LLM Configuration — mode-based selection
  // - "backend": use remote backend service (requires backendUrl)
  // - "local": call LLM directly via AI SDK (uses offload.model or main agent model)
  //
  // User identity: prefer offloadConfig.userId; fall back to the host's
  // primary non-loopback IPv4 address.
  const _resolvedUserId = resolveUserId(offloadConfig.userId ?? null);
  logger.debug?.(
    `[context-offload] user-id resolved: "${_resolvedUserId}" (source=${getUserIdSource() ?? "?"})`,
  );

  let backendClient: BackendClient | LocalLlmClient | null = null;

  if (offloadConfig.mode === "backend" || offloadConfig.mode === "collect") {
    // Remote backend mode (or collect mode with backend)
    if (!offloadConfig.backendUrl) {
      logger.error(`[context-offload] mode=${offloadConfig.mode} but backendUrl not configured. L1/L1.5/L2/L4 disabled.`);
    } else {
      backendClient = new BackendClient(
        offloadConfig.backendUrl,
        logger,
        offloadConfig.backendApiKey,
        offloadConfig.backendTimeoutMs,
        () => _lastActiveSessionKey,
        () => _resolvedUserId,
        () => { try { return _lastActiveMgr?.getLastSessionKey?.() ?? _lastActiveSessionKey; } catch { return _lastActiveSessionKey; } },
      );
    }
  } else {
    // Local LLM mode — resolve model from offload.model or fall back to agents.defaults.model
    let resolvedModelRef = offloadConfig.model;
    if (!resolvedModelRef) {
      // Fallback: use main agent model from openclaw.json agents.defaults.model
      const mainConfig = api.config as Record<string, unknown> | undefined;
      const agents = mainConfig?.agents as Record<string, unknown> | undefined;
      const defaults = agents?.defaults as Record<string, unknown> | undefined;
      const modelCfg = defaults?.model;
      if (typeof modelCfg === "string" && modelCfg.includes("/")) {
        resolvedModelRef = modelCfg;
      } else if (modelCfg && typeof modelCfg === "object") {
        const primary = (modelCfg as Record<string, unknown>).primary;
        if (typeof primary === "string" && primary.includes("/")) {
          resolvedModelRef = primary;
        }
      }
      if (resolvedModelRef) {
        logger.debug?.(`[context-offload] offload.model not set, using main agent model: ${resolvedModelRef}`);
      }
    }

    if (resolvedModelRef) {
      const modelParts = resolvedModelRef.split("/", 2);
      const providerKey = modelParts[0];
      const modelId = modelParts[1] ?? resolvedModelRef;
      const models = (api.config as any)?.models;
      const providerCfg = models?.providers?.[providerKey];
      const baseUrl = providerCfg?.baseUrl ?? providerCfg?.baseURL;
      const apiKey = providerCfg?.apiKey;

      if (baseUrl && apiKey) {
        backendClient = new LocalLlmClient(
          { baseUrl, apiKey, model: modelId, temperature: offloadConfig.temperature, timeoutMs: offloadConfig.backendTimeoutMs },
          logger,
        );
      } else {
        logger.error(
          `[context-offload] Local LLM mode failed: provider "${providerKey}" not found or missing baseUrl/apiKey in models.providers. ` +
          `L1/L1.5/L2 disabled.`,
        );
      }
    } else {
      logger.warn("[context-offload] No model resolved (offload.model not set, agents.defaults.model not found). L1/L1.5/L2 disabled.");
    }
  }

  // Track last active session key for BackendClient header
  let _lastActiveSessionKey: string | null = null;

  if (backendClient && (offloadConfig.mode === "backend" || offloadConfig.mode === "collect")) {
    logger.debug?.(`[context-offload] LLM mode: backend (${offloadConfig.backendUrl})`);
  } else if (backendClient) {
    logger.debug?.(`[context-offload] LLM mode: local (${offloadConfig.model ?? "main-agent-model"})`);
  } else {
    logger.warn("[context-offload] LLM client not available. L1/L1.5/L2/L4 disabled (L3 compression still active).");
  }

  // ─── Fault tolerance constants ──────────────────────────────────────────────
  const MAX_L1_CHUNK_RETRIES = 3;
  const L1_BATCH_SIZE = 5; // matches backend toolPairs limit (1-5)
  const L2_BATCH_SIZE = 30; // max entries per L2 backend call to avoid oversized requests / timeouts

  // ─── Backend-aware L1 flush helper (with batching + retry + fallback) ──────
  // Backend mode only: take pairs → filter → split into batches → per-batch HTTP
  // → on failure: retry up to MAX_L1_CHUNK_RETRIES → then generate local fallback entries.
  const flushL1 = async (stateManager: OffloadStateManager, triggerSource: string, fireAndForget = false, maxCount?: number): Promise<void> => {
    if (!backendClient) return;
    if (!stateManager.hasPending()) return;

    const release = await stateManager.acquireL1Lock();
    try {
      // Take and filter pairs
      const pendingCount = stateManager.getPendingCount();
      const takeCount = maxCount != null ? Math.min(maxCount, pendingCount) : pendingCount;
      let takenPairs = stateManager.takePending(takeCount);
      if (takenPairs.length === 0) return;

      // Filter heartbeat pairs
      const isHeartbeat = (p: typeof takenPairs[0]) => {
        try {
          const raw = typeof p.params === "string" ? p.params : JSON.stringify(p.params ?? "");
          return raw.includes("HEARTBEAT.md");
        } catch { return false; }
      };
      const beforeFilter = takenPairs.length;
      const pairs = takenPairs.filter((p) => !isHeartbeat(p));
      if (beforeFilter > pairs.length) {
        logger.debug?.(`[context-offload] L1: filtered ${beforeFilter - pairs.length} heartbeat pair(s)`);
      }
      if (pairs.length === 0) return;

      // L1.1: Write ref MD files locally (preserves raw tool results for L3 recovery)
      const refByToolCallId = new Map<string, string>();
      for (const p of pairs) {
        try {
          const resultStr = typeof p.result === "string"
            ? sanitizeText(p.result)
            : sanitizeText(JSON.stringify(p.result, null, 2));
          const content = `**Tool:** ${p.toolName}\n**Call ID:** ${p.toolCallId}\n\n**Result:**\n\`\`\`\n${resultStr}\n\`\`\``;
          const refPath = await writeRefMd(stateManager.ctx, p.timestamp, p.toolName, content);
          refByToolCallId.set(p.toolCallId, refPath);
        } catch (err) {
          logger.error(`[context-offload] L1.1 ref write error (${p.toolCallId}): ${err}`);
        }
      }

      // Split into batches of L1_BATCH_SIZE
      const batches: typeof pairs[] = [];
      for (let i = 0; i < pairs.length; i += L1_BATCH_SIZE) {
        batches.push(pairs.slice(i, i + L1_BATCH_SIZE));
      }
      logger.debug?.(`[context-offload] L1 (${triggerSource}): ${pairs.length} pairs → ${batches.length} batch(es) of ≤${L1_BATCH_SIZE}`);

      const recentMessages = _buildL1RecentContext(stateManager);
      logger.debug?.(`[context-offload] L1 recentMessages (${recentMessages.length} chars):\n${recentMessages}`);

      for (const chunk of batches) {
        const chunkKey = chunk[0].toolCallId; // track by first toolCallId
        const prevFails = stateManager._l1ChunkFailCounts.get(chunkKey) ?? 0;

        try {
          const req: L1Request = {
            recentMessages,
            toolPairs: chunk.map((p) => ({
              toolName: p.toolName,
              toolCallId: p.toolCallId,
              params: typeof p.params === "string" ? sanitizeText(p.params) : p.params,
              result: typeof p.result === "string" ? sanitizeText(p.result as string) : p.result,
              timestamp: p.timestamp,
            })),
          };
          const resp = await backendClient.l1Summarize(req);

          // Success — reset fail count, write entries
          stateManager._l1ChunkFailCounts.delete(chunkKey);
          if (resp.entries && resp.entries.length > 0) {
            for (const entry of resp.entries) {
              if (!entry.result_ref && refByToolCallId.has(entry.tool_call_id)) {
                entry.result_ref = refByToolCallId.get(entry.tool_call_id)!;
              }
            }
            await appendOffloadEntries(stateManager.ctx, resp.entries, undefined, logger);
            stateManager.entryCounter += resp.entries.length;
            logger.debug?.(`[context-offload] L1 batch OK: ${resp.entries.length} entries from ${chunk.length} pairs (entryCounter=${stateManager.entryCounter})`);
          }
        } catch (err) {
          const newFails = prevFails + 1;
          logger.warn(`[context-offload] L1 batch FAILED (${chunkKey}, attempt ${newFails}/${MAX_L1_CHUNK_RETRIES}): ${err}`);

          if (newFails >= MAX_L1_CHUNK_RETRIES) {
            // Exceeded retry limit — generate local fallback entries (no LLM summary)
            logger.warn(`[context-offload] L1 batch DEGRADED: ${chunk.length} pairs → fallback entries (no LLM summary)`);
            stateManager._l1ChunkFailCounts.delete(chunkKey);
            const fallbackEntries: import("./types.js").OffloadEntry[] = [];
            for (const p of chunk) {
              const resultStr = typeof p.result === "string" ? p.result : JSON.stringify(p.result ?? "");
              const truncResult = resultStr.length > 300 ? resultStr.slice(0, 297) + "..." : resultStr;
              const truncParams = typeof p.params === "string"
                ? (p.params.length > 200 ? p.params.slice(0, 197) + "..." : p.params)
                : JSON.stringify(p.params ?? "").slice(0, 200);
              fallbackEntries.push({
                timestamp: p.timestamp,
                node_id: null,
                tool_call: `${p.toolName}(${truncParams})`,
                summary: `[L1 degraded] ${p.toolName}: ${truncResult}`,
                result_ref: refByToolCallId.get(p.toolCallId) ?? "",
                tool_call_id: p.toolCallId,
                score: 0,
              });
            }
            await appendOffloadEntries(stateManager.ctx, fallbackEntries, undefined, logger);
            stateManager.entryCounter += fallbackEntries.length;
            logger.debug?.(`[context-offload] L1 fallback: wrote ${fallbackEntries.length} degraded entries`);
          } else {
            // Under retry limit — re-enqueue this chunk for next flush
            stateManager._l1ChunkFailCounts.set(chunkKey, newFails);
            for (const p of chunk) {
              stateManager.processedToolCallIds.delete(p.toolCallId);
              stateManager.pendingToolPairs.push(p as any);
            }
            logger.debug?.(`[context-offload] L1 batch: re-enqueued ${chunk.length} pairs (retry ${newFails}/${MAX_L1_CHUNK_RETRIES})`);
          }
        }
      }
    } finally {
      release();
    }
  };

  // ─── Backend-aware L1.5 judge helper (1 retry, fail-safe) ──────────────────
  // L1.5 determines task boundary. On failure (after 1 retry):
  //   - activeMmd cleared to null → L2 won't trigger
  //   - All null entries marked as "short" → won't pollute future L2
  //   - This turn has no MMD construction
  _l15Disposed = false; // Reset on re-registration
  const L15_RETRY_DELAY_MS = 3000;

  /** L1.5 fail-safe: push a short boundary instead of marking entries on disk. */
  const _l15FailSafe = async (stateManager: OffloadStateManager, startIndex: number) => {
    stateManager.setActiveMmd(null, null);
    stateManager.pushBoundary({ startIndex, result: "short", targetMmd: null });
    await stateManager.save();
    stateManager.setMmdInjectionReady(false);
    stateManager.l15Settled = true;
    logger.warn(`[context-offload] L1.5 fail-safe: settled (boundary short @${startIndex}, activeMmd=null)`);
  };

  const attemptL15 = async (stateManager: OffloadStateManager, startIndex: number): Promise<boolean> => {
    try {
      // Build request
      const allMmds = await listMmds(stateManager.ctx);
      const availableMmds = allMmds.slice(-10);
      const { join } = await import("node:path");
      const mmdMetas: L15Request["availableMmdMetas"] = [];
      for (const mmdFile of availableMmds) {
        try {
          const content = await readMmd(stateManager.ctx, mmdFile);
          if (content) {
            mmdMetas.push(parseMmdMeta(mmdFile, join(stateManager.ctx.mmdsDir, mmdFile), content));
          }
        } catch { /* skip */ }
      }
      const currentMmdFilename = stateManager.getActiveMmdFile();
      let currentMmd: L15Request["currentMmd"] = null;
      if (currentMmdFilename) {
        const content = await readMmd(stateManager.ctx, currentMmdFilename);
        if (content) {
          currentMmd = { filename: currentMmdFilename, content, path: join(stateManager.ctx.mmdsDir, currentMmdFilename) };
        }
      }
      const recentMessages = _buildL15RecentContext(stateManager);

      stateManager.setMmdInjectionReady(false);
      const resp = await backendClient!.l15Judge({ recentMessages, currentMmd, availableMmdMetas: mmdMetas });

      // Normalize backend response (handles null fields from fallback)
      const judgment = normalizeJudgment(resp as unknown as Record<string, unknown>);
      if (!judgment) {
        logger.warn("[context-offload] L1.5: all-null response (backend LLM unavailable)");
        return false; // trigger retry
      }

      // Success
      logger.debug?.(
        `[context-offload] L1.5: completed=${judgment.taskCompleted}, continuation=${judgment.isContinuation}, longTask=${judgment.isLongTask}, label=${judgment.newTaskLabel ?? "none"}, contFile=${judgment.continuationMmdFile ?? "none"}`,
      );

      // ── Flush residual null entries for the OLD mmd before task transition ──
      // When the user switches tasks, the old mmd may have < l2NullThreshold
      // null entries that would never reach the threshold trigger. We detect
      // the mmd change and fire a forced L2 for the old mmd's remaining entries
      // so they are not orphaned or mis-attributed to the new mmd.
      const prevMmdFile = currentMmdFilename; // captured before handleTaskTransition

      // Apply task transition
      await handleTaskTransition(stateManager, judgment, logger);

      const newMmdFile = stateManager.getActiveMmdFile();
      const mmdSwitched = prevMmdFile && newMmdFile !== prevMmdFile;
      if (mmdSwitched) {
        // Fire-and-forget: flush residual null entries for the OLD mmd.
        // Only include entries whose index < startIndex (they belong to the
        // previous boundary, not the new one being pushed below).
        const _flushStartIndex = startIndex;
        const _flushPrevMmd = prevMmdFile!;
        (async () => {
          try {
            const allEntries = await readAllOffloadEntries(stateManager.ctx);
            const residualEntries: typeof allEntries = [];
            for (let idx = 0; idx < allEntries.length && idx < _flushStartIndex; idx++) {
              const e = allEntries[idx];
              if ((e.node_id === null || e.node_id === "wait") && !(e.tool_call ?? "").includes("HEARTBEAT.md")) {
                residualEntries.push(e);
              }
            }
            if (residualEntries.length === 0) return;

            // Build a synthetic entriesByMmd for the old mmd only
            const residualByMmd = new Map<string, typeof residualEntries>();
            residualByMmd.set(_flushPrevMmd, residualEntries);

            logger.debug?.(
              `[context-offload] L1.5 task-switch flush: ${residualEntries.length} residual null entries (idx<${_flushStartIndex}) for old mmd=${_flushPrevMmd}, triggering forced L2`,
            );
            await runL2WithBackend(stateManager, residualByMmd, "task_switch_flush");
          } catch (flushErr) {
            logger.warn(`[context-offload] L1.5 task-switch flush failed: ${flushErr}`);
          }
        })().catch(() => {});
      }

      // Push boundary based on L1.5 result
      const activeMmdFile = stateManager.getActiveMmdFile();
      if (activeMmdFile) {
        stateManager.pushBoundary({ startIndex, result: "long", targetMmd: activeMmdFile });
        logger.debug?.(`[context-offload] L1.5 boundary: long @${startIndex} → ${activeMmdFile}`);
      } else {
        stateManager.pushBoundary({ startIndex, result: "short", targetMmd: null });
        logger.debug?.(`[context-offload] L1.5 boundary: short @${startIndex}`);
      }

      await stateManager.save();
      stateManager.setMmdInjectionReady(true);
      stateManager.l15Settled = true;
      logger.debug?.("[context-offload] L1.5: settled, MMD injection ready");
      return true;
    } catch (err) {
      logger.warn(`[context-offload] L1.5 attempt failed: ${err}`);
      return false;
    }
  };

  const judgeL15 = async (stateManager: OffloadStateManager, event: any, ctx: any): Promise<void> => {
    if (!backendClient) return;
    stateManager.l15Settled = false;

    // Flush only the pairs that existed BEFORE this user message
    const snapshotCount = stateManager.getPendingCount();
    if (snapshotCount > 0) {
      try {
        await flushL1(stateManager, "l15_pre_flush", false, snapshotCount);
      } catch (err) {
        logger.warn(`[context-offload] L1.5 pre-flush failed: ${err}`);
      }
    }

    // Record the dividing line: entries after this index belong to this turn
    const startIndex = stateManager.entryCounter;
    logger.debug?.(`[context-offload] L1.5 boundary startIndex=${startIndex} (pending flushed=${snapshotCount})`);

    // First attempt
    if (await attemptL15(stateManager, startIndex)) return;

    // Single retry after delay (fire-and-forget)
    const retry = async () => {
      await new Promise((r) => setTimeout(r, L15_RETRY_DELAY_MS));
      if (_l15Disposed || stateManager.l15Settled) return;
      logger.debug?.("[context-offload] L1.5 retrying... (1/1)");
      if (await attemptL15(stateManager, startIndex)) return;
      // Both attempts failed — activate fail-safe
      logger.warn("[context-offload] L1.5 FAILED after 1 retry, activating fail-safe");
      await _l15FailSafe(stateManager, startIndex);
    };
    retry().catch(() => {});
  };

  // ─── Backend-aware L2 trigger helper ───────────────────────────────────────
  const runL2WithBackend = async (stateManager: OffloadStateManager, entriesByMmd: Map<string, any[]>, triggerSource: string): Promise<void> => {
    if (!backendClient) return;
    try {
      for (const [mmdFile, mmdEntries] of entriesByMmd) {
        const taskLabel = mmdFile.replace(/^\d+-/, "").replace(/\.mmd$/, "") || "unnamed-task";
        const prefixMatch = mmdFile.match(/^(\d+)-/);
        const mmdPrefix = prefixMatch ? prefixMatch[1] : "000";

        // Split entries into batches to avoid oversized requests
        const batches: any[][] = [];
        for (let i = 0; i < mmdEntries.length; i += L2_BATCH_SIZE) {
          batches.push(mmdEntries.slice(i, i + L2_BATCH_SIZE));
        }
        logger.debug?.(`[context-offload] L2 (${triggerSource}): mmd=${mmdFile}, ${mmdEntries.length} entries → ${batches.length} batch(es) of ≤${L2_BATCH_SIZE}`);

        for (let bIdx = 0; bIdx < batches.length; bIdx++) {
          const batch = batches[bIdx];
          const batchWaitIds = new Set(batch.map((e: any) => e.tool_call_id as string));

          // Read fresh MMD for each batch (previous batch may have updated it)
          const existingMmd = await readMmd(stateManager.ctx, mmdFile);

          const req: L2Request = {
            existingMmd,
            newEntries: batch.map((e: any) => ({
              tool_call_id: e.tool_call_id,
              tool_call: e.tool_call,
              summary: e.summary,
              timestamp: e.timestamp,
            })),
            recentHistory: stateManager.cachedRecentHistory || null,
            currentTurn: stateManager.cachedLatestTurnMessages || null,
            taskLabel,
            mmdPrefix,
            mmdCharCount: existingMmd ? existingMmd.length : 0,
          };

          // Mark batch entries as "wait" before calling backend
          const allEntries = await readAllOffloadEntries(stateManager.ctx);
          let changed = false;
          for (const entry of allEntries) {
            if (batchWaitIds.has(entry.tool_call_id) && entry.node_id === null) {
              entry.node_id = "wait";
              changed = true;
            }
          }
          if (changed) await rewriteAllOffloadEntries(stateManager.ctx, allEntries);
          if (bIdx === 0) {
            stateManager.setLastL2TriggerTime(nowChinaISO());
            await stateManager.save();
          }

          try {
            const resp = await backendClient.l2Generate(req);

            // Handle backend degraded response (empty fileAction = LLM unavailable)
            if (!resp.fileAction) {
              logger.warn(`[context-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length}: degraded response, applying fallback backfill`);
              await backfillNodeIds(stateManager.ctx, resp.nodeMapping ?? {}, batchWaitIds, logger, {
                mmdFallbackText: existingMmd ?? "",
                mmdPrefix,
              });
              continue;
            }

            // Apply MMD file changes
            if (resp.fileAction === "replace" && resp.replaceBlocks && resp.replaceBlocks.length > 0) {
              const patchOk = await patchMmd(stateManager.ctx, mmdFile, resp.replaceBlocks);
              logger.debug?.(`[context-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length}: patchMmd: ${patchOk ? "ok" : "FAILED"} (${resp.replaceBlocks.length} blocks)`);
              if (!patchOk && resp.mmdContent) {
                await writeMmd(stateManager.ctx, mmdFile, resp.mmdContent);
                logger.debug?.(`[context-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length}: fallback writeMmd: ${resp.mmdContent.length} chars`);
              }
            } else if (resp.mmdContent) {
              await writeMmd(stateManager.ctx, mmdFile, resp.mmdContent);
              logger.debug?.(`[context-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length}: writeMmd: ${resp.mmdContent.length} chars`);
            }

            // Backfill node_ids
            const mmdAfterWrite = await readMmd(stateManager.ctx, mmdFile);
            const mmdForBackfill =
              typeof mmdAfterWrite === "string" && mmdAfterWrite.trim().length > 0
                ? mmdAfterWrite
                : typeof existingMmd === "string" && existingMmd.trim().length > 0
                  ? existingMmd
                  : "";
            await backfillNodeIds(stateManager.ctx, resp.nodeMapping ?? {}, batchWaitIds, logger, {
              mmdFallbackText: mmdForBackfill,
              mmdPrefix,
            });

            logger.debug?.(`[context-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length} (${triggerSource}): applied, action=${resp.fileAction}, mapping=${Object.keys(resp.nodeMapping ?? {}).length}`);
          } catch (err) {
            logger.error(`[context-offload] L2 [${mmdFile}] batch ${bIdx + 1}/${batches.length} failed: ${err}`);
            // Continue with remaining batches — failed entries stay as "wait" for retry
          }
        }
      }
    } catch (err) {
      logger.error(`[context-offload] L2 failed: ${err}`);
    }
  };

  // ─── Backend-aware L4 skill helper ─────────────────────────────────────────
  const createSkillWithBackend = async (
    stateManager: OffloadStateManager,
    skillCommand: { mmdName: string | null; skillFocus: string | null },
  ): Promise<any> => {
    if (!backendClient || !skillCommand.mmdName) return null;
    try {
      // Read MMD + offload entries locally, send to backend
      const allMmds = await listMmds(stateManager.ctx);
      const mmdFilename = allMmds.find((f) => f.includes(skillCommand.mmdName!)) ?? null;
      if (mmdFilename) {
        const mmdContent = await readMmd(stateManager.ctx, mmdFilename);
        if (mmdContent) {
          const allEntries = await readAllOffloadEntries(stateManager.ctx);
          const nodeIdPattern = /\b(\d{3}-N\d+)\b/g;
          const nodeIds = new Set<string>();
          let match: RegExpExecArray | null;
          while ((match = nodeIdPattern.exec(mmdContent)) !== null) {
            nodeIds.add(match[1]);
          }
          const filtered = allEntries.filter((e) => e.node_id && nodeIds.has(e.node_id));
          const resp = await (backendClient as any).l4Generate({
            mmdFilename,
            mmdContent,
            offloadEntries: filtered,
            skillFocus: skillCommand.skillFocus,
          });
          if (!resp) return null;
          // Write skill file locally
          const { mkdir, writeFile } = await import("node:fs/promises");
          const { join } = await import("node:path");
          // NOTE: stateManager.ctx.dataDir 已经是 <dataRoot>/<agentName>（见 storage.ts createStorageContext），
          // 因此 skill 的 owner 隔离已经天然由 dataDir 提供，**不要**在路径里再拼一层 agentName，
          // 否则会得到 <dataRoot>/<agentName>/skills/<agentName>/<skillName>/SKILL.md（agent 重复）。
          // 与 core/skill 模块统一的目录契约：<dataDir>/skills/<skillName>/SKILL.md
          const skillsDir = join(stateManager.ctx.dataDir, "skills", resp.skillName);
          await mkdir(skillsDir, { recursive: true });
          await writeFile(join(skillsDir, "SKILL.md"), resp.skillContent, "utf-8");
          const resultPrompt = `<l4_skill_result>\n【Skill 生成完成】\n\n**Skill 名称:** ${resp.skillName}\n**描述:** ${resp.skillDescription}\n**文件路径:** ${join(skillsDir, "SKILL.md")}\n\n---\n${resp.skillContent}\n---\n</l4_skill_result>`;
          return { appendSystemContext: resultPrompt, phase: "completed", skillName: resp.skillName };
        }
      }
    } catch (err) {
      logger.error(`[context-offload] Backend L4 failed: ${err}`);
    }
    return null;
  };

  // Resolve context window — prioritize model's actual contextWindow from openclaw.json
  const getContextWindow = (): number => {
    try {
      const config = api.config;
      const agents = config?.agents;
      const defaults = agents?.defaults;
      const defaultModel = typeof defaults?.model === "string"
        ? defaults.model
        : (typeof defaults?.model === "object" && typeof (defaults?.model as any)?.primary === "string")
          ? (defaults.model as any).primary
          : null;
      const models = config?.models;
      // 1. If we know the model, find its exact contextWindow from providers
      if (defaultModel && models) {
        const [providerKey, modelId] = defaultModel.split("/", 2);
        const provider = models.providers?.[providerKey];
        if (provider?.models) {
          const modelList = Array.isArray(provider.models) ? provider.models : [];
          for (const m of modelList) {
            if (m.id === modelId && typeof m.contextWindow === "number") return m.contextWindow;
          }
        }
      }
      // 2. Fallback: top-level models.contextWindow
      if (models?.contextWindow && typeof models.contextWindow === "number") return models.contextWindow;
      // NOTE: fallback 3 (scan all providers) was removed — it could return
      // contextWindow from an unrelated model (e.g. 262144 from Claude-3.5
      // when the active model is GPT with 200000).
    } catch { /* ignore */ }
    // 3. Plugin config fallback
    if (typeof pCfg.defaultContextWindow === "number" && pCfg.defaultContextWindow > 0) {
      return pCfg.defaultContextWindow;
    }
    return PLUGIN_DEFAULTS.defaultContextWindow;
  };

  // Track last active manager for L2 scheduler (L2 is global, needs any session's ctx to read agent-shared files)
  let _lastActiveMgr: OffloadStateManager | null = null;

  /** Helper: resolve session manager and update last-active tracking */
  const _resolveSession = async (sessionKey: string, sessionId?: string): Promise<OffloadStateManager | null> => {
    if (!sessionKey) return null;
    const entry = await sessions.resolveIfAllowed(sessionKey, sessionId);
    if (!entry) return null;
    _lastActiveMgr = entry.manager;
    _lastActiveSessionKey = sessionKey;
    return entry.manager;
  };

  // L2 Scheduler — uses module-level state (_l2Running, _l2PollHandle, _l2FirstNotifyAt)
  // Clean up any lingering poll timer from previous registerOffload() call
  if (_l2PollHandle !== null) { clearTimeout(_l2PollHandle); _l2PollHandle = null; }
  _l2FirstNotifyAt = null;
  _l2Running = false;

  const l2TimeoutMs = (pCfg.l2TimeoutSeconds ?? PLUGIN_DEFAULTS.l2TimeoutSeconds) * 1000;
  const l2Threshold = pCfg.l2NullThreshold ?? PLUGIN_DEFAULTS.l2NullThreshold;

  const clearL2Poll = () => {
    if (_l2PollHandle !== null) { clearTimeout(_l2PollHandle); _l2PollHandle = null; }
    _l2FirstNotifyAt = null;
  };

  const armL2Poll = () => {
    if (_l2PollHandle !== null) return;
    if (_l2FirstNotifyAt === null) _l2FirstNotifyAt = Date.now();
    const tick = async () => {
      _l2PollHandle = null;
      const mgr = _lastActiveMgr;
      if (!mgr) return;
      // Gate: L2 must wait for L1.5 to settle (task boundary determined)
      // Timeout: if L1.5 hasn't settled after 60s (e.g. no Context Engine / assemble not called),
      // force-settle to unblock L2.
      if (!mgr.l15Settled) {
        const l15WaitAge = _l2FirstNotifyAt ? Date.now() - _l2FirstNotifyAt : 0;
        if (l15WaitAge > 60_000) {
          mgr.l15Settled = true;
          logger.warn("[context-offload] L2 poll: L1.5 settle timeout (60s), force-settling to unblock L2");
        } else {
          logger.debug?.("[context-offload] L2 poll: waiting for L1.5 to settle, deferring...");
          scheduleNextTick();
          return;
        }
      }
      try {
        const allEntries = await readAllOffloadEntries(mgr.ctx);
        const nullCount = allEntries.filter((e) => e.node_id === null).length;
        if (nullCount === 0) { _l2FirstNotifyAt = null; return; }
        if (_l2Running) { scheduleNextTick(); return; }
        const age = Date.now() - (_l2FirstNotifyAt ?? Date.now());
        if (nullCount >= l2Threshold) {
          _l2FirstNotifyAt = null;
          tryTriggerL2("null_threshold").catch(() => {});
        } else if (age >= l2TimeoutMs) {
          _l2FirstNotifyAt = null;
          tryTriggerL2("timer").catch(() => {});
        } else {
          scheduleNextTick();
        }
      } catch {
        scheduleNextTick();
      }
    };
    const scheduleNextTick = () => {
      if (_l2PollHandle !== null) return;
      _l2PollHandle = setTimeout(tick, 5000);
      if (_l2PollHandle && typeof _l2PollHandle === "object" && "unref" in _l2PollHandle) {
        (_l2PollHandle as any).unref();
      }
    };
    _l2PollHandle = setTimeout(tick, 0);
    if (_l2PollHandle && typeof _l2PollHandle === "object" && "unref" in _l2PollHandle) {
      (_l2PollHandle as any).unref();
    }
  };

  const notifyL2NewNullEntries = (newNullCount: number) => {
    if (!_lastActiveMgr || newNullCount <= 0) return;
    armL2Poll();
  };

  const tryTriggerL2 = async (triggerSource = "unknown") => {
    if (_l2Running) return;
    const mgr = _lastActiveMgr;
    if (!mgr) return;
    // Set _l2Running BEFORE any await to prevent concurrent triggers
    _l2Running = true;
    try {
      const { shouldTrigger, reason, entriesByMmd } = await checkL2Trigger(mgr, pCfg, logger);
      if (!shouldTrigger) return;
      const totalEntries = Array.from(entriesByMmd.values()).reduce((s, a) => s + a.length, 0);
      logger.debug?.(`[context-offload] L2 triggered (${triggerSource}): ${reason}, ${totalEntries} entries across ${entriesByMmd.size} mmd(s)`);
      await runL2WithBackend(mgr, entriesByMmd, triggerSource);
    } catch (err) {
      logger.error(`[context-offload] L2 trigger error: ${err}`);
    } finally {
      _l2Running = false;
      try {
        const postEntries = await readAllOffloadEntries(mgr.ctx);
        const postNullCount = postEntries.filter((e) => e.node_id === null).length;
        if (postNullCount >= l2Threshold) {
          clearL2Poll();
          tryTriggerL2("post_completion").catch(() => {});
        } else if (postNullCount > 0) {
          clearL2Poll();
          armL2Poll();
        } else {
          clearL2Poll();
        }
      } catch {
        armL2Poll();
      }
    }
  };

  // ─── Register Hooks ────────────────────────────────────────────────────────
  //
  // api.on() in OpenClaw 4.1 is a direct wrapper around registerTypedHook():
  //   (hookName, handler, opts) => registerTypedHook(record, hookName, handler, opts, hookPolicy)
  //
  // NOTE: api.registerHook() is a different API that requires a `name` field
  // on the handler — do NOT use it here (causes "hook registration missing name").
  //
  const _hookNames: string[] = [];
  const _trackedOn = (hookName: string, handler: (...args: any[]) => any) => {
    _hookNames.push(hookName);
    if (typeof api.on === "function") {
      api.on(hookName, (...args: any[]) => {
        if (_contextEngineRejected) return; // slot not acquired — all offload disabled
        return handler(...args);
      });
    } else {
      logger.error(`[context-offload] api.on not available for hook "${hookName}"! Hook will not fire.`);
    }
  };

  // before_tool_call
  _trackedOn("before_tool_call", async (event: any, ctx: any) => {
    const sk = ctx?.sessionKey;
    if (!sk) return;
    const mgr = await _resolveSession(sk, ctx?.sessionId);
    if (!mgr) return;
    const toolCallId = event.toolCallId ?? ctx.toolCallId;
    if (toolCallId && event.params != null) {
      mgr.cacheToolParams(toolCallId, event.params);
    }
  });

  // after_tool_call
  _trackedOn("after_tool_call", async (event: any, ctx: any) => {
    const _atcStart = Date.now();
    const _toolName = event.toolName ?? "unknown";
    const _toolCallId = event.toolCallId ?? "N/A";
    logger.debug?.(`[context-offload] >>> after_tool_call START: tool=${_toolName} id=${_toolCallId}`);
    try {
      const sk = ctx?.sessionKey;
      const _mgr = sk ? await _resolveSession(sk, ctx?.sessionId) : _lastActiveMgr;
      if (!_mgr) {
        logger.debug?.(`[context-offload] <<< after_tool_call SKIP: no session manager (${Date.now() - _atcStart}ms)`);
        return;
      }
      const afterToolCallHandler = createAfterToolCallHandler(_mgr, logger, getContextWindow, pCfg, backendClient as any);
      await afterToolCallHandler(event, ctx);
      const _handlerDone = Date.now();
      logger.debug?.(`[context-offload] after_tool_call handler done: ${_handlerDone - _atcStart}ms`);

      const pending = _mgr.getPendingCount();
      const threshold = pCfg.forceTriggerThreshold ?? 4;
      if (shouldForceL1(_mgr, pCfg)) {
        logger.debug?.(`[context-offload] L1 TRIGGERED: pending=${pending} >= threshold=${threshold}, flushing...`);
        flushL1(_mgr, "force_threshold", true).then(async () => {
          try {
            const allEntries = await readAllOffloadEntries(_mgr.ctx);
            const nullCount = allEntries.filter((e) => e.node_id === null).length;
            notifyL2NewNullEntries(nullCount);
          } catch { /* ignore */ }
        }).catch(() => {});
      } else {
        logger.debug?.(`[context-offload] L1 pending: ${pending}/${threshold} (not yet)`);
      }
      logger.debug?.(`[context-offload] <<< after_tool_call END: tool=${_toolName} total=${Date.now() - _atcStart}ms`);
    } catch (err) {
      logger.error(`[context-offload] <<< after_tool_call ERROR: tool=${_toolName} ${err} (${Date.now() - _atcStart}ms)`);
    }
  });

  // llm_output — simplified for backend mode (just logs pending count)
  _trackedOn("llm_output", async (event: any, ctx: any) => {
    const sk = ctx?.sessionKey;
    const mgr = sk ? sessions.get(sk)?.manager : _lastActiveMgr;
    if (!mgr) return;
    const pendingCount = mgr.getPendingCount();
    if (pendingCount > 0) {
      logger.debug?.(
        `[context-offload] llm_output: ${pendingCount} pending tool pairs (will be flushed at next llm_input or after_tool_call batch)`,
      );
    }
  });

  // llm_input (token cache + L2 context cache only — L1.5 is triggered exclusively from assemble)
  _trackedOn("llm_input", async (event: any, _ctx: any) => {
    const _llmInputStart = Date.now();
    if (isInternalMemorySession(_ctx?.sessionKey)) return;
    logger.debug?.(`[context-offload] >>> llm_input START`);
    const _sk = _ctx?.sessionKey;
    const _mgr = _sk ? await _resolveSession(_sk, _ctx?.sessionId) : _lastActiveMgr;
    if (!_mgr) return;
    try {
      const historyMessages = Array.isArray(event.historyMessages) ? event.historyMessages : [];
      const sysPrompt = typeof event.systemPrompt === "string" ? event.systemPrompt : null;
      const promptText = typeof event.prompt === "string" ? event.prompt : null;
      _mgr.cachedSystemPrompt = sysPrompt;
      _mgr.cachedUserPrompt = promptText;

      const snap = buildTiktokenContextSnapshot("llm_input", historyMessages, sysPrompt, promptText);
      _mgr.cachedSystemPromptTokens = snap.systemTokens;
      _mgr.cachedUserPromptTokens = snap.userPromptTokens;
      if (snap.systemTokens > 0) {
        _mgr.setEstimatedSystemOverhead(snap.systemTokens);
        if (_mgr.isLoaded()) _mgr.save().catch(() => {});
      }

      if (historyMessages.length > 0) {
        _mgr.cachedLatestTurnMessages = _extractLatestTurn(historyMessages, promptText);
        _mgr.cachedRecentHistory = _extractRecentHistory(historyMessages, promptText);
      }

      logger.debug?.(`[context-offload] <<< llm_input END: ${Date.now() - _llmInputStart}ms`);
    } catch (err) {
      logger.error(`[context-offload] <<< llm_input ERROR: ${err} (${Date.now() - _llmInputStart}ms)`);
    }
  });

  // before_agent_start (L4 + session fallback)
  const l4State = { pendingResult: null as any };
  _trackedOn("before_agent_start", async (event: any, ctx: any) => {
    if (isInternalMemorySession(ctx?.sessionKey)) return;
    const sk = ctx?.sessionKey;
    const mgr = sk ? await _resolveSession(sk, ctx?.sessionId) : null;
    if (!mgr) return;
    const userPrompt = event.prompt ?? "";
    const skillCommand = parseCreateSkillCommand(userPrompt);
    if (skillCommand) {
      try {
        const result = await createSkillWithBackend(mgr, skillCommand);
        if (result?.appendSystemContext) l4State.pendingResult = result;
      } catch { /* ignore */ }
    }
  });

  // before_prompt_build — primary hook for Responses API (gateway HTTP mode).
  //
  // OpenClaw's Responses API (/v1/responses) does NOT invoke the Context Engine
  // lifecycle (bootstrap → assemble → afterTurn). Only the pi-embedded-runner
  // (CLI/terminal mode) calls context engine methods.
  //
  // This hook provides the SAME functionality as OffloadContextEngine.assemble():
  //   1. L1.5 task judgment (fire-and-forget)
  //   2. L1 flush (fire-and-forget)
  //   3. Fast-path re-apply (confirmed/deleted offload replacements)
  //   4. L3 compression (aggressive/mild/emergency)
  //   5. MMD injection
  //
  // When assemble() IS called (CLI mode), it sets a per-turn flag so this hook
  // skips redundant work.
  // In "collect" mode: only L1 flush, skip L3 compression and MMD injection.
  _trackedOn("before_prompt_build", async (event: any, ctx: any) => {
    if (isInternalMemorySession(ctx?.sessionKey)) return;
    const sk = ctx?.sessionKey;
    const mgr = sk ? await _resolveSession(sk, ctx?.sessionId) : _lastActiveMgr;
    if (!mgr) return;

    // L1 flush (fire-and-forget)
    if (mgr.getPendingCount() > 0) {
      flushL1(mgr, "before_prompt_build_flush", true).then(async () => {
        try {
          const allEntries = await readAllOffloadEntries(mgr.ctx);
          const nullCount = allEntries.filter((e: any) => e.node_id === null).length;
          if (nullCount > 0) notifyL2NewNullEntries(nullCount);
        } catch { /* ignore */ }
      }).catch(() => {});
    }

    // In collect mode: trigger L1.5 (fire-and-forget) then skip L3 compression
    if (offloadConfig.mode === "collect") {
      // L1.5 task judgment — same logic as assemble, fire-and-forget
      const _prompt = typeof event?.prompt === "string" ? event.prompt : null;
      if (_prompt && _prompt.length > 0 && backendClient) {
        const promptHash = simpleHash(_prompt);
        const lastHash = mgr.lastL15PromptHash;
        if (promptHash !== lastHash) {
          mgr.lastL15PromptHash = promptHash;
          mgr.l15Settled = false;
          judgeL15(mgr, { prompt: _prompt, messages: event.messages ?? [] }, { sessionKey: ctx?.sessionKey }).catch((err) => {
            logger.warn(`[context-offload] collect L1.5 judge failed: ${err}`);
          });
        }
      }
      return;
    }

    // Fast-path re-apply + L3 compression + MMD injection
    const bpbHandler = createBeforePromptBuildHandler(mgr, logger, getContextWindow, pCfg);
    await bpbHandler(event, ctx);
  });

  // ─── Register Context Engine ───────────────────────────────────────────────
  logger.debug?.(`[context-offload] [DIAG] Hooks registered via api.on: [${_hookNames.join(", ")}] (${_hookNames.length} total)`);

  // In "collect" mode: skip Context Engine entirely, use legacy compaction.
  // L1/L1.5/L2 still run async but L3 is disabled.
  if (offloadConfig.mode === "collect") {
    const _configSlotCE = (api.config as any)?.plugins?.slots?.contextEngine;
    if (_configSlotCE === "memory-tencentdb") {
      logger.warn(`[context-offload] Mode "collect" but slots.contextEngine="${_configSlotCE}". Context Engine will NOT be registered in collect mode - consider removing the slot or switching to mode "backend".`);
    }
    logger.info(`[context-offload] Mode "collect": L3 disabled, context engine NOT registered (using legacy compaction). L1/L1.5/L2 active.`);
    // Force L1.5 settled so L2 poll doesn't block forever
    if (_lastActiveMgr) (_lastActiveMgr as any).l15Settled = true;
    // Start reclaim scheduler if needed, then skip to end
    _contextEngineRegistered = true; // prevent future registration attempts
  } else {
  // ─── Normal mode: register Context Engine ─────────────────────────────────
  const engineOpts = {
    sessions, logger, pCfg, getContextWindow, dataRoot,
    notifyL2NewNullEntries, clearL2Timeout: clearL2Poll, l4State,
    flushL1, backendClient, judgeL15,
    disposeL15: () => { _l15Disposed = true; },
  };

  // Singleton pattern: create engine once, update on subsequent calls.
  // OpenClaw's registerContextEngine() only succeeds on the FIRST call for
  // a given id. But only the LAST register() invocation produces live hooks.
  // So we hot-update the singleton engine's internal refs on every call.
  if (!_sharedEngine) {
    _sharedEngine = new OffloadContextEngine(engineOpts);
  } else {
    _sharedEngine.update(engineOpts);
    logger.debug?.("[context-offload] Context engine singleton updated with latest closures");
  }
  const engine = _sharedEngine;

  if (!_contextEngineRegistered) {
    // Pre-check: verify config slots.contextEngine points to this plugin.
    // If the slot is configured for another engine (e.g. "legacy"), we must NOT
    // register — even if api.registerContextEngine() would return ok:true,
    // the framework won't actually call our assemble(), causing L1.5 to never settle.
    const CE_PLUGIN_ID = "memory-tencentdb";
    const configSlotCE = (api.config as any)?.plugins?.slots?.contextEngine;
    if (configSlotCE !== CE_PLUGIN_ID) {
      logger.warn(`[context-offload] Config plugins.slots.contextEngine="${configSlotCE ?? "(not set)"}" (expected "${CE_PLUGIN_ID}"). Context engine slot not assigned to this plugin - ALL offload functions disabled.`);
      _contextEngineRejected = true;
      return;
    }

    // First registration — actually register with the framework
    let ceSlotOccupied = false;
    try {
      const result = api.registerContextEngine(CE_PLUGIN_ID, () => engine) as any;
      if (result && result.ok === false) {
        logger.error(`[context-offload] registerContextEngine returned { ok: false, existingOwner: ${result.existingOwner ?? "?"} }. Context engine slot occupied — ALL offload functions disabled!`);
        ceSlotOccupied = true;
      } else {
        _contextEngineRegistered = true;
        logger.debug?.("[context-offload] Context engine registered successfully (first call)");
      }
    } catch (ceErr) {
      logger.warn(`[context-offload] registerContextEngine factory failed: ${ceErr}, trying direct object`);
      try {
        const result2 = api.registerContextEngine(CE_PLUGIN_ID, engine) as any;
        if (result2 && result2.ok === false) {
          logger.error(`[context-offload] registerContextEngine direct returned { ok: false }. Context engine slot occupied — ALL offload functions disabled!`);
          ceSlotOccupied = true;
        } else {
          _contextEngineRegistered = true;
          logger.debug?.("[context-offload] Context engine registered successfully (direct mode)");
        }
      } catch (ceErr2) {
        logger.error(`[context-offload] registerContextEngine direct also failed: ${ceErr2}. ALL offload functions disabled!`);
        ceSlotOccupied = true;
      }
    }
    if (ceSlotOccupied) {
      _contextEngineRejected = true;
      logger.error("[context-offload] Offload module DISABLED: context engine slot occupied by another plugin. All hooks will be no-ops.");
      return; // Early exit — do not start reclaim scheduler either
    }
  } else {
    logger.debug?.("[context-offload] Context engine already registered, singleton updated (hot-refresh)");
  }
  } // end else (non-collect mode)

  // ─── Reclaim Scheduler ──────────────────────────────────────────────────────
  // Clean up any lingering reclaim timer from previous registerOffload() call
  if (_reclaimTimer !== null) { clearTimeout(_reclaimTimer); _reclaimTimer = null; }

  const _retentionDays = offloadConfig.offloadRetentionDays;
  const _logMaxSizeMb = offloadConfig.logMaxSizeMb;
  if (_retentionDays >= 3) {
    const INITIAL_DELAY_MS = 5 * 60 * 1000; // 5 min after startup
    const RECLAIM_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24h

    const scheduleReclaim = (delayMs: number) => {
      _reclaimTimer = setTimeout(async () => {
        try {
          const stats = await reclaimOffloadData(dataRoot, {
            retentionDays: _retentionDays,
            logMaxSizeMb: _logMaxSizeMb,
          }, logger);
          logger.debug?.(
            `[context-offload] Reclaim done: jsonl=${stats.deletedJsonl}, refs=${stats.deletedRefs}, ` +
            `mmds=${stats.deletedMmds}, logs=${stats.truncatedLogs}, registry=${stats.prunedRegistryEntries}`,
          );
        } catch (err) {
          logger.warn(`[context-offload] Reclaim failed: ${err}`);
        }
        scheduleReclaim(RECLAIM_INTERVAL_MS);
      }, delayMs);
      if (_reclaimTimer && typeof _reclaimTimer === "object" && "unref" in _reclaimTimer) {
        (_reclaimTimer as any).unref();
      }
    };
    scheduleReclaim(INITIAL_DELAY_MS);
    logger.debug?.(`[context-offload] Reclaim scheduler started: retentionDays=${_retentionDays}, logMaxSizeMb=${_logMaxSizeMb}`);
  }

  logger.debug?.("[context-offload] Offload module registration complete.");
}

// ─── OffloadContextEngine ────────────────────────────────────────────────────

class OffloadContextEngine {
  private _sessions: SessionRegistry;
  private _logger: PluginLogger;
  private _pCfg: Partial<PluginConfig>;
  private _getContextWindow: () => number;
  private _notifyL2NewNullEntries: (count: number) => void;
  private _clearL2Timeout: () => void;
  private _l4State: { pendingResult: any };
  private _flushL1: (mgr: OffloadStateManager, triggerSource: string, fireAndForget?: boolean, maxCount?: number) => Promise<void>;
  private _backendClient: BackendClient | null;
  private _judgeL15: (mgr: OffloadStateManager, event: any, ctx: any) => Promise<void>;
  private _disposeL15: () => void;

  constructor(opts: any) {
    this.update(opts);
  }

  /**
   * Hot-update all internal references. Called on every registerOffload()
   * invocation so the singleton engine always delegates to the LATEST
   * closures (hooks, sessions, flushL1, etc.) produced by the most recent
   * register() call — which is the only one whose hooks are actually live.
   */
  update(opts: any): void {
    this._sessions = opts.sessions;
    this._logger = opts.logger;
    this._pCfg = opts.pCfg;
    this._getContextWindow = opts.getContextWindow;
    this._notifyL2NewNullEntries = opts.notifyL2NewNullEntries;
    this._clearL2Timeout = opts.clearL2Timeout;
    this._l4State = opts.l4State;
    this._flushL1 = opts.flushL1;
    this._backendClient = opts.backendClient;
    this._judgeL15 = opts.judgeL15;
    this._disposeL15 = opts.disposeL15 ?? (() => {});
  }

  get info() {
    return { id: "openclaw-context-offload", name: "Context Offload Engine", version: "0.7.0", ownsCompaction: true };
  }

  async bootstrap(params: any) {
    const { sessionId, sessionKey } = params;
    const logger = this._logger;
    logger.debug?.(`[context-offload] >>> CE.bootstrap CALLED: sessionKey=${sessionKey}, sessionId=${sessionId?.slice(0, 12)}...`);
    if (isInternalMemorySession(sessionKey)) {
      logger.debug?.(`[context-offload] bootstrap SKIP: internal memory session (${sessionKey})`);
      return { bootstrapped: false, reason: "internal_memory_session" };
    }
    try {
      if (sessionKey) {
        const entry = await this._sessions.resolveIfAllowed(sessionKey, sessionId);
        if (entry) {
          // Attach per-session manager to params for assemble/afterTurn
          params._offloadManager = entry.manager;
        }
      }
      return { bootstrapped: true };
    } catch (err) {
      return { bootstrapped: false, reason: String(err) };
    }
  }

  async ingest(params: any) {
    const { message } = params;
    if (!message) return { ingested: false };
    const role = message.role ?? message.message?.role;
    if (role === "toolResult" || role === "tool") {
      const toolCallId = message.toolCallId ?? message.tool_call_id ?? message.message?.toolCallId ?? message.message?.tool_call_id;
      if (toolCallId) {
        let mgr: OffloadStateManager | undefined = params._offloadManager;
        if (!mgr && params.sessionKey) {
          mgr = this._sessions.get(params.sessionKey)?.manager;
        }
        if (mgr) mgr.processedToolCallIds.add(toolCallId);
        return { ingested: true };
      }
    }
    return { ingested: false };
  }

  async assemble(params: any) {
    const { messages, tokenBudget, prompt } = params;
    const logger = this._logger;
    logger.debug?.(`[context-offload] assemble CALLED: msgs=${messages?.length ?? 0}, budget=${tokenBudget ?? "N/A"}, prompt=${typeof prompt === "string" ? prompt.length + " chars" : "none"}, sessionKey=${params.sessionKey ?? "?"}`);
    // Resolve stateManager: prefer params._offloadManager (set by bootstrap),
    // then fall back to SessionRegistry resolve (framework may pass different params objects).
    let stateManager: OffloadStateManager | undefined = params._offloadManager;
    if (!stateManager && params.sessionKey) {
      try {
        const entry = await this._sessions.resolveIfAllowed(params.sessionKey, params.sessionId);
        if (entry) {
          stateManager = entry.manager;
          params._offloadManager = entry.manager; // cache for compact/afterTurn
          logger.debug?.(`[context-offload] assemble: resolved manager from SessionRegistry for ${params.sessionKey}`);
        }
      } catch (err) {
        logger.warn(`[context-offload] assemble: failed to resolve session ${params.sessionKey}: ${err}`);
      }
    }
    const pCfg = this._pCfg;

    if (!stateManager) {
      logger.debug?.(`[context-offload] assemble SKIP: no stateManager (sessionKey=${params.sessionKey ?? "none"})`);
      return { messages: messages ? [...messages] : [], estimatedTokens: 0 };
    }

    const workMessages = messages ? [...messages] : [];
    const _asmStart = Date.now();
    logger.debug?.(`[context-offload] assemble START: msgCount=${workMessages.length}, budget=${tokenBudget ?? "N/A"}, pending=${stateManager.getPendingCount()}, confirmed=${stateManager.confirmedOffloadIds?.size ?? 0}, deleted=${stateManager.deletedOffloadIds?.size ?? 0}`);

    // Cache prompt early so _buildL1RecentContext() has it when L1.5 fires
    // (assemble runs before llm_input, which is where cachedUserPrompt was previously set)
    if (typeof prompt === "string" && prompt.length > 0) {
      stateManager.cachedUserPrompt = prompt;
    }

    if (workMessages.length > 0) {
      stateManager.cachedLatestTurnMessages = _extractLatestTurn(workMessages, prompt);
      stateManager.cachedRecentHistory = _extractRecentHistory(workMessages, prompt);
    }

    try {

      // L1.5 task judgment — fire-and-forget (sole trigger point)
      if (!prompt || typeof prompt !== "string" || prompt.length === 0) {
        logger.debug?.(`[context-offload] assemble L1.5 SKIP: no prompt (prompt=${typeof prompt}, len=${prompt?.length ?? 0})`);
      } else if (!this._backendClient) {
        logger.debug?.(`[context-offload] assemble L1.5 SKIP: no backendClient`);
      } else {
        const promptHash = simpleHash(prompt);
        const lastHash = stateManager.lastL15PromptHash;
        if (promptHash === lastHash) {
          logger.debug?.(`[context-offload] assemble L1.5 SKIP: same prompt hash (${promptHash}), l15Settled=${stateManager.l15Settled}`);
        } else {
          stateManager.lastL15PromptHash = promptHash;
          stateManager.l15Settled = false;
          logger.debug?.(`[context-offload] assemble L1.5 TRIGGERED: new prompt hash (${promptHash}), l15Settled=false (reset), activeMmd=${stateManager.getActiveMmdFile() ?? "null"}`);
          this._judgeL15(
            stateManager,
            { prompt, messages: workMessages },
            { sessionKey: stateManager.getLastSessionKey() },
          ).catch((err) => {
            logger.warn(`[context-offload] assemble L1.5 judge failed: ${err}`);
          });
        }
      }

      // L1 flush is now handled inside judgeL15 (l15_pre_flush) before
      // recording the boundary startIndex. No separate flush needed here.

      // ── Raw token snapshot BEFORE fast-path re-apply ──
      // This captures what the framework originally passed in, before any offload
      // replacements. Crucial for understanding the delta between after_tool_call
      // and assemble traces.
      // Use the same sys tokens basis as L3 compression to ensure consistent comparisons.
      const _rawMsgCountBeforeFP = workMessages.length;
      // Use fast estimate for raw token count (only used for logging/tracing, not for compression decisions)
      const _rawMsgTokens = fastEstimateMessages(workMessages);

      // Fast-path re-apply
      const hasConfirmed = stateManager.confirmedOffloadIds?.size > 0;
      const hasDeleted = stateManager.deletedOffloadIds?.size > 0;
      let offloadEntries: any[] | null = null;
      let offloadMap: Map<string, any> | null = null;
      let _fpReplacedCount = 0;
      let _fpDeletedCount = 0;
      let _fpCompressedCount = 0;

      // ── FP-BOUNDARY-DELETE: fast head-delete based on last aggressive boundary ──
      // After aggressive deletes N messages from the head, the framework replays
      // the full history next time (including those already-deleted messages).
      // We record the boundary message's index + fingerprint after aggressive,
      // then on next assemble we verify the boundary is still at the same position
      // with the same content and splice everything before it.
      const _boundary = stateManager._lastAggressiveBoundary;
      let _fpBoundaryDeleted = 0;
      if (_boundary && prompt && prompt.length > 0
          && workMessages.length > _boundary.originalIndex && _boundary.originalIndex > 0) {
        const candidateMsg = workMessages[_boundary.originalIndex];
        if (_msgFingerprint(candidateMsg) === _boundary.fingerprint) {
          let headDeleteEnd = _boundary.originalIndex;
          // Forward: if the boundary message itself is a toolResult, extend to consume
          // all consecutive toolResults (their tool_use is in the delete zone).
          while (headDeleteEnd < workMessages.length && isToolResultMessage(workMessages[headDeleteEnd])) {
            headDeleteEnd++;
          }
          // Backward: if the last kept message before cut is assistant(tool_use),
          // its tool_results may be right after the cut — include them in deletion.
          // (This shouldn't happen since aggressive guarantees clean cuts, but safety.)
          if (headDeleteEnd > 0 && headDeleteEnd < workMessages.length) {
            const lastDeleted = workMessages[headDeleteEnd - 1];
            if (isAssistantMessageWithToolUse(lastDeleted)) {
              // Extend to include following toolResults that belong to this tool_use
              while (headDeleteEnd < workMessages.length && isToolResultMessage(workMessages[headDeleteEnd])) {
                headDeleteEnd++;
              }
            }
          }
          // Don't delete everything
          if (headDeleteEnd > 0 && headDeleteEnd < workMessages.length) {
            workMessages.splice(0, headDeleteEnd);
            _fpDeletedCount += headDeleteEnd;
            _fpBoundaryDeleted = headDeleteEnd;
            logger.debug?.(`[context-offload] assemble FP-BOUNDARY-DELETE: spliced ${headDeleteEnd} old msgs (boundaryIdx=${_boundary.originalIndex}, was=${workMessages.length + headDeleteEnd}, now=${workMessages.length})`);
          }
        } else {
          // Fingerprint mismatch — boundary invalid, clear it
          logger.debug?.(`[context-offload] assemble FP-BOUNDARY-DELETE: fingerprint mismatch at idx=${_boundary.originalIndex}, skipping (expected=${_boundary.fingerprint}, got=${_msgFingerprint(candidateMsg)})`);
          stateManager._lastAggressiveBoundary = null;
        }
      }

      if (hasConfirmed || hasDeleted) {
        offloadEntries = await readOffloadEntries(stateManager.ctx);
        offloadMap = new Map();
        populateOffloadLookupMap(offloadMap, offloadEntries);
        stateManager.setCachedOffloadMap(offloadMap);

        const indicesToDelete: number[] = [];
        for (let i = 0; i < workMessages.length; i++) {
          const msg = workMessages[i];
          const tid = extractToolCallId(msg);
          const tidNorm = tid ? normalizeToolCallIdForLookup(tid) : null;
          if (tid && hasDeleted && (stateManager.deletedOffloadIds.has(tid) || (tidNorm && stateManager.deletedOffloadIds.has(tidNorm)))) {
            indicesToDelete.push(i); _fpDeletedCount++; continue;
          }
          if (hasDeleted && isOnlyToolUseAssistant(msg)) {
            const tuIds = extractAllToolUseIds(msg);
            if (tuIds.length > 0 && tuIds.every((id) => stateManager.deletedOffloadIds.has(id) || stateManager.deletedOffloadIds.has(normalizeToolCallIdForLookup(id)))) {
              indicesToDelete.push(i); _fpDeletedCount++; continue;
            }
          }
          // FIX: For mixed assistant messages (text + tool_use), strip deleted tool_use
          // blocks to prevent orphaned tool_use without matching tool_result (Anthropic 400).
          if (hasDeleted && isAssistantMessageWithToolUse(msg) && !isOnlyToolUseAssistant(msg)) {
            const content = msg.type === "message" ? msg.message?.content : msg.content;
            if (Array.isArray(content)) {
              for (let j = content.length - 1; j >= 0; j--) {
                const block = content[j] as any;
                if ((block.type === "tool_use" || block.type === "toolCall") && block.id) {
                  const blockIdNorm = normalizeToolCallIdForLookup(block.id);
                  if (stateManager.deletedOffloadIds.has(block.id) || stateManager.deletedOffloadIds.has(blockIdNorm)) {
                    content.splice(j, 1);
                  }
                }
              }
            }
          }
          if (msg._offloaded) continue;
          if (tid && hasConfirmed && (stateManager.confirmedOffloadIds.has(tid) || (tidNorm && stateManager.confirmedOffloadIds.has(tidNorm)))) {
            const entry = getOffloadEntry(offloadMap, tid);
            if (entry && isToolResultMessage(msg)) { replaceWithSummary(msg, entry); msg._offloaded = true; _fpReplacedCount++; }
          }
          if (isOnlyToolUseAssistant(msg)) {
            const tuIds = extractAllToolUseIds(msg);
            const allConfirmed = tuIds.length > 0 && tuIds.every((id) => stateManager.confirmedOffloadIds.has(id) || stateManager.confirmedOffloadIds.has(normalizeToolCallIdForLookup(id)));
            if (allConfirmed) {
              const tuEntries = tuIds.map((id) => getOffloadEntry(offloadMap!, id)).filter(Boolean) as any[];
              if (tuEntries.length === tuIds.length) { replaceAssistantToolUseWithSummary(msg, tuEntries); msg._offloaded = true; _fpCompressedCount++; }
            }
          } else if (isAssistantMessageWithToolUse(msg)) {
            compressNonCurrentToolUseBlocks(msg, offloadMap, new Set(), stateManager.confirmedOffloadIds);
          }
        }
        if (indicesToDelete.length > 0) {
          for (let k = indicesToDelete.length - 1; k >= 0; k--) workMessages.splice(indicesToDelete[k], 1);
        }
      }

      // ── Post fast-path summary ──
      const _fpMsgCountAfter = workMessages.length;
      logger.debug?.(`[context-offload] assemble FAST-PATH: rawMsgTokens≈${_rawMsgTokens} (${_rawMsgCountBeforeFP} msgs) → ` +
        `replaced=${_fpReplacedCount} toolResults, compressed=${_fpCompressedCount} assistants, deleted=${_fpDeletedCount} msgs → ` +
        `${_fpMsgCountAfter} msgs remaining, confirmed=${stateManager.confirmedOffloadIds?.size ?? 0}, deleted=${stateManager.deletedOffloadIds?.size ?? 0}`,
      );

      // Active MMD injection is now handled by after_tool_call hook (which has
      // access to event.messages after the openclaw patch). The hook checks
      // L1.5 settled status and reads the latest MMD content (reflecting L2 updates).
      // assemble no longer injects active MMD — it only handles L3 compression
      // and history MMD injection (AGGRESSIVE).

      // L3 compression
      const contextWindow = this._getContextWindow();
      // Use the smaller of framework budget and model context window to avoid overflow.
      const effectiveBudget = tokenBudget ? Math.min(tokenBudget, contextWindow) : contextWindow;
      const mildRatio = pCfg.mildOffloadRatio ?? PLUGIN_DEFAULTS.mildOffloadRatio;
      const aggressiveRatio = pCfg.aggressiveCompressRatio ?? PLUGIN_DEFAULTS.aggressiveCompressRatio;
      const mildThreshold = Math.floor(effectiveBudget * mildRatio);
      const aggressiveThreshold = Math.floor(effectiveBudget * aggressiveRatio);

      // Include system prompt tokens in all token calculations.
      // assemble() doesn't receive systemPrompt directly, so use cached/estimated value.
      const _sysFromCache = stateManager.cachedSystemPromptTokens;
      const _sysFromOverhead = stateManager.getEstimatedSystemOverhead();
      const _sysFromRatio = Math.floor(effectiveBudget * (pCfg.defaultSystemOverheadRatio ?? PLUGIN_DEFAULTS.defaultSystemOverheadRatio));
      const systemTokensEstimate = _sysFromCache ?? _sysFromOverhead ?? _sysFromRatio;
      const _sysSource = _sysFromCache != null ? "cachedSystemPromptTokens" : _sysFromOverhead != null ? "estimatedSystemOverhead" : "defaultRatio";
      logger.debug?.(`[context-offload] assemble sys tokens: estimate=${systemTokensEstimate} (source=${_sysSource}, cache=${_sysFromCache ?? "null"}, overhead=${_sysFromOverhead ?? "null"}, ratio=${_sysFromRatio})`,
      );
      const precomputed = { systemTokens: systemTokensEstimate, userPromptTokens: 0 };

      // _rawTokensBefore uses the same sys basis as L3 compression for consistent comparisons
      const _rawTokensBefore = _rawMsgTokens + systemTokensEstimate;

      // ── Fast estimate: skip tiktoken when clearly below threshold ──
      // Use fast character-based estimation (~5ms) instead of tiktoken (~3-10s).
      // Only trigger precise tiktoken when estimate is near compression thresholds.
      const _fastEstStart = Date.now();
      const fastEst = fastEstimateMessages(workMessages) + systemTokensEstimate + (prompt ? Math.ceil(prompt.length / 4) : 0);
      const _fastEstMs = Date.now() - _fastEstStart;
      const FAST_EST_SAFETY_MARGIN = 0.85; // 15% safety margin for estimation error

      let workingTokens: number;
      let snap: ReturnType<typeof buildTiktokenContextSnapshot> | null = null;
      let _usedFastPath = false;

      // ── Incremental estimation: if boundary-delete fired and we have cached
      // aggressive results, estimate tokens incrementally from new messages only.
      // This avoids tiktoken entirely for the common case of 1-2 new messages.
      const _boundaryCache = stateManager._lastAggressiveBoundary;
      const BOUNDARY_NEW_MSG_TOLERANCE = 20; // max new messages before forcing full recount
      if (_fpBoundaryDeleted > 0 && _boundaryCache
          && workMessages.length <= _boundaryCache.keptMsgCount + BOUNDARY_NEW_MSG_TOLERANCE
          && _boundaryCache.remainingTokens < aggressiveThreshold) {
        // Estimate: last aggressive tokens + new messages token delta
        const newMsgCount = Math.max(0, workMessages.length - _boundaryCache.keptMsgCount);
        const newMsgTokens = newMsgCount > 0
          ? fastEstimateMessages(workMessages.slice(workMessages.length - newMsgCount)) + (prompt ? Math.ceil(prompt.length / 4) : 0)
          : (prompt ? Math.ceil(prompt.length / 4) : 0);
        const incrementalEst = _boundaryCache.remainingTokens + newMsgTokens;
        if (incrementalEst < aggressiveThreshold) {
          workingTokens = incrementalEst;
          _usedFastPath = true;
          logger.debug?.(`[context-offload] assemble BOUNDARY-INCR-SKIP: incremental≈${incrementalEst} (base=${_boundaryCache.remainingTokens}+new=${newMsgTokens}, newMsgs=${newMsgCount}) < aggressive@${aggressiveThreshold}, skipping tiktoken`);
        } else {
          // Incremental estimate exceeds threshold — need precise tiktoken
          snap = buildTiktokenContextSnapshot("assemble", workMessages, null, prompt ?? null, precomputed);
          workingTokens = snap.totalTokens;
          logger.debug?.(`[context-offload] assemble L3 check (boundary-incr exceeded): total≈${workingTokens} (incr-est was ${incrementalEst}), msgs=${workMessages.length}, aggressive@${aggressiveThreshold}`);
        }
      } else if (fastEst < aggressiveThreshold * FAST_EST_SAFETY_MARGIN) {
        // Below aggressive threshold — use estimate for mild/skip decisions.
        // Mild only replaces tool results with summaries (no precise token math needed).
        // Only aggressive needs precise tiktoken (to compute exact delete count).
        workingTokens = fastEst;
        _usedFastPath = true;
        logger.debug?.(`[context-offload] assemble L3 FAST-SKIP: fastEst≈${fastEst} < ${Math.floor(aggressiveThreshold * FAST_EST_SAFETY_MARGIN)} (${(FAST_EST_SAFETY_MARGIN * 100).toFixed(0)}% aggressive), ` +
          `budget=${effectiveBudget}, msgs=${workMessages.length}, fastEstMs=${_fastEstMs}ms`,
        );
      } else if (!stateManager._lastAggressiveBoundary && prompt && prompt.length > 0) {
        // No boundary + has prompt + clearly above threshold → skip full tiktoken.
        // TAIL-ACCUMULATE will do its own precise calculation from the tail.
        workingTokens = fastEst;
        logger.debug?.(`[context-offload] assemble L3 TAIL-ACCUM-PENDING: fastEst≈${fastEst} (no boundary, will tail-accumulate), skipping full tiktoken`);
      } else {
        // Near/above aggressive threshold — do precise tiktoken
        snap = buildTiktokenContextSnapshot("assemble", workMessages, null, prompt ?? null, precomputed);
        workingTokens = snap.totalTokens;
        logger.debug?.(`[context-offload] assemble L3 check: total≈${workingTokens} (sys≈${systemTokensEstimate}, msgs≈${snap.messagesTokens}, user≈${snap.userPromptTokens}), ` +
          `budget=${effectiveBudget} (contextWindow=${contextWindow}, tokenBudget=${tokenBudget ?? "N/A"}), ` +
          `utilisation=${((workingTokens / effectiveBudget) * 100).toFixed(1)}%, mild@${mildThreshold}, aggressive@${aggressiveThreshold}, msgs=${workMessages.length}, fastEst=${fastEst}, fastEstMs=${_fastEstMs}ms`,
        );
      }

      let _aggDeletedCount = 0;
      let _aggRounds = 0;
      let _aggDeletedIds: string[] = [];
      let _aggTokensBefore = workingTokens;
      let _aggTokensAfter = workingTokens;
      let _aggDurationMs = 0;
      let _aggMmdInjected = 0;
      let _aggMmdTokens = 0;
      if (workingTokens >= aggressiveThreshold) {
        // ── TAIL-ACCUMULATE: when no boundary cache exists (first run), compute
        // tokens from tail until reaching 60% of budget, then discard the head.
        // This avoids the expensive full-tiktoken + multi-round aggressive loop.
        const TAIL_ACCUM_TARGET_RATIO = 0.60;
        const tailAccumTarget = Math.floor(effectiveBudget * TAIL_ACCUM_TARGET_RATIO) - systemTokensEstimate;
        if (!stateManager._lastAggressiveBoundary && workMessages.length > 0 && prompt && prompt.length > 0) {
          const _tailStart = Date.now();
          let accum = 0;
          let keepFrom = 0; // will keep [keepFrom ... end]
          for (let i = workMessages.length - 1; i >= 0; i--) {
            const msgTokens = tiktokenCount(JSON.stringify(workMessages[i], jsonReplacer));
            if (accum + msgTokens > tailAccumTarget) {
              keepFrom = i + 1;
              break;
            }
            accum += msgTokens;
          }
          // Tool-pair safety: extend keepFrom forward to not orphan toolResults
          while (keepFrom < workMessages.length && isToolResultMessage(workMessages[keepFrom])) {
            accum += tiktokenCount(JSON.stringify(workMessages[keepFrom], jsonReplacer));
            keepFrom++;
          }
          // Tool-pair safety (backward): if last deleted msg is assistant(tool_use),
          // its tool_results are in the keep zone — extend deletion to include them
          if (keepFrom > 0 && keepFrom < workMessages.length) {
            const lastDeleted = workMessages[keepFrom - 1];
            if (isAssistantMessageWithToolUse(lastDeleted)) {
              while (keepFrom < workMessages.length && isToolResultMessage(workMessages[keepFrom])) {
                accum += tiktokenCount(JSON.stringify(workMessages[keepFrom], jsonReplacer));
                keepFrom++;
              }
            }
          }
          // User message protection: don't cut past the last user message
          // (ensure the most recent user turn is always kept)
          for (let u = workMessages.length - 1; u >= keepFrom; u--) {
            const role = workMessages[u].role ?? workMessages[u].message?.role ?? workMessages[u].type;
            if (role === "user" || role === "human") {
              // Found last user msg in keep zone — good
              break;
            }
            if (u === keepFrom) {
              // No user message in keep zone — find the last one and adjust keepFrom
              for (let u2 = keepFrom - 1; u2 >= 0; u2--) {
                const r2 = workMessages[u2].role ?? workMessages[u2].message?.role ?? workMessages[u2].type;
                if (r2 === "user" || r2 === "human") {
                  keepFrom = u2;
                  break;
                }
              }
            }
          }
          // Minimum keep: always keep at least 10 messages
          const MIN_KEEP = 10;
          if (workMessages.length - keepFrom < MIN_KEEP) {
            keepFrom = Math.max(0, workMessages.length - MIN_KEEP);
          }
          // Don't delete everything
          if (keepFrom > 0 && keepFrom < workMessages.length) {
            // Collect deleted tool call IDs for offload tracking
            const tailDeletedIds: string[] = [];
            for (let d = 0; d < keepFrom; d++) {
              const msg = workMessages[d];
              const tid = extractToolCallId(msg) ?? (isOnlyToolUseAssistant(msg) ? extractAllToolUseIds(msg)[0] : null);
              if (tid) tailDeletedIds.push(tid);
            }
            workMessages.splice(0, keepFrom);
            _aggDeletedCount = keepFrom;
            _aggDeletedIds = tailDeletedIds;
            workingTokens = accum + systemTokensEstimate;
            _aggTokensAfter = workingTokens;
            _aggDurationMs = Date.now() - _tailStart;
            logger.info(`[context-offload] assemble TAIL-ACCUMULATE: kept ${workMessages.length} msgs from tail, deleted ${keepFrom} from head, tokens≈${workingTokens}, target=${tailAccumTarget}+sys=${systemTokensEstimate}, duration=${_aggDurationMs}ms`);
            // Mark deleted IDs
            if (tailDeletedIds.length > 0) {
              const statusUpdates = new Map<string, string | boolean>();
              for (const id of tailDeletedIds) { statusUpdates.set(id, "deleted"); stateManager.confirmedOffloadIds.add(id); stateManager.deletedOffloadIds.add(id); }
              markOffloadStatus(stateManager.ctx, statusUpdates).catch(() => {});
            }
            // Record boundary
            const boundaryFp = _msgFingerprint(workMessages[0]);
            let boundaryOrigIdx = -1;
            for (let bi = 0; bi < messages.length; bi++) {
              if (_msgFingerprint(messages[bi]) === boundaryFp) {
                if (bi + 1 < messages.length && workMessages.length > 1) {
                  if (_msgFingerprint(messages[bi + 1]) === _msgFingerprint(workMessages[1])) {
                    boundaryOrigIdx = bi; break;
                  }
                } else {
                  boundaryOrigIdx = bi; break;
                }
              }
            }
            if (boundaryOrigIdx >= 0) {
              stateManager._lastAggressiveBoundary = {
                originalIndex: boundaryOrigIdx,
                fingerprint: boundaryFp,
                keptMsgCount: workMessages.length,
                remainingTokens: workingTokens,
              };
              logger.info(`[context-offload] assemble TAIL-ACCUMULATE BOUNDARY recorded: idx=${boundaryOrigIdx}, kept=${workMessages.length}, tokens≈${workingTokens}`);
            }
          }
        } else {
          // Has boundary cache — use standard aggressive path
          logger.debug?.(`[context-offload] assemble L3-AGGRESSIVE: tokens≈${workingTokens} >= ${aggressiveThreshold}, starting...`);
          if (!offloadEntries) { offloadEntries = await readOffloadEntries(stateManager.ctx); offloadMap = new Map(); populateOffloadLookupMap(offloadMap!, offloadEntries); }
          const countTokens = createL3TokenCounter(pCfg, logger);
          const aggressiveDeleteRatio = (pCfg as any).aggressiveDeleteRatio ?? PLUGIN_DEFAULTS.aggressiveDeleteRatio;
          const currentTaskNodeIds = await getCurrentTaskNodeIds(stateManager);
          const _aggStart = Date.now();
          // aggressiveThreshold includes systemTokensEstimate, but the internal
          // function computes remainingTokens WITHOUT system tokens (sysPrompt=null).
          // Subtract systemTokensEstimate so the comparison is consistent.
          // Target 85% of threshold to leave buffer for subsequent tool loop messages.
          // Without buffer: tokens hover at 109K (threshold=108.8K) → every tool call re-triggers.
          const AGGRESSIVE_TARGET_RATIO = 0.85;
          const aggressiveTargetForMsgs = Math.max(0, Math.floor(aggressiveThreshold * AGGRESSIVE_TARGET_RATIO) - systemTokensEstimate);
          const result = await aggressiveCompressUntilBelowThreshold(
            workMessages, offloadMap!, currentTaskNodeIds, aggressiveDeleteRatio, stateManager, logger, aggressiveTargetForMsgs, countTokens, null, prompt ?? null,
          );
          _aggDeletedCount = result.deletedCount;
          _aggRounds = result.rounds;
          _aggDeletedIds = result.allDeletedToolCallIds;
          workingTokens = result.remainingTokens + systemTokensEstimate;

          _aggTokensAfter = workingTokens;
          _aggDurationMs = Date.now() - _aggStart;
          logger.debug?.(`[context-offload] assemble L3-AGGRESSIVE done: rounds=${result.rounds}, deleted=${result.deletedCount}, remaining≈${workingTokens} (raw=${result.remainingTokens}+sys=${systemTokensEstimate}), deletedIds=${result.allDeletedToolCallIds.length}, stalledByUserMsg=${result.stalledByUserMsg ?? false}, duration=${_aggDurationMs}ms`);
          if (_aggDurationMs > 10_000) {
            logger.warn(`[context-offload] assemble L3-AGGRESSIVE SLOW: ${_aggDurationMs}ms (rounds=${result.rounds}, deleted=${result.deletedCount}, remaining≈${workingTokens})`);
          }
          // Record boundary for FP-BOUNDARY-DELETE on next replay (only when prompt present)
          if (result.deletedCount > 0 && workMessages.length > 0 && prompt && prompt.length > 0) {
            const boundaryFp = _msgFingerprint(workMessages[0]);
            // Find the boundary message's position in the original framework input
            let boundaryOrigIdx = -1;
            for (let bi = 0; bi < messages.length; bi++) {
              if (_msgFingerprint(messages[bi]) === boundaryFp) {
                // Verify with next message too to avoid hash collision on duplicate content
                if (bi + 1 < messages.length && workMessages.length > 1) {
                  if (_msgFingerprint(messages[bi + 1]) === _msgFingerprint(workMessages[1])) {
                    boundaryOrigIdx = bi;
                    break;
                  }
                } else {
                  boundaryOrigIdx = bi;
                  break;
                }
              }
            }
            if (boundaryOrigIdx >= 0) {
              stateManager._lastAggressiveBoundary = {
                originalIndex: boundaryOrigIdx,
                fingerprint: boundaryFp,
                keptMsgCount: workMessages.length,
                remainingTokens: workingTokens,
              };
              logger.debug?.(`[context-offload] assemble BOUNDARY recorded: idx=${boundaryOrigIdx}, fp=${boundaryFp}, kept=${workMessages.length}, tokens≈${workingTokens}`);
            } else {
              // Could not locate boundary in original messages — clear stale boundary
              stateManager._lastAggressiveBoundary = null;
              logger.debug?.(`[context-offload] assemble BOUNDARY: could not locate in original msgs, cleared`);
            }
          }
          if (result.allDeletedToolCallIds.length > 0) {
            const statusUpdates = new Map<string, string | boolean>();
            for (const id of result.allDeletedToolCallIds) { statusUpdates.set(id, "deleted"); stateManager.confirmedOffloadIds.add(id); stateManager.deletedOffloadIds.add(id); }
            markOffloadStatus(stateManager.ctx, statusUpdates).catch(() => {});
            const mmdInj = await buildHistoryMmdInjection(result.allDeletedToolCallIds, offloadMap!, offloadEntries, stateManager, logger, countTokens, effectiveBudget, pCfg);
            if (mmdInj.injectedMessages.length > 0) {
              removeExistingMmdInjections(workMessages);
              const histInsertIdx = findHistoryMmdInsertionPoint(workMessages);
              workMessages.splice(histInsertIdx, 0, ...mmdInj.injectedMessages);
              _aggMmdInjected = mmdInj.injectedMessages.length;
              _aggMmdTokens = mmdInj.totalMmdTokens;
              workingTokens += mmdInj.totalMmdTokens;
              logger.debug?.(`[context-offload] assemble L3-AGGRESSIVE MMD injection: ${mmdInj.injectedMessages.length} msgs, ${mmdInj.totalMmdTokens} tokens, budget=${Math.floor(effectiveBudget * (pCfg.mmdMaxTokenRatio ?? PLUGIN_DEFAULTS.mmdMaxTokenRatio))}, files=[${mmdInj.mmdFiles.join(",")}], workingTokens now=${workingTokens}`);

              // Debug: dump injected MMD message content
              for (let ii = 0; ii < mmdInj.injectedMessages.length; ii++) {
                const im = mmdInj.injectedMessages[ii] as any;
                let ic = "";
                if (typeof im.content === "string") ic = im.content;
                else if (Array.isArray(im.content)) ic = im.content.map((c: any) => typeof c === "string" ? c : (c.text ?? "")).join(" ");
                const lines = ic.split("\n");
                logger.debug?.(`[context-offload]   MMD-inject[${ii}] role=${im.role}, lines=${lines.length}, preview=${ic.replace(/\n/g, "\\n").slice(0, 200)}${ic.length > 200 ? "..." : ""}`);
              }
            } else {
              logger.debug?.(`[context-offload] assemble L3-AGGRESSIVE MMD injection: no history MMDs to inject`);
            }
          }
          // If aggressive stalled due to user message protection, force emergency
          if (result.stalledByUserMsg && workingTokens >= aggressiveThreshold) {
            logger.warn(`[context-offload] assemble L3-AGGRESSIVE stalled, forcing emergency fallback`);
            stateManager._forceEmergencyNext = true;
          }
        } // end else (standard aggressive path)
      } else {
        logger.debug?.(`[context-offload] assemble L3-AGGRESSIVE: SKIP (tokens≈${workingTokens} < ${aggressiveThreshold})`);
      }

      // Summary after AGGRESSIVE (was full dump, now aggregated)
      if (_aggDeletedCount > 0) {
        const mmdCount = workMessages.filter((m: any) => m._mmdContextMessage || m._mmdInjection).length;
        const offloadedCount = workMessages.filter((m: any) => m._offloaded).length;
        logger.debug?.(`[context-offload] POST-AGGRESSIVE: ${workMessages.length} msgs remaining, mmd=${mmdCount}, offloaded=${offloadedCount}, deleted=${_aggDeletedCount}`);
      }

      let _mildReplacedCount = 0;
      let _mildFinalThreshold = 0;
      let _mildDurationMs = 0;
      let _mildTokensBefore = workingTokens;
      let _mildReplacedIds: string[] = [];
      if (workingTokens >= mildThreshold) {
        logger.debug?.(`[context-offload] assemble L3-MILD: tokens≈${workingTokens} >= ${mildThreshold}, starting...`);
        if (!offloadEntries) { offloadEntries = await readOffloadEntries(stateManager.ctx); offloadMap = new Map(); populateOffloadLookupMap(offloadMap!, offloadEntries); }
        const currentTaskNodeIds = await getCurrentTaskNodeIds(stateManager);
        const mildScanRatio = (pCfg as any).mildOffloadScanRatio ?? PLUGIN_DEFAULTS.mildOffloadScanRatio;
        const _mildStart = Date.now();
        const cascadeResult = compressByScoreCascade(workMessages, offloadMap!, currentTaskNodeIds, mildScanRatio, logger);
        _mildReplacedCount = cascadeResult.replacedCount;
        _mildFinalThreshold = cascadeResult.finalThreshold;
        _mildDurationMs = Date.now() - _mildStart;
        _mildReplacedIds = cascadeResult.replacedToolCallIds;
        logger.debug?.(`[context-offload] assemble L3-MILD done: replaced=${cascadeResult.replacedCount}, finalThreshold=${cascadeResult.finalThreshold}, ids=[${cascadeResult.replacedToolCallIds.slice(0, 5).join(",")}${cascadeResult.replacedToolCallIds.length > 5 ? "..." : ""}], duration=${_mildDurationMs}ms`);
        if (cascadeResult.replacedCount > 0) {
          for (const id of cascadeResult.replacedToolCallIds) stateManager.confirmedOffloadIds.add(id);
          const mildUpdates = new Map<string, string | boolean>();
          for (const id of cascadeResult.replacedToolCallIds) mildUpdates.set(id, true);
          markOffloadStatus(stateManager.ctx, mildUpdates).catch(() => {});

          // Summary after MILD replacement (was full dump, now aggregated)
          const replacedCount = workMessages.filter((m: any) => {
            const c = typeof m.content === "string" ? m.content : "";
            return c.includes("[Offload summary") || c.includes("⚡ offload");
          }).length;
          logger.debug?.(`[context-offload] POST-MILD: ${workMessages.length} msgs, replaced=${replacedCount}`);
        }
      } else {
        logger.debug?.(`[context-offload] assemble L3-MILD: SKIP (tokens≈${workingTokens} < ${mildThreshold})`);
      }

      // Emergency — reuse workingTokens instead of redundant full tiktoken snapshot
      const emergencyRatio = pCfg.emergencyCompressRatio ?? PLUGIN_DEFAULTS.emergencyCompressRatio;
      const emergencyTargetRatio = pCfg.emergencyTargetRatio ?? PLUGIN_DEFAULTS.emergencyTargetRatio;
      const emergencyThreshold = Math.floor(effectiveBudget * emergencyRatio);
      const emergencyTarget = Math.floor(effectiveBudget * emergencyTargetRatio);
      let _emDeletedCount = 0;
      let _emTokensBefore = workingTokens;
      let _emTriggered = false;
      const forceEmergency = stateManager._forceEmergencyNext === true;
      if (forceEmergency) stateManager._forceEmergencyNext = false;
      if ((workingTokens >= emergencyThreshold || forceEmergency) && workMessages.length > EMERGENCY_MIN_MESSAGES_TO_KEEP) {
        _emTriggered = true;
        _usedFastPath = false; // force precise finalSnap after emergency
        logger.warn(`[context-offload] assemble EMERGENCY: tokens≈${workingTokens} >= ${emergencyThreshold} (${(emergencyRatio * 100).toFixed(0)}%), force=${forceEmergency}, target=${emergencyTarget} (${(emergencyTargetRatio * 100).toFixed(0)}%), msgTarget=${emergencyTarget - systemTokensEstimate}`);
        const countTokensEmg = createL3TokenCounter(pCfg, logger);
        const _emStart = Date.now();
        const emResult = emergencyCompress(workMessages, emergencyTarget - systemTokensEstimate, countTokensEmg, null, prompt ?? null, logger);
        _emDeletedCount = emResult.deletedCount;
        workingTokens = emResult.remainingTokens + systemTokensEstimate;
        const _emDurationMs = Date.now() - _emStart;
        if (_emDurationMs > 10_000) {
          logger.warn(`[context-offload] assemble EMERGENCY SLOW: ${_emDurationMs}ms (deleted=${emResult.deletedCount}, remaining≈${workingTokens})`);
        } else {
          logger.debug?.(`[context-offload] assemble EMERGENCY done: deleted=${emResult.deletedCount} msgs, remaining≈${workingTokens} (raw=${emResult.remainingTokens}+sys=${systemTokensEstimate}), deletedIds=${emResult.deletedToolCallIds.length}, duration=${_emDurationMs}ms`);
        }
        if (emResult.deletedToolCallIds.length > 0) {
          const emUpdates = new Map<string, string | boolean>();
          for (const id of emResult.deletedToolCallIds) { emUpdates.set(id, "deleted"); stateManager.confirmedOffloadIds.add(id); stateManager.deletedOffloadIds.add(id); }
          markOffloadStatus(stateManager.ctx, emUpdates).catch(() => {});
        }
        // Re-record boundary after emergency (only when prompt present)
        if (emResult.deletedCount > 0 && workMessages.length > 0 && prompt && prompt.length > 0) {
          const boundaryFp = _msgFingerprint(workMessages[0]);
          let boundaryOrigIdx = -1;
          for (let bi = 0; bi < messages.length; bi++) {
            if (_msgFingerprint(messages[bi]) === boundaryFp) {
              if (bi + 1 < messages.length && workMessages.length > 1) {
                if (_msgFingerprint(messages[bi + 1]) === _msgFingerprint(workMessages[1])) {
                  boundaryOrigIdx = bi; break;
                }
              } else {
                boundaryOrigIdx = bi; break;
              }
            }
          }
          if (boundaryOrigIdx >= 0) {
            stateManager._lastAggressiveBoundary = {
              originalIndex: boundaryOrigIdx,
              fingerprint: boundaryFp,
              keptMsgCount: workMessages.length,
              remainingTokens: workingTokens,
            };
            logger.debug?.(`[context-offload] assemble EMERGENCY BOUNDARY recorded: idx=${boundaryOrigIdx}, kept=${workMessages.length}, tokens≈${workingTokens}`);
          } else {
            stateManager._lastAggressiveBoundary = null;
          }
        }
      } else {
        logger.debug?.(`[context-offload] assemble EMERGENCY: SKIP (tokens≈${workingTokens} < ${emergencyThreshold}, force=${forceEmergency}, msgs=${workMessages.length})`);
      }

      // L4 injection
      let systemPromptAddition: string | undefined;
      if (this._l4State.pendingResult?.appendSystemContext) {
        systemPromptAddition = this._l4State.pendingResult.appendSystemContext;
        this._l4State.pendingResult = null;
      }

      const finalSnap = _usedFastPath
        ? { totalTokens: workingTokens, messagesTokens: workingTokens - systemTokensEstimate, systemTokens: systemTokensEstimate, userPromptTokens: 0 }
        : buildTiktokenContextSnapshot("assemble_final", workMessages, null, prompt ?? null, precomputed);
      const tokensBefore = snap?.totalTokens ?? fastEst;
      const tokensSaved = tokensBefore - finalSnap.totalTokens;
      const _asmDuration = Date.now() - _asmStart;
      logger.debug?.(`[context-offload] assemble END (ok): ${messages?.length ?? 0}→${workMessages.length} msgs, rawTokens≈${_rawTokensBefore}, tokensBefore≈${tokensBefore} (FP: -${_rawTokensBefore - tokensBefore}, replaced=${_fpReplacedCount}, compressed=${_fpCompressedCount}, deleted=${_fpDeletedCount}), tokensAfter≈${finalSnap.totalTokens} (sys≈${systemTokensEstimate}), tokensSaved≈${tokensSaved}, totalSaved≈${_rawTokensBefore - finalSnap.totalTokens}, hasL4=${!!systemPromptAddition}, duration=${_asmDuration}ms`);

      // Async trace — fire-and-forget, must not block assemble return
      try {
        traceOffloadDecision({
          sessionKey: stateManager.getLastSessionKey(),
          stage: "L3.assemble.completed",
          input: {
            messagesBefore: messages?.length ?? 0,
            rawTokensBefore: _rawTokensBefore,
            rawMsgTokens: _rawMsgTokens,
            tokensBefore,
            budget: effectiveBudget,
            contextWindow,
            systemTokensEstimate,
            mildThreshold,
            aggressiveThreshold,
            emergencyThreshold,
            durationMs: _asmDuration,
          },
          output: {
            // Overall
            messagesAfter: workMessages.length,
            messagesRemoved: (messages?.length ?? 0) - workMessages.length,
            tokensAfter: finalSnap.totalTokens,
            tokensSaved,
            totalTokensSaved: _rawTokensBefore - finalSnap.totalTokens,
            utilisation: `${((finalSnap.totalTokens / effectiveBudget) * 100).toFixed(1)}%`,
            utilisationBefore: `${((_rawTokensBefore / effectiveBudget) * 100).toFixed(1)}%`,
            hasL4: !!systemPromptAddition,
            // Fast-path re-apply details
            fastPath: {
              rawTokens: _rawTokensBefore,
              tokensAfterFP: tokensBefore,
              tokensSavedByFP: _rawTokensBefore - tokensBefore,
              replacedToolResults: _fpReplacedCount,
              compressedAssistants: _fpCompressedCount,
              deletedMsgs: _fpDeletedCount,
              confirmedIds: stateManager.confirmedOffloadIds?.size ?? 0,
              deletedIds: stateManager.deletedOffloadIds?.size ?? 0,
            },
            // AGGRESSIVE details
            aggressive: {
              triggered: _aggDeletedCount > 0,
              tokensBefore: _aggTokensBefore,
              tokensAfter: _aggTokensAfter,
              deletedMsgs: _aggDeletedCount,
              deletedIds: _aggDeletedIds.slice(0, 20),
              rounds: _aggRounds,
              durationMs: _aggDurationMs,
              historyMmdInjected: _aggMmdInjected,
              historyMmdTokens: _aggMmdTokens,
            },
            // MILD details
            mild: {
              triggered: _mildReplacedCount > 0,
              tokensBefore: _mildTokensBefore,
              replacedCount: _mildReplacedCount,
              finalThreshold: _mildFinalThreshold,
              replacedIds: _mildReplacedIds.slice(0, 20),
              durationMs: _mildDurationMs,
            },
            // EMERGENCY details
            emergency: {
              triggered: _emTriggered,
              tokensBefore: _emTokensBefore,
              deletedMsgs: _emDeletedCount,
              forceEmergency,
            },
          },
          logger,
        });
      } catch { /* trace failure must not affect assemble */ }

      // Trace messages snapshots — original input vs processed output
      try {
        traceMessagesSnapshot({
          sessionKey: stateManager.getLastSessionKey(),
          stage: "assemble.input",
          messages: messages ?? [],
          label: "original messages (before assemble)",
          extra: {
            rawTokensBefore: _rawTokensBefore,
            budget: effectiveBudget,
            contextWindow,
          },
          logger,
        });
        traceMessagesSnapshot({
          sessionKey: stateManager.getLastSessionKey(),
          stage: "assemble.output",
          messages: workMessages,
          label: "workMessages (after assemble)",
          extra: {
            tokensAfter: finalSnap.totalTokens,
            tokensSaved,
            totalTokensSaved: _rawTokensBefore - finalSnap.totalTokens,
            budget: effectiveBudget,
            hasL4: !!systemPromptAddition,
          },
          logger,
        });
      } catch { /* trace failure must not affect assemble */ }

      // Upload plugin state + L3 token accounting to backend /store.
      try {
        const _triggerReason = _rawTokensBefore >= aggressiveThreshold
          ? "above_aggressive"
          : _rawTokensBefore >= mildThreshold
            ? "above_mild"
            : "below_mild";
        const _report = buildL3TriggerReport({
          stage: "assemble",
          triggerReason: _triggerReason,
          stateManager,
          event: { messages: workMessages }, // assemble has its own shape — patch check is n/a here
          contextWindow,
          mildThreshold,
          aggressiveThreshold,
          tokensBefore: _rawTokensBefore,
          tokensAfter: finalSnap.totalTokens,
          messagesBefore: messages?.length ?? 0,
          messagesAfter: workMessages.length,
          durationMs: _asmDuration,
          aboveMild: _rawTokensBefore >= mildThreshold,
          aboveAggressive: _rawTokensBefore >= aggressiveThreshold,
          mildReplacedCount: _mildReplacedCount,
          aggressiveDeletedCount: _aggDeletedCount,
          emergencyTriggered: _emTriggered,
          emergencyDeletedCount: _emDeletedCount,
        });
        reportL3Trigger(this._backendClient ?? null, _report, logger);
      } catch (reportErr) {
        logger.warn(`[context-offload] assemble L3 state-report build failed: ${reportErr}`);
      }

      return { messages: workMessages, estimatedTokens: finalSnap.totalTokens, systemPromptAddition };
    } catch (err) {
      logger.error(`[context-offload] assemble error: ${err}`);
      if (isTokenOverflowError(err)) stateManager._forceEmergencyNext = true;
      return { messages: workMessages, estimatedTokens: 0 };
    }
  }

  async compact(params: any) {
    const _compactStart = Date.now();
    const logger = this._logger;
    logger.debug?.(`[context-offload] >>> CE.compact CALLED: sessionKey=${params.sessionKey ?? "?"}`);
    let stateManager: OffloadStateManager | undefined = params._offloadManager;
    if (!stateManager && params.sessionKey) {
      try {
        const entry = await this._sessions.resolveIfAllowed(params.sessionKey, params.sessionId);
        if (entry) stateManager = entry.manager;
      } catch { /* ignore */ }
    }
    const pCfg = this._pCfg;
    logger.debug?.(`[context-offload] >>> compact START: params=${JSON.stringify(params ?? {}).slice(0, 500)}`);
    if (!stateManager) {
      logger.warn(`[context-offload] <<< compact SKIP: no session manager (${Date.now() - _compactStart}ms)`);
      return { ok: false, compacted: false, reason: "no_session_manager" };
    }
    try {
      // Try delegating to runtime's built-in compaction first
      let delegateFn: any;
      try {
        const { createRequire } = await import("node:module");
        const globalRequire = createRequire("/usr/local/lib/node_modules/openclaw/");
        const sdk = globalRequire("openclaw/plugin-sdk");
        delegateFn = sdk.delegateCompactionToRuntime;
        logger.debug?.(`[context-offload] compact: resolved via createRequire (global path)`);
      } catch (e1) {
        logger.debug?.(`[context-offload] compact: createRequire failed: ${e1}`);
        try {
          const paths = [
            "/usr/local/lib/node_modules/openclaw/dist/plugin-sdk/index.js",
            "/usr/lib/node_modules/openclaw/dist/plugin-sdk/index.js",
          ];
          for (const p of paths) {
            try {
              const sdk = await import(p);
              delegateFn = sdk.delegateCompactionToRuntime;
              logger.debug?.(`[context-offload] compact: resolved via absolute path: ${p}`);
              break;
            } catch (ep) {
              logger.debug?.(`[context-offload] compact: absolute path failed: ${p} → ${ep}`);
            }
          }
        } catch { /* ignore */ }
        if (!delegateFn) {
          try {
            const sdk = await import("openclaw/plugin-sdk" as any);
            delegateFn = sdk.delegateCompactionToRuntime;
            logger.debug?.(`[context-offload] compact: resolved via direct import`);
          } catch { /* ignore */ }
        }
      }

      if (typeof delegateFn === "function") {
        logger.debug?.(`[context-offload] compact: >>> delegateCompactionToRuntime START`);
        const result = await delegateFn(params);
        logger.debug?.(`[context-offload] <<< compact END (delegated) ${Date.now() - _compactStart}ms — compacted=${result.compacted}`);
        return result;
      }

      // Fallback: self-execute emergency compression when runtime delegation unavailable
      logger.info(`[context-offload] compact: delegateCompactionToRuntime unavailable, self-executing emergency compression`);
      const messages = params.messages;
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        logger.debug?.(`[context-offload] <<< compact END (no_messages) ${Date.now() - _compactStart}ms`);
        return { ok: true, compacted: false, reason: "no_messages" };
      }

      const contextWindow = this._getContextWindow();
      const budget = params.tokenBudget ? Math.min(params.tokenBudget, contextWindow) : contextWindow;
      const mildRatio = pCfg.mildOffloadRatio ?? PLUGIN_DEFAULTS.mildOffloadRatio;
      const targetTokens = Math.floor(budget * mildRatio);
      const systemTokensEstimate = stateManager.cachedSystemPromptTokens
        ?? stateManager.getEstimatedSystemOverhead()
        ?? Math.floor(budget * (pCfg.defaultSystemOverheadRatio ?? PLUGIN_DEFAULTS.defaultSystemOverheadRatio));

      const countTokens = createL3TokenCounter(pCfg, logger);
      logger.info(`[context-offload] compact: msgs=${messages.length}, target=${targetTokens}, msgTarget=${targetTokens - systemTokensEstimate}`);
      const emergencyResult = emergencyCompress(messages, targetTokens - systemTokensEstimate, countTokens, null, null, logger);

      if (emergencyResult.deletedToolCallIds.length > 0) {
        for (const id of emergencyResult.deletedToolCallIds) {
          stateManager.confirmedOffloadIds.add(id);
          stateManager.confirmedOffloadIds.add(normalizeToolCallIdForLookup(id));
          stateManager.deletedOffloadIds.add(id);
          stateManager.deletedOffloadIds.add(normalizeToolCallIdForLookup(id));
        }
        const statusUpdates = new Map<string, string | boolean>();
        for (const id of emergencyResult.deletedToolCallIds) statusUpdates.set(id, "deleted");
        markOffloadStatus(stateManager.ctx, statusUpdates).catch(() => {});
      }

      // Invalidate assemble boundary cache after compact modifies messages
      if (emergencyResult.deletedCount > 0) {
        stateManager._lastAggressiveBoundary = null;
      }

      logger.info(`[context-offload] <<< compact END (self_emergency) ${Date.now() - _compactStart}ms — deleted=${emergencyResult.deletedCount} msgs, remaining≈${emergencyResult.remainingTokens}+sys≈${systemTokensEstimate}`);
      return { ok: true, compacted: emergencyResult.deletedCount > 0, reason: "self_emergency", messages };
    } catch (err) {
      logger.error(`[context-offload] <<< compact ERROR: ${err} (${Date.now() - _compactStart}ms)`);
      return { ok: false, compacted: false, reason: String(err) };
    }
  }

  async afterTurn(_params: any) {
    const logger = this._logger;
    logger.debug?.(`[context-offload] >>> CE.afterTurn CALLED: sessionKey=${_params?.sessionKey ?? "?"}`);
    let stateManager: OffloadStateManager | undefined = _params?._offloadManager;
    if (!stateManager && _params?.sessionKey && !isInternalMemorySession(_params.sessionKey)) {
      try {
        const entry = this._sessions.get(_params.sessionKey);
        stateManager = entry?.manager;
      } catch { /* ignore */ }
    }
    if (!stateManager) return;
    try {
      // Flush remaining pending tool pairs — fire-and-forget to avoid blocking
      // the next turn's assemble(). L1 Lock guarantees data integrity; the next
      // judgeL15's pre_flush will pick up any pairs that haven't been flushed yet.
      const pendingCount = stateManager.getPendingCount();
      if (pendingCount > 0) {
        logger.debug?.(`[context-offload] afterTurn: fire-and-forget flushing ${pendingCount} remaining pending pairs`);
        this._flushL1(stateManager, "afterTurn_flush").then(async () => {
          try {
            const allEntries = await readAllOffloadEntries(stateManager!.ctx);
            const nullCount = allEntries.filter((e) => e.node_id === null).length;
            if (nullCount > 0) this._notifyL2NewNullEntries(nullCount);
          } catch { /* ignore */ }
        }).catch((err) => {
          logger.warn(`[context-offload] afterTurn: L1 flush failed: ${err}`);
        });
      }

      if (stateManager.isLoaded()) await stateManager.save();
    } catch { /* ignore */ }
  }

  async maintain(_params: any) {
    return { changed: false, bytesFreed: 0, rewrittenEntries: 0 };
  }

  async dispose() {
    this._logger.debug?.("[context-offload] dispose: cleaning up");
    this._disposeL15();
    this._clearL2Timeout();
    if (_reclaimTimer !== null) { clearTimeout(_reclaimTimer); _reclaimTimer = null; }
  }
}

// ─── Test-only exports (internal functions for unit testing) ────────────────
export const _testExports = {
  _isHeartbeatText,
  _extractMsgText,
  _normalizePromptForCompare,
  _extractLatestTurn,
  _extractRecentHistory,
  _buildL1RecentContext,
  _buildL15RecentContext,
  isInternalMemorySession,
  simpleHash,
  OffloadContextEngine,
};
