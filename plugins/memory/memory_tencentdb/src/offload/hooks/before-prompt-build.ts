/**
 * before_prompt_build hook handler.
 *
 * Three-phase context cleanup before llm_input:
 * 1. Fast-path re-apply: re-offload confirmed mild replacements + delete aggressive-deleted messages
 * 2. Token guard: if still above thresholds, run full L3 (Aggressive + Mild) inline
 * 3. MMD injection: injects active/history MMD into messages
 */
import { PLUGIN_DEFAULTS } from "../types.js";
import { readOffloadEntries, markOffloadStatus } from "../storage.js";
import { buildTiktokenContextSnapshot } from "../context-token-tracker.js";
import { traceOffloadDecision } from "../opik-tracer.js";
import { injectMmdIntoMessages, findHistoryMmdInsertionPoint } from "../mmd-injector.js";
import { createL3TokenCounter } from "../l3-token-counter.js";
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
} from "../l3-helpers.js";
import {
  compressByScoreCascade,
  aggressiveCompressUntilBelowThreshold,
  buildHistoryMmdInjection,
  removeExistingMmdInjections,
  emergencyCompress,
  EMERGENCY_MIN_MESSAGES_TO_KEEP,
  isTokenOverflowError,
  filterHeartbeatMessages,
  dumpMessagesSnapshot,
} from "./llm-input-l3.js";
import type { OffloadStateManager } from "../state-manager.js";
import type { PluginConfig, PluginLogger } from "../types.js";

export function createBeforePromptBuildHandler(
  stateManager: OffloadStateManager,
  logger: PluginLogger,
  getContextWindow: (() => number) | undefined,
  pluginConfig: Partial<PluginConfig> | undefined,
) {
  return async (event: any, _ctx: any) => {
    // Skip internal memory-pipeline sessions
    const _sk = stateManager.getLastSessionKey() ?? _ctx?.sessionKey;
    if (typeof _sk === "string" && /memory-.*-session-\d+/.test(_sk)) return;

    logger.debug?.(`[context-offload] before_prompt_build CALLED, msgs=${event?.messages?.length ?? "?"}`);
    try {
      const messages = event.messages;
      if (!messages || !Array.isArray(messages) || messages.length === 0) return;

      filterHeartbeatMessages(messages, logger);

      const sessionKey = stateManager.getLastSessionKey();
      const hasConfirmed = stateManager.confirmedOffloadIds && stateManager.confirmedOffloadIds.size > 0;
      const hasDeleted = stateManager.deletedOffloadIds && stateManager.deletedOffloadIds.size > 0;

      if (!hasConfirmed && !hasDeleted) {
        await injectMmdIntoMessages(messages, stateManager, logger, getContextWindow, pluginConfig, { waitForL15: true });
        return undefined;
      }

      // Phase 1: Fast-path
      const snapBefore = buildTiktokenContextSnapshot("before_prompt_pre", messages, null, null);
      const tokensBefore = snapBefore.totalTokens;

      const offloadEntries = await readOffloadEntries(stateManager.ctx);
      const offloadMap = new Map();
      populateOffloadLookupMap(offloadMap, offloadEntries);
      stateManager.setCachedOffloadMap(offloadMap);

      let fastReplaceApplied = 0;
      const indicesToDelete: number[] = [];
      const deletedToolCallIdsForMmd: string[] = [];

      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        const tid = extractToolCallId(msg);
        const tidNorm = tid ? normalizeToolCallIdForLookup(tid) : null;

        if (tid && hasDeleted && (stateManager.deletedOffloadIds.has(tid) || (tidNorm && stateManager.deletedOffloadIds.has(tidNorm)))) {
          indicesToDelete.push(i);
          if (isToolResultMessage(msg)) deletedToolCallIdsForMmd.push(tid);
          continue;
        }
        if (hasDeleted && isOnlyToolUseAssistant(msg)) {
          const tuIds = extractAllToolUseIds(msg);
          const allDeleted = tuIds.length > 0 && tuIds.every((id) =>
            stateManager.deletedOffloadIds.has(id) || stateManager.deletedOffloadIds.has(normalizeToolCallIdForLookup(id)));
          if (allDeleted) { indicesToDelete.push(i); continue; }
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
          if (entry && isToolResultMessage(msg)) {
            replaceWithSummary(msg, entry);
            msg._offloaded = true;
            fastReplaceApplied++;
          }
        }
        if (isOnlyToolUseAssistant(msg)) {
          const tuIds = extractAllToolUseIds(msg);
          const allConfirmed = tuIds.length > 0 && tuIds.every((id) =>
            stateManager.confirmedOffloadIds.has(id) || stateManager.confirmedOffloadIds.has(normalizeToolCallIdForLookup(id)));
          if (allConfirmed) {
            const tuEntries = tuIds.map((id) => getOffloadEntry(offloadMap, id)).filter(Boolean) as any[];
            if (tuEntries.length === tuIds.length) {
              replaceAssistantToolUseWithSummary(msg, tuEntries);
              msg._offloaded = true;
              fastReplaceApplied++;
            }
          }
        } else if (isAssistantMessageWithToolUse(msg)) {
          compressNonCurrentToolUseBlocks(msg, offloadMap, new Set(), stateManager.confirmedOffloadIds);
        }
      }

      if (indicesToDelete.length > 0) {
        for (let k = indicesToDelete.length - 1; k >= 0; k--) {
          messages.splice(indicesToDelete[k], 1);
        }
      }

      // Phase 2: Token guard
      const contextWindow = typeof getContextWindow === "function" ? getContextWindow() : PLUGIN_DEFAULTS.defaultContextWindow;
      const mildRatio = pluginConfig?.mildOffloadRatio ?? PLUGIN_DEFAULTS.mildOffloadRatio;
      const aggressiveRatio = pluginConfig?.aggressiveCompressRatio ?? PLUGIN_DEFAULTS.aggressiveCompressRatio;
      const mildThreshold = Math.floor(contextWindow * mildRatio);
      const aggressiveThreshold = Math.floor(contextWindow * aggressiveRatio);

      const snapGuard = buildTiktokenContextSnapshot("before_prompt_guard", messages, null, null);
      let workingTokens = snapGuard.totalTokens;

      if (workingTokens >= aggressiveThreshold) {
        const countTokens = createL3TokenCounter(pluginConfig, logger);
        const aggressiveDeleteRatio = (pluginConfig as any)?.aggressiveDeleteRatio ?? PLUGIN_DEFAULTS.aggressiveDeleteRatio;
        const currentTaskNodeIds = await getCurrentTaskNodeIds(stateManager);
        const _bpbAggStart = Date.now();
        const result = await aggressiveCompressUntilBelowThreshold(
          messages, offloadMap, currentTaskNodeIds, aggressiveDeleteRatio,
          stateManager, logger, aggressiveThreshold, countTokens, null, null,
        );
        workingTokens = result.remainingTokens;
        const _bpbAggDuration = Date.now() - _bpbAggStart;
        if (_bpbAggDuration > 10_000) {
          logger.warn(`[context-offload] L3(before_prompt_build) AGGRESSIVE SLOW: ${_bpbAggDuration}ms (rounds=${result.rounds}, deleted=${result.deletedCount}, remaining≈${workingTokens})`);
        }
        dumpMessagesSnapshot("bpb-after-aggressive", messages, logger);
        if (result.allDeletedToolCallIds.length > 0) {
          const statusUpdates = new Map<string, string | boolean>();
          for (const id of result.allDeletedToolCallIds) {
            statusUpdates.set(id, "deleted");
            statusUpdates.set(normalizeToolCallIdForLookup(id), "deleted");
            stateManager.confirmedOffloadIds.add(id);
            stateManager.deletedOffloadIds.add(id);
          }
          markOffloadStatus(stateManager.ctx, statusUpdates).catch((err: any) =>
            logger.error(`[context-offload] markOffloadStatus error: ${err}`));
          const mmdInjection = await buildHistoryMmdInjection(
            result.allDeletedToolCallIds, offloadMap, offloadEntries,
            stateManager, logger, countTokens, contextWindow, pluginConfig,
          );
          if (mmdInjection.injectedMessages.length > 0) {
            removeExistingMmdInjections(messages);
            const histInsertIdx = findHistoryMmdInsertionPoint(messages);
            messages.splice(histInsertIdx, 0, ...mmdInjection.injectedMessages);
            workingTokens += mmdInjection.totalMmdTokens;
            dumpMessagesSnapshot("bpb-after-aggressive-mmd-injection", messages, logger);
          }
        }
        // If aggressive stalled due to user message protection, force emergency
        if (result.stalledByUserMsg && workingTokens >= aggressiveThreshold) {
          logger.warn(`[context-offload] before_prompt_build AGGRESSIVE stalled, forcing emergency fallback`);
          stateManager._forceEmergencyNext = true;
        }
      }

      if (workingTokens >= mildThreshold) {
        const currentTaskNodeIds = await getCurrentTaskNodeIds(stateManager);
        const mildScanRatio = (pluginConfig as any)?.mildOffloadScanRatio ?? PLUGIN_DEFAULTS.mildOffloadScanRatio;
        const cascadeResult = compressByScoreCascade(messages, offloadMap, currentTaskNodeIds, mildScanRatio, logger);
        if (cascadeResult.replacedCount > 0) {
          for (const id of cascadeResult.replacedToolCallIds) {
            stateManager.confirmedOffloadIds.add(id);
          }
          const mildStatusUpdates = new Map<string, string | boolean>();
          for (const id of cascadeResult.replacedToolCallIds) {
            mildStatusUpdates.set(id, true);
          }
          markOffloadStatus(stateManager.ctx, mildStatusUpdates).catch((err: any) =>
            logger.error(`[context-offload] markOffloadStatus error: ${err}`));
        }
        dumpMessagesSnapshot("bpb-after-mild", messages, logger);
      }
      {
        const emergencyRatio = pluginConfig?.emergencyCompressRatio ?? PLUGIN_DEFAULTS.emergencyCompressRatio;
        const emergencyTargetRatio = pluginConfig?.emergencyTargetRatio ?? PLUGIN_DEFAULTS.emergencyTargetRatio;
        const emergencyThreshold = Math.floor(contextWindow * emergencyRatio);
        const emergencyTarget = Math.floor(contextWindow * emergencyTargetRatio);
        const preEmergencySnap = buildTiktokenContextSnapshot("before_prompt_pre_emergency", messages, null, null);
        workingTokens = preEmergencySnap.totalTokens;
        const forceEmergency = stateManager._forceEmergencyNext === true;
        if (forceEmergency) stateManager._forceEmergencyNext = false;
        if ((workingTokens >= emergencyThreshold || forceEmergency) && messages.length > EMERGENCY_MIN_MESSAGES_TO_KEEP) {
          const countTokensBpb = createL3TokenCounter(pluginConfig, logger);
          const _bpbEmStart = Date.now();
          const emergencyResult = emergencyCompress(messages, emergencyTarget, countTokensBpb, null, null, logger);
          workingTokens = emergencyResult.remainingTokens;
          const _bpbEmDuration = Date.now() - _bpbEmStart;
          if (_bpbEmDuration > 10_000) {
            logger.warn(`[context-offload] L3(before_prompt_build) EMERGENCY SLOW: ${_bpbEmDuration}ms (deleted=${emergencyResult.deletedCount}, remaining≈${workingTokens})`);
          }
          if (emergencyResult.deletedToolCallIds.length > 0) {
            const emergencyStatusUpdates = new Map<string, string | boolean>();
            for (const id of emergencyResult.deletedToolCallIds) {
              emergencyStatusUpdates.set(id, "deleted");
              stateManager.confirmedOffloadIds.add(id);
              stateManager.deletedOffloadIds.add(id);
            }
            markOffloadStatus(stateManager.ctx, emergencyStatusUpdates).catch((err: any) =>
              logger.error(`[context-offload] markOffloadStatus error: ${err}`));
          }
          dumpMessagesSnapshot("bpb-after-emergency", messages, logger);
        }
      }

      // Phase 3: MMD Injection
      await injectMmdIntoMessages(messages, stateManager, logger, getContextWindow, pluginConfig, { waitForL15: true });

      traceOffloadDecision({
        sessionKey: stateManager.getLastSessionKey(),
        stage: "L3.before_prompt_build.completed",
        input: {
          phase: "before_prompt_build",
          confirmedOffloadIds: stateManager.confirmedOffloadIds.size,
          deletedOffloadIds: stateManager.deletedOffloadIds.size,
        },
        output: {
          messagesAfter: messages.length,
        },
        logger,
      });

      return undefined;
    } catch (err) {
      logger.error(`[context-offload] before_prompt_build error: ${err}`);
      if (isTokenOverflowError(err)) {
        stateManager._forceEmergencyNext = true;
      }
      return;
    }
  };
}
