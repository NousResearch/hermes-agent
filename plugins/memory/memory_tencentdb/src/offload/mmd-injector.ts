/**
 * Unified MMD injector.
 *
 * Maintains a single marked message in event.messages containing the active
 * MMD (+ history MMDs). Used by both before_prompt_build (full inject after
 * L1.5 judgment) and after_tool_call (incremental update when L2 refreshes
 * the MMD file during the tool loop).
 *
 * The marker property `_mmdContextMessage` is used to locate the message for
 * replacement. L3 compression must skip messages carrying this marker.
 */
import { readMmd, listMmds } from "./storage.js";
import { PLUGIN_DEFAULTS, type PluginConfig, type PluginLogger } from "./types.js";
import { createL3TokenCounter } from "./l3-token-counter.js";
import { traceOffloadDecision } from "./opik-tracer.js";
import { isToolResultMessage, isAssistantMessageWithToolUse } from "./l3-helpers.js";
import type { OffloadStateManager } from "./state-manager.js";

/** Marker property on the injected message object. */
export const MMD_MESSAGE_MARKER = "_mmdContextMessage";

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Full inject — called from assemble / before_prompt_build (every user-message round)
 * and from llm_input (every LLM call).
 *
 * Only injects the ACTIVE MMD (determined by L1.5).
 * History MMDs are NOT injected here — they are only injected by L3 aggressive
 * compression (buildHistoryMmdInjection) after messages are deleted, as a
 * replacement for lost conversation context.
 */
export async function injectMmdIntoMessages(
  messages: any[],
  stateManager: OffloadStateManager,
  logger: PluginLogger,
  getContextWindow: (() => number) | undefined,
  pluginConfig: Partial<PluginConfig> | undefined,
  options?: { waitForL15?: boolean },
): Promise<{ mmdTokens: number }> {
  // When waitForL15 is set (assemble path), skip injection entirely if L1.5 hasn't settled yet.
  // This preserves any previously injected MMD messages without removing or replacing them.
  if (options?.waitForL15 && !stateManager.l15Settled) {
    logger.debug?.(
      `[context-offload] mmd-injector inject: SKIPPED — L1.5 not settled yet (waitForL15=true), msgs=${messages.length}`,
    );
    return { mmdTokens: stateManager.lastMmdInjectedTokens };
  }

  const injReady = stateManager.isMmdInjectionReady();
  const actFile = stateManager.getActiveMmdFile();
  logger.debug?.(
    `[context-offload] mmd-injector inject: injectionReady=${injReady}, activeMmdFile=${actFile ?? "null"}, msgs=${messages.length}`,
  );
  if (!injReady) {
    removeMmdMessages(messages);
    stateManager.lastMmdInjectedTokens = 0;
    return { mmdTokens: 0 };
  }

  const contextWindow =
    typeof getContextWindow === "function"
      ? getContextWindow()
      : PLUGIN_DEFAULTS.defaultContextWindow;
  const mmdMaxTokenRatio =
    pluginConfig?.mmdMaxTokenRatio ?? PLUGIN_DEFAULTS.mmdMaxTokenRatio;
  const countTokens = createL3TokenCounter(pluginConfig, logger);

  const activeMmdText = await buildActiveMmdText(stateManager, logger);
  logger.debug?.(
    `[context-offload] mmd-injector inject: activeMmdText=${activeMmdText ? `${activeMmdText.length} chars` : "null"}, contextWindow=${contextWindow}`,
  );
  removeMmdMessages(messages);

  let totalMmdTokens = 0;

  if (activeMmdText) {
    const activeMsg: any = {
      role: "user",
      content: [{ type: "text", text: activeMmdText }],
      [MMD_MESSAGE_MARKER]: "active",
    };
    const insertIdx = findActiveMmdInsertionPoint(messages);
    messages.splice(insertIdx, 0, activeMsg);
    totalMmdTokens += countTokens(activeMmdText);
  }

  stateManager.lastMmdInjectedTokens = totalMmdTokens;

  const activeMmd = stateManager.getActiveMmdFile();
  logger.debug?.(
    `[context-offload] mmd-injector: injected active MMD into messages (${totalMmdTokens} tokens, file=${activeMmd})`,
  );

  // Summary after active MMD injection (was full dump, now aggregated)
  if (totalMmdTokens > 0) {
    const mmdCount = messages.filter((m: any) => m[MMD_MESSAGE_MARKER] === "active" || m._mmdInjection).length;
    const offloadedCount = messages.filter((m: any) => m._offloaded).length;
    logger.debug?.(`[context-offload] POST-ACTIVE-MMD-INJECT: ${messages.length} msgs, mmd=${mmdCount}, offloaded=${offloadedCount}`);
  }

  traceOffloadDecision({
    sessionKey: stateManager.getLastSessionKey(),
    stage: "mmd-injector.inject",
    input: {
      activeMmd,
      mmdInjectionReady: true,
      contextWindow,
      mmdMaxTokenRatio,
    },
    output: {
      result: `MMD 注入 messages：${totalMmdTokens} tokens (active only)`,
      mmdTokens: totalMmdTokens,
      hasActive: !!activeMmdText,
      hasHistory: false,
      mmdTokenBudget: Math.floor(contextWindow * mmdMaxTokenRatio),
    },
    logger,
  });

  return { mmdTokens: totalMmdTokens };
}

/**
 * Incremental update — called from after_tool_call (every tool-loop iteration).
 */
export async function maybeUpdateMmdInMessages(
  messages: any[],
  stateManager: OffloadStateManager,
  logger: PluginLogger,
  getContextWindow: (() => number) | undefined,
  pluginConfig: Partial<PluginConfig> | undefined,
): Promise<boolean> {
  const injectionReady = stateManager.isMmdInjectionReady();
  const activeMmdFile = stateManager.getActiveMmdFile();
  logger.debug?.(
    `[context-offload] mmd-injector maybeUpdate: injectionReady=${injectionReady}, activeMmdFile=${activeMmdFile ?? "null"}, msgs=${messages.length}`,
  );
  if (!injectionReady) return false;
  if (!activeMmdFile) return false;

  let mmdContent: string | null;
  try {
    mmdContent = await readMmd(stateManager.ctx, activeMmdFile);
    logger.debug?.(
      `[context-offload] mmd-injector maybeUpdate: readMmd result=${mmdContent ? `${mmdContent.length} chars` : "null"}`,
    );
  } catch (e) {
    logger.debug?.(`[context-offload] mmd-injector maybeUpdate: readMmd error=${e}`);
    return false;
  }
  if (!mmdContent) return false;

  const newFp = computeFingerprint(mmdContent);
  const lastFp = stateManager.getInjectedMmdVersion(activeMmdFile);
  if (newFp === lastFp) return false;

  logger.debug?.(
    `[context-offload] mmd-injector: MMD updated (${activeMmdFile}), refreshing in-loop`,
  );
  await injectMmdIntoMessages(
    messages,
    stateManager,
    logger,
    getContextWindow,
    pluginConfig,
  );
  return true;
}

// ─── Insertion point helpers (exported for after-tool-call & llm-input-l3) ──

function findLatestUserMessageIndex(messages: any[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg[MMD_MESSAGE_MARKER]) continue;
    if (msg._mmdInjection) continue;
    const role = msg.role ?? msg.message?.role ?? msg.type;
    if (role === "user") return i;
  }
  return -1;
}

/**
 * Find the best insertion point for the active MMD message.
 *
 * Strategy: insert AFTER the latest user message (in the second half of the
 * conversation), so the MMD sits between the user's question and the ongoing
 * tool loop — not at position 0 which pollutes the oldest context.
 *
 * Fallback: if the latest user message is in the first half (unlikely during
 * active tool loops), insert at the start of the trailing tool-result/assistant
 * block, clamped to within 30 messages from the tail.
 *
 * IMPORTANT: The insertion point must NOT split a tool_call / tool_result pair.
 * If the candidate position is between an assistant message containing tool_use
 * and its corresponding tool_result(s), shift backwards to before the assistant
 * message so the pair stays intact.
 */
export function findActiveMmdInsertionPoint(messages: any[]): number {
  if (messages.length <= 2) return 0;

  const halfIdx = Math.floor(messages.length / 2);
  const latestUserIdx = findLatestUserMessageIndex(messages);
  let insertIdx: number;
  if (latestUserIdx >= halfIdx) {
    insertIdx = latestUserIdx + 1;
  } else {
    let loopStart = messages.length;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg[MMD_MESSAGE_MARKER]) continue;
      if (msg._mmdInjection) continue;
      const role = msg.role ?? msg.message?.role ?? msg.type;
      if (role === "toolResult" || role === "tool" || role === "assistant") {
        loopStart = i;
      } else {
        break;
      }
    }

    const maxDistFromTail = 30;
    const minInsertIdx = Math.max(0, messages.length - maxDistFromTail);
    insertIdx = Math.max(loopStart, minInsertIdx);
    insertIdx = Math.min(insertIdx, Math.max(0, messages.length - 1));
  }

  // Guard: don't insert between an assistant tool_use message and its tool_result(s).
  // If the message at insertIdx is a tool_result, walk backwards past the tool_result
  // cluster and the preceding assistant tool_use message.
  insertIdx = adjustForToolCallPair(messages, insertIdx);

  return insertIdx;
}

/**
 * Adjusts an insertion index so it does not land between an assistant message
 * containing tool_use blocks and the subsequent tool_result messages.
 *
 * Walk backwards: if we see tool_result messages at `idx`, keep going back;
 * if we then land on an assistant message with tool_use, step before it too.
 */
function adjustForToolCallPair(messages: any[], idx: number): number {
  if (idx <= 0 || idx >= messages.length) return idx;

  // Check if the message AT idx (or the preceding context) forms a tool pair boundary.
  // Case 1: idx points at a tool_result → we're inside a tool pair, walk back.
  let cur = idx;
  while (cur > 0 && cur < messages.length) {
    const msg = messages[cur];
    if (msg[MMD_MESSAGE_MARKER] || msg._mmdInjection) { cur--; continue; }
    if (!isToolResultMessage(msg)) break;
    cur--;
  }

  // After skipping tool_results, check if the message at `cur` is an assistant with tool_use.
  // If so, we must insert BEFORE this assistant message to keep the pair intact.
  if (cur >= 0 && cur < messages.length) {
    const msg = messages[cur];
    if (!msg[MMD_MESSAGE_MARKER] && !msg._mmdInjection && isAssistantMessageWithToolUse(msg)) {
      return cur;
    }
  }

  // Also check the message just BEFORE idx — if it's an assistant with tool_use,
  // and idx's message is a tool_result, we already handled above. But if idx-1 is
  // assistant with tool_use and idx is tool_result, the while-loop above would
  // have caught it. This covers the edge case where idx is right after an assistant
  // tool_use (before any tool_result arrives yet).
  if (idx > 0 && idx < messages.length) {
    const prevMsg = messages[idx - 1];
    if (!prevMsg[MMD_MESSAGE_MARKER] && !prevMsg._mmdInjection && isAssistantMessageWithToolUse(prevMsg)) {
      const curMsg = messages[idx];
      if (isToolResultMessage(curMsg)) {
        return idx - 1;
      }
    }
  }

  // If we moved backward, return the adjusted position; otherwise return original.
  return cur < idx ? cur : idx;
}

/**
 * Find insertion point for history MMD messages (injected after AGGRESSIVE deletion).
 *
 * Strategy: insert BEFORE the active MMD (if present) or at the same position
 * where the active MMD would go. History context should precede active context
 * so the LLM reads chronologically: history → active → recent tool loop.
 *
 * Unlike active MMD, history MMD should NOT go to index 0 — it should sit in
 * the middle of the conversation, just before the active task context.
 */
export function findHistoryMmdInsertionPoint(messages: any[]): number {
  // If there's an existing active MMD, insert just before it
  for (let i = 0; i < messages.length; i++) {
    if (messages[i][MMD_MESSAGE_MARKER] === "active") return i;
  }
  // No active MMD — use the same heuristic as active MMD insertion
  return findActiveMmdInsertionPoint(messages);
}

function removeMmdMessages(messages: any[]): void {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i][MMD_MESSAGE_MARKER]) {
      messages.splice(i, 1);
    }
  }
}

async function buildActiveMmdText(
  stateManager: OffloadStateManager,
  logger: PluginLogger,
): Promise<string | null> {
  const activeMmdFile = stateManager.getActiveMmdFile();
  if (!activeMmdFile) return null;
  return await buildActiveMmdBlock(activeMmdFile, stateManager, logger);
}

async function buildActiveMmdBlock(
  activeMmdFile: string,
  stateManager: OffloadStateManager,
  logger: PluginLogger,
): Promise<string | null> {
  try {
    const mmdContent = await readMmd(stateManager.ctx, activeMmdFile);
    if (!mmdContent) return null;
    stateManager.setInjectedMmdVersion(
      activeMmdFile,
      computeFingerprint(mmdContent),
    );
    const metaMatch = mmdContent.match(/^%%\{\s*(.*?)\s*\}%%/);
    let taskGoal = "";
    if (metaMatch) {
      try {
        const meta = JSON.parse(`{${metaMatch[1]}}`);
        taskGoal = meta.taskGoal || "";
      } catch {
        /* ignore */
      }
    }
    const nodePattern = /\b(\d+-N\d+|N\d+)\b/g;
    const nodeIds: string[] = [];
    let match: RegExpExecArray | null;
    while ((match = nodePattern.exec(mmdContent)) !== null) {
      if (!nodeIds.includes(match[1])) nodeIds.push(match[1]);
    }
    return [
      `<current_task_context>`,
      `【当前活跃任务的mermaid流程图】这是你最近正在执行的任务的阶段性记录（此条下方的tool use未被汇总，进程可能有延迟，仅供参考）。`,
      taskGoal ? `**任务目标:** ${taskGoal}` : "",
      `**任务文件:** ${activeMmdFile}`,
      nodeIds.length > 0
        ? `**节点索引:** 可通过 node_id 在 offload.{sessionid}.jsonl 中查找对应的工具调用记录。如需查看某个节点对应的原始工具调用与完整结果，请在 offload.{sessionid}.jsonl 中找到对应条目的 result_ref 并读取该文件。`
        : "",
      "```mermaid",
      mmdContent,
      "```",
      `标记为 "doing" 的节点是近期焦点（注：可能有延迟，下方的tool use未被统计，仅供参考），"done" 的已完成。请参考此保持方向感，避免重复已完成的工作。`,
      `</current_task_context>`,
    ]
      .filter((line) => line !== "")
      .join("\n");
  } catch (err) {
    logger.error(
      `[context-offload] mmd-injector: Error building active MMD block: ${err}`,
    );
    return null;
  }
}

function computeFingerprint(content: string): string {
  return `${content.length}:${content.slice(0, 64)}`;
}
