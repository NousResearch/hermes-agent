/**
 * L3 Compressor — pure functions for mild, aggressive, and emergency compression.
 */
import type { OffloadEntry } from "../types.js";
import type { Message } from "./helpers.js";
import {
  extractToolResultId,
  isToolResultMessage,
  isMmdMessage,
  replaceWithSummary,
  buildOffloadMap,
  extractAllToolUseIds,
  isAssistantWithToolUse,
} from "./helpers.js";
import { getEncoding, type Tiktoken } from "js-tiktoken";

// ─── Tiktoken Precise Token Counter ─────────────────────────────────────────

let _encoder: Tiktoken | null = null;
function getEncoder(): Tiktoken {
  if (!_encoder) _encoder = getEncoding("o200k_base");
  return _encoder;
}

/** JSON replacer: skip large binary-like fields to speed up serialization. */
function jsonReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "string" && value.length > 50000) {
    return value.slice(0, 50000) + "...[truncated]";
  }
  return value;
}

/**
 * Extract the LLM-visible portion of a message (role + content only).
 * This matches what the LLM provider actually tokenizes — excludes metadata
 * like model, usage, timestamp, api, provider, etc.
 */
function extractLlmVisibleText(msg: Message): string {
  const role = msg.role ?? (msg as any).message?.role ?? "";
  const rawContent = msg.content ?? (msg as any).message?.content ?? "";

  let contentStr: string;
  if (typeof rawContent === "string") {
    contentStr = rawContent;
  } else if (Array.isArray(rawContent)) {
    // Flatten content blocks: text, tool_use arguments, tool_result content
    const parts: string[] = [];
    for (const block of rawContent) {
      if (typeof block === "string") {
        parts.push(block);
      } else if (block?.type === "text" && typeof block.text === "string") {
        parts.push(block.text);
      } else if (block?.type === "tool_use" || block?.type === "toolCall") {
        // Include tool name + arguments (they count toward tokens)
        parts.push(block.name ?? block.toolName ?? "");
        if (block.arguments) parts.push(typeof block.arguments === "string" ? block.arguments : JSON.stringify(block.arguments, jsonReplacer));
        if (block.input) parts.push(typeof block.input === "string" ? block.input : JSON.stringify(block.input, jsonReplacer));
      } else if (block?.type === "tool_result") {
        if (typeof block.content === "string") parts.push(block.content);
        else if (block.content) parts.push(JSON.stringify(block.content, jsonReplacer));
      } else {
        // Unknown block: serialize it
        parts.push(JSON.stringify(block, jsonReplacer));
      }
    }
    contentStr = parts.join("\n");
  } else {
    contentStr = JSON.stringify(rawContent, jsonReplacer);
  }

  // role overhead: ~4 tokens per message (OpenAI format overhead)
  // OpenAI format: tool_calls as a separate field (not in content blocks)
  if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
    const tcParts: string[] = [];
    for (const tc of msg.tool_calls) {
      const fn = tc.function ?? tc;
      if (fn.name) tcParts.push(fn.name);
      if (fn.arguments) tcParts.push(typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments, jsonReplacer));
    }
    if (tcParts.length > 0) {
      contentStr += "\n" + tcParts.join("\n");
    }
  }

  return `${role}\n${contentStr}`;
}

/** Precise token count for a message using tiktoken BPE encoding.
 *  Only counts LLM-visible content (role + content), not metadata. */
export function preciseMessageTokens(msg: Message): number {
  try {
    const text = extractLlmVisibleText(msg);
    return getEncoder().encode(text).length + 4; // +4 for message framing overhead
  } catch {
    // Fallback: rough estimate
    const text = JSON.stringify(msg);
    return Math.ceil(text.length / 4);
  }
}

// ─── Constants ───────────────────────────────────────────────────────────────

export const MILD_SCAN_RATIO = 0.7;
export const MILD_SCORE_FLOOR = 4;
export const AGGRESSIVE_MIN_KEEP = 10;
export const EMERGENCY_MIN_KEEP = 10;

// ─── Level Resolution ────────────────────────────────────────────────────────

export type CompactionLevel = "fastpath" | "mild" | "aggressive" | "emergency";

/**
 * Determine compression level based on token usage ratio.
 */
export function resolveLevel(
  ratio: number,
  config?: { mildRatio?: number; aggressiveRatio?: number; emergencyRatio?: number },
): CompactionLevel {
  const mild = config?.mildRatio ?? 0.5;
  const aggressive = config?.aggressiveRatio ?? 0.85;
  const emergency = config?.emergencyRatio ?? 0.95;

  if (ratio >= emergency) return "emergency";
  if (ratio >= aggressive) return "aggressive";
  if (ratio >= mild) return "mild";
  return "fastpath";
}

// ─── Mild Compression ────────────────────────────────────────────────────────

export interface MildResult {
  replacedCount: number;
  confirmedIds: string[];
}

/**
 * Mild compression: replace tool_result content with summary based on score cascade.
 * Only processes messages in the first `scanRatio` of the array.
 * Mutates messages in place.
 */
export function mildCompress(
  messages: Message[],
  entries: OffloadEntry[],
  scanRatio: number = MILD_SCAN_RATIO,
  scoreFloor: number = MILD_SCORE_FLOOR,
): MildResult {
  const offloadMap = buildOffloadMap(entries);
  const confirmedIds: string[] = [];
  const scanEnd = Math.floor(messages.length * scanRatio);

  // Collect candidates
  const candidates: Array<{ idx: number; msg: Message; entry: OffloadEntry; tid: string }> = [];
  for (let i = 0; i < scanEnd; i++) {
    const msg = messages[i];
    if (msg._offloaded || isMmdMessage(msg)) continue;
    const tid = extractToolResultId(msg);
    if (!tid) continue;
    if (!isToolResultMessage(msg)) continue;
    const entry = offloadMap.get(tid);
    if (!entry) continue;
    candidates.push({ idx: i, msg, entry, tid });
  }

  // Sort by score descending (higher score = more suitable for replacement)
  candidates.sort((a, b) => b.entry.score - a.entry.score);

  // Replace from highest score down to floor
  let replacedCount = 0;
  for (const c of candidates) {
    if (c.entry.score < scoreFloor) break;
    // Skip if summary would be larger than original content
    const originalLen = getTextContent(c.msg).length;
    const summaryLen = (c.entry.summary ?? "").length + 50; // +50 for "[Offloaded...]" prefix
    if (summaryLen >= originalLen) continue;
    replaceWithSummary(c.msg, c.entry);
    confirmedIds.push(c.tid);
    replacedCount++;
  }

  return { replacedCount, confirmedIds };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Check if a message is a system-reminder (injected by Claude Code as role:user). */
function isSystemReminder(msg: Message): boolean {
  const content = msg.content ?? msg.message?.content ?? "";
  if (typeof content === "string") {
    return content.trimStart().startsWith("<system-reminder>");
  }
  if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block === "string" && block.trimStart().startsWith("<system-reminder>")) return true;
      if (block?.type === "text" && typeof block.text === "string" && block.text.trimStart().startsWith("<system-reminder>")) return true;
    }
  }
  return false;
}

/** Max characters to keep when truncating a large tool_result. */
export const TOOL_RESULT_TRUNCATE_CHARS = 2000;

/**
 * Find index of the last user message (excluding MMD injections).
 * Returns -1 if no user message found.
 */
export function findLastUserMessageIndex(messages: Message[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
    if (role === "user" && !isMmdMessage(messages[i])) return i;
  }
  return -1;
}

/**
 * In the protected tail zone (from protectedTailIdx onward), find tool_result messages
 * and truncate the one with the largest content to TOOL_RESULT_TRUNCATE_CHARS.
 * Repeats until tokensToFree <= 0 or no more truncatable messages.
 * Mutates messages and tokenArray in place. Returns tokens freed by truncation.
 */
export function truncateTailToolResults(
  messages: Message[],
  tokenArray: number[],
  protectedTailIdx: number,
  tokensToFree: number,
  estimateTokensFn: (text: string) => number,
  truncateChars: number = TOOL_RESULT_TRUNCATE_CHARS,
): number {
  let totalFreed = 0;

  while (tokensToFree > 0) {
    // Find the largest tool_result in protected zone
    let maxIdx = -1;
    let maxLen = 0;

    for (let i = protectedTailIdx; i < messages.length; i++) {
      if (!isToolResultMessage(messages[i])) continue;
      const content = getTextContent(messages[i]);
      if (content.length <= truncateChars) continue; // already small enough
      if (content.length > maxLen) {
        maxLen = content.length;
        maxIdx = i;
      }
    }

    if (maxIdx === -1) break; // nothing left to truncate

    // Truncate
    const msg = messages[maxIdx];
    const oldContent = getTextContent(msg);
    const truncated = oldContent.slice(0, truncateChars) + `\n\n[... content truncated, only first ${truncateChars} characters retained ...]`;
    setTextContent(msg, truncated);

    // Recalculate token for this message
    const newTokens = estimateTokensFn(truncated);
    const freed = tokenArray[maxIdx] - newTokens;
    tokenArray[maxIdx] = newTokens;
    tokensToFree -= freed;
    totalFreed += freed;
  }

  return totalFreed;
}

/**
 * Truncate oversized tool_result content in messages that are NOT marked for deletion.
 * Used when min-keep prevents further message deletion but tokens still exceed budget.
 * Finds the largest tool_result content first and truncates to TOOL_RESULT_TRUNCATE_CHARS.
 * Mutates messages and tokenArray in place. Returns total tokens freed.
 */
function truncateRemainingToolResults(
  messages: Message[],
  tokenArray: number[],
  deleteIndices: Set<number>,
  tokensToFree: number,
  usePrecise: boolean,
  truncateChars: number = TOOL_RESULT_TRUNCATE_CHARS,
): number {
  let totalFreed = 0;

  while (tokensToFree > 0) {
    // Find the largest tool_result content among non-deleted messages
    let maxIdx = -1;
    let maxLen = 0;

    for (let i = 0; i < messages.length; i++) {
      if (deleteIndices.has(i)) continue;
      if (!isToolResultMessage(messages[i])) continue;
      const content = getTextContent(messages[i]);
      if (content.length <= truncateChars) continue;
      if (content.length > maxLen) {
        maxLen = content.length;
        maxIdx = i;
      }
    }

    if (maxIdx === -1) break;

    // Truncate
    const oldContent = getTextContent(messages[maxIdx]);
    const truncated = oldContent.slice(0, truncateChars) +
      `\n\n[... content truncated, only first ${truncateChars} characters retained ...]`;
    setTextContent(messages[maxIdx], truncated);

    // Recalculate token
    const oldTokens = tokenArray[maxIdx];
    const newTokens = usePrecise
      ? preciseMessageTokens(messages[maxIdx])
      : Math.max(1, Math.ceil(truncated.length / 4));
    tokenArray[maxIdx] = newTokens;
    const freed = oldTokens - newTokens;
    tokensToFree -= freed;
    totalFreed += freed;
  }

  return totalFreed;
}

/**
 * Extract text content from a message (handles various formats).
 * Supports Anthropic content blocks: tool_result (.content), tool_use (.input), text (.text).
 */
function getTextContent(msg: Message): string {
  const content = msg.content ?? msg.message?.content ?? "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((b: any) => {
      if (typeof b === "string") return b;
      if (b?.type === "text") return b.text ?? "";
      if (b?.type === "tool_result") {
        return typeof b.content === "string" ? b.content : JSON.stringify(b.content ?? "");
      }
      if (b?.type === "tool_use") {
        const input = typeof b.input === "string" ? b.input : JSON.stringify(b.input ?? {});
        return `${b.name ?? ""}:${input}`;
      }
      return b?.text ?? JSON.stringify(b ?? "");
    }).join("");
  }
  return JSON.stringify(content);
}

/**
 * Set text content on a message (handles wrapped vs direct format).
 * For Anthropic user messages with tool_result blocks, updates the block's .content in-place.
 */
function setTextContent(msg: Message, text: string): void {
  if (msg.type === "message" && msg.message) {
    msg.message.content = text;
  } else if (msg.role === "user" && Array.isArray(msg.content)) {
    // Anthropic: update tool_result block content in-place to preserve structure
    for (const block of msg.content) {
      if (block?.type === "tool_result") {
        block.content = text;
        return;
      }
    }
    msg.content = text;
  } else {
    msg.content = text;
  }
}

export interface ProtectedTailResult {
  freedTokens: number;
  deletedCount: number;
}

/**
 * Compress the protected tail zone (last user message and after).
 * Strategy (ordered by priority):
 *   1. Truncate oversized tool_results (content > TOOL_RESULT_TRUNCATE_CHARS) from largest first
 *   2. If still above target, delete oldest tool pairs in the tail (assistant + its tool_results)
 *      while preserving the last user message itself and at least the most recent tool pair
 *
 * Mutates messages and tokenArray in place. Appends deleted IDs to deletedIds.
 */
export function compressProtectedTail(
  messages: Message[],
  tokenArray: number[],
  targetTokens: number,
  currentTokens: number,
  deletedIds: string[],
): ProtectedTailResult {
  let tokensToFree = currentTokens - targetTokens;
  let totalFreed = 0;
  let deletedCount = 0;

  if (tokensToFree <= 0) return { freedTokens: 0, deletedCount: 0 };

  const protectedIdx = findLastUserMessageIndex(messages);
  const tailStart = protectedIdx >= 0 ? protectedIdx : 0;

  // Phase 1: Delete tool pairs in the tail zone (fast batch deletion)
  // A "tool pair" = assistant(with tool_use) + consecutive tool_result(s) after it
  const tailPairs = collectToolPairsInRange(messages, tailStart, messages.length);

  const hasNonToolAssistant = messages.slice(tailStart).some((m) => {
    const r = m.role ?? m.message?.role ?? m.type;
    return r === "assistant" && !isAssistantWithToolUse(m);
  });
  // Reserve the last pair for fine-grained compression (unless there's a text assistant after it)
  const reserveLastPair = !hasNonToolAssistant;
  const fastDeleteLimit = reserveLastPair
    ? Math.max(0, tailPairs.length - 1)
    : tailPairs.length;

  // Phase 1: Fast batch deletion of entire tool pairs (oldest first)
  const pairDeleteIndices: number[] = [];
  for (let p = 0; p < fastDeleteLimit && tokensToFree > 0; p++) {
    for (const idx of tailPairs[p].indices) {
      pairDeleteIndices.push(idx);
      tokensToFree -= tokenArray[idx];
      totalFreed += tokenArray[idx];
    }
  }

  if (tokensToFree <= 0 || tailPairs.length === 0) {
    // Fast path sufficient: batch delete and return
    pairDeleteIndices.sort((a, b) => b - a);
    for (const idx of pairDeleteIndices) {
      const tid = extractToolResultId(messages[idx]);
      if (tid) deletedIds.push(tid);
      messages.splice(idx, 1);
      tokenArray.splice(idx, 1);
      deletedCount++;
    }
    if (tokensToFree <= 0) return { freedTokens: totalFreed, deletedCount };
  } else {
    // Phase 2: Fine-grained compression on the last remaining pair(s)
    // Only truncate when we've exhausted fast deletion and still need more space

    // Step 2a: Truncate oversized tool_results in remaining pairs
    const estimateText = (text: string) => Math.max(1, Math.ceil(text.length / 4));
    const remainingStart = tailPairs.length > 0 ? tailPairs[fastDeleteLimit]?.indices[0] ?? tailStart : tailStart;
    const truncFreed = truncateTailToolResults(messages, tokenArray, remainingStart, tokensToFree, estimateText);
    tokensToFree -= truncFreed;
    totalFreed += truncFreed;

    if (tokensToFree <= 0) {
      // Truncation was enough — still batch-delete the fast pairs
      pairDeleteIndices.sort((a, b) => b - a);
      for (const idx of pairDeleteIndices) {
        const tid = extractToolResultId(messages[idx]);
        if (tid) deletedIds.push(tid);
        messages.splice(idx, 1);
        tokenArray.splice(idx, 1);
        deletedCount++;
      }
      return { freedTokens: totalFreed, deletedCount };
    }

    // Step 2b: Truncate tool_call arguments in remaining assistant messages
    const TOOL_CALL_ARG_TRUNCATE = 200;
    for (let i = tailStart; i < messages.length && tokensToFree > 0; i++) {
      if (!isAssistantWithToolUse(messages[i])) continue;
      if (pairDeleteIndices.includes(i)) continue;
      let msgFreed = 0;

      // OpenAI format: tool_calls array
      const toolCalls = messages[i].tool_calls as any[] | undefined;
      if (toolCalls && Array.isArray(toolCalls)) {
        for (const tc of toolCalls) {
          const args = tc.function?.arguments;
          if (typeof args === "string" && args.length > TOOL_CALL_ARG_TRUNCATE) {
            const truncated = args.slice(0, TOOL_CALL_ARG_TRUNCATE) + "...[truncated]";
            msgFreed += Math.ceil((args.length - truncated.length) / 4);
            tc.function.arguments = truncated;
          }
        }
      }

      // Anthropic format: content array with tool_use blocks
      const content = messages[i].type === "message" ? messages[i].message?.content : messages[i].content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block?.type === "tool_use" && block.input) {
            const inputStr = typeof block.input === "string" ? block.input : JSON.stringify(block.input);
            if (inputStr.length > TOOL_CALL_ARG_TRUNCATE) {
              const truncated = inputStr.slice(0, TOOL_CALL_ARG_TRUNCATE) + "...[truncated]";
              msgFreed += Math.ceil((inputStr.length - truncated.length) / 4);
              block.input = truncated;
            }
          }
        }
      }

      if (msgFreed > 0) {
        const newTokens = Math.max(1, tokenArray[i] - msgFreed);
        const freed = tokenArray[i] - newTokens;
        tokenArray[i] = newTokens;
        tokensToFree -= freed;
        totalFreed += freed;
      }
    }

    // Step 2c: Delete individual tool_results (largest first) from remaining pairs
    if (tokensToFree > 0) {
      const remainingPairStart = fastDeleteLimit;
      for (let p = remainingPairStart; p < tailPairs.length && tokensToFree > 0; p++) {
        const pair = tailPairs[p];
        const toolResultIndices = pair.indices.slice(1)
          .sort((a, b) => tokenArray[b] - tokenArray[a]);

        for (const idx of toolResultIndices) {
          if (tokensToFree <= 0) break;
          pairDeleteIndices.push(idx);
          tokensToFree -= tokenArray[idx];
          totalFreed += tokenArray[idx];
        }
        // If ALL tool_results in this pair were deleted, also delete the assistant
        const allResultsDeleted = pair.indices.slice(1).every((idx) => pairDeleteIndices.includes(idx));
        if (allResultsDeleted) {
          const assistantIdx = pair.indices[0];
          pairDeleteIndices.push(assistantIdx);
          tokensToFree -= tokenArray[assistantIdx];
          totalFreed += tokenArray[assistantIdx];
        }
      }
    }

    // Batch delete all marked indices
    pairDeleteIndices.sort((a, b) => b - a);
    for (const idx of pairDeleteIndices) {
      const tid = extractToolResultId(messages[idx]);
      if (tid) deletedIds.push(tid);
      messages.splice(idx, 1);
      tokenArray.splice(idx, 1);
      deletedCount++;
    }
  }

  // Phase 3: Orphan protection — strip dangling tool_use from remaining assistants
  const allDeletedIds = new Set(deletedIds);
  for (let j = 0; j < messages.length; j++) {
    if (!isAssistantWithToolUse(messages[j])) continue;
    const tuIds = extractAllToolUseIds(messages[j]);
    const danglingIds = tuIds.filter((id) => allDeletedIds.has(id));
    if (danglingIds.length === 0) continue;

    if (danglingIds.length === tuIds.length) {
      tokensToFree -= tokenArray[j];
      totalFreed += tokenArray[j];
      messages.splice(j, 1);
      tokenArray.splice(j, 1);
      deletedCount++;
      j--;
      continue;
    }

    const content = messages[j].type === "message" ? messages[j].message?.content : messages[j].content;
    if (Array.isArray(content)) {
      const danglingSet = new Set(danglingIds);
      for (let k = content.length - 1; k >= 0; k--) {
        const block = content[k];
        if ((block?.type === "tool_use" || block?.type === "toolCall") && block.id && danglingSet.has(block.id)) {
          content.splice(k, 1);
        }
      }
    }
  }

  return { freedTokens: totalFreed, deletedCount };
}

interface ToolPairGroup {
  /** Indices of messages in this tool pair (assistant + tool_results). */
  indices: number[];
}

/**
 * Collect tool pairs (assistant + consecutive tool_results) within a range.
 * Returns in chronological order (oldest first).
 */
function collectToolPairsInRange(messages: Message[], start: number, end: number): ToolPairGroup[] {
  const pairs: ToolPairGroup[] = [];
  let i = start;

  while (i < end) {
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;

    // Find an assistant message (potentially with tool_use)
    if (role === "assistant") {
      const group: number[] = [i];
      let j = i + 1;
      // Collect consecutive tool_result messages following it
      while (j < end) {
        const r = messages[j].role ?? messages[j].message?.role ?? messages[j].type;
        if (r === "tool" || r === "toolResult" || r === "tool_result") {
          group.push(j);
          j++;
        } else {
          break;
        }
      }
      // Only consider it a "tool pair" if there are tool_results following
      if (group.length > 1) {
        pairs.push({ indices: group });
      }
      i = j;
    } else {
      i++;
    }
  }

  return pairs;
}

/**
 * Check if a message has a tool role (tool_result / toolResult).
 * Used to expand deletion to include consecutive tool messages and prevent orphans.
 */
export function isToolRole(msg: Message): boolean {
  const role = msg.role ?? msg.message?.role ?? msg.type;
  return role === "tool" || role === "toolResult" || role === "tool_result";
}

/**
 * Ensure tool pair integrity after deletion decisions.
 * If a tool_result is deleted, ensure its assistant(tool_use) is also deleted.
 * If an assistant(tool_use) is deleted, ensure its tool_result is also deleted.
 * This prevents Anthropic ValidationException about mismatched tool pairs.
 */
function ensureToolPairIntegrity(messages: Message[], deleteIndices: Set<number>): void {
  // Build tool_use_id → index mappings for all messages
  const toolUseIdToAssistantIdx = new Map<string, number>();
  const toolResultIdToIdx = new Map<string, number>();

  for (let i = 0; i < messages.length; i++) {
    // Collect tool_use IDs from assistant messages
    const toolUseIds = extractAllToolUseIds(messages[i]);
    for (const id of toolUseIds) {
      toolUseIdToAssistantIdx.set(id, i);
    }
    // Collect tool_result IDs
    const resultId = extractToolResultId(messages[i]);
    if (resultId) {
      toolResultIdToIdx.set(resultId, i);
    }
    // Anthropic: user message may have multiple tool_result blocks
    if (messages[i].role === "user" && Array.isArray(messages[i].content)) {
      for (const block of messages[i].content) {
        if (block?.type === "tool_result" && block.tool_use_id) {
          toolResultIdToIdx.set(block.tool_use_id, i);
        }
      }
    }
  }

  // If a tool_result msg is deleted → also delete its assistant(tool_use)
  let changed = true;
  while (changed) {
    changed = false;
    for (const [toolUseId, resultIdx] of toolResultIdToIdx) {
      const assistantIdx = toolUseIdToAssistantIdx.get(toolUseId);
      if (assistantIdx === undefined) continue;

      if (deleteIndices.has(resultIdx) && !deleteIndices.has(assistantIdx)) {
        // Check if ALL tool_results for this assistant are deleted
        const allUseIds = extractAllToolUseIds(messages[assistantIdx]);
        const allResultsDeleted = allUseIds.every((id) => {
          const rIdx = toolResultIdToIdx.get(id);
          return rIdx !== undefined && deleteIndices.has(rIdx);
        });
        if (allResultsDeleted) {
          deleteIndices.add(assistantIdx);
          changed = true;
        }
      }

      if (deleteIndices.has(assistantIdx) && !deleteIndices.has(resultIdx)) {
        deleteIndices.add(resultIdx);
        changed = true;
      }
    }
  }
}

/**
 * Count non-MMD messages excluding a set of indices marked for deletion.
 */
export function countNonMmdExcluding(messages: Message[], excludeIndices: Set<number>): number {
  let count = 0;
  for (let i = 0; i < messages.length; i++) {
    if (!excludeIndices.has(i) && !isMmdMessage(messages[i])) count++;
  }
  return count;
}

// ─── Aggressive Compression ──────────────────────────────────────────────────

export interface AggressiveResult {
  deletedCount: number;
  deletedIds: string[];
  remainingTokens: number;
}

/**
 * Aggressive compression: adaptive scan direction for minimal token calculation.
 * - If excess (tokens to delete) <= retain budget → scan from HEAD (calculate less)
 * - If excess > retain budget → scan from TAIL (calculate less)
 * Protects the last user message, MMD, and at least AGGRESSIVE_MIN_KEEP non-MMD messages.
 * Mutates messages and tokenArray in place.
 */
export function aggressiveCompress(
  messages: Message[],
  targetTokens: number,
  tokenArray: number[],
  clientTotalTokens?: number,
): AggressiveResult {
  // ── Baseline ──────────────────────────────────────────────────────────────
  // clientTotalTokens (from API usage) is the authoritative token count.
  // tokenArray (local tiktoken estimate) is only used for relative weighting
  // between messages — it decides "which message to delete" not "how many tokens".
  // scale maps local token values back to the client baseline.
  let tokenArraySum = tokenArray.reduce((s, v) => s + v, 0);
  if (tokenArraySum <= 0 && messages.length > 0) {
    for (let i = 0; i < messages.length; i++) {
      tokenArray[i] = preciseMessageTokens(messages[i]);
    }
    tokenArraySum = tokenArray.reduce((s, v) => s + v, 0);
  }
  const currentTokens = clientTotalTokens && clientTotalTokens > 0
    ? clientTotalTokens
    : tokenArraySum;
  if (currentTokens <= targetTokens) {
    return { deletedCount: 0, deletedIds: [], remainingTokens: currentTokens };
  }
  const scale = tokenArraySum > 0 ? currentTokens / tokenArraySum : 1;
  const excess = currentTokens - targetTokens;

  const protectedTailIdx = findLastUserMessageIndex(messages);
  const protectedStart = protectedTailIdx >= 0 ? protectedTailIdx : messages.length;

  // ── Phase 1: Truncate oversized tool_result content ────────────────────
  // Before deleting any messages, try to free tokens by truncating large
  // tool results to TOOL_RESULT_TRUNCATE_CHARS. This preserves conversation
  // context while reducing token count.
  let freedClient = 0; // tracks freed tokens in client baseline
  for (let round = 0; round < messages.length && freedClient < excess; round++) {
    let maxIdx = -1;
    let maxLen = 0;
    for (let i = 0; i < messages.length; i++) {
      if (isMmdMessage(messages[i])) continue;
      if (!isToolResultMessage(messages[i])) continue;
      const content = getTextContent(messages[i]);
      if (content.length <= TOOL_RESULT_TRUNCATE_CHARS) continue;
      if (content.length > maxLen) {
        maxLen = content.length;
        maxIdx = i;
      }
    }
    if (maxIdx === -1) break;

    const oldTokens = tokenArray[maxIdx];
    const oldContent = getTextContent(messages[maxIdx]);
    const truncated = oldContent.slice(0, TOOL_RESULT_TRUNCATE_CHARS) +
      `\n\n[... content truncated, only first ${TOOL_RESULT_TRUNCATE_CHARS} characters retained ...]`;
    setTextContent(messages[maxIdx], truncated);
    const newTokens = preciseMessageTokens(messages[maxIdx]);
    tokenArray[maxIdx] = newTokens;
    const localFreed = oldTokens - newTokens;
    freedClient += localFreed * scale;
  }

  // If truncation alone brought us below target, we're done
  if (freedClient >= excess) {
    return {
      deletedCount: 0,
      deletedIds: [],
      remainingTokens: Math.round(currentTokens - freedClient),
    };
  }

  // ── Phase 2: Delete old messages from head ─────────────────────────────
  // tokenArray decides relative priority (oldest first), scale maps freed
  // back to client baseline. Respects min-keep and protected tail.
  const remainingExcess = excess - freedClient;
  const deleteIndices: Set<number> = new Set();
  let deleteFreedLocal = 0;
  const scaledExcess = tokenArraySum > 0 ? remainingExcess / scale : remainingExcess;

  // Pass 1: tool results first (from head)
  for (let i = 0; i < protectedStart && deleteFreedLocal < scaledExcess; i++) {
    if (isMmdMessage(messages[i])) continue;
    if (isSystemReminder(messages[i])) continue;
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
    if (role === "system") continue;
    if (!isToolResultMessage(messages[i])) continue;
    const nonMmdRemaining = countNonMmdExcluding(messages, new Set([...deleteIndices, i]));
    if (nonMmdRemaining < AGGRESSIVE_MIN_KEEP) break;
    deleteIndices.add(i);
    deleteFreedLocal += tokenArray[i];
  }

  // Pass 2: other messages (assistant, user) from head
  if (deleteFreedLocal < scaledExcess) {
    for (let i = 0; i < protectedStart && deleteFreedLocal < scaledExcess; i++) {
      if (deleteIndices.has(i)) continue;
      if (isMmdMessage(messages[i])) continue;
      if (isSystemReminder(messages[i])) continue;
      const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
      if (role === "system") continue;
      const nonMmdRemaining = countNonMmdExcluding(messages, new Set([...deleteIndices, i]));
      if (nonMmdRemaining < AGGRESSIVE_MIN_KEEP) break;
      deleteIndices.add(i);
      deleteFreedLocal += tokenArray[i];
    }
  }

  // Ensure tool pair integrity
  ensureToolPairIntegrity(messages, deleteIndices);

  // ── Phase 3: Execute deletion + orphan cleanup ─────────────────────────
  const deletedIds: string[] = [];

  // Strip dangling tool_use blocks from assistant messages
  const deletedToolCallIds = new Set<string>();
  for (const idx of deleteIndices) {
    const tid = extractToolResultId(messages[idx]);
    if (tid) deletedToolCallIds.add(tid);
  }
  for (let j = 0; j < messages.length; j++) {
    if (deleteIndices.has(j)) continue;
    if (!isAssistantWithToolUse(messages[j])) continue;
    const tuIds = extractAllToolUseIds(messages[j]);
    const danglingIds = tuIds.filter((id) => deletedToolCallIds.has(id));
    if (danglingIds.length === 0) continue;
    if (danglingIds.length === tuIds.length) {
      // All tool_uses orphaned — delete the assistant too
      deleteIndices.add(j);
      continue;
    }
    // Partial: strip only dangling blocks
    const content = messages[j].type === "message" ? messages[j].message?.content : messages[j].content;
    if (Array.isArray(content)) {
      const danglingSet = new Set(danglingIds);
      for (let k = content.length - 1; k >= 0; k--) {
        const block = content[k];
        if ((block?.type === "tool_use" || block?.type === "toolCall") && block.id && danglingSet.has(block.id)) {
          content.splice(k, 1);
        }
      }
    }
  }

  // Batch delete in reverse order
  let totalDeleteFreedLocal = 0;
  const sortedIndices = [...deleteIndices].sort((a, b) => b - a);
  for (const idx of sortedIndices) {
    const tid = extractToolResultId(messages[idx]);
    if (tid) deletedIds.push(tid);
    totalDeleteFreedLocal += tokenArray[idx];
    messages.splice(idx, 1);
    tokenArray.splice(idx, 1);
  }

  freedClient += totalDeleteFreedLocal * scale;
  const remainingTokens = Math.round(currentTokens - freedClient);

  return { deletedCount: deleteIndices.size, deletedIds, remainingTokens };
}

/**
 * Scan from head: accumulate tokens to delete until freed >= excess.
 * Skips MMD, system, and protected tail.
 *
 * Priority: delete tool results first (small, enables orphan cleanup of large assistant),
 * then other messages (assistant, user text). This avoids over-deleting when one assistant
 * message contains large tool_call arguments.
 */
function scanFromHeadForDeletion(
  messages: Message[],
  tokenArray: number[],
  excess: number,
  protectedStart: number,
): Set<number> {
  const deleteIndices = new Set<number>();
  let freed = 0;

  // Pass 1: delete tool results first (from head, within unprotected zone)
  for (let i = 0; i < protectedStart && freed < excess; i++) {
    if (isMmdMessage(messages[i])) continue;
    if (isSystemReminder(messages[i])) continue;
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
    if (role === "system") continue;
    if (!isToolResultMessage(messages[i])) continue;

    const nonMmdRemaining = countNonMmdExcluding(messages, new Set([...deleteIndices, i]));
    if (nonMmdRemaining < AGGRESSIVE_MIN_KEEP) break;

    deleteIndices.add(i);
    freed += tokenArray[i];
  }

  // Pass 2: if still need more, delete other messages (assistant, user) from head
  if (freed < excess) {
    for (let i = 0; i < protectedStart && freed < excess; i++) {
      if (deleteIndices.has(i)) continue;
      if (isMmdMessage(messages[i])) continue;
      if (isSystemReminder(messages[i])) continue;
      const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
      if (role === "system") continue;

      const nonMmdRemaining = countNonMmdExcluding(messages, new Set([...deleteIndices, i]));
      if (nonMmdRemaining < AGGRESSIVE_MIN_KEEP) break;

      deleteIndices.add(i);
      freed += tokenArray[i];
    }
  }

  // Pass 3: if still need more tokens but hit min-keep limit,
  // truncate oversized tool_result content to TOOL_RESULT_TRUNCATE_CHARS in remaining messages
  if (freed < excess) {
    freed += truncateRemainingToolResults(
      messages, tokenArray, deleteIndices, excess - freed, false,
    );
  }

  // Pass 4: ensure tool pair integrity — if a tool_result is deleted,
  // its assistant(tool_use) must also be deleted, and vice versa.
  ensureToolPairIntegrity(messages, deleteIndices);

  return deleteIndices;
}

/**
 * Scan from tail: accumulate tokens to retain until retained >= targetTokens.
 * Everything not retained is marked for deletion. Used when excess is large.
 */
function scanFromTailForRetention(
  messages: Message[],
  tokenArray: number[],
  targetTokens: number,
  protectedStart: number,
): Set<number> {
  const retainIndices = new Set<number>();
  let retainedTokens = 0;

  // Retain protected tail unconditionally
  for (let i = protectedStart; i < messages.length; i++) {
    retainIndices.add(i);
    retainedTokens += tokenArray[i];
  }

  // Retain MMD unconditionally
  for (let i = 0; i < messages.length; i++) {
    if (isMmdMessage(messages[i]) && !retainIndices.has(i)) {
      retainIndices.add(i);
      retainedTokens += tokenArray[i];
    }
  }

  // Retain system message unconditionally
  for (let i = 0; i < messages.length; i++) {
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
    if (role === "system" && !retainIndices.has(i)) {
      retainIndices.add(i);
      retainedTokens += tokenArray[i];
    }
  }

  // Scan backward, retain until budget filled
  for (let i = protectedStart - 1; i >= 0 && retainedTokens < targetTokens; i--) {
    if (retainIndices.has(i)) continue;
    retainIndices.add(i);
    retainedTokens += tokenArray[i];
  }

  // Ensure AGGRESSIVE_MIN_KEEP non-MMD messages retained
  let nonMmdRetained = 0;
  for (const idx of retainIndices) {
    if (!isMmdMessage(messages[idx])) nonMmdRetained++;
  }
  for (let i = protectedStart - 1; i >= 0 && nonMmdRetained < AGGRESSIVE_MIN_KEEP; i--) {
    if (retainIndices.has(i)) continue;
    if (!isMmdMessage(messages[i])) {
      retainIndices.add(i);
      retainedTokens += tokenArray[i];
      nonMmdRetained++;
    }
  }

  // Everything not retained → delete
  const deleteIndices = new Set<number>();
  for (let i = 0; i < messages.length; i++) {
    if (!retainIndices.has(i)) deleteIndices.add(i);
  }

  // If min-keep forced us to retain more messages than budget allows,
  // truncate oversized tool_result content in retained messages to free tokens
  const deletedTokens = [...deleteIndices].reduce(
    (s, idx) => s + tokenArray[idx], 0,
  );
  const totalTokensSum = tokenArray.reduce((s, v) => s + v, 0);
  const remainAfterDelete = totalTokensSum - deletedTokens;
  if (remainAfterDelete > targetTokens) {
    truncateRemainingToolResults(
      messages, tokenArray, deleteIndices, remainAfterDelete - targetTokens, false,
    );
  }

  return deleteIndices;
}

// ─── Emergency Compression ───────────────────────────────────────────────────

export interface EmergencyResult {
  deletedCount: number;
  deletedIds: string[];
  remainingTokens: number;
}

/**
 * Emergency compression: adaptive scan with MMD extraction/restoration.
 * Chooses scan direction (head vs tail) based on which is cheaper.
 * Mutates messages and tokenArray in place.
 */
export function emergencyCompress(
  messages: Message[],
  targetTokens: number,
  tokenArray: number[],
  clientRemainingTokens?: number,
): EmergencyResult {
  // ── 1. Extract MMD messages (preserve them) ────────────────────────────
  const mmdItems: Array<{ msg: Message; tokens: number }> = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (isMmdMessage(messages[i])) {
      mmdItems.unshift({ msg: messages.splice(i, 1)[0], tokens: tokenArray.splice(i, 1)[0] });
    }
  }

  // ── Baseline (same as aggressiveCompress) ──────────────────────────────
  let tokenArraySum = tokenArray.reduce((s, v) => s + v, 0);
  if (tokenArraySum <= 0 && messages.length > 0) {
    for (let i = 0; i < messages.length; i++) {
      tokenArray[i] = preciseMessageTokens(messages[i]);
    }
    tokenArraySum = tokenArray.reduce((s, v) => s + v, 0);
  }
  let currentTokens = clientRemainingTokens && clientRemainingTokens > 0
    ? clientRemainingTokens
    : tokenArraySum;
  const scale = tokenArraySum > 0 ? currentTokens / tokenArraySum : 1;

  const deletedIds: string[] = [];
  let deletedCount = 0;

  if (currentTokens <= targetTokens || messages.length <= 2) {
    // Restore MMD and return
    for (const { msg, tokens } of mmdItems) {
      messages.splice(0, 0, msg);
      tokenArray.splice(0, 0, tokens);
      currentTokens += tokens;
    }
    return { deletedCount: 0, deletedIds: [], remainingTokens: currentTokens };
  }

  const excess = currentTokens - targetTokens;
  const protectedTailIdx = findLastUserMessageIndex(messages);
  const protectedStart = protectedTailIdx >= 0 ? protectedTailIdx : messages.length;

  // ── Phase 1: Truncate oversized tool_result content ────────────────────
  let freedClient = 0;
  for (let round = 0; round < messages.length && freedClient < excess; round++) {
    let maxIdx = -1;
    let maxLen = 0;
    for (let i = 0; i < messages.length; i++) {
      if (!isToolResultMessage(messages[i])) continue;
      const content = getTextContent(messages[i]);
      if (content.length <= TOOL_RESULT_TRUNCATE_CHARS) continue;
      if (content.length > maxLen) {
        maxLen = content.length;
        maxIdx = i;
      }
    }
    if (maxIdx === -1) break;

    const oldTokens = tokenArray[maxIdx];
    const oldContent = getTextContent(messages[maxIdx]);
    const truncated = oldContent.slice(0, TOOL_RESULT_TRUNCATE_CHARS) +
      `\n\n[... content truncated, only first ${TOOL_RESULT_TRUNCATE_CHARS} characters retained ...]`;
    setTextContent(messages[maxIdx], truncated);
    const newTokens = preciseMessageTokens(messages[maxIdx]);
    tokenArray[maxIdx] = newTokens;
    const localFreed = oldTokens - newTokens;
    freedClient += localFreed * scale;
  }

  if (freedClient >= excess) {
    // Truncation alone was enough — restore MMD and return
    for (const { msg, tokens } of mmdItems) {
      messages.splice(0, 0, msg);
      tokenArray.splice(0, 0, tokens);
    }
    return {
      deletedCount: 0,
      deletedIds: [],
      remainingTokens: Math.round(currentTokens - freedClient),
    };
  }

  // ── Phase 2: Delete old messages from head ─────────────────────────────
  const remainingExcess = excess - freedClient;
  const scaledExcess = tokenArraySum > 0 ? remainingExcess / scale : remainingExcess;
  const deleteIndices = new Set<number>();
  let deleteFreedLocal = 0;

  // Pass 1: tool results first
  for (let i = 0; i < protectedStart && deleteFreedLocal < scaledExcess; i++) {
    if (isMmdMessage(messages[i])) continue;
    if (isSystemReminder(messages[i])) continue;
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
    if (role === "system") continue;
    if (!isToolResultMessage(messages[i])) continue;
    const nonMmdRemaining = countNonMmdExcluding(messages, new Set([...deleteIndices, i]));
    if (nonMmdRemaining < EMERGENCY_MIN_KEEP) break;
    deleteIndices.add(i);
    deleteFreedLocal += tokenArray[i];
  }

  // Pass 2: other messages
  if (deleteFreedLocal < scaledExcess) {
    for (let i = 0; i < protectedStart && deleteFreedLocal < scaledExcess; i++) {
      if (deleteIndices.has(i)) continue;
      if (isMmdMessage(messages[i])) continue;
      if (isSystemReminder(messages[i])) continue;
      const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
      if (role === "system") continue;
      const nonMmdRemaining = countNonMmdExcluding(messages, new Set([...deleteIndices, i]));
      if (nonMmdRemaining < EMERGENCY_MIN_KEEP) break;
      deleteIndices.add(i);
      deleteFreedLocal += tokenArray[i];
    }
  }

  // Ensure tool pair integrity
  ensureToolPairIntegrity(messages, deleteIndices);

  // ── Phase 3: Orphan cleanup + execute deletion ─────────────────────────
  const deletedToolCallIds = new Set<string>();
  for (const idx of deleteIndices) {
    const tid = extractToolResultId(messages[idx]);
    if (tid) deletedToolCallIds.add(tid);
  }
  for (let j = 0; j < messages.length; j++) {
    if (deleteIndices.has(j)) continue;
    if (!isAssistantWithToolUse(messages[j])) continue;
    const tuIds = extractAllToolUseIds(messages[j]);
    const danglingIds = tuIds.filter((id) => deletedToolCallIds.has(id));
    if (danglingIds.length === 0) continue;
    if (danglingIds.length === tuIds.length) {
      deleteIndices.add(j);
      continue;
    }
    const content = messages[j].type === "message" ? messages[j].message?.content : messages[j].content;
    if (Array.isArray(content)) {
      const danglingSet = new Set(danglingIds);
      for (let k = content.length - 1; k >= 0; k--) {
        const block = content[k];
        if ((block?.type === "tool_use" || block?.type === "toolCall") && block.id && danglingSet.has(block.id)) {
          content.splice(k, 1);
        }
      }
    }
  }

  // Batch delete
  let totalDeleteFreedLocal = 0;
  const sortedIndices = [...deleteIndices].sort((a, b) => b - a);
  for (const idx of sortedIndices) {
    const tid = extractToolResultId(messages[idx]);
    if (tid) deletedIds.push(tid);
    totalDeleteFreedLocal += tokenArray[idx];
    messages.splice(idx, 1);
    tokenArray.splice(idx, 1);
    deletedCount++;
  }

  freedClient += totalDeleteFreedLocal * scale;

  // ── 4. Restore MMD messages at head ────────────────────────────────────
  for (const { msg, tokens } of mmdItems) {
    messages.splice(0, 0, msg);
    tokenArray.splice(0, 0, tokens);
  }

  return {
    deletedCount,
    deletedIds,
    remainingTokens: Math.round(currentTokens - freedClient),
  };
}
