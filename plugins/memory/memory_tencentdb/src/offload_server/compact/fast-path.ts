/**
 * L3 Fast-path — re-apply confirmed/deleted offload state to messages.
 * Runs on every compaction call as the first step.
 */
import type { OffloadEntry, CompactState } from "../types.js";
import type { Message } from "./helpers.js";
import {
  extractToolResultId,
  isToolResultMessage,
  isOnlyToolUseAssistant,
  isAssistantWithToolUse,
  extractAllToolUseIds,
  replaceWithSummary,
  buildOffloadMap,
} from "./helpers.js";

export interface FastPathResult {
  replacedCount: number;
  deletedCount: number;
}

/**
 * Apply fast-path: replace confirmed tool results with summary,
 * delete messages whose tool_call_id is in deletedOffloadIds,
 * strip orphaned toolCall blocks from mixed assistant messages.
 * Mutates messages in place.
 */
export function applyFastPath(
  messages: Message[],
  entries: OffloadEntry[],
  compactState: CompactState,
): FastPathResult {
  const offloadMap = buildOffloadMap(entries);
  const confirmedSet = new Set(compactState.confirmedOffloadIds);
  const deletedSet = new Set(compactState.deletedOffloadIds);

  if (confirmedSet.size === 0 && deletedSet.size === 0) {
    return { replacedCount: 0, deletedCount: 0 };
  }

  const indicesToDelete: number[] = [];
  let replacedCount = 0;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const tid = extractToolResultId(msg);

    // 1. Deleted tool_result — handle Anthropic multi-block user messages carefully
    if (tid && deletedSet.has(tid)) {
      // Anthropic format: user message may have multiple tool_result blocks
      if (msg.role === "user" && Array.isArray(msg.content)) {
        const allToolResultIds = msg.content
          .filter((b: any) => b?.type === "tool_result" && b.tool_use_id)
          .map((b: any) => b.tool_use_id as string);

        if (allToolResultIds.length > 1) {
          // Multi-block: only remove the blocks that are in deletedSet, keep the rest
          const allDeleted = allToolResultIds.every((id: string) => deletedSet.has(id));
          if (allDeleted) {
            indicesToDelete.push(i);
          } else {
            // Partial: strip only deleted tool_result blocks
            for (let j = msg.content.length - 1; j >= 0; j--) {
              const block = msg.content[j];
              if (block?.type === "tool_result" && block.tool_use_id && deletedSet.has(block.tool_use_id)) {
                msg.content.splice(j, 1);
              }
            }
          }
          continue;
        }
      }
      // Single tool_result or non-Anthropic format: delete entire message
      indicesToDelete.push(i);
      continue;
    }

    // 2. Confirmed tool_result → replace with summary
    if (tid && confirmedSet.has(tid) && isToolResultMessage(msg) && !msg._offloaded) {
      const entry = offloadMap.get(tid);
      if (entry) {
        replaceWithSummary(msg, entry);
        replacedCount++;
      }
    }

    // 3. Pure toolCall assistant — all IDs deleted → mark for deletion
    if (!tid && isOnlyToolUseAssistant(msg)) {
      const tuIds = extractAllToolUseIds(msg);
      if (tuIds.length > 0 && tuIds.every((id) => deletedSet.has(id))) {
        indicesToDelete.push(i);
      }
    }

    // 4. Mixed assistant (text + toolCall) — strip deleted toolCall blocks
    //    to prevent orphaned tool_use without matching tool_result (provider 400)
    if (!tid && isAssistantWithToolUse(msg) && !isOnlyToolUseAssistant(msg)) {
      const content = msg.type === "message" ? msg.message?.content : msg.content;
      if (Array.isArray(content)) {
        for (let j = content.length - 1; j >= 0; j--) {
          const block = content[j];
          if ((block?.type === "tool_use" || block?.type === "toolCall") && block.id) {
            if (deletedSet.has(block.id)) {
              content.splice(j, 1);
            }
          }
        }
      }
    }
  }

  // Delete in reverse order to preserve indices
  const uniqueIndices = [...new Set(indicesToDelete)].sort((a, b) => b - a);
  for (const idx of uniqueIndices) {
    messages.splice(idx, 1);
  }

  // Post-pass: ensure no orphaned tool_use or tool_result after deletion
  // Collect all remaining tool_use IDs and tool_result IDs
  const remainingToolUseIds = new Set<string>();
  const remainingToolResultIds = new Set<string>();
  for (const msg of messages) {
    for (const id of extractAllToolUseIds(msg)) {
      remainingToolUseIds.add(id);
    }
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block?.type === "tool_result" && block.tool_use_id) {
          remainingToolResultIds.add(block.tool_use_id);
        }
      }
    }
    const singleTid = extractToolResultId(msg);
    if (singleTid) remainingToolResultIds.add(singleTid);
  }

  // Remove orphaned tool_use blocks from assistant messages (no matching tool_result)
  for (const msg of messages) {
    if (!isAssistantWithToolUse(msg)) continue;
    const content = msg.type === "message" ? msg.message?.content : msg.content;
    if (!Array.isArray(content)) continue;
    for (let j = content.length - 1; j >= 0; j--) {
      const block = content[j];
      if ((block?.type === "tool_use" || block?.type === "toolCall") && block.id) {
        if (!remainingToolResultIds.has(block.id)) {
          content.splice(j, 1);
        }
      }
    }
  }

  // Remove orphaned tool_result messages/blocks (no matching tool_use)
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (let j = msg.content.length - 1; j >= 0; j--) {
        const block = msg.content[j];
        if (block?.type === "tool_result" && block.tool_use_id && !remainingToolUseIds.has(block.tool_use_id)) {
          msg.content.splice(j, 1);
        }
      }
      // If user message is now empty, remove it
      if (msg.content.length === 0) {
        messages.splice(i, 1);
        uniqueIndices.push(i); // count it
      }
    } else {
      const singleTid = extractToolResultId(msg);
      if (singleTid && !remainingToolUseIds.has(singleTid) && isToolResultMessage(msg)) {
        messages.splice(i, 1);
        uniqueIndices.push(i);
      }
    }
  }

  // Remove empty assistant messages (all tool_use blocks were stripped)
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const role = msg.role ?? msg.message?.role ?? msg.type;
    if (role !== "assistant") continue;
    const content = msg.type === "message" ? msg.message?.content : msg.content;
    if (Array.isArray(content) && content.length === 0) {
      messages.splice(i, 1);
      uniqueIndices.push(i);
    }
  }

  return { replacedCount, deletedCount: uniqueIndices.length };
}
