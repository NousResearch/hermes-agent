/**
 * L3 MMD Injector — inject active/history MMD into messages.
 * Handles version dedup and insertion point calculation.
 */
import type { StorageAdapter } from "../../core/storage/adapter.js";
import type { OffloadEntry, OffloadState } from "../types.js";
import type { Message } from "./helpers.js";
import { MMD_CONTEXT_MARKER, MMD_INJECTION_MARKER, buildOffloadMap } from "./helpers.js";

// ─── Active MMD Injection ────────────────────────────────────────────────────

export interface MmdInjectionResult {
  injectedCount: number;
  mmdTokensEstimate: number;
}

/**
 * Inject the active MMD into messages. Performs version dedup.
 */
export async function injectActiveMmd(
  messages: Message[],
  state: OffloadState,
  storage: StorageAdapter,
  basePath: string,
): Promise<MmdInjectionResult> {
  // Resolve the MMD file to inject: use activeMmdFile, or fallback to the latest .mmd
  let mmdFile = state.activeMmdFile;
  if (!mmdFile) {
    // Fallback: find the most recent .mmd file (sorted by prefix number, highest = latest)
    const allMmds = await storage.readdirNames(`${basePath}/mmds/`, ".mmd");
    if (allMmds.length > 0) {
      // Sort descending by prefix number (e.g., "006-current-task.mmd" → 6)
      allMmds.sort((a, b) => {
        const numA = parseInt(a.split("-")[0], 10) || 0;
        const numB = parseInt(b.split("-")[0], 10) || 0;
        return numB - numA;
      });
      // Pick the first non-empty one
      for (const candidate of allMmds) {
        const content = await storage.readFile(`${basePath}/mmds/${candidate}`);
        if (content && content.trim().length > 0) {
          mmdFile = candidate;
          break;
        }
      }
    }
  }

  if (!mmdFile) {
    removeMmdMessages(messages, "active");
    return { injectedCount: 0, mmdTokensEstimate: 0 };
  }

  const mmdContent = await storage.readFile(`${basePath}/mmds/${mmdFile}`);
  if (!mmdContent) {
    removeMmdMessages(messages, "active");
    return { injectedCount: 0, mmdTokensEstimate: 0 };
  }

  const newVersion = hashContent(mmdContent);

  // Version dedup: check existing MMD message version and position
  const existingIdx = messages.findIndex((m) => m[MMD_CONTEXT_MARKER] === "active");
  if (existingIdx >= 0 && messages[existingIdx]._mmdVersion === newVersion) {
    // Content unchanged — but still need to ensure correct position
    // (aggressive compression may have shifted it)
    const correctIdx = findActiveMmdInsertionPoint(
      messages.filter((m) => m[MMD_CONTEXT_MARKER] !== "active"),
    );
    // Check if position is already correct (accounting for removal offset)
    const effectiveCorrectIdx = correctIdx <= existingIdx ? correctIdx : correctIdx;
    if (existingIdx === effectiveCorrectIdx) {
      return { injectedCount: 0, mmdTokensEstimate: 0 };
    }
    // Reposition: remove and re-insert at correct point
    const [mmdMsg] = messages.splice(existingIdx, 1);
    const newInsertIdx = findActiveMmdInsertionPoint(messages);
    messages.splice(newInsertIdx, 0, mmdMsg);
    return { injectedCount: 0, mmdTokensEstimate: 0 };
  }

  // Remove old active MMD message
  removeMmdMessages(messages, "active");

  // Build MMD text
  const mmdText = buildActiveMmdText(mmdFile, mmdContent);
  const mmdMsg: Message = {
    role: "user",
    content: mmdText,
    [MMD_CONTEXT_MARKER]: "active",
    _mmdVersion: newVersion,
    _mmdFilename: mmdFile,
  };

  // Insert at calculated point
  const insertIdx = findActiveMmdInsertionPoint(messages);
  messages.splice(insertIdx, 0, mmdMsg);

  return { injectedCount: 1, mmdTokensEstimate: Math.ceil(mmdContent.length / 4) };
}

// ─── History MMD Injection ───────────────────────────────────────────────────

export interface HistoryMmdResult {
  injectedCount: number;
  mmdFiles: string[];
  totalTokensEstimate: number;
}

/**
 * Inject history MMD files for entries that were deleted during aggressive compression.
 */
export async function injectHistoryMmds(
  messages: Message[],
  deletedIds: string[],
  entries: OffloadEntry[],
  state: OffloadState,
  storage: StorageAdapter,
  basePath: string,
  tokenBudget: number,
): Promise<HistoryMmdResult> {
  if (deletedIds.length === 0) {
    return { injectedCount: 0, mmdFiles: [], totalTokensEstimate: 0 };
  }

  // 1. Find MMD prefixes from deleted entries' node_ids
  const offloadMap = buildOffloadMap(entries);
  const mmdPrefixes = new Set<string>();
  for (const id of deletedIds) {
    const entry = offloadMap.get(id);
    if (entry?.node_id) {
      const prefix = entry.node_id.split("-")[0];
      if (prefix) mmdPrefixes.add(prefix);
    }
  }
  if (mmdPrefixes.size === 0) {
    return { injectedCount: 0, mmdFiles: [], totalTokensEstimate: 0 };
  }

  // 2. Find matching history MMD files (exclude active)
  const allMmds = await storage.readdirNames(`${basePath}/mmds/`, ".mmd");
  const candidates = allMmds.filter((f) => {
    const prefix = f.split("-")[0];
    return mmdPrefixes.has(prefix) && f !== state.activeMmdFile;
  });
  if (candidates.length === 0) {
    return { injectedCount: 0, mmdFiles: [], totalTokensEstimate: 0 };
  }

  // 3. Read and inject (respecting token budget), most recent first
  candidates.reverse();
  const injected: Message[] = [];
  const mmdFiles: string[] = [];
  let usedTokens = 0;

  for (const filename of candidates) {
    const content = await storage.readFile(`${basePath}/mmds/${filename}`);
    if (!content) continue;

    const text = buildHistoryMmdText(filename, content);
    const tokens = Math.ceil(text.length / 4);
    if (usedTokens + tokens > tokenBudget) continue;

    injected.push({
      role: "user",
      content: text,
      [MMD_INJECTION_MARKER]: true,
      _mmdFilename: filename,
    });
    mmdFiles.push(filename);
    usedTokens += tokens;
  }

  if (injected.length === 0) {
    return { injectedCount: 0, mmdFiles: [], totalTokensEstimate: 0 };
  }

  // 4. Remove old history MMD injections and insert new ones
  removeExistingMmdInjections(messages);
  const insertIdx = findHistoryMmdInsertionPoint(messages);
  // Reverse back to chronological order (oldest first)
  injected.reverse();
  mmdFiles.reverse();
  messages.splice(insertIdx, 0, ...injected);

  return { injectedCount: injected.length, mmdFiles, totalTokensEstimate: usedTokens };
}

// ─── Insertion Point Calculation ─────────────────────────────────────────────

/**
 * Find insertion point for active MMD.
 * Strategy: insert after the latest user message in the second half.
 * Guard: don't split tool_call / tool_result pairs.
 */
export function findActiveMmdInsertionPoint(messages: Message[]): number {
  if (messages.length <= 2) {
    const idx = Math.min(1, messages.length);
    return adjustForToolCallPair(messages, idx);
  }

  let latestUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
    if (role === "user" && !messages[i][MMD_CONTEXT_MARKER] && !messages[i][MMD_INJECTION_MARKER]) {
      latestUserIdx = i;
      break;
    }
  }

  let insertIdx: number;
  if (latestUserIdx >= 0) {
    // Insert relative to latest user message:
    // - If it's the last message → insert before it (user prompt stays last)
    // - Otherwise → insert after it (between user and tool loop, matching plugin behavior)
    if (latestUserIdx === messages.length - 1) {
      insertIdx = latestUserIdx;
    } else {
      insertIdx = latestUserIdx + 1;
    }
  } else {
    // No user message found — fallback: before the trailing tool loop
    let loopStart = messages.length;
    for (let i = messages.length - 1; i >= 0; i--) {
      const role = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
      if (messages[i][MMD_CONTEXT_MARKER] || messages[i][MMD_INJECTION_MARKER]) continue;
      if (role === "toolResult" || role === "tool" || role === "assistant") {
        loopStart = i;
      } else {
        break;
      }
    }
    const maxDistFromTail = 30;
    const minInsertIdx = Math.max(1, messages.length - maxDistFromTail);
    insertIdx = Math.max(loopStart, minInsertIdx);
  }

  // Guard: don't insert between assistant(tool_use) and its tool_result
  insertIdx = adjustForToolCallPair(messages, insertIdx);

  // Hard guard: never insert before system message
  if (insertIdx === 0 && messages.length > 0) {
    const firstRole = messages[0].role ?? messages[0].message?.role ?? messages[0].type;
    if (firstRole === "system") {
      insertIdx = 1;
    }
  }

  return insertIdx;
}

/**
 * Find insertion point for history MMD (before active MMD).
 */
export function findHistoryMmdInsertionPoint(messages: Message[]): number {
  for (let i = 0; i < messages.length; i++) {
    if (messages[i][MMD_CONTEXT_MARKER] === "active") return i;
  }
  return findActiveMmdInsertionPoint(messages);
}

// ─── MMD Message Management ─────────────────────────────────────────────────

/**
 * Remove MMD messages by type ("active", "history", or all).
 */
export function removeMmdMessages(messages: Message[], type?: "active" | "history"): number {
  let removed = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (type === "active" && messages[i][MMD_CONTEXT_MARKER] === "active") {
      messages.splice(i, 1);
      removed++;
    } else if (type === "history" && messages[i][MMD_INJECTION_MARKER]) {
      messages.splice(i, 1);
      removed++;
    } else if (!type && (messages[i][MMD_CONTEXT_MARKER] || messages[i][MMD_INJECTION_MARKER])) {
      messages.splice(i, 1);
      removed++;
    }
  }
  return removed;
}

/**
 * Remove existing history MMD injection messages.
 */
export function removeExistingMmdInjections(messages: Message[]): number {
  let removed = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i][MMD_INJECTION_MARKER]) {
      messages.splice(i, 1);
      removed++;
    }
  }
  return removed;
}

// ─── Text Builders ───────────────────────────────────────────────────────────

/**
 * Build active MMD injection text.
 */
export function buildActiveMmdText(filename: string, mmdContent: string): string {
  let taskGoal = "";
  const metaMatch = mmdContent.match(/^%%\{\s*(.*?)\s*\}%%/);
  if (metaMatch) {
    try {
      const meta = JSON.parse(`{${metaMatch[1]}}`);
      taskGoal = meta.taskGoal || "";
    } catch { /* ignore */ }
  }

  return [
    `<current_task_context>`,
    `【当前活跃任务的mermaid流程图】这是你最近正在执行的任务的阶段性记录。`,
    taskGoal ? `**任务目标:** ${taskGoal}` : "",
    `**任务文件:** ${filename}`,
    "```mermaid",
    mmdContent,
    "```",
    `标记为 "doing" 的节点是近期焦点，"done" 的已完成。请参考此保持方向感，避免重复已完成的工作。`,
    `</current_task_context>`,
  ]
    .filter((line) => line !== "")
    .join("\n");
}

/**
 * Build history MMD injection text.
 */
export function buildHistoryMmdText(filename: string, mmdContent: string): string {
  let taskGoal = "";
  const metaMatch = mmdContent.match(/^%%\{\s*(.*?)\s*\}%%/);
  if (metaMatch) {
    try {
      const meta = JSON.parse(`{${metaMatch[1]}}`);
      taskGoal = meta.taskGoal || "";
    } catch { /* ignore */ }
  }

  return [
    `<history_task_context>`,
    `【历史任务记录】以下是此前完成的任务的概要。`,
    taskGoal ? `**任务目标:** ${taskGoal}` : "",
    `**任务文件:** ${filename}`,
    "```mermaid",
    mmdContent,
    "```",
    `</history_task_context>`,
  ]
    .filter((line) => line !== "")
    .join("\n");
}

// ─── Internal Helpers ────────────────────────────────────────────────────────

function hasToolResultContent(msg: Message): boolean {
  const content = msg?.content;
  if (!Array.isArray(content)) return false;
  return content.some(
    (block: any) =>
      typeof block === "object" &&
      block !== null &&
      (block.type === "tool_result" || block.type === "toolResult"),
  );
}

function hasToolUseContent(msg: Message): boolean {
  const content = msg?.content;
  if (!Array.isArray(content)) return false;
  return content.some(
    (block: any) =>
      typeof block === "object" &&
      block !== null &&
      (block.type === "tool_use" || block.type === "toolUse"),
  );
}

function adjustForToolCallPair(messages: Message[], insertIdx: number): number {
  if (insertIdx <= 0 || insertIdx >= messages.length) return insertIdx;
  const msgAtIdx = messages[insertIdx];
  const role = msgAtIdx?.role ?? msgAtIdx?.message?.role ?? msgAtIdx?.type;

  // Check both role-level (OpenAI: role="tool") and content-level (Anthropic: content[].type="tool_result")
  const isToolResult =
    role === "tool" ||
    role === "toolResult" ||
    role === "tool_result" ||
    (role === "user" && hasToolResultContent(msgAtIdx));

  if (isToolResult) {
    // Walk back to before the assistant tool_use
    let i = insertIdx - 1;
    while (i >= 0) {
      const r = messages[i].role ?? messages[i].message?.role ?? messages[i].type;
      if (r === "assistant" && hasToolUseContent(messages[i])) {
        return i;
      }
      if (r === "assistant") {
        return i;
      }
      const prevIsToolResult =
        r === "tool" ||
        r === "toolResult" ||
        r === "tool_result" ||
        (r === "user" && hasToolResultContent(messages[i]));
      if (!prevIsToolResult) break;
      i--;
    }
  }

  // Also check: don't insert right after an assistant with tool_use (before its tool_result)
  if (insertIdx > 0) {
    const prevMsg = messages[insertIdx - 1];
    const prevRole = prevMsg?.role ?? prevMsg?.message?.role ?? prevMsg?.type;
    if (prevRole === "assistant" && hasToolUseContent(prevMsg)) {
      return insertIdx - 1;
    }
  }

  return insertIdx;
}

function hashContent(s: string): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  return Math.abs(hash).toString(36).padStart(6, "0").slice(0, 6);
}
