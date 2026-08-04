/**
 * L3 Compaction — Message helper utilities.
 * Handles multiple message formats (OpenAI, Anthropic, OpenClaw).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OpenClaw 消息格式说明（来源：@mariozechner/pi-ai 类型定义）
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * OpenClaw 内部使用统一的 AgentMessage 联合类型，包含以下几种消息：
 *
 * ── 1. UserMessage ──
 * {
 *   role: "user",
 *   content: string | ContentBlock[],  // string 或 [{type:"text",text:"..."}] 或含图片块
 *   timestamp: number,
 * }
 *
 * ── 2. AssistantMessage（纯文本回复）──
 * {
 *   role: "assistant",
 *   content: [{ type: "text", text: "..." }],
 *   model: "gpt-5.2",
 *   stopReason: "stop",
 *   timestamp: number,
 *   api: "messages" | "chat" | "responses",
 *   provider: "anthropic" | "openai" | "google",
 *   usage: { input, output, totalTokens, ... },
 * }
 *
 * ── 3. AssistantMessage（含 tool_use / toolCall）──
 * {
 *   role: "assistant",
 *   content: [
 *     { type: "text", text: "I'll read the file..." },     // 可选文本块
 *     { type: "toolCall", id: "call_abc123", name: "read_file", arguments: { path: "..." } },
 *     { type: "toolCall", id: "call_def456", name: "exec", arguments: { cmd: "..." } },
 *   ],
 *   stopReason: "toolUse",
 *   ...同上
 * }
 * 注: Anthropic 原生格式使用 { type: "tool_use", id, name, input }
 *     OpenClaw 内部统一为 { type: "toolCall", id, name, arguments }
 *     但消息到达后端时**两种格式都可能出现**（取决于客户端是否已转换），
 *     因此本模块所有判断都同时匹配 "tool_use" 和 "toolCall"。
 *
 * ── 4. ToolResultMessage ──
 * {
 *   role: "toolResult",
 *   toolCallId: "call_abc123",           // 对应 AssistantMessage 中 toolCall 的 id
 *   toolName: "read_file",
 *   content: [{ type: "text", text: "文件内容..." }],
 *   isError: false,
 *   timestamp: number,
 *   details?: any,                       // 可选的详细信息（不发给 LLM）
 * }
 *
 * ── 5. 消息配对规则 ──
 * - 每个 AssistantMessage 中的 toolCall 必须有对应的 ToolResultMessage
 * - toolCallId 是配对的唯一标识
 * - 删除 tool_result 时必须同时删除对应的 assistant toolCall（否则 provider 返回 400）
 * - 一个 assistant 消息可包含多个 toolCall（并行工具调用）
 *
 * ── 6. 转换到 LLM Provider 原生格式 ──
 * OpenAI:     toolResult → { role: "tool", tool_call_id: "...", content: "..." }
 * Anthropic:  toolResult → { role: "user", content: [{ type: "tool_result", tool_use_id: "...", content: "..." }] }
 * 注: 本模块处理的是 **转换后** 的消息（发送给 provider 前的格式），
 *     因此需要兼容所有上述格式。
 *
 * ── 7. 特殊标记字段（本模块使用）──
 * - _offloaded: boolean          — 已被 summary 替换的 tool_result
 * - _mmdContextMessage: string   — MMD 注入消息标记（"active" | "history"）
 * - _mmdInjection: boolean       — 历史 MMD 注入消息标记
 * - _mmdVersion: string          — MMD 内容哈希，用于版本去重
 * ═══════════════════════════════════════════════════════════════════════════
 */
import type { OffloadEntry } from "../types.js";

// ─── Message type aliases ────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Message = Record<string, any>;

// ─── Tool Result ID Extraction ───────────────────────────────────────────────

/**
 * Extract the linked tool_use_id from a tool-result message (supports multiple formats).
 */
export function extractToolResultId(msg: Message): string | null {
  // OpenAI tool result: { role: "tool", tool_call_id: "..." }
  if (msg.tool_call_id) return msg.tool_call_id;
  // Anthropic tool_result: { type: "tool_result", tool_use_id: "..." }
  if (msg.tool_use_id) return msg.tool_use_id;
  // OpenClaw wrapped: { type: "message", message: { id: "...", role: "toolResult" } }
  if (msg.type === "message" && msg.message?.id) return msg.message.id;
  // Anthropic content block tool_result: { role: "user", content: [{type:"tool_result", tool_use_id}] }
  if (msg.role === "user" && Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (block?.type === "tool_result" && block.tool_use_id) return block.tool_use_id;
    }
  }
  return null;
}

// ─── Message Type Checks ─────────────────────────────────────────────────────

/**
 * Check if a message is a tool result (contains tool output).
 */
export function isToolResultMessage(msg: Message): boolean {
  if (msg.role === "tool") return true;
  if (msg.type === "tool_result") return true;
  const innerRole = msg.message?.role ?? msg.type;
  if (innerRole === "toolResult" || innerRole === "tool_result") return true;
  // Anthropic user message with tool_result content blocks
  if (msg.role === "user" && Array.isArray(msg.content)) {
    return msg.content.some((b: any) => b?.type === "tool_result");
  }
  return false;
}

/**
 * Check if a message is an assistant message containing only tool_use blocks.
 */
export function isOnlyToolUseAssistant(msg: Message): boolean {
  const role = msg.role ?? msg.message?.role;
  if (role !== "assistant") return false;
  const content = msg.type === "message" ? msg.message?.content : msg.content;
  if (!Array.isArray(content)) return false;
  if (content.length === 0) return false;
  return content.every((block: any) =>
    block?.type === "tool_use" || block?.type === "toolCall",
  );
}

/**
 * Check if a message is an assistant message that contains tool_use (may also have text).
 * Supports both Anthropic format (content blocks) and OpenAI format (tool_calls field).
 */
export function isAssistantWithToolUse(msg: Message): boolean {
  const role = msg.role ?? msg.message?.role;
  if (role !== "assistant") return false;
  // OpenAI format: tool_calls field
  if (msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) return true;
  // Anthropic format: content blocks
  const content = msg.type === "message" ? msg.message?.content : msg.content;
  if (!Array.isArray(content)) return false;
  return content.some((block: any) =>
    block?.type === "tool_use" || block?.type === "toolCall",
  );
}

/**
 * Extract all tool_use IDs from an assistant message.
 * Supports both Anthropic format (content blocks) and OpenAI format (tool_calls field).
 */
export function extractAllToolUseIds(msg: Message): string[] {
  const ids: string[] = [];
  // OpenAI format: tool_calls field
  if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      if (tc.id) ids.push(tc.id);
    }
  }
  // Anthropic format: content blocks
  const content = msg.type === "message" ? msg.message?.content : msg.content;
  if (Array.isArray(content)) {
    for (const block of content) {
      if ((block?.type === "tool_use" || block?.type === "toolCall") && block.id) {
        ids.push(block.id);
      }
    }
  }
  return ids;
}

// ─── Replace With Summary ────────────────────────────────────────────────────

/**
 * Replace a tool result message's content with the offload summary.
 */
export function replaceWithSummary(msg: Message, entry: OffloadEntry): void {
  const parts = [
    `[Offloaded Tool Result | node: ${entry.node_id ?? "N/A"}]`,
    `Summary: ${entry.summary}`,
  ];
  if (entry.result_ref) {
    parts.push(`原始工具结果已存档，如需查看完整内容请调用 tdai_read_cos(path="${entry.result_ref}")`);
  }
  const summaryContent = parts.join("\n");

  if (msg.type === "message" && msg.message) {
    if (Array.isArray(msg.message.content)) {
      msg.message.content = [{ type: "text", text: summaryContent }];
    } else {
      msg.message.content = summaryContent;
    }
  } else if (msg.role === "user" && Array.isArray(msg.content)) {
    // Anthropic tool_result in user content blocks
    for (let i = 0; i < msg.content.length; i++) {
      if (msg.content[i]?.type === "tool_result") {
        msg.content[i].content = summaryContent;
        break;
      }
    }
  } else {
    if (Array.isArray(msg.content)) {
      msg.content = [{ type: "text", text: summaryContent }];
    } else {
      msg.content = summaryContent;
    }
  }
  msg._offloaded = true;
}

// ─── Offload Map ─────────────────────────────────────────────────────────────

/**
 * Build a lookup map from tool_call_id → OffloadEntry.
 */
export function buildOffloadMap(entries: OffloadEntry[]): Map<string, OffloadEntry> {
  const map = new Map<string, OffloadEntry>();
  for (const entry of entries) {
    if (entry.tool_call_id) {
      map.set(entry.tool_call_id, entry);
    }
  }
  return map;
}

// ─── Token Estimation ────────────────────────────────────────────────────────

/**
 * Count CJK characters (CJK Unified Ideographs + Extension A + Compatibility).
 */
function countCjkChars(text: string): number {
  let n = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if (
      (c >= 0x4e00 && c <= 0x9fff) ||
      (c >= 0x3400 && c <= 0x4dbf) ||
      (c >= 0xf900 && c <= 0xfaff)
    ) {
      n++;
    }
  }
  return n;
}

/**
 * Estimate tokens for a text string using CJK-aware heuristic.
 * CJK characters ≈ 1 token / 1.7 chars, non-CJK ≈ 1 token / 4 chars.
 * Aligned with plugin-side estimateL3MixedTokensHeuristic.
 */
function estimateTextTokens(text: string): number {
  const cjk = countCjkChars(text);
  const rest = Math.max(0, text.length - cjk);
  return Math.max(1, Math.ceil(cjk / 1.7 + rest / 4));
}

/**
 * Estimate tokens for a message using CJK-aware heuristic (中文/1.7 + 非中文/4).
 */
export function estimateMessageTokens(msg: Message): number {
  const content = msg.content ?? msg.message?.content ?? "";
  let text: string;
  if (typeof content === "string") {
    text = content;
  } else if (Array.isArray(content)) {
    text = content.map((b: any) => (typeof b === "string" ? b : b?.text ?? JSON.stringify(b) ?? "")).join("");
  } else {
    text = JSON.stringify(content);
  }

  // Include tool_calls arguments (OpenAI format)
  const toolCalls = (msg as any).tool_calls;
  if (toolCalls && Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      if (tc.function?.name) text += tc.function.name;
      if (tc.function?.arguments) text += tc.function.arguments;
    }
  }

  return estimateTextTokens(text);
}

// ─── MMD Message Markers ─────────────────────────────────────────────────────

export const MMD_CONTEXT_MARKER = "_mmdContextMessage";
export const MMD_INJECTION_MARKER = "_mmdInjection";

/**
 * Check if a message is a MMD-related message (should be preserved during compression).
 */
export function isMmdMessage(msg: Message): boolean {
  return !!msg[MMD_CONTEXT_MARKER] || !!msg[MMD_INJECTION_MARKER];
}
