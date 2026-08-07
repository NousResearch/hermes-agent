/**
 * Capture hook (v3 client mode):
 *   1. Extract user/assistant messages from the agent_end raw message array
 *   2. Apply position slice + (optional) timestamp cursor to keep only this turn
 *   3. Replace the polluted user message with the cached original prompt
 *   4. Sanitize text + strip code blocks (assistant) + filter noise
 *   5. POST the cleaned messages to the gateway via SDK addConversation (/v3)
 *
 * Mirrors the structural cleanup in extensions/memory-tencentdb/src/core/conversation/l0-recorder.ts
 * (recordConversation), but does not write any local JSONL — the server is
 * authoritative for L0 storage.
 */

import type { MemoryClient } from "@tencentdb-agent-memory/memory-sdk-ts-v2";
import { sanitizeText, stripCodeBlocks, shouldCaptureL0 } from "../sanitize.js";

const TAG = "[memory-client-v3][capture]";

interface Logger {
  debug?: (msg: string) => void;
  info: (msg: string) => void;
  warn: (msg: string) => void;
  error: (msg: string) => void;
}

interface ExtractedMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface CaptureContext {
  /** sessionKey from agent_end ctx — used as conversation key on the server. */
  sessionKey: string;
  /** sessionId from agent_end ctx — passed to addConversation when present. */
  sessionId?: string;
  /** Raw event.messages array (full session history at agent_end time). */
  rawMessages: unknown[];
  /** Clean original user prompt (cached at before_prompt_build, pre-pollution). */
  originalUserText?: string;
  /**
   * Number of messages in the session at before_prompt_build time.
   * Used to position-slice rawMessages so we only re-send messages added in this turn.
   */
  originalUserMessageCount?: number;
  /**
   * Epoch ms cursor: only messages with timestamp > this are sent.
   * Used as a fallback when position slice is unavailable.
   */
  afterTimestamp?: number;
}

export interface CaptureResult {
  /** Number of messages actually sent to the gateway (post-filter). */
  capturedCount: number;
  /** Server-reported total count after this batch. */
  serverTotalCount?: number;
  /** Max timestamp among captured messages — caller should advance its cursor to this. */
  maxTimestamp?: number;
}

/**
 * Run a single agent_end capture.
 *
 * Returns immediately with `capturedCount: 0` if there is nothing to send.
 * Errors from the gateway are thrown — caller should wrap in try/catch.
 */
export async function performCapture(
  client: MemoryClient,
  ctx: CaptureContext,
  logger?: Logger,
): Promise<CaptureResult> {
  const { sessionKey, sessionId, rawMessages, originalUserText, originalUserMessageCount, afterTimestamp } = ctx;

  // ── Step 1. Position slice ──
  // Only consider messages added AFTER before_prompt_build, i.e. this turn's input.
  const usePositionSlice =
    originalUserMessageCount != null &&
    originalUserMessageCount > 0 &&
    originalUserMessageCount <= rawMessages.length;
  const slicedMessages = usePositionSlice
    ? rawMessages.slice(originalUserMessageCount)
    : rawMessages;

  if (usePositionSlice) {
    logger?.debug?.(
      `${TAG} Position slice: ${rawMessages.length} raw → ${slicedMessages.length} new ` +
      `(sliceStart=${originalUserMessageCount})`,
    );
  }

  // ── Step 2. Extract user/assistant messages ──
  const allExtracted = extractUserAssistantMessages(slicedMessages);
  logger?.debug?.(
    `${TAG} Extracted ${allExtracted.length} user/assistant messages from ${slicedMessages.length} raw`,
  );

  // ── Step 3. Timestamp cursor (fallback when position slice unavailable) ──
  const cursor = afterTimestamp ?? 0;
  const filteredByTime = cursor !== 0
    ? allExtracted.filter((m) => m.timestamp > cursor)
    : allExtracted;

  if (cursor > 0) {
    logger?.debug?.(
      `${TAG} Timestamp filter: ${allExtracted.length} → ${filteredByTime.length} (cursor=${cursor})`,
    );
  }

  if (filteredByTime.length === 0) {
    logger?.debug?.(`${TAG} No new messages to capture`);
    return { capturedCount: 0 };
  }

  // ── Step 4. Replace polluted user message with cached original ──
  // The framework appends the user's message AFTER before_prompt_build and
  // injects prependContext into it. Without this swap, the captured user
  // text would contain the recall blob, causing a feedback loop.
  if (originalUserText) {
    const targetRaw = usePositionSlice
      ? (slicedMessages[0] as Record<string, unknown> | undefined)
      : (originalUserMessageCount != null && originalUserMessageCount >= 0 && originalUserMessageCount < rawMessages.length)
        ? (rawMessages[originalUserMessageCount] as Record<string, unknown> | undefined)
        : undefined;
    const targetTs = typeof targetRaw?.timestamp === "number" ? targetRaw.timestamp : undefined;

    if (targetTs != null) {
      let replaced = false;
      for (let i = 0; i < filteredByTime.length; i++) {
        if (filteredByTime[i].role === "user" && filteredByTime[i].timestamp === targetTs) {
          logger?.debug?.(
            `${TAG} Replacing polluted user message (ts=${targetTs}, ` +
            `${filteredByTime[i].content.length}→${originalUserText.length} chars)`,
          );
          filteredByTime[i] = { ...filteredByTime[i], content: originalUserText };
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        logger?.warn?.(`${TAG} Could not match cached prompt to any extracted user message — relying on sanitizeText()`);
      }
    }
  }

  // ── Step 5. Sanitize + strip code + filter ──
  const cleaned = filteredByTime
    .map((m) => {
      let content = sanitizeText(m.content);
      if (m.role === "assistant") content = stripCodeBlocks(content);
      return { role: m.role, content, timestamp: m.timestamp };
    })
    .filter((m) => shouldCaptureL0(m.content));

  logger?.debug?.(
    `${TAG} After sanitize+filter: ${cleaned.length} messages (from ${filteredByTime.length})`,
  );

  if (cleaned.length === 0) {
    logger?.info(`${TAG} All messages filtered out, skipping POST`);
    return { capturedCount: 0 };
  }

  // ── Step 6. POST to gateway ──
  const result = await client.addConversation({
    session_id: sessionId ?? sessionKey,
    messages: cleaned.map((m) => ({
      role: m.role,
      content: m.content,
      timestamp: new Date(m.timestamp).toISOString(),
    })),
  });

  const maxTimestamp = Math.max(...cleaned.map((m) => m.timestamp));
  logger?.info(
    `${TAG} Captured ${cleaned.length} message(s) (server total=${result.total_count}, ` +
    `sessionKey=${sessionKey.slice(0, 32)}${sessionKey.length > 32 ? "…" : ""})`,
  );

  return {
    capturedCount: cleaned.length,
    serverTotalCount: result.total_count,
    maxTimestamp,
  };
}

/**
 * Extract user/assistant entries from the framework's raw message array.
 *
 * Handles both content shapes the framework may produce:
 *   - `content: string`
 *   - `content: Array<{ type: "text", text: string } | ...>`
 *
 * Strips inline base64 image data URIs (replaces with `[image]`) so they do
 * not bloat the request payload or pollute downstream FTS / embeddings.
 */
function extractUserAssistantMessages(messages: unknown[]): ExtractedMessage[] {
  const result: ExtractedMessage[] = [];

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    const role = m.role as string | undefined;
    if (role !== "user" && role !== "assistant") continue;

    let content: string | undefined;
    if (typeof m.content === "string") {
      content = m.content;
    } else if (Array.isArray(m.content)) {
      const parts: string[] = [];
      for (const part of m.content) {
        if (part && typeof part === "object" && (part as Record<string, unknown>).type === "text") {
          const text = (part as Record<string, unknown>).text;
          if (typeof text === "string") parts.push(text);
        }
      }
      content = parts.join("\n");
    }

    if (content && /data:image\/[a-z+]+;base64,/i.test(content)) {
      content = content.replace(/data:image\/[a-z+]+;base64,[A-Za-z0-9+/=]+/gi, "[image]");
    }

    if (content && content.trim()) {
      const ts = typeof m.timestamp === "number" ? m.timestamp : Date.now();
      result.push({
        role: role as "user" | "assistant",
        content: content.trim(),
        timestamp: ts,
      });
    }
  }

  return result;
}
