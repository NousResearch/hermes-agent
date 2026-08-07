/**
 * L0 Conversation Recorder: records raw conversation messages to local JSONL files.
 *
 * Triggered from agent_end hook. Receives the conversation messages directly from
 * the hook context (no file I/O needed), sanitizes them, filters out noise, and
 * writes to ~/.openclaw/memory-tdai/conversations/YYYY-MM-DD.jsonl
 *
 * Design decisions:
 * - Uses JSONL format (**one message per line** — flat, easy to grep/stream)
 * - One file per day (all sessions merged into the same daily file)
 * - sessionKey is stored as a field in each JSONL line, not in the filename
 * - Independent from system session files — format fully controlled by plugin
 * - Messages are sanitized to remove injected tags (prevent feedback loops)
 * - Short/long/command messages are filtered out
 */

import crypto from "node:crypto";
import { sanitizeText, stripCodeBlocks, shouldCaptureL0 } from "../../utils/sanitize.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";
import type { Logger } from "../types.js";
import { DEFAULT_ISOLATION_ID } from "../store/types.js";

// ============================
// Types
// ============================

export interface ConversationMessage {
  /** Unique message ID (used by L1 prompt for source_message_ids tracking) */
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number; // epoch ms
}

/**
 * Generate a short unique message ID.
 */
function generateMessageId(): string {
  return `msg_${Date.now()}_${crypto.randomBytes(3).toString("hex")}`;
}

/**
 * New flat format: one message per JSONL line.
 */
export interface L0MessageRecord {
  sessionKey: string;
  sessionId: string;
  /** Three-dim tenancy isolation. Optional during rollout — see
   *  docs/l0l3-tenant-isolation-design.md. */
  userId?: string;
  agentId?: string;
  recordedAt: string; // ISO timestamp
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number; // epoch ms
}

/**
 * A group of conversation messages (used by downstream consumers).
 * Each L0ConversationRecord represents one or more messages from the same recording event.
 */
export interface L0ConversationRecord {
  sessionKey: string;
  sessionId: string;
  recordedAt: string; // ISO timestamp
  messageCount: number;
  messages: ConversationMessage[];
}

const TAG = "[memory-tdai][l0]";

// ============================
// Core function
// ============================

/**
 * Record a conversation round to the L0 JSONL file.
 *
 * Only records **incremental** messages (new since the last capture).
 * Uses `afterTimestamp` as the primary filter to skip already-captured history.
 *
 * @param sessionKey - The session key for this conversation
 * @param rawMessages - Raw messages from the agent_end hook context (full session history)
 * @param baseDir - Base data directory (~/.openclaw/memory-tdai/)
 * @param logger - Optional logger
 * @param originalUserText - Clean original user prompt (pre-prependContext)
 * @param afterTimestamp - Epoch ms cursor: only messages with timestamp > this are new.
 *                         Pass 0 or omit for the first capture of a session.
 * @returns Filtered messages (for L1 to use directly), or empty array if nothing worth recording
 */
export async function recordConversation(params: {
  sessionKey: string;
  sessionId?: string;
  /** Three-dim tenancy isolation, propagated into every L0 record written. */
  userId?: string;
  agentId?: string;
  rawMessages: unknown[];
  baseDir: string;
  logger?: Logger;
  /** Clean original user prompt (pre-prependContext) */
  originalUserText?: string;
  /** Epoch ms cursor: only process messages with timestamp strictly greater than this. */
  afterTimestamp?: number;
  /**
   * Number of messages in the session at before_prompt_build time.
   * Used to locate the exact user message that originalUserText corresponds to:
   * rawMessages[originalUserMessageCount] is the user message appended by the framework
   * AFTER before_prompt_build, i.e. the one whose content was polluted by prependContext.
   */
  originalUserMessageCount?: number;
  /** StorageAdapter for file operations (COS/local). Falls back to fs when absent. */
  storage?: StorageAdapter;
}): Promise<ConversationMessage[]> {
  const { sessionKey, sessionId, userId, agentId, rawMessages, baseDir, logger, originalUserText, afterTimestamp, originalUserMessageCount, storage } = params;

  // Step 1: Position slice + extract user/assistant messages.
  //
  // Dual protection against duplicate capture:
  //   Layer 1 (position slice): Use originalUserMessageCount (cached at before_prompt_build)
  //     to slice rawMessages — only keep messages added AFTER the prompt build, i.e. this
  //     turn's new messages. This is immune to timestamp drift after gateway restarts.
  //   Layer 2 (timestamp cursor): The existing afterTimestamp filter below acts as a fallback
  //     when the position slice is unavailable (cache expired, process restart, etc.).
  const usePositionSlice = originalUserMessageCount != null && originalUserMessageCount > 0
    && originalUserMessageCount <= rawMessages.length;
  const slicedMessages = usePositionSlice
    ? rawMessages.slice(originalUserMessageCount)
    : rawMessages;

  const allExtracted = extractUserAssistantMessages(slicedMessages);

  if (usePositionSlice) {
    logger?.debug?.(
      `${TAG} Position slice: ${rawMessages.length} raw → ${slicedMessages.length} new (sliceStart=${originalUserMessageCount})`,
    );
  }

  // Diagnostic: check whether the framework actually provides timestamp on raw messages.
  // If all raw timestamps are missing, the timestamp cursor is effectively useless and
  // position slice becomes the sole incremental mechanism.
  if (slicedMessages.length > 0) {
    const firstRaw = slicedMessages[0] as Record<string, unknown> | undefined;
    const rawTs = firstRaw?.timestamp;
    const hasRawTs = typeof rawTs === "number";
    logger?.debug?.(
      `${TAG} Raw message[0] timestamp probe: ${hasRawTs ? `present (${rawTs})` : `missing (type=${typeof rawTs}, value=${String(rawTs)})`}`,
    );
  }

  logger?.debug?.(`${TAG} Extracted ${allExtracted.length} user/assistant messages from ${slicedMessages.length} total`);

  // Step 1.5: Incremental filter — only keep messages newer than the cursor.
  //
  // Uses strict greater-than (>) which is safe because:
  //   - The cursor is set to max(timestamps) of the LAST recorded batch.
  //   - The next agent turn's messages will have timestamps strictly greater than
  //     the previous turn (there's at least one LLM API call between turns, which
  //     takes hundreds of milliseconds minimum — no same-millisecond collision).
  //   - All messages within a single turn are captured together as one batch,
  //     so even if multiple messages share the same timestamp, they are either
  //     all included (new batch) or all excluded (already captured).
  //   - If a message lacks a timestamp field, extractUserAssistantMessages()
  //     assigns Date.now() at extraction time, which is always > previous cursor.
  const cursor = afterTimestamp ?? 0;
  const extracted = cursor !== 0
    ? allExtracted.filter((m) => m.timestamp > cursor)
    : allExtracted;

  if (extracted.length > 0) {
    const first = extracted[0];
    logger?.debug?.(
      `${TAG} First captured message: role=${first.role}, ts=${first.timestamp}, ` +
      `date=${new Date(first.timestamp).toISOString()}, content=${first.content.slice(0, 80)}${first.content.length > 80 ? "…" : ""}`,
    );
  }

  if (cursor > 0) {
    logger?.debug?.(
      `${TAG} Incremental filter: ${allExtracted.length} total → ${extracted.length} new (cursor=${cursor})`,
    );

    // Safety valve: if timestamp filter passed everything through and position slice
    // was not available, this likely indicates timestamp drift after a gateway restart.
    if (!usePositionSlice && extracted.length === allExtracted.length && allExtracted.length > 8) {
      logger?.warn?.(
        `${TAG} ⚠ Safety valve: all ${allExtracted.length} messages passed timestamp filter (cursor=${cursor}) — ` +
        `possible timestamp drift after gateway restart. Position slice was not available (no cached messageCount).`,
      );
    }
  }

  if (extracted.length === 0) {
    logger?.debug?.(`${TAG} No new user/assistant messages to record`);
    return [];
  }

  // Step 2: Replace polluted user messages with cached original prompt.
  //
  // Background:
  //   The framework appends the user's message to the session after before_prompt_build,
  //   then injects prependContext into it. So the user message in rawMessages is polluted.
  //   We cached the clean prompt (originalUserText) and the message count at
  //   before_prompt_build time (originalUserMessageCount) to identify which raw message
  //   is the real user input.
  //
  // Strategy:
  //   When position slice is active, the polluted user message is slicedMessages[0].
  //   Otherwise, fall back to rawMessages[originalUserMessageCount].
  //   In both cases, find the timestamp and match it in `extracted` for replacement.
  //   If matching fails, skip replacement — sanitizeText() in Step 3 is the safety net.
  if (originalUserText) {
    // Determine the target raw message that contains the polluted user prompt
    const targetRaw: Record<string, unknown> | undefined = usePositionSlice
      ? slicedMessages[0] as Record<string, unknown> | undefined
      : (originalUserMessageCount != null && originalUserMessageCount >= 0 && originalUserMessageCount < rawMessages.length)
        ? rawMessages[originalUserMessageCount] as Record<string, unknown> | undefined
        : undefined;

    const targetTs = targetRaw && typeof targetRaw.timestamp === "number" ? targetRaw.timestamp : undefined;

    if (targetTs != null) {
      let replaced = false;
      for (let i = 0; i < extracted.length; i++) {
        if (extracted[i].role === "user" && extracted[i].timestamp === targetTs) {
          logger?.debug?.(
            `${TAG} Replacing user message at timestamp=${targetTs} with cached original prompt ` +
            `(${originalUserText.length} chars, was ${extracted[i].content.length} chars) [positionSlice=${usePositionSlice}]`,
          );
          extracted[i] = { ...extracted[i], content: originalUserText };
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        logger?.warn?.(
          `${TAG} Target user message (ts=${targetTs}) not found in extracted batch — ` +
          `possibly filtered by cursor. Skipping replacement, will rely on sanitizeText().`,
        );
      }
    } else if (targetRaw) {
      logger?.warn?.(
        `${TAG} Target raw message has no valid timestamp — ` +
        `skipping replacement, will rely on sanitizeText().`,
      );
    } else {
      logger?.warn?.(
        `${TAG} Have originalUserText but cannot locate target raw message — ` +
        `skipping replacement, will rely on sanitizeText().`,
      );
    }
  }

  // Step 3: Sanitize and filter
  const filtered = extracted
    .map((m) => {
      let content = sanitizeText(m.content);
      // Strip fenced code blocks from assistant replies to reduce embedding noise
      if (m.role === "assistant") {
        content = stripCodeBlocks(content);
      }
      return { id: m.id, role: m.role, content, timestamp: m.timestamp };
    })
    .filter((m) => shouldCaptureL0(m.content));

  logger?.debug?.(`${TAG} After sanitize+filter: ${filtered.length} messages (from ${extracted.length})`);

  if (filtered.length === 0) {
    logger?.debug?.(`${TAG} All messages filtered out, skipping L0 write`);
    return [];
  }

  // Step 4: Write to JSONL file — one message per line (flat format)
  const now = new Date().toISOString();
  const lines: string[] = [];
  for (const msg of filtered) {
    const record: L0MessageRecord = {
      sessionKey,
      sessionId: sessionId || DEFAULT_ISOLATION_ID,
      // Tenancy isolation — missing fields go to the default compatibility bucket.
      userId: userId || DEFAULT_ISOLATION_ID,
      agentId: agentId || DEFAULT_ISOLATION_ID,
      recordedAt: now,
      id: msg.id,
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp,
    };
    lines.push(JSON.stringify(record));
  }

  const shardDate = formatLocalDate(new Date());
  const recordKey = StoragePaths.conversation(shardDate);

  try {
    if (storage) {
      await storage.appendFile(recordKey, lines.join("\n") + "\n");
    } else {
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const outDir = path.default.join(baseDir, "conversations");
      const outPath = path.default.join(outDir, `${shardDate}.jsonl`);
      await fs.default.mkdir(outDir, { recursive: true });
      await fs.default.appendFile(outPath, lines.join("\n") + "\n", "utf-8");
    }
    logger?.debug?.(`${TAG} Recorded ${filtered.length} messages to ${recordKey}`);
  } catch (err) {
    logger?.error(`${TAG} Failed to write L0 file: ${err instanceof Error ? err.message : String(err)}`);
    // Return filtered messages anyway so L1 can still process them
  }

  return filtered;
}

/**
 * Read all L0 conversation records for a session.
 * Returns records in chronological order.
 *
 * File format: `YYYY-MM-DD.jsonl` (daily files, all sessions merged).
 * Each line is an L0MessageRecord; filtered by sessionKey at line level.
 */
export async function readConversationRecords(
  sessionKey: string,
  baseDir: string,
  logger?: Logger,
  storage?: StorageAdapter,
): Promise<L0ConversationRecord[]> {
  // Daily file pattern: YYYY-MM-DD.jsonl
  const dateFilePattern = /^\d{4}-\d{2}-\d{2}\.jsonl$/;

  let entries: string[];
  try {
    if (storage) {
      entries = await storage.readdirNames(StoragePaths.conversationsDir, ".jsonl");
    } else {
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const conversationsDir = path.default.join(baseDir, "conversations");
      const dirEntries = await fs.default.readdir(conversationsDir, { withFileTypes: true });
      entries = dirEntries
        .filter((entry) => entry.isFile())
        .map((entry) => entry.name);
    }
  } catch {
    // Directory doesn't exist yet — normal for first conversation
    return [];
  }

  const targetFiles = entries
    .filter((name) => dateFilePattern.test(name))
    .sort();

  if (targetFiles.length === 0) {
    return [];
  }

  const records: L0ConversationRecord[] = [];

  for (const fileName of targetFiles) {
    let raw: string | null;
    try {
      if (storage) {
        raw = await storage.readFile(`${StoragePaths.conversationsDir}${fileName}`);
      } else {
        const fs = await import("node:fs/promises");
        const path = await import("node:path");
        raw = await fs.default.readFile(path.default.join(baseDir, "conversations", fileName), "utf-8");
      }
    } catch {
      logger?.warn?.(`${TAG} Failed to read L0 file: ${fileName}`);
      continue;
    }

    if (!raw) continue;

    const lines = raw.split("\n").filter((line: string) => line.trim());
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>;

        // Filter by sessionKey at line level
        const lineSessionKey = parsed.sessionKey as string | undefined;
        if (lineSessionKey !== sessionKey) continue;

        if (typeof parsed.role === "string" && typeof parsed.content === "string") {
          // Flat format: { sessionKey, sessionId, recordedAt, id, role, content, timestamp }
          // Wrap into L0ConversationRecord for uniform downstream consumption
          const msg: ConversationMessage = {
            id: (typeof parsed.id === "string" && parsed.id) ? parsed.id : generateMessageId(),
            role: parsed.role as "user" | "assistant",
            content: parsed.content as string,
            timestamp: typeof parsed.timestamp === "number" ? parsed.timestamp : Date.now(),
          };
          records.push({
            sessionKey: (parsed.sessionKey as string) || sessionKey,
            sessionId: (parsed.sessionId as string) || DEFAULT_ISOLATION_ID,
            recordedAt: (parsed.recordedAt as string) || new Date().toISOString(),
            messageCount: 1,
            messages: [msg],
          });
        } else {
          logger?.warn?.(`${TAG} Unrecognized JSONL line format in ${fileName}:${i + 1}`);
        }
      } catch {
        logger?.warn?.(`${TAG} Skipping malformed JSONL line in ${fileName}:${i + 1}`);
      }
    }
  }

  records.sort((a, b) => {
    const ta = Date.parse(a.recordedAt);
    const tb = Date.parse(b.recordedAt);
    const na = Number.isFinite(ta) ? ta : Number.POSITIVE_INFINITY;
    const nb = Number.isFinite(tb) ? tb : Number.POSITIVE_INFINITY;
    return na - nb;
  });

  return records;
}

/**
 * Read L0 messages across all conversation records for a session,
 * optionally filtered by a cursor timestamp (messages after the cursor).
 *
 * When `limit` is provided, only the **newest** `limit` messages are returned
 * (matching the DB path's `ORDER BY timestamp DESC LIMIT ?` behavior).
 * Returned messages are always in chronological order (oldest → newest).
 *
 * NOTE: potential optimization — records are chronologically ordered (append-only JSONL),
 * so a reverse scan could skip entire old records. Deferred for now; see Issue 5 in
 * docs/05-known-issues.md.
 */
export async function readConversationMessages(
  sessionKey: string,
  baseDir: string,
  afterTimestamp?: number,
  logger?: Logger,
  limit?: number,
): Promise<ConversationMessage[]> {
  const records = await readConversationRecords(sessionKey, baseDir, logger);
  const allMessages: ConversationMessage[] = [];

  for (const record of records) {
    for (const msg of record.messages) {
      if (afterTimestamp && msg.timestamp <= afterTimestamp) continue;
      allMessages.push(msg);
    }
  }

  // Truncate to newest `limit` messages (keep tail, since array is chronological)
  if (limit != null && limit > 0 && allMessages.length > limit) {
    logger?.debug?.(
      `${TAG} readConversationMessages: truncating ${allMessages.length} → ${limit} (newest)`,
    );
    return allMessages.slice(-limit);
  }

  return allMessages;
}

/**
 * A group of conversation messages sharing the same sessionId.
 */
export interface SessionIdMessageGroup {
  sessionId: string;
  messages: Array<ConversationMessage & { recordedAtMs: number }>;
}

/**
 * Read L0 messages for a session, grouped by sessionId.
 *
 * Within the same sessionKey, different sessionIds represent different conversation
 * instances (e.g. after /reset). L1 extraction should process each group independently
 * so that each group's sessionId is correctly associated with its extracted memories.
 *
 * When `limit` is provided, only the **newest** `limit` messages (across all groups)
 * are retained — matching the DB path's `ORDER BY recorded_at DESC LIMIT ?` behavior.
 * Groups that become empty after truncation are dropped.
 *
 * Groups are returned in chronological order (by earliest message timestamp).
 * Messages within each group are also in chronological order.
 *
 * @param afterRecordedAtMs - Epoch ms cursor: only messages with recordedAt > this are included.
 */
export async function readConversationMessagesGroupedBySessionId(
  sessionKey: string,
  baseDir: string,
  afterRecordedAtMs?: number,
  logger?: Logger,
  limit?: number,
): Promise<SessionIdMessageGroup[]> {
  const records = await readConversationRecords(sessionKey, baseDir, logger);

  // Collect all messages with their sessionId, filtering by recorded_at cursor
  const allMessages: Array<{ sessionId: string; msg: ConversationMessage & { recordedAtMs: number } }> = [];

  for (const record of records) {
    const sid = record.sessionId || "";
    const recMs = Date.parse(record.recordedAt) || 0;
    if (afterRecordedAtMs && recMs <= afterRecordedAtMs) continue;
    for (const msg of record.messages) {
      allMessages.push({ sessionId: sid, msg: { ...msg, recordedAtMs: recMs } });
    }
  }

  // Sort by timestamp ASC (chronological) — records are already roughly ordered
  // by recordedAt, but messages within may not be perfectly sorted by timestamp.
  allMessages.sort((a, b) => a.msg.timestamp - b.msg.timestamp);

  // Truncate to newest `limit` messages (keep tail)
  let selected = allMessages;
  if (limit != null && limit > 0 && allMessages.length > limit) {
    logger?.debug?.(
      `${TAG} readConversationMessagesGroupedBySessionId: truncating ${allMessages.length} → ${limit} (newest)`,
    );
    selected = allMessages.slice(-limit);
  }

  // Re-group by sessionId
  const groupMap = new Map<string, Array<ConversationMessage & { recordedAtMs: number }>>();
  for (const { sessionId, msg } of selected) {
    let group = groupMap.get(sessionId);
    if (!group) {
      group = [];
      groupMap.set(sessionId, group);
    }
    group.push(msg);
  }

  // Convert to array, sorted by earliest message timestamp in each group
  const groups: SessionIdMessageGroup[] = [];
  for (const [sessionId, messages] of groupMap) {
    if (messages.length > 0) {
      groups.push({ sessionId, messages });
    }
  }
  groups.sort((a, b) => a.messages[0].timestamp - b.messages[0].timestamp);

  return groups;
}

// ============================
// Helpers
// ============================

/**
 * Extract user and assistant messages from raw hook message array.
 */
function extractUserAssistantMessages(messages: unknown[]): ConversationMessage[] {
  const result: ConversationMessage[] = [];

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    const role = m.role as string | undefined;

    if (role !== "user" && role !== "assistant") continue;

    let content: string | undefined;
    if (typeof m.content === "string") {
      content = m.content;
    } else if (Array.isArray(m.content)) {
      const textParts: string[] = [];
      for (const part of m.content) {
        if (
          part &&
          typeof part === "object" &&
          (part as Record<string, unknown>).type === "text"
        ) {
          const text = (part as Record<string, unknown>).text;
          if (typeof text === "string") textParts.push(text);
        }
      }
      content = textParts.join("\n");
    }

    // Strip inline base64 image data URIs that some providers embed in string content.
    // These are not useful for memory and would pollute FTS / embedding indexes.
    if (content && /data:image\/[a-z+]+;base64,/i.test(content)) {
      content = content.replace(/data:image\/[a-z+]+;base64,[A-Za-z0-9+/=]+/gi, "[image]");
    }

    if (content && content.trim()) {
      const ts = typeof m.timestamp === "number" ? m.timestamp : Date.now();
      result.push({
        id: (typeof m.id === "string" && m.id) ? m.id : generateMessageId(),
        role: role as "user" | "assistant",
        content: content.trim(),
        timestamp: ts,
      });
    }
  }

  return result;
}

/**
 * Format local date as YYYY-MM-DD.
 */
function formatLocalDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
