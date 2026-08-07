/**
 * offload-client — OffloadContextEngine.
 * Occupies the Context Engine slot and delegates compression to the server.
 */
import type { OffloadClientConfig, RecentMessage, Logger } from "./types.js";
import type { OffloadApiClient } from "./offload-api-client.js";
import { estimateAllTokens, estimateMessageTokens } from "./token-estimator.js";

const DEFAULT_CONTEXT_WINDOW = 128000;
/** compact target: keep messages until total <= contextWindow * TARGET_RATIO */
const COMPACT_TARGET_RATIO = 0.5;
/** When truncating a single large tool_result, keep at most this many chars */
const TOOL_RESULT_TRUNCATE_CHARS = 2000;

// ─── Message role helpers (handle multiple formats) ─────────────────────────

function getMsgRole(msg: any): string {
  return msg?.role ?? msg?.message?.role ?? msg?.type ?? "";
}

function isToolResult(msg: any): boolean {
  const role = getMsgRole(msg);
  if (role === "tool" || role === "toolResult" || role === "tool_result") return true;
  // Anthropic: user message with tool_result content blocks
  if (role === "user" && Array.isArray(msg?.content)) {
    return msg.content.some((b: any) => b?.type === "tool_result");
  }
  return false;
}

function isAssistantWithToolUse(msg: any): boolean {
  const role = getMsgRole(msg);
  if (role !== "assistant") return false;
  const content = msg?.type === "message" ? msg?.message?.content : msg?.content;
  if (!Array.isArray(content)) return false;
  return content.some((b: any) => b?.type === "tool_use" || b?.type === "toolCall");
}

/**
 * Truncate tool_result content in-place, returning a shallow-cloned message.
 */
function truncateToolResult(msg: any, maxChars: number): any {
  const clone = JSON.parse(JSON.stringify(msg));
  const content = clone.type === "message" ? clone.message?.content : clone.content;
  if (typeof content === "string" && content.length > maxChars) {
    const truncated = content.slice(0, maxChars) + "\n...[truncated]";
    if (clone.type === "message") clone.message.content = truncated;
    else clone.content = truncated;
    return clone;
  }
  if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block?.text === "string" && block.text.length > maxChars) {
        block.text = block.text.slice(0, maxChars) + "\n...[truncated]";
      }
      if (typeof block?.content === "string" && block.content.length > maxChars) {
        block.content = block.content.slice(0, maxChars) + "\n...[truncated]";
      }
    }
    return clone;
  }
  return clone;
}

export class OffloadContextEngine {
  /** Per-session state cache. */
  private sessions = new Map<string, {
    lastAccessMs: number;  // ← NEW: track last access time for cleanup
    lastKnownTotalTokens: number;
    lastKnownMsgCount: number;
    lastL15PromptHash: string;
    cachedPrompt?: string;
    cachedRecentMessages: RecentMessage[];
    cachedRecentContext?: string;
  }>();

  /** Get or create per-session state. */
  private getSession(sessionKey: string) {
    let s = this.sessions.get(sessionKey);
    if (!s) {
      s = {
        lastAccessMs: Date.now(),  // ← NEW
        lastKnownTotalTokens: 0,
        lastKnownMsgCount: 0,
        lastL15PromptHash: "",
        cachedRecentMessages: [],
      };
      this.sessions.set(sessionKey, s);
    } else {
      s.lastAccessMs = Date.now();  // ← NEW: update on access
    }
    return s;
  }

  /**
   * Reset session state (call when session is /new'd or destroyed).
   */
  resetSession(sessionKey: string): void {
    this.sessions.delete(sessionKey);
  }

  /**
   * Clear all session states (emergency shutdown).
   */
  clearAllSessions(): void {
    const n = this.sessions.size;
    this.sessions.clear();
    if (n > 0) {
      this.logger.info(`[offload-client] cleared ${n} session states`);
    }
  }

  /**
   * Get the cached context for a session (for after_tool_call hook to send with ingest).
   */
  getContext(sessionKey?: string): { prompt?: string; recentMessages?: RecentMessage[] } | undefined {
    if (!sessionKey) return undefined;
    const s = this.sessions.get(sessionKey);
    if (!s || (!s.cachedPrompt && s.cachedRecentMessages.length === 0)) return undefined;
    return { prompt: s.cachedPrompt, recentMessages: s.cachedRecentMessages };
  }

  /**
   * Get the cached formatted context string for a session (for legacy compatibility).
   */
  getRecentContext(sessionKey?: string): string | undefined {
    if (!sessionKey) return undefined;
    return this.sessions.get(sessionKey)?.cachedRecentContext;
  }

  constructor(
    private client: OffloadApiClient,
    private config: OffloadClientConfig,
    private logger: Logger,
  ) {}

  get info() {
    return {
      id: "memory-tencentdb",
      name: "Offload Client Context Engine",
      version: "2.0.0",
      ownsCompaction: true,
    };
  }

  /**
   * bootstrap — called when a new session starts (e.g. /new command).
   * Resets per-session cached state.
   */
  async bootstrap(params: { sessionKey?: string; sessionId?: string }) {
    const sk = params.sessionKey ?? params.sessionId;
    if (sk) this.resetSession(sk);
    return { bootstrapped: true };
  }

  /**
   * ingest — no-op for client mode (ingest is handled by after_tool_call hook).
   * Required by the framework ContextEngine interface.
   */
  async ingest(_params: {
    sessionId: string;
    sessionKey?: string;
    message: any;
    isHeartbeat?: boolean;
  }) {
    return { ingested: true };
  }

  /**
   * compact — record the framework's authoritative token count for calibration,
   * then defer actual compaction to assemble().
   */
  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: any;
  }) {
    // Record framework token count for calibration
    if (params.currentTokenCount && params.currentTokenCount > 0) {
      const sk = params.sessionKey ?? params.sessionId;
      const s = this.getSession(sk);
      s.lastKnownTotalTokens = params.currentTokenCount;
      this.logger.debug?.(
        `[offload-client] compact: calibration updated, knownTokens=${s.lastKnownTotalTokens}`,
      );
    }

    const contextWindow = params.tokenBudget ?? DEFAULT_CONTEXT_WINDOW;
    const targetTokens = Math.floor(contextWindow * COMPACT_TARGET_RATIO);

    this.logger.info(
      `[offload-client] compact: sessionKey=${params.sessionKey ?? params.sessionId}, ` +
      `budget=${contextWindow}, target=${targetTokens}, ` +
      `currentTokens=${params.currentTokenCount}, force=${params.force}`,
    );

    return {
      ok: true,
      compacted: false,
      reason: "offload-client: compaction deferred to assemble()",
    };
  }

  /**
   * Resolve the best known token total for calibration.
   * Uses lastKnownTotalTokens only if message count hasn't changed significantly.
   */
  private resolveCalibrationTokens(sessionKey: string, msgCount: number): number | undefined {
    const s = this.sessions.get(sessionKey);
    if (!s || s.lastKnownTotalTokens <= 0) return undefined;
    // If message count changed by >20%, the cached total is stale — skip calibration
    if (s.lastKnownMsgCount > 0 && Math.abs(msgCount - s.lastKnownMsgCount) / s.lastKnownMsgCount > 0.2) {
      return undefined;
    }
    return s.lastKnownTotalTokens;
  }

  /**
   * Local brute-force compaction: keep tail messages up to target budget,
   * respecting tool pairs and truncating oversized tool_results.
   * Used as fallback when server compaction is unavailable.
   */
  private localCompact(messages: any[], contextWindow: number, sessionKey: string): any[] {
    const targetTokens = Math.floor(contextWindow * COMPACT_TARGET_RATIO);
    const knownTokens = this.resolveCalibrationTokens(sessionKey, messages.length);
    const { perMessage } = estimateAllTokens(messages, knownTokens);
    const n = messages.length;

    // Step 1: scan from tail, find the cut index
    let cumTokens = 0;
    let cutIdx = n; // everything before cutIdx gets deleted
    for (let i = n - 1; i >= 0; i--) {
      cumTokens += perMessage[i];
      if (cumTokens > targetTokens) {
        cutIdx = i + 1;
        break;
      }
      cutIdx = i;
    }

    // Never delete the very first user message (index 0)
    if (cutIdx <= 0) cutIdx = 0;

    // Step 2: expand cut boundary to respect tool pairs
    // If cutIdx lands inside a tool pair, move it to include the full pair.

    // 2a: If msg at cutIdx is a tool_result, its paired assistant+tool_use is
    //     before cutIdx (would be deleted). Move cutIdx back to include the pair.
    while (cutIdx < n && isToolResult(messages[cutIdx])) {
      cutIdx++;
    }

    // 2b: If msg at cutIdx-1 (last deleted) is assistant+tool_use, its tool_result
    //     at cutIdx would be orphaned. Pull cutIdx back to keep the pair.
    while (cutIdx > 0 && cutIdx < n && isAssistantWithToolUse(messages[cutIdx - 1])) {
      cutIdx--;
    }

    // Step 3: build retained array
    const retained = messages.slice(cutIdx);

    if (retained.length === 0) {
      return [...messages]; // safety: don't delete everything
    }

    const deletedCount = cutIdx;
    let retainedTokens = 0;
    for (let i = cutIdx; i < n; i++) retainedTokens += perMessage[i];

    this.logger.info(
      `[offload-client] localCompact: deleted ${deletedCount}/${n} msgs, ` +
      `retained ${retained.length} msgs, tokens=${retainedTokens}/${targetTokens} target`,
    );

    // Step 4: if still over target and there's a large tool_result, truncate it
    if (retainedTokens > targetTokens) {
      let maxTrIdx = -1;
      let maxTrTokens = 0;
      for (let i = 0; i < retained.length; i++) {
        if (isToolResult(retained[i])) {
          const t = estimateMessageTokens(retained[i]);
          if (t > maxTrTokens) {
            maxTrTokens = t;
            maxTrIdx = i;
          }
        }
      }
      if (maxTrIdx >= 0 && maxTrTokens > TOOL_RESULT_TRUNCATE_CHARS / 4) {
        retained[maxTrIdx] = truncateToolResult(retained[maxTrIdx], TOOL_RESULT_TRUNCATE_CHARS);
        const newTokens = estimateMessageTokens(retained[maxTrIdx]);
        retainedTokens = retainedTokens - maxTrTokens + newTokens;
        this.logger.info(
          `[offload-client] localCompact: truncated tool_result[${maxTrIdx}] ` +
          `${maxTrTokens}→${newTokens} tokens, total now=${retainedTokens}`,
        );
      }
    }

    return retained;
  }

  // ─── L1.5 Trigger ───────────────────────────────────────────────────────────

  /**
   * Trigger L1.5 task judgment via ingest API when a new user prompt is detected.
   * Fire-and-forget — does not block the assemble flow.
   */
  private triggerL15IfNeeded(prompt: string | undefined, messages: any[], sessionKey: string): void {
    if (!prompt || typeof prompt !== "string" || prompt.length === 0) return;

    // Skip system/internal prompts that are not user-initiated
    if (this.isInternalPrompt(prompt)) {
      this.logger.debug?.(`[offload-client] L1.5 skipped: internal prompt (${prompt.slice(0, 60)})`);
      return;
    }

    // Always update cached context for after_tool_call hook (L1 needs it)
    const recentMsgs = this.buildRecentMessages(prompt, messages);
    const s = this.getSession(sessionKey);
    s.cachedPrompt = prompt.slice(0, 500);
    s.cachedRecentMessages = recentMsgs;
    s.cachedRecentContext = this.formatContextForL1(prompt, recentMsgs);

    // Dedup: skip L1.5 if same prompt as last trigger for this session
    const hash = this.simpleHash(prompt);
    if (s.lastL15PromptHash === hash) {
      this.logger.debug?.(`[offload-client] L1.5 skipped: same prompt hash (${hash})`);
      return;
    }
    s.lastL15PromptHash = hash;

    this.logger.info(
      `[offload-client] L1.5 triggered: promptHash=${hash}, recentMsgs=${recentMsgs.length}`,
    );

    // Fire-and-forget L1.5
    this.client.ingestL15(sessionKey, prompt.slice(0, 500), recentMsgs).catch((err) => {
      this.logger.warn(`[offload-client] L1.5 ingestL15 failed: ${err}`);
    });
  }

  /**
   * Detect internal/system prompts that should not trigger L1.5.
   * These are framework-generated messages, not user-initiated conversations.
   */
  private isInternalPrompt(prompt: string): boolean {
    // Compaction flush prompts
    if (prompt.startsWith("Pre-compaction")) return true;
    // Inter-session routing messages
    if (prompt.startsWith("[Inter-session message]")) return true;
    // Heartbeat/keepalive
    if (prompt.includes("HEARTBEAT") || prompt.includes("heartbeat")) return true;
    return false;
  }

  /**
   * Build structured RecentMessage[] for ingest API.
   * Filters: user/assistant text only, no tool calls, no heartbeats.
   * Max 5 recent turns, 400 chars per message.
   */
  private buildRecentMessages(prompt: string, messages: any[]): RecentMessage[] {
    const normalizedPrompt = prompt.trim().slice(0, 200).toLowerCase();

    // Scan messages, collect user/assistant pairs
    const pairs: RecentMessage[] = [];
    for (const msg of messages) {
      const role = getMsgRole(msg);

      // Skip tool messages entirely
      if (isToolResult(msg) || isAssistantWithToolUse(msg)) continue;
      if (role === "tool" || role === "toolResult" || role === "tool_result") continue;

      if (role === "user") {
        const text = this.extractMsgText(msg);
        if (!text || text.length <= 5) continue;
        if (text.includes("HEARTBEAT") || text.includes("heartbeat")) continue;
        const trimmed = text.slice(0, 400);
        // Skip if it matches current prompt
        const normalizedText = trimmed.slice(0, 200).toLowerCase();
        if (normalizedPrompt && (normalizedText === normalizedPrompt || normalizedText.startsWith(normalizedPrompt) || normalizedPrompt.startsWith(normalizedText))) continue;
        pairs.push({ role: "user", content: trimmed });
      } else if (role === "assistant") {
        const text = this.extractMsgText(msg);
        if (!text || text.length <= 10) continue;
        if (text.includes("HEARTBEAT") || text.includes("heartbeat")) continue;
        pairs.push({ role: "assistant", content: text.slice(0, 400) });
      }
    }

    // Keep last N messages (max 10 messages ≈ 5 turns)
    const recent = pairs.slice(-10);
    return recent;
  }

  /**
   * Format context string for L1 executor (recent-context.txt).
   */
  private formatContextForL1(prompt: string, recentMsgs: RecentMessage[]): string {
    const parts: string[] = [];
    if (recentMsgs.length > 0) {
      parts.push("历史消息，可作为参考：");
      for (const m of recentMsgs) {
        parts.push(`[${m.role === "user" ? "User" : "Assistant"}]: ${m.content}`);
      }
    }
    parts.push(`\n最新user message：\n[User]: ${prompt.slice(0, 500)}`);
    return parts.join("\n");
  }

  /**
   * Extract text content from a message.
   */
  private extractMsgText(msg: any): string {
    const content = msg?.content ?? msg?.message?.content ?? "";
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((b: any) => (typeof b === "string" ? b : b?.text ?? ""))
        .join("");
    }
    return "";
  }

  /**
   * Simple string hash for prompt deduplication.
   */
  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return hash.toString(36);
  }

  /**
   * assemble — estimate ratio → call server compaction → fallback to localCompact.
   * Framework calls this to build the model context for each turn.
   */
  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages?: any[];
    tokenBudget?: number;
    prompt?: string;
    availableTools?: Set<string>;
    citationsMode?: string;
    model?: string;
  }) {
    const { messages, sessionKey, sessionId } = params;
    if (!messages || messages.length === 0) {
      return { messages: messages ? [...messages] : [], estimatedTokens: 0 };
    }

    const sk = sessionKey ?? sessionId ?? "unknown";

    // ── L1.5 trigger: fire-and-forget on new user prompt ──
    this.triggerL15IfNeeded(params.prompt, messages, sk);

    const contextWindow = params.tokenBudget ?? DEFAULT_CONTEXT_WINDOW;
    // Don't use framework's knownTokens for calibration — our tiktoken is already precise.
    // Framework's currentTokenCount may use a different calculation method (e.g. chars/4).
    const { total, perMessage } = estimateAllTokens(messages);
    const ratio = total / contextWindow;

    // Update calibration state for future calls
    const s = this.getSession(sk);
    s.lastKnownTotalTokens = total;
    s.lastKnownMsgCount = messages.length;

    // Below client threshold — skip compaction
    if (ratio < this.config.compactionRatio) {
      this.logger.debug?.(
        `[offload-client] assemble: ratio=${(ratio * 100).toFixed(1)}% < ${(this.config.compactionRatio * 100).toFixed(0)}%, skip`,
      );
      return { messages: [...messages], estimatedTokens: total };
    }

    this.logger.info(
      `[offload-client] assemble: ratio=${(ratio * 100).toFixed(1)}%, msgs=${messages.length}, calling compaction...`,
    );

    // Try server-side compaction first
    const result = await this.client.compaction({
      sessionId: sessionKey ?? sessionId ?? "unknown",
      messages,
      ratio,
      contextWindow,
      totalTokens: total,
      messageTokens: perMessage,
    });

    if (result) {
      const compactedTokens = result.messages.reduce(
        (sum: number, msg: any) => sum + estimateMessageTokens(msg),
        0,
      );
      this.logger.info(
        `[offload-client] server compaction done: level=${result.report.resolvedLevel}, ` +
        `${result.report.originalCount}→${result.report.compactedCount} msgs, ` +
        `mild=${result.report.mildReplacements}, agg=${result.report.aggressiveDeleted}, ` +
        `em=${result.report.emergencyDeleted}, mmd=${result.report.mmdInjected}`,
      );
      return { messages: result.messages, estimatedTokens: compactedTokens };
    }

    // Fallback: local brute-force compaction
    this.logger.warn("[offload-client] server compaction failed, falling back to local compact");
    const compacted = this.localCompact(messages, contextWindow, sk);
    const compactedTokens = compacted.reduce(
      (sum: number, msg: any) => sum + estimateMessageTokens(msg),
      0,
    );
    return { messages: compacted, estimatedTokens: compactedTokens };
  }

  /**
   * afterTurn — no-op (ingest is handled by after_tool_call hook).
   */
  async afterTurn() {}
}
