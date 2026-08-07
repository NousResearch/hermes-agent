/**
 * offload-client — HTTP client for Offload Server v2 API.
 */
import type { OffloadClientConfig, ToolPairPayload, RecentMessage, CompactionResult, Logger } from "./types.js";

export class OffloadApiClient {
  constructor(
    private config: OffloadClientConfig,
    private logger: Logger,
  ) {}

  /** Health check: GET /v2/offload/health. Returns true if server is reachable (any HTTP response = reachable). */
  async checkHealth(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${this.config.serverUrl}/v2/offload/health`, {
        method: "GET",
        headers: { Authorization: `Bearer ${this.config.apiKey}` },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      // Any HTTP response (including 401/403) means server is reachable
      return res.status < 500;
    } catch {
      return false;
    }
  }

  /**
   * Fire-and-forget: send tool pairs to ingest endpoint.
   * Does not throw — failures are logged as warnings.
   */
  async ingest(sessionId: string, toolPairs: ToolPairPayload[]): Promise<void> {
    return this.ingestWithContext(sessionId, toolPairs, undefined, undefined);
  }

  /**
   * Fire-and-forget: send tool pairs + optional context to ingest endpoint.
   * When prompt/recentMessages are provided, server triggers L1 with context (skip L1.5).
   */
  async ingestWithContext(
    sessionId: string,
    toolPairs: ToolPairPayload[],
    prompt?: string,
    recentMessages?: RecentMessage[],
  ): Promise<void> {
    const url = `${this.config.serverUrl}/v2/offload/ingest`;
    const payload: Record<string, unknown> = {
      session_id: sessionId,
      tool_pairs: toolPairs.map((tp) => ({
        tool_name: tp.toolName,
        tool_call_id: tp.toolCallId,
        params: tp.params,
        result: tp.result,
        error: tp.error,
        timestamp: tp.timestamp,
        duration_ms: tp.durationMs,
      })),
    };
    if (prompt) payload.prompt = prompt;
    if (recentMessages && recentMessages.length > 0) payload.recent_messages = recentMessages;
    const body = JSON.stringify(payload);

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.config.ingestTimeoutMs);

      await fetch(url, {
        method: "POST",
        headers: this.buildHeaders(),
        body,
        signal: controller.signal,
      });

      clearTimeout(timer);
    } catch (err) {
      this.logger.warn(`[offload-client] ingest failed: ${err}`);
    }
  }

  /**
   * Fire-and-forget: trigger L1.5 task judgment via ingest endpoint.
   * Sends prompt + recentMessages (empty toolPairs) to activate the L1.5 path on the server.
   */
  async ingestL15(sessionId: string, prompt: string, recentMessages?: RecentMessage[]): Promise<void> {
    const url = `${this.config.serverUrl}/v2/offload/ingest`;
    const payload: Record<string, unknown> = {
      session_id: sessionId,
      tool_pairs: [],
      prompt,
    };
    if (recentMessages && recentMessages.length > 0) payload.recent_messages = recentMessages;
    const body = JSON.stringify(payload);

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.config.ingestTimeoutMs);

      const response = await fetch(url, {
        method: "POST",
        headers: this.buildHeaders(),
        body,
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) {
        this.logger.warn(`[offload-client] ingestL15 returned ${response.status}`);
      }
    } catch (err) {
      this.logger.warn(`[offload-client] ingestL15 failed: ${err}`);
    }
  }

  /**
   * Synchronous compaction call. Returns compressed messages + report.
   * Returns null on timeout/failure (caller should keep original messages).
   */
  async compaction(req: {
    sessionId: string;
    messages: any[];
    ratio: number;
    contextWindow: number;
    totalTokens: number;
    messageTokens?: number[];
  }): Promise<CompactionResult | null> {
    const url = `${this.config.serverUrl}/v2/offload/compact`;
    const body = JSON.stringify({
      session_id: req.sessionId,
      messages: req.messages,
      ratio: req.ratio,
      context_window: req.contextWindow,
      total_tokens: req.totalTokens,
      message_tokens: req.messageTokens,
    });

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.config.compactionTimeoutMs);

      const response = await fetch(url, {
        method: "POST",
        headers: this.buildHeaders(),
        body,
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) {
        this.logger.warn(`[offload-client] compaction returned ${response.status}`);
        return null;
      }

      const json = (await response.json()) as any;
      if (json.code !== 0 || !json.data) {
        this.logger.warn(`[offload-client] compaction error: ${json.message ?? "unknown"}`);
        return null;
      }

      return { messages: json.data.messages, report: json.data.report };
    } catch (err) {
      this.logger.warn(`[offload-client] compaction failed: ${err}`);
      return null;
    }
  }

  private buildHeaders(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.config.apiKey}`,
      "X-TDAI-Service-Id": this.config.serviceId,
    };
  }
}
