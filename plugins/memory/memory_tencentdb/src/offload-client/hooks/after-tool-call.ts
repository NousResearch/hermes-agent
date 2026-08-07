/**
 * offload-client — after_tool_call hook handler.
 * Fire-and-forget: sends tool pair + context to ingest API for L1 processing.
 */
import type { OffloadApiClient } from "../offload-api-client.js";
import type { OffloadClientConfig, ToolPairPayload, RecentMessage, Logger } from "../types.js";

export interface AfterToolCallEvent {
  toolName: string;
  toolCallId: string;
  params?: unknown;
  result?: unknown;
  error?: string;
  durationMs?: number;
}

/**
 * Create the after_tool_call hook handler.
 * Sends each tool call result to the server for L1 processing.
 *
 * @param getContext Optional getter for { prompt, recentMessages } context per session.
 */
export function createAfterToolCallHandler(
  client: OffloadApiClient,
  config: OffloadClientConfig,
  logger: Logger,
  getContext?: (sessionKey: string) => { prompt?: string; recentMessages?: RecentMessage[] } | undefined,
) {
  return (event: AfterToolCallEvent, ctx: { sessionKey?: string; sessionId?: string }) => {
    const sessionId = ctx.sessionKey ?? ctx.sessionId;
    logger.debug?.(
      `[offload-client] after_tool_call: tool=${event.toolName}, session=${sessionId ?? "(none)"}, callId=${event.toolCallId ?? "(none)"}`,
    );
    if (!sessionId) return;

    const toolPair: ToolPairPayload = {
      toolName: event.toolName,
      toolCallId: event.toolCallId,
      params: event.params ?? {},
      result: event.result,
      error: event.error,
      timestamp: new Date().toISOString(),
      durationMs: event.durationMs,
    };

    const context = getContext?.(sessionId);

    // Fire-and-forget — do not block the LLM flow
    client.ingestWithContext(sessionId, [toolPair], context?.prompt, context?.recentMessages).catch((err) => {
      logger.warn(`[offload-client] ingest fire-and-forget error: ${err}`);
    });
  };
}
