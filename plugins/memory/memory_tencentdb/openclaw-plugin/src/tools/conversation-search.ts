/**
 * tdai_conversation_search tool — delegates to v3 SDK searchConversation.
 */

import type { MemoryClient } from "@tencentdb-agent-memory/memory-sdk-ts-v2";

interface Logger {
  debug?: (msg: string) => void;
  warn: (msg: string) => void;
}

export async function handleConversationSearch(
  client: MemoryClient,
  params: { query: string; limit?: number; session_key?: string },
  logger?: Logger,
) {
  const { query, limit = 5, session_key } = params;

  if (!query?.trim()) {
    return { content: [{ type: "text" as const, text: "Query cannot be empty." }] };
  }

  try {
    logger?.debug?.(`[conversation-search] query="${query}", limit=${limit}, session=${session_key ?? "(all)"}`);
    const result = await client.searchConversation({
      query,
      limit,
      session_id: session_key,
    });
    const messages = result.messages ?? [];
    logger?.debug?.(`[conversation-search] ✅ ${messages.length} results`);

    if (messages.length === 0) {
      return { content: [{ type: "text" as const, text: "No matching conversation messages found." }] };
    }

    const lines: string[] = [`Found ${messages.length} matching message(s):`, ""];
    for (const msg of messages) {
      const scoreStr = msg.score != null ? ` (score: ${msg.score.toFixed(3)})` : "";
      const dateStr = msg.timestamp ? ` [${msg.timestamp}]` : "";
      lines.push(`---`);
      lines.push(`**[${msg.role}]**${dateStr}${scoreStr}`);
      lines.push("");
      lines.push(msg.content);
      lines.push("");
    }

    return {
      content: [{ type: "text" as const, text: lines.join("\n") }],
      details: { count: messages.length },
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger?.warn(`[conversation-search] Failed: ${msg}`);
    return { content: [{ type: "text" as const, text: `Conversation search failed: ${msg}` }] };
  }
}
