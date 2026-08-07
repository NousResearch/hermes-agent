/**
 * tdai_memory_search tool — delegates to v3 SDK searchAtomic.
 */

import type { MemoryClient } from "@tencentdb-agent-memory/memory-sdk-ts-v2";

interface Logger {
  debug?: (msg: string) => void;
  warn: (msg: string) => void;
}

export async function handleMemorySearch(
  client: MemoryClient,
  params: { query: string; limit?: number; type?: string },
  logger?: Logger,
) {
  const { query, limit = 5, type } = params;

  if (!query?.trim()) {
    return { content: [{ type: "text" as const, text: "Query cannot be empty." }] };
  }

  try {
    logger?.debug?.(`[memory-search] query="${query}", limit=${limit}, type=${type ?? "(all)"}`);
    const result = await client.searchAtomic({ query, limit, type });
    const items = result.items ?? [];
    logger?.debug?.(`[memory-search] ✅ ${items.length} results`);

    if (items.length === 0) {
      return { content: [{ type: "text" as const, text: "No matching memories found." }] };
    }

    const lines: string[] = [`Found ${items.length} matching memories:`, ""];
    for (const item of items) {
      const scoreStr = item.score != null ? ` (score: ${item.score.toFixed(3)})` : "";
      lines.push(`- **[${item.type}]**${scoreStr}`);
      lines.push(`  ${item.content}`);
      lines.push("");
    }

    return {
      content: [{ type: "text" as const, text: lines.join("\n") }],
      details: { count: items.length },
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger?.warn(`[memory-search] Failed: ${msg}`);
    return { content: [{ type: "text" as const, text: `Memory search failed: ${msg}` }] };
  }
}
