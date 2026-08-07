/**
 * conversation_search tool: Agent-callable tool for searching L0 conversation records.
 *
 * Supports three search strategies with automatic degradation:
 *   1. **hybrid** (default) — FTS5 keyword + vector embedding in parallel,
 *      merged via Reciprocal Rank Fusion (RRF).
 *   2. **embedding** — pure vector similarity (when FTS5 is unavailable).
 *   3. **fts** — pure FTS5 keyword search (when embedding is unavailable).
 *
 * The tool is registered via `api.registerTool()` in index.ts.
 */

import type { IMemoryStore, IsolationFilter, L0SearchResult } from "../store/types.js";
import { buildFtsQuery } from "../store/sqlite.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { Logger } from "../types.js";

// ============================
// Types
// ============================

export interface ConversationSearchResultItem {
  id: string;
  session_key: string;
  session_id: string;
  user_id: string;
  agent_id: string;
  /** Role of the message sender: "user" or "assistant" */
  role: string;
  /** Text content of this single message */
  content: string;
  score: number;
  recorded_at: string;
}

export interface ConversationSearchResult {
  results: ConversationSearchResultItem[];
  total: number;
  /** Actual search strategy used: "hybrid", "embedding", "fts", or "none". */
  strategy: string;
  /** Optional message, e.g. when embedding is not configured. */
  message?: string;
}

const TAG = "[memory-tdai][tdai_conversation_search]";

// ============================
// RRF (Reciprocal Rank Fusion)
// ============================

/** Standard RRF constant from the original RRF paper. */
const RRF_K = 60;

/**
 * Merge multiple ranked lists of `ConversationSearchResultItem` via Reciprocal
 * Rank Fusion. Items appearing in multiple lists get their RRF scores summed.
 *
 * Returns items sorted by descending RRF score. The `score` field of each
 * returned item is replaced by the RRF score for consistent ranking semantics.
 */
function rrfMergeL0(...lists: ConversationSearchResultItem[][]): ConversationSearchResultItem[] {
  const map = new Map<string, { item: ConversationSearchResultItem; rrfScore: number }>();

  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      const score = 1 / (RRF_K + rank + 1);
      const existing = map.get(item.id);
      if (existing) {
        existing.rrfScore += score;
      } else {
        map.set(item.id, { item, rrfScore: score });
      }
    }
  }

  return [...map.values()]
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(({ item, rrfScore }) => ({ ...item, score: rrfScore }));
}

// ============================
// Search implementation
// ============================

export async function executeConversationSearch(params: {
  query: string;
  limit: number;
  sessionKey?: string;
  filter?: IsolationFilter;
  vectorStore?: IMemoryStore;
  embeddingService?: EmbeddingService;
  logger?: Logger;
}): Promise<ConversationSearchResult> {
  const {
    query,
    limit,
    sessionKey: sessionFilter,
    filter: isolationFilter,
    vectorStore,
    embeddingService,
    logger,
  } = params;

  logger?.debug?.(
    `${TAG} CALLED: query="${query.slice(0, 100)}", limit=${limit}, ` +
    `sessionFilter=${sessionFilter ?? "(none)"}, ` +
    `vectorStore=${vectorStore ? "available" : "UNAVAILABLE"}, ` +
    `embeddingService=${embeddingService ? "available" : "UNAVAILABLE"}`,
  );

  if (!query || query.trim().length === 0) {
    logger?.debug?.(`${TAG} Empty query, returning empty`);
    return { results: [], total: 0, strategy: "none" };
  }

  if (!vectorStore) {
    logger?.warn?.(`${TAG} VectorStore not available`);
    return { results: [], total: 0, strategy: "none" };
  }

  // ── Determine available capabilities ──
  const hasEmbedding = !!embeddingService;
  const hasFts = vectorStore.isFtsAvailable();

  if (!hasEmbedding && !hasFts) {
    logger?.warn?.(`${TAG} Neither EmbeddingService nor FTS5 available — cannot search`);
    return {
      results: [],
      total: 0,
      strategy: "none",
      message:
        "Embedding service is not configured and FTS is not available. " +
        "Conversation search requires an embedding provider or FTS5 support. " +
        "Please configure an embedding provider in the embedding.provider setting (e.g. openai_compatible).",
    };
  }

  // ── Over-retrieve for later filtering and RRF merging ──
  const candidateK = sessionFilter ? limit * 4 : limit * 3;

  // ── Native hybrid short-circuit (TCVDB) ──
  // If the store natively supports hybrid search (dense + sparse + RRF in a
  // single API call), skip the dual-path FTS+Vector logic to avoid a redundant
  // second HTTP request with garbled FTS tokens as embedding input.
  if (vectorStore.getCapabilities().nativeHybridSearch && vectorStore.searchL0Hybrid) {
    logger?.debug?.(`${TAG} [native-hybrid] Single-call hybrid search...`);
    const results = await vectorStore.searchL0Hybrid(
      isolationFilter ? { query, topK: candidateK, filter: isolationFilter } : { query, topK: candidateK },
    );
    let items: ConversationSearchResultItem[] = results.map((r) => ({
      id: r.record_id,
      session_key: r.session_key,
      session_id: r.session_id,
      user_id: r.user_id,
      agent_id: r.agent_id,
      role: r.role,
      content: r.message_text,
      score: r.score,
      recorded_at: r.recorded_at,
    }));

    // Apply session filter
    if (sessionFilter) {
      items = items.filter((r) => r.session_key === sessionFilter);
    }
    const trimmed = items.slice(0, limit);
    logger?.debug?.(
      `${TAG} RESULT (strategy=native-hybrid): returning ${trimmed.length} messages ` +
      `(scores: [${trimmed.map((r) => r.score.toFixed(3)).join(", ")}])`,
    );
    return { results: trimmed, total: trimmed.length, strategy: "hybrid" };
  }

  // ── SQLite dual-path: run FTS5 + Vector in parallel, merge with client-side RRF ──
  const [ftsItems, vecItems] = await Promise.all([
    // FTS5 keyword search on L0
    (async (): Promise<ConversationSearchResultItem[]> => {
      if (!hasFts) return [];
      try {
        const ftsQuery = buildFtsQuery(query);
        if (!ftsQuery) {
          logger?.debug?.(`${TAG} [hybrid-fts] No usable FTS tokens from query`);
          return [];
        }
        logger?.debug?.(`${TAG} [hybrid-fts] FTS5 query: "${ftsQuery}"`);
        const ftsResults = isolationFilter
          ? await vectorStore.searchL0Fts(ftsQuery, candidateK, isolationFilter)
          : await vectorStore.searchL0Fts(ftsQuery, candidateK);
        logger?.debug?.(`${TAG} [hybrid-fts] FTS5 returned ${ftsResults.length} candidates`);
        return ftsResults.map((r) => ({
          id: r.record_id,
          session_key: r.session_key,
          role: r.role,
          content: r.message_text,
          score: r.score,
          recorded_at: r.recorded_at,
        }));
      } catch (err) {
        logger?.warn?.(
          `${TAG} [hybrid-fts] FTS5 search failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
        );
        return [];
      }
    })(),

    // Vector embedding search on L0
    (async (): Promise<ConversationSearchResultItem[]> => {
      if (!hasEmbedding) return [];
      try {
        logger?.debug?.(`${TAG} [hybrid-vec] Generating query embedding...`);
        const queryEmbedding = await embeddingService!.embed(query);
        logger?.debug?.(
          `${TAG} [hybrid-vec] Embedding OK, dims=${queryEmbedding.length}, searching top-${candidateK}...`,
        );
        const vecResults: L0SearchResult[] = isolationFilter
          ? await vectorStore.searchL0Vector(queryEmbedding, candidateK, query, isolationFilter)
          : await vectorStore.searchL0Vector(queryEmbedding, candidateK, query);
        logger?.debug?.(`${TAG} [hybrid-vec] Vector search returned ${vecResults.length} candidates`);
        return vecResults.map((r) => ({
          id: r.record_id,
          session_key: r.session_key,
          session_id: r.session_id,
          user_id: r.user_id,
          agent_id: r.agent_id,
          role: r.role,
          content: r.message_text,
          score: r.score,
          recorded_at: r.recorded_at,
        }));
      } catch (err) {
        logger?.warn?.(
          `${TAG} [hybrid-vec] Embedding search failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
        );
        return [];
      }
    })(),
  ]);

  // ── Determine effective strategy ──
  const ftsOk = ftsItems.length > 0;
  const vecOk = vecItems.length > 0;
  let strategy: string;

  if (ftsOk && vecOk) {
    strategy = "hybrid";
  } else if (vecOk) {
    strategy = "embedding";
  } else if (ftsOk) {
    strategy = "fts";
  } else {
    logger?.debug?.(`${TAG} Both search paths returned 0 results`);
    return { results: [], total: 0, strategy: hasEmbedding ? "embedding" : "fts" };
  }

  // ── Merge results ──
  let results: ConversationSearchResultItem[];
  if (strategy === "hybrid") {
    results = rrfMergeL0(ftsItems, vecItems);
    logger?.debug?.(
      `${TAG} [hybrid] RRF merged: fts=${ftsItems.length}, vec=${vecItems.length} → ${results.length} unique`,
    );
  } else {
    // Single-source: use whichever list has results (already sorted by score)
    results = ftsOk ? ftsItems : vecItems;
  }

  // ── Apply session key filter ──
  if (sessionFilter) {
    const preFilterCount = results.length;
    results = results.filter((r) => r.session_key === sessionFilter);
    logger?.debug?.(`${TAG} After session filter "${sessionFilter}": ${results.length}/${preFilterCount}`);
  }

  // ── Trim to requested limit ──
  const trimmed = results.slice(0, limit);

  logger?.debug?.(
    `${TAG} RESULT (strategy=${strategy}): returning ${trimmed.length} messages ` +
    `(scores: [${trimmed.map((r) => r.score.toFixed(3)).join(", ")}])`,
  );

  return {
    results: trimmed,
    total: trimmed.length,
    strategy,
  };
}

// ============================
// Tool response formatter
// ============================

export function formatConversationSearchResponse(result: ConversationSearchResult): string {
  if (result.message) {
    return result.message;
  }
  if (result.results.length === 0) {
    return "No matching conversation messages found.";
  }

  const lines: string[] = [
    `Found ${result.total} matching message(s):`,
    "",
  ];

  for (const item of result.results) {
    const scoreStr = typeof item.score === "number" ? ` (score: ${item.score.toFixed(3)})` : "";
    const dateStr = item.recorded_at ? ` [${item.recorded_at}]` : "";
    lines.push(`---`);
    lines.push(`**[${item.role}]** Session: ${item.session_key}${dateStr}${scoreStr}`);
    lines.push("");
    lines.push(item.content);
    lines.push("");
  }

  return lines.join("\n");
}
