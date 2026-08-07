/**
 * memory_search tool: Agent-callable tool for searching L1 memory records.
 *
 * Supports three search strategies with automatic degradation:
 *   1. **hybrid** (default) — FTS5 keyword + vector embedding in parallel,
 *      merged via Reciprocal Rank Fusion (RRF).
 *   2. **embedding** — pure vector similarity (when FTS5 is unavailable).
 *   3. **fts** — pure FTS5 keyword search (when embedding is unavailable).
 *
 * The tool is registered via `api.registerTool()` in index.ts.
 */

import type { IMemoryStore, IsolationFilter, L1SearchResult } from "../store/types.js";
import { buildFtsQuery } from "../store/sqlite.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { Logger } from "../types.js";

// ============================
// Types
// ============================

export interface MemorySearchResultItem {
  id: string;
  content: string;
  type: string;
  team_id?: string;
  user_id?: string;
  agent_id?: string;
  task_id?: string;
  priority: number;
  scene_name: string;
  score: number;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface MemorySearchResult {
  results: MemorySearchResultItem[];
  total: number;
  strategy: string;
  /** Optional message, e.g. when embedding is not configured. */
  message?: string;
}

const TAG = "[memory-tdai][tdai_memory_search]";

// ============================
// RRF (Reciprocal Rank Fusion)
// ============================

/** Standard RRF constant from the original RRF paper. */
const RRF_K = 60;

/**
 * Merge multiple ranked lists of `MemorySearchResultItem` via Reciprocal Rank
 * Fusion. Items appearing in multiple lists get their RRF scores summed.
 *
 * Returns items sorted by descending RRF score. The `score` field of each
 * returned item is replaced by the RRF score for consistent ranking semantics.
 */
function rrfMergeL1(...lists: MemorySearchResultItem[][]): MemorySearchResultItem[] {
  const map = new Map<string, { item: MemorySearchResultItem; rrfScore: number }>();

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

export async function executeMemorySearch(params: {
  query: string;
  limit: number;
  type?: string;
  scene?: string;
  filter?: IsolationFilter;
  vectorStore?: IMemoryStore;
  embeddingService?: EmbeddingService;
  logger?: Logger;
}): Promise<MemorySearchResult> {
  const {
    query,
    limit,
    type: typeFilter,
    scene: sceneFilter,
    filter: isolationFilter,
    vectorStore,
    embeddingService,
    logger,
  } = params;

  logger?.debug?.(
    `${TAG} CALLED: query="${query.slice(0, 100)}", limit=${limit}, ` +
    `typeFilter=${typeFilter ?? "(none)"}, sceneFilter=${sceneFilter ?? "(none)"}, ` +
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
        "Memory search requires an embedding provider or FTS5 support. " +
        "Please configure an embedding provider in the embedding.provider setting (e.g. openai_compatible).",
    };
  }

  // ── Over-retrieve for later filtering and RRF merging ──
  const candidateK = limit * 3;

  // ── Native hybrid short-circuit (TCVDB) ──
  // If the store natively supports hybrid search (dense + sparse + RRF in a
  // single API call), skip the dual-path FTS+Vector logic to avoid a redundant
  // second HTTP request with garbled FTS tokens as embedding input.
  if (vectorStore.getCapabilities().nativeHybridSearch && vectorStore.searchL1Hybrid) {
    logger?.debug?.(`${TAG} [native-hybrid] Single-call hybrid search...`);
    const results = await vectorStore.searchL1Hybrid(
      isolationFilter ? { query, topK: candidateK, filter: isolationFilter } : { query, topK: candidateK },
    );
    let items: MemorySearchResultItem[] = results.map((r) => ({
      id: r.record_id,
      content: r.content,
      type: r.type,
      priority: r.priority,
      scene_name: r.scene_name,
      score: r.score,
      team_id: r.team_id,
      user_id: r.user_id,
      agent_id: r.agent_id,
      task_id: r.task_id,
      version: r.version ?? 0,
      created_at: r.timestamp_start,
      updated_at: r.timestamp_end,
    }));

    // Apply secondary filters
    if (typeFilter) items = items.filter((r) => r.type === typeFilter);
    if (sceneFilter) {
      const ns = sceneFilter.toLowerCase();
      items = items.filter((r) => r.scene_name.toLowerCase().includes(ns));
    }
    const trimmed = items.slice(0, limit);
    logger?.debug?.(
      `${TAG} RESULT (strategy=native-hybrid): returning ${trimmed.length} memories ` +
      `(scores: [${trimmed.map((r) => r.score.toFixed(3)).join(", ")}])`,
    );
    return { results: trimmed, total: trimmed.length, strategy: "hybrid" };
  }

  // ── SQLite dual-path: run FTS5 + Vector in parallel, merge with client-side RRF ──
  const [ftsItems, vecItems] = await Promise.all([
    // FTS5 keyword search
    (async (): Promise<MemorySearchResultItem[]> => {
      if (!hasFts) return [];
      try {
        const ftsQuery = buildFtsQuery(query);
        if (!ftsQuery) {
          logger?.debug?.(`${TAG} [hybrid-fts] No usable FTS tokens from query`);
          return [];
        }
        logger?.debug?.(`${TAG} [hybrid-fts] FTS5 query: "${ftsQuery}"`);
        const ftsResults = isolationFilter
          ? await vectorStore.searchL1Fts(ftsQuery, candidateK, isolationFilter)
          : await vectorStore.searchL1Fts(ftsQuery, candidateK);
        logger?.debug?.(`${TAG} [hybrid-fts] FTS5 returned ${ftsResults.length} candidates`);
        return ftsResults.map((r) => ({
          id: r.record_id,
          content: r.content,
          type: r.type,
          priority: r.priority,
          scene_name: r.scene_name,
          score: r.score,
          team_id: r.team_id,
      user_id: r.user_id,
      agent_id: r.agent_id,
      task_id: r.task_id,
      version: r.version ?? 0,
          created_at: r.timestamp_start,
          updated_at: r.timestamp_end,
        }));
      } catch (err) {
        logger?.warn?.(
          `${TAG} [hybrid-fts] FTS5 search failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
        );
        return [];
      }
    })(),

    // Vector embedding search
    (async (): Promise<MemorySearchResultItem[]> => {
      if (!hasEmbedding) return [];
      try {
        logger?.debug?.(`${TAG} [hybrid-vec] Generating query embedding...`);
        const queryEmbedding = await embeddingService!.embed(query);
        logger?.debug?.(
          `${TAG} [hybrid-vec] Embedding OK, dims=${queryEmbedding.length}, searching top-${candidateK}...`,
        );
        const vecResults: L1SearchResult[] = isolationFilter
          ? await vectorStore.searchL1Vector(queryEmbedding, candidateK, query, isolationFilter)
          : await vectorStore.searchL1Vector(queryEmbedding, candidateK, query);
        logger?.debug?.(`${TAG} [hybrid-vec] Vector search returned ${vecResults.length} candidates`);
        return vecResults.map((r) => ({
          id: r.record_id,
          content: r.content,
          type: r.type,
          priority: r.priority,
          scene_name: r.scene_name,
          score: r.score,
          team_id: r.team_id,
      user_id: r.user_id,
      agent_id: r.agent_id,
      task_id: r.task_id,
      version: r.version ?? 0,
          created_at: r.timestamp_start,
          updated_at: r.timestamp_end,
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
  let results: MemorySearchResultItem[];
  if (strategy === "hybrid") {
    results = rrfMergeL1(ftsItems, vecItems);
    logger?.debug?.(
      `${TAG} [hybrid] RRF merged: fts=${ftsItems.length}, vec=${vecItems.length} → ${results.length} unique`,
    );
  } else {
    // Single-source: use whichever list has results (already sorted by score)
    results = ftsOk ? ftsItems : vecItems;
  }

  // ── Apply secondary filters (type, scene) ──
  const preFilterCount = results.length;
  if (typeFilter) {
    results = results.filter((r) => r.type === typeFilter);
    logger?.debug?.(`${TAG} After type filter "${typeFilter}": ${results.length}/${preFilterCount}`);
  }
  if (sceneFilter) {
    const normalizedScene = sceneFilter.toLowerCase();
    results = results.filter((r) =>
      r.scene_name.toLowerCase().includes(normalizedScene),
    );
    logger?.debug?.(`${TAG} After scene filter "${sceneFilter}": ${results.length}/${preFilterCount}`);
  }

  // ── Trim to requested limit ──
  const trimmed = results.slice(0, limit);

  logger?.debug?.(
    `${TAG} RESULT (strategy=${strategy}): returning ${trimmed.length} memories ` +
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

export function formatSearchResponse(result: MemorySearchResult): string {
  if (result.message) {
    return result.message;
  }
  if (result.results.length === 0) {
    return "No matching memories found.";
  }

  const lines: string[] = [
    `Found ${result.total} matching memories:`,
    "",
  ];

  for (const item of result.results) {
    const scoreStr = typeof item.score === "number" ? ` (score: ${item.score.toFixed(3)})` : "";
    const sceneStr = item.scene_name ? ` [scene: ${item.scene_name}]` : "";
    const priorityStr = item.priority >= 0 ? ` (priority: ${item.priority})` : " (global instruction)";
    lines.push(`- **[${item.type}]**${priorityStr}${sceneStr}${scoreStr}`);
    lines.push(`  ${item.content}`);
    lines.push("");
  }

  return lines.join("\n");
}
