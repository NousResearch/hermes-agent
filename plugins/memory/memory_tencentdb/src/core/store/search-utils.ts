/**
 * Search utilities — shared helpers for memory search across backends.
 *
 * Contains:
 * - RRF (Reciprocal Rank Fusion) merge — used by SQLite hybrid search
 *   (eliminates the 3x duplication in auto-recall, memory-search, conversation-search)
 * - FTS query building — re-exported from sqlite for convenience
 */

// ============================
// RRF (Reciprocal Rank Fusion)
// ============================

/**
 * Standard RRF constant from the original RRF paper.
 * Higher k → more weight on lower-ranked items (smoother distribution).
 */
export const RRF_K = 60;

/**
 * Merge multiple ranked lists via Reciprocal Rank Fusion.
 *
 * Each item's RRF score = sum over all lists of 1/(k + rank + 1).
 * Items appearing in multiple lists get their scores summed.
 *
 * @param lists   Array of ranked lists. Each list must have items with an `id` field.
 * @param k       RRF constant (default: 60).
 * @returns       Merged list sorted by descending RRF score, with `rrfScore` attached.
 *
 * @example
 * ```ts
 * const merged = rrfMerge(
 *   [ftsResults, vecResults],
 *   (item) => item.record_id,
 * );
 * ```
 */
export function rrfMerge<T>(
  lists: T[][],
  getId: (item: T) => string,
  k: number = RRF_K,
): Array<T & { rrfScore: number }> {
  const map = new Map<string, { item: T; rrfScore: number }>();

  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      const id = getId(item);
      const score = 1 / (k + rank + 1);
      const existing = map.get(id);
      if (existing) {
        existing.rrfScore += score;
      } else {
        map.set(id, { item, rrfScore: score });
      }
    }
  }

  return [...map.values()]
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(({ item, rrfScore }) => ({ ...item, rrfScore }));
}
