// ── Content-addressed syntax-highlight cache (#95595) ────────────────────────
// Switching to a warm session remounts the incoming transcript, and every
// mounted code block used to be re-tokenized from scratch by shiki — N blocks
// × full tokenization on the renderer main thread, on every switch, even
// though the code had not changed. The fix is a module-level LRU cache keyed
// by (scope, language, code) holding the final highlighted HTML, so a remount
// of an unchanged block paints the cached markup synchronously and never
// touches the highlighter.
//
// Bounds: shiki's tokenized HTML is ~5-10x the source size, so an unbounded
// cache would leak renderer memory over a long session list. Cap both the
// entry count and the total cached characters; evict oldest-first.
//
// This module is intentionally dependency-free (no React, no shiki) so the
// cache logic can be unit-tested in isolation.

export const HIGHLIGHT_CACHE_MAX_ENTRIES = 512
export const HIGHLIGHT_CACHE_MAX_CHARS = 6 * 1024 * 1024

/** Unique key for one highlighted block: scope + language + exact code. */
export function highlightCacheKey(scope: string, language: string, code: string): string {
  return `${scope}\u0000${language}\u0000${code}`
}

/**
 * Bounded LRU map of highlight cache keys to rendered HTML. `get` refreshes
 * recency (Map insertion order is used as the LRU clock); `set` evicts the
 * oldest entries until both caps hold.
 */
export class HighlightCache {
  private readonly entries = new Map<string, string>()
  private chars = 0

  constructor(
    private readonly maxEntries: number = HIGHLIGHT_CACHE_MAX_ENTRIES,
    private readonly maxChars: number = HIGHLIGHT_CACHE_MAX_CHARS
  ) {}

  get size(): number {
    return this.entries.size
  }

  get totalChars(): number {
    return this.chars
  }

  has(key: string): boolean {
    return this.entries.has(key)
  }

  get(key: string): string | undefined {
    const value = this.entries.get(key)

    if (value !== undefined) {
      // Refresh recency: re-inserting moves the entry to the newest position.
      this.entries.delete(key)
      this.entries.set(key, value)
    }

    return value
  }

  set(key: string, html: string): void {
    if (this.entries.has(key)) {
      this.chars -= this.entries.get(key)!.length
      this.entries.delete(key)
    }

    this.entries.set(key, html)
    this.chars += html.length
    this.evict()
  }

  clear(): void {
    this.entries.clear()
    this.chars = 0
  }

  private evict(): void {
    while ((this.entries.size > this.maxEntries || this.chars > this.maxChars) && this.entries.size > 0) {
      const oldestKey = this.entries.keys().next().value as string
      const oldest = this.entries.get(oldestKey)!
      this.entries.delete(oldestKey)
      this.chars -= oldest.length
    }
  }
}

/** The renderer-wide highlight cache. Lives for the lifetime of the module. */
export const highlightCache = new HighlightCache()
