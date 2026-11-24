// Canonical text micro-helpers. Do not redefine these per-page.

export const asText = (v: unknown): string => (typeof v === 'string' ? v : v == null ? '' : String(v))

export const includesQuery = (v: unknown, q: string) => asText(v).toLowerCase().includes(q)

export const prettyName = (v: string) => v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

/** Search-key normalization: the exact `value.trim().toLowerCase()` idiom that
 *  was hand-written at ~30 filter/lookup sites. */
export const normalize = (v: unknown): string => asText(v).trim().toLowerCase()

/** Uppercase the first character, leave the rest. Matches the
 *  `s.charAt(0).toUpperCase() + s.slice(1)` idiom (empty-safe). */
export const capitalize = (v: string): string => (v ? v.charAt(0).toUpperCase() + v.slice(1) : v)

/** First non-empty string among `keys`, trimmed. For reading tool args and
 *  results, where the key carrying the interesting value varies by tool. */
export const firstStringField = (record: Record<string, unknown>, keys: readonly string[]): string => {
  for (const key of keys) {
    const value = record[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return ''
}

/** Search-equivalence fold: separators (`-`, `_`, `.`) and case stop matter —
 *  `qwen3.8-flash`, `Qwen3.8 Flash` and `qwen3 8_flash` all fold to the same
 *  string. One char in, one char out (length preserved), so highlight ranges
 *  computed on folded text index the ORIGINAL text unchanged. Model ids use
 *  hyphens, display names use spaces, quants use underscores — a picker search
 *  must not care. Filter and highlight MUST call this on BOTH sides; a fold on
 *  only one is what made hyphen queries match without ever highlighting. */
export function searchFold(v: unknown): string {
  return asText(v).toLowerCase().replace(/[-_.]/g, ' ')
}

/** `searchFold` + substring: the one matcher every searchable picker uses. */
export function foldIncludes(text: unknown, query: unknown): boolean {
  return searchFold(text).includes(searchFold(query))
}
