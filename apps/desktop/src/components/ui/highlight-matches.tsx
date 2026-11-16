import type { ReactNode } from 'react'

import { normalize, searchFold } from '@/lib/text'
import { cn } from '@/lib/utils'

/**
 * Emphasize every case-insensitive occurrence of `query` inside `text` — the
 * "why is this row in my filtered list" affordance for searchable pickers.
 * Renders semantic `<mark>`s (screen readers announce them as highlighted)
 * restyled to the accent token instead of the browser's yellow slab, so the
 * emphasis reads as text hierarchy, not a highlighter pen.
 *
 * The query must mirror the surface's OWN filter semantics, or the emphasis
 * lies about why a row matched:
 * - literal matching by default, including per-term string[] queries in the
 *   command palette;
 * - `foldSeparators` only for model surfaces whose filter uses `foldIncludes`.
 *   Replacing separators one-for-one keeps ranges aligned with the label.
 */
export function HighlightMatches({
  className,
  foldSeparators = false,
  query,
  text
}: {
  className?: string
  foldSeparators?: boolean
  query: string | string[]
  text: string
}) {
  const terms = (Array.isArray(query) ? query : [query]).map(normalize).filter(Boolean)

  if (terms.length === 0) {
    return <>{text}</>
  }

  const ranges = foldSeparators
    ? matchRanges(searchFold(text), terms.map(searchFold))
    : matchRanges(text.toLowerCase(), terms)

  if (ranges.length === 0) {
    // No occurrence (the row matched on its id/slug/keywords, not this label).
    return <>{text}</>
  }

  const parts: ReactNode[] = []
  let cursor = 0

  for (const [start, end] of ranges) {
    if (start > cursor) {
      parts.push(text.slice(cursor, start))
    }

    parts.push(
      <mark className={cn('bg-transparent font-medium text-(--ui-accent)', className)} key={start}>
        {text.slice(start, end)}
      </mark>
    )

    cursor = end
  }

  if (cursor < text.length) {
    parts.push(text.slice(cursor))
  }

  return <>{parts}</>
}

/** All occurrences of every term in `lower`, as sorted, merged [start, end). */
function matchRanges(lower: string, terms: string[]): Array<[number, number]> {
  const raw: Array<[number, number]> = []

  for (const term of terms) {
    for (let index = lower.indexOf(term); index >= 0; index = lower.indexOf(term, index + 1)) {
      raw.push([index, index + term.length])
    }
  }

  raw.sort((a, b) => a[0] - b[0] || a[1] - b[1])

  const merged: Array<[number, number]> = []

  for (const [start, end] of raw) {
    const last = merged[merged.length - 1]

    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end)
    } else {
      merged.push([start, end])
    }
  }

  return merged
}
