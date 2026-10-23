// Pure matching behind in-log search (LogTail + useLogSearch). Kept free of
// React so the offsets, segment cuts, and severity column are testable alone.

export interface LogSearchHit {
  end: number
  line: number
  start: number
}

export type LogSeverity = 'critical' | 'error' | 'warning'

export interface LogLineSegment {
  /** Index into the full hit list this segment belongs to, or null. */
  hit: null | number
  text: string
}

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/**
 * Every case-insensitive literal occurrence of `query`, with offsets into the
 * ORIGINAL line. Lowercasing a copy and slicing the original by its offsets
 * drifts whenever case mapping changes length (U+0130 lowercases to two code
 * units), so the match runs on the original text with the `iu` flags instead.
 */
export function findLogSearchHits(lines: readonly string[], query: string): LogSearchHit[] {
  const needle = query.trim()

  if (!needle) {
    return []
  }

  const pattern = new RegExp(escapeRegExp(needle), 'giu')
  const hits: LogSearchHit[] = []

  lines.forEach((text, line) => {
    for (const match of text.matchAll(pattern)) {
      hits.push({ end: match.index + match[0].length, line, start: match.index })
    }
  })

  return hits
}

// hermes_logging._LOG_FORMAT puts the level right after the timestamp. Anchoring
// to that column keeps a message that merely mentions "ERROR" from being
// marked as one.
const LEVEL_COLUMN = /^\d{4}-\d{2}-\d{2} [\d:.,]+ (WARNING|ERROR|CRITICAL)\b/

export function logLineSeverity(line: string): LogSeverity | null {
  const level = LEVEL_COLUMN.exec(line)?.[1]

  return level ? (level.toLowerCase() as LogSeverity) : null
}

/** Cut one line at its hits. `hits` pairs each of this line's hits with its
 *  index in the full list so the active one can be told apart. */
export function logLineSegments(text: string, hits: readonly { hit: LogSearchHit; index: number }[]): LogLineSegment[] {
  const segments: LogLineSegment[] = []
  let cursor = 0

  for (const { hit, index } of hits) {
    if (hit.start > cursor) {
      segments.push({ hit: null, text: text.slice(cursor, hit.start) })
    }

    segments.push({ hit: index, text: text.slice(hit.start, hit.end) })
    cursor = hit.end
  }

  if (cursor < text.length || segments.length === 0) {
    segments.push({ hit: null, text: text.slice(cursor) })
  }

  return segments
}
