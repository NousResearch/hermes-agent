export type CommandCenterLogSeverity = 'CRITICAL' | 'DEBUG' | 'ERROR' | 'INFO' | 'WARNING'

const LEVEL_RE = /\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b/

export interface HighlightSegment {
  match: boolean
  text: string
}

export function commandCenterLogSeverity(line: string): CommandCenterLogSeverity | null {
  const match = LEVEL_RE.exec(line)

  if (!match) {
    return null
  }

  return match[1] === 'WARN' ? 'WARNING' : (match[1] as CommandCenterLogSeverity)
}

export function highlightLogSegments(text: string, query: string): HighlightSegment[] {
  const needle = query.trim()

  if (!needle) {
    return [{ match: false, text }]
  }

  const lowerText = text.toLocaleLowerCase()
  const lowerNeedle = needle.toLocaleLowerCase()
  const segments: HighlightSegment[] = []
  let cursor = 0

  while (cursor < text.length) {
    const index = lowerText.indexOf(lowerNeedle, cursor)

    if (index < 0) {
      segments.push({ match: false, text: text.slice(cursor) })
      break
    }

    if (index > cursor) {
      segments.push({ match: false, text: text.slice(cursor, index) })
    }

    segments.push({ match: true, text: text.slice(index, index + needle.length) })
    cursor = index + needle.length
  }

  return segments.length ? segments : [{ match: false, text }]
}
