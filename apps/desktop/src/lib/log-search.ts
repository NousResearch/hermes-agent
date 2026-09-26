export interface LogSearchSegment {
  match: boolean
  text: string
}

function normalizedNeedle(query: string): string {
  return query.trim().toLocaleLowerCase()
}

export function splitLogSearchMatches(text: string, query: string): LogSearchSegment[] {
  const needle = normalizedNeedle(query)

  if (!needle) {
    return [{ match: false, text }]
  }

  const lowerText = text.toLocaleLowerCase()
  const segments: LogSearchSegment[] = []
  let cursor = 0

  while (cursor < text.length) {
    const index = lowerText.indexOf(needle, cursor)

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

export function countLogSearchMatches(lines: readonly string[], query: string): number {
  const needle = normalizedNeedle(query)

  if (!needle) {
    return 0
  }

  let count = 0

  for (const line of lines) {
    const lowerLine = line.toLocaleLowerCase()
    let cursor = 0

    while (cursor < lowerLine.length) {
      const index = lowerLine.indexOf(needle, cursor)

      if (index < 0) {
        break
      }

      count += 1
      cursor = index + needle.length
    }
  }

  return count
}

export function firstLogSearchMatchLine(lines: readonly string[], query: string): number {
  const needle = normalizedNeedle(query)

  if (!needle) {
    return -1
  }

  return lines.findIndex(line => line.toLocaleLowerCase().includes(needle))
}
