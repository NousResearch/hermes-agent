export interface LogSearchSegment {
  match: boolean
  text: string
}

function normalizedNeedle(query: string): string {
  return query.trim().toLocaleLowerCase()
}

interface NormalizedText {
  value: string
  sourceEnd: number[]
  sourceStart: number[]
}

/**
 * Lowercase text while retaining the source offsets for each resulting UTF-16
 * code unit. Some Unicode characters expand when lowercased (for example İ),
 * so offsets in the lowered string cannot safely slice the original text.
 */
function normalizeText(text: string): NormalizedText {
  let value = ''
  const sourceStart: number[] = []
  const sourceEnd: number[] = []
  let offset = 0

  for (const character of text) {
    const normalized = character.toLocaleLowerCase()

    for (let index = 0; index < normalized.length; index += 1) {
      sourceStart.push(offset)
      sourceEnd.push(offset + character.length)
    }

    value += normalized
    offset += character.length
  }

  return { value, sourceEnd, sourceStart }
}

export function splitLogSearchMatches(text: string, query: string): LogSearchSegment[] {
  const needle = normalizedNeedle(query)

  if (!needle) {
    return [{ match: false, text }]
  }

  const normalizedText = normalizeText(text)
  const segments: LogSearchSegment[] = []
  let cursor = 0

  while (cursor < normalizedText.value.length) {
    const index = normalizedText.value.indexOf(needle, cursor)

    if (index < 0) {
      segments.push({ match: false, text: text.slice(normalizedText.sourceStart[cursor] ?? text.length) })
      break
    }

    const start = normalizedText.sourceStart[index]
    const end = normalizedText.sourceEnd[index + needle.length - 1]

    if (start > (normalizedText.sourceStart[cursor] ?? 0)) {
      segments.push({ match: false, text: text.slice(normalizedText.sourceStart[cursor] ?? 0, start) })
    }

    segments.push({ match: true, text: text.slice(start, end) })
    cursor = index + needle.length

    while (cursor < normalizedText.value.length && normalizedText.sourceStart[cursor] < end) {
      cursor += 1
    }
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
