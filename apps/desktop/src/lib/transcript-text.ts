export interface TranscriptTextPart {
  text: string
  href?: string
}

export function safeTranscriptLink(value: string): boolean {
  if (value.length > 4096 || /[\s<>"'\p{Surrogate}]/u.test(value) || [...value].some(c => c.charCodeAt(0) <= 32)) {
    return false
  }

  try {
    const url = new URL(value)

    return (url.protocol === 'http:' || url.protocol === 'https:') && !url.username && !url.password
  } catch {
    return false
  }
}

export function trimTranscriptUrl(value: string): string {
  let end = value.length
  const pairs: Record<string, string> = { ')': '(', ']': '[', '}': '{' }

  while (end > 0) {
    const char = value[end - 1]

    if (/[.,;:!?]/.test(char)) {
      end--

      continue
    }

    const open = pairs[char]

    if (open) {
      const current = value.slice(0, end)

      if ([...current].filter(c => c === char).length > [...current].filter(c => c === open).length) {
        end--

        continue
      }
    }

    break
  }

  return value.slice(0, end)
}

function transcriptUrlBoundary(text: string, start: number): { end: number; closed: boolean } {
  let depth = 0
  let end = start

  for (; end < text.length; end++) {
    const char = text[end]

    if (/[\s<>"']/.test(char)) {
      break
    }

    if (char === '(') {
      depth++
    } else if (char === ')') {
      if (depth === 0) {
        return { end, closed: true }
      }

      depth--
    }
  }

  return { end, closed: false }
}

interface TranscriptLinkScan {
  end: number
  sourceEnd: number
  part?: TranscriptTextPart
}

function readTranscriptLink(text: string, match: RegExpExecArray): TranscriptLinkScan {
  const named = match[1] !== undefined
  const start = named ? match.index + match[0].length - match[2].length : match.index
  const boundary = transcriptUrlBoundary(text, start)
  const end = boundary.end + (named && boundary.closed ? 1 : 0)
  const skipped = { end, sourceEnd: match.index }

  if (!named && match.index > 0 && /[A-Za-z0-9_:/@.-]/.test(text[match.index - 1])) {
    return skipped
  }

  if ((named && !boundary.closed) || boundary.end - start > 4096) {
    return skipped
  }

  const candidate = named ? text.slice(start, boundary.end) : trimTranscriptUrl(text.slice(start, boundary.end))

  if (!safeTranscriptLink(candidate)) {
    return skipped
  }

  return {
    end,
    sourceEnd: named ? end : start + candidate.length,
    part: { text: named ? match[1] : candidate, href: candidate }
  }
}

/** Preserve the source exactly when links are off. Link labels may replace Markdown syntax only when on. */
export function transcriptTextParts(text: string, links: boolean, literal = false): TranscriptTextPart[] {
  if (!links) {
    return [{ text }]
  }

  // Structured JSON is literal even when displayed by a prose-oriented consumer.
  if (!literal && /^\s*[[{]/.test(text)) {
    try {
      JSON.parse(text)
      literal = true
    } catch {
      // Not structured JSON; retain normal Markdown label handling.
    }
  }

  const result: TranscriptTextPart[] = []

  const pattern = literal ? /https?:\/\//gi : /\[([^\]\n]{1,512})\]\((https?:\/\/)|https?:\/\//gi

  let cursor = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(text)) !== null) {
    const link = readTranscriptLink(text, match)
    // Do not rediscover a nested scheme inside a rejected destination.
    pattern.lastIndex = link.end

    if (!link.part) {
      continue
    }

    if (match.index > cursor) {
      result.push({ text: text.slice(cursor, match.index) })
    }

    result.push(link.part)
    cursor = link.sourceEnd
  }

  if (cursor < text.length) {
    result.push({ text: text.slice(cursor) })
  }

  return result.length ? result : [{ text }]
}
