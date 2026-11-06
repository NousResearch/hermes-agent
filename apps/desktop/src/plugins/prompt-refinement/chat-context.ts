/**
 * Opt-in chat context: each composer registers a reader for its own chat (the
 * drawer host does, through the SDK's `useThisComposer().recentMessages`), and
 * a refinement for that composer takes a short, trimmed excerpt of it — enough
 * for "do the same for the other file" to make sense, small enough to stay
 * cheap. Nothing is read unless the user turned chat context on.
 */

export interface TranscriptLine {
  role: 'assistant' | 'user'
  text: string
}

type Reader = (limit: number) => TranscriptLine[]

export const CONTEXT_MESSAGES = 6
export const CONTEXT_CHARS_EACH = 600
export const CONTEXT_CHARS_TOTAL = 3000

const readers = new Map<string, Reader>()

export function registerContextReader(target: string, reader: Reader): () => void {
  readers.set(target, reader)

  return () => {
    if (readers.get(target) === reader) {
      readers.delete(target)
    }
  }
}

const clip = (text: string, limit: number) => (text.length <= limit ? text : `${text.slice(0, limit - 1).trimEnd()}…`)

/** "User: …\nAssistant: …", newest kept when trimming, oldest first in the
 *  result; '' when the composer has no reader or no messages. */
export function formatContext(lines: readonly TranscriptLine[]): string {
  const formatted: string[] = []
  let total = 0

  for (const line of [...lines].reverse()) {
    const entry = `${line.role === 'user' ? 'User' : 'Assistant'}: ${clip(line.text.replace(/\s+/g, ' ').trim(), CONTEXT_CHARS_EACH)}`

    if (total + entry.length > CONTEXT_CHARS_TOTAL) {
      break
    }

    formatted.unshift(entry)
    total += entry.length + 1
  }

  return formatted.join('\n')
}

export function recentContext(target: string): string {
  const reader = readers.get(target)

  try {
    return reader ? formatContext(reader(CONTEXT_MESSAGES)) : ''
  } catch {
    // Context is a nicety: a reader that fails leaves the refinement without it.
    return ''
  }
}
