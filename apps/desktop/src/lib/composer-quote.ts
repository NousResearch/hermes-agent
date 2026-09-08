export interface ComposerQuote {
  body: string
  label: string
}

const QUOTE_PREFIX = 'q1.'
export const COMPOSER_QUOTE_RE = /@quote:(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/g

function base64UrlEncode(value: string): string {
  const bytes = new TextEncoder().encode(value)
  let binary = ''

  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000))
  }

  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
}

function base64UrlDecode(value: string): string {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) {
    throw new Error('invalid base64url')
  }

  const base64 = value.replaceAll('-', '+').replaceAll('_', '/')
  const binary = atob(base64 + '='.repeat((4 - (base64.length % 4)) % 4))
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
}

export function encodeComposerQuote(quote: ComposerQuote): string {
  return QUOTE_PREFIX + base64UrlEncode(JSON.stringify(quote))
}

export function decodeComposerQuote(value: string): ComposerQuote | null {
  if (!value.startsWith(QUOTE_PREFIX)) {
    return null
  }

  try {
    const parsed = JSON.parse(base64UrlDecode(value.slice(QUOTE_PREFIX.length))) as Partial<ComposerQuote> | null

    return parsed &&
      typeof parsed.body === 'string' &&
      parsed.body.trim() &&
      typeof parsed.label === 'string' &&
      parsed.label.trim()
      ? { body: parsed.body, label: parsed.label }
      : null
  } catch {
    return null
  }
}

function unquote(value: string): string {
  const head = value[0]
  const tail = value[value.length - 1]

  return (head === '`' && tail === '`') || (head === '"' && tail === '"') || (head === "'" && tail === "'")
    ? value.slice(1, -1)
    : value.replace(/[,.;!?]+$/, '')
}

export function composerQuoteLabel(value: string): string | null {
  return decodeComposerQuote(unquote(value))?.label ?? null
}

export function composerQuoteLabelsIn(text: string): string {
  return text.replace(COMPOSER_QUOTE_RE, (match, value: string) => composerQuoteLabel(value) ?? match)
}

export function expandComposerQuotes(draft: string): string {
  if (!draft.includes('@quote:')) {
    return draft
  }

  let cursor = 0
  let expanded = ''

  for (const match of draft.matchAll(COMPOSER_QUOTE_RE)) {
    const start = match.index ?? 0
    const end = start + match[0].length
    const quote = decodeComposerQuote(unquote(match[1] || ''))

    expanded += draft.slice(cursor, start)

    if (!quote) {
      expanded += match[0]
      cursor = end
      continue
    }

    if (expanded && !expanded.endsWith('\n\n')) {
      expanded += expanded.endsWith('\n') ? '\n' : '\n\n'
    }

    expanded += quote.body.trim()

    const remainder = draft.slice(end)

    if (remainder && !remainder.startsWith('\n\n')) {
      expanded += remainder.startsWith('\n') ? '\n' : '\n\n'
    }

    // The rich editor emits one delimiter space after an atomic chip. Consume
    // exactly that separator so the response starts flush after the blank line.
    cursor = end + (remainder.startsWith(' ') ? 1 : 0)
  }

  return expanded + draft.slice(cursor)
}
