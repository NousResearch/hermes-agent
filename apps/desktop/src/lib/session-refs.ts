/**
 * Pure helpers for `@session:<profile>/<id>` references.
 *
 * Kept free of React/store/API imports so the markdown preprocessor (a hot
 * per-flush path) can use them without pulling in the resolver; the stateful
 * title lookup lives in `session-link-title.ts`.
 */

/** Mirrors the composer/transcript form in `directive-text.tsx`: a bare value,
 *  or one fenced in backticks/quotes so a value with spaces survives. The
 *  lookbehinds keep `foo@session:` and URL paths from matching, and skip a ref
 *  a model already wrapped in a markdown link — rewriting that would nest. */
export const SESSION_REF_RE = /(?<![\w/])(?<!]\()@session:(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/g

const TRAILING_PUNCTUATION_RE = /[,.;:!?)\]}]+$/

function unwrapQuotes(raw: string): null | string {
  if (raw.length < 2) {
    return null
  }

  const head = raw[0]
  const tail = raw[raw.length - 1]

  if ((head === '`' && tail === '`') || (head === '"' && tail === '"') || (head === "'" && tail === "'")) {
    return raw.slice(1, -1)
  }

  return null
}

/** Splits a matched value into the reference itself and any prose punctuation
 *  that the greedy `\S+` branch swallowed. Quoted values are already fenced. */
export function splitSessionRefValue(raw: string): { trailing: string; value: string } {
  const quoted = unwrapQuotes(raw)

  if (quoted !== null) {
    return { trailing: '', value: quoted }
  }

  const value = raw.replace(TRAILING_PUNCTUATION_RE, '')

  return { trailing: raw.slice(value.length), value }
}

/** Session ids never contain a slash, so a slash unambiguously means
 *  `<profile>/<id>` — same split as `tools/session_search_tool.py`. */
export function parseSessionRefValue(value: string): { profile?: string; sessionId: string } {
  const trimmed = value.trim()
  const slash = trimmed.indexOf('/')

  if (slash === -1) {
    return { sessionId: trimmed }
  }

  const profile = trimmed.slice(0, slash).trim()
  const sessionId = trimmed.slice(slash + 1).trim()

  return sessionId ? { profile: profile || undefined, sessionId } : { sessionId: trimmed }
}

export function sessionRefCacheKey(value: string): string {
  const { profile, sessionId } = parseSessionRefValue(value)

  return sessionId ? `${profile ?? ''}/${sessionId}` : ''
}

/** Chip label before (or without) a resolved title — a short, still-identifying id. */
export function sessionRefFallbackLabel(value: string): string {
  const { sessionId } = parseSessionRefValue(value)

  if (!sessionId) {
    return value
  }

  return sessionId.length > 10 ? `${sessionId.slice(0, 8)}…` : sessionId
}

/** A fragment href, matching the `#preview/` convention in `preview-targets.ts`
 *  — no custom URL scheme to survive markdown sanitization. */
export function sessionMarkdownHref(value: string): string {
  return `#session/${encodeURIComponent(value)}`
}

export function sessionRefFromMarkdownHref(href?: string): null | string {
  if (!href?.startsWith('#session/')) {
    return null
  }

  try {
    return decodeURIComponent(href.slice('#session/'.length)) || null
  } catch {
    return null
  }
}

/**
 * Rewrites bare `@session:<profile>/<id>` tokens into markdown links so an
 * agent-authored reference reaches `MarkdownLink` and renders as a chip.
 * Callers must exclude code spans/fences — `preprocessMarkdown` already does.
 */
export function linkifySessionRefs(text: string): string {
  if (!text.includes('@session:')) {
    return text
  }

  return text.replace(SESSION_REF_RE, (match, raw: string) => {
    const { trailing, value } = splitSessionRefValue(raw)

    if (!parseSessionRefValue(value).sessionId) {
      return match
    }

    const label = sessionRefFallbackLabel(value).replace(/[[\]\\]/g, '\\$&')

    return `[${label}](${sessionMarkdownHref(value)})${trailing}`
  })
}
