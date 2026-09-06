import type { ReactNode } from 'react'

/**
 * TRANSCRIPT DIRECTIVES — the transcript as a contribution area.
 *
 * A plugin registers a named directive; the model addresses it by emitting a
 * paragraph of the form `::name{key="value"}` and that leaf renders as the
 * plugin's component, inline in the assistant message. This is the deliberate
 * counterpart to artifact promotion: artifacts are heuristic (substantial
 * fences get promoted whether or not the model asked), directives are
 * addressed (nothing renders unless a plugin claimed the name).
 *
 * The parse is deliberately narrow — a directive must be the entire
 * paragraph, so it can never hijack mid-prose text, and an unclaimed or
 * malformed directive falls back to the plain paragraph it always was.
 * Attributes are untrusted model output: plugins validate their own fields.
 */

export const TRANSCRIPT_DIRECTIVE_AREA = 'transcript.directives'

/** Props handed to a directive contribution's `render`. */
export interface TranscriptDirectiveProps {
  /** Parsed, untrusted attributes (e.g. `{ file: 'demo.html' }`). */
  attrs: Readonly<Record<string, string>>
  /** Original directive source text (diagnostics / fallback rendering). */
  source: string
  /** True while the surrounding message is still streaming. */
  streaming: boolean
}

/** Payload of a `transcript.directives` contribution's `data`. */
export interface TranscriptDirectiveContribution {
  /** The name the model addresses: `::<name>{...}`. Lowercase, `[a-z0-9-]`,
   *  unique across plugins — first registration wins on collision. */
  name: string
  /** Renders the directive leaf. Mounted inside the contribution error
   *  boundary, so a throw degrades to an inline error, not a dead message. */
  render: (props: TranscriptDirectiveProps) => ReactNode
}

export interface ParsedTranscriptDirective {
  name: string
  attrs: Record<string, string>
  source: string
}

const TRANSCRIPT_DIRECTIVE_PLACEHOLDER_PREFIX = 'hermestranscriptdirectivev1'
// The whole paragraph, nothing else on the line: `::name` or `::name{...}`.
// Length caps bound the attr scan on adversarial input.
const DIRECTIVE_RE = /^::([a-z][a-z0-9-]{0,63})(?:\{([^{}]{0,1024})\})?$/

// `key="value"` pairs; single quotes accepted for model sloppiness.
const ATTR_RE = /([a-z][\w-]{0,63})=(?:"([^"]*)"|'([^']*)')/gi

/**
 * Parse a paragraph as a transcript directive. Returns null unless the ENTIRE
 * trimmed text is one directive — prose containing `::` stays prose.
 * Pure and synchronous — safe to call during render.
 */
export function parseTranscriptDirective(text: string): ParsedTranscriptDirective | null {
  const trimmed = text.trim()

  // Cheap reject before the regex: directives are short single lines.
  if (!trimmed.startsWith('::') || trimmed.length > 1200 || trimmed.includes('\n')) {
    return null
  }

  const match = DIRECTIVE_RE.exec(trimmed)

  if (!match) {
    return null
  }

  const attrs: Record<string, string> = {}

  if (match[2]) {
    for (const pair of match[2].matchAll(ATTR_RE)) {
      attrs[pair[1].toLowerCase()] = pair[2] ?? pair[3] ?? ''
    }
  }

  return { name: match[1], attrs, source: trimmed }
}

export interface TranscriptDirectiveCodec {
  encode(text: string): string | null
  decode(text: string): string | null
  restoreOwnedTokens(text: string): string
}

const MAX_TRANSCRIPT_DIRECTIVE_CODEC_ENTRIES = 256

function createCodecNonce(): string | null {
  const values = new Uint32Array(4)
  const cryptoSource = globalThis.crypto

  if (!cryptoSource?.getRandomValues) {
    return null
  }

  try {
    cryptoSource.getRandomValues(values)
  } catch {
    return null
  }

  return Array.from(values, value => value.toString(16).padStart(8, '0'))
    .join('')
    .toUpperCase()
}

/**
 * Create a per-Markdown-surface codec. The token map is private to the
 * surface, and decoding is an exact map lookup rather than a parser round
 * trip, so model-authored lookalikes cannot claim a contribution.
 */
export function createTranscriptDirectiveCodec(): TranscriptDirectiveCodec {
  const tokenToSource = new Map<string, string>()
  const sourceToToken = new Map<string, string>()
  const nonce = createCodecNonce()
  const ownedTokenPrefix = nonce ? TRANSCRIPT_DIRECTIVE_PLACEHOLDER_PREFIX + nonce : null
  const ownedTokenPattern = ownedTokenPrefix ? new RegExp(`${ownedTokenPrefix}[A-F0-9]{8}`, 'gu') : null

  return {
    encode(text) {
      if (!nonce) {
        return null
      }

      const parsed = parseTranscriptDirective(text)

      if (!parsed) {
        return null
      }

      const existing = sourceToToken.get(parsed.source)

      if (existing) {
        return existing
      }

      if (sourceToToken.size >= MAX_TRANSCRIPT_DIRECTIVE_CODEC_ENTRIES) {
        return null
      }

      const index = sourceToToken.size.toString(16).padStart(8, '0').toUpperCase()
      const token = TRANSCRIPT_DIRECTIVE_PLACEHOLDER_PREFIX + nonce + index
      sourceToToken.set(parsed.source, token)
      tokenToSource.set(token, parsed.source)

      return token
    },
    decode(text) {
      return tokenToSource.get(text) ?? null
    },
    restoreOwnedTokens(text) {
      if (!ownedTokenPrefix || !ownedTokenPattern || !text.includes(ownedTokenPrefix)) {
        return text
      }

      return text.replace(ownedTokenPattern, token => tokenToSource.get(token) ?? token)
    }
  }
}

interface StandaloneDirectiveBlock {
  leading: string
  source: string
  trailing: string
}

interface StandaloneOwnedTokenBlock {
  leading: string
  token: string
  trailing: string
}

// Do not attempt to duplicate the downstream HTML parser. Once an unprotected
// block contains HTML-like markup, later directive-looking blocks in the same
// message stay ordinary Markdown. Protected directives are inert tokens here,
// so HTML-looking attribute text cannot taint a following directive.
const HTML_LIKE_MARKUP_RE = /<(?:[a-z!/]|\?)/iu

function standaloneDirectiveBlock(block: string): StandaloneDirectiveBlock | null {
  const leading = block.match(/^[ \t]*/u)?.[0] ?? ''

  // Four-space indentation is an indented code block; tabs are structural
  // indentation too. Neither context is a directive paragraph.
  if (leading.includes('\t') || leading.length >= 4) {
    return null
  }

  const remaining = block.slice(leading.length)
  const trailing = remaining.match(/[ \t\r\n]*$/u)?.[0] ?? ''
  const source = remaining.slice(0, remaining.length - trailing.length)

  if (!source || source.includes('\n') || source.includes('\r')) {
    return null
  }

  const parsed = parseTranscriptDirective(source)

  return parsed?.source === source ? { leading, source, trailing } : null
}

function standaloneOwnedTokenBlock(
  block: string,
  codec: TranscriptDirectiveCodec
): StandaloneOwnedTokenBlock | null {
  const leading = block.match(/^[ \t]*/u)?.[0] ?? ''

  if (leading.includes('\t') || leading.length >= 4) {
    return null
  }

  const remaining = block.slice(leading.length)
  const trailing = remaining.match(/[ \t\r\n]*$/u)?.[0] ?? ''
  const token = remaining.slice(0, remaining.length - trailing.length)

  if (!token || token.includes('\n') || token.includes('\r') || codec.decode(token) === null) {
    return null
  }

  return { leading, token, trailing }
}

function normalizeBlockLineEndings(value: string): string {
  return value.replace(/\r\n?/g, '\n')
}

function rawEndForNormalizedLength(markdown: string, start: number, normalizedLength: number): number | null {
  let normalizedOffset = 0
  let rawOffset = start

  while (normalizedOffset < normalizedLength) {
    if (rawOffset >= markdown.length) {
      return null
    }

    if (markdown[rawOffset] === '\r' && markdown[rawOffset + 1] === '\n') {
      rawOffset += 2
    } else {
      rawOffset += 1
    }

    normalizedOffset += 1
  }

  return rawOffset
}

/**
 * Protect only Markdown blocks that are truly standalone directive
 * paragraphs. Structural whitespace is copied around the inert token, while
 * lists, quotes, HTML, fences, indented code, and multi-line prose remain
 * untouched for the normal Markdown parser.
 */
export function protectTranscriptDirectiveBlocks(
  markdown: string,
  parseBlocks: (value: string) => readonly string[],
  codec: TranscriptDirectiveCodec
): string {
  if (!markdown.includes('::')) {
    return markdown
  }

  let blocks: readonly string[]

  try {
    blocks = parseBlocks(markdown)
  } catch {
    return markdown
  }

  if (blocks.join('') !== normalizeBlockLineEndings(markdown)) {
    return markdown
  }

  const protectedBlocks: string[] = []
  let htmlLikeMarkupSeen = false
  let rawOffset = 0

  for (const block of blocks) {
    const rawEnd = rawEndForNormalizedLength(markdown, rawOffset, block.length)

    if (rawEnd === null) {
      return markdown
    }

    const rawBlock = markdown.slice(rawOffset, rawEnd)

    if (normalizeBlockLineEndings(rawBlock) !== block) {
      return markdown
    }

    const directive = htmlLikeMarkupSeen ? null : standaloneDirectiveBlock(rawBlock)
    const token = directive ? codec.encode(directive.source) : null
    const protectedBlock = token && directive ? directive.leading + token + directive.trailing : rawBlock

    protectedBlocks.push(protectedBlock)
    htmlLikeMarkupSeen ||= HTML_LIKE_MARKUP_RE.test(protectedBlock)
    rawOffset = rawEnd
  }

  return rawOffset === markdown.length ? protectedBlocks.join('') : markdown
}

/**
 * Re-check owned directive tokens after generic Markdown preprocessing. Only
 * tokens that remain complete standalone blocks in the final Markdown may
 * reach Streamdown. Structural changes, synthesized HTML, or parser ambiguity
 * restore the token's original source, which is visible but cannot be claimed.
 */
export function finalizeTranscriptDirectiveBlocks(
  markdown: string,
  parseBlocks: (value: string) => readonly string[],
  protectionCodec: TranscriptDirectiveCodec,
  claimCodec: TranscriptDirectiveCodec
): string {
  if (!markdown.includes(TRANSCRIPT_DIRECTIVE_PLACEHOLDER_PREFIX)) {
    return markdown
  }

  const restoreAll = () => protectionCodec.restoreOwnedTokens(markdown)
  let blocks: readonly string[]

  try {
    blocks = parseBlocks(markdown)
  } catch {
    return restoreAll()
  }

  if (blocks.join('') !== normalizeBlockLineEndings(markdown)) {
    return restoreAll()
  }

  const finalizedBlocks: string[] = []
  let htmlLikeMarkupSeen = false
  let rawOffset = 0

  for (const block of blocks) {
    const rawEnd = rawEndForNormalizedLength(markdown, rawOffset, block.length)

    if (rawEnd === null) {
      return restoreAll()
    }

    const rawBlock = markdown.slice(rawOffset, rawEnd)

    if (normalizeBlockLineEndings(rawBlock) !== block) {
      return restoreAll()
    }

    const ownedToken = htmlLikeMarkupSeen ? null : standaloneOwnedTokenBlock(rawBlock, protectionCodec)
    const source = ownedToken ? protectionCodec.decode(ownedToken.token) : null
    const claimToken = source ? claimCodec.encode(source) : null

    const finalizedBlock =
      ownedToken && claimToken
        ? ownedToken.leading + claimToken + ownedToken.trailing
        : protectionCodec.restoreOwnedTokens(rawBlock)

    finalizedBlocks.push(finalizedBlock)
    htmlLikeMarkupSeen ||= HTML_LIKE_MARKUP_RE.test(finalizedBlock)
    rawOffset = rawEnd
  }

  return rawOffset === markdown.length ? finalizedBlocks.join('') : restoreAll()
}
