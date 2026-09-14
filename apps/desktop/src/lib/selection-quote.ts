/**
 * Turning a transcript selection into side-chat context.
 *
 * A selection travels as TEXT in the side chat's composer draft, never as an
 * attachment or an `@…:` reference: the draft stash persists text only
 * (`loadPersistedDraftTexts` restores `{ attachments: [], text }`), no existing
 * reference kind carries raw text (the ones that expand server-side resolve a
 * path), and `@session:` is display-only. Text also needs no backend change, and
 * it keeps the quote and the user's question inside ONE user turn — the app's
 * strict role-alternation rule forbids a synthetic user turn followed by a real
 * one.
 *
 * Everything here is pure so the quoting rules are testable without a DOM.
 */

/** Cap on a quoted selection. A selection can be a whole message, and the quote
 *  plus the question still has to fit one composer draft comfortably. */
export const SELECTION_QUOTE_MAX_CHARS = 4000

const TRUNCATION_MARKER = '…(truncated)'

const TITLE_MAX_CHARS = 40

export interface SelectionQuote {
  text: string
  truncated: boolean
}

/**
 * Quote every line, empty ones included. A blank line left unquoted would END
 * the blockquote and drop the rest of the selection — inside a fenced code block
 * that silently mangles the context the model is handed.
 */
export function quoteBlock(text: string): string {
  return text
    .split('\n')
    .map(line => {
      const trimmed = line.trimEnd()

      return trimmed ? `> ${trimmed}` : '>'
    })
    .join('\n')
}

/**
 * The quote block for a raw selection, clipped at a LINE boundary when it is
 * over budget (a half-line quote reads like a typo) with a marker line making
 * the omission visible to the reader and to the model.
 */
export function buildSelectionQuote(raw: string, max: number = SELECTION_QUOTE_MAX_CHARS): SelectionQuote {
  const text = raw.replace(/\r\n?/g, '\n').trim()

  if (!text) {
    return { text: '', truncated: false }
  }

  const whole = quoteBlock(text)

  if (whole.length <= max) {
    return { text: whole, truncated: false }
  }

  const markerCost = TRUNCATION_MARKER.length + 3 // `> ` + the marker + the joining newline
  const kept: string[] = []

  for (const line of text.split('\n')) {
    const candidate = kept.length ? quoteBlock([...kept, line].join('\n')) : quoteBlock(line)

    if (candidate.length + markerCost > max) {
      break
    }

    kept.push(line)
  }

  // One pathological line (a whole file's worth of text with no newline) still
  // has to yield something quotable, even though it cannot respect the cap.
  if (!kept.length) {
    kept.push(text.slice(0, Math.max(1, max - markerCost)))
  }

  return { text: `${quoteBlock(kept.join('\n'))}\n> ${TRUNCATION_MARKER}`, truncated: true }
}

/**
 * The side chat's session title, derived from the selection. Selections are
 * usually code, so leading punctuation and backticks go first — otherwise the
 * sidebar fills up with rows starting `About: ``` or `About: const`.
 *
 * The prefix is passed in rather than baked in: it is user-facing copy and
 * belongs to the caller's translations.
 */
export function deriveSideChatTitle(raw: string, prefix = 'About:'): string {
  const flat = raw
    .replace(/[\r\n]+/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/^[\s`'"([{<*#>_~=|/\\-]+/, '')
    // Trailing markdown fences and quotes only: brackets are left alone so
    // `new Map()` does not lose its call parens.
    .replace(/[\s`'"]+$/, '')
    .trim()

  if (!flat) {
    return `${prefix} selection`.trim()
  }

  if (flat.length <= TITLE_MAX_CHARS) {
    return `${prefix} ${flat}`.trim()
  }

  // Prefer a word boundary, but never return an empty tail from a long
  // single-token selection (minified code, a URL).
  const clipped = flat.slice(0, TITLE_MAX_CHARS).replace(/\s+\S*$/, '').trim() || flat.slice(0, TITLE_MAX_CHARS)

  return `${prefix} ${clipped}…`.trim()
}
