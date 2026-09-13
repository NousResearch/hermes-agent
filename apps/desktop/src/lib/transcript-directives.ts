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
 * The product parser requires the entire paragraph to be one directive, so
 * mid-prose text and malformed or unclaimed directives stay prose.
 *
 * For the guided chat segmenter, the guard is the CLAIM, not its position:
 * a name nobody registered — and a malformed one — stays exactly the text it
 * always was. Position used to be the guard too (a directive had to be the
 * whole paragraph), and that cost more than it bought: a model that wrote the
 * directive at the end of its sentence instead of alone under it put raw
 * `::onboarding{step="look"}` in front of the user AND swallowed the card,
 * which on a step whose card is the only way forward stops the conversation
 * dead. So a directive is recognised wherever it starts a word, and the
 * paragraph around it keeps rendering as prose.
 *
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

export type TranscriptParagraphSegment =
  { kind: 'prose'; text: string } | { kind: 'directive'; directive: ParsedTranscriptDirective }

// The whole paragraph, nothing else on the line: `::name` or `::name{...}`.
//
// BODY: a run of quoted strings and plain characters, so a brace inside a
// quoted value (`p1="stash@{0}"`) is content while a brace outside one still
// fails the match. The alternation branches start with distinct characters, so
// there is nothing ambiguous to backtrack over. The repetition counts
// alternation groups rather than characters, so the real bound on body length
// is the 1200-char guard in parseTranscriptDirective below.
//
// TRAILING DEBRIS is tolerated after the closing brace. A directive's
// attribute values are natural language, so an unpaired `*`, `_`, backtick or
// `~~` inside a prompt makes an incomplete-markdown repair append a synthetic
// closer AFTER the `}` (`::followup{p1="wt-* worktrees"}*`). Strict matching
// turned that one stray character into a silently unrendered panel.
//
// Only markdown's inline CLOSER punctuation is forgiven, never letters,
// digits or `}`: real prose after a directive still disqualifies the
// paragraph, so this cannot start hijacking mid-sentence text.
const DIRECTIVE_RE = /^::([a-z][a-z0-9-]{0,63})(?:\{((?:"[^"]*"|'[^']*'|[^{}"']){0,1024})\})?([*_`~\s]{0,8})$/

// `::name` or `::name{...}`, anywhere a word can start — so `std::vector` is
// never a directive. Length caps bound the attr scan on adversarial input.
const SEGMENT_RE = /(?<=^|\s)::([a-z][a-z0-9-]{0,63})(?:\{([^{}]{0,1024})\})?/g

// `key="value"` pairs; single quotes accepted for model sloppiness.
const ATTR_RE = /([a-z][\w-]{0,63})=(?:"([^"]*)"|'([^']*)')/gi

/** Cheap gate: could this paragraph be addressing a directive at all? */
export function looksLikeDirective(text: string): boolean {
  return /^\s*::[a-z]/.test(text)
}

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
  const body = match[2] ?? ''

  if (body) {
    for (const pair of body.matchAll(ATTR_RE)) {
      attrs[pair[1].toLowerCase()] = pair[2] ?? pair[3] ?? ''
    }

    // A brace body that yields no attributes is a malformed directive, not an
    // attribute-less one: `::followup{p1=unquoted}` would otherwise parse
    // "successfully" into an empty-props panel that renders blank. Reject it
    // so the caller reports a drop instead of mounting an empty widget.
    if (Object.keys(attrs).length === 0 && body.trim() !== '') {
      return null
    }
  }

  // `source` is the directive proper — trailing repair debris is not part of
  // what the model addressed, and plugins echo `source` in diagnostics.
  const debris = match[3] ?? ''
  const source = debris ? trimmed.slice(0, trimmed.length - debris.length) : trimmed

  return { name: match[1], attrs, source }
}

/**
 * Why a directive-looking paragraph did not parse, in one human sentence, or
 * null when there is nothing to report.
 *
 * The failure this exists for is silent by construction: the paragraph renders
 * as its own raw source, which reads like the model emitted junk rather than
 * like the app dropped a widget. Callers log this so the NEXT such regression
 * announces itself instead of needing a bisect.
 */
export function describeDirectiveParseFailure(text: string): string | null {
  const trimmed = text.trim()

  if (!looksLikeDirective(trimmed) || parseTranscriptDirective(trimmed) !== null) {
    return null
  }

  if (trimmed.includes('\n')) {
    return 'directive spans multiple lines (must be one paragraph)'
  }

  if (trimmed.length > 1200) {
    return `directive is ${trimmed.length} chars (max 1200)`
  }

  const open = trimmed.indexOf('{')

  if (open >= 0 && !trimmed.includes('}')) {
    return 'attribute brace is never closed'
  }

  // A brace inside a quoted value is content, so the closer is the last `}`
  // that is not inside quotes. Scan once rather than guessing with indexOf.
  const close = open >= 0 ? unquotedClosingBrace(trimmed, open) : -1

  if (close >= 0 && close < trimmed.length - 1) {
    return `unexpected text after the closing brace: ${JSON.stringify(trimmed.slice(close + 1))}`
  }

  if (!/^::[a-z][a-z0-9-]{0,63}/.test(trimmed)) {
    return 'directive name must be lowercase [a-z][a-z0-9-]*'
  }

  if (open >= 0 && close > open && !/=\s*["']/.test(trimmed.slice(open + 1, close))) {
    return 'attribute values must be quoted, e.g. key="value"'
  }

  return 'directive did not match ::name{key="value"}'
}

/** Index of the attribute block's closing brace, ignoring quoted braces. */
function unquotedClosingBrace(text: string, open: number): number {
  let quote: string | null = null

  for (let index = open + 1; index < text.length; index += 1) {
    const char = text[index]

    if (quote !== null) {
      if (char === quote) {
        quote = null
      }

      continue
    }

    if (char === '"' || char === "'") {
      quote = char

      continue
    }

    if (char === '}') {
      return index
    }
  }

  return -1
}

function parseAttrs(body: string | undefined): ParsedTranscriptDirective['attrs'] {
  const attrs: Record<string, string> = {}

  for (const pair of (body ?? '').matchAll(ATTR_RE)) {
    attrs[pair[1].toLowerCase()] = pair[2] ?? pair[3] ?? ''
  }

  return attrs
}

/**
 * True when a STILL-STREAMING paragraph should be withheld as a directive in
 * progress. Deltas land ~3 chars at a time, and `::ask{question="Wha` cannot
 * parse until the final `}` lands — exactly the window where raw directive
 * text used to flash. A lone `:` is the same line one delta earlier. The
 * check covers the paragraph-leading case (the authored shape for onboarding
 * cards); a directive a model appends mid-sentence streams as prose until it
 * completes, which reads as ordinary typing rather than leaked markup.
 *
 * Only ever consult this while the message is streaming: a SETTLED paragraph
 * that starts with `::` but doesn't parse is an authoring bug the user should
 * see as text, and callers must keep that behavior.
 */
export function isDirectiveInProgress(text: string): boolean {
  const trimmed = text.trimStart()

  return trimmed === ':' || trimmed.startsWith('::')
}

/**
 * Split a paragraph into its prose runs and the directives embedded in them,
 * in the order they were written. Null when it holds no directive at all.
 *
 * Pure and synchronous — safe to call during render. Deciding which of these
 * are real is the caller's job: only a claimed name becomes a card, so an
 * unregistered `::whatever` is folded straight back into the prose it came in.
 */
export function segmentTranscriptDirectives(text: string): TranscriptParagraphSegment[] | null {
  if (!text.includes('::') || text.length > 4800) {
    return null
  }

  const out: TranscriptParagraphSegment[] = []
  let cursor = 0

  SEGMENT_RE.lastIndex = 0

  for (const match of text.matchAll(SEGMENT_RE)) {
    const start = match.index ?? 0

    // A brace the attr group refused (unclosed, or past the length cap) means
    // the name matched but its attributes did not. Half of a directive is not
    // one: render a card with the attributes silently dropped and it is broken
    // in a way nobody can see. Leave the whole thing as the text it is.
    if (match[2] === undefined && text[start + match[0].length] === '{') {
      continue
    }

    if (start > cursor) {
      out.push({ kind: 'prose', text: text.slice(cursor, start) })
    }

    out.push({
      kind: 'directive',
      directive: { name: match[1], attrs: parseAttrs(match[2]), source: match[0] }
    })
    cursor = start + match[0].length
  }

  if (out.length === 0) {
    return null
  }

  if (cursor < text.length) {
    out.push({ kind: 'prose', text: text.slice(cursor) })
  }

  return out
}
