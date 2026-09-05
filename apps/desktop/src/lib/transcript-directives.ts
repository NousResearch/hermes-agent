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

// The whole paragraph, nothing else on the line: `::name` or `::name{...}`.
// Length caps bound the attr scan on adversarial input.
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
const DIRECTIVE_RE = /^::([a-z][a-z0-9-]{0,63})(?:\{([^{}]{0,1024})\})?([*_`~\s]{0,8})$/

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

  // Attribute values cannot contain braces, so the FIRST `}` after the opener
  // is the real closer — anything past it is debris. `lastIndexOf` would miss
  // the case where the debris IS a brace (`::name{…}}`).
  const close = open >= 0 ? trimmed.indexOf('}', open) : -1

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

