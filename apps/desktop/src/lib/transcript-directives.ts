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
 * What keeps a directive from hijacking prose is the CLAIM, not its position:
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
  | { kind: 'prose'; text: string }
  | { kind: 'directive'; directive: ParsedTranscriptDirective }

// `::name` or `::name{...}`, anywhere a word can start — so `std::vector` is
// never a directive. Length caps bound the attr scan on adversarial input.
const DIRECTIVE_RE = /(?<=^|\s)::([a-z][a-z0-9-]{0,63})(?:\{([^{}]{0,1024})\})?/g

// `key="value"` pairs; single quotes accepted for model sloppiness.
const ATTR_RE = /([a-z][\w-]{0,63})=(?:"([^"]*)"|'([^']*)')/gi

function parseAttrs(body: string | undefined): Record<string, string> {
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

  DIRECTIVE_RE.lastIndex = 0

  for (const match of text.matchAll(DIRECTIVE_RE)) {
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
