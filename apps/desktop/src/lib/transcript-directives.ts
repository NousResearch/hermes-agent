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
 * The guard is the CLAIM, not its position: a name nobody registered — and a malformed one — stays exactly the text it
 * always was. Position used to be the guard too (a directive had to be the
 * whole paragraph), and that cost more than it bought: a model that wrote the
 * directive at the end of its sentence instead of alone under it put the raw
 * directive in front of the user AND swallowed the card. So a directive is recognised wherever it starts a word, and the
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

// Brace-body cap. A signed CDN URL alone runs 500–1,000+ chars (Meshy's are
// ~520), and a directive must fit two of them plus a title. Still a bound, so
// the attr scan stays linear on adversarial input.
export const DIRECTIVE_BODY_MAX = 2048

// `::name` or `::name{...}`, anywhere a word can start — so `std::vector` is
// never a directive.
const SEGMENT_RE = new RegExp(
  String.raw`(?<=^|\s)::([a-z][a-z0-9-]{0,63})(?:\{([^{}]{0,${DIRECTIVE_BODY_MAX}})\})?`,
  'g'
)

/** A directive with a brace body on one line: the span markdown preprocessing
 *  must hand to the parser untouched (its attributes are data, not prose). */
export const DIRECTIVE_WITH_BODY_RE = new RegExp(
  String.raw`(?<=^|\s)::[a-z][a-z0-9-]{0,63}\{[^{}\n]{0,${DIRECTIVE_BODY_MAX}}\}`,
  'g'
)

// A text node that opens a directive brace it never closes.
const OPEN_DIRECTIVE_RE = /(?<=^|\s)::[a-z][a-z0-9-]{0,63}\{[^{}]*$/

interface MdNode {
  children?: MdNode[]
  type: string
  value?: string
}

function inlineText(node: MdNode): string | null {
  if (node.type === 'text') {
    return node.value ?? ''
  }

  // A GFM literal autolink keeps its source verbatim, backslash escapes
  // included (preprocess escapes `_`/`&`/`~` inside directives); a URL never
  // holds a literal `\`, so drop the CommonMark escapes to get it as written.
  // Only a literal autolink (its text IS its URL) folds: an explicit
  // `[label](url)` is the author's link and stays clickable.
  if (node.type === 'link' && node.children?.every(child => child.type === 'text')) {
    // GFM keeps the escapes in both the literal's text and its url, so compare them raw.
    const raw = node.children.map(child => child.value ?? '').join('')
    const url = (node as MdNode & { url?: string }).url ?? ''

    if (raw !== url && `http://${raw}` !== url && `mailto:${raw}` !== url) {
      return null
    }

    return raw.replace(/\\([!-/:-@[-`{-~])/g, '$1')
  }

  return null
}

function foldDirectiveRuns(node: MdNode) {
  const kids = node.children

  if (!kids) {
    return
  }

  const out: MdNode[] = []

  for (let i = 0; i < kids.length; i += 1) {
    const child = kids[i]

    if (child.type === 'text' && OPEN_DIRECTIVE_RE.test(child.value ?? '')) {
      let merged = child.value ?? ''
      let closed = -1

      for (let j = i + 1; j < kids.length && merged.length <= 4800; j += 1) {
        const piece = inlineText(kids[j])

        if (piece === null) {
          break
        }

        merged += piece

        if (piece.includes('}')) {
          closed = j

          break
        }
      }

      if (closed !== -1) {
        // Re-scan the folded node: its tail may open the next directive.
        kids[closed] = { type: 'text', value: merged }
        i = closed - 1

        continue
      }
    }

    foldDirectiveRuns(child)
    out.push(child)
  }

  node.children = out
}

/**
 * remark plugin. GFM links every bare URL in a text node — including one
 * inside a directive attribute (`url="https://…"`) — and that link element
 * splits the paragraph, so it is no longer text-only and the card is never
 * claimed. Fold a run that opens a directive brace in a text node and closes
 * it within text/link siblings back into one text node; the link's text is
 * the URL as written, so the directive comes back byte-for-byte.
 */
export function remarkTranscriptDirectives() {
  return (tree: MdNode) => foldDirectiveRuns(tree)
}

// `key="value"` pairs; single quotes accepted for model sloppiness.
const ATTR_RE = /([a-z][\w-]{0,63})=(?:"([^"]*)"|'([^']*)')/gi

function parseAttrs(body: string | undefined): ParsedTranscriptDirective['attrs'] {
  const attrs: Record<string, string> = {}

  for (const pair of (body ?? '').matchAll(ATTR_RE)) {
    // Models copy the `<https://…>` autolink form into attributes too.
    attrs[pair[1].toLowerCase()] = (pair[2] ?? pair[3] ?? '').replace(/^<(https?:\/\/[^<>\s]+)>$/i, '$1')
  }

  return attrs
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
