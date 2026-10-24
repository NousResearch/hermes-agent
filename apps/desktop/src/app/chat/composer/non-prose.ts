/**
 * `composer.nonProse` — the app side.
 *
 * A plugin declares half-open `[start, end)` ranges of the DRAFT as non-prose
 * (a code span, a path, a flag) and the app paints each range as a
 * `<span data-composer-nonprose spellcheck="false">` inside its own render
 * pass, then owns that node from there on.
 *
 * Why a wrapper at all: Chromium paints its squiggle from the DOM it holds,
 * and `spellcheck` is an inherited per-element attribute with no per-range
 * opt-out — a wrapper is the only lever. Why the app and not the plugin: the
 * wrapper is a node on a tree `normalizeComposerEditorDom` rewrites on every
 * keystroke, so a plugin writing it is a second writer, and the caret jumps
 * whenever the reconciler moves what it wrote. The rule the context-menu slot
 * states holds here too — a provider contributes DATA (offsets into a string),
 * never an element, a `Range` or a selection; resolve, normalize, cap, paint
 * and unwrap are this module's, and `rich-editor` is the only caller.
 *
 * Nothing here runs unless a provider is registered AND the surface has opted
 * into spell checking: a surface Chromium isn't checking has nothing for a
 * wrapper to suppress, and with no provider there is no call, no attribute and
 * no DOM diff.
 */
import { registry } from '@/contrib/registry'
import type { Contribution } from '@/contrib/types'

import {
  COMPOSER_AREAS,
  type ComposerNonProseContext,
  type ComposerNonProseProvider,
  type ComposerNonProseRange
} from './contrib'

/** The attribute core paints on every wrapper it emits. Its presence is the
 *  contract: a node carrying it is core's, never junk, and never a chip. */
export const COMPOSER_NON_PROSE_ATTR = 'data-composer-nonprose'

/** The attribute a composer surface stamps its session address with. It makes
 *  the `sessionId` a provider sees the same value in the render pass and in
 *  the reconcile pass; without it a session-scoped provider would answer
 *  differently between the two and its wrappers would thrash. */
export const COMPOSER_SESSION_ATTR = 'data-composer-session'

/** Ranges one provider may declare per pass; the rest are dropped. */
export const COMPOSER_NON_PROSE_PER_PROVIDER = 64

/** Ranges core paints per pass across every provider, after merging. */
export const COMPOSER_NON_PROSE_TOTAL = 256

export type ComposerNonProseKind = NonNullable<ComposerNonProseRange['kind']>

/** A declared range after clamping, line splitting, merging and capping — the
 *  only shape the renderer and the reconciler ever see. */
export interface NormalizedComposerNonProseRange {
  end: number
  kind: ComposerNonProseKind
  start: number
}

/** A range reduced to a render atom: the same shape `chipSpans` returns, so
 *  `appendComposerContents` can interleave chips and wrappers in one pass. */
export interface ComposerNonProseSpan {
  end: number
  node: () => HTMLElement
  start: number
}

const EMPTY: readonly NormalizedComposerNonProseRange[] = Object.freeze([])

const KINDS: readonly string[] = ['code', 'path', 'other']

/**
 * Whether `element` effectively has spell checking on — the seam's gate.
 *
 * Read from the ATTRIBUTE, walking ancestors, because that is exactly what
 * Chromium does: `spellcheck` is inherited, absent means "inherit", and a
 * nearest `spellcheck="false"` turns it off for the whole subtree. Reading the
 * attribute (rather than the IDL property) also keeps the gate honest in a
 * test DOM that doesn't implement the property for every element.
 */
function spellCheckEnabled(element: HTMLElement): boolean {
  for (let node: HTMLElement | null = element; node; node = node.parentElement) {
    const declared = node.getAttribute('spellcheck')

    if (declared !== null) {
      return declared.toLowerCase() !== 'false'
    }
  }

  return true
}

/** `kind` is a paint hint only; anything unrecognised paints `code`. */
function kindOf(raw: unknown): ComposerNonProseKind {
  return typeof raw === 'string' && KINDS.includes(raw) ? (raw as ComposerNonProseKind) : 'code'
}

/**
 * One declared range → the pieces of it a wrapper can actually cover: clamped
 * to `text`, split at every `\n` (a wrapper cannot straddle a `<br>`), with
 * whitespace-only pieces dropped (they'd paint an element around nothing and
 * read as content to the placeholder marker).
 */
function splitRange(raw: unknown, text: string): NormalizedComposerNonProseRange[] {
  if (!raw || typeof raw !== 'object') {
    return []
  }

  const { end, kind, start } = raw as Partial<ComposerNonProseRange>

  // Offsets are UTF-16 indices, the units `String.prototype.slice` and the
  // composer's own serialization use. A non-integer is a broken provider, not
  // a range to guess at.
  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    return []
  }

  const from = Math.max(0, Math.min(start as number, text.length))
  const to = Math.max(from, Math.min(end as number, text.length))
  const pieces: NormalizedComposerNonProseRange[] = []
  let cursor = from

  while (cursor < to) {
    const newline = text.indexOf('\n', cursor)
    const stop = newline === -1 || newline >= to ? to : newline

    if (stop > cursor && /\S/.test(text.slice(cursor, stop))) {
      pieces.push({ end: stop, kind: kindOf(kind), start: cursor })
    }

    cursor = stop + 1
  }

  return pieces
}

/**
 * Sort, merge overlapping AND adjacent ranges into one, and cap the result.
 * Adjacency counts as overlap on purpose: two providers naming halves of one
 * span is one wrapper, not two nodes with a seam in it. The earliest range in
 * order keeps its `kind`.
 */
function mergeRanges(
  ranges: readonly NormalizedComposerNonProseRange[],
  cap: number
): NormalizedComposerNonProseRange[] {
  const merged: NormalizedComposerNonProseRange[] = []

  for (const range of [...ranges].sort((a, b) => a.start - b.start || a.end - b.end)) {
    const last = merged[merged.length - 1]

    if (last && range.start <= last.end) {
      last.end = Math.max(last.end, range.end)
    } else if (merged.length < cap) {
      merged.push({ ...range })
    }
  }

  return merged
}

/**
 * Resolve every `composer.nonProse` contribution against one draft into the
 * single range set core paints (or reconciles against).
 *
 * Per provider: rows are read in the order returned, invalid rows are skipped,
 * and at most `COMPOSER_NON_PROSE_PER_PROVIDER` pieces are kept. Across
 * providers: merged, then capped at `COMPOSER_NON_PROSE_TOTAL`. A provider that
 * throws, or returns a non-array, contributes nothing and leaves every other
 * provider's ranges standing.
 */
export function composerNonProseRanges(
  contributions: readonly Contribution[],
  context: ComposerNonProseContext
): readonly NormalizedComposerNonProseRange[] {
  if (!context.text || contributions.length === 0) {
    return EMPTY
  }

  const pieces: NormalizedComposerNonProseRange[] = []

  for (const contribution of contributions) {
    const provider = contribution.data as Partial<ComposerNonProseProvider> | undefined

    if (typeof provider?.ranges !== 'function') {
      continue
    }

    let declared: unknown

    try {
      declared = provider.ranges(context)
    } catch (error) {
      console.warn(`[composer.nonProse] ${contribution.source ?? 'core'} provider failed`, error)

      continue
    }

    if (!Array.isArray(declared)) {
      continue
    }

    let kept = 0

    for (const row of declared) {
      if (kept >= COMPOSER_NON_PROSE_PER_PROVIDER) {
        break
      }

      const range = splitRange(row, context.text)

      pieces.push(...range)
      kept += range.length
    }
  }

  return mergeRanges(pieces, COMPOSER_NON_PROSE_TOTAL)
}

/**
 * The ranges this surface paints right now, or `[]`.
 *
 * The gate: only a surface that has opted into spell checking is decorated
 * (`spellcheck` is inherited, so this reads the value the surface effectively
 * has — the same attribute that decides whether Chromium checks it at all),
 * and only while at least one provider is registered.
 */
export function composerNonProseSurfaceRanges(
  target: DocumentFragment | HTMLElement,
  text: string
): readonly NormalizedComposerNonProseRange[] {
  if (!(target instanceof HTMLElement) || !spellCheckEnabled(target)) {
    return EMPTY
  }

  const contributions = registry.getArea(COMPOSER_AREAS.nonProse)

  if (contributions.length === 0) {
    return EMPTY
  }

  return composerNonProseRanges(contributions, {
    sessionId: target.closest(`[${COMPOSER_SESSION_ATTR}]`)?.getAttribute(COMPOSER_SESSION_ATTR) ?? '',
    text
  })
}

/**
 * The wrapper core paints for one range.
 *
 * Deliberately NO `data-ref-text`: with it absent the wrapper serializes as its
 * own `textContent`, so submit, undo, draft restore and the `@`/`/` scanners
 * read exactly the string they read today — and the reconciler's chip rules
 * (`dataset.refText`) never mistake it for a chip. It also keeps the wrapper
 * editable: no `contenteditable="false"`, unlike a chip.
 */
export function composerNonProseElement(text: string, kind: string): HTMLElement {
  const wrapper = document.createElement('span')

  wrapper.dataset.composerNonprose = kind
  wrapper.setAttribute('spellcheck', 'false')
  wrapper.textContent = text

  return wrapper
}

function spanOf(text: string, start: number, end: number, kind: ComposerNonProseKind): ComposerNonProseSpan {
  return { end, node: () => composerNonProseElement(text.slice(start, end), kind), start }
}

/**
 * The render atoms for one pass, with every range clipped against `blocked` —
 * the chip spans already emitted for the same text, sorted by `start`. A chip
 * inside a wrapper would lose its atomic caret and its backspace behaviour, so
 * the chip wins the overlap the same way it wins one against another chip.
 */
export function composerNonProseSpans(
  target: DocumentFragment | HTMLElement,
  text: string,
  blocked: readonly { end: number; start: number }[]
): ComposerNonProseSpan[] {
  const ranges = composerNonProseSurfaceRanges(target, text)

  if (ranges.length === 0) {
    return []
  }

  const spans: ComposerNonProseSpan[] = []

  for (const range of ranges) {
    let cursor = range.start

    for (const block of blocked) {
      if (block.end <= cursor) {
        continue
      }

      if (block.start >= range.end) {
        break
      }

      if (block.start > cursor) {
        spans.push(spanOf(text, cursor, Math.min(block.start, range.end), range.kind))
      }

      cursor = Math.max(cursor, block.end)
    }

    if (cursor < range.end) {
      spans.push(spanOf(text, cursor, range.end, range.kind))
    }
  }

  return spans
}
