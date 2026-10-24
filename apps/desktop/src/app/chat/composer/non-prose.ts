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
 * no DOM diff. That gate is also the seam's dependency: both desktop composers
 * hard-code `spellcheck="false"` today (#44415), so only
 * `host.composer.setSpellcheck` — plugin-sdk #143057 — can open it. Until that
 * lands a provider is registered and never asked for a range, which is why the
 * public docs call the seam inert on a shell without it.
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

/** Pieces (not declarations) one provider may contribute per pass. A single
 *  declaration spanning N lines is N wrappers, and each one is a node in the
 *  same budget, so the count is taken as the pieces are produced. */
export const COMPOSER_NON_PROSE_PER_PROVIDER = 64

/** Ranges core paints per pass across every provider, after merging. Filled
 *  round-robin (see `composerNonProseRanges`) so one provider cannot spend it
 *  all. */
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

/** True when `offset` sits between the halves of a surrogate pair. */
function splitsSurrogatePair(text: string, offset: number): boolean {
  if (offset <= 0 || offset >= text.length) {
    return false
  }

  const high = text.charCodeAt(offset - 1)
  const low = text.charCodeAt(offset)

  return high >= 0xd800 && high <= 0xdbff && low >= 0xdc00 && low <= 0xdfff
}

/**
 * One declared range → the pieces of it a wrapper can actually cover: clamped
 * to `text`, split at every `\n` (a wrapper cannot straddle a `<br>`), with
 * whitespace-only pieces dropped (they'd paint an element around nothing and
 * read as content to the placeholder marker).
 *
 * Both ends snap OUTWARD off a surrogate pair: a DOM node cannot hold half an
 * emoji, so `[1, 2)` of `a😀b` covers the whole pair rather than leaving an
 * unpaired high surrogate in the text node (which serializes back to the same
 * string, but mangles any innerHTML path).
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

  const declaredFrom = Math.max(0, Math.min(start as number, text.length))
  const declaredTo = Math.max(declaredFrom, Math.min(end as number, text.length))
  const from = splitsSurrogatePair(text, declaredFrom) ? declaredFrom - 1 : declaredFrom
  const to = splitsSurrogatePair(text, declaredTo) ? Math.min(declaredTo + 1, text.length) : declaredTo
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
 * and at most `COMPOSER_NON_PROSE_PER_PROVIDER` PIECES are kept — a row that
 * splits into more pieces than the budget has room for is cut, not added
 * whole. Across providers: the shared `COMPOSER_NON_PROSE_TOTAL` budget is
 * handed out round-robin, so one provider's long declaration cannot crowd the
 * others out of a pass; the result is then merged and capped. A provider that
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

  const perProvider: NormalizedComposerNonProseRange[][] = []

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

    const pieces: NormalizedComposerNonProseRange[] = []

    for (const row of declared) {
      const room = COMPOSER_NON_PROSE_PER_PROVIDER - pieces.length

      if (room <= 0) {
        break
      }

      const range = splitRange(row, context.text)

      // Count PIECES against the budget: one declaration over a 400-line draft
      // is 400 wrappers, and 400 is what it used to add (the cap was consulted
      // once per declaration, before the split).
      if (range.length > room) {
        pieces.push(...range.slice(0, room))
      } else {
        pieces.push(...range)
      }
    }

    if (pieces.length > 0) {
      perProvider.push(pieces)
    }
  }

  // Round-robin: every provider's first piece, then every provider's second,
  // and so on. A provider with fewer pieces than its turn leaves the slack to
  // the rest, so the full budget is still spent when there is enough to spend,
  // and no provider is served only after another has taken all 256.
  const picked: NormalizedComposerNonProseRange[] = []

  for (let turn = 0; picked.length < COMPOSER_NON_PROSE_TOTAL; turn++) {
    let served = false

    for (const pieces of perProvider) {
      if (turn >= pieces.length) {
        continue
      }

      picked.push(pieces[turn])
      served = true

      if (picked.length >= COMPOSER_NON_PROSE_TOTAL) {
        break
      }
    }

    if (!served) {
      break
    }
  }

  return mergeRanges(picked, COMPOSER_NON_PROSE_TOTAL)
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
 * Clip every range around `blocked` — the chip spans already emitted for the
 * same text, sorted by `start`. A chip inside a wrapper would lose its atomic
 * caret and its backspace behaviour, so the chip wins the overlap the same way
 * it wins one against another chip.
 *
 * Shared by both passes on purpose: the render pass turns the result into
 * nodes, and the reconcile pass compares it with the nodes already on screen.
 * If they clipped differently the reconciler would unwrap and re-paint the
 * same span on every keystroke.
 */
export function clipComposerNonProseRanges(
  ranges: readonly NormalizedComposerNonProseRange[],
  blocked: readonly { end: number; start: number }[]
): NormalizedComposerNonProseRange[] {
  const clipped: NormalizedComposerNonProseRange[] = []

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
        clipped.push({ end: Math.min(block.start, range.end), kind: range.kind, start: cursor })
      }

      cursor = Math.max(cursor, block.end)
    }

    if (cursor < range.end) {
      clipped.push({ end: range.end, kind: range.kind, start: cursor })
    }
  }

  return clipped
}

/** The render atoms for one pass. */
export function composerNonProseSpans(
  target: DocumentFragment | HTMLElement,
  text: string,
  blocked: readonly { end: number; start: number }[]
): ComposerNonProseSpan[] {
  const ranges = clipComposerNonProseRanges(composerNonProseSurfaceRanges(target, text), blocked)

  return ranges.map(range => spanOf(text, range.start, range.end, range.kind))
}
