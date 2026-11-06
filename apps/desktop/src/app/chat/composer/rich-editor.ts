/**
 * Helpers for the contenteditable composer surface: serialize refs to chip
 * HTML, walk the DOM back to plain `@kind:value` text, and place the caret.
 *
 * Chip values are always wrapped in backticks/quotes so REF_RE stops at the
 * fence — without that, typing after a chip would get re-absorbed on the next
 * plain-text round-trip.
 */
import {
  directiveIconElement,
  directiveIconSvg,
  formatRefValue,
  refAttrsHtml,
  refChipLabel,
  type SlashChipKind,
  slashIconElement
} from '@/components/assistant-ui/directive-text'
import { referenceKind, referenceRe, unwrapReferenceValue } from '@/components/assistant-ui/reference-kinds'
import { registry } from '@/contrib/registry'

import { COMPOSER_AREAS } from './contrib'
import {
  clipComposerNonProseRanges,
  COMPOSER_NON_PROSE_ATTR,
  composerNonProseElement,
  composerNonProseEnabled,
  type ComposerNonProseKind,
  composerNonProseSpans,
  composerNonProseSurfaceRanges
} from './non-prose'
import { slashCommandMatches, type SlashCommandScanOptions } from './slash-refs'

export const RICH_INPUT_SLOT = 'composer-rich-input'

/** Chromium's litter: editing beside a `contenteditable=false` chip splits the
 *  line and leaves zero-length text nodes behind. They render as nothing and
 *  serialize as nothing, so no reader of the editor should count them. */
function isEmptyTextNode(node: ChildNode | null): boolean {
  return node?.nodeType === Node.TEXT_NODE && !node.textContent
}

/** The node before `node`, stepping over that litter. */
function meaningfulPreviousSibling(node: ChildNode | null): ChildNode | null {
  let prev = node?.previousSibling ?? null

  while (isEmptyTextNode(prev)) {
    prev = prev?.previousSibling ?? null
  }

  return prev
}

/** The node after `node`, stepping over that litter. */
function meaningfulNextSibling(node: ChildNode | null): ChildNode | null {
  let next = node?.nextSibling ?? null

  while (isEmptyTextNode(next)) {
    next = next?.nextSibling ?? null
  }

  return next
}

/** The live collapsed selection container, if it belongs to this editor. */
function composerCollapsedSelectionContainer(editor: HTMLElement): Node | null {
  const selection = window.getSelection()

  if (!selection?.isCollapsed || selection.rangeCount === 0) {
    return null
  }

  const range = selection.getRangeAt(0)

  return editor.contains(range.startContainer) ? range.startContainer : null
}

/** Keep the `data-empty` marker the placeholder paints on in step with the
 *  editor root's contents.
 *
 *  `:empty` can't be the whole test: a cleared editor keeps a scaffolding <br>
 *  so the contenteditable doesn't collapse, and that break makes `:empty`
 *  false. Nor can CSS infer it on its own — a text node is invisible to
 *  selectors, so `one<br>` and a lone `<br>` are the same shape, and
 *  `:has(> br:only-child)` would paint the placeholder over the user's text.
 *  The code that empties the editor is what knows, so it marks it.
 *
 *  Zero-length text nodes don't count as contents. Chromium leaves them behind
 *  whenever an edit lands next to a `contenteditable=false` chip, and counting
 *  them left an editor the user had emptied looking occupied. */
export function markEditorEmptiness(editor: HTMLElement) {
  if (Array.from(editor.childNodes).every(isEmptyTextNode)) {
    editor.dataset.empty = ''
  } else {
    delete editor.dataset.empty
  }
}

/** Editors with an IME composition in flight — see `beginComposerComposition`. */
const composingEditors = new WeakSet<HTMLElement>()

/** Editors that currently hold at least one non-prose wrapper core painted.
 *
 *  A WeakSet, so an unmounted editor is collected with everything else. It is
 *  what lets the reconciler answer "is there anything to unwrap here?" without
 *  serializing the draft: `normalizeComposerEditorDom` runs on every keystroke
 *  of every composer, and an app that never registered a provider must not pay
 *  a walk for that. */
const decoratedEditors = new WeakSet<HTMLElement>()

/** Editors already wired to the composition escape hatches below (so a second
 *  composition on the same editor adds no listeners). */
const compositionGuards = new WeakSet<HTMLElement>()

/** End the composition marker for `editor`.
 *
 *  Exported because the composer's own `onBlur` already clears its
 *  `composingRef` for exactly this reason (#44135): Chromium can drop
 *  `compositionend` when focus jumps, the input source changes, or a
 *  programmatic DOM swap aborts the composition mid-preedit. A wedged flag here
 *  is worse than a lost one — it would silently defer every non-prose
 *  reconcile for the rest of that editor's life, so the wrapper set would
 *  freeze at whatever text was on screen when the composition started. */
export function endComposerComposition(editor: HTMLElement) {
  composingEditors.delete(editor)
}

/** Drop the marker as IME composition starts, before any preedit text lands.
 *
 *  Input events during composition are deliberately skipped (they carry
 *  uncommitted preedit text), so nothing else clears the marker until
 *  `compositionend` — and the hint would otherwise sit behind the hiragana the
 *  user is composing. `normalizeComposerEditorDom` restores it if composition
 *  ends with nothing committed.
 *
 *  Composition also defers non-prose re-decoration (see `reconcileNonProse`):
 *  during it the DOM holds preedit text that is not the draft yet, so the
 *  ranges a provider would resolve describe text the user has not committed.
 *  The flag clears on Chromium's own `compositionend`, which always fires
 *  before the composer's React handler flushes — and, because that event can be
 *  missed, on `blur` as well: a composition never survives focus loss, and the
 *  listener is native, so it fires even when React's `onBlur` is what went
 *  missing. */
export function beginComposerComposition(editor: HTMLElement) {
  delete editor.dataset.empty
  composingEditors.add(editor)

  if (!compositionGuards.has(editor)) {
    compositionGuards.add(editor)
    editor.addEventListener('compositionend', () => endComposerComposition(editor))
    editor.addEventListener('blur', () => endComposerComposition(editor))
  }
}

/** @see referenceRe — the shared pattern every surface recognises a reference
 *  with. Module-level `/g` regexes carry `lastIndex`, so call sites reset it. */
export const REF_RE = referenceRe()

const ESC: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }

export function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, ch => ESC[ch] || ch)
}

export function unquoteRef(raw: string) {
  return unwrapReferenceValue(raw)
}

/** Always-quote variant of formatRefValue — chips need a fence even for safe values. */
export function quoteRefValue(value: string) {
  if (!value.includes('`')) {
    return `\`${value}\``
  }

  if (!value.includes('"')) {
    return `"${value}"`
  }

  if (!value.includes("'")) {
    return `'${value}'`
  }

  return formatRefValue(value)
}

/** The serialized `@kind:value` text a ref chip submits as. One function, used
 *  to build the chip AND to measure the draft the non-prose seam resolves
 *  against, so the offsets a provider returns describe the string
 *  `composerPlainText` reads back out of the DOM. */
export function refChipText(kind: string, rawValue: string) {
  return `@${kind}:${quoteRefValue(unquoteRef(rawValue))}`
}

export function refChipHtml(kind: string, rawValue: string, displayLabel?: string) {
  const id = unquoteRef(rawValue)
  const text = refChipText(kind, rawValue)

  const label = displayLabel || refChipLabel(kind, id)

  return `<span contenteditable="false" title="${escapeHtml(id)}" data-ref-text="${escapeHtml(text)}" data-ref-id="${escapeHtml(id)}" data-ref-kind="${escapeHtml(kind)}" ${refAttrsHtml(kind)}>${directiveIconSvg(kind)}${escapeHtml(label)}</span>`
}

export function refChipElement(kind: string, rawValue: string, displayLabel?: string) {
  const id = unquoteRef(rawValue)
  const text = refChipText(kind, rawValue)
  const chip = document.createElement('span')

  chip.contentEditable = 'false'
  chip.title = id
  chip.dataset.refText = text
  chip.dataset.refId = id
  chip.dataset.refKind = kind
  chip.className = 'ref'
  chip.dataset.ref = referenceKind(kind)
  chip.append(directiveIconElement(kind), document.createTextNode(displayLabel || refChipLabel(kind, id)))

  return chip
}

/** A non-editable reference for a picked slash command (`/skin nous`, `/tropes`).
 *  `data-ref-text` carries the literal command so `composerPlainText` round-trips
 *  it back to the exact text that gets submitted. */
export function slashChipElement(command: string, kind: SlashChipKind, label?: string) {
  const chip = document.createElement('span')

  chip.contentEditable = 'false'
  chip.dataset.refText = command
  chip.dataset.slashKind = kind
  chip.className = 'ref'
  chip.dataset.ref = kind
  chip.append(slashIconElement(kind), document.createTextNode(label || command))

  return chip
}

function appendTextWithBreaks(target: DocumentFragment | HTMLElement, text: string) {
  const lines = text.split('\n')

  lines.forEach((line, index) => {
    if (index > 0) {
      target.append(document.createElement('br'))
    }

    if (line) {
      target.append(document.createTextNode(line))
    }
  })
}

/** A chip atom, plus the exact text it contributes to the serialized draft. */
interface ComposerChipSpan {
  end: number
  node: () => HTMLElement
  serialized: string
  start: number
}

/** Every span of `text` that renders as a chip, in source order. */
function chipSpans(text: string, options: SlashCommandScanOptions): ComposerChipSpan[] {
  REF_RE.lastIndex = 0

  const refs = Array.from(text.matchAll(REF_RE)).map(match => {
    const start = match.index ?? 0
    const kind = match[1] || 'file'
    const value = match[2] || ''

    return {
      end: start + match[0].length,
      node: () => refChipElement(kind, value),
      serialized: refChipText(kind, value),
      start
    }
  })

  const commands = slashCommandMatches(text, options).map(match => ({
    end: match.end,
    node: () => slashChipElement(match.command, match.kind),
    // A slash pill serializes as the token it matched, so the source form is
    // already the submitted form.
    serialized: match.command,
    start: match.start
  }))

  return [...refs, ...commands].sort((a, b) => a.start - b.start)
}

/**
 * The draft `text` renders to, with every chip span measured in THAT string.
 *
 * `composerPlainText` reads a chip back as its `data-ref-text`, which is the
 * quoted form even when the draft arrived unquoted (`@file:src/a.ts` →
 * `@file:`src/a.ts``). Resolving a `composer.nonProse` provider against the
 * source form in the render pass and against the serialized form in the
 * reconcile pass therefore handed it two different strings for the same editor
 * — and every offset answered for one landed on the wrong character in the
 * other. Both passes now measure this one string, which is also what submit
 * sends.
 */
function composerRenderedText(text: string, chips: readonly ComposerChipSpan[]) {
  const spans: Array<Omit<ComposerChipSpan, 'serialized'>> = []
  let cursor = 0
  let rendered = ''

  for (const chip of chips) {
    // Same overlap rule as the emit loop below: a chip that starts inside the
    // previous one is not a chip, it is that chip's own text.
    if (chip.start < cursor) {
      continue
    }

    rendered += text.slice(cursor, chip.start)

    const start = rendered.length

    rendered += chip.serialized
    spans.push({ end: rendered.length, node: chip.node, start })
    cursor = chip.end
  }

  rendered += text.slice(cursor)

  return { chips: spans, text: rendered }
}

/** Build the chip/text DOM for `text`. Directives hydrate back to their pills —
 *  `@kind:value` refs and `/command` invocations both — so text that arrives
 *  whole (a paste, a restored draft, an undo step, a rebuilt line) carries the
 *  same chips the typed path would have committed.
 *
 *  `composer.nonProse` ranges are painted here too: this is the one place text
 *  becomes DOM, so it is the one place a wrapper can be emitted — and the only
 *  one that knows which chip spans a wrapper must leave alone. The ranges are
 *  resolved against `rendered.text`, the string the DOM will serialize back to,
 *  so the render pass and the reconcile pass agree on what a provider saw. */
export function appendComposerContents(
  target: DocumentFragment | HTMLElement,
  text: string,
  options: SlashCommandScanOptions = {}
) {
  const rendered = composerRenderedText(text, chipSpans(text, options))
  const regions = composerNonProseSpans(target, rendered.text, rendered.chips)

  if (regions.length > 0) {
    if (target instanceof HTMLElement) {
      decoratedEditors.add(target)
    }

    // First wrapper painted on this app: from here on, an unload can leave one
    // behind with no pass to clean it up (see `sweepNonProseOnUnload`).
    sweepNonProseOnUnload()
  }

  let cursor = 0

  // Chips first on a tie: `sort` is stable, and a wrapper never swallows one.
  for (const span of [...rendered.chips, ...regions].sort((a, b) => a.start - b.start)) {
    // A `@` ref wins an overlap: a command token can't contain an `@`, so the
    // only way spans collide is a slash inside a quoted ref value
    // (`` @url:`a /clean` ``), which belongs to that value.
    if (span.start < cursor) {
      continue
    }

    appendTextWithBreaks(target, rendered.text.slice(cursor, span.start))
    target.append(span.node())
    cursor = span.end
  }

  appendTextWithBreaks(target, rendered.text.slice(cursor))
}

export function renderComposerContents(target: HTMLElement, text: string, options?: SlashCommandScanOptions) {
  target.replaceChildren()

  // Defaults to live editing, where a token ending the text is still being
  // typed (`/wor`) and must stay editable. Callers repainting inert text (a
  // restored draft, a sent message opened for edit) pass `trailingCommitted`.
  appendComposerContents(target, text, options)

  // The other writer that reshapes the editor root: painting a restored draft
  // in clears the marker, clearing back to '' sets it.
  markEditorEmptiness(target)
}

/** Caret range when the selection lives inside `editor`; else null. */
function composerSelectionRange(editor: HTMLElement) {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null

  if (!selection || !range || !editor.contains(range.commonAncestorContainer)) {
    return null
  }

  return { range, selection }
}

/** Serialized text from the editor's start up to (`container`, `offset`).
 *
 *  Chips are ATOMIC here: each contributes an object-replacement placeholder
 *  rather than leaking its label text, and a <br> contributes a newline. That
 *  makes a chip edge read as a token boundary, which is what both trigger
 *  detection and directive recognition need. */
export function serializeTextBefore(editor: HTMLElement, container: Node, offset: number): string {
  const probe = document.createRange()

  probe.selectNodeContents(editor)
  probe.setEnd(container, offset)

  const scratch = document.createElement('div')

  scratch.append(probe.cloneContents())

  for (const chip of scratch.querySelectorAll('[data-ref-text]')) {
    chip.replaceWith('\uFFFC')
  }

  for (const br of scratch.querySelectorAll('br')) {
    br.replaceWith('\n')
  }

  return scratch.textContent ?? ''
}

/** True when the insertion point starts a token — the editor's start, or after
 *  whitespace or a chip. `foo` + a pasted `/clean` is `foo/clean`, not a
 *  command; `foo ` + the same paste is. */
function atTokenBoundary(editor: HTMLElement, range: Range | null): boolean {
  // No caret means the insert lands at the end, so the question is about the
  // editor's last character either way.
  const before = range
    ? serializeTextBefore(editor, range.startContainer, range.startOffset)
    : serializeTextBefore(editor, editor, editor.childNodes.length)

  const last = before.slice(-1)

  return !last || /[\s\uFFFC]/.test(last)
}

/** Insert text at the caret (replacing any selection), with any directives in
 *  it landing as chips. Pastes use this instead of `execCommand('insertText')`
 *  — Chromium's editing pipeline is ~O(n²) on large multiline blobs.
 *
 *  The text arrives whole rather than typed, so a `/command` ending it is
 *  complete rather than half-written and chips like the rest.
 *
 *  `consumeBefore` characters immediately before the caret are swallowed by the
 *  insert. That's how a paste into an open `@url:` scope replaces the scope
 *  instead of stacking on it (`@url:@url:\`https://…\``). */
export function insertComposerContentsAtCaret(editor: HTMLElement, text: string, consumeBefore = 0) {
  const scoped = consumeBefore > 0 ? rangeBeforeCaret(editor, consumeBefore) : null

  if (scoped) {
    scoped.deleteContents()
    scoped.collapse(true)

    const selection = window.getSelection()

    selection?.removeAllRanges()
    selection?.addRange(scoped)
  }

  const hit = composerSelectionRange(editor)
  const fragment = document.createDocumentFragment()

  // Before measuring the boundary — a replaced selection puts the insertion
  // point where the selection started, not where it ended.
  if (hit) {
    hit.range.deleteContents()
  }

  appendComposerContents(fragment, text, {
    boundaryBefore: atTokenBoundary(editor, hit?.range ?? null),
    trailingCommitted: true
  })

  // A slash pill ending the insert gets the trailing space the typed commit
  // path appends, or the next full re-render reads it as a half-typed token
  // and demotes it. `@` refs need no marker — REF_RE re-chips them either way.
  if ((fragment.lastChild as HTMLElement | null)?.dataset?.slashKind) {
    fragment.append(document.createTextNode(' '))
  }

  const tail = fragment.lastChild

  if (hit) {
    hit.range.insertNode(fragment)
  } else {
    editor.append(fragment)
  }

  if (tail) {
    const caret = document.createRange()
    caret.setStartAfter(tail)
    caret.collapse(true)
    const selection = hit?.selection ?? window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(caret)
  }

  revealCaret(editor)
}

/** Range covering exactly `length` serialized characters immediately before a
 *  collapsed caret, spanning Chromium's split text nodes. Null when the caret
 *  isn't a collapsed selection in `editor`, or when a chip/<br>/block boundary
 *  interrupts before `length` characters are covered — a trigger token is
 *  always contiguous text, so anything else means "don't touch the DOM here".
 *
 *  This is what keeps chip insertion stable: Chromium fragments text nodes
 *  around contenteditable=false chips on every edit, so any commit path that
 *  demands the whole token inside ONE text node (the old check) degrades to a
 *  full re-render as soon as a chip exists anywhere in the line. */
export function rangeBeforeCaret(editor: HTMLElement, length: number): Range | null {
  const hit = composerSelectionRange(editor)

  if (!hit?.range.collapsed || length <= 0) {
    return null
  }

  let node: Node | null = hit.range.startContainer
  let offset = hit.range.startOffset

  // An element-positioned caret (common right after programmatic caret moves)
  // resolves to the end of the text node before it. A chip or <br> there means
  // no text token precedes the caret — bail rather than guess.
  if (node.nodeType !== Node.TEXT_NODE) {
    node = node.childNodes[offset - 1] ?? null

    if (node?.nodeType !== Node.TEXT_NODE) {
      return null
    }

    offset = (node.textContent || '').length
  }

  let startNode = node as Text
  let startOffset = offset
  let remaining = length

  while (remaining > 0) {
    if (startOffset >= remaining) {
      startOffset -= remaining
      remaining = 0

      break
    }

    remaining -= startOffset

    const prev: Node | null = startNode.previousSibling

    if (prev?.nodeType !== Node.TEXT_NODE) {
      return null
    }

    startNode = prev as Text
    startOffset = (prev.textContent || '').length
  }

  const range = document.createRange()

  range.setStart(startNode, startOffset)
  range.setEnd(hit.range.startContainer, hit.range.startOffset)

  return range
}

/** Swap the `length` characters immediately before a collapsed caret for
 *  `fragment`, leaving the caret after it. Returns whether it ran. Spans split
 *  text nodes (see rangeBeforeCaret) — a token typed around existing chips
 *  still commits in place instead of falling back to a full re-render. */
export function replaceBeforeCaret(editor: HTMLElement, length: number, fragment: DocumentFragment) {
  const range = rangeBeforeCaret(editor, length)

  if (!range) {
    return false
  }

  const tail = fragment.lastChild

  range.deleteContents()
  range.insertNode(fragment)

  if (tail) {
    range.setStartAfter(tail)
  }

  range.collapse(true)

  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)

  return true
}

/** Backspace at a collapsed caret immediately after a chip: delete the chip AND
 *  the single trailing space we auto-insert after it, atomically — so removing a
 *  directive never strands an orphaned space (the contenteditable-driven cleanup
 *  was unreliable). Returns whether it ran.
 *
 *  "Immediately after" has to be read through Chromium's litter. Committing a
 *  completion empties the typed token's text node rather than removing it, and
 *  `Range.insertNode` splits around the caret, so the chip routinely sits
 *  between zero-length text nodes. Reading those as content made the caret look
 *  like it was after plain text; the delete declined and Chromium's own
 *  backspace bounced between the leftovers instead of removing the chip. */
export function deleteChipBeforeCaret(editor: HTMLElement): boolean {
  const hit = composerSelectionRange(editor)

  if (!hit || !hit.range.collapsed) {
    return false
  }

  const { startContainer, startOffset } = hit.range
  let chip: ChildNode | null = null

  if (startContainer === editor) {
    chip = startOffset > 0 ? (editor.childNodes[startOffset - 1] ?? null) : null

    if (isEmptyTextNode(chip)) {
      chip = meaningfulPreviousSibling(chip)
    }
  } else if (startContainer.nodeType === Node.TEXT_NODE && startOffset === 0) {
    chip = meaningfulPreviousSibling(startContainer as ChildNode)
  }

  if (chip?.nodeType !== Node.ELEMENT_NODE || !(chip as HTMLElement).dataset.refText) {
    return false
  }

  const after = meaningfulNextSibling(chip)
  chip.remove()

  // Drop the auto-inserted trailing space; keep any real following text.
  if (after?.nodeType === Node.TEXT_NODE) {
    const text = after.textContent ?? ''

    if (text === ' ') {
      after.remove()
    } else if (text.startsWith(' ')) {
      after.textContent = text.slice(1)
    }
  }

  const caret = document.createRange()

  if (after?.isConnected) {
    caret.setStartBefore(after)
  } else {
    caret.selectNodeContents(editor)
    caret.collapse(false)
  }

  caret.collapse(true)
  hit.selection.removeAllRanges()
  hit.selection.addRange(caret)

  return true
}

/** Remove a non-collapsed selection in-editor. Skips collapsed carets so word/
 *  line delete (Opt/Cmd+Backspace) stays native. Returns whether anything ran. */
export function deleteSelectionInEditor(editor: HTMLElement) {
  const hit = composerSelectionRange(editor)

  if (!hit || hit.range.collapsed) {
    return false
  }

  hit.range.deleteContents()
  hit.range.collapse(true)
  hit.selection.removeAllRanges()
  hit.selection.addRange(hit.range)

  return true
}

/** Serialize a draft string into chip-HTML for the contenteditable surface. */
export function composerHtml(text: string) {
  let cursor = 0
  let html = ''

  REF_RE.lastIndex = 0

  for (const match of text.matchAll(REF_RE)) {
    const index = match.index ?? 0
    html += escapeHtml(text.slice(cursor, index)).replace(/\n/g, '<br>')
    html += refChipHtml(match[1] || 'file', match[2] || '')
    cursor = index + match[0].length
  }

  return html + escapeHtml(text.slice(cursor)).replace(/\n/g, '<br>')
}

/** Walk a DOM subtree back to the plain `@kind:value` text it represents. */
export function composerPlainText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent || ''
  }

  if (node.nodeType !== Node.ELEMENT_NODE) {
    return ''
  }

  const el = node as HTMLElement

  if (el.dataset.refText) {
    return el.dataset.refText
  }

  // An editor holding nothing but the placeholder <br> is EMPTY. That <br> is
  // scaffolding normalizeComposerEditorDom adds so the contenteditable keeps
  // its height — not a line the user typed. Reading it as "\n" is how a
  // just-cleared composer stayed non-empty: the newline got stashed as the
  // session's draft and painted back on return.
  if (el.dataset.slot === RICH_INPUT_SLOT && el.childNodes.length === 1 && el.firstChild?.nodeName === 'BR') {
    return ''
  }

  if (el.tagName === 'BR') {
    return '\n'
  }

  const text = Array.from(node.childNodes).map(composerPlainText).join('')
  const block = el.tagName === 'DIV' || el.tagName === 'P'

  return block && text && el.dataset.slot !== RICH_INPUT_SLOT ? `${text}\n` : text
}

export function placeCaretEnd(element: HTMLElement) {
  // A repaint can land on an editor React has already unmounted (the chat
  // bar toggles with the thread's loading gate). Selecting into a detached
  // node throws `addRange(): The given range isn't in document` from inside
  // the commit phase, and React re-renders in a loop on it (#117285).
  if (!element.isConnected) {
    return
  }

  const range = document.createRange()
  const selection = window.getSelection()

  range.selectNodeContents(element)
  range.collapse(false)
  selection?.removeAllRanges()
  selection?.addRange(range)
  revealCaret(element)
}

/** The editor `scrollTop` that brings a caret spanning `caret.top..bottom`
 *  (client px) inside the visible band `viewport`, or null when it's already
 *  visible. Moves the nearest edge only, like `block: 'nearest'`. */
export function caretRevealScrollTop(
  caret: { bottom: number; top: number },
  viewport: { bottom: number; top: number },
  scrollTop: number
): number | null {
  if (caret.top < viewport.top) {
    return scrollTop - (viewport.top - caret.top)
  }

  if (caret.bottom > viewport.bottom) {
    return scrollTop + (caret.bottom - viewport.bottom)
  }

  return null
}

function caretClientRect(range: Range): { bottom: number; top: number } | null {
  const end = range.cloneRange()
  end.collapse(false)

  const rects = typeof end.getClientRects === 'function' ? end.getClientRects() : null

  if (rects?.length) {
    return rects[rects.length - 1]
  }

  // Chromium gives a caret between elements (after a chip or <br>, or at the
  // editor's end, which is where inserts leave it) no box. Measure a probe
  // there instead. A text-node caret always has a box, so the probe never
  // splits text.
  if (end.startContainer.nodeType === Node.TEXT_NODE) {
    return null
  }

  const probe = document.createElement('span')
  probe.textContent = '\u200b'
  end.insertNode(probe)

  const rect = probe.getBoundingClientRect()

  probe.remove()

  return rect
}

const pendingReveals = new WeakSet<HTMLElement>()

/** Scroll the editor so its caret is visible, on the next frame (one per
 *  frame across a burst of inserts). Chromium only does this itself for native
 *  edit commands; programmatic inserts (paste, voice transcripts, repaints)
 *  leave the caret wherever the old scroll position puts it (#79806). Local to
 *  the editor: `scrollIntoView` would also move the transcript. */
export function revealCaret(editor: HTMLElement) {
  if (pendingReveals.has(editor)) {
    return
  }

  pendingReveals.add(editor)
  window.requestAnimationFrame(() => {
    pendingReveals.delete(editor)

    const selection = window.getSelection()
    const range = selection?.rangeCount ? selection.getRangeAt(0) : null

    if (!editor.isConnected || !range || !editor.contains(range.endContainer)) {
      return
    }

    const caret = caretClientRect(range)

    if (!caret) {
      return
    }

    const top = editor.getBoundingClientRect().top + editor.clientTop
    const next = caretRevealScrollTop(caret, { bottom: top + editor.clientHeight, top }, editor.scrollTop)

    if (next !== null) {
      editor.scrollTop = next
    }
  })
}

/** The caret's offset in `composerPlainText` coordinates, so it can be restored
 *  after the editor is re-rendered from text (undo/redo). A chip counts as its
 *  whole `@kind:value` text — the same units the snapshot measures. */
export function caretOffsetInEditor(editor: HTMLElement): number {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null

  if (!range || !editor.contains(range.commonAncestorContainer)) {
    return composerPlainText(editor).length
  }

  const before = range.cloneRange()
  before.selectNodeContents(editor)
  before.setEnd(range.startContainer, range.startOffset)

  // The scratch container must carry the editor's slot marker: composerPlainText
  // appends a trailing "\n" to any other block element, which would inflate
  // every offset by one and land the restored caret a character late.
  const container = document.createElement('div')
  container.dataset.slot = RICH_INPUT_SLOT
  container.append(before.cloneContents())

  return composerPlainText(container).length
}

/** Place the caret `offset` characters into the editor, in the same
 *  `composerPlainText` coordinates `caretOffsetInEditor` reports. Lands after a
 *  chip it would otherwise split, since a chip is a single atomic unit. */
export function placeCaretAtOffset(editor: HTMLElement, offset: number) {
  const selection = window.getSelection()

  if (!selection) {
    return
  }

  let remaining = offset

  const walk = (node: Node): Range | null => {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === Node.TEXT_NODE) {
        const length = (child.textContent || '').length

        if (remaining <= length) {
          const range = document.createRange()
          range.setStart(child, remaining)
          range.collapse(true)

          return range
        }

        remaining -= length

        continue
      }

      if (child.nodeType !== Node.ELEMENT_NODE) {
        continue
      }

      const el = child as HTMLElement

      // Chips and <br> are atomic: consume their serialized length whole.
      if (el.dataset.refText || el.tagName === 'BR') {
        const length = el.dataset.refText ? el.dataset.refText.length : 1

        if (remaining < length) {
          const range = document.createRange()
          range.setStartBefore(el)
          range.collapse(true)

          return range
        }

        remaining -= length

        continue
      }

      const hit = walk(el)

      if (hit) {
        return hit
      }
    }

    return null
  }

  const range = editor.isConnected ? walk(editor) : null

  if (range?.startContainer.isConnected) {
    selection.removeAllRanges()
    selection.addRange(range)

    return
  }

  placeCaretEnd(editor)
}

/** Snapshot only when cleanup actually removes the focused caret's container;
 * cloning the draft on every input flush makes ordinary typing needlessly costly. */
function removeComposerJunk(editor: HTMLElement, node: ChildNode) {
  const selected = composerCollapsedSelectionContainer(editor)

  const offset =
    document.activeElement === editor && selected && node.contains(selected) ? caretOffsetInEditor(editor) : null

  node.remove()

  if (offset !== null) {
    placeCaretAtOffset(editor, Math.min(offset, composerPlainText(editor).length))
  }
}

/** Nothing but a break / whitespace (recursively) — i.e. no real text or chip. */
function isBlankNode(node: ChildNode | null): boolean {
  if (!node) {
    return false
  }

  if (node.nodeName === 'BR') {
    return true
  }

  if (node.nodeType === Node.TEXT_NODE) {
    return !(node.textContent || '').trim()
  }

  if (node.nodeType === Node.ELEMENT_NODE) {
    const el = node as HTMLElement

    return !el.dataset.refText && Array.from(el.childNodes).every(isBlankNode)
  }

  return false
}

/** Swap a non-prose wrapper for its children in place, keeping the selection.
 *
 *  Moving nodes out resets Chromium's selection, so this saves the DOM
 *  endpoints (and direction) first and re-applies them after — the same reason
 *  the block unwrap below does. Every endpoint the unwrap can move is mapped,
 *  not only the ones INSIDE the wrapper: the shape `placeCaretEnd`, Ctrl+A and
 *  a click below the last line leave is a caret anchored on the PARENT, right
 *  after the wrapper. Re-applying that endpoint's stale child index put the
 *  caret a few characters into the unwrapped text, and not restoring it at all
 *  (what this did whenever neither endpoint was inside the wrapper) let
 *  Chromium collapse it to the editor's start. */
function unwrapNonProse(editor: HTMLElement, wrapper: HTMLElement) {
  const selection = editor.ownerDocument.getSelection()
  const anchorNode = selection?.anchorNode ?? null
  const focusNode = selection?.focusNode ?? null
  const anchorOffset = selection?.anchorOffset ?? 0
  const focusOffset = selection?.focusOffset ?? 0
  const parent = wrapper.parentNode
  const index = parent ? Array.from(parent.childNodes).indexOf(wrapper) : -1
  const moved = wrapper.childNodes.length

  // Map one endpoint across the unwrap:
  //  - ON the wrapper: its offset indexes the children, so it becomes
  //    `index + offset` in the parent (the wrapper's index is where its content
  //    now begins).
  //  - ON THE PARENT after the wrapper: the parent loses one child and gains
  //    `moved`, so every later index shifts by `moved - 1`.
  //  - anywhere else (a text node inside the wrapper, elsewhere in the editor,
  //    outside it): unchanged — the children are MOVED, not re-created, so a
  //    text node keeps its identity and its offsets.
  const map = (node: Node | null, offset: number): { moved: boolean; node: Node | null; offset: number } => {
    if (!node || !parent) {
      return { moved: false, node, offset }
    }

    if (node === wrapper) {
      return { moved: true, node: parent, offset: index + Math.min(offset, moved) }
    }

    if (node === parent && offset > index) {
      return { moved: true, node: parent, offset: offset + moved - 1 }
    }

    return { moved: wrapper.contains(node), node, offset }
  }

  const anchor = map(anchorNode, anchorOffset)
  const focus = map(focusNode, focusOffset)

  wrapper.replaceWith(...Array.from(wrapper.childNodes))

  // Nothing of this selection sat in or beside the wrapper: the browser's own
  // live range already points where it should, so leave it alone.
  if (!anchor.moved && !focus.moved) {
    return
  }

  if (!editor.isConnected || !anchor.node?.isConnected || !focus.node?.isConnected) {
    return
  }

  selection?.setBaseAndExtent(anchor.node, anchor.offset, focus.node, focus.offset)
}

/** One pass over the editor: its serialized text (the string
 *  `composerPlainText` returns), the text nodes carrying it, the chip spans, and
 *  every non-prose wrapper with its offsets — at ANY depth, since Chromium
 *  block-wraps a line routinely. */
interface ComposerTextProbe {
  chips: Array<{ end: number; start: number }>
  nodes: Array<{ end: number; node: Text; start: number }>
  text: string
  wrappers: Array<{ element: HTMLElement; end: number; start: number }>
}

/** Serialize the editor the way `composerPlainText` does, recording where every
 *  piece sits. `null` when the walk cannot reproduce that string exactly (an
 *  exotic tree shape this pass does not understand): painting against offsets
 *  that do not line up would move the draft, so the caller does nothing. */
function probeComposerText(editor: HTMLElement): ComposerTextProbe | null {
  const chips: ComposerTextProbe['chips'] = []
  const nodes: ComposerTextProbe['nodes'] = []
  const wrappers: ComposerTextProbe['wrappers'] = []
  let text = ''

  const walk = (node: Node) => {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeType === Node.TEXT_NODE) {
        const value = child.textContent ?? ''
        const start = text.length

        text += value

        if (value) {
          nodes.push({ end: text.length, node: child as Text, start })
        }

        continue
      }

      if (child.nodeType !== Node.ELEMENT_NODE) {
        continue
      }

      const element = child as HTMLElement

      // A chip and a <br> are atomic, exactly as `composerPlainText` reads them.
      if (element.dataset.refText !== undefined) {
        const start = text.length

        text += element.dataset.refText
        chips.push({ end: text.length, start })

        continue
      }

      if (element.tagName === 'BR') {
        text += '\n'

        continue
      }

      if (element.hasAttribute(COMPOSER_NON_PROSE_ATTR)) {
        const start = text.length

        walk(element)
        wrappers.push({ element, end: text.length, start })

        continue
      }

      // A block contributes its own trailing newline — the rule
      // `composerPlainText` uses, and the reason a wrapper Chromium
      // block-wrapped is measured where it actually sits.
      const block = element.tagName === 'DIV' || element.tagName === 'P'
      const from = text.length

      walk(element)

      if (block && text.length > from && element.dataset.slot !== RICH_INPUT_SLOT) {
        text += '\n'
      }
    }
  }

  walk(editor)

  return text === composerPlainText(editor) ? { chips, nodes, text, wrappers } : null
}

/** Wrap `[start, end)` of `probe.text` in a fresh non-prose element.
 *
 *  The resolver splits ranges at every `\n` and clips them around chips, so a
 *  range the reconciler paints always lies inside text — never across a `<br>`,
 *  a chip or a block edge — and both ends resolve to a point inside a text
 *  node. Returns whether it painted. */
function wrapNonProse(probe: ComposerTextProbe, start: number, end: number, kind: ComposerNonProseKind) {
  const pointAt = (offset: number): [Text, number] | null => {
    const hit = probe.nodes.find(entry => entry.start <= offset && offset <= entry.end)

    return hit ? [hit.node, offset - hit.start] : null
  }

  const from = pointAt(start)
  const to = pointAt(end)

  if (!from || !to) {
    return false
  }

  const range = document.createRange()

  range.setStart(from[0], from[1])
  range.setEnd(to[0], to[1])

  // Never extract a range the DOM does not read as that slice: a wrapper around
  // the wrong span moves the user's text into (and out of) it.
  if (range.toString() !== probe.text.slice(start, end)) {
    return false
  }

  const wrapper = composerNonProseElement(probe.text.slice(start, end), kind)

  wrapper.replaceChildren(range.extractContents())
  range.insertNode(wrapper)

  return true
}

/** Own the non-prose wrappers: a `[data-composer-nonprose]` element is a node
 *  core emits, so no rule above may drop it as junk — but it is only valid
 *  while a resolved range still covers it. Re-resolve for the text this pass
 *  leaves behind and make the DOM match that answer: unwrap the stale ones,
 *  re-paint the ones the provider asks for again, clip or extend the ones whose
 *  range moved. A plugin's code span therefore keeps its styling on the very
 *  next keystroke instead of going plain until something re-renders the whole
 *  draft.
 *
 *  Two kinds of stale wrapper: one no range covers any more (the user edited
 *  the span out of shape, the provider changed its answer, or the plugin
 *  unloaded), and one core would never emit — holding a chip, a `<br>` or a
 *  block, which a wrapper cannot legally straddle.
 *
 *  When the answer still describes what is on screen — the common case, every
 *  keystroke — nothing is touched at all: no node moves, so the caret does not
 *  either. */
function reconcileNonProse(editor: HTMLElement) {
  // An IME composition defers the whole pass (see `beginComposerComposition`):
  // the DOM holds preedit text that is not the draft yet.
  if (composingEditors.has(editor)) {
    return
  }

  // Nothing to own — the ordinary case, and the reconciler runs on every
  // keystroke of every composer. With no provider registered and no wrapper on
  // screen it must cost two lookups and nothing else: no serialization, no
  // walk. (`decoratedEditors` is what keeps the unload case working for an
  // editor the sweep could not reach — one that is not in the document yet.)
  if (!decoratedEditors.has(editor) && !composerNonProseEnabled(editor)) {
    return
  }

  const probe = probeComposerText(editor)

  if (!probe) {
    return
  }

  const desired = clipComposerNonProseRanges(composerNonProseSurfaceRanges(editor, probe.text), probe.chips)

  const matches =
    probe.wrappers.length === desired.length &&
    probe.wrappers.every((wrapper, index) => {
      const range = desired[index]

      return (
        wrapper.element.children.length === 0 &&
        wrapper.start === range.start &&
        wrapper.end === range.end &&
        (wrapper.element.dataset.composerNonprose ?? 'code') === range.kind
      )
    })

  if (matches) {
    if (probe.wrappers.length === 0) {
      decoratedEditors.delete(editor)
    }

    return
  }

  // Only a re-paint that actually wraps something can move the caret, and only
  // a collapsed one is worth carrying across it.
  const selection = editor.ownerDocument.getSelection()
  const caret = selection?.isCollapsed && editor.contains(selection.anchorNode) ? caretOffsetInEditor(editor) : null

  // Unwrap first: a stale wrapper still holds text a NEW range may cover (a
  // span that grew over the wrapper's edge), and wrapping around it would nest
  // one wrapper inside the other.
  for (const wrapper of probe.wrappers) {
    unwrapNonProse(editor, wrapper.element)
  }

  let current = probeComposerText(editor)
  let painted = false

  for (const range of desired) {
    if (!current || current.text !== probe.text) {
      break
    }

    if (wrapNonProse(current, range.start, range.end, range.kind)) {
      painted = true
      // Wrapping splits the text nodes the offsets index into, so the map is
      // re-read between ranges. Only ranges that actually paint pay for it.
      current = probeComposerText(editor)
    }
  }

  if (painted && caret !== null) {
    placeCaretAtOffset(editor, Math.min(caret, composerPlainText(editor).length))
  }

  // Keep the fast-path flag honest: with nothing left to paint, the next pass
  // can bail on two lookups instead of a walk.
  if (desired.length > 0) {
    decoratedEditors.add(editor)
  } else {
    decoratedEditors.delete(editor)
  }
}

/** Unwrap every wrapper the moment the last `composer.nonProse` provider goes
 *  away.
 *
 *  The reconciler would get there on the next pass, but an unload schedules no
 *  pass — the composer only normalizes on an edit — so the wrapper would sit
 *  there until the user typed. Subscribed lazily, on the first pass that
 *  actually paints: an app with no provider pays for no listener. */
let nonProseSweepInstalled = false

function sweepNonProseOnUnload() {
  if (nonProseSweepInstalled) {
    return
  }

  nonProseSweepInstalled = true

  registry.subscribeArea(COMPOSER_AREAS.nonProse, () => {
    if (registry.getArea(COMPOSER_AREAS.nonProse).length > 0) {
      return
    }

    // Nothing can cover a wrapper any more, so all of them are stale.
    document.querySelectorAll<HTMLElement>(`[${COMPOSER_NON_PROSE_ATTR}]`).forEach(wrapper => {
      const editor = wrapper.closest<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)

      if (editor) {
        unwrapNonProse(editor, wrapper)
        decoratedEditors.delete(editor)
      }
    })
  })
}

/** Drop contenteditable junk that serializes as `\n` and falsely expands the
 *  composer. Editing around a contenteditable=false chip makes Chromium wrap the
 *  remainder in stray block <div>s / trailing <br>s — none of which our own
 *  rendering emits (we use text nodes + <br> + chips). Real <br> line breaks
 *  (Shift+Enter, which sit after actual text) are preserved. */
export function normalizeComposerEditorDom(editor: HTMLElement) {
  const selectedContainer = composerCollapsedSelectionContainer(editor)

  // Chromium's zero-length text nodes first: every check below reads siblings,
  // and litter between them makes a chip look like it has text either side.
  for (const child of Array.from(editor.childNodes)) {
    if (isEmptyTextNode(child) && child !== selectedContainer) {
      child.remove()
    }
  }

  // A trailing block wrapper holding only a break/whitespace is the phantom
  // "new line" Chromium adds after a chip on backspace — drop it.
  const tailBlock = editor.lastChild as HTMLElement | null

  if (
    tailBlock?.nodeType === Node.ELEMENT_NODE &&
    (tailBlock.tagName === 'DIV' || tailBlock.tagName === 'P') &&
    isBlankNode(tailBlock)
  ) {
    removeComposerJunk(editor, tailBlock)
  }

  // Unwrap a lone block wrapper back to inline content.
  if (editor.childNodes.length === 1 && editor.firstChild?.nodeType === Node.ELEMENT_NODE) {
    const wrapper = editor.firstChild as HTMLElement

    if ((wrapper.tagName === 'DIV' || wrapper.tagName === 'P') && wrapper.dataset.slot !== RICH_INPUT_SLOT) {
      // Moving the text nodes out resets Chromium's selection to the editor's
      // start. Keep DOM endpoints (and direction), not serialized text offsets:
      // chips are atomic and the wrapper's trailing newline is being removed.
      const selection = editor.ownerDocument.getSelection()
      const anchorNode = selection?.anchorNode
      const focusNode = selection?.focusNode
      const anchorOffset = selection?.anchorOffset ?? 0
      const focusOffset = selection?.focusOffset ?? 0
      const ownsSelection = anchorNode && focusNode && wrapper.contains(anchorNode) && wrapper.contains(focusNode)

      editor.replaceChildren(...Array.from(wrapper.childNodes))

      if (ownsSelection && editor.isConnected) {
        selection?.setBaseAndExtent(
          anchorNode === wrapper ? editor : anchorNode,
          anchorOffset,
          focusNode === wrapper ? editor : focusNode,
          focusOffset
        )
      }
    }
  }

  // A trailing <br> right after a chip / only whitespace is a phantom line.
  const last = editor.lastChild

  if (last?.nodeName === 'BR') {
    let prev: ChildNode | null = last.previousSibling

    while (prev?.nodeType === Node.TEXT_NODE && !(prev.textContent || '').trim()) {
      prev = prev.previousSibling
    }

    if (!prev || (prev as HTMLElement).dataset?.refText) {
      removeComposerJunk(editor, last)
    }
  }

  // A `[data-composer-nonprose]` wrapper is a node core emits, so every rule
  // above leaves it alone — this is the one pass that owns it. Runs after the
  // block rules so it measures the text this normalization actually leaves.
  reconcileNonProse(editor)

  // ContentEditable elements with no children can visually collapse to
  // near-zero height in some browsers (especially Chromium), causing the
  // composer to appear as a tiny dot/pixel. Ensure there's always at least
  // one <br> so the element maintains intrinsic height. The CSS min-height
  // is a belt; the <br> is suspenders — together they prevent the shrink.
  // That break is also why emptiness has to be marked, not inferred.
  markEditorEmptiness(editor)

  if (editor.childNodes.length === 0) {
    editor.appendChild(document.createElement('br'))
  }
}
