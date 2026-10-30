/**
 * The composer's context-menu contribution area, app side.
 *
 * Plugins contribute DATA (`composer.contextMenu` → `ComposerContextMenuProvider`);
 * the app owns everything else: it decides the click landed on a composer,
 * captures the word and selection there, hands a provider only those two
 * strings, orders and caps the rows, renders them with its own menu item, and
 * runs the chosen row after the menu closed. The live `Range` and the editor
 * element never leave this module — the only edit a plugin can make through it
 * is replacing the exact word the user right-clicked, once.
 */

import type {
  ComposerContextMenuAction,
  ComposerContextMenuContext,
  ComposerContextMenuItem,
  ComposerContextMenuProvider
} from '@/app/chat/composer/contrib'
import { recordComposerUndoPoint } from '@/app/chat/composer/hooks/use-composer-undo'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import type { Contribution } from '@/contrib/types'

/** Rows one plugin may put in the menu — the app's own spell-check section is
 *  five suggestions plus "Add to dictionary". */
export const COMPOSER_CONTEXT_MENU_ITEM_CAP = 6

const LABEL_MAX = 80
const REPLACEMENT_MAX = 200
const SELECTION_MAX = 2000
const WORD_MAX = 64

// C0/C1 controls and the bidi overrides/isolates a label could use to dress up
// as a different row.
const UNSAFE_LABEL_CHARS = /[\p{Cc}\u202a-\u202e\u2066-\u2069]/gu
const CONTROL_CHARS = /\p{Cc}/u
const WORD_CHAR = /[\p{L}\p{M}\p{N}'\u2019-]/u
const WORD_EDGE = /['\u2019-]/

/** The app-held half of a composer right-click. `context` is all a provider
 *  ever sees; `editor` and `wordRange` stay here. */
export interface ComposerMenuTarget {
  context: ComposerContextMenuContext
  editor: HTMLElement
  wordRange: Range | null
}

/** One rendered row: the app's label and a runner that builds the action. */
export interface ComposerMenuRow {
  disabled: boolean
  key: string
  label: string
  run: () => void
}

/** One whole character (code point) at / before a UTF-16 offset, so letters
 *  outside the BMP (`𠮷`) count as one word character, not two halves. */
function charAt(text: string, index: number): string {
  const point = text.codePointAt(index)

  return point === undefined ? '' : String.fromCodePoint(point)
}

function charBefore(text: string, index: number): string {
  const low = text.charCodeAt(index - 1)

  return text.slice(low >= 0xdc00 && low <= 0xdfff && index >= 2 ? index - 2 : Math.max(0, index - 1), index)
}

const isWordChar = (char: string) => char !== '' && WORD_CHAR.test(char)

/** Word bounds around `caret` in `text`. A caret sits on a boundary between
 *  characters, so step to the nearest word character first; otherwise the
 *  word found depends on which side of a gap the caret happened to land.
 *  Quotes and dashes join a word's inside ("don't", "re-run"), never its edges. */
function wordBounds(text: string, caret: number): null | { end: number; start: number } {
  let start = caret

  if (!isWordChar(charAt(text, start))) {
    const before = charBefore(text, start)

    if (!isWordChar(before)) {
      return null
    }

    start -= before.length
  }

  let end = start

  for (let char = charBefore(text, start); start > 0 && isWordChar(char); char = charBefore(text, start)) {
    start -= char.length
  }

  for (let char = charAt(text, end); end < text.length && isWordChar(char); char = charAt(text, end)) {
    end += char.length
  }

  while (start < end && WORD_EDGE.test(charAt(text, start))) {
    start++
  }

  while (end > start && WORD_EDGE.test(charBefore(text, end))) {
    end--
  }

  return end > start && end - start <= WORD_MAX ? { end, start } : null
}

/** The text nodes directly beside `node` with no element between them: a paste
 *  lands as its own node, so one word can span several. Chips and line breaks
 *  are elements, so a word never crosses them. */
function textRun(node: Text): Text[] {
  const run = [node]

  for (let prev = node.previousSibling; prev instanceof Text; prev = prev.previousSibling) {
    run.unshift(prev)
  }

  for (let next = node.nextSibling; next instanceof Text; next = next.nextSibling) {
    run.push(next)
  }

  return run
}

/** Map an offset in the joined run back to a (node, offset) boundary point.
 *  `end` boundaries prefer the earlier node at a seam, `start` the later. */
function pointIn(run: Text[], offset: number, edge: 'end' | 'start'): [Text, number] {
  let rest = offset

  for (const [index, node] of run.entries()) {
    const fits = edge === 'end' ? rest <= node.length : rest < node.length

    if (fits || index === run.length - 1) {
      return [node, Math.min(rest, node.length)]
    }

    rest -= node.length
  }

  return [run[0], 0]
}

/** The word around a collapsed caret, or at a selection's start, as a live
 *  Range over the composer's own text nodes. */
function wordRangeAt(selected: Range): Range | null {
  const node = selected.startContainer

  if (!(node instanceof Text)) {
    return null
  }

  const run = textRun(node)
  const before = run.slice(0, run.indexOf(node)).reduce((sum, item) => sum + item.length, 0)
  const bounds = wordBounds(run.map(item => item.data).join(''), before + Math.min(selected.startOffset, node.length))

  if (!bounds) {
    return null
  }

  const range = document.createRange()

  range.setStart(...pointIn(run, bounds.start, 'start'))
  range.setEnd(...pointIn(run, bounds.end, 'end'))

  return range
}

/** Capture the right-click when (and only when) it landed in a composer —
 *  the main input or the message-edit composer, both marked by the rich
 *  input slot. Any other editable (settings fields, API keys, dialogs)
 *  resolves to null and contributes nothing. */
export function captureComposerMenuTarget(editable: HTMLElement | null): ComposerMenuTarget | null {
  if (!editable || editable.dataset.slot !== RICH_INPUT_SLOT) {
    return null
  }

  const selection = window.getSelection()
  const selected = selection?.rangeCount ? selection.getRangeAt(0) : null

  const inside = Boolean(
    selected && editable.contains(selected.startContainer) && editable.contains(selected.endContainer)
  )

  const wordRange = inside && selected ? wordRangeAt(selected) : null

  return {
    context: Object.freeze({
      selection: inside && selected ? selected.toString().slice(0, SELECTION_MAX) : '',
      word: wordRange ? wordRange.toString() : null
    }),
    editor: editable,
    wordRange
  }
}

function safeLabel(raw: unknown): null | string {
  if (typeof raw !== 'string') {
    return null
  }

  const label = raw.replace(UNSAFE_LABEL_CHARS, ' ').replace(/\s+/g, ' ').trim()

  return label ? (label.length > LABEL_MAX ? `${label.slice(0, LABEL_MAX - 1)}…` : label) : null
}

/** The action an item's `onSelect` receives: the same context plus a replace
 *  of the right-clicked word, refused once the composer no longer holds that
 *  word in that place — which a first replace itself ends, so it works once. */
function composerMenuAction(target: ComposerMenuTarget): ComposerContextMenuAction {
  const { context, editor, wordRange } = target

  const replaceWord = (text: string): boolean => {
    const stale =
      !wordRange ||
      !editor.isConnected ||
      !editor.contains(wordRange.startContainer) ||
      wordRange.toString() !== context.word

    if (stale || typeof text !== 'string' || !text || text.length > REPLACEMENT_MAX || CONTROL_CHARS.test(text)) {
      return false
    }

    editor.focus()

    const selection = window.getSelection()

    selection?.removeAllRanges()
    selection?.addRange(wordRange)

    // The composer owns ⌘Z, and Chromium's editing command below never reaches
    // its stack on its own: bank the pre-edit state so the swap is undoable.
    recordComposerUndoPoint(editor)

    // Chromium's own editing command, so the composer's input handling sees a
    // normal edit (draft sync, caret after the new word).
    return document.execCommand('insertText', false, text)
  }

  return Object.freeze({ ...context, replaceWord })
}

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

function rowFor(item: unknown, key: string, target: ComposerMenuTarget, source: string): ComposerMenuRow | null {
  const candidate = item as Partial<ComposerContextMenuItem> | null
  const label = safeLabel(candidate?.label)
  const onSelect = candidate?.onSelect

  if (!label || typeof onSelect !== 'function') {
    return null
  }

  return {
    disabled: Boolean(candidate?.disabled),
    key,
    label,
    run: () => {
      try {
        onSelect(composerMenuAction(target))
      } catch (error) {
        console.warn(`[composer.contextMenu] ${source} item failed`, error)
      }
    }
  }
}

/**
 * Resolve the area's contributions into menu sections: one per plugin, plugins
 * in code-point order of their source id (then contribution id) — never load
 * order or a plugin-chosen `order` — and at most
 * `COMPOSER_CONTEXT_MENU_ITEM_CAP` rows per plugin across all its
 * contributions. A provider that throws or returns garbage contributes nothing
 * and leaves every other row standing.
 */
export function resolveComposerMenuSections(
  contributions: readonly Contribution[],
  target: ComposerMenuTarget
): ComposerMenuRow[][] {
  const bySource = new Map<string, ComposerMenuRow[]>()

  const ordered = [...contributions].sort(
    (a, b) => byCodePoint(a.source ?? 'core', b.source ?? 'core') || byCodePoint(a.id, b.id)
  )

  for (const contribution of ordered) {
    const source = contribution.source ?? 'core'
    const rows = bySource.get(source) ?? []
    const provider = contribution.data as Partial<ComposerContextMenuProvider> | undefined

    bySource.set(source, rows)

    if (rows.length >= COMPOSER_CONTEXT_MENU_ITEM_CAP || typeof provider?.items !== 'function') {
      continue
    }

    let items: unknown

    try {
      items = provider.items(target.context)
    } catch (error) {
      console.warn(`[composer.contextMenu] ${source} provider failed`, error)

      continue
    }

    for (const [index, item] of (Array.isArray(items) ? items : []).entries()) {
      const row =
        rows.length < COMPOSER_CONTEXT_MENU_ITEM_CAP && rowFor(item, `${contribution.id}:${index}`, target, source)

      if (row) {
        rows.push(row)
      }
    }
  }

  return [...bySource.values()].filter(rows => rows.length > 0)
}
