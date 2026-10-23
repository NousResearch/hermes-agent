import { atom } from 'nanostores'

/**
 * The area a plugin contributes CONTEXT-MENU ROWS to.
 *
 * The app's own context menu (Cut / Copy / Paste / Select All) is a React
 * tree built in `app-context-menu.tsx`. Until now it had no extension point:
 * a plugin that wanted a row had to reach into the app's DOM and inject one,
 * which fights the menu's own focus trap and its keyboard handling.
 *
 * A contribution here is DATA, not a React node: the menu renders the rows
 * with its own `Item` component, so a plugin's row is visually and
 * behaviourally indistinguishable from a built-in one.
 *
 * `data` is a {@link ContextMenuContribution}: the rows to show, and the
 * context they were resolved against.
 */
export const CONTEXT_MENU_AREA = 'contextMenu'

/**
 * What a plugin is told about the menu it is contributing to.
 *
 * `word` is the word under the caret when the menu opened over an editable,
 * or null when the click was not on a word. `selectedText` is the current
 * selection. Both are captured at OPEN time, before the menu's focus trap
 * takes focus away from the field.
 */
export interface ContextMenuContext {
  /** The editable the menu was opened over, when there was one. */
  editable: boolean
  /** The word under the caret, or null when the caret was not on a word. */
  word: string | null
  /** The text selected in the field, or an empty string. */
  selectedText: string
}

/**
 * One row a plugin contributes to the context menu.
 *
 * `onSelect` runs when the user picks the row. The menu closes first, then
 * the app restores focus to the field, and only then does the action run --
 * the same order the built-in edit verbs use, so a plugin row can act on the
 * field the user right-clicked.
 *
 * The order matters and is not incidental. Radix fires a menu item's
 * `onSelect` before it closes the menu content, and the content is a focus
 * trap that prevents focus restoration on close. A handler that ran directly
 * would therefore run with the menu still mounted and focus still inside it,
 * so a live selection read -- or an edit command aimed at the focused
 * element -- would hit the menu rather than the field. Closing first and
 * refocusing the field on the next frame puts the action back on the field.
 *
 * The context the row was resolved against (`editable`, `word`,
 * `selectedText`) is captured when the menu OPENS, so it is still valid by
 * the time the action runs.
 */
export interface ContextMenuRow {
  /** The text shown in the row. */
  label: string
  /** A codicon shown before the label, e.g. `'edit'` or `'book'`. */
  icon?: string
  /** Draws the row greyed out and unclickable. */
  disabled?: boolean
  /** Runs when the row is chosen. */
  onSelect: () => void
}

/**
 * The `data` payload of a contribution registered in {@link CONTEXT_MENU_AREA}.
 */
export interface ContextMenuContribution {
  /**
   * The rows to add, resolved for the menu that is open. Returning an empty
   * array contributes nothing.
   */
  rows: (context: ContextMenuContext) => ContextMenuRow[]
  /**
   * Where the rows go. `'top'` puts them at the top of the menu, above the
   * app's own rows; `'bottom'` puts them after the last section. Defaults to
   * `'top'`.
   */
  position?: 'top' | 'bottom'
  /**
   * Sort key among the other contributors in the same position. Ascending;
   * ties keep registration order. Defaults to 0.
   */
  order?: number
}

/** The open menu's context, or null when no menu is open. */
export const $contextMenuContext = atom<ContextMenuContext | null>(null)

/**
 * Publish the context for the menu that just opened, so contributors can
 * resolve their rows against it. Cleared when the menu closes.
 */
export function setContextMenuContext(context: ContextMenuContext | null): void {
  $contextMenuContext.set(context)
}

/**
 * The word under the caret inside an editable, or null when the caret is not
 * on one.
 *
 * The editable's text lives in a text node inside the contenteditable, so the
 * selection's start container is that node and the offsets index into its
 * value. A collapsed caret sits on a BOUNDARY between two words, so the
 * probe steps to the nearest word character first -- otherwise the word found
 * depends on which side of the gap the caret happened to land.
 */
export function wordUnderCaret(editable: HTMLElement | null | undefined): string | null {
  if (!editable || typeof document === 'undefined') {
    return null
  }

  const selection = document.getSelection()

  if (!selection || !selection.rangeCount) {
    return null
  }

  const range = selection.getRangeAt(0)
  const node = range.startContainer

  if (node.nodeType !== Node.TEXT_NODE) {
    return null
  }

  const text = node.nodeValue || ''
  const isWord = (ch: string) => /[\p{L}'\u2019-]/u.test(ch)
  // The selection can outlive the node it points at (the text changed under
  // it), so the offset may be past the end. Clamp rather than read past it.
  let start = Math.min(Math.max(range.startOffset, 0), text.length)
  let end = start

  if (!isWord(text[start] || '')) {
    let probe = start

    while (probe > 0 && !isWord(text[probe - 1])) {
      probe--
    }

    if (probe === start) {
      while (probe < text.length && !isWord(text[probe])) {
        probe++
      }
    }

    start = probe
    end = probe
  }

  while (start > 0 && isWord(text[start - 1])) {
    start--
  }

  while (end < text.length && isWord(text[end])) {
    end++
  }

  const word = text.slice(start, end)

  return word.length >= 2 ? word : null
}
