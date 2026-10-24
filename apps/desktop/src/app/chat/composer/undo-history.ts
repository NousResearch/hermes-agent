/**
 * The composer's own undo stack.
 *
 * The rich editor mutates its DOM through `Range` rather than the browser's
 * editing commands — `execCommand('insertText')` is ~O(n²) on large multiline
 * blobs and froze the composer for seconds on a big paste (#45812). The cost of
 * that bypass is that those mutations never reach Chromium's undo stack, so
 * ⌘Z skipped straight past a paste and undid whatever came *before* it, leaving
 * the pasted text stranded.
 *
 * Owning the whole stack is the only coherent fix: a half-owned one interleaves
 * our snapshots with Chromium's own typing entries and undoes them out of order.
 * So every composer edit — typed or programmatic — records here, and the editor
 * intercepts ⌘Z / ⌘⇧Z instead of letting the native command run.
 *
 * Snapshots are plain text + a caret offset, not DOM: the editor already
 * round-trips losslessly through `composerPlainText`/`renderComposerContents`,
 * so text is the smallest thing that fully restores a state.
 */

export interface ComposerSnapshot {
  caret: number
  text: string
}

/** Consecutive typing inside this window collapses into one undo entry, so ⌘Z
 *  steps back by a burst the way a native editor does — not one character. */
const COALESCE_WINDOW_MS = 600

const DEFAULT_LIMIT = 200

/** True for ordinary typing / deleting `beforeinput` types that should share one
 *  undo entry. Paste, drop, and history stay discrete. */
export function isCoalescableTypingInput(inputType: string): boolean {
  if (
    inputType === 'insertFromPaste' ||
    inputType === 'insertFromDrop' ||
    inputType === 'insertFromYank' ||
    inputType === 'historyUndo' ||
    inputType === 'historyRedo'
  ) {
    return false
  }

  return inputType.startsWith('insert') || inputType.startsWith('delete')
}

export interface ComposerUndoHistory {
  /** Drop all history and start over from `snapshot`'s state (session swap). */
  reset: () => void
  /** Redo one step. `current` is the live state, banked for a subsequent undo. */
  redo: (current: ComposerSnapshot) => ComposerSnapshot | null
  /** Bank the state that existed *before* an edit. `coalesce` merges this into
   *  the previous entry when it lands inside the typing window. */
  record: (previous: ComposerSnapshot, options?: { coalesce?: boolean }) => void
  /** Undo one step. `current` is the live state, banked for a subsequent redo. */
  undo: (current: ComposerSnapshot) => ComposerSnapshot | null
}

export function createComposerUndoHistory(
  limit = DEFAULT_LIMIT,
  now: () => number = () => Date.now()
): ComposerUndoHistory {
  let past: ComposerSnapshot[] = []
  let future: ComposerSnapshot[] = []
  let lastRecordedAt = 0
  let lastWasCoalescable = false

  const record: ComposerUndoHistory['record'] = (previous, options) => {
    const coalesce = options?.coalesce ?? false
    const at = now()
    const merges = coalesce && lastWasCoalescable && past.length > 0 && at - lastRecordedAt < COALESCE_WINDOW_MS

    lastRecordedAt = at
    lastWasCoalescable = coalesce
    // A fresh edit invalidates anything the user had redone past.
    future = []

    // Merging keeps the OLDER snapshot — the entry already holds the state from
    // the start of the burst, which is what ⌘Z should step back to.
    if (merges) {
      return
    }

    // A no-op edit (same text) would make ⌘Z look broken for one press.
    if (past[past.length - 1]?.text === previous.text) {
      return
    }

    past.push(previous)

    if (past.length > limit) {
      past = past.slice(past.length - limit)
    }
  }

  const step = (from: ComposerSnapshot[], to: ComposerSnapshot[], current: ComposerSnapshot) => {
    const next = from.pop()

    if (!next) {
      return null
    }

    to.push(current)
    // Any traversal ends the typing burst, so the next keystroke opens a new entry.
    lastWasCoalescable = false

    return next
  }

  return {
    record,
    redo: current => step(future, past, current),
    reset: () => {
      past = []
      future = []
      lastRecordedAt = 0
      lastWasCoalescable = false
    },
    undo: current => step(past, future, current)
  }
}

/** True for the keystroke that means "undo" (⌘Z / Ctrl+Z, without Shift). */
export function isUndoShortcut(event: Pick<KeyboardEvent, 'altKey' | 'ctrlKey' | 'key' | 'metaKey' | 'shiftKey'>) {
  return (event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'z'
}

/** True for "redo" — ⌘⇧Z everywhere, plus Ctrl+Y on Windows/Linux. */
export function isRedoShortcut(event: Pick<KeyboardEvent, 'altKey' | 'ctrlKey' | 'key' | 'metaKey' | 'shiftKey'>) {
  if (event.altKey) {
    return false
  }

  const key = event.key.toLowerCase()

  if ((event.metaKey || event.ctrlKey) && event.shiftKey && key === 'z') {
    return true
  }

  return event.ctrlKey && !event.metaKey && !event.shiftKey && key === 'y'
}
