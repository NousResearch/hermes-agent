import type { TouchEvent as ReactTouchEvent } from 'react'

import { placeCaretEnd } from './rich-editor'

// A tap, not a scroll, drag or long press (which iOS hands to selection).
const TAP_MAX_MOVE_PX = 10
const TAP_MAX_MS = 500

/** Text-entry attributes for the composer editor. Spelling is checked on every
 *  host that has a keyboard: the composer carries prose as often as it carries
 *  paths, and an unmarked typo reaches the model as a wrong instruction. On
 *  macOS Electron this is the SYSTEM spell checker (NSSpellChecker), so the
 *  squiggle, the suggestion menu and the words the user already taught the Mac
 *  all behave exactly as they do in TextEdit or Word.
 *
 *  Automatic substitution stays off everywhere: the same field takes code,
 *  commands and paths, and a smart quote or an em dash substituted for `"` or
 *  `--` corrupts them. Casing stays off with it — the editor is not a sentence
 *  and must not capitalize after a colon or a bullet. A touch host types prose
 *  on a soft keyboard and gets that keyboard's casing, correction and spelling. */
export function composerTextEntryProps(touch: boolean) {
  return touch
    ? ({ autoCapitalize: 'sentences', autoCorrect: 'on', spellCheck: true } as const)
    : ({ autoCapitalize: 'off', autoCorrect: 'off', spellCheck: true } as const)
}

const stillAt = (from: { x: number; y: number }, touch: { clientX: number; clientY: number }) =>
  Math.hypot(touch.clientX - from.x, touch.clientY - from.y) <= TAP_MAX_MOVE_PX

function placeCaretAt(editor: HTMLElement, x: number, y: number) {
  const range = document.caretRangeFromPoint?.(x, y)
  const selection = window.getSelection()

  if (range && selection && editor.contains(range.startContainer)) {
    selection.removeAllRanges()
    selection.addRange(range)
  } else {
    placeCaretEnd(editor)
  }
}

/** Focus the editor from the touchend of a short, still tap. iPhone Safari
 *  drops the synthetic click of a single tap on the composer (a long press
 *  still works), so the editor never gets focus or the keyboard. A focus()
 *  inside touchend runs in the user gesture, so iOS opens the keyboard. An
 *  already focused editor is left to the browser, which moves the caret. */
export function touchTapFocusHandlers() {
  let start: { at: number; x: number; y: number } | null = null

  return {
    onTouchEnd(event: ReactTouchEvent<HTMLElement>) {
      const touch = event.changedTouches[0]
      const editor = event.currentTarget
      const began = start
      start = null

      // Another finger still down makes this a pinch or a two-finger gesture.
      if (
        !began ||
        !touch ||
        event.touches.length > 0 ||
        editor.getAttribute('contenteditable') === 'false' ||
        document.activeElement === editor
      ) {
        return
      }

      if (!stillAt(began, touch) || event.timeStamp - began.at > TAP_MAX_MS) {
        return
      }

      editor.focus()
      placeCaretAt(editor, touch.clientX, touch.clientY)
    },
    // A scroll that comes back near its start before lifting is still a scroll.
    // A second finger landing outside the editor never fires touchstart here,
    // but every touchmove lists it.
    onTouchMove(event: ReactTouchEvent<HTMLElement>) {
      const touch = event.touches.length === 1 ? event.touches[0] : undefined

      if (start && (!touch || !stillAt(start, touch))) {
        start = null
      }
    },
    onTouchStart(event: ReactTouchEvent<HTMLElement>) {
      const touch = event.touches.length === 1 ? event.touches[0] : undefined
      start = touch ? { at: event.timeStamp, x: touch.clientX, y: touch.clientY } : null
    },
    onTouchCancel() {
      start = null
    }
  }
}
