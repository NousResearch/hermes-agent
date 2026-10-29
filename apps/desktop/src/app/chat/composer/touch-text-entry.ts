import type { TouchEvent as ReactTouchEvent } from 'react'

import { isMacPlatform } from '@/lib/platform'

import { placeCaretEnd } from './rich-editor'

// A tap, not a scroll, drag or long press (which iOS hands to selection).
const TAP_MAX_MOVE_PX = 10
const TAP_MAX_MS = 500

/** Text-entry attributes for the composer editor. Desktop keeps them off: the
 *  composer takes paths, code and commands, and Chromium's macOS
 *  text-replacement path shares the autocorrect gate (spellcheck stays off, so
 *  smart quotes and dashes stay out). A touch host types prose on a soft
 *  keyboard and gets the platform's casing, correction and spelling. */
export function composerTextEntryProps(touch: boolean) {
  return touch
    ? ({ autoCapitalize: 'sentences', autoCorrect: 'on', spellCheck: true } as const)
    : ({ autoCapitalize: 'off', autoCorrect: isMacPlatform() ? 'on' : 'off', spellCheck: false } as const)
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

      if (!began || !touch || editor.getAttribute('contenteditable') === 'false' || document.activeElement === editor) {
        return
      }

      if (!stillAt(began, touch) || event.timeStamp - began.at > TAP_MAX_MS) {
        return
      }

      editor.focus()
      placeCaretAt(editor, touch.clientX, touch.clientY)
    },
    // A scroll that comes back near its start before lifting is still a scroll.
    onTouchMove(event: ReactTouchEvent<HTMLElement>) {
      const touch = event.touches[0]

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
