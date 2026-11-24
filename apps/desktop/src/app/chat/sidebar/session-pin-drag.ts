/**
 * The redesigned sidebar's drag-to-pin: a chat row carried onto the Pinned section pins it.
 *
 * Classic rows ride `startSessionDrag` (../session-drag.ts), whose pin drop writes the local pin
 * set by bare session id. The redesign lists chats from every machine and profile and pins through
 * its own owner-qualified write, so its rows take only the pin target from that drag language and
 * commit through the same action as the row menu's "Pin". The machinery is the shared pointer drag
 * session: threshold, pointer chip, Esc to cancel, and no click after a drag.
 */

import type { PointerEvent as ReactPointerEvent } from 'react'

import { queryVisible } from '@/components/pane-shell/pane-visibility'
import { rectContains, startDragSession } from '@/components/pane-shell/tree/renderer/drag-session'

/** Marks the section a dragged chat pins into. */
export const sessionPinDropProps = { 'data-session-pin-drop': '' }

// The attribute classic's drop targets light up with; the section's ring keys off it.
const DROP_HOVER_ATTR = 'data-session-drop-hover'

interface SessionPinDrag {
  /** The pointer chip's text: the chat's title. */
  label: string
  /** Released on Pinned. */
  onPin: () => void
  /** Called off mid-drag (Esc, or the pointer was taken away): the session keeps Esc to itself, so
   *  a second drag riding the same press learns of it here. */
  onAbort: () => void
}

export function startSessionPinDrag(event: ReactPointerEvent<HTMLElement>, { label, onPin, onAbort }: SessionPinDrag) {
  let target: HTMLElement | null = null
  let rect: DOMRect | null = null
  let over = false
  let lifted = false

  const setOver = (next: boolean) => {
    over = next

    if (next) {
      target?.setAttribute(DROP_HOVER_ATTR, 'true')
    } else {
      target?.removeAttribute(DROP_HOVER_ATTR)
    }
  }

  startDragSession(event, {
    ghost: { label },
    onEngage() {
      lifted = true
      target = queryVisible<HTMLElement>('[data-session-pin-drop]')
      rect = target?.getBoundingClientRect() ?? null
    },
    // Never a pane drop: the sidebar's own target is the only one this drag has.
    resolveMove(x, y) {
      setOver(Boolean(rect && rectContains(rect, x, y)))

      return null
    },
    onCommit() {
      lifted = false

      if (over) {
        onPin()
      }
    },
    // Runs after a commit too, which has already set the row down.
    onEnd() {
      setOver(false)

      if (lifted) {
        onAbort()
      }
    }
  })
}
