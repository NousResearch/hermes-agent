import { atom } from 'nanostores'
import { useEffect } from 'react'
import { type NavigationType, useLocation, useNavigationType } from 'react-router'

import { hostHas } from '@/lib/host-capabilities'
import { isBrowserHostedDesktop } from '@/lib/platform'

/** Where this window stands in its own router history: the entry it shows and the newest entry a
 *  Forward can reach. The browser keeps the stack; this keeps only what the band's Back/Forward need
 *  to know (a History API cannot be asked whether a forward entry exists). */
export interface HistoryPosition {
  index: number
  last: number
}

export const $historyPosition = atom<HistoryPosition>({ index: 0, last: 0 })

/** A push drops every forward entry; a pop or replace moves within (or onto) the known stack. */
export function nextHistoryPosition(
  previous: HistoryPosition,
  action: `${NavigationType}`,
  index: number
): HistoryPosition {
  return action === 'PUSH' ? { index, last: index } : { index, last: Math.max(previous.last, index) }
}

export const canGoBack = (position: HistoryPosition) => position.index > 0

/** Whether a band draws its own Back and Forward. A compact band has no room, and the browser-hosted
 *  app leaves them to the browser at any width: its buttons walk the same history. */
export const bandOffersHistory = (compact: boolean) => !compact && !isBrowserHostedDesktop()
export const canGoForward = (position: HistoryPosition) => position.index < position.last

/** The router's own entry number (react-router keeps `idx` in `history.state`). */
function routerEntryIndex(): null | number {
  const index = (window.history.state as { idx?: unknown } | null)?.idx

  return typeof index === 'number' ? index : null
}

/** Follows every navigation of the window's router. Mount once, inside it. */
export function useTrackHistoryPosition() {
  const location = useLocation()
  const action = useNavigationType()

  useEffect(() => {
    const index = routerEntryIndex()

    if (index !== null) {
      $historyPosition.set(nextHistoryPosition($historyPosition.get(), action, index))
    }
  }, [action, location])
}

export const historyBack = () => window.history.back()
export const historyForward = () => window.history.forward()

/** A mouse's back/forward buttons (3/4) walk this history in an app window (a browser tab's own
 *  buttons already do). The preview pane claims them first for its page (it prevents the default),
 *  so they never move the app while the pointer is over a preview. */
export function useMouseHistoryButtons(enabled: boolean) {
  useEffect(() => {
    if (!enabled || !hostHas('appMouseHistoryButtons')) {
      return
    }

    const onMouseDown = (event: MouseEvent) => {
      if ((event.button !== 3 && event.button !== 4) || event.defaultPrevented) {
        return
      }

      event.preventDefault()

      if (event.button === 3) {
        historyBack()
      } else {
        historyForward()
      }
    }

    window.addEventListener('mousedown', onMouseDown)

    return () => window.removeEventListener('mousedown', onMouseDown)
  }, [enabled])
}
