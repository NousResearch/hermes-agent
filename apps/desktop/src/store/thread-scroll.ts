import { atom, type WritableAtom } from 'nanostores'

import { readKey, writeKey } from '@/lib/storage'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these atoms for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// `$threadScrolledUp` dims the composer / status stack; `$threadJumpButtonVisible`
// shows the floating jump control. Both track `!isAtBottom` today, but stay
// separate so their thresholds can diverge again without touching consumers.
export const $threadScrolledUp = atom(false)
export const $threadJumpButtonVisible = atom(false)

// Skip no-op writes so subscribers don't churn on every scroll tick.
const setter = (target: WritableAtom<boolean>) => (value: boolean) => {
  if (target.get() !== value) {
    target.set(value)
  }
}

const setScrolledUp = setter($threadScrolledUp)
const setJumpButtonVisible = setter($threadJumpButtonVisible)

export const setThreadAtBottom = (isAtBottom: boolean) => {
  setScrolledUp(!isAtBottom)
  setJumpButtonVisible(!isAtBottom)
}

export const resetThreadScroll = () => setThreadAtBottom(true)

// Cross-component bridge: the jump button lives by the composer, the viewport's
// `scrollToBottom` lives inside the thread. The bridge registers a handler; the
// button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Set<() => void>()

export const onScrollToBottomRequest = (handler: () => void) => {
  handlers.add(handler)

  return () => void handlers.delete(handler)
}

export const requestScrollToBottom = () => handlers.forEach(handler => handler())

// Inline edit grows a sticky human bubble. Fire on pointerdown so the viewport
// escapes stick-to-bottom before focus/layout; close clears the edit flag when
// the inline composer unmounts.
const editOpenHandlers = new Set<() => void>()
const editCloseHandlers = new Set<() => void>()

export const onThreadEditOpen = (handler: () => void) => {
  editOpenHandlers.add(handler)

  return () => void editOpenHandlers.delete(handler)
}

export const notifyThreadEditOpen = () => editOpenHandlers.forEach(handler => handler())

export const onThreadEditClose = (handler: () => void) => {
  editCloseHandlers.add(handler)

  return () => void editCloseHandlers.delete(handler)
}

export const notifyThreadEditClose = () => editCloseHandlers.forEach(handler => handler())

// ── Per-session scroll position persistence ──────────────────────────────────
// When the user scrolls up to read history, their scrollTop is saved keyed by
// sessionKey. On return, it's restored in the useLayoutEffect before the default
// settle-to-bottom, so the reading position survives session switches. Positions
// are cleared automatically when the user returns to the bottom of a session.
const SCROLL_POS_KEY = 'hermes.desktop.threadScroll.v1'

type ScrollPositions = Record<string, number>

const MAX_SESSIONS = 120 // matches sessionPreviews cap in preview.ts

let scrollPositions: ScrollPositions = {}

function loadPositions(): ScrollPositions {
  const raw = readKey(SCROLL_POS_KEY)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw) as unknown

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).filter(
        (entry): entry is [string, number] => typeof entry[1] === 'number'
      )
    )
  } catch {
    return {}
  }
}

function persistPositions() {
  // Prune oldest entries when over the cap (JS objects preserve insertion order)
  const keys = Object.keys(scrollPositions)

  while (keys.length > MAX_SESSIONS) {
    delete scrollPositions[keys[0]!]
    keys.shift()
  }

  writeKey(SCROLL_POS_KEY, keys.length === 0 ? null : JSON.stringify(scrollPositions))
}

scrollPositions = loadPositions()

export function getScrollPosition(key: string): number | undefined {
  return scrollPositions[key]
}

export function saveScrollPosition(key: string, scrollTop: number) {
  // Delete then re-add to track recency (JS insertion order = LRU anchor)
  delete scrollPositions[key]
  scrollPositions[key] = scrollTop
  persistPositions()
}

export function clearScrollPosition(key: string) {
  if (scrollPositions[key] === undefined) {
    return
  }

  delete scrollPositions[key]
  persistPositions()
}
