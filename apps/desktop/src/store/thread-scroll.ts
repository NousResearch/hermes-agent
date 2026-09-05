import { atom } from 'nanostores'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these maps for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// Maps are keyed by session id so split panes publish independently. A global
// singleton lit every pane's jump button whenever any sibling scrolled up
// (#103586). Keep-alive tabs stay mounted with a real layout box, so only the
// on-screen pane may publish or reset its session's composer-facing mirror.
// Jump-to-bottom requests are keyed by session so a click (or an input-request
// snap) cannot scroll every mounted transcript.
export const $threadScrolledUpBySession = atom<Record<string, boolean>>({})
export const $threadJumpButtonVisibleBySession = atom<Record<string, boolean>>({})

function sessionKey(sessionId: string | null | undefined): string {
  return sessionId ?? ''
}

function setSessionFlag(
  target: typeof $threadScrolledUpBySession,
  sessionId: string | null | undefined,
  value: boolean
): void {
  const key = sessionKey(sessionId)
  const current = target.get()

  if (Boolean(current[key]) === value) {
    return
  }

  if (value) {
    target.set({ ...current, [key]: true })
    return
  }

  if (!(key in current)) {
    return
  }

  const next = { ...current }
  delete next[key]
  target.set(next)
}

export const setThreadAtBottom = (isAtBottom: boolean, sessionId: string | null = null) => {
  setSessionFlag($threadScrolledUpBySession, sessionId, !isAtBottom)
  setSessionFlag($threadJumpButtonVisibleBySession, sessionId, !isAtBottom)
}

export const resetThreadScroll = (sessionId: string | null = null) => setThreadAtBottom(true, sessionId)

export const publishThreadAtBottom = (
  isAtBottom: boolean,
  publisher: { paneVisible: boolean; sessionId?: string | null }
): void => {
  if (!publisher.paneVisible) {
    return
  }

  setThreadAtBottom(isAtBottom, publisher.sessionId ?? null)
}

export const resetPublishedThreadScroll = (publisher: {
  paneVisible: boolean
  sessionId?: string | null
}): void => {
  if (!publisher.paneVisible) {
    return
  }

  resetThreadScroll(publisher.sessionId ?? null)
}

export const isThreadScrolledUp = (sessionId: string | null | undefined): boolean =>
  Boolean($threadScrolledUpBySession.get()[sessionKey(sessionId)])

export const isThreadJumpButtonVisible = (sessionId: string | null | undefined): boolean =>
  Boolean($threadJumpButtonVisibleBySession.get()[sessionKey(sessionId)])

// Cross-component bridge: the jump button lives by the composer, the viewport's
// `scrollToBottom` lives inside the thread. The bridge registers a handler; the
// button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Map<string | null, Set<() => void>>()

export const onScrollToBottomRequest = (handler: () => void, sessionId: string | null = null) => {
  const scoped = handlers.get(sessionId) ?? new Set<() => void>()

  scoped.add(handler)
  handlers.set(sessionId, scoped)

  return () => {
    scoped.delete(handler)

    if (scoped.size === 0) {
      handlers.delete(sessionId)
    }
  }
}

export const requestScrollToBottom = (sessionId: string | null = null) => {
  handlers.get(sessionId)?.forEach(handler => handler())
}

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
