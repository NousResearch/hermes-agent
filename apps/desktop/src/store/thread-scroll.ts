import { atom, type WritableAtom } from 'nanostores'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these atoms for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// `$threadScrolledUp` dims the composer / status stack; `$threadJumpButtonVisible`
// shows the floating jump control. Both track `!isAtBottom` today, but stay
// separate so their thresholds can diverge again without touching consumers.
//
// Keep-alive tabs stay mounted with a real layout box, so only the on-screen
// pane may publish or reset this composer-facing mirror. Jump-to-bottom
// requests are keyed by session so a click (or an input-request snap) cannot
// scroll every mounted transcript.
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

export const publishThreadAtBottom = (isAtBottom: boolean, publisher: { paneVisible: boolean }): void => {
  if (!publisher.paneVisible) {
    return
  }

  setThreadAtBottom(isAtBottom)
}

export const resetPublishedThreadScroll = (publisher: { paneVisible: boolean }): void => {
  if (!publisher.paneVisible) {
    return
  }

  resetThreadScroll()
}

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

// Timeline rail → mounted transcript list. A timeline target can already be in
// assistant-ui's repository while still sitting outside the smaller DOM paint
// budget. Lists report whether they own the requested durable message id; only
// that list raises its render budget, so other visible panes remain untouched.
const revealHandlers = new Set<(scope: string, id: string) => boolean>()

export const onRevealMessageRequest = (handler: (scope: string, id: string) => boolean) => {
  revealHandlers.add(handler)

  return () => void revealHandlers.delete(handler)
}

export const requestRevealMessage = (scope: string, id: string): boolean => {
  let handled = false

  revealHandlers.forEach(handler => {
    handled = handler(scope, id) || handled
  })

  return handled
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
