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
// Factory-shaped for the split pane: each mounted thread + its chrome share ONE
// instance (two threads writing a shared atom would fight over it), delivered
// through the pane bundle (see app/chat/pane-view.ts). The module-level exports
// below stay the MAIN pane's instance, so every existing consumer is untouched.

export interface ThreadScrollInstance {
  $threadScrolledUp: WritableAtom<boolean>
  $threadJumpButtonVisible: WritableAtom<boolean>
  setThreadAtBottom: (isAtBottom: boolean) => void
  resetThreadScroll: () => void
  /** Fire the registered viewport handler(s) — the jump button's click path. */
  requestScrollToBottom: () => void
  /** Cross-component bridge: the jump button lives by the composer, the
   *  viewport's `scrollToBottom` lives inside the thread. The thread registers
   *  a handler; the button fires it. Mirrors the composer focus/insert emitter
   *  pattern, scoped per pane so one pane's jump can't scroll the other. */
  onScrollToBottomRequest: (handler: () => void) => () => void
}

export function createThreadScrollAtoms(): ThreadScrollInstance {
  const $threadScrolledUp = atom(false)
  const $threadJumpButtonVisible = atom(false)

  // Skip no-op writes so subscribers don't churn on every scroll tick.
  const setter = (target: WritableAtom<boolean>) => (value: boolean) => {
    if (target.get() !== value) {
      target.set(value)
    }
  }

  const setScrolledUp = setter($threadScrolledUp)
  const setJumpButtonVisible = setter($threadJumpButtonVisible)

  const setThreadAtBottom = (isAtBottom: boolean) => {
    setScrolledUp(!isAtBottom)
    setJumpButtonVisible(!isAtBottom)
  }

  const handlers = new Set<() => void>()

  return {
    $threadScrolledUp,
    $threadJumpButtonVisible,
    setThreadAtBottom,
    resetThreadScroll: () => setThreadAtBottom(true),
    requestScrollToBottom: () => handlers.forEach(handler => handler()),
    onScrollToBottomRequest: handler => {
      handlers.add(handler)

      return () => void handlers.delete(handler)
    }
  }
}

/** The main pane's instance — the atoms every pre-split consumer imported. */
export const mainThreadScroll = createThreadScrollAtoms()

export const $threadScrolledUp = mainThreadScroll.$threadScrolledUp
export const $threadJumpButtonVisible = mainThreadScroll.$threadJumpButtonVisible
export const setThreadAtBottom = mainThreadScroll.setThreadAtBottom
export const resetThreadScroll = mainThreadScroll.resetThreadScroll
export const onScrollToBottomRequest = mainThreadScroll.onScrollToBottomRequest
export const requestScrollToBottom = mainThreadScroll.requestScrollToBottom

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
