import { atom, type WritableAtom } from 'nanostores'

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

// Long-session render budget: ThreadMessageList caps the rendered-DOM to the
// newest ~300 parts (RENDER_BUDGET) and slices older turns out of the viewport.
// The right-edge prompt rail subscribes to the FULL message list (every user
// prompt) and so can be clicked for turns whose DOM nodes don't exist. The
// timeline calls requestExpandRenderBudget(targetMessageId) to ask
// ThreadMessageList to lower its firstVisible cutoff so the target group's
// DOM mounts — without this, scrollToPrompt silently no-ops at
// thread-timeline.tsx on clicks past the budget boundary (issue #52816).
//
// The bridge carries a message id (not a numeric index) because the rail's
// `entries` are filtered (blank + process-notification user messages dropped)
// while ThreadMessageList's `groups` include every user message. An index
// would silently mis-aim whenever a filtered message precedes the target
// (teknium1 review, 2026-07-15). The consumer resolves the id against the
// `groups` array via `groups.findIndex` so it lines up exactly with what
// `firstVisible` slices — earlier drafts used a user-ordinal helper, but that
// mis-aimed when a standalone non-user group (e.g. a system message at
// session start) preceded the target (Gemini 3.5 Flash + GPT-OSS code
// review, 2026-07-16).
const expandBudgetHandlers = new Set<(targetMessageId: string) => void>()

export const onExpandRenderBudgetRequest = (handler: (targetMessageId: string) => void) => {
  expandBudgetHandlers.add(handler)

  return () => void expandBudgetHandlers.delete(handler)
}

export const requestExpandRenderBudget = (targetMessageId: string) => {
  expandBudgetHandlers.forEach(handler => handler(targetMessageId))
}
