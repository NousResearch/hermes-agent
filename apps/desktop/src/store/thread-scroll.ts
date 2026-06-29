import { atom, type WritableAtom } from 'nanostores'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these atoms for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// `$threadScrolledUp` dims the composer / status stack. `$threadJumpState`
// describes the single floating navigation pill: legacy bottom jump, latest
// answer start, or latest answer end. Keeping them separate lets the pill stay
// visible at the bottom of a long answer without pretending the thread is
// scrolled up.
export const $threadScrolledUp = atom(false)

export type ThreadJumpTarget = 'answer-end' | 'answer-start' | 'bottom'

export interface ThreadJumpState {
  target: ThreadJumpTarget
  visible: boolean
}

export interface LatestAnswerJumpGeometry {
  answerHeight: number
  answerTop: number
  clientHeight: number
  scrollTop: number
}

export const DEFAULT_THREAD_JUMP_STATE: ThreadJumpState = { target: 'bottom', visible: false }

const ANSWER_JUMP_MIN_HEIGHT_PX = 420
const ANSWER_JUMP_MIN_VIEWPORT_RATIO = 0.95
const ANSWER_JUMP_NEAR_START_PX = 120

export const $threadJumpState = atom<ThreadJumpState>(DEFAULT_THREAD_JUMP_STATE)

// Back-compat surface for any remaining consumers that only care about
// visibility. New code should prefer `$threadJumpState`.
export const $threadJumpButtonVisible = atom(false)

// Skip no-op writes so subscribers don't churn on every scroll tick.
const setter = (target: WritableAtom<boolean>) => (value: boolean) => {
  if (target.get() !== value) {
    target.set(value)
  }
}

const setScrolledUp = setter($threadScrolledUp)
const setJumpButtonVisible = setter($threadJumpButtonVisible)

function syncJumpButtonVisible() {
  setJumpButtonVisible($threadScrolledUp.get() || $threadJumpState.get().visible)
}

export function latestAnswerJumpState(geometry: LatestAnswerJumpGeometry): ThreadJumpState {
  const { answerHeight, answerTop, clientHeight, scrollTop } = geometry

  if (
    !Number.isFinite(answerHeight) ||
    !Number.isFinite(answerTop) ||
    !Number.isFinite(clientHeight) ||
    !Number.isFinite(scrollTop) ||
    answerHeight <= 0 ||
    clientHeight <= 0
  ) {
    return DEFAULT_THREAD_JUMP_STATE
  }

  const usefulHeight = Math.max(ANSWER_JUMP_MIN_HEIGHT_PX, clientHeight * ANSWER_JUMP_MIN_VIEWPORT_RATIO)

  if (answerHeight < usefulHeight) {
    return DEFAULT_THREAD_JUMP_STATE
  }

  return {
    target: scrollTop <= answerTop + ANSWER_JUMP_NEAR_START_PX ? 'answer-end' : 'answer-start',
    visible: true
  }
}

export const setThreadAtBottom = (isAtBottom: boolean) => {
  setScrolledUp(!isAtBottom)
  syncJumpButtonVisible()
}

export const setThreadJumpState = (state: ThreadJumpState) => {
  const current = $threadJumpState.get()

  if (current.visible !== state.visible || current.target !== state.target) {
    $threadJumpState.set(state)
  }

  syncJumpButtonVisible()
}

export const resetThreadScroll = () => {
  setThreadAtBottom(true)
  setThreadJumpState(DEFAULT_THREAD_JUMP_STATE)
}

// Cross-component bridge: the jump button lives by the composer, the viewport's
// scroll owner lives inside the thread. The bridge registers a handler; the
// button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Set<(target: ThreadJumpTarget) => void>()

export const onThreadJumpRequest = (handler: (target: ThreadJumpTarget) => void) => {
  handlers.add(handler)

  return () => void handlers.delete(handler)
}

export const requestThreadJump = (target = $threadJumpState.get().target) => handlers.forEach(handler => handler(target))

// Legacy aliases for the old bottom-only control.
export const onScrollToBottomRequest = (handler: () => void) =>
  onThreadJumpRequest(target => {
    if (target === 'bottom') {
      handler()
    }
  })

export const requestScrollToBottom = () => requestThreadJump('bottom')

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
