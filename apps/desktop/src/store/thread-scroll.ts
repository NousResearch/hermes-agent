import { atom, type WritableAtom } from 'nanostores'

// `$threadScrolledUp` flips the instant the viewport leaves the bottom (dims the
// composer / status stack). `$threadJumpButtonVisible` trips a little further up
// (~10px) so the floating jump control only shows once meaningfully away.
export const $threadScrolledUp = atom(false)
export const $threadJumpButtonVisible = atom(false)

// Skip no-op writes so subscribers don't churn on every scroll tick.
const setter = (target: WritableAtom<boolean>) => (value: boolean) => {
  if (target.get() !== value) {
    target.set(value)
  }
}

export const setThreadScrolledUp = setter($threadScrolledUp)
export const setThreadJumpButtonVisible = setter($threadJumpButtonVisible)

export const resetThreadScroll = () => {
  setThreadScrolledUp(false)
  setThreadJumpButtonVisible(false)
}

// Cross-component bridge: the jump button lives by the composer, the re-arm +
// pin machinery lives in the virtualizer. The virtualizer registers a handler;
// the button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Set<() => void>()

export const onScrollToBottomRequest = (handler: () => void) => {
  handlers.add(handler)

  return () => void handlers.delete(handler)
}

export const requestScrollToBottom = () => handlers.forEach(handler => handler())
