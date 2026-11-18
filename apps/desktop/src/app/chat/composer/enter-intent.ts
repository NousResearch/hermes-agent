/**
 * Who owns Enter on this surface?
 *
 * A phone keyboard has no Shift to pair with Enter, and a hardware keyboard
 * (or an IME's "return" variant) can still deliver `shiftKey: true` on a touch
 * host. Both forms must stay a line break there — a shifted Return that fell
 * through to the browser was the one path that still bypassed the explicit
 * `insertComposerLineBreak`, which is exactly the "my Return does nothing"
 * report. Desktop keeps Shift+Enter as the browser's native newline.
 */
export type EnterIntent = 'queue' | 'break' | 'send' | 'native'

export function resolveEnterIntent(input: {
  key: string
  shiftKey?: boolean
  metaKey?: boolean
  ctrlKey?: boolean
  touch: boolean
}): EnterIntent {
  if (input.key !== 'Enter') {
    return 'native'
  }

  const submitModifier = Boolean(input.metaKey || input.ctrlKey)

  // Cmd/Ctrl+Enter queues a follow-up while a turn runs, on every host.
  if (submitModifier && !input.shiftKey) {
    return 'queue'
  }

  // Touch: any Return shape opens a line; only the explicit Send button sends.
  if (input.touch) {
    return 'break'
  }

  if (submitModifier || input.shiftKey) {
    return 'native'
  }

  return 'send'
}
