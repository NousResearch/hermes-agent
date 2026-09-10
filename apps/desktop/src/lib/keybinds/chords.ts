import { isMacPlatform } from '@/lib/platform'

/**
 * True when the event is the ⌘/Ctrl+L chord (no shift). The chord routes
 * input to the composer: a terminal or preview selection goes in as context,
 * and a bare press moves focus. The priority ladder between those consumers
 * lives in app/chat/composer/focus-chord.ts.
 */
export function isComposerChord(event: KeyboardEvent): boolean {
  const mod = isMacPlatform() ? event.metaKey : event.ctrlKey

  return mod && !event.shiftKey && event.key.toLowerCase() === 'l'
}

/**
 * True when the event is the universal blocking-prompt confirm chord:
 * ⌘⏎ on macOS, Ctrl+Enter elsewhere (⌃⏎ also folds into it via the
 * explicit ctrlKey check). Shift/Alt variants are different gestures, not
 * confirms. Consumers: the tool-approval bar, the batch clarify card's
 * confirm, and the type-to-focus gate that must yield exactly these keys.
 */
export function isConfirmChord(event: {
  key: string
  metaKey: boolean
  ctrlKey: boolean
  shiftKey: boolean
  altKey: boolean
}): boolean {
  return event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey && !event.altKey
}
