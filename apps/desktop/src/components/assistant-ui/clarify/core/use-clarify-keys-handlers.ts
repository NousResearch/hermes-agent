import { visibleClarifyCard } from '@/lib/keybinds/composer-focus-keys'

interface ArrowMove {
  delta: number
  wrap: boolean
}

const INTERACTIVE = 'a[href], button, input, select, textarea, [role="button"]'

function focusOwnsKeys(focused: HTMLElement | null): boolean {
  return Boolean(
    focused && (focused.isContentEditable || (focused.matches(INTERACTIVE) && !focused.matches('button[data-choice]')))
  )
}

/** True when this card must leave the keystroke alone. */
export function isForeignKeystroke(event: globalThis.KeyboardEvent, form: HTMLFormElement | null): boolean {
  if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented) {
    return true
  }

  // Not the visible card ⇒ not our keystroke. Inactive tabs stay MOUNTED,
  // so every parked clarify keeps a live `window` listener; without this the
  // card that acts is whichever mounted first, and answering the question in
  // front of you silently answers a background session's question instead —
  // resuming an agent turn the user never saw. Same resolver the composer's
  // `clarifyCardOwnsKey` yields to, so the two cannot disagree about which
  // card is live.
  if (visibleClarifyCard() !== form) {
    return true
  }

  return focusOwnsKeys(document.activeElement as HTMLElement | null)
}

const ARROWS = new Map<string, (columns: number | undefined) => ArrowMove | null>([
  ['ArrowDown', columns => ({ delta: columns ?? 1, wrap: columns === undefined })],
  ['ArrowLeft', columns => (columns === undefined ? null : { delta: -1, wrap: true })],
  ['ArrowRight', columns => (columns === undefined ? null : { delta: 1, wrap: true })],
  ['ArrowUp', columns => ({ delta: -(columns ?? 1), wrap: columns === undefined })]
])

/** Cursor move for an arrow key; left/right only move in a grid. */
export function arrowMove(key: string, columns: number | undefined): ArrowMove | null {
  return ARROWS.get(key)?.(columns) ?? null
}

/** Row index for a 1-9 / a-z shortcut key, or null. */
export function shortcutIndex(key: string): null | number {
  if (/^[1-9]$/.test(key)) {
    return Number(key) - 1
  }

  const lower = key.toLowerCase()

  // Only the letters this card actually renders a row for (the caller checks
  // the bound). Anything past the last row belongs to the composer — the user
  // is typing a message instead of picking an option, and swallowing the
  // keystroke here would make the first letter of it vanish.
  return lower.length === 1 && lower >= 'a' && lower <= 'z' ? lower.charCodeAt(0) - 97 : null
}
