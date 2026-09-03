// Pure logic for the find-in-page bar (⌘F). Kept out of the component so the
// match-counter projection and the in-bar key routing can be unit-tested
// without jsdom, a BrowserWindow, or the preload bridge.
//
// The Electron side of the feature lives in electron/find-in-page.ts; this is
// strictly renderer presentation logic.

/**
 * Counter shown next to the input, e.g. `"3/12"`.
 *
 * Three distinct states, and they are not the same thing:
 * - No query → `''`. The counter hides entirely rather than claiming "0/0"
 *   before the user has asked anything.
 * - A query with no matches → `'0/0'`. An explicit, honest zero.
 * - A query with matches → `'<ordinal>/<count>'`.
 *
 * `activeMatchOrdinal` is 1-indexed and can legitimately arrive as 0 from
 * Electron for the frame between issuing a search and the first match being
 * selected, so the ordinal is clamped into `[0, count]` rather than trusted.
 */
export function formatMatchLabel(query: string, activeMatchOrdinal: number, matchCount: number): string {
  if (!query) {
    return ''
  }

  const count = Number.isFinite(matchCount) && matchCount > 0 ? Math.floor(matchCount) : 0

  if (count === 0) {
    return '0/0'
  }

  const raw = Number.isFinite(activeMatchOrdinal) ? Math.floor(activeMatchOrdinal) : 0
  const ordinal = Math.min(Math.max(raw, 0), count)

  return `${ordinal}/${count}`
}

/** What a keypress means to an open find bar. `null` = not ours, let it through. */
export type FindBarKeyAction = 'close' | 'next' | 'previous' | null

/** The subset of a keyboard event the matcher needs — works for DOM and React events. */
export interface FindBarKeyEvent {
  key: string
  shiftKey?: boolean
  metaKey?: boolean
  ctrlKey?: boolean
  altKey?: boolean
}

/**
 * Map a keypress to a find-bar action while the bar is open.
 *
 * Two families, matching the platform convention Chrome/Safari/VS Code (and
 * Claude Desktop's `findInPage` accelerators) all share:
 * - Bare `Enter` / `Shift+Enter` step forward / backward. Only valid while
 *   focus is in the find input, so callers pass `inInput: true` there.
 * - `⌘G` / `⌘⇧G` (Ctrl+G / Ctrl+Shift+G off macOS) step forward / backward
 *   from anywhere while the bar is open — that is the accelerator pair, and it
 *   must not require the input to hold focus.
 * - `Escape` closes from anywhere.
 *
 * `Alt` is treated as disqualifying so ⌥⌘G and friends fall through to
 * whatever else may want them instead of being silently swallowed.
 */
export function findBarKeyAction(event: FindBarKeyEvent, options: { inInput?: boolean } = {}): FindBarKeyAction {
  if (event.altKey) {
    return null
  }

  const mod = Boolean(event.metaKey || event.ctrlKey)

  if (event.key === 'Escape') {
    return mod ? null : 'close'
  }

  // `event.key` for the G key is 'g' unshifted and 'G' with Shift held, so
  // compare case-insensitively and read direction from `shiftKey` alone.
  if (mod && event.key.toLowerCase() === 'g') {
    return event.shiftKey ? 'previous' : 'next'
  }

  if (event.key === 'Enter' && !mod && options.inInput) {
    return event.shiftKey ? 'previous' : 'next'
  }

  return null
}

/**
 * Combos the open find bar owns, in canonical `comboFromEvent` form.
 *
 * The global keybind dispatcher (app/hooks/use-keybinds.ts) consults this
 * before routing a combo to the registry. Without it, three real collisions
 * fire alongside the find bar:
 * - `mod+g` → `view.toggleReview` (⌘G is the review pane's default).
 * - `mod+shift+g` → whatever a user has bound there.
 * - `escape` → `composer.cancel`, which would abort a running turn while the
 *   user only meant to dismiss the find bar.
 *
 * `stopPropagation` cannot solve this: both listeners sit on `window` in the
 * capture phase, and propagation control does not suppress sibling listeners
 * on the same target. Ownership has to be decided by the dispatcher, which is
 * the documented single owner of combo dispatch. This matches the
 * "keyboard ownership follows focus / one cancel gesture does one thing"
 * invariant in apps/desktop/AGENTS.md.
 */
export function findBarClaimsCombo(combo: string): boolean {
  return combo === 'mod+g' || combo === 'mod+shift+g' || combo === 'escape'
}
