/**
 * Imperative theme switching, for callers with no component to hang a hook on.
 *
 * `useTheme().setTheme` is the React door and covers every in-app surface, but
 * not every switch happens during a render. The backend announces a `/skin`
 * change over the gateway, and a plugin may want to repaint when a connection
 * comes up or a background event fires. Those callers write a name to
 * `$pendingSkinApply`; the ThemeProvider drains it through `setTheme`, so an
 * imperative switch normalizes, persists per profile, and drops any preview
 * exactly like a manual pick — one policy, one owner.
 */

import { $pendingSkinApply } from './backend-sync'
import { resolveTheme } from './user-themes'

/**
 * Ask for a theme switch from outside React. Returns whether the name resolved.
 *
 * An unknown name is refused rather than coerced. `setTheme` falls back to the
 * default skin for anything it can't resolve, which is right for a person
 * picking from a list but wrong for code reacting to an event: a gateway naming
 * a theme the user never installed would silently reset their appearance
 * instead of leaving it alone. The boolean doubles as the availability check,
 * so a caller needs no separate lookup to know it got what it asked for.
 */
export function requestTheme(name: string): boolean {
  if (!resolveTheme(name)) {
    return false
  }

  $pendingSkinApply.set(name)

  return true
}
