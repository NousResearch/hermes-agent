/**
 * HUD summon — a global OS chord that brings the HUD up from ANY app.
 *
 * Before this, the only ways into HUD mode were the in-app keybind and the
 * titlebar button — both of which need Hermes to have focus first. That is
 * three context switches (Alt-Tab to Hermes, toggle, Alt-Tab back) to use a
 * surface whose whole point is staying in the app you were already in.
 *
 * Registered for the life of the app (unlike the snap chord, which lives only
 * while the HUD is up). Main owns registration — same authority split as
 * Quick Entry and snap. Electron's globalShortcut is press-only.
 *
 * The chord is deliberately the SAME as the in-app `view.toggleHud` keybind
 * (⌘⇧H). While the global registration holds, it intercepts the press before
 * the renderer sees it and does the same toggle; if another app already owns
 * the chord, registration fails (logged, never silent) and the in-app keybind
 * keeps working exactly as before — graceful degradation, not a dead key.
 */

import type { GlobalShortcutLike } from './quick-entry'

export const DEFAULT_HUD_SUMMON_SHORTCUT = 'CommandOrControl+Shift+H'

/** How a summon should behave, decided from where the user pressed it. */
export type HudSummonMode = 'close' | 'open-external' | 'open-in-app'

/**
 * Pure decision for a summon press.
 *
 * - HUD already up → the chord is a toggle, so dismiss it. One gesture does
 *   exactly one thing in both directions (the Quick Entry convention).
 * - HUD down and a Hermes window has focus → the user is IN the app; open the
 *   HUD the way the titlebar toggle does (take focus, step the app aside).
 * - HUD down and focus is elsewhere → the user is in Figma / a browser / a
 *   terminal. Open the HUD WITHOUT stealing focus and WITHOUT hiding the main
 *   window: they summoned a companion, not a mode switch.
 */
export function resolveHudSummon(state: { hermesFocused: boolean; hudOpen: boolean }): HudSummonMode {
  if (state.hudOpen) {
    return 'close'
  }

  return state.hermesFocused ? 'open-in-app' : 'open-external'
}

export interface HudSummonShortcutController {
  /** Register the global chord. Returns false when another app owns it. */
  register(): boolean
  /** Release the chord (quit). Idempotent. */
  dispose(): void
  /** The accelerator currently held, or null when unregistered/taken. */
  current(): null | string
}

export function createHudSummonShortcut(
  globalShortcut: GlobalShortcutLike,
  onSummon: () => void
): HudSummonShortcutController {
  let active: null | string = null

  const release = () => {
    if (active) {
      try {
        globalShortcut.unregister(active)
      } catch {
        // Best effort — a dead accelerator must not block re-register.
      }

      active = null
    }
  }

  return {
    register() {
      release()

      const accelerator = DEFAULT_HUD_SUMMON_SHORTCUT
      let ok = false

      try {
        ok = globalShortcut.isRegistered(accelerator) ? false : globalShortcut.register(accelerator, onSummon)
      } catch {
        ok = false
      }

      active = ok ? accelerator : null

      return ok
    },
    dispose() {
      release()
    },
    current() {
      return active
    }
  }
}
