/**
 * HUD global summon shortcut — toggles the floating HUD bar from anywhere.
 *
 * Fixed chord CommandOrControl+Shift+U; no keyup handling (press-only,
 * like snap). Exported DEFAULT_HUD_SUMMON_SHORTCUT for main wiring.
 */

import type { GlobalShortcutLike } from './quick-entry'

export const DEFAULT_HUD_SUMMON_SHORTCUT = 'CommandOrControl+Shift+U'

export interface HudSummonShortcutController {
  /** Register the global chord. Returns false when another app owns it. */
  register(): boolean
  /** Release the chord (quit / disabled). Idempotent. */
  dispose(): void
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
        ok = globalShortcut.isRegistered(accelerator)
          ? false
          : globalShortcut.register(accelerator, onSummon)
      } catch {
        ok = false
      }

      active = ok ? accelerator : null

      return ok
    },
    dispose() {
      release()
    }
  }
}
