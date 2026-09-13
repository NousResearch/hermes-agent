/**
 * Keep-awake — hold a single machine-global power-save blocker.
 *
 * `prevent-app-suspension` stops the system from sleeping (long overnight
 * agent runs keep going) while still letting the display dim. The renderer
 * owns the preference (persisted in localStorage) and mirrors it here over
 * IPC; the main process owns the one native blocker, same authority split as
 * translucency/zoom. Electron auto-releases the blocker on quit.
 *
 * The preference alone does not hold the blocker: it is scoped to agent runs
 * (see `shouldHoldAwake`), so an idle Hermes lets the machine sleep normally.
 */

export type KeepAwakeType = 'prevent-app-suspension' | 'prevent-display-sleep'

/** The slice of Electron's `powerSaveBlocker` we use (injected for testing). */
export interface PowerSaveBlockerLike {
  start(type: KeepAwakeType): number
  stop(id: number): void
  isStarted(id: number): boolean
}

export interface KeepAwake {
  /** Turn the blocker on/off (idempotent). Returns the resulting state. */
  set(on: boolean): boolean
  isActive(): boolean
}

/**
 * Whether the blocker should be held right now: only while the user has the
 * preference on AND at least one agent turn is in flight. Releasing at idle is
 * the whole point — a finished overnight run must let the machine sleep.
 */
export function shouldHoldAwake(preferenceOn: boolean, activeTurnCount: number): boolean {
  return preferenceOn && activeTurnCount > 0
}

export function createKeepAwake(
  blocker: PowerSaveBlockerLike,
  type: KeepAwakeType = 'prevent-app-suspension'
): KeepAwake {
  let id: null | number = null

  const isActive = () => id !== null && blocker.isStarted(id)

  return {
    isActive,
    set(on) {
      if (on && !isActive()) {
        id = blocker.start(type)
      } else if (!on && id !== null) {
        if (blocker.isStarted(id)) {
          blocker.stop(id)
        }

        id = null
      }

      return isActive()
    }
  }
}
