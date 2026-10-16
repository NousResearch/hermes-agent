/**
 * Keep-awake — hold a single machine-global power-save blocker.
 *
 * `prevent-app-suspension` stops the system from sleeping (long overnight
 * agent runs keep going) while still letting the display dim. The renderer
 * owns the preference (persisted in localStorage) and mirrors it here over
 * IPC; the main process owns the one native blocker, same authority split as
 * translucency/zoom. Electron auto-releases the blocker on quit.
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

/**
 * How the blocker follows the user: never, only while a turn is in flight, or
 * around the clock. 'while-working' is the mode the module doc-comment always
 * described — overnight runs survive without the user predicting, before they
 * start, that they will need it, and without a laptop pinned awake all week.
 */
export type KeepAwakeMode = 'always' | 'off' | 'while-working'

const KEEP_AWAKE_MODES: readonly KeepAwakeMode[] = ['off', 'while-working', 'always']

/**
 * Coerce a wire or persisted value to a mode. The pre-mode toggle sent and
 * stored a boolean; `true` meant what 'always' means now, `false` meant 'off'.
 */
export function parseKeepAwakeMode(value: unknown): KeepAwakeMode | null {
  if (value === true) {
    return 'always'
  }

  if (value === false) {
    return 'off'
  }

  return typeof value === 'string' && (KEEP_AWAKE_MODES as readonly string[]).includes(value)
    ? (value as KeepAwakeMode)
    : null
}

/** The main process's persisted copy: the `{ mode }` shape, or the legacy `{ on }`. */
export function readKeepAwakeMode(persisted: unknown): KeepAwakeMode {
  if (typeof persisted !== 'object' || persisted === null) {
    return 'off'
  }

  const record = persisted as Record<string, unknown>

  return parseKeepAwakeMode(record.mode) ?? parseKeepAwakeMode(record.on) ?? 'off'
}

/** Whether the blocker should be held right now, given the mode and the live turn picture. */
export function keepAwakeWanted(mode: KeepAwakeMode, working: boolean): boolean {
  return mode === 'always' || (mode === 'while-working' && working)
}

/**
 * How long 'while-working' keeps holding the blocker after the live turn
 * picture goes idle (#134434).
 *
 * On Windows the sleep idle-timer counts from the last user INPUT, not from
 * the release of a power request. On an unattended overnight run the user has
 * been idle far past the sleep timeout, so releasing the blocker at turn end
 * lets the machine sleep within SECONDS — before the next follow-up turn
 * (process_complete, bg-review, compression) can re-arm `busy`. `busy` only
 * re-arms on `message.start`, which the backend emits after it accepts the
 * follow-up, so a cold model request can spend tens of seconds in the gap.
 * A grace hold after the last turn bridges every such gap.
 */
export const KEEP_AWAKE_RELEASE_GRACE_MS = 5 * 60_000

/**
 * Decide whether the blocker should be held, factoring the release grace.
 *
 * `withinGrace` is true while the hold-after-idle window from the last turn is
 * still open. Any live work resets the window, so a follow-up turn that
 * re-arms busy simply continues the hold (no stop/start churn on the native
 * blocker).
 *
 * The grace only applies to 'while-working': 'always' never releases and
 * 'off' never holds, so neither has a window to bridge.
 */
export function keepAwakeHeld(mode: KeepAwakeMode, working: boolean, withinGrace: boolean): boolean {
  if (mode === 'off') {
    return false
  }

  if (mode === 'always') {
    return true
  }

  return working || withinGrace
}
