/**
 * Main's half of keep-awake: the one power-save blocker, its persisted mode,
 * and the one-time offer after a timed sleep interrupted a turn. Kept apart
 * from power-save.ts so the renderer bundle never pulls in Node's fs.
 */

import fs from 'node:fs'
import path from 'node:path'

import {
  createKeepAwake,
  type KeepAwakeMode,
  keepAwakeWanted,
  parseKeepAwakeMode,
  type PowerSaveBlockerLike,
  readKeepAwakeMode
} from './power-save'

/**
 * A sleep that starts after at least this long without input is a timed sleep, the kind
 * keep-awake prevents. Closing the lid or choosing Sleep is not, and keep-awake can't stop it.
 * Windows' shortest sleep timeout is one minute; the idle reading is in whole seconds and can
 * land just under it.
 */
export const IDLE_SLEEP_MIN_SECONDS = 55

export interface KeepAwakeControllerDeps {
  blocker: PowerSaveBlockerLike
  /** `keep-awake.json` in the app's userData: the mode, and whether the offer was made. */
  configPath: string
  isWorking: () => boolean
  log: (line: string) => void
  /** Show the one-time "keep awake while working" offer; false when no chat window can show it. */
  offer: () => boolean
}

export interface KeepAwakeController {
  /** Reconcile the blocker with the mode and the live turn picture. */
  apply(working?: boolean): void
  /** App ready: restore the saved mode, or the default when nothing usable is saved. */
  restore(): void
  /** A mode from the renderer, or the boolean the pre-mode toggle sent. */
  setMode(value: unknown): void
  noteSuspend(idleSeconds: number): void
  noteResume(): void
}

/**
 * Main's half of keep-awake: the one blocker, its persisted mode, and the offer. When a timed
 * sleep interrupts a turn while keep-awake is off, the next wake offers "While working", once
 * per computer.
 */
export function createKeepAwakeController(deps: KeepAwakeControllerDeps): KeepAwakeController {
  const keepAwake = createKeepAwake(deps.blocker)
  let mode: KeepAwakeMode = 'off'
  let offered = false
  let sleptMidTurn = false

  const persist = () => {
    try {
      fs.mkdirSync(path.dirname(deps.configPath), { recursive: true })
      fs.writeFileSync(deps.configPath, JSON.stringify({ mode, offered }, null, 2), 'utf8')
    } catch (error) {
      deps.log(`[keep-awake] write failed: ${(error as Error).message}`)
    }
  }

  const apply = (working = deps.isWorking()) => {
    keepAwake.set(keepAwakeWanted(mode, working))
  }

  return {
    apply,
    restore() {
      let saved: unknown = null

      try {
        saved = JSON.parse(fs.readFileSync(deps.configPath, 'utf8'))
      } catch {
        // First launch, or an unreadable file: the defaults stand.
      }

      mode = readKeepAwakeMode(saved)
      offered = typeof saved === 'object' && saved !== null && (saved as Record<string, unknown>).offered === true
      apply()
    },
    setMode(value) {
      const next = parseKeepAwakeMode(value)

      if (next === null) {
        return
      }

      mode = next
      apply()
      persist()
    },
    noteSuspend(idleSeconds) {
      sleptMidTurn = mode === 'off' && deps.isWorking() && idleSeconds >= IDLE_SLEEP_MIN_SECONDS
    },
    noteResume() {
      if (sleptMidTurn && !offered && deps.offer()) {
        offered = true
        persist()
      }

      sleptMidTurn = false
    }
  }
}
