// Electron userData isolates installations that share HERMES_HOME.
// The marker survives a package swap and records the version that started it.
// The separate waiter owns automatic relaunch. Cancelling an update stops
// that waiter and removes its marker, including when startup stays manual.

import * as fs from 'node:fs'
import * as path from 'node:path'

import type { App } from 'electron'

import type { RelaunchWaiterHandle } from './relaunch-waiter'

export interface PendingRelaunchMarker {
  schemaVersion: 1
  /** Version of the app that wrote the marker (pre-update). */
  fromVersion: string
  /** Wall-clock ms when the update was triggered. */
  startedAt: number
}

const MARKER_FILENAME = 'pending-update-relaunch.json'

function markerPath(app: Pick<App, 'getPath'>): string {
  return path.join(app.getPath('userData'), MARKER_FILENAME)
}

/** Pure: the filename constant, for tests. */
export const PENDING_RELAUNCH_FILENAME = MARKER_FILENAME

/**
 * Register the one-shot post-update relaunch marker. Best-effort — a failure
 * to write never blocks the update; relaunch just stays manual.
 */
export function writePendingRelaunch(
  app: Pick<App, 'getPath'>,
  fromVersion: string,
  writeFile: (file: string, contents: string) => void = (f: string, c: string): void => fs.writeFileSync(f, c)
): boolean {
  try {
    const marker: PendingRelaunchMarker = { schemaVersion: 1, fromVersion, startedAt: Date.now() }
    writeFile(markerPath(app), JSON.stringify(marker))

    return true
  } catch {
    return false
  }
}

export interface UpdateRelaunchDeps {
  /** Return a ready waiter, or no handle after a safely stopped failure. */
  relaunch: () => RelaunchWaiterHandle | undefined | Promise<RelaunchWaiterHandle | undefined>
}

export interface RelaunchRegistration {
  automatic: boolean
  cancel: () => Promise<void>
}

/** Marker failure permits an update, but failed waiter cleanup must abort it. */
export async function registerUpdateRelaunch(
  app: Pick<App, 'getPath'>,
  fromVersion: string,
  deps: UpdateRelaunchDeps,
  writeFile: (file: string, contents: string) => void = (f: string, c: string): void => fs.writeFileSync(f, c)
): Promise<RelaunchRegistration> {
  const ownsMarker: boolean = writePendingRelaunch(app, fromVersion, writeFile)

  const removeMarker = (): void => {
    if (!ownsMarker) {
      return
    }

    try {
      fs.unlinkSync(markerPath(app))
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error
      }
    }
  }

  let waiter: RelaunchWaiterHandle | undefined

  try {
    waiter = await deps.relaunch()
  } catch (error) {
    try {
      removeMarker()
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], 'Relaunch registration cleanup failed', { cause: error })
    }

    throw error
  }

  let cancellation: Promise<void> | undefined

  return {
    automatic: waiter !== undefined,
    cancel: () => {
      cancellation ??= (async () => {
        const errors: unknown[] = []

        try {
          await waiter?.cancel()
        } catch (error) {
          errors.push(error)
        }

        try {
          removeMarker()
        } catch (error) {
          errors.push(error)
        }

        if (errors.length) {
          throw new AggregateError(errors, 'Relaunch cancellation failed')
        }
      })()

      return cancellation
    }
  }
}

export interface ConsumedRelaunch {
  /** True when this launch IS the post-update relaunch (new version). */
  wasUpdateRelaunch: boolean
  /** The version the update started from, when the marker was present. */
  fromVersion?: string
}

export interface RelaunchFsDeps {
  existsSync?: (file: string) => boolean
  readFileSync?: (file: string) => string
  unlinkSync?: (file: string) => void
}

/**
 * First-run hook: detect + consume the pending-relaunch marker. Returns
 * wasUpdateRelaunch=true only when the CURRENT version differs from the
 * marker's fromVersion — the same version means the update never landed
 * (cancelled/failed OS install), so the marker is deleted and no toast fires.
 */
export function consumePendingRelaunch(
  app: Pick<App, 'getPath'>,
  currentVersion: string,
  deps: RelaunchFsDeps = {}
): ConsumedRelaunch {
  const existsSync = deps.existsSync ?? ((file: string) => fs.existsSync(file))
  const readFileSync = deps.readFileSync ?? ((file: string) => fs.readFileSync(file, 'utf8'))
  const unlinkSync = deps.unlinkSync ?? ((file: string) => fs.unlinkSync(file))

  const file: string = markerPath(app)

  if (!existsSync(file)) {
    return { wasUpdateRelaunch: false }
  }

  let fromVersion: string | undefined

  try {
    const raw = JSON.parse(readFileSync(file)) as PendingRelaunchMarker
    fromVersion = typeof raw.fromVersion === 'string' ? raw.fromVersion : undefined
  } catch {
    fromVersion = undefined
  }

  // Consume unconditionally — the marker is one-shot.
  try {
    unlinkSync(file)
  } catch {
    // Best-effort cleanup; a leftover marker is re-consumed harmlessly.
  }

  return { wasUpdateRelaunch: fromVersion !== undefined && fromVersion !== currentVersion, fromVersion }
}
