// updater/relaunch.ts — unconditional relaunch after an App Installer swap.
//
// Contract: clicking Update always ends in Hermes reopening on the new
// version — no user action between quit and relaunch. Implementation: a
// one-shot pending-relaunch marker written before we quit. The OS App
// Installer swaps the package; on next launch (whenever it happens) Hermes
// reads the marker, shows the "updated" toast, and deletes it.
//
// The marker is a JSON file under HERMES_HOME (not a run-key / scheduled
// task): a registry entry can't be made one-shot safely from a dying process,
// and a file survives the package swap (HERMES_HOME lives outside the
// package). For the actual relaunch we rely on Windows restarting the app
// when an MSIX package update completes while the app registered itself
// for restart (RegisterApplicationRestart) — the marker's job is only to
// tell the NEW version it was an update relaunch, so it can toast + clean up.
//
// If the OS does NOT relaunch (update applied later, on next launch), the
// marker is still correct: the first launch after the swap detects it and
// toasts. A stale marker (update cancelled/failed) self-deletes when its
// expected-version no longer matches a later install... no — cancelled
// updates mean the version never changed, so we stamp with "expected NEW
// version unknown" instead: the marker only records that an update was
// STARTED; the new version decides "was it really me?" by comparing versions
// recorded at write time vs launch time.

import * as fs from 'node:fs'
import * as path from 'node:path'

export interface PendingRelaunchMarker {
  schemaVersion: 1
  /** Version of the app that wrote the marker (pre-update). */
  fromVersion: string
  /** Wall-clock ms when the update was triggered. */
  startedAt: number
}

const MARKER_FILENAME = 'pending-update-relaunch.json'

function markerPath(hermesHome: string): string {
  return path.join(hermesHome, MARKER_FILENAME)
}

/** Pure: the filename constant, for tests. */
export const PENDING_RELAUNCH_FILENAME = MARKER_FILENAME

/**
 * Register the one-shot post-update relaunch marker. Best-effort — a failure
 * to write never blocks the update; relaunch just stays manual.
 */
export function writePendingRelaunch(
  hermesHome: string,
  fromVersion: string,
  writeFile: (file: string, contents: string) => void = (f, c) => fs.writeFileSync(f, c)
): boolean {
  try {
    const marker: PendingRelaunchMarker = { schemaVersion: 1, fromVersion, startedAt: Date.now() }
    writeFile(markerPath(hermesHome), JSON.stringify(marker))

    return true
  } catch {
    return false
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
  hermesHome: string,
  currentVersion: string,
  deps: RelaunchFsDeps = {}
): ConsumedRelaunch {
  const existsSync = deps.existsSync ?? ((file: string) => fs.existsSync(file))
  const readFileSync = deps.readFileSync ?? ((file: string) => fs.readFileSync(file, 'utf8'))
  const unlinkSync = deps.unlinkSync ?? ((file: string) => fs.unlinkSync(file))

  const file = markerPath(hermesHome)

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
