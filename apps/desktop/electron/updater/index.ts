// updater/index.ts — the desktop update strategy surface.
//
// Every install shape updates through a different owner:
//   app-installer   out-of-store MSIX on win32 — the OS App Installer owns
//                   the apply (the .appinstaller feed registered at install).
//   microsoft-store StoreContext checks and requests the package update.
//   external        Steward-owned deployments with no in-app updater.
//   windows-handoff git checkout on win32 — detached updater binary or the
//                   repo hand-off script owns the swap.
//   posix-handoff   git checkout on macOS/Linux — the repo posix hand-off
//                   script owns the swap.
//   manual          checkout with no staged updater — the user runs
//                   `hermes update` themselves.
//
// The build stamp declares the owner. Runtime dispatch needs no payload probe
// or Store inference; the strategy reports its mechanism to the renderer.

import type { InstallStamp } from '../install-stamp'

export type UpdaterMechanism =
  | 'app-installer'
  | 'electron-updater'
  | 'external'
  | 'microsoft-store'
  | 'windows-handoff'
  | 'posix-handoff'
  | 'manual'

/** The facts the mechanism dispatch keys on. Pure data — injectable for tests. */
export interface MechanismFacts {
  platform: NodeJS.Platform
  updateMechanism: InstallStamp['updateMechanism'] | undefined
}

/**
 * The stamp names the artifact owner. A missing payload must never turn a
 * packaged app into a checkout, and Light needs no payload to update itself.
 */
export function resolveUpdaterMechanism(facts: MechanismFacts): UpdaterMechanism {
  if (facts.updateMechanism && facts.updateMechanism !== 'self') {
    return facts.updateMechanism
  }

  return facts.platform === 'win32' ? 'windows-handoff' : 'posix-handoff'
}

/** The status shape main.ts already sends over `hermes:updates:check`. */
export interface UpdaterStatusWire {
  supported: boolean
  mechanism?: UpdaterMechanism
  updateAvailable?: boolean
  branch?: string
  currentBranch?: string
  reason?: string
  message?: string
  advice?: string
  error?: string
  behind?: number | null
  currentSha?: string
  currentVersion?: string
  channel?: 'stable' | 'canary'
  latestTag?: string | null
  targetSha?: string
  commits?: { sha: string; summary: string; author: string; at: number }[]
  dirty?: boolean
  hermesRoot?: string
  fetchedAt?: number
}

/** The result shape main.ts already sends over `hermes:updates:apply`. */
export interface UpdaterApplyResultWire {
  ok: boolean
  mechanism?: UpdaterMechanism
  error?: string
  message?: string
  manual?: boolean
  bundled?: boolean
  command?: string
  hermesRoot?: string
  handedOff?: boolean
  updater?: string
  [key: string]: unknown
}

export interface UpdaterStrategy {
  readonly mechanism: UpdaterMechanism
  check(opts?: { force?: boolean }): Promise<UpdaterStatusWire>
  apply(opts: { stopSafeBlockers?: boolean }): Promise<UpdaterApplyResultWire>
}
