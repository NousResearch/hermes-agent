// updater/index.ts — the desktop update strategy surface.
//
// Every install shape updates through a different owner:
//   app-installer   out-of-store MSIX on win32 — the OS App Installer owns
//                   the apply (the .appinstaller feed registered at install).
//   external        Microsoft Store / steward-owned deployments — no in-app
//                   updater at all.
//   windows-handoff git checkout on win32 — detached updater binary or the
//                   repo hand-off script owns the swap.
//   posix-handoff   git checkout on macOS/Linux — the repo posix hand-off
//                   script owns the swap.
//   manual          checkout with no staged updater — the user runs
//                   `hermes update` themselves.
//
// The mechanism is resolved ONCE from runtime facts (payload presence,
// platform, store flag) by resolveUpdaterMechanism — a pure function, unit
// tested — and every strategy reports it on the wire so the renderer can
// tailor copy per mechanism without probing the install shape itself.

export type UpdaterMechanism =
  | 'app-installer'
  | 'external'
  | 'windows-handoff'
  | 'posix-handoff'
  | 'manual'

/** The facts the mechanism dispatch keys on. Pure data — injectable for tests. */
export interface MechanismFacts {
  /** A bundled payload ships inside this artifact (sealed runtime). */
  isBundled: boolean
  isWindows: boolean
  /** This process is a Microsoft Store deployment. */
  isWindowsStore: boolean
}

/**
 * Resolve which mechanism owns updates for this install. Precedence mirrors
 * the historical checkUpdates/applyUpdates ladder in main.ts exactly:
 * payload-probe first, store-flag second, platform third. Behavior-preserving.
 */
export function resolveUpdaterMechanism(facts: MechanismFacts): UpdaterMechanism {
  if (facts.isBundled) {
    return facts.isWindows && !facts.isWindowsStore ? 'app-installer' : 'external'
  }

  return facts.isWindows ? 'windows-handoff' : 'posix-handoff'
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
  check(): Promise<UpdaterStatusWire>
  apply(opts: { stopSafeBlockers?: boolean }): Promise<UpdaterApplyResultWire>
}
