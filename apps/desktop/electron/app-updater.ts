// app-updater.ts — electron-updater integration for bundled desktop installs.
//
// Bundled installs update through GitHub Releases: electron-updater reads
// latest*.yml from the release that the desktop-bundled-release workflow
// attached, downloads the new installer, and applies it. The swapped-in app
// carries the new runtime in its own resources (embedded mode), so there is
// no post-update install step at all.
//
// Source installs never reach this module. The callers gate on the install
// manifest first and fall through to the git-based update path.
//
// The decision helpers are pure so vitest covers them. The impure wrapper
// at the bottom lazy-loads electron-updater, because the module must not
// cost anything on thin builds.

import type { AppUpdater } from 'electron-updater'

export interface UpdaterGateFacts {
  stampHasPayload: boolean
  isPackaged: boolean
}

/**
 * True when this launch must use electron-updater for app updates.
 *
 * Both conditions are necessary:
 * - the build carries an embedded payload (an external build has no
 *   matching feed artifacts),
 * - the app is packaged (dev runs have no app-update.yml).
 *
 * This is a constant of the artifact, not of machine state. An eject
 * replaces the whole app with a source-built external one (no embedded
 * stamp), so no "ejected embedded install" state exists to gate on.
 */
export function shouldUseAppUpdater(facts: UpdaterGateFacts): boolean {
  return facts.stampHasPayload === true && facts.isPackaged === true
}

/**
 * Map an electron-updater check result to the renderer's update-check shape
 * (the shape hermes:updates:check already returns for the git path). The
 * renderer then needs no new states: `updateAvailable` plus `mechanism`
 * drive the existing UI.
 */
export function describeFeedCheck(
  current: string,
  info: { version?: string } | null | undefined,
  isUpdateAvailable?: boolean
): {
  supported: true
  mechanism: 'app-updater'
  channel: 'stable'
  currentVersion: string
  latestVersion: string | null
  latestTag: string | null
  updateAvailable: boolean
  fetchedAt: number
} {
  const latest = info && typeof info.version === 'string' ? info.version : null

  return {
    supported: true,
    mechanism: 'app-updater',
    // Bundled installs are locked to the stable channel; saying so here
    // lets every renderer surface pick release vocabulary without a
    // separate probe of the install manifest.
    channel: 'stable',
    currentVersion: current,
    latestVersion: latest,
    latestTag: latest ? `v${latest}` : null,
    // Prefer electron-updater's own semver verdict: a plain string compare
    // would offer a locally-newer dev build a downgrade.
    updateAvailable: isUpdateAvailable ?? (latest !== null && latest !== current),
    fetchedAt: Date.now()
  }
}

// ── impure wrapper ──────────────────────────────────────────────────────────

let cachedUpdater: AppUpdater | null = null

/**
 * Lazy singleton for electron-updater's autoUpdater. The require sits inside
 * the function so thin builds and tests never pay for the module load.
 * autoDownload stays off: the renderer asks the user before the download
 * starts (same consent model as the git path).
 */
export function getAutoUpdater(): AppUpdater {
  if (cachedUpdater) {
    return cachedUpdater
  }

  const { autoUpdater } = require('electron-updater') as { autoUpdater: AppUpdater }

  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = true
  cachedUpdater = autoUpdater

  return autoUpdater
}

/** Check the GitHub Releases feed. Returns the renderer-shaped result. */
export async function checkAppUpdate(currentVersion: string): Promise<ReturnType<typeof describeFeedCheck>> {
  const updater = getAutoUpdater()
  const result = await updater.checkForUpdates()

  return describeFeedCheck(currentVersion, result?.updateInfo, result?.isUpdateAvailable)
}

/**
 * Download the update, then quit and install. `onProgress` receives percent
 * values from electron-updater's download events. The returned promise
 * resolves after the download; quitAndInstall exits the process.
 */
export async function applyAppUpdate(onProgress?: (percent: number) => void): Promise<{ ok: true }> {
  const updater = getAutoUpdater()
  const handler = onProgress ? (p: { percent: number }) => onProgress(p.percent) : null

  if (handler) {
    updater.on('download-progress', handler)
  }

  // The listener must come off on failure too: the updater is a process-wide
  // singleton, and a retry after a failed download would stack a second
  // listener that fires ghost progress events.
  try {
    await updater.downloadUpdate()
  } finally {
    if (handler) {
      updater.removeListener('download-progress', handler)
    }
  }

  updater.quitAndInstall()

  return { ok: true }
}
