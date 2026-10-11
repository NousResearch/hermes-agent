/**
 * Build identity for the launch-time recovery markers.
 *
 * The sandbox / GPU ladders clear a sticky fallback when the app changes
 * version. That signal is dead on a source or `hermes update` install: every
 * build reports the `0.0.0` placeholder, so a marker promoted by a bad launch
 * is never re-probed and the app keeps starting with the degraded launch flags
 * until a reinstall. The install stamp's commit (and its build time) does move
 * between those builds, so it is the identity to compare there.
 *
 * Pure and dependency-free so it can be unit-tested without Electron.
 */

import type { InstallStamp } from './install-stamp'

/** Versions that mean "no release identity" rather than a real release. */
const PLACEHOLDER_APP_VERSIONS = new Set(['', '0.0.0', 'v0.0.0', 'unknown'])

export function isPlaceholderAppVersion(appVersion: string | null | undefined): boolean {
  return PLACEHOLDER_APP_VERSIONS.has(String(appVersion ?? '').trim())
}

/**
 * The identity a sticky marker is re-probed against: the app version when it
 * carries one, and the install stamp's commit/build time when it does not.
 *
 * A real release version changes on every update, so it is the whole identity.
 * A placeholder version does not, and the stamp is what actually differs
 * between two source builds. A dev run with no stamp at all keeps the bare
 * version, i.e. today's behavior.
 */
export function launchBuildIdentity(options: {
  appVersion?: string | null
  installStamp?: Readonly<Pick<InstallStamp, 'commit' | 'builtAt' | 'dirty'>> | null
}): string {
  const appVersion = String(options.appVersion ?? '').trim()
  const stamp = options.installStamp ?? null
  const commit = String(stamp?.commit ?? '').trim()
  const builtAt = String(stamp?.builtAt ?? '').trim()

  if (!isPlaceholderAppVersion(appVersion)) {
    return appVersion
  }

  if (!commit && !builtAt) {
    return appVersion
  }

  const commitPart = commit ? `g${commit.slice(0, 12)}` : 'unknown'
  const dirtyPart = stamp?.dirty === true ? '-dirty' : ''
  const buildPart = builtAt ? `@${builtAt}` : ''

  return `${appVersion || '0.0.0'}+${commitPart}${dirtyPart}${buildPart}`
}

/**
 * True when a sticky marker predates this launch's build and the sandbox (or
 * GPU) deserves one more attempt instead of staying degraded forever.
 *
 * A marker written before build identities existed has no `build` to compare,
 * so the version check alone is all it gets — the same contract as before,
 * except that a placeholder version can never satisfy it. In that case the
 * marker is re-probed once, which is the only way a source install ever
 * recovers from a launch the user did not cause.
 */
export function launchMarkerNeedsReprobe(
  marker: { version?: string | null; build?: string | null } | null | undefined,
  options: { appVersion?: string | null; buildIdentity?: string | null }
): boolean {
  const appVersion = String(options.appVersion ?? '').trim()
  const buildIdentity = String(options.buildIdentity ?? '').trim()

  if (!marker || !appVersion) {
    return false
  }

  if (marker.version && marker.version !== appVersion) {
    return true
  }

  if (marker.build) {
    return Boolean(buildIdentity) && marker.build !== buildIdentity
  }

  return isPlaceholderAppVersion(appVersion)
}
