import type { DesktopVersionInfo } from '@/global'

/**
 * Pure distribution-label selection for the version-details "Distribution"
 * row, shared by every surface that names an install's provenance.
 *
 * The install stamp carries three independent facts and the label keys on the
 * pair that answers "what did the user install":
 *  - `distribution`  the steward of a sealed tree (`desktop-app`) — gates the
 *                    desktop-app rows below.
 *  - `payload`       the artifact kind of the app itself: `bootstrap` is the
 *                    old-style installer shell (a .app/.exe over a managed
 *                    checkout), `bundled` ships the runtime in the artifact,
 *                    `light` carries no runtime.
 *  - `updateMechanism` who applies the next update (Store, App Installer,
 *                    electron-updater, …). `self` on a `bootstrap` artifact is
 *                    truthful forever: the shell never updates itself, the
 *                    managed checkout underneath does.
 * `installedByScript` refines an external/git install into "installed by
 * install.sh / install.ps1" vs a manual clone — those checkouts carry the
 * bootstrap installers' `.hermes-bootstrap-complete` receipt.
 */

/** Keys of the updates i18n section this resolver may return. */
export type DistributionLabelKey =
  | 'versionDetailsDistributionStore'
  | 'versionDetailsDistributionDesktopMsix'
  | 'versionDetailsDistributionDesktop'
  | 'versionDetailsDistributionDesktopInstaller'
  | 'versionDetailsDistributionSourceInstaller'
  | 'versionDetailsDistributionSource'

export interface DistributionLabelInput {
  distribution?: DesktopVersionInfo['distribution']
  updateMechanism?: DesktopVersionInfo['updateMechanism']
  payload?: DesktopVersionInfo['payload']
  installedByScript?: boolean
  /** Live provenance for non-stamped installs (git/unknown/…). A manual
   *  clone reports git; nothing known means no Distribution row. */
  source?: DesktopVersionInfo['source']
}

/**
 * The i18n key for the Distribution row, or null when nothing is known well
 * enough to show a row (no distribution stamp at all).
 */
export function distributionLabelKey(version: DistributionLabelInput): DistributionLabelKey | null {
  if (version.distribution === 'nix' || version.distribution === 'docker') {
    // Not localized facts; the component renders them verbatim.
    return null
  }

  if (version.updateMechanism === 'microsoft-store') {
    return 'versionDetailsDistributionStore'
  }

  if (version.distribution !== 'desktop-app') {
    // A plain source install: the bootstrap receipt separates the script
    // installers from a manual clone. No provenance at all shows no row —
    // an unproven build must not claim "Source".
    if (version.installedByScript) {
      return 'versionDetailsDistributionSourceInstaller'
    }
    return version.source && version.source !== 'unknown' ? 'versionDetailsDistributionSource' : null
  }

  if (version.payload === 'bootstrap') {
    return 'versionDetailsDistributionDesktopInstaller'
  }

  if (version.updateMechanism === 'app-installer') {
    return 'versionDetailsDistributionDesktopMsix'
  }

  return 'versionDetailsDistributionDesktop'
}
