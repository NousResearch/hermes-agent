import { isCanaryTag } from './feature-flags'
import type { InstallStamp } from './install-stamp'
import { COMMIT_BUILD_UPDATE_MESSAGE } from './updater/external'

export interface AppVersionInfo {
  appVersion: string
  baseVersion?: string
  channel?: 'stable' | 'canary' | null
  distance?: number
  commit?: string | null
  branch?: string | null
  source?: InstallStamp['source']
  distribution?: string
  updateMechanism?: InstallStamp['updateMechanism']
  dirty?: boolean
}

/** A release channel is an artifact identity, not a user preference. */
export function packagedReleaseChannel(stamp: Readonly<InstallStamp> | null): 'stable' | 'canary' | null {
  if (!stamp?.tag || stamp.source === 'commit-build') { return null }

  return isCanaryTag(stamp.tag) ? 'canary' : 'stable'
}

/** The backend may live on another machine and run a different release. */
export function appVersionInfo(stamp: Readonly<InstallStamp> | null, runtimeVersion: string, packageVersion: string): AppVersionInfo {
  if (!stamp) { return { appVersion: runtimeVersion, baseVersion: packageVersion } }

  return {
    appVersion: stamp.payload === 'bootstrap' ? runtimeVersion : stamp.displayVersion || packageVersion,
    baseVersion: stamp.baseVersion ?? undefined,
    channel: packagedReleaseChannel(stamp),
    distance: stamp.distance ?? undefined,
    commit: stamp.commit,
    branch: stamp.branch,
    source: stamp.source ?? undefined,
    distribution: stamp.distribution ?? undefined,
    updateMechanism: stamp.updateMechanism,
    dirty: stamp.dirty
  }
}

export function assertSourceUpdateChannel(stamp: Readonly<InstallStamp> | null): void {
  if (stamp?.source === 'commit-build') { throw new Error(COMMIT_BUILD_UPDATE_MESSAGE) }

  if (stamp && stamp.payload !== 'bootstrap') {
    throw new Error('This package has a fixed update channel. Install the other package to change channels.')
  }
}
