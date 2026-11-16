import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import { $defaultProfileRoute, noteDefaultProfileRoute } from '@/store/default-profile'
import { $connection } from '@/store/session'

import {
  browserFolderBasePath,
  browserPageBasePath,
  dropBrowserFolderPref,
  migrateBrowserFolderPref
} from './browser-settings'
import {
  dropBrowserStartupProfile,
  migrateBrowserStartupProfile,
  readBrowserStartupProfile
} from './browser-startup-profile'
import { hostHas } from './host-capabilities'

function hostedBasePath(): string | null {
  return hostHas('originScopedPrefs') ? browserPageBasePath() : null
}

// The gateway a profile was created, renamed or deleted on: the one the window is on unless the
// caller names it. A folder names a path on that gateway; the startup profile is this page's own.
function ownerOf(connectionId?: null | string): string {
  return (connectionId === undefined ? $connection.get()?.connectionId : connectionId)?.trim() || LOCAL_CONNECTION_ID
}

function namesProfile(profile: string | undefined, target: string): boolean {
  return (profile ?? '').trim() === target.trim()
}

export function migrateBrowserProfilePrefs(from: string, to: string, connectionId?: null | string): void {
  const basePath = hostedBasePath()
  const fromName = from.trim()
  const toName = to.trim()
  const owner = ownerOf(connectionId)

  if (basePath === null || !fromName || !toName || fromName === toName) {
    return
  }

  // A rename never lands on a live profile's name, so a folder or startup
  // profile already stored under the new name was left by a deleted profile.
  // The renamed profile's own prefs, or their absence, replace it.
  dropBrowserProfilePrefs(toName, owner)
  migrateBrowserFolderPref(browserFolderBasePath(owner), fromName, toName)

  if (owner !== LOCAL_CONNECTION_ID) {
    return
  }

  const movedStartup = migrateBrowserStartupProfile(basePath, fromName, toName)
  const current = $defaultProfileRoute.get()

  if (!current || !namesProfile(current.profile, fromName)) {
    return
  }

  // Same-window storage writes do not fire a storage event, so the mirror
  // has to follow the profile this window was actually applying.
  const saved = movedStartup ? readBrowserStartupProfile(basePath) : null

  noteDefaultProfileRoute(saved ?? { ...current, profile: toName })
}

export function dropBrowserProfilePrefs(profile: string, connectionId?: null | string): void {
  const basePath = hostedBasePath()
  const name = profile.trim()
  const owner = ownerOf(connectionId)

  if (basePath === null || !name) {
    return
  }

  dropBrowserFolderPref(browserFolderBasePath(owner), name)

  if (owner !== LOCAL_CONNECTION_ID) {
    return
  }

  dropBrowserStartupProfile(basePath, name)

  if (namesProfile($defaultProfileRoute.get()?.profile, name)) {
    noteDefaultProfileRoute(null)
  }
}
