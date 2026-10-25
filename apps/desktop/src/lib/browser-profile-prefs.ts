import { $defaultProfileRoute, noteDefaultProfileRoute } from '@/store/default-profile'

import { browserPageBasePath, dropBrowserFolderPref, migrateBrowserFolderPref } from './browser-settings'
import {
  dropBrowserStartupProfile,
  migrateBrowserStartupProfile,
  readBrowserStartupProfile
} from './browser-startup-profile'
import { isBrowserHostedDesktop } from './platform'

function hostedBasePath(): string | null {
  return isBrowserHostedDesktop() ? browserPageBasePath() : null
}

function namesProfile(profile: string | undefined, target: string): boolean {
  return (profile ?? '').trim() === target.trim()
}

export function migrateBrowserProfilePrefs(from: string, to: string): void {
  const basePath = hostedBasePath()
  const fromName = from.trim()
  const toName = to.trim()

  if (basePath === null || !fromName || !toName || fromName === toName) {
    return
  }

  migrateBrowserFolderPref(basePath, fromName, toName)

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

export function dropBrowserProfilePrefs(profile: string): void {
  const basePath = hostedBasePath()
  const name = profile.trim()

  if (basePath === null || !name) {
    return
  }

  dropBrowserFolderPref(basePath, name)
  dropBrowserStartupProfile(basePath, name)

  if (namesProfile($defaultProfileRoute.get()?.profile, name)) {
    noteDefaultProfileRoute(null)
  }
}
