import { normalizeBasePath } from '@hermes/shared'

import { type BrowserHostGlobals, browserLaunchProfile } from './platform'
import { readKey, writeKey } from './storage'

interface BrowserSettingsOptions {
  basePath: string
  currentProfile: () => null | string
}

/** This page's hosted gateway, the same scope `createBrowserSettings` keys on. */
export function browserPageBasePath(): string {
  return normalizeBasePath((window as Window & BrowserHostGlobals).__HERMES_BASE_PATH__)
}

/** A profile not named is the one the serving gateway runs as, the profile its
 * REST answers for when a request names none. */
export function browserDefaultProjectDirKey(basePath: string, profile: string): string {
  const owner = profile.trim() || browserLaunchProfile() || 'default'

  return `hermes:browser:default-project-dir:${JSON.stringify([basePath || '/', owner])}`
}

/**
 * An older build keyed the folder on the profile the tab named, else 'default'.
 * A tab opened without `?profile=` names none, so on a gateway served as
 * another profile the folder that profile's tab chose sits under 'default'.
 * The first read on this build moves it to the served profile, unless that
 * profile already has its own. The marker keeps a later launch as another
 * profile from taking default's own folder.
 */
function adoptLegacyLaunchFolder(basePath: string): void {
  const marker = `hermes:browser:default-project-dir-adopted:${JSON.stringify([basePath || '/'])}`

  if (readKey(marker)) {
    return
  }

  writeKey(marker, '1')

  const launch = browserLaunchProfile()

  if (!launch || launch === 'default') {
    return
  }

  const legacyKey = browserDefaultProjectDirKey(basePath, 'default')
  const ownKey = browserDefaultProjectDirKey(basePath, launch)
  const legacy = readKey(legacyKey)?.trim()

  if (legacy && !readKey(ownKey)?.trim()) {
    writeKey(ownKey, legacy)
    writeKey(legacyKey, null)
  }
}

function folderKey(basePath: string, profile: string): string {
  adoptLegacyLaunchFolder(basePath)

  return browserDefaultProjectDirKey(basePath, profile)
}

/** The profile's own default folder, or '' when it never chose one. */
export function readBrowserDefaultProjectDir(basePath: string, profile: string): string {
  return readKey(folderKey(basePath, profile))?.trim() || ''
}

/** Rename moves the folder with the profile, or the lack of one. */
export function migrateBrowserFolderPref(basePath: string, from: string, to: string): void {
  const sourceKey = folderKey(basePath, from)
  const destKey = folderKey(basePath, to)

  if (sourceKey === destKey) {
    return
  }

  writeKey(destKey, readKey(sourceKey)?.trim() || null)
  writeKey(sourceKey, null)
}

export function dropBrowserFolderPref(basePath: string, profile: string): void {
  writeKey(folderKey(basePath, profile), null)
}

/**
 * The default project folder for new chats, kept in this origin's storage.
 *
 * Electron keeps it in userData; a browser tab has only localStorage. The
 * folder names a path on the machine the profile's workspace browses (this
 * host, or an SSH machine), so the key is scoped to the hosted gateway and the
 * profile. A write that storage refuses rejects, so Settings reports the
 * failure instead of a choice that is gone on the next load.
 */
export function createBrowserSettings({
  basePath,
  currentProfile
}: BrowserSettingsOptions): Window['hermesDesktop']['settings'] {
  const key = () => folderKey(basePath, currentProfile() ?? '')

  return {
    getDefaultProjectDir: async () => ({
      defaultLabel: 'Server workspace',
      dir: readKey(key())?.trim() || null,
      resolvedCwd: ''
    }),
    // The browser picks through the server folder browser (sessions-settings.tsx).
    pickDefaultProjectDir: async () => ({ canceled: true, dir: null }),
    setDefaultProjectDir: async (dir: null | string) => {
      const next = dir?.trim() || null

      if (next) {
        window.localStorage.setItem(key(), next)
      } else {
        window.localStorage.removeItem(key())
      }

      return { dir: next }
    }
  }
}
