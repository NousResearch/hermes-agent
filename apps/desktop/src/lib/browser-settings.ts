import { normalizeBasePath } from '@hermes/shared'

import type { BrowserHostGlobals } from './platform'
import { readKey, writeKey } from './storage'

interface BrowserSettingsOptions {
  basePath: string
  currentProfile: () => null | string
}

/** This page's hosted gateway, the same scope `createBrowserSettings` keys on. */
export function browserPageBasePath(): string {
  return normalizeBasePath((window as Window & BrowserHostGlobals).__HERMES_BASE_PATH__)
}

export function browserDefaultProjectDirKey(basePath: string, profile: string): string {
  return `hermes:browser:default-project-dir:${JSON.stringify([basePath || '/', profile.trim() || 'default'])}`
}

/** The profile's own default folder, or '' when it never chose one. */
export function readBrowserDefaultProjectDir(basePath: string, profile: string): string {
  return readKey(browserDefaultProjectDirKey(basePath, profile))?.trim() || ''
}

/** Rename keeps the folder with the profile. A destination that already chose
 * one keeps it; the old key is dropped either way so the name can be reused. */
export function migrateBrowserFolderPref(basePath: string, from: string, to: string): void {
  const sourceKey = browserDefaultProjectDirKey(basePath, from)
  const destKey = browserDefaultProjectDirKey(basePath, to)

  if (sourceKey === destKey) {
    return
  }

  const source = readKey(sourceKey)

  if (source === null) {
    return
  }

  const next = source.trim()

  if (next && readKey(destKey) === null) {
    writeKey(destKey, next)
  }

  writeKey(sourceKey, null)
}

export function dropBrowserFolderPref(basePath: string, profile: string): void {
  writeKey(browserDefaultProjectDirKey(basePath, profile), null)
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
  const key = () => browserDefaultProjectDirKey(basePath, currentProfile()?.trim() || 'default')

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
