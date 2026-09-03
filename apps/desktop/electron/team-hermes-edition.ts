import fs from 'node:fs'
import path from 'node:path'

export const TEAM_HERMES_APP_NAME = 'Team Hermes Desktop'
export const TEAM_HERMES_APP_USER_MODEL_ID = 'com.bkashjee.teamhermes.desktop'
export const TEAM_HERMES_PROTOCOL = 'team-hermes'
export const TEAM_HERMES_USER_DATA_DIRNAME = 'Team Hermes Desktop'
export const LEGACY_HERMES_USER_DATA_DIRNAME = 'Hermes'

export function teamHermesUserDataPath(appDataPath: string): string {
  return path.join(appDataPath, TEAM_HERMES_USER_DATA_DIRNAME)
}

export function legacyHermesUserDataPath(appDataPath: string): string {
  return path.join(appDataPath, LEGACY_HERMES_USER_DATA_DIRNAME)
}

/** Copy the legacy UI state only when Team Hermes has no established state. */
export function shouldMigrateLegacyUserData(legacyHasState: boolean, teamHasState: boolean): boolean {
  return legacyHasState && !teamHasState
}

/** Chromium process locks are instance-specific and must never be migrated. */
export function shouldCopyLegacyUserDataEntry(sourcePath: string): boolean {
  return !/^Singleton(?:Cookie|Lock|Socket)$/i.test(path.basename(sourcePath))
}

/**
 * Seed the separate Team Hermes profile from an existing Hermes desktop
 * profile. This is deliberately one-way and only runs while the destination
 * has no state, so later official Hermes launches cannot overwrite Team Hermes.
 */
export function migrateLegacyUserData(legacyUserData: string, teamUserData: string): boolean {
  const hasEntries = (candidate: string) => {
    try {
      return fs.existsSync(candidate) && fs.readdirSync(candidate).length > 0
    } catch {
      return false
    }
  }

  if (!shouldMigrateLegacyUserData(hasEntries(legacyUserData), hasEntries(teamUserData))) {
    return false
  }

  fs.mkdirSync(teamUserData, { recursive: true })
  fs.cpSync(legacyUserData, teamUserData, {
    filter: shouldCopyLegacyUserDataEntry,
    force: false,
    recursive: true
  })

  return true
}
