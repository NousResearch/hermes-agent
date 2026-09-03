import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  legacyHermesUserDataPath,
  migrateLegacyUserData,
  shouldCopyLegacyUserDataEntry,
  shouldMigrateLegacyUserData,
  TEAM_HERMES_APP_NAME,
  TEAM_HERMES_APP_USER_MODEL_ID,
  TEAM_HERMES_PROTOCOL,
  teamHermesUserDataPath
} from './team-hermes-edition'

describe('Team Hermes edition identity', () => {
  it('uses an independent desktop identity and data directory', () => {
    expect(TEAM_HERMES_APP_NAME).toBe('Team Hermes Desktop')
    expect(TEAM_HERMES_APP_USER_MODEL_ID).toBe('com.bkashjee.teamhermes.desktop')
    expect(TEAM_HERMES_PROTOCOL).toBe('team-hermes')
    const appData = path.join('Users', 'Me', 'AppData', 'Roaming')
    expect(teamHermesUserDataPath(appData)).toBe(path.join(appData, 'Team Hermes Desktop'))
    expect(legacyHermesUserDataPath(appData)).toBe(path.join(appData, 'Hermes'))
  })

  it('migrates only when legacy state exists and Team Hermes is fresh', () => {
    expect(shouldMigrateLegacyUserData(true, false)).toBe(true)
    expect(shouldMigrateLegacyUserData(true, true)).toBe(false)
    expect(shouldMigrateLegacyUserData(false, false)).toBe(false)
  })

  it('excludes Chromium singleton locks from migration', () => {
    expect(shouldCopyLegacyUserDataEntry('C:\\state\\SingletonLock')).toBe(false)
    expect(shouldCopyLegacyUserDataEntry('C:\\state\\SingletonCookie')).toBe(false)
    expect(shouldCopyLegacyUserDataEntry('C:\\state\\Local Storage')).toBe(true)
  })

  it('copies real legacy state once and excludes process locks', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'team-hermes-migration-'))
    const legacy = path.join(root, 'Hermes')
    const team = path.join(root, 'Team Hermes Desktop')

    try {
      fs.mkdirSync(path.join(legacy, 'Local Storage'), { recursive: true })
      fs.writeFileSync(path.join(legacy, 'Local Storage', 'state.log'), 'appearance and profile state')
      fs.writeFileSync(path.join(legacy, 'connections.json'), '{"profiles":[]}')
      fs.writeFileSync(path.join(legacy, 'SingletonLock'), 'stale process lock')

      expect(migrateLegacyUserData(legacy, team)).toBe(true)
      expect(fs.readFileSync(path.join(team, 'connections.json'), 'utf8')).toBe('{"profiles":[]}')
      expect(fs.readFileSync(path.join(team, 'Local Storage', 'state.log'), 'utf8')).toBe(
        'appearance and profile state'
      )
      expect(fs.existsSync(path.join(team, 'SingletonLock'))).toBe(false)

      fs.writeFileSync(path.join(team, 'team-only.json'), '{}')
      fs.writeFileSync(path.join(legacy, 'official-later.json'), '{}')

      expect(migrateLegacyUserData(legacy, team)).toBe(false)
      expect(fs.existsSync(path.join(team, 'official-later.json'))).toBe(false)
    } finally {
      fs.rmSync(root, { force: true, recursive: true })
    }
  })
})
