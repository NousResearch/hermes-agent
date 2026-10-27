import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, test } from 'vitest'

import { createDesktopProfilePreferences, type MissingLocalProfileCheck } from './desktop-profile'

// The preference file lives in Electron userData, which a Hermes reinstall
// does not touch; the profiles live under HERMES_HOME, which it can wipe.
let root = ''
let target = ''
let profilesRoot = ''
let logs: string[] = []

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-profile-missing-'))
  target = path.join(root, 'user-data', 'active-profile.json')
  profilesRoot = path.join(root, 'hermes-home', 'profiles')
  fs.mkdirSync(profilesRoot, { recursive: true })
  logs = []
})

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

function check(overrides: Partial<MissingLocalProfileCheck> = {}): MissingLocalProfileCheck {
  return {
    globalRemote: false,
    hasRemoteOverride: () => false,
    launchProfile: null,
    log: line => logs.push(line),
    profileExists: profile => fs.existsSync(path.join(profilesRoot, profile)),
    ...overrides
  }
}

test('a remembered local profile missing from the Hermes home falls back to default on disk', () => {
  createDesktopProfilePreferences(target).remember('hermes-contributor')

  const preferences = createDesktopProfilePreferences(target)

  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), ['hermes-contributor'])
  assert.equal(preferences.readActive(), 'default')
  // The next launch reads the repaired file, not a cached value.
  assert.equal(createDesktopProfilePreferences(target).readActive(), 'default')
  assert.match(logs.join('\n'), /last-used profile "hermes-contributor" no longer exists/)
  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), [])
})

test('existing, default and explicitly launched profiles are kept', () => {
  const preferences = createDesktopProfilePreferences(target)

  fs.mkdirSync(path.join(profilesRoot, 'work'))
  preferences.remember('work')
  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), [])
  assert.equal(preferences.readActive(), 'work')

  preferences.remember('default')
  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), [])
  assert.equal(preferences.readActive(), 'default')

  // `Hermes --profile typo` must still surface the backend's refusal.
  preferences.remember('typo')
  assert.deepEqual(preferences.forgetMissingLocalProfiles(check({ launchProfile: 'typo' })), [])
  assert.equal(preferences.readActive(), 'typo')
  assert.deepEqual(logs, [])
})

test('profiles served by a remote gateway are never judged by the local disk', () => {
  const preferences = createDesktopProfilePreferences(target)
  preferences.remember('remote-only')
  preferences.setDefault({ connectionId: 'remote-work', profile: 'remote-only' })

  assert.deepEqual(preferences.forgetMissingLocalProfiles(check({ globalRemote: true })), [])
  assert.deepEqual(
    preferences.forgetMissingLocalProfiles(check({ hasRemoteOverride: profile => profile === 'remote-only' })),
    []
  )
  assert.equal(preferences.readActive(), 'remote-only')
  assert.deepEqual(preferences.getDefault(), { connectionId: 'remote-work', profile: 'remote-only' })
})

test.each([
  ['the legacy local route', null, false],
  ['the registry local entry while the app is in remote mode', 'local', true]
])('an explicit default on %s to a missing profile is cleared', (_label, connectionId, globalRemote) => {
  const changes: unknown[] = []
  const preferences = createDesktopProfilePreferences(target, { onDefaultChanged: route => changes.push(route) })
  preferences.setDefault({ connectionId, profile: 'gone' })
  changes.length = 0

  assert.deepEqual(preferences.forgetMissingLocalProfiles(check({ globalRemote })), ['gone'])
  assert.equal(createDesktopProfilePreferences(target).getDefault(), null)
  assert.deepEqual(changes, [null])
})

test('a preference file that cannot be rewritten is logged without aborting startup', () => {
  const preferences = createDesktopProfilePreferences(target)
  preferences.remember('gone')
  fs.mkdirSync(`${target}.tmp`)

  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), [])
  assert.equal(preferences.readActive(), 'gone')
  assert.match(logs.join('\n'), /could not forget missing local profiles/)
})
