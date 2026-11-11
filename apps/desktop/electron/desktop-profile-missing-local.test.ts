import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, test } from 'vitest'

import { createDesktopProfilePreferences, type MissingLocalProfileCheck } from './desktop-profile'
import { localProfileIsLive } from './local-profile-liveness'

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
    profileExists: profile => localProfileIsLive(path.dirname(profilesRoot), profile),
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
  fs.writeFileSync(path.join(profilesRoot, 'work', 'config.yaml'), '')
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

// hermes_constants.named_profile_is_live refuses both shapes, so `--profile`
// would fail every launch while the preference survived.
test.each([
  [
    'tombstoned by `hermes profile delete`',
    (home: string) => {
      fs.writeFileSync(path.join(home, 'config.yaml'), '')
      fs.mkdirSync(path.join(profilesRoot, '.deleted'))
      fs.writeFileSync(path.join(profilesRoot, '.deleted', 'ghost'), 'deleted\n')
    }
  ],
  ['a marker-less runtime leftover', (home: string) => fs.mkdirSync(path.join(home, 'logs'))]
])('a remembered profile directory that is %s is forgotten', (_label, shape) => {
  const home = path.join(profilesRoot, 'ghost')
  fs.mkdirSync(home)
  shape(home)
  const preferences = createDesktopProfilePreferences(target)
  preferences.remember('ghost')
  preferences.setDefault({ connectionId: null, profile: 'ghost' })

  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), ['ghost', 'ghost'])
  assert.equal(preferences.readActive(), 'default')
  assert.equal(preferences.getDefault(), null)
})

// Unprivileged Windows cannot create symlinks. Python's path predicates read a
// dangling or looping link as one absent path and keep checking the rest.
test.skipIf(process.platform === 'win32').each([
  [
    'its only identity marker is a dangling symlink',
    (home: string) => {
      fs.symlinkSync(path.join(root, 'missing-config.yaml'), path.join(home, 'config.yaml'))
    }
  ],
  [
    'a marker is a symlink loop next to a real .env',
    (home: string) => {
      fs.symlinkSync('config.yaml', path.join(home, 'config.yaml'))
      fs.writeFileSync(path.join(home, '.env'), '')
    }
  ],
  [
    'its tombstone path is a symlink loop',
    (home: string) => {
      fs.writeFileSync(path.join(home, 'config.yaml'), '')
      fs.mkdirSync(path.join(profilesRoot, '.deleted'))
      fs.symlinkSync('cloned', path.join(profilesRoot, '.deleted', 'cloned'))
    }
  ]
])('a profile where %s stays live, as in the CLI', (_label, shape) => {
  const home = path.join(profilesRoot, 'cloned')
  fs.mkdirSync(home)
  shape(home)
  const preferences = createDesktopProfilePreferences(target)
  preferences.remember('cloned')

  assert.deepEqual(preferences.forgetMissingLocalProfiles(check()), [])
  assert.equal(preferences.readActive(), 'cloned')
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
