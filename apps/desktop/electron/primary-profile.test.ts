import assert from 'node:assert/strict'
import path from 'node:path'

import { test, vi } from 'vitest'

import { serveBackendArgs } from './backend-command'
import {
  createPrimaryProfileOwner,
  parseDesktopProfilePreference,
  resolveEffectivePrimaryProfile
} from './primary-profile'

const ROOT = path.join(path.parse(process.cwd()).root, 'tmp', 'hermes-home')

test('explicit desktop preference wins without reading legacy state', () => {
  const readFile = vi.fn(() => 'legacy')

  assert.equal(resolveEffectivePrimaryProfile({ desktopProfile: ' life ', hermesHome: ROOT, readFile }), 'life')
  assert.equal(readFile.mock.calls.length, 0)
})

test('invalid explicit desktop preference fails without reading legacy state', () => {
  const readFile = vi.fn(() => 'life')

  assert.throws(
    () => resolveEffectivePrimaryProfile({ desktopProfile: 'root', hermesHome: ROOT, readFile }),
    /Invalid profile name in Desktop preference/
  )
  assert.equal(readFile.mock.calls.length, 0)
})

test('malformed or wrongly typed persisted Desktop preferences fail closed', () => {
  assert.throws(() => parseDesktopProfilePreference('{'), SyntaxError)
  assert.throws(() => parseDesktopProfilePreference('{"profile":42}'), /Invalid profile name in Desktop preference/)
  assert.equal(parseDesktopProfilePreference('{"profile":null}'), null)
  assert.equal(parseDesktopProfilePreference('{"profile":"  "}'), null)
})

test('profile-scoped HERMES_HOME mirrors the CLI profile override', () => {
  const readFile = vi.fn(() => 'legacy')
  const hermesHome = path.join(ROOT, 'profiles', 'coder')

  assert.equal(resolveEffectivePrimaryProfile({ desktopProfile: null, hermesHome, readFile }), 'coder')
  assert.equal(readFile.mock.calls.length, 0)
})

test('invalid profile-scoped HERMES_HOME fails before backend spawn', () => {
  assert.throws(
    () =>
      resolveEffectivePrimaryProfile({
        desktopProfile: null,
        hermesHome: path.join(ROOT, 'profiles', 'sudo'),
        readFile: vi.fn(() => 'life')
      }),
    /Invalid profile name in HERMES_HOME/
  )
})

test('legacy active_profile owns an unpinned root launch', () => {
  const readFile = vi.fn(() => ' life\n')

  assert.equal(resolveEffectivePrimaryProfile({ desktopProfile: null, hermesHome: ROOT, readFile }), 'life')
  assert.deepEqual(readFile.mock.calls, [[path.join(ROOT, 'active_profile'), 'utf8']])
})

test('missing or empty legacy state falls back to default', () => {
  const missing = Object.assign(new Error('missing'), { code: 'ENOENT' })

  assert.equal(
    resolveEffectivePrimaryProfile({
      desktopProfile: null,
      hermesHome: ROOT,
      readFile: () => {
        throw missing
      }
    }),
    'default'
  )
  assert.equal(
    resolveEffectivePrimaryProfile({ desktopProfile: null, hermesHome: ROOT, readFile: () => '  ' }),
    'default'
  )
})

test('non-missing active_profile read failures surface to desktop boot', () => {
  const denied = Object.assign(new Error('permission denied'), { code: 'EACCES' })

  assert.throws(
    () =>
      resolveEffectivePrimaryProfile({
        desktopProfile: null,
        hermesHome: ROOT,
        readFile: () => {
          throw denied
        }
      }),
    error => error === denied
  )
})

test('non-empty invalid legacy state fails instead of silently booting default', () => {
  assert.throws(
    () => resolveEffectivePrimaryProfile({ desktopProfile: null, hermesHome: ROOT, readFile: () => '../bad' }),
    /Invalid profile name in active_profile/
  )
})

test('reserved legacy state fails instead of reaching backend spawn', () => {
  assert.throws(
    () => resolveEffectivePrimaryProfile({ desktopProfile: null, hermesHome: ROOT, readFile: () => 'root' }),
    /Invalid profile name in active_profile/
  )
})

test('primary owner stays frozen until the backend lifecycle resets', () => {
  let resolved = 'life'
  const owner = createPrimaryProfileOwner(() => resolved)

  assert.equal(owner.get(), 'life')
  resolved = 'work'
  assert.equal(owner.get(), 'life')

  owner.reset()
  assert.equal(owner.get(), 'work')
})

test('local respawns stay pinned when the legacy sticky profile changes', () => {
  let stickyProfile = 'life'
  const owner = createPrimaryProfileOwner(() => stickyProfile)

  assert.deepEqual(serveBackendArgs(owner.get()), ['--profile', 'life', 'serve', '--host', '127.0.0.1', '--port', '0'])

  stickyProfile = 'work'
  assert.deepEqual(serveBackendArgs(owner.get()), ['--profile', 'life', 'serve', '--host', '127.0.0.1', '--port', '0'])
})
