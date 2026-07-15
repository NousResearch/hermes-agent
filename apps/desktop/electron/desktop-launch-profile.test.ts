import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { splitHermesHomeRootAndProfile } from './backend-env'
import {
  preferredDesktopLaunchProfile,
  primaryBackendArgsFromLaunch,
  primaryProfileKeyFromLaunch,
  sanitizeProfileHint
} from './desktop-launch-profile'

test('sanitizeProfileHint accepts default and valid profile ids', () => {
  assert.equal(sanitizeProfileHint('Oracle'), 'oracle')
  assert.equal(sanitizeProfileHint('default'), 'default')
  assert.equal(sanitizeProfileHint('bad name'), null)
  assert.equal(sanitizeProfileHint(''), null)
})

test('preferredDesktopLaunchProfile prefers env hint over desktop preference', () => {
  assert.equal(
    preferredDesktopLaunchProfile({ profileHint: 'oracle', preference: 'coder' }),
    'oracle'
  )
  assert.equal(preferredDesktopLaunchProfile({ profileHint: null, preference: 'coder' }), 'coder')
  assert.equal(preferredDesktopLaunchProfile({ profileHint: null, preference: null }), null)
})

test('launch contract: inferred profiles/<name> selects primary and --profile when preference unset', () => {
  const parsed = splitHermesHomeRootAndProfile('/Users/test/.hermes/profiles/oracle', {
    pathModule: path.posix
  })
  const profileHint = sanitizeProfileHint(parsed.profileHint)
  const preference = null

  assert.equal(parsed.root, '/Users/test/.hermes')
  assert.equal(profileHint, 'oracle')
  assert.equal(primaryProfileKeyFromLaunch({ profileHint, preference }), 'oracle')
  assert.deepEqual(primaryBackendArgsFromLaunch({ profileHint, preference }), [
    '--profile',
    'oracle',
    'serve',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('launch contract: unset hint + unset preference keeps legacy launch (no --profile)', () => {
  assert.equal(primaryProfileKeyFromLaunch({ profileHint: null, preference: null }), 'default')
  assert.deepEqual(primaryBackendArgsFromLaunch({ profileHint: null, preference: null }), [
    'serve',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})
