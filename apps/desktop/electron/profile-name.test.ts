import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeDesktopProfile } from './profile-name'

test('normalizeDesktopProfile accepts canonical desktop profile names', () => {
  assert.equal(normalizeDesktopProfile(' life_2 '), 'life_2')
  assert.equal(normalizeDesktopProfile('default'), 'default')
})

test('normalizeDesktopProfile canonicalizes mixed-case CLI profile input', () => {
  assert.equal(normalizeDesktopProfile(' Life_2 '), 'life_2')
  assert.equal(normalizeDesktopProfile('Default'), 'default')
})

test('normalizeDesktopProfile rejects malformed and absent values', () => {
  assert.equal(normalizeDesktopProfile('../life'), null)
  assert.equal(normalizeDesktopProfile('bad profile'), null)
  assert.equal(normalizeDesktopProfile(''), null)
  assert.equal(normalizeDesktopProfile(undefined), null)
})

test('normalizeDesktopProfile rejects every non-default CLI reserved name', () => {
  for (const profile of ['hermes', 'test', 'tmp', 'root', 'sudo', 'Hermes', 'SUDO']) {
    assert.equal(normalizeDesktopProfile(profile), null, profile)
  }
})
