/**
 * Tests for electron/cli-profile-override.cjs.
 *
 * Run with: node --test electron/cli-profile-override.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const { parseCliProfile } = require('./cli-profile-override.cjs')

test('returns null when --profile is missing', () => {
  assert.equal(parseCliProfile([]), null)
  assert.equal(parseCliProfile(['--other', 'value']), null)
  assert.equal(parseCliProfile(['electron', 'main.cjs']), null)
})

test('returns null when --profile has no value', () => {
  assert.equal(parseCliProfile(['--profile']), null)
  assert.equal(parseCliProfile(['--profile']), null)
})

test('returns null for empty or whitespace-only value', () => {
  assert.equal(parseCliProfile(['--profile', '']), null)
  assert.equal(parseCliProfile(['--profile', '   ']), null)
})

test('accepts "default" as a valid profile', () => {
  assert.equal(parseCliProfile(['--profile', 'default']), 'default')
})

test('accepts valid lowercase profile names', () => {
  assert.equal(parseCliProfile(['--profile', 'desktop']), 'desktop')
  assert.equal(parseCliProfile(['--profile', 'my-profile']), 'my-profile')
  assert.equal(parseCliProfile(['--profile', 'test_1']), 'test_1')
  assert.equal(parseCliProfile(['--profile', 'a']), 'a')
})

test('rejects uppercase profile names', () => {
  assert.equal(parseCliProfile(['--profile', 'Desktop']), null)
  assert.equal(parseCliProfile(['--profile', 'MY_PROFILE']), null)
})

test('rejects names starting with hyphen or underscore', () => {
  assert.equal(parseCliProfile(['--profile', '-desktop']), null)
  assert.equal(parseCliProfile(['--profile', '_desktop']), null)
})

test('rejects names longer than 64 characters', () => {
  const longName = 'a'.repeat(65)
  assert.equal(parseCliProfile(['--profile', longName]), null)
})

test('accepts names up to 64 characters', () => {
  const maxName = 'a'.repeat(64)
  assert.equal(parseCliProfile(['--profile', maxName]), maxName)
})

test('finds --profile among other arguments', () => {
  assert.equal(
    parseCliProfile(['electron', 'main.cjs', '--other', 'val', '--profile', 'work']),
    'work'
  )
})

test('trims whitespace from profile name', () => {
  assert.equal(parseCliProfile(['--profile', '  desktop  ']), 'desktop')
})

test('returns null for names with special characters', () => {
  assert.equal(parseCliProfile(['--profile', 'my profile']), null)
  assert.equal(parseCliProfile(['--profile', 'profile@home']), null)
  assert.equal(parseCliProfile(['--profile', 'a/b']), null)
})
