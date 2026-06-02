const assert = require('node:assert/strict')
const test = require('node:test')

const { buildPosixUpdateArgs, chooseDesktopUpdateStrategy } = require('./update-strategy.cjs')

test('macOS/Linux use POSIX in-app update even when a staged updater exists', () => {
  assert.equal(
    chooseDesktopUpdateStrategy({ isWindows: false, updater: '/Users/me/.hermes/hermes-setup' }),
    'posix-in-app'
  )
  assert.equal(chooseDesktopUpdateStrategy({ isWindows: false, updater: null }), 'posix-in-app')
})

test('Windows keeps staged-updater handoff when available', () => {
  assert.equal(
    chooseDesktopUpdateStrategy({ isWindows: true, updater: 'C:\\Users\\me\\AppData\\Local\\hermes\\hermes-setup.exe' }),
    'staged-updater'
  )
})

test('Windows without staged updater surfaces manual command', () => {
  assert.equal(chooseDesktopUpdateStrategy({ isWindows: true, updater: null }), 'manual')
})

test('POSIX update args pin the configured update branch', () => {
  assert.deepEqual(buildPosixUpdateArgs('main'), ['update', '--yes', '--branch', 'main'])
  assert.deepEqual(buildPosixUpdateArgs('bb/gui'), ['update', '--yes', '--branch', 'bb/gui'])
  assert.deepEqual(buildPosixUpdateArgs('  release/test  '), ['update', '--yes', '--branch', 'release/test'])
})

test('POSIX update args fall back to bare update when branch is absent', () => {
  assert.deepEqual(buildPosixUpdateArgs(''), ['update', '--yes'])
  assert.deepEqual(buildPosixUpdateArgs(null), ['update', '--yes'])
})
