import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hasWindowsPathPrefix, isHermesOwnedVenvDaemon } from './venv-holder-select'

const SCRIPTS = 'C:\\Hermes\\venv\\Scripts'

test('matches the hindsight daemon shim (exe under venv Scripts + hindsight cmdline)', () => {
  assert.equal(
    isHermesOwnedVenvDaemon(
      'C:\\Hermes\\venv\\Scripts\\pythonw.exe',
      'C:\\Hermes\\venv\\Scripts\\pythonw.exe -m hindsight_api.main --daemon --idle-timeout 300 --port 9177',
      SCRIPTS
    ),
    true
  )
})

test('Windows path prefix match is ordinal case-insensitive', () => {
  assert.equal(
    isHermesOwnedVenvDaemon(
      'c:\\hermes\\venv\\scripts\\python.exe',
      'python.exe -m hindsight_api.main --daemon',
      'C:\\Hermes\\venv\\Scripts'
    ),
    true
  )
})

test('excludes external venv holders that are not the hindsight daemon', () => {
  // a user terminal running the hermes CLI from the venv — must NOT be killed
  assert.equal(isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\hermes.exe', 'hermes chat -q "hi"', SCRIPTS), false)
  // an unrelated python script using the venv interpreter
  assert.equal(
    isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\python.exe', 'python C:\\tools\\import.py', SCRIPTS),
    false
  )
})

test('excludes exes outside the venv even when the cmdline mentions hindsight', () => {
  assert.equal(
    isHermesOwnedVenvDaemon('C:\\Other\\pythonw.exe', 'pythonw -m hindsight_api.main --daemon', SCRIPTS),
    false
  )
})

test('prefix boundary: sibling dirs (ScriptsX) do not match', () => {
  assert.equal(hasWindowsPathPrefix('C:\\Hermes\\venv\\ScriptsX\\python.exe', SCRIPTS), false)
  assert.equal(hasWindowsPathPrefix('C:\\Hermes\\venv\\Scripts\\python.exe', SCRIPTS), true)
})

test('null/undefined fields never match', () => {
  assert.equal(isHermesOwnedVenvDaemon(null, 'x', SCRIPTS), false)
  assert.equal(isHermesOwnedVenvDaemon('C:\\Hermes\\venv\\Scripts\\pythonw.exe', null, SCRIPTS), false)
  assert.equal(isHermesOwnedVenvDaemon(undefined, undefined, SCRIPTS), false)
})
