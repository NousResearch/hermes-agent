import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hasWindowsPathPrefix, isExternalVenvHolder, isHermesOwnedVenvDaemon } from './venv-holder-select'

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

// --- isExternalVenvHolder (#62311) ------------------------------------------

test('matches the autostart gateway shim (hermes.exe under venv Scripts)', () => {
  assert.equal(
    isExternalVenvHolder(
      'C:\\Hermes\\venv\\Scripts\\hermes.exe',
      '"C:\\Hermes\\venv\\Scripts\\hermes.exe" gateway run --external-supervisor',
      SCRIPTS
    ),
    true
  )
})

test('matches the dashboard scheduled task (python -m hermes_cli / -m hermes)', () => {
  assert.equal(
    isExternalVenvHolder(
      'C:\\Hermes\\venv\\Scripts\\python.exe',
      '"C:\\Hermes\\venv\\Scripts\\python.exe" -m hermes_cli.main dashboard',
      SCRIPTS
    ),
    true
  )
  assert.equal(
    isExternalVenvHolder('C:\\Hermes\\venv\\Scripts\\pythonw.exe', 'pythonw.exe -m hermes serve', SCRIPTS),
    true
  )
})

test('never matches an unrelated process that merely borrows the venv interpreter', () => {
  // a user's own script running on the venv python — NOT Hermes, must NOT be killed
  assert.equal(
    isExternalVenvHolder('C:\\Hermes\\venv\\Scripts\\python.exe', 'python C:\\tools\\import.py', SCRIPTS),
    false
  )
  // hindsight daemon is selected by isHermesOwnedVenvDaemon, not here
  assert.equal(
    isExternalVenvHolder('C:\\Hermes\\venv\\Scripts\\pythonw.exe', 'pythonw -m hindsight_api.main --daemon', SCRIPTS),
    false
  )
})

test('never matches a process outside the venv, even with hermes in the cmdline', () => {
  // an editor / shell whose command line mentions the install root (#62445 regression guard)
  assert.equal(
    isExternalVenvHolder('C:\\Windows\\System32\\cmd.exe', 'cmd /c cd C:\\Hermes\\venv\\Scripts && dir', SCRIPTS),
    false
  )
  assert.equal(isExternalVenvHolder('C:\\Other\\hermes.exe', 'hermes gateway run', SCRIPTS), false)
})

test('sibling-dir and boundary safety for the external selector', () => {
  assert.equal(
    isExternalVenvHolder('C:\\Hermes\\venv\\ScriptsX\\hermes.exe', 'hermes gateway run', SCRIPTS),
    false
  )
  assert.equal(isExternalVenvHolder(null, 'hermes gateway run', SCRIPTS), false)
  assert.equal(isExternalVenvHolder('C:\\Hermes\\venv\\Scripts\\hermes.exe', null, SCRIPTS), false)
})
