import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hasWindowsPathPrefix, isExternalVenvHolder, isRabbitOwnedVenvDaemon } from './venv-holder-select'

const SCRIPTS = 'C:\\Rabbit\\venv\\Scripts'

test('matches the hindsight daemon shim (exe under venv Scripts + hindsight cmdline)', () => {
  assert.equal(
    isRabbitOwnedVenvDaemon(
      'C:\\Rabbit\\venv\\Scripts\\pythonw.exe',
      'C:\\Rabbit\\venv\\Scripts\\pythonw.exe -m hindsight_api.main --daemon --idle-timeout 300 --port 9177',
      SCRIPTS
    ),
    true
  )
})

test('Windows path prefix match is ordinal case-insensitive', () => {
  assert.equal(
    isRabbitOwnedVenvDaemon(
      'c:\\rabbit\\venv\\scripts\\python.exe',
      'python.exe -m hindsight_api.main --daemon',
      'C:\\Rabbit\\venv\\Scripts'
    ),
    true
  )
})

test('excludes external venv holders that are not the hindsight daemon', () => {
  // a user terminal running the rabbit CLI from the venv — must NOT be killed
  assert.equal(isRabbitOwnedVenvDaemon('C:\\Rabbit\\venv\\Scripts\\rabbit.exe', 'rabbit chat -q "hi"', SCRIPTS), false)
  // an unrelated python script using the venv interpreter
  assert.equal(
    isRabbitOwnedVenvDaemon('C:\\Rabbit\\venv\\Scripts\\python.exe', 'python C:\\tools\\import.py', SCRIPTS),
    false
  )
})

test('excludes exes outside the venv even when the cmdline mentions hindsight', () => {
  assert.equal(
    isRabbitOwnedVenvDaemon('C:\\Other\\pythonw.exe', 'pythonw -m hindsight_api.main --daemon', SCRIPTS),
    false
  )
})

test('prefix boundary: sibling dirs (ScriptsX) do not match', () => {
  assert.equal(hasWindowsPathPrefix('C:\\Rabbit\\venv\\ScriptsX\\python.exe', SCRIPTS), false)
  assert.equal(hasWindowsPathPrefix('C:\\Rabbit\\venv\\Scripts\\python.exe', SCRIPTS), true)
})

test('null/undefined fields never match', () => {
  assert.equal(isRabbitOwnedVenvDaemon(null, 'x', SCRIPTS), false)
  assert.equal(isRabbitOwnedVenvDaemon('C:\\Rabbit\\venv\\Scripts\\pythonw.exe', null, SCRIPTS), false)
  assert.equal(isRabbitOwnedVenvDaemon(undefined, undefined, SCRIPTS), false)
})

// --- isExternalVenvHolder (#62311) ------------------------------------------

test('matches the autostart gateway shim (rabbit.exe under venv Scripts)', () => {
  assert.equal(
    isExternalVenvHolder(
      'C:\\Rabbit\\venv\\Scripts\\rabbit.exe',
      '"C:\\Rabbit\\venv\\Scripts\\rabbit.exe" gateway run --external-supervisor',
      SCRIPTS
    ),
    true
  )
})

test('matches the dashboard scheduled task (python -m rabbit_cli / -m rabbit)', () => {
  assert.equal(
    isExternalVenvHolder(
      'C:\\Rabbit\\venv\\Scripts\\python.exe',
      '"C:\\Rabbit\\venv\\Scripts\\python.exe" -m rabbit_cli.main dashboard',
      SCRIPTS
    ),
    true
  )
  assert.equal(
    isExternalVenvHolder('C:\\Rabbit\\venv\\Scripts\\pythonw.exe', 'pythonw.exe -m rabbit serve', SCRIPTS),
    true
  )
})

test('never matches an unrelated process that merely borrows the venv interpreter', () => {
  // a user's own script running on the venv python — NOT Rabbit, must NOT be killed
  assert.equal(
    isExternalVenvHolder('C:\\Rabbit\\venv\\Scripts\\python.exe', 'python C:\\tools\\import.py', SCRIPTS),
    false
  )
  // hindsight daemon is selected by isRabbitOwnedVenvDaemon, not here
  assert.equal(
    isExternalVenvHolder('C:\\Rabbit\\venv\\Scripts\\pythonw.exe', 'pythonw -m hindsight_api.main --daemon', SCRIPTS),
    false
  )
})

test('never matches a process outside the venv, even with rabbit in the cmdline', () => {
  // an editor / shell whose command line mentions the install root (#62445 regression guard)
  assert.equal(
    isExternalVenvHolder('C:\\Windows\\System32\\cmd.exe', 'cmd /c cd C:\\Rabbit\\venv\\Scripts && dir', SCRIPTS),
    false
  )
  assert.equal(isExternalVenvHolder('C:\\Other\\rabbit.exe', 'rabbit gateway run', SCRIPTS), false)
})

test('sibling-dir and boundary safety for the external selector', () => {
  assert.equal(isExternalVenvHolder('C:\\Rabbit\\venv\\ScriptsX\\rabbit.exe', 'rabbit gateway run', SCRIPTS), false)
  assert.equal(isExternalVenvHolder(null, 'rabbit gateway run', SCRIPTS), false)
  assert.equal(isExternalVenvHolder('C:\\Rabbit\\venv\\Scripts\\rabbit.exe', null, SCRIPTS), false)
})
