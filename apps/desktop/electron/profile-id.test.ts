import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { serveBackendArgs } from './backend-command'
import { parseProfileDisplayName, readProfileIdCandidates, resolveProfileId } from './profile-id'

// A profile whose display_name differs from its directory name — the shape in
// the report (`profiles/smarthome/profile.yaml` -> `display_name: SmartHome`).
const SMARTHOME = [{ displayName: 'SmartHome', name: 'smarthome' }]
const EMAIL = [{ displayName: 'E-Mail', name: 'e-mail' }]
// display_name equal to the directory name: the behavior that must not change.
const UNCHANGED = [{ displayName: 'virtualisierung', name: 'virtualisierung' }]

const SERVE_TAIL = ['serve', '--host', '127.0.0.1', '--port', '0']

test('serveBackendArgs spawns the profile DIRECTORY name, never its display_name label', () => {
  assert.deepEqual(serveBackendArgs('SmartHome', SMARTHOME), ['--profile', 'smarthome', ...SERVE_TAIL])
})

test('serveBackendArgs resolves a hyphens-only-differing label to its directory', () => {
  assert.deepEqual(serveBackendArgs('E-Mail', EMAIL), ['--profile', 'e-mail', ...SERVE_TAIL])
})

test('serveBackendArgs is unchanged when display_name equals the directory name', () => {
  assert.deepEqual(serveBackendArgs('virtualisierung', UNCHANGED), ['--profile', 'virtualisierung', ...SERVE_TAIL])
  // A canonical id needs no candidate list at all (the hot path stays I/O-free).
  assert.deepEqual(serveBackendArgs('worker'), ['--profile', 'worker', ...SERVE_TAIL])
  // Unset profile keeps the legacy launch.
  assert.deepEqual(serveBackendArgs(), SERVE_TAIL)
  assert.deepEqual(serveBackendArgs(undefined, SMARTHOME), SERVE_TAIL)
})

test('resolveProfileId prefers the directory name and never guesses', () => {
  const candidates = [
    { displayName: 'SmartHome', name: 'smarthome' },
    { displayName: 'Kiosk', name: 'kiosk-a' },
    { displayName: 'Kiosk', name: 'kiosk-b' }
  ]

  // Exact id, case-folded directory, and unique label all land on the directory.
  assert.equal(resolveProfileId('smarthome', candidates), 'smarthome')
  assert.equal(resolveProfileId('SmartHome', candidates), 'smarthome')
  assert.equal(resolveProfileId('  SMARTHOME  ', candidates), 'smarthome')
  // A label that also spells a directory resolves to that directory, not the label.
  assert.equal(resolveProfileId('Kiosk', [...candidates, { displayName: 'Kiosk', name: 'kiosk' }]), 'kiosk')
  // An ambiguous label (two directories share it) is NOT resolved to a coin flip.
  assert.equal(resolveProfileId('Kiosk', candidates), 'Kiosk')
  // Unknown names keep today's behavior: the backend owns the error message.
  assert.equal(resolveProfileId('nope', candidates), 'nope')
  assert.equal(resolveProfileId('', candidates), '')
  assert.equal(resolveProfileId(null, candidates), '')
})

test('parseProfileDisplayName reads the label, never a nested key', () => {
  assert.equal(parseProfileDisplayName('description: x\ndisplay_name: SmartHome\n'), 'SmartHome')
  assert.equal(parseProfileDisplayName("display_name: 'E-Mail'\n"), 'E-Mail')
  assert.equal(parseProfileDisplayName('display_name: "Kiosk"\n'), 'Kiosk')
  assert.equal(parseProfileDisplayName('display_name: Kiosk # the box in the hall\n'), 'Kiosk')
  assert.equal(parseProfileDisplayName('ui_meta:\n  display_name: nested\n'), '')
  assert.equal(parseProfileDisplayName('description: only\n'), '')
  assert.equal(parseProfileDisplayName(''), '')
})

test('readProfileIdCandidates lists profile directories with their labels', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-profile-id-'))
  fs.mkdirSync(path.join(root, 'smarthome'))
  fs.writeFileSync(path.join(root, 'smarthome', 'profile.yaml'), 'display_name: SmartHome\n')
  fs.mkdirSync(path.join(root, 'plain'))
  fs.mkdirSync(path.join(root, '.hidden'))
  fs.writeFileSync(path.join(root, 'stray.yaml'), 'display_name: Stray\n')

  try {
    assert.deepEqual(readProfileIdCandidates(root), [
      { name: 'plain' },
      { displayName: 'SmartHome', name: 'smarthome' }
    ])
    // An unreadable root is advisory only — never a spawn-blocking throw.
    assert.deepEqual(readProfileIdCandidates(path.join(root, 'missing')), [])
  } finally {
    fs.rmSync(root, { force: true, recursive: true })
  }
})
