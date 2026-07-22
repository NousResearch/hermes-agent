import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { loadOrCreateInstallationId, parseInstallationId, sshOwnershipId } from './desktop-installation'

const ID_A = '11111111-1111-4111-8111-111111111111'
const ID_B = '22222222-2222-4222-8222-222222222222'

function withTempDir(run) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-installation-'))

  try {
    return run(directory)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

test('parseInstallationId accepts only a version-4 UUID record', () => {
  assert.equal(parseInstallationId(JSON.stringify({ installationId: ID_A.toUpperCase() })), ID_A)
  assert.equal(parseInstallationId(JSON.stringify({ installationId: 'not-an-id' })), '')
  assert.equal(parseInstallationId('{}'), '')
  assert.equal(parseInstallationId('{'), '')
})

test('loadOrCreateInstallationId persists and reuses one installation ID', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_B),
      ID_A
    )
    assert.equal(fs.statSync(filePath).mode & 0o777, 0o600)
  }))

test('loadOrCreateInstallationId tightens an existing identity file', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(filePath, JSON.stringify({ installationId: ID_A }), { mode: 0o644 })
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_B),
      ID_A
    )

    if (process.platform !== 'win32') {
      assert.equal(fs.statSync(filePath).mode & 0o777, 0o600)
    }
  }))

test('loadOrCreateInstallationId replaces a malformed existing record', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(filePath, '{', { mode: 0o600 })
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(JSON.parse(fs.readFileSync(filePath, 'utf8')).installationId, ID_A)
  }))

test('loadOrCreateInstallationId replaces an existing symlink', () =>
  withTempDir(directory => {
    if (process.platform === 'win32') {
      return
    }

    const target = path.join(directory, 'target.json')
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(target, JSON.stringify({ installationId: ID_B }), { mode: 0o600 })
    fs.symlinkSync(target, filePath)
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(fs.lstatSync(filePath).isSymbolicLink(), false)
    assert.equal(JSON.parse(fs.readFileSync(target, 'utf8')).installationId, ID_B)
  }))

test('loadOrCreateInstallationId replaces a malformed destination without a repair lock', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(filePath, '{', { mode: 0o600 })
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(fs.existsSync(`${filePath}.lock`), false)
  }))

test('sshOwnershipId is stable, scoped, and does not disclose the UUID', () => {
  const global = sshOwnershipId(ID_A, '')
  assert.match(global, /^[0-9a-f]{32}$/)
  assert.equal(global, sshOwnershipId(ID_A, ''))
  assert.notEqual(global, sshOwnershipId(ID_A, 'worker'))
  assert.ok(!global.includes(ID_A.slice(0, 8)))
  assert.throws(() => sshOwnershipId('bad', ''))
})
