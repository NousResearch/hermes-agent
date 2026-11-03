// msix-shared — native package versions come from the immutable build time.
// Stable keeps the Store quad; canary uses yy.mmdd.hh.mmss. The git-backed
// lookup is deterministic here because node:child_process.execFileSync is mocked.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { beforeEach, test, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFileSync: vi.fn(),
}))

// Re-import AFTER the mock so the module binds the mocked execFileSync.
const { execFileSync } = await import('node:child_process')
const msix = await import('../../../scripts/msix-shared.mjs')

// A fake desktop dir with just enough for appIdentity: product-identity.cjs
// (the bundled variant) + a package.json version.
function makeFakeDesktop(version) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-ident-'))
  fs.writeFileSync(
    path.join(dir, 'product-identity.cjs'),
    "module.exports = { store: false, light: false, displayName: 'Hermes', appId: 'com.nousresearch.hermes-bundled', channel: 'latest', artifactNamePascal: 'HermesBundled', msixAppIdWithOrg: 'NousResearch.HermesBundled' }\n"
  )
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name: 'hermes-desktop', version }))
  return dir
}

beforeEach(() => {
  execFileSync.mockReset()
})

test('explicit stable tag owns the package version, independent of checkout metadata', () => {
  const desktop = makeFakeDesktop('0.1.0')
  execFileSync.mockImplementation((cmd, args) => {
    if (args[0] === 'for-each-ref') return `${Math.floor(Date.UTC(2026, 7, 29, 1, 2, 3) / 1000)}\n`
    throw new Error(`unexpected git call ${cmd} ${args.join(' ')}`)
  })
  try {
    const { version, fileVersion } = msix.appIdentity(desktop, 'v0.27.1')
    assert.equal(version, '2026.5761.123.0')
    assert.equal(fileVersion, '0.27.1')
    assert.equal(msix.appIdentity(desktop, '').version, '0.1.0.0')
    assert.throws(() => msix.appIdentity(desktop, 'v0.27.1-invalid'), /release tag/)
  } finally {
    fs.rmSync(desktop, { recursive: true, force: true })
  }
})

test('stable and canary use their own native build-time quads', () => {
  const desktop = makeFakeDesktop('0.27.1')
  const epoch = Math.floor(Date.UTC(2026, 7, 29, 1, 2, 3) / 1000)
  execFileSync.mockImplementation((cmd, args) => {
    if (args[0] === 'for-each-ref') return `${epoch}\n`
    throw new Error(`unexpected git call ${cmd} ${args.join(' ')}`)
  })
  try {
    const stable = msix.appIdentity(desktop, 'v0.27.1')
    const canary = msix.appIdentity(desktop, 'v0.27.1+canary.20260829T010203Z')
    assert.equal(stable.version, '2026.5761.123.0')
    assert.equal(canary.version, '26.829.1.203')
    assert.equal(canary.fileVersion, '0.27.1+canary.20260829T010203Z')
  } finally {
    fs.rmSync(desktop, { recursive: true, force: true })
  }
})
