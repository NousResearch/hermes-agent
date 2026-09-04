// msix-shared — the 4-part MSIX version derivation (minutes-since-stable
// for canaries). The git-backed lookup is deterministic here because
// node:child_process.execFileSync is mocked; the math it feeds is the
// contract App Installer compares.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
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
    "module.exports = { store: false, light: false, displayName: 'Hermes', appId: 'com.nousresearch.hermes-bundled', channel: 'latest', appNamePascal: 'HermesBundled', msixAppIdWithOrg: 'NousResearch.HermesBundled' }\n"
  )
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name: 'hermes-desktop', version }))
  return dir
}

function gitMock(tags, stableEpoch) {
  execFileSync.mockImplementation((cmd, args) => {
    if (args[0] === 'tag') return `${tags.join('\n')}\n`
    if (args[0] === 'log') return `${stableEpoch}\n`
    throw new Error(`unexpected git call ${cmd} ${args.join(' ')}`)
  })
}

beforeEach(() => {
  execFileSync.mockReset()
})

test('stable tag keeps the 3-part version plus .0', () => {
  const desktop = makeFakeDesktop('0.27.1')
  const { version, fileVersion } = msix.appIdentity(desktop, 'v0.27.1')
  assert.equal(version, '0.27.1.0')
  assert.equal(fileVersion, '0.27.1')
})

test('canary version is tag base + minutes since the same-minor stable', () => {
  const desktop = makeFakeDesktop('0.27.1')
  // Stable v0.27.1 committed 2026-08-01T00:00:00Z; canary cut 2026-08-29T01:02:03Z.
  const stableEpoch = Math.floor(Date.UTC(2026, 7, 1) / 1000)
  gitMock(['v0.27.1', 'v0.27.2-canary.20260829010203'], stableEpoch)
  const { version, fileVersion } = msix.appIdentity(desktop, 'v0.27.2-canary.20260829010203')
  const expectedMinutes = Math.floor((Date.UTC(2026, 7, 29, 1, 2, 3) - Date.UTC(2026, 7, 1)) / 60000)
  assert.equal(version, `0.27.2.${expectedMinutes}`)
  // The artifact FILENAME carries the full canary string (appInfo.version),
  // not the 4-part feed version.
  assert.equal(fileVersion, '0.27.2-canary.20260829010203')
})

test('canaryBuildMinutesFor is pure minutes math', () => {
  const base = Date.UTC(2026, 7, 1) / 1000
  const minutes = msix.canaryBuildMinutesFor('v0.27.2-canary.20260829010203', base)
  assert.equal(minutes, Math.floor((Date.UTC(2026, 7, 29, 1, 2, 3) - Date.UTC(2026, 7, 1)) / 60000))
})

test('canaryBuildMinutesFor returns null for a stable tag', () => {
  assert.equal(msix.canaryBuildMinutesFor('v0.27.1', 0), null)
})

test('canaryBuildMinutesFor rejects a stable base older than 45 days', () => {
  const base = Date.UTC(2026, 5, 1) / 1000 // 2026-06-01
  assert.throws(
    () => msix.canaryBuildMinutesFor('v0.27.2-canary.20260829010203', base),
    /45 days|16-bit/i
  )
})

test('legacy 8-digit canary stamp still computes minutes (midnight of that day)', () => {
  const base = Date.UTC(2026, 7, 1) / 1000
  const minutes = msix.canaryBuildMinutesFor('v0.27.2-canary.20260801', base)
  assert.equal(minutes, 0)
})

test('buildAppInstaller pins the derived 4-part version everywhere', () => {
  const xml = msix.buildAppInstaller({
    baseUrl: 'https://updates.example.com',
    variantChannelPath: 'win32/canary',
    identityName: 'NousResearch.HermesBundled',
    version: '0.27.2.1234',
    bundleFilename: 'HermesBundled-0.27.2.1234-win.msixbundle'
  })
  assert.match(xml, /Uri="https:\/\/updates\.example\.com\/win32\/canary\/HermesBundled-0\.27\.2\.1234-win\.msixbundle"/)
  const versionCount = (xml.match(/Version="0\.27\.2\.1234"/g) || []).length
  assert.equal(versionCount, 2)
})
