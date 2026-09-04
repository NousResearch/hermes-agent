// MSIX BUILD_NUMBER round-trip (plan item 3.2): scripts/msix-shared.mjs's
// appIdentity() derivation must equal the Version= the built manifest ships.
//
// The "built manifest" here is the REAL one the win32 lane packs: the repo's
// custom template (assets/msix-manifest.xml) run through app-builder-lib's
// substituteManifestMacros — the same helper MsixTarget.writeManifest uses —
// with the version macro fed by appIdentity(), exactly as
// scripts/gen-msix-manifest.mjs (the offline inspection twin of the build)
// does. The test then reads the Version= back out of the XML and compares.
//
// The git-backed stable lookup is deterministic here because
// node:child_process.execFileSync is mocked; the math it feeds is the
// contract App Installer and makeappx compare.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { beforeEach, test, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFileSync: vi.fn(),
}))

const { execFileSync } = await import('node:child_process')
const msix = await import('../../../scripts/msix-shared.mjs')
const require = createRequire(import.meta.url)

// The real helpers + template the build itself uses.
const { substituteManifestMacros } = require('../../../node_modules/app-builder-lib/dist/targets/win/winAppUtil.js')
const scriptsDir = path.dirname(fileURLToPath(import.meta.url))
const desktopDir = path.resolve(scriptsDir, '..')
const template = fs.readFileSync(path.join(desktopDir, 'assets', 'msix-manifest.xml'), 'utf8')

// A fake app dir with just enough for appIdentity (same shape as
// msix-shared.test.mjs's fixture).
function makeFakeDesktop(version) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-roundtrip-'))
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

// Substitute the template the way the build does: the version macro comes
// from appIdentity(); every other macro gets a deterministic placeholder —
// they are identity/packaging strings, not the contract under test here.
function buildManifest(appDir, tag) {
  const { version } = msix.appIdentity(appDir, tag)
  return { version, xml: substituteManifestMacros(template, (m) => (m === 'version' ? version : `test-${m}`)) }
}

function identityVersion(xml) {
  const identity = /<Identity\b[^>]*>/.exec(xml)
  assert.ok(identity, 'substituted manifest has no Identity element')
  const version = /Version="([^"]*)"/.exec(identity[0])
  assert.ok(version, 'Identity element carries no Version attribute')
  return version[1]
}

beforeEach(() => {
  execFileSync.mockReset()
})

test('stable tag: appIdentity derivation round-trips into the manifest Version', () => {
  const app = makeFakeDesktop('0.27.1')
  const { version, xml } = buildManifest(app, 'v0.27.1')
  assert.equal(version, '0.27.1.0')
  assert.equal(identityVersion(xml), version)
})

test('canary tag: minutes-since-stable build number round-trips into the manifest Version', () => {
  const app = makeFakeDesktop('0.27.1')
  // Stable v0.27.1 committed 2026-08-01T00:00:00Z; canary cut 2026-08-29T01:02:03Z.
  const stableEpoch = Math.floor(Date.UTC(2026, 7, 1) / 1000)
  gitMock(['v0.27.1', 'v0.27.2-canary.20260829010203'], stableEpoch)
  const { version, xml } = buildManifest(app, 'v0.27.2-canary.20260829010203')
  const expectedMinutes = Math.floor((Date.UTC(2026, 7, 29, 1, 2, 3) - Date.UTC(2026, 7, 1)) / 60000)
  assert.equal(version, `0.27.2.${expectedMinutes}`)
  assert.equal(identityVersion(xml), version)
})

test('manifest Version components are 16-bit (makeappx rejects anything larger)', () => {
  const app = makeFakeDesktop('0.27.1')
  const stableEpoch = Math.floor(Date.UTC(2026, 7, 1) / 1000)
  gitMock(['v0.27.1', 'v0.27.2-canary.20260829010203'], stableEpoch)
  const { xml } = buildManifest(app, 'v0.27.2-canary.20260829010203')
  for (const part of identityVersion(xml).split('.')) {
    const n = Number(part)
    assert.ok(Number.isInteger(n) && n >= 0 && n <= 65535, `component ${part} outside 16 bits`)
  }
})

test('a later canary stamps a strictly larger BUILD_NUMBER than an earlier one', () => {
  // The build number exists so Windows can order canaries on the same
  // base version; monotonicity across the stamp is the actual contract.
  const app = makeFakeDesktop('0.27.1')
  const stableEpoch = Math.floor(Date.UTC(2026, 7, 1) / 1000)
  const early = 'v0.27.2-canary.20260815080000'
  const late = 'v0.27.2-canary.20260829010203'
  gitMock([early, late, 'v0.27.1'], stableEpoch)
  const earlyBuild = Number(buildManifest(app, early).version.split('.')[3])
  const lateBuild = Number(buildManifest(app, late).version.split('.')[3])
  assert.ok(lateBuild > earlyBuild, `${lateBuild} must exceed ${earlyBuild}`)
})
