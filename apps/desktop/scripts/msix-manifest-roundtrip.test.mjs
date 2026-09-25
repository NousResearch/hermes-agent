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
    "module.exports = { store: false, light: false, displayName: 'Hermes', appId: 'com.nousresearch.hermes-bundled', channel: 'latest', artifactNamePascal: 'HermesBundled', msixAppIdWithOrg: 'NousResearch.HermesBundled' }\n"
  )
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name: 'hermes-desktop', version }))
  return dir
}

function gitMock(epoch) {
  execFileSync.mockImplementation((cmd, args) => {
    if (args[0] === 'for-each-ref') return `${epoch}\n`
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

test('stable tag: the build-time quad round-trips into the manifest Version', () => {
  const app = makeFakeDesktop('0.27.1')
  gitMock(Math.floor(Date.UTC(2026, 7, 29, 1, 2, 3) / 1000))
  const { version, xml } = buildManifest(app, 'v0.27.1')
  assert.equal(version, '2026.5761.123.0')
  assert.equal(identityVersion(xml), version)
})

test('canary tag: the build-time quad round-trips into the manifest Version', () => {
  const app = makeFakeDesktop('0.27.1')
  const { version, xml } = buildManifest(app, 'v0.27.1+canary.20260829T010203Z')
  assert.equal(version, '26.829.1.203')
  assert.equal(identityVersion(xml), version)
})

test('manifest Version components are 16-bit (makeappx rejects anything larger)', () => {
  const app = makeFakeDesktop('0.27.1')
  const { xml } = buildManifest(app, 'v0.27.1+canary.20260829T010203Z')
  for (const part of identityVersion(xml).split('.')) {
    const n = Number(part)
    assert.ok(Number.isInteger(n) && n >= 0 && n <= 65535, `component ${part} outside 16 bits`)
  }
})

test('a later build stamps a strictly larger quad than an earlier one', () => {
  const app = makeFakeDesktop('0.27.1')
  const early = Number(buildManifest(app, 'v0.27.1+canary.20260815T080000Z').version.split('.')[1])
  const late = Number(buildManifest(app, 'v0.27.1+canary.20260829T010203Z').version.split('.')[1])
  assert.ok(late > early, `${late} must exceed ${early}`)
})
