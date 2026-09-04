import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import {
  chromiumRoots,
  isMachO,
  listLooseMachO,
  listTopLevelApps,
  parseDeveloperId,
  repairFrameworkLinks,
  resolveSigningIdentity,
  signNestedChromium
} from './sign-nested-chromium.mjs'

function tempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sign-chrome-'))
}

function machoBuf() {
  const buf = Buffer.alloc(16)
  buf.writeUInt32BE(0xfeedfacf, 0)
  return buf
}

test('isMachO accepts a 64-bit Mach-O magic and rejects text', () => {
  const dir = tempRoot()
  try {
    const bin = path.join(dir, 'a')
    const txt = path.join(dir, 'b.txt')
    fs.writeFileSync(bin, machoBuf())
    fs.writeFileSync(txt, 'not a binary')
    assert.equal(isMachO(bin), true)
    assert.equal(isMachO(txt), false)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('chromiumRoots only names chromium store entries', () => {
  const payload = tempRoot()
  try {
    fs.mkdirSync(path.join(payload, 'tools', 'chromium-1208'), { recursive: true })
    fs.mkdirSync(path.join(payload, 'tools', 'chromium_headless_shell-1208'), { recursive: true })
    fs.mkdirSync(path.join(payload, 'tools', 'uv-0.12.3-darwin-arm64'), { recursive: true })
    const roots = chromiumRoots(payload).map(p => path.basename(p)).sort()
    assert.deepEqual(roots, ['chromium-1208', 'chromium_headless_shell-1208'])
  } finally {
    fs.rmSync(payload, { recursive: true, force: true })
  }
})

test('listTopLevelApps finds .app dirs and listLooseMachO skips them', () => {
  const root = tempRoot()
  try {
    const app = path.join(root, 'Google Chrome for Testing.app')
    fs.mkdirSync(path.join(app, 'Contents', 'MacOS'), { recursive: true })
    fs.writeFileSync(path.join(app, 'Contents', 'MacOS', 'Chrome'), machoBuf())
    const loose = path.join(root, 'chrome-headless-shell')
    fs.writeFileSync(loose, machoBuf())
    assert.deepEqual(listTopLevelApps(root), [app])
    assert.deepEqual(listLooseMachO(root), [loose])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('repairFrameworkLinks turns a flattened Foo.framework/Foo into a symlink', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const fw = path.join(root, 'F.framework')
    const versioned = path.join(fw, 'Versions', 'A')
    fs.mkdirSync(path.join(versioned, 'Resources'), { recursive: true })
    fs.writeFileSync(path.join(versioned, 'F'), machoBuf())
    fs.writeFileSync(path.join(versioned, 'Resources', 'Info.plist'), '<plist/>')
    fs.writeFileSync(path.join(fw, 'F'), machoBuf())
    fs.mkdirSync(path.join(fw, 'Resources'))
    fs.writeFileSync(path.join(fw, 'Resources', 'Info.plist'), '<plist/>')

    const n = repairFrameworkLinks(root)
    assert.ok(n >= 2)
    assert.equal(fs.lstatSync(path.join(fw, 'F')).isSymbolicLink(), true)
    assert.equal(fs.lstatSync(path.join(fw, 'Versions', 'Current')).isSymbolicLink(), true)
    assert.equal(fs.readFileSync(path.join(fw, 'F')).equals(machoBuf()), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('parseDeveloperId takes the first Developer ID Application line', () => {
  const listing = `
  1) ABC "Apple Development: someone@example.com (TEAM)"
  2) DEF "Developer ID Application: Nous Research Inc (763K57MW7Z)"
  3) GHI "Developer ID Installer: Nous Research Inc (763K57MW7Z)"
`
  assert.equal(parseDeveloperId(listing), 'Developer ID Application: Nous Research Inc (763K57MW7Z)')
  assert.equal(parseDeveloperId('nothing here'), null)
})

test('signNestedChromium no-ops without an identity', () => {
  const payload = tempRoot()
  try {
    const r = signNestedChromium(payload, { identity: null, entitlements: path.join(payload, 'missing.plist') })
    assert.deepEqual(r, { signed: 0, repaired: 0, identity: null })
  } finally {
    fs.rmSync(payload, { recursive: true, force: true })
  }
})

test('signNestedChromium --deep signs the .app and file-signs loose Mach-O', () => {
  const payload = tempRoot()
  try {
    const entitlements = path.join(payload, 'entitlements.plist')
    fs.writeFileSync(entitlements, '<plist/>')
    const app = path.join(payload, 'tools', 'chromium-1208', 'Google Chrome for Testing.app')
    fs.mkdirSync(path.join(app, 'Contents', 'MacOS'), { recursive: true })
    fs.writeFileSync(path.join(app, 'Contents', 'MacOS', 'Chrome'), machoBuf())
    const looseDir = path.join(payload, 'tools', 'chromium_headless_shell-1208')
    fs.mkdirSync(looseDir, { recursive: true })
    const loose = path.join(looseDir, 'chrome-headless-shell')
    fs.writeFileSync(loose, machoBuf())
    const calls = []
    const r = signNestedChromium(payload, {
      identity: 'Developer ID Application: Test',
      entitlements,
      exec: (cmd, args) => {
        calls.push([cmd, ...args])
      }
    })
    assert.equal(r.signed, 2)
    assert.equal(calls.length, 2)
    const appCall = calls.find(c => c.includes(app))
    const fileCall = calls.find(c => c.includes(loose))
    assert.ok(appCall.includes('--deep'))
    assert.ok(!fileCall.includes('--deep'))
  } finally {
    fs.rmSync(payload, { recursive: true, force: true })
  }
})

test('resolveSigningIdentity reads the packager keychain', async () => {
  const seen = []
  const packager = {
    codeSigningInfo: {
      value: Promise.resolve({ keychainFile: '/tmp/builder.keychain' })
    }
  }
  const listing = '  1) DEF "Developer ID Application: Nous Research Inc (763K57MW7Z)"\n'
  const r = await resolveSigningIdentity(packager, (_cmd, args) => {
    seen.push(args)
    return listing
  })
  assert.equal(r.identity, 'Developer ID Application: Nous Research Inc (763K57MW7Z)')
  assert.equal(r.keychain, '/tmp/builder.keychain')
  assert.ok(seen[0].includes('/tmp/builder.keychain'))
})
