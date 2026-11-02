import assert from 'node:assert/strict'
import childProcess from 'node:child_process'
import fs from 'node:fs'
import { syncBuiltinESMExports } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { MacTargetHelper } from 'app-builder-lib/internal'
import { test, vi } from 'vitest'

import {
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
    const loose = path.join(payload, 'tools', 'chromium-1208', 'libEGL.dylib')
    fs.writeFileSync(loose, machoBuf())
    fs.writeFileSync(path.join(path.dirname(loose), 'README'), 'not Mach-O')
    for (const name of ['chromium_headless_shell-1208', 'uv-0.12.3-darwin-arm64']) {
      const other = path.join(payload, 'tools', name)
      fs.mkdirSync(other, { recursive: true })
      fs.writeFileSync(path.join(other, 'binary'), machoBuf())
    }
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

test('nested Chromium uses the same keychain certificate selector as the installed builder', async () => {
  const payload = tempRoot()
  const keychain = path.join(payload, 'builder.keychain')
  const hash = '0123456789ABCDEF0123456789ABCDEF01234567'
  const otherHash = 'F'.repeat(40)
  const listing = `  1) ${hash} "Developer ID Application: Signing Fixture (TEAM)"\n  2) ${otherHash} "Developer ID Application: Other Fixture (TEAM)"\n  2 valid identities found\n`
  let ready = false
  const packager = {
    config: {},
    resourceList: Promise.resolve([]),
    codeSigningInfo: {
      value: Promise.resolve().then(() => {
        ready = true
        return { keychainFile: keychain }
      })
    }
  }
  const readIdentity = (cmd, args) => {
    assert.ok(ready, 'identity discovery must await builder keychain initialization')
    assert.equal(path.basename(cmd), 'security')
    assert.equal(args[0], 'find-identity', 'never import credentials in this test')
    assert.equal(args.at(-1), keychain)
    return listing
  }
  // Exercise the installed builder's discovery and options, replacing only the
  // native process boundary. This proves selector parity, not native signing.
  const native = vi.spyOn(childProcess, 'execFile').mockImplementation((cmd, args, options, callback) => {
    callback(null, readIdentity(cmd, args), '')
    return undefined
  })
  syncBuiltinESMExports()
  vi.stubEnv('CSC_NAME', undefined)
  vi.stubEnv('CSC_IDENTITY_AUTO_DISCOVERY', 'true')
  try {
    const app = path.join(payload, 'tools', 'chromium-1208', 'Browser.app')
    fs.mkdirSync(app, { recursive: true })
    const entitlements = path.join(payload, 'entitlements.plist')
    fs.writeFileSync(entitlements, '<plist/>')
    const helper = new MacTargetHelper(packager)
    const nested = await resolveSigningIdentity(packager, readIdentity)
    const identity = await helper.findSigningIdentity('mac', undefined, keychain, false)
    const outer = await helper.buildSignOptions(app, identity, undefined, keychain, undefined, 'mac')
    const calls = []
    const signed = signNestedChromium(payload, {
      ...nested, entitlements, exec: (cmd, args) => calls.push([cmd, ...args])
    })
    assert.equal(signed.signed, 1)
    assert.equal(calls.length, 1)
    const args = calls[0]
    assert.equal(outer.identity, hash)
    assert.equal(args[args.indexOf('--sign') + 1], outer.identity)
    assert.equal(args[args.indexOf('--keychain') + 1], outer.keychain)
    assert.ok(args.includes('--deep') && args.includes('--timestamp') && args.includes('runtime'))
    assert.equal(args.at(-1), app)
    vi.stubEnv('CSC_NAME', 'Other Fixture')
    const qualified = await resolveSigningIdentity(packager, readIdentity)
    const selected = await helper.findSigningIdentity('mac', undefined, keychain, false)
    assert.equal(selected.hash, otherHash)
    assert.equal(qualified.identity, selected.hash, 'CSC_NAME is a qualifier, not a codesign selector')
  } finally {
    native.mockRestore()
    syncBuiltinESMExports()
    vi.unstubAllEnvs()
    fs.rmSync(payload, { recursive: true, force: true })
  }
})

test('identity preparation failures abort instead of selecting another keychain or skipping signing', async () => {
  vi.stubEnv('CSC_NAME', 'Signing Fixture')
  try {
    const importFailure = new Error('builder keychain initialization failed')
    const packager = { codeSigningInfo: { value: Promise.reject(importFailure) } }
    await assert.rejects(resolveSigningIdentity(packager, () => {
      assert.fail('must not discover identities after the builder keychain failed')
    }), error => error === importFailure)
    const keychain = path.join(os.tmpdir(), 'builder.keychain')
    packager.codeSigningInfo = { value: Promise.resolve({ keychainFile: keychain }) }
    const discoveryFailure = new Error('security could not read the keychain')
    await assert.rejects(resolveSigningIdentity(packager, () => { throw discoveryFailure }),
      error => error === discoveryFailure)
    vi.stubEnv('CSC_NAME', undefined)
    await assert.rejects(resolveSigningIdentity(packager, () => '  0 valid identities found\n'), /Developer ID/)
  } finally {
    vi.unstubAllEnvs()
  }
})
