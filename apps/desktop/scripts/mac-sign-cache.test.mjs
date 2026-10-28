import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { expect, test, vi } from 'vitest'

import { createMacSignCache, insideBundle, looseMachOFiles, verifySigned } from './mac-sign-cache.mjs'
import { createMacSigner } from './mac-sign.mjs'

const MACHO = Buffer.from([0xcf, 0xfa, 0xed, 0xfe])
const SIGNED = Buffer.from('\nsigned-by-test')

/** @param {string} marker @returns {Buffer} */
const machO = marker => Buffer.concat([MACHO, Buffer.from(marker)])

/**
 * @typedef {object} SigningFixture
 * @property {string} root
 * @property {string} app
 * @property {(relative: string, bytes: Buffer | string) => string} put
 * @property {import('vitest').Mock<(file: string) => Promise<void>>} codesign
 * @property {import('vitest').Mock<(files: string[]) => Promise<Set<string>>>} verify
 * @property {Parameters<typeof createMacSignCache>[0] & { root: string }} opts
 * @property {() => void} cleanup
 */

/** @returns {SigningFixture} */
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mac-sign-cache-test-'))
  const entitlements = path.join(root, 'inherit.plist')
  fs.writeFileSync(entitlements, '<plist/>')
  const app = path.join(root, 'Hermes.app')
  /** @param {string} relative @param {Buffer | string} bytes @returns {string} */
  const put = (relative, bytes) => {
    const file = path.join(app, 'Contents', relative)
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fs.writeFileSync(file, bytes)
    return file
  }
  // Fake codesign appends a signature; fake verify accepts exactly those bytes.
  const codesign = vi.fn(async (/** @type {string} */ file) => {
    fs.writeFileSync(file, Buffer.concat([fs.readFileSync(file), SIGNED]))
  })
  const verify = vi.fn(async (/** @type {string[]} */ files) =>
    new Set(files.filter(file => fs.readFileSync(file).subarray(-SIGNED.length).equals(SIGNED))))
  const opts = { root: path.join(root, 'cache'), identity: 'ABC123', teamId: 'TEAMID1234',
    entitlements, codesign, verify }
  return { root, app, put, codesign, verify, opts,
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }) }
}

/**
 * @param {SigningFixture} f
 * @param {string[]} files
 * @param {Parameters<typeof createMacSignCache>[0]} [opts]
 * @returns {Promise<import('./mac-sign-cache.mjs').SignPlan>}
 */
async function signAll(f, files, opts = f.opts) {
  const cache = createMacSignCache(opts)
  if (!cache) throw new Error('cache disabled')
  const plan = await cache.prepare(files)
  await cache.sign(plan)
  return plan
}

test('loose payload binaries are found; nested bundles, the main executable and non-Mach-O are not', () => {
  const f = fixture()
  try {
    const loose = f.put('Resources/agent-payload/venv/lib/_core.so', machO('a'))
    f.put('Resources/agent-payload/README.txt', 'text')
    f.put('MacOS/Hermes', machO('main'))
    f.put('Frameworks/Electron Framework.framework/Versions/A/Electron Framework', machO('fw'))
    f.put('Resources/agent-payload/tools/chromium-1/Chromium.app/Contents/MacOS/Chromium', machO('c'))
    expect(looseMachOFiles(f.app, () => false)).toEqual([loose])
    expect(looseMachOFiles(f.app, file => file === loose)).toEqual([])
    expect(insideBundle(f.app, path.join(f.app, 'Contents/Frameworks/X.framework/Versions/A/X'))).toBe(true)
    expect(insideBundle(f.app, loose)).toBe(false)
  } finally {
    f.cleanup()
  }
})

test('an identical input restores the stored signed bytes without signing again', async () => {
  const f = fixture()
  try {
    const first = f.put('Resources/agent-payload/venv/lib/_core.so', machO('a'))
    await signAll(f, [first])
    const signed = fs.readFileSync(first)
    expect(f.codesign).toHaveBeenCalledTimes(1)

    const again = f.put('Resources/agent-payload/venv/lib/_core.so', machO('a'))
    const plan = await signAll(f, [again])
    expect(plan.restored).toBe(1)
    expect(f.codesign).toHaveBeenCalledTimes(1)
    expect(fs.readFileSync(again).equals(signed)).toBe(true)
  } finally {
    f.cleanup()
  }
})

test('duplicate inputs with the same name are signed once and all receive the signature', async () => {
  const f = fixture()
  try {
    const files = ['venv/lib/_core.so', 'uv-cache/wheels/_core.so', 'uv-cache/archive/_core.so']
      .map(relative => f.put(`Resources/agent-payload/${relative}`, machO('same')))
    const plan = await signAll(f, files)
    expect(plan.pending).toHaveLength(1)
    expect(f.codesign).toHaveBeenCalledTimes(1)
    for (const file of files) expect(fs.readFileSync(file).subarray(-SIGNED.length).equals(SIGNED)).toBe(true)
  } finally {
    f.cleanup()
  }
})

test('the file name, team, identity and entitlements are part of the key', async () => {
  const f = fixture()
  try {
    await signAll(f, [f.put('Resources/a/_core.so', machO('x'))])
    // Same bytes, different name: codesign derives the identifier from the name.
    await signAll(f, [f.put('Resources/a/_other.so', machO('x'))])
    expect(f.codesign).toHaveBeenCalledTimes(2)
    await signAll(f, [f.put('Resources/a/_core.so', machO('x'))], { ...f.opts, teamId: 'OTHERTEAM1' })
    await signAll(f, [f.put('Resources/a/_core.so', machO('x'))], { ...f.opts, identity: 'DEF456' })
    fs.writeFileSync(f.opts.entitlements, '<plist>changed</plist>')
    await signAll(f, [f.put('Resources/a/_core.so', machO('x'))])
    expect(f.codesign).toHaveBeenCalledTimes(5)
  } finally {
    f.cleanup()
  }
})

test('a cached entry the verifier rejects is evicted and the file is signed afresh', async () => {
  const f = fixture()
  try {
    const file = f.put('Resources/a/_core.so', machO('x'))
    await signAll(f, [file])
    const [policy] = fs.readdirSync(f.opts.root)
    const [key] = fs.readdirSync(path.join(f.opts.root, policy))
    fs.writeFileSync(path.join(f.opts.root, policy, key, 'signed'), machO('tampered'))

    const again = f.put('Resources/a/_core.so', machO('x'))
    const plan = await signAll(f, [again])
    expect(plan.restored).toBe(0)
    expect(f.codesign).toHaveBeenCalledTimes(2)
    expect(fs.readFileSync(again).subarray(-SIGNED.length).equals(SIGNED)).toBe(true)
  } finally {
    f.cleanup()
  }
})

test('a signature that fails verification is never stored', async () => {
  const f = fixture()
  try {
    const file = f.put('Resources/a/_core.so', machO('x'))
    const cache = createMacSignCache({ ...f.opts, verify: async () => new Set() })
    if (!cache) throw new Error('cache disabled')
    await expect(cache.sign(await cache.prepare([file]))).rejects.toThrow('failed verification')
    expect(fs.existsSync(f.opts.root) ? fs.readdirSync(f.opts.root).flatMap(p =>
      fs.readdirSync(path.join(f.opts.root, p)).filter(e => !e.startsWith('.tmp-'))) : []).toEqual([])
  } finally {
    f.cleanup()
  }
})

test('without a cache root, team or identity the cache is off', () => {
  const f = fixture()
  try {
    expect(createMacSignCache({ ...f.opts, root: undefined })).toBeNull()
    expect(createMacSignCache({ ...f.opts, teamId: undefined })).toBeNull()
    expect(createMacSignCache({ ...f.opts, identity: undefined })).toBeNull()
  } finally {
    f.cleanup()
  }
})

test('the signer hands cache-signed binaries to osx-sign as ignored and still seals everything else', async () => {
  const f = fixture()
  try {
    const loose = f.put('Resources/agent-payload/venv/lib/_core.so', machO('a'))
    const framework = f.put('Frameworks/X.framework/Versions/A/X', machO('fw'))
    const policyIgnored = f.put('Resources/policy.so', machO('policy'))
    const inheritedIgnored = f.put('Resources/inherited.so', machO('inherited'))
    const keychain = path.join(f.root, 'release.keychain-db')
    const sign = vi.fn(async (/** @type {import('@electron/osx-sign').SignOptions} */ opts) => {
      expect(fs.readFileSync(loose)).toEqual(Buffer.concat([machO('a'), SIGNED]))
      assert(opts.ignore instanceof Function)
      expect(opts.ignore(loose)).toBe(true)
      expect(opts.ignore(policyIgnored)).toBe(true)
      expect(opts.ignore(inheritedIgnored)).toBe(true)
      expect(opts.ignore(framework)).toBe(false)
      expect(opts.ignore(f.app)).toBe(false)
      expect(opts.identity).toBe(f.opts.identity)
      expect(opts.keychain).toBe(keychain)
      expect(opts.batchCodesignCalls).toBe(false)
    })
    const signer = createMacSigner(
      { entitlements: 'app.plist', entitlementsInherit: f.opts.entitlements, hardenedRuntime: true,
        ignore: file => file === policyIgnored },
      { env: { HERMES_MAC_SIGNATURE_CACHE: f.opts.root }, sign,
        team: async () => 'TEAMID1234',
        makeCache: options => createMacSignCache({ ...options, codesign: f.codesign, verify: f.verify }) })
    const packager = { helper: { getOptionsForFile: async () => () => ({}) } }
    /** @type {import('@electron/osx-sign').SignOptions} */
    const opts = { app: f.app, platform: 'darwin', identity: f.opts.identity, keychain,
      ignore: file => file === inheritedIgnored }
    await signer(opts, packager)
    expect(f.codesign).toHaveBeenCalledExactlyOnceWith(loose, [
      '--sign', f.opts.identity, '--force', '--keychain', keychain,
      '--timestamp', '--options', 'runtime', '--entitlements', f.opts.entitlements
    ])
    f.put('Resources/agent-payload/venv/lib/_core.so', machO('a'))
    await signer(opts, packager)
    expect(f.codesign).toHaveBeenCalledTimes(1)
    expect(sign).toHaveBeenCalledTimes(2)
  } finally {
    f.cleanup()
  }
})

test('the signer does not seal an app when cache signing fails verification', async () => {
  const f = fixture()
  try {
    f.put('Resources/agent-payload/venv/lib/_core.so', machO('a'))
    const sign = vi.fn(async (/** @type {import('@electron/osx-sign').SignOptions} */ _opts) => {})
    const signer = createMacSigner(
      { entitlementsInherit: f.opts.entitlements, hardenedRuntime: true, ignore: () => false },
      { env: { HERMES_MAC_SIGNATURE_CACHE: f.opts.root }, sign,
        team: async () => 'TEAMID1234',
        makeCache: options => createMacSignCache({ ...options, codesign: f.codesign, verify: async () => new Set() }) })
    const packager = { helper: { getOptionsForFile: async () => () => ({}) } }
    await expect(signer({ app: f.app, platform: 'darwin', identity: f.opts.identity }, packager))
      .rejects.toThrow('failed verification')
    expect(f.codesign).toHaveBeenCalledTimes(1)
    expect(sign).not.toHaveBeenCalled()
  } finally {
    f.cleanup()
  }
})

test.runIf(process.platform === 'darwin')('native verification accepts only a Developer ID, timestamped, hardened signature', async () => {
  const f = fixture()
  try {
    // An ad-hoc signature verifies, but has no team, timestamp or Developer ID.
    const file = path.join(f.root, 'true')
    fs.copyFileSync('/usr/bin/true', file)
    execFileSync('codesign', ['--force', '--sign', '-', '--options', 'runtime', file])
    expect(await verifySigned([file], 'TEAMID1234')).toEqual(new Set())
  } finally {
    f.cleanup()
  }
})
