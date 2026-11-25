import { createHash } from 'node:crypto'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import zlib from 'node:zlib'

import { Arch, MacPackager, Packager, Platform, PublishManager } from 'app-builder-lib'
import { buildBlockMap, createBlockmap, validateConfiguration } from 'app-builder-lib/internal'
import jsYaml from 'js-yaml'
import { afterEach, expect, test, vi } from 'vitest'

import { createMacSigner } from './mac-sign.mjs'
import notarize, { COMBINED_NOTARY_BUDGET_MS, DMG_VERIFY_RESERVE_MS } from './notarize.mjs'
import notarizeDmg, { resolveDmgSigningIdentity } from './notarize-dmg.mjs'

const require = createRequire(import.meta.url)
const desktopDir = path.resolve(import.meta.dirname, '..')
const builderConfig = require(path.join(desktopDir, 'electron-builder.config.cjs'))

const IDENTITY_HASH = '0123456789ABCDEF0123456789ABCDEF01234567'
const SECOND_IDENTITY_HASH = 'FEDCBA9876543210FEDCBA9876543210FEDCBA98'
const IDENTITY_LIST = `  1) ${IDENTITY_HASH} "Developer ID Application: Nous Research (TEAM123456)"\n  1 valid identities found\n`
const TWO_IDENTITY_LIST =
  `  1) ${IDENTITY_HASH} "Developer ID Application: Other Org (OTHER12345)"\n` +
  `  2) ${SECOND_IDENTITY_HASH} "Developer ID Application: Nous Research (TEAM123456)"\n` +
  '  2 valid identities found\n'
const SUBMISSION_ID = '00000000-0000-4000-8000-000000000042'
const NOTARY_TIMEOUT = `Error: HTTPError(statusCode: nil, error: Error Domain=NSURLErrorDomain Code=-1001 "The request timed out." UserInfo={NSErrorFailingURLStringKey=https://appstoreconnect.apple.com/notary/v2/submissions/${SUBMISSION_ID}?})`
const CONNECTION_LOST = 'error is Error Domain=NSURLErrorDomain Code=-1005 "The network connection was lost."\nThe staple and validate action failed! Error 68.'
const MISSING_TICKET = Object.assign(
  new Error('CloudKit query for Hermes.dmg failed due to "Record not found".\nThe staple and validate action failed! Error 65.'),
  { code: 65 }
)

/** @type {string[]} */
const roots = []
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

function sha512Base64(bytes) {
  return createHash('sha512').update(bytes).digest('base64')
}

async function createBuilderFixture({
  keychainFile,
  platformOptions = { sign: () => {} },
  preStapleBytes = Buffer.from('udif-dmg-unsigned-payload-'.repeat(600))
} = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dmg-notarize-'))
  roots.push(root)
  const effectiveKeychain = keychainFile === undefined ? path.join(root, 'builder.keychain') : keychainFile
  const outDir = path.join(root, 'release')
  fs.mkdirSync(outDir, { recursive: true })
  fs.writeFileSync(
    path.join(root, 'package.json'),
    JSON.stringify({ name: 'hermes-fixture', version: '1.2.3', description: 'fixture', author: 'Test' })
  )

  const info = new Packager({
    projectDir: root,
    config: {
      appId: 'ai.hermes.fixture',
      productName: 'Hermes Fixture',
      directories: { output: outDir },
      mac: {
        publish: [{ provider: 'generic', url: 'https://updates.example/releases/darwin/stable/', channel: 'stable' }]
      }
    }
  })
  await info.validateConfig()
  info._appInfo = new (await import('app-builder-lib')).AppInfo(info, null)

  const publishManager = new PublishManager(info, { publish: 'never' })
  const packager = new MacPackager(info)
  Object.assign(packager.platformSpecificBuildOptions, platformOptions)
  packager.codeSigningInfo = { value: Promise.resolve({ keychainFile: effectiveKeychain }) }

  const dmgPath = path.join(outDir, 'HermesBundled-1.2.3-mac-arm64.dmg')
  const zipPath = path.join(outDir, 'HermesBundled-1.2.3-mac-arm64.zip')
  fs.writeFileSync(dmgPath, preStapleBytes)
  fs.writeFileSync(zipPath, Buffer.from('signed-app-zip-bytes'))

  const dmgTarget = { name: 'dmg', outDir, options: {} }
  const zipTarget = { name: 'zip', outDir, options: {} }
  const keyFile = path.join(root, 'api-key.p8')
  fs.writeFileSync(keyFile, 'test-key-file')

  return {
    root,
    outDir,
    keychainFile: effectiveKeychain,
    info,
    publishManager,
    packager,
    dmgPath,
    zipPath,
    dmgTarget,
    zipTarget,
    preStapleBytes,
    signedEnv: {
      APPLE_API_KEY: keyFile,
      APPLE_API_KEY_ID: 'KEY123',
      APPLE_API_ISSUER: 'ISSUER456'
    }
  }
}

/** Sign the fixture's app through createMacSigner, as electron-builder does before afterSign and the DMG hook. */
async function signAppWith(f, identity = IDENTITY_HASH) {
  const appPath = path.join(f.outDir, 'Hermes Fixture.app')
  fs.mkdirSync(path.join(appPath, 'Contents', 'Resources', 'agent-payload'), { recursive: true })
  const signer = createMacSigner(
    { hardenedRuntime: true, entitlements: 'ent.plist', entitlementsInherit: 'ent.plist', ignore: () => false },
    { env: {}, sign: vi.fn(async () => {}) }
  )
  await signer(
    { app: appPath, platform: 'darwin', type: 'distribution', identity, keychain: f.keychainFile },
    Object.assign(f.packager, { helper: { getOptionsForFile: async () => () => ({}) } })
  )
}

test('signed and notarized DMG signs with timestamp, retries notary/stapler/validate, verifies Gatekeeper, and updates blockmap and stable-mac.yml to post-staple bytes', async () => {
  const f = await createBuilderFixture()
  await signAppWith(f)
  const signedTrailer = Buffer.from('\n--CODESIGN-CMS-TIMESTAMP-SIGNATURE--')
  const ticketTrailer = Buffer.from('\n--APPLE-NOTARIZATION-STAPLED-TICKET--')
  const calls = []
  let notaryCalls = 0
  let stapleCalls = 0
  let validateCalls = 0
  let elapsed = 0

  const run = vi.fn(async (command, args) => {
    calls.push([command, ...args])
    if (command === 'security') return { stdout: IDENTITY_LIST, stderr: '' }
    if (command === 'codesign' && args.includes('--sign')) {
      fs.appendFileSync(args.at(-1), signedTrailer)
      return { stdout: '', stderr: '' }
    }
    if (command === 'codesign' && args.includes('--verify')) {
      return { stdout: '', stderr: '' }
    }
    if (command === 'xcrun' && args[0] === 'notarytool') {
      notaryCalls += 1
      if (notaryCalls === 1) {
        elapsed += 60_000
        throw Object.assign(new Error(NOTARY_TIMEOUT), { code: 1, stderr: NOTARY_TIMEOUT })
      }
      return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
    }
    if (command === 'xcrun' && args[0] === 'stapler' && args[1] === 'staple') {
      stapleCalls += 1
      if (stapleCalls === 1) throw MISSING_TICKET
      if (stapleCalls === 2) throw Object.assign(new Error(CONNECTION_LOST), { code: 68 })
      fs.appendFileSync(args.at(-1), ticketTrailer)
      return { stdout: 'The staple and validate action worked!', stderr: '' }
    }
    if (command === 'xcrun' && args[0] === 'stapler' && args[1] === 'validate') {
      validateCalls += 1
      if (validateCalls === 1) throw Object.assign(new Error(CONNECTION_LOST), { code: 68 })
      return { stdout: 'The validate action worked!', stderr: '' }
    }
    if (command === 'spctl') {
      return { stdout: '', stderr: `${f.dmgPath}: accepted\nsource=Notarized Developer ID\n` }
    }
    throw new Error(`Unexpected command: ${command} ${args.join(' ')}`)
  })
  const sleep = vi.fn(async ms => {
    elapsed += ms
  })
  const log = vi.fn()

  f.info.eventEmitter.on(
    'artifactBuildCompleted',
    event => notarizeDmg(event, { run, env: f.signedEnv, sleep, log, now: () => elapsed }),
    'user'
  )

  const preStapleUpdateInfo = await createBlockmap(f.dmgPath, f.dmgTarget, f.packager, path.basename(f.dmgPath))
  expect(preStapleUpdateInfo.size).toBe(f.preStapleBytes.length)

  await f.packager.emitArtifactBuildCompleted({
    file: f.dmgPath,
    safeArtifactName: path.basename(f.dmgPath),
    target: f.dmgTarget,
    arch: Arch.arm64,
    packager: f.packager,
    isWriteUpdateInfo: true,
    updateInfo: preStapleUpdateInfo
  })
  const zipUpdateInfo = await createBlockmap(f.zipPath, f.zipTarget, f.packager, path.basename(f.zipPath))
  await f.packager.emitArtifactBuildCompleted({
    file: f.zipPath,
    safeArtifactName: path.basename(f.zipPath),
    target: f.zipTarget,
    arch: Arch.arm64,
    packager: f.packager,
    isWriteUpdateInfo: true,
    updateInfo: zipUpdateInfo
  })
  await f.publishManager.awaitTasks()

  const finalBytes = fs.readFileSync(f.dmgPath)
  expect(finalBytes.equals(Buffer.concat([f.preStapleBytes, signedTrailer, ticketTrailer]))).toBe(true)

  expect(calls.filter(c => c[0] === 'ditto')).toHaveLength(0)
  expect(calls).toContainEqual([
    'codesign', '--force', '--sign', IDENTITY_HASH, '--timestamp', '--verbose', '--keychain', f.keychainFile, f.dmgPath
  ])
  expect(calls.filter(c => c[0] === 'codesign' && c[1] === '--verify' && c[2] === '--verbose=2' && c[3] === f.dmgPath)).toHaveLength(2)
  expect(calls.filter(c => c[0] === 'xcrun' && c[1] === 'notarytool' && c[2] === 'submit')).toHaveLength(1)
  expect(calls.filter(c => c[0] === 'xcrun' && c[1] === 'notarytool' && c[2] === 'wait' && c[3] === SUBMISSION_ID)).toHaveLength(1)
  expect(stapleCalls).toBe(3)
  expect(validateCalls).toBe(2)
  expect(calls).toContainEqual(['spctl', '-a', '-vv', '-t', 'open', '--context', 'context:primary-signature', f.dmgPath])

  const blockmap = JSON.parse(zlib.gunzipSync(fs.readFileSync(`${f.dmgPath}.blockmap`)).toString('utf8'))
  const coveredBytes = blockmap.files[0].sizes.reduce((sum, n) => sum + n, 0)
  expect(coveredBytes).toBe(finalBytes.length)
  expect(coveredBytes).not.toBe(f.preStapleBytes.length)
  const freshBlockmapPath = path.join(f.root, 'fresh.blockmap')
  const fresh = await buildBlockMap(f.dmgPath, 'gzip', freshBlockmapPath)
  expect(fs.readFileSync(`${f.dmgPath}.blockmap`).equals(fs.readFileSync(freshBlockmapPath))).toBe(true)

  const feed = jsYaml.load(fs.readFileSync(path.join(f.outDir, 'stable-mac.yml'), 'utf8'))
  const dmgEntry = feed.files.find(entry => entry.url === path.basename(f.dmgPath))
  expect(dmgEntry).toBeDefined()
  expect(dmgEntry.size).toBe(finalBytes.length)
  expect(dmgEntry.sha512).toBe(sha512Base64(finalBytes))
  expect(dmgEntry.sha512).toBe(fresh.sha512)
  expect(dmgEntry.sha512).not.toBe(sha512Base64(f.preStapleBytes))
  expect(feed.files[0].url).toBe(path.basename(f.zipPath))
})

test('signed local build without Apple notary credentials signs DMG and refreshes blockmap/updateInfo without notarizing', async () => {
  const f = await createBuilderFixture()
  await signAppWith(f)
  const signedTrailer = Buffer.from('\n--LOCAL-SIGN-ONLY--')
  const calls = []
  const run = vi.fn(async (command, args) => {
    calls.push([command, ...args])
    if (command === 'security') return { stdout: IDENTITY_LIST, stderr: '' }
    if (command === 'codesign' && args.includes('--sign')) {
      fs.appendFileSync(args.at(-1), signedTrailer)
    }
    return { stdout: '', stderr: '' }
  })
  const updateInfo = await createBlockmap(f.dmgPath, f.dmgTarget, f.packager, path.basename(f.dmgPath))
  await notarizeDmg(
    {
      file: f.dmgPath,
      target: f.dmgTarget,
      packager: f.packager,
      isWriteUpdateInfo: true,
      updateInfo
    },
    { run, env: {}, sleep: vi.fn(), log: vi.fn() }
  )
  const signedBytes = fs.readFileSync(f.dmgPath)
  expect(signedBytes.equals(Buffer.concat([f.preStapleBytes, signedTrailer]))).toBe(true)
  expect(calls.some(c => c[0] === 'xcrun')).toBe(false)
  expect(calls.some(c => c[0] === 'spctl')).toBe(false)
  expect(updateInfo.size).toBe(signedBytes.length)
  expect(updateInfo.sha512).toBe(sha512Base64(signedBytes))
  const blockmap = JSON.parse(zlib.gunzipSync(fs.readFileSync(`${f.dmgPath}.blockmap`)).toString('utf8'))
  expect(blockmap.files[0].sizes.reduce((sum, n) => sum + n, 0)).toBe(signedBytes.length)
})

test('unsigned local builds skip outer DMG signing even when valid Developer ID certificates exist in the keychain', async () => {
  const cases = [
    { name: 'CSC_IDENTITY_AUTO_DISCOVERY=false', env: { CSC_IDENTITY_AUTO_DISCOVERY: 'false' }, platformOptions: {}, keychainFile: null, stdout: TWO_IDENTITY_LIST },
    { name: 'mac.sign=null', env: {}, platformOptions: { sign: null }, keychainFile: null, stdout: TWO_IDENTITY_LIST },
    { name: 'mac.sign.identity=null', env: {}, platformOptions: { sign: { identity: null } }, keychainFile: null, stdout: TWO_IDENTITY_LIST },
    { name: 'ad-hoc CSC_NAME=-', env: { CSC_NAME: '-' }, platformOptions: {}, keychainFile: null, stdout: TWO_IDENTITY_LIST },
    { name: 'no Developer ID in default keychain', env: {}, platformOptions: {}, keychainFile: null, stdout: '  0 valid identities found\n' }
  ]
  for (const scenario of cases) {
    const f = await createBuilderFixture({ keychainFile: scenario.keychainFile, platformOptions: scenario.platformOptions })
    const run = vi.fn(async command => {
      if (command === 'security') return { stdout: scenario.stdout, stderr: '' }
      throw new Error(`Unexpected command for ${scenario.name}: ${command}`)
    })
    await notarizeDmg(
      { file: f.dmgPath, target: f.dmgTarget, packager: f.packager, isWriteUpdateInfo: true, updateInfo: { size: f.preStapleBytes.length, sha512: sha512Base64(f.preStapleBytes) } },
      { run, env: scenario.env, sleep: vi.fn(), log: vi.fn() }
    )
    expect(fs.readFileSync(f.dmgPath).equals(f.preStapleBytes), scenario.name).toBe(true)
    if (scenario.stdout === TWO_IDENTITY_LIST) {
      expect(run, scenario.name).not.toHaveBeenCalled()
    } else {
      expect(run.mock.calls.every(([cmd]) => cmd === 'security'), scenario.name).toBe(true)
    }
  }
})

test('DMG reuses the exact identity and keychain recorded by createMacSigner when multiple certificates exist', async () => {
  const f = await createBuilderFixture()
  const appPath = path.join(f.outDir, 'Hermes Fixture.app')
  fs.mkdirSync(path.join(appPath, 'Contents', 'Resources', 'agent-payload'), { recursive: true })
  const signer = createMacSigner(
    { hardenedRuntime: true, entitlements: 'ent.plist', entitlementsInherit: 'ent.plist', ignore: () => false },
    { env: {}, sign: vi.fn(async () => {}) }
  )
  await signer(
    { app: appPath, platform: 'darwin', type: 'distribution', identity: SECOND_IDENTITY_HASH, keychain: f.keychainFile },
    Object.assign(f.packager, { helper: { getOptionsForFile: async () => () => ({}) } })
  )

  const calls = []
  const run = vi.fn(async (command, args) => {
    calls.push([command, ...args])
    if (command === 'security') throw new Error('DMG hook must not re-query security find-identity after app signing')
    if (command === 'xcrun' && args[0] === 'notarytool') {
      return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
    }
    return { stdout: '', stderr: '' }
  })

  await notarizeDmg(
    { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
    { run, env: f.signedEnv, sleep: vi.fn(), log: vi.fn() }
  )
  expect(calls.some(c => c[0] === 'security')).toBe(false)
  expect(calls).toContainEqual([
    'codesign', '--force', '--sign', SECOND_IDENTITY_HASH, '--timestamp', '--verbose', '--keychain', f.keychainFile, f.dmgPath
  ])
})

test('DMG notarization shares the remaining combined budget after app notarization and reserves verify headroom', async () => {
  const f = await createBuilderFixture()
  await signAppWith(f)
  const appOutDir = path.join(f.root, 'mac-arm64')
  fs.mkdirSync(path.join(appOutDir, `${f.packager.appInfo.productFilename}.app`), { recursive: true })
  let elapsed = 0
  const calls = []
  const run = vi.fn(async (command, args) => {
    calls.push([command, ...args])
    if (command === 'security') return { stdout: IDENTITY_LIST, stderr: '' }
    if (command === 'xcrun' && args[0] === 'notarytool') {
      if (args[2].endsWith('.zip')) {
        elapsed += 210 * 60 * 1000
      }
      return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
    }
    return { stdout: '', stderr: '' }
  })

  await notarize(
    { electronPlatformName: 'darwin', appOutDir, packager: f.packager },
    { run, env: f.signedEnv, sleep: vi.fn(), log: vi.fn(), now: () => elapsed }
  )
  await notarizeDmg(
    { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
    { run, env: f.signedEnv, sleep: vi.fn(), log: vi.fn(), now: () => elapsed }
  )

  const dmgSubmit = calls.find(c => c[0] === 'xcrun' && c[1] === 'notarytool' && c[2] === 'submit' && c[3] === f.dmgPath)
  const expectedSeconds = Math.ceil((COMBINED_NOTARY_BUDGET_MS - 210 * 60 * 1000 - DMG_VERIFY_RESERVE_MS) / 1000)
  expect(dmgSubmit[dmgSubmit.indexOf('--timeout') + 1]).toBe(`${expectedSeconds}s`)

  elapsed = COMBINED_NOTARY_BUDGET_MS - DMG_VERIFY_RESERVE_MS
  const exhaustedRun = vi.fn(async command => {
    if (command === 'security') return { stdout: IDENTITY_LIST, stderr: '' }
    if (command === 'xcrun') throw new Error('notarytool must not run when combined budget is exhausted')
    return { stdout: '', stderr: '' }
  })
  await expect(
    notarizeDmg(
      { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
      { run: exhaustedRun, env: f.signedEnv, sleep: vi.fn(), log: vi.fn(), now: () => elapsed }
    )
  ).rejects.toThrow(/exceeded the wait budget/)
})

test('a build electron-builder left unsigned never signs its DMG with another keychain certificate', async () => {
  const cases = [
    { name: 'default keychain, no notary credentials', keychainFile: null, env: {}, expected: 'unsigned' },
    { name: 'default keychain, notary credentials', keychainFile: null, env: 'signed', expected: /without a Developer ID Application signing identity/ },
    { name: 'CSC_LINK keychain', keychainFile: undefined, env: {}, expected: /electron-builder did not sign the app/ }
  ]
  for (const scenario of cases) {
    const f = await createBuilderFixture({ keychainFile: scenario.keychainFile })
    const run = vi.fn(async command => {
      if (command === 'security') return { stdout: TWO_IDENTITY_LIST, stderr: '' }
      return { stdout: '', stderr: '' }
    })
    const done = notarizeDmg(
      { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
      { run, env: scenario.env === 'signed' ? f.signedEnv : scenario.env, sleep: vi.fn(), log: vi.fn() }
    )
    if (scenario.expected === 'unsigned') {
      await done
    } else {
      await expect(done, scenario.name).rejects.toThrow(scenario.expected)
    }
    expect(run.mock.calls.some(([command]) => command === 'codesign'), scenario.name).toBe(false)
    expect(fs.readFileSync(f.dmgPath).equals(f.preStapleBytes), scenario.name).toBe(true)
  }
})

test('a recorded app signature signs the DMG even when the signing settings would skip an unsigned build', async () => {
  for (const env of [{ CSC_IDENTITY_AUTO_DISCOVERY: 'false' }, { CSC_NAME: '-' }]) {
    const name = JSON.stringify(env)
    const f = await createBuilderFixture()
    await signAppWith(f)
    const calls = []
    const run = vi.fn(async (command, args) => {
      calls.push([command, ...args])
      if (command === 'xcrun' && args[0] === 'notarytool') {
        return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
      }
      return { stdout: '', stderr: '' }
    })
    await notarizeDmg(
      { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
      { run, env: { ...f.signedEnv, ...env }, sleep: vi.fn(), log: vi.fn() }
    )
    expect(calls, name).toContainEqual([
      'codesign', '--force', '--sign', IDENTITY_HASH, '--timestamp', '--verbose', '--keychain', f.keychainFile, f.dmgPath
    ])
    expect(calls.some(c => c[0] === 'xcrun' && c[1] === 'notarytool' && c[2] === 'submit'), name).toBe(true)
  }
})

test('the DMG notary budget also counts the packing and signing time before afterSign', async () => {
  const f = await createBuilderFixture()
  await signAppWith(f)
  const appOutDir = path.join(f.root, 'mac-arm64')
  fs.mkdirSync(path.join(appOutDir, `${f.packager.appInfo.productFilename}.app`), { recursive: true })
  let elapsed = 40 * 60 * 1000
  const calls = []
  const run = vi.fn(async (command, args) => {
    calls.push([command, ...args])
    if (command === 'xcrun' && args[0] === 'notarytool') {
      if (args[2].endsWith('.zip')) elapsed += 210 * 60 * 1000
      return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
    }
    return { stdout: '', stderr: '' }
  })
  const deps = { run, env: f.signedEnv, sleep: vi.fn(), log: vi.fn(), now: () => elapsed }

  await notarize({ electronPlatformName: 'darwin', appOutDir, packager: f.packager }, deps)
  await notarizeDmg({ file: f.dmgPath, target: f.dmgTarget, packager: f.packager }, deps)

  const dmgSubmit = calls.find(c => c[0] === 'xcrun' && c[1] === 'notarytool' && c[2] === 'submit' && c[3] === f.dmgPath)
  const expectedSeconds = Math.ceil((COMBINED_NOTARY_BUDGET_MS - 250 * 60 * 1000 - DMG_VERIFY_RESERVE_MS) / 1000)
  expect(dmgSubmit[dmgSubmit.indexOf('--timeout') + 1]).toBe(`${expectedSeconds}s`)
})

test('non-DMG artifacts and non-darwin packagers are ignored', async () => {
  const f = await createBuilderFixture()
  const run = vi.fn()
  await notarizeDmg({ file: `${f.dmgPath}.blockmap`, target: f.dmgTarget, packager: f.packager }, { run, env: f.signedEnv })
  await notarizeDmg({ file: f.zipPath, target: f.zipTarget, packager: f.packager }, { run, env: f.signedEnv })
  await notarizeDmg(
    { file: f.dmgPath, target: f.dmgTarget, packager: { ...f.packager, platform: Platform.LINUX } },
    { run, env: f.signedEnv }
  )
  expect(run).not.toHaveBeenCalled()
})

test('APPLE_NOTARY_PROFILE auth signs, notarizes, staples, and verifies DMG with --keychain-profile', async () => {
  const f = await createBuilderFixture()
  await signAppWith(f)
  const calls = []
  const run = vi.fn(async (command, args) => {
    calls.push([command, ...args])
    if (command === 'security') return { stdout: IDENTITY_LIST, stderr: '' }
    if (command === 'xcrun' && args[0] === 'notarytool') {
      return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
    }
    return { stdout: '', stderr: '' }
  })
  await notarizeDmg(
    { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
    { run, env: { APPLE_NOTARY_PROFILE: 'release-notary-profile' }, sleep: vi.fn(), log: vi.fn() }
  )
  const submitCall = calls.find(c => c[0] === 'xcrun' && c[1] === 'notarytool' && c[2] === 'submit')
  expect(submitCall).toEqual(expect.arrayContaining(['--keychain-profile', 'release-notary-profile', '--wait']))
  expect(calls).toContainEqual(['spctl', '-a', '-vv', '-t', 'open', '--context', 'context:primary-signature', f.dmgPath])
})

test('misconfigured signing or Gatekeeper/stapler validation failures fail loudly', async () => {
  const f = await createBuilderFixture({ keychainFile: null })
  const noIdentityRun = vi.fn(async () => ({ stdout: '  0 valid identities found\n', stderr: '' }))
  const inlineKey = '-----BEGIN PRIVATE KEY-----\ntest-inline-fixture\n-----END PRIVATE KEY-----'
  const beforeTempKeys = new Set(fs.readdirSync(os.tmpdir()).filter(n => n.startsWith('hermes-notary-')))
  await expect(
    notarizeDmg(
      { file: f.dmgPath, target: f.dmgTarget, packager: f.packager },
      {
        run: noIdentityRun,
        env: {
          APPLE_API_KEY: inlineKey,
          APPLE_API_KEY_ID: 'KEY123',
          APPLE_API_ISSUER: 'ISSUER456',
          CSC_IDENTITY_AUTO_DISCOVERY: 'false'
        },
        sleep: vi.fn(),
        log: vi.fn()
      }
    )
  ).rejects.toThrow(/without a Developer ID Application signing identity/)
  const afterTempKeys = new Set(fs.readdirSync(os.tmpdir()).filter(n => n.startsWith('hermes-notary-')))
  expect([...afterTempKeys].filter(n => !beforeTempKeys.has(n))).toEqual([])

  await expect(
    resolveDmgSigningIdentity(
      { codeSigningInfo: { value: Promise.resolve({ keychainFile: path.join(f.root, 'missing.keychain') }) } },
      { env: {} }
    )
  ).rejects.toThrow(/electron-builder did not sign the app/)

  for (const failAt of ['codesign-verify', 'stapler-validate-permanent', 'spctl']) {
    const f2 = await createBuilderFixture()
    await signAppWith(f2)
    const sleep = vi.fn()
    const run = vi.fn(async (command, args) => {
      if (command === 'security') return { stdout: IDENTITY_LIST, stderr: '' }
      if (command === 'codesign' && args.includes('--verify') && failAt === 'codesign-verify') {
        throw new Error('code object is not signed at all')
      }
      if (command === 'xcrun' && args[0] === 'notarytool') {
        return { stdout: JSON.stringify({ id: SUBMISSION_ID, status: 'Accepted' }), stderr: '' }
      }
      if (command === 'xcrun' && args[0] === 'stapler' && args[1] === 'validate' && failAt === 'stapler-validate-permanent') {
        throw Object.assign(new Error('The validate action failed! Error 65.'), { code: 65 })
      }
      if (command === 'spctl' && failAt === 'spctl') {
        throw Object.assign(new Error(`${f2.dmgPath}: rejected`), { code: 3 })
      }
      return { stdout: '', stderr: '' }
    })
    await expect(
      notarizeDmg(
        { file: f2.dmgPath, target: f2.dmgTarget, packager: f2.packager },
        { run, env: f2.signedEnv, sleep, log: vi.fn() }
      ),
      failAt
    ).rejects.toThrow()
    expect(sleep, failAt).not.toHaveBeenCalled()
  }
})

test('electron-builder config keeps dmg.sign off and validates artifactBuildCompleted against app-builder-lib schema', async () => {
  expect(builderConfig.dmg?.sign).not.toBe(true)
  await expect(
    validateConfiguration({
      ...builderConfig,
      afterSign: 'scripts/notarize.mjs',
      artifactBuildCompleted: 'scripts/notarize-dmg.mjs'
    })
  ).resolves.toBeUndefined()
})

test.runIf(process.platform === 'darwin')('macOS builder config wires both afterSign and outer DMG artifactBuildCompleted hooks', () => {
  expect(builderConfig.afterSign).toBe('scripts/notarize.mjs')
  expect(builderConfig.artifactBuildCompleted).toBe('scripts/notarize-dmg.mjs')
})
