// product-identity.cjs is the single derivation of the desktop product
// identity; electron/product-identity.ts re-exports it. These tests hold
// the identity contract: the TS accessor resolves to the .cjs object, and
// the two variants disagree on every OS-visible marker (side-by-side
// installs must not collide).
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { build } from 'esbuild'
import { afterEach, beforeEach, test, vi } from 'vitest'

const require = createRequire(import.meta.url)

beforeEach(() => {
  vi.resetModules()
})

afterEach(() => {
  delete process.env.HERMES_DESKTOP_VARIANT
  delete process.env.HERMES_PAYLOAD_TAG
  delete process.env.HERMES_BUILD_COMMIT
  delete process.env.HERMES_PAYLOAD_VERSION
  vi.resetModules()
})

async function identityForVariant(variant: string | undefined) {
  if (variant === undefined) {
    delete process.env.HERMES_DESKTOP_VARIANT
  } else {
    process.env.HERMES_DESKTOP_VARIANT = variant
  }

  delete require.cache[require.resolve('../product-identity.cjs')]
  vi.resetModules()

  return (await import('./product-identity')).PRODUCT_IDENTITY
}

test('baked runtime identity never evaluates ambient build selectors', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'baked-identity-'))

  try {
    const identity = await identityForVariant('bundled')

    const result = await build({
      stdin: { contents: "import {PRODUCT_IDENTITY} from './product-identity'; console.log(JSON.stringify(PRODUCT_IDENTITY))", resolveDir: import.meta.dirname },
      bundle: true, platform: 'node', format: 'esm', write: false,
      define: { __HERMES_PRODUCT_IDENTITY__: JSON.stringify(identity) }
    })

    const file = path.join(dir, 'identity.mjs')
    fs.writeFileSync(file, result.outputFiles[0].text)

    const actual = JSON.parse(execFileSync(process.execPath, [file], {
      encoding: 'utf8', env: { ...process.env, HERMES_DESKTOP_VARIANT: 'store', HERMES_BUILD_COMMIT: 'a'.repeat(40) }
    }))

    assert.deepEqual(actual, identity)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('nonstable runtime pins userData before the app name can change', async () => {
  const stable = await identityForVariant('bundled')
  process.env.HERMES_PAYLOAD_TAG = 'v0.28.0-canary.20260818'
  const canary = await identityForVariant('bundled')
  const runtime = await import('./product-identity')
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'identity-userdata-'))
  const paths: Record<string, string> = { appData: root, userData: path.join(root, 'Hermes') }
  let name = 'Hermes'

  const app = {
    getPath: (key: string) => paths[key],
    setPath: (key: string, value: string) => { assert.ok(fs.statSync(value).isDirectory()); paths[key] = value },
    setName: (value: string) => { name = value }
  }

  try {
    assert.equal(runtime.applyDesktopIdentity(app, stable), null)
    assert.equal(paths.userData, path.join(root, 'Hermes'))
    assert.equal(runtime.applyDesktopIdentity(app, canary), canary.displayName)
    assert.equal(paths.userData, path.join(paths.appData, canary.appNamePascal))
    assert.equal(name, canary.displayName)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('light identity is fully distinct from the full identity', async () => {
  const full = await identityForVariant(undefined)
  const light = await identityForVariant('light')

  assert.equal(light.light, true)

  // Enumerate the OS-visible identity markers explicitly rather than looping
  // Object.keys: `store` is a build-mode flag, legitimately equal (false)
  // across variants, so a keys-loop would fail for the wrong reason. Only the
  // markers Windows/electron keys on must differ for side-by-side installs.
  for (const prop of ['displayName', 'appId', 'appNamePascal', 'msixAppIdWithOrg', 'channel'] as const) {
    assert.notEqual(light[prop], full[prop], `${prop} must differ between light and full`)
  }
})

test('a canary payload tag moves BOTH variants onto their canary feed channel', async () => {
  process.env.HERMES_PAYLOAD_TAG = 'v0.28.0-canary.20260818'
  const full = await identityForVariant(undefined)
  assert.equal(full.channel, 'canary')

  process.env.HERMES_PAYLOAD_TAG = 'v0.28.0-canary.20260818'
  const light = await identityForVariant('light')
  assert.equal(light.channel, 'light-canary')
})

test('canary installs alongside stable with its own GUI and CLI names', async () => {
  for (const variant of [undefined, 'bundled', 'light']) {
    delete process.env.HERMES_PAYLOAD_TAG
    const stable = await identityForVariant(variant)
    process.env.HERMES_PAYLOAD_TAG = 'v0.28.0-canary.20260818'
    const canary = await identityForVariant(variant)

    for (const prop of ['displayName', 'appId', 'appNamePascal', 'msixAppIdWithOrg', 'windowsExecutableName', 'cliName'] as const) {
      assert.notEqual(canary[prop], stable[prop], `${prop} must isolate canary`)
    }

    assert.equal(canary.artifactNamePascal, stable.artifactNamePascal)
    assert.equal(canary.cliName, variant === 'light' ? 'hermes-light-canary' : 'hermes-canary')
    assert.equal(canary.windowsExecutableName, canary.cliName)
  }
})

test('each commit owns a deterministic identity and has no release channel', async () => {
  const firstSha = 'abcdef1234567890abcdef1234567890abcdef12'
  const secondSha = '1234567890abcdef1234567890abcdef12345678'

  for (const variant of [undefined, 'bundled', 'light']) {
    delete process.env.HERMES_BUILD_COMMIT
    const stable = await identityForVariant(variant)
    process.env.HERMES_BUILD_COMMIT = firstSha
    const first = await identityForVariant(variant)
    assert.deepEqual(first, await identityForVariant(variant))
    process.env.HERMES_BUILD_COMMIT = secondSha
    const second = await identityForVariant(variant)

    for (const prop of ['displayName', 'appId', 'appNamePascal', 'msixAppIdWithOrg', 'windowsExecutableName', 'cliName'] as const) {
      assert.notEqual(first[prop], stable[prop], `${prop} must isolate commit from stable`)
      assert.notEqual(first[prop], second[prop], `${prop} must isolate two commits`)
      assert.ok(first[prop].includes(firstSha.slice(0, 7)))
    }

    assert.equal(first.channel, null)
    assert.equal(first.cliName, `${variant === 'light' ? 'hermes-light' : 'hermes'}-${firstSha.slice(0, 7)}`)
    assert.equal(first.artifactNamePascal, stable.artifactNamePascal)
  }
})

test('a commit build names the SHA in the display name', async () => {
  process.env.HERMES_BUILD_COMMIT = 'abcdef1234567890abcdef1234567890abcdef12'
  const full = await identityForVariant(undefined)
  assert.equal(full.displayName, 'Hermes abcdef1')

  const bundled = await identityForVariant('bundled')
  assert.equal(bundled.displayName, 'Hermes Agent abcdef1')

  delete process.env.HERMES_BUILD_COMMIT
  const plain = await identityForVariant(undefined)
  assert.equal(plain.displayName, 'Hermes')

  // Malformed commit values must not leak into the name (commit builds
  // validate the full SHA elsewhere; the display derivation stays total).
  process.env.HERMES_BUILD_COMMIT = 'not-a-sha'
  const malformed = await identityForVariant(undefined)
  assert.equal(malformed.displayName, 'Hermes')
})

test('packaging isolates boot metadata and executable names without renaming release artifacts', async () => {
  const pkg = require('../package.json')

  const load = () => {
    delete require.cache[require.resolve('../electron-builder.config.cjs')]

    return require('../electron-builder.config.cjs')
  }

  const stableIdentity = await identityForVariant('bundled')
  const stable = load()
  assert.equal(stable.extraMetadata.productName || pkg.productName, pkg.productName)

  for (const build of ['canary', 'abcdef1234567890abcdef1234567890abcdef12']) {
    process.env.HERMES_PAYLOAD_TAG = build === 'canary' ? 'v0.28.0-canary.20260818' : ''
    process.env.HERMES_BUILD_COMMIT = build === 'canary' ? '' : build
    const identity = await identityForVariant('bundled')
    const config = load()
    // Electron bootstrap gives productName precedence over name. appId alone
    // changes neither its early userData lookup nor its single-instance lock.
    assert.equal(config.extraMetadata.productName, identity.displayName)
    assert.equal(config.extraMetadata.name, identity.appNamePascal)
    assert.notEqual(config.extraMetadata.name, stableIdentity.appNamePascal)
    assert.equal(config.win.executableName, identity.windowsExecutableName)
    const { AppInfo } = require('../../../node_modules/app-builder-lib/dist/appInfo.js')
    const appInfo = new AppInfo({ config, metadata: { ...pkg, ...config.extraMetadata } }, null, config.win)
    // Windows packaging, afterPack, rollback preservation and final signing
    // all consume this resolved name, not the display name.
    assert.equal(appInfo.productFilename, identity.windowsExecutableName)
    assert.equal(config.mac.extendInfo.CFBundleExecutable, config.executableName)
    assert.equal(config.artifactName, stable.artifactName)
    const { appIdentity } = require('../../../scripts/msix-shared.mjs')
    process.env.HERMES_PAYLOAD_VERSION = '0.28.0'
    const artifact = appIdentity(fileURLToPath(new URL('../', import.meta.url)))
    assert.equal(artifact.name, identity.artifactNamePascal)
    assert.equal(artifact.identity.msixAppIdWithOrg, config.msix.identityName)
    assert.deepEqual(config.protocols[0].schemes, ['hermes'])

    if (build !== 'canary') {
      assert.equal(config.publish, null)
      assert.equal(config.mac.publish, null)
    }
  }
})

test('stable tags and tagless dev builds publish to the stable channels', async () => {
  process.env.HERMES_PAYLOAD_TAG = 'v0.28.0'
  assert.equal((await identityForVariant(undefined)).channel, 'latest')

  delete process.env.HERMES_PAYLOAD_TAG
  assert.equal((await identityForVariant('light')).channel, 'light')
})

test('bundled variant has a distinct identity from the full variant', async () => {
  const full = await identityForVariant(undefined)
  const bundled = await identityForVariant('bundled')

  assert.equal(bundled.light, false)
  assert.notEqual(bundled.displayName, full.displayName)
  assert.notEqual(bundled.appNamePascal, full.appNamePascal)
  assert.notEqual(bundled.appId, full.appId)
  assert.equal(bundled.channel, 'latest')
})

test('store inherits the bundled app identity (shared userData) but swaps the MSIX packaging identity', async () => {
  const bundled = await identityForVariant('bundled')
  const store = await identityForVariant('store')

  // Same Electron app: displayName/appNamePascal (-> shared userData dir),
  // appId, and the out-of-store org-prefixed name are all inherited.
  assert.equal(store.store, true)
  assert.equal(store.light, false)
  assert.equal(store.displayName, bundled.displayName)
  assert.equal(store.appNamePascal, bundled.appNamePascal)
  assert.equal(store.appId, bundled.appId)
  assert.equal(store.msixAppIdWithOrg, bundled.msixAppIdWithOrg)
  // The store build never publishes to a feed.
  assert.equal(store.channel, null)
})

test('nonstable builds cannot claim the official Store package', async () => {
  process.env.HERMES_PAYLOAD_TAG = 'v0.28.0-canary.20260818'
  await assert.rejects(identityForVariant('store'), /Store.*stable/)
  delete process.env.HERMES_PAYLOAD_TAG
  process.env.HERMES_BUILD_COMMIT = 'abcdef1234567890abcdef1234567890abcdef12'
  await assert.rejects(identityForVariant('store'), /Store.*stable/)
})

test('store carries the Partner Center MSIX identity and no other variant does', async () => {
  const store = await identityForVariant('store')
  assert.deepEqual(store.storeMsix, {
    identityName: 'NousResearchInc.HermesAgent',
    publisher: 'CN=EE6D86E4-606F-4E38-B940-AD7248C9D519',
    publisherDisplayName: 'Nous Research Inc.'
  })

  for (const v of [undefined, 'bundled', 'light'] as const) {
    assert.equal((await identityForVariant(v)).storeMsix, undefined, `variant ${v} must carry no storeMsix`)
  }
})
