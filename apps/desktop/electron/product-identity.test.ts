// product-identity.cjs is the single derivation of the desktop product
// identity; electron/product-identity.ts re-exports it. These tests hold
// the identity contract: the TS accessor resolves to the .cjs object, and
// the two variants disagree on every OS-visible marker (side-by-side
// installs must not collide).
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

import { afterEach, beforeEach, test, vi } from 'vitest'

const require = createRequire(import.meta.url)

beforeEach(() => {
  vi.resetModules()
})

afterEach(() => {
  delete process.env.HERMES_DESKTOP_VARIANT
  delete process.env.HERMES_PAYLOAD_TAG
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

test('stable tags and tagless dev builds publish to the stable channels', async () => {
  process.env.HERMES_PAYLOAD_TAG = 'v0.28.0'
  assert.equal((await identityForVariant(undefined)).channel, 'latest')

  delete process.env.HERMES_PAYLOAD_TAG
  assert.equal((await identityForVariant('light')).channel, 'light')
})

test('bundled variant keeps the Hermes display name and a distinct appId', async () => {
  const full = await identityForVariant(undefined)
  const bundled = await identityForVariant('bundled')

  assert.equal(bundled.light, false)
  assert.equal(bundled.displayName, full.displayName)
  assert.equal(bundled.appNamePascal, 'HermesBundled')
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
