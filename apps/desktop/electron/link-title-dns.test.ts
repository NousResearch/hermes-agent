import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createLinkTitlePinnedResolver, type LinkTitleAddress } from './link-title-dns'

const PUBLIC_V4 = '93.184.216.34'
const PUBLIC_V6 = '2606:4700:4700::1111'
const PRIVATE_V4 = '10.0.0.8'

function address(address: string, family: 4 | 6): LinkTitleAddress {
  return { address, family }
}

function isPublicAddress(value: string): boolean {
  return value === PUBLIC_V4 || value === PUBLIC_V6
}

test('normalizes hostname cache keys and reuses the immutable approved answer set inside the TTL', async () => {
  const lookup = vi.fn(async () => [address(PUBLIC_V4, 4), address(PUBLIC_V6, 6), address(PUBLIC_V4, 4)])
  const resolver = createLinkTitlePinnedResolver({ isPublicAddress, lookup, now: () => 1_000, ttlMs: 30_000 })

  const first = await resolver.resolve('ExAmPlE.COM.')
  const second = await resolver.resolve('example.com')

  assert.equal(first, second)
  assert.deepEqual(first, [address(PUBLIC_V4, 4), address(PUBLIC_V6, 6)])
  assert.equal(Object.isFrozen(first), true)
  assert.equal(Object.isFrozen(first[0]), true)
  assert.deepEqual(lookup.mock.calls, [['example.com']])
})

test('public IP literals are normalized without DNS and private literals are rejected', async () => {
  const lookup = vi.fn<() => Promise<readonly LinkTitleAddress[]>>()
  const resolver = createLinkTitlePinnedResolver({ isPublicAddress, lookup, ttlMs: 30_000 })

  assert.deepEqual(await resolver.resolve(`[${PUBLIC_V6}]`), [address(PUBLIC_V6, 6)])
  assert.deepEqual(await resolver.resolve(PUBLIC_V4), [address(PUBLIC_V4, 4)])
  await assert.rejects(resolver.resolve(PRIVATE_V4), /non-public/i)
  assert.equal(lookup.mock.calls.length, 0)
})

test('rejects empty, private-only, and mixed DNS answers without caching them', async () => {
  const lookup = vi
    .fn<(hostname: string) => Promise<readonly LinkTitleAddress[]>>()
    .mockResolvedValueOnce([])
    .mockResolvedValueOnce([address(PRIVATE_V4, 4)])
    .mockResolvedValueOnce([address(PUBLIC_V4, 4), address(PRIVATE_V4, 4)])
    .mockResolvedValueOnce([address(PUBLIC_V4, 4)])

  const resolver = createLinkTitlePinnedResolver({ isPublicAddress, lookup, now: () => 0, ttlMs: 30_000 })

  await assert.rejects(resolver.resolve('empty.example'), /empty/i)
  await assert.rejects(resolver.resolve('private.example'), /non-public/i)
  await assert.rejects(resolver.resolve('mixed.example'), /non-public/i)
  assert.deepEqual(await resolver.resolve('mixed.example'), [address(PUBLIC_V4, 4)])
  assert.deepEqual(lookup.mock.calls, [['empty.example'], ['private.example'], ['mixed.example'], ['mixed.example']])
})

test('renews an active pin for a full TTL so a near-expiry fetch cannot rebind mid-flight', async () => {
  let now = 100
  let finishFirstLookup: ((value: readonly LinkTitleAddress[]) => void) | undefined

  const firstLookup = new Promise<readonly LinkTitleAddress[]>(resolve => {
    finishFirstLookup = resolve
  })

  const lookup = vi
    .fn<(hostname: string) => Promise<readonly LinkTitleAddress[]>>()
    .mockReturnValueOnce(firstLookup)
    .mockResolvedValueOnce([address(PRIVATE_V4, 4)])
    .mockResolvedValueOnce([address(PUBLIC_V6, 6)])

  const resolver = createLinkTitlePinnedResolver({ isPublicAddress, lookup, now: () => now, ttlMs: 30_000 })

  const first = resolver.resolve('rebind.example')
  const concurrent = resolver.resolve('REBIND.EXAMPLE.')
  assert.equal(lookup.mock.calls.length, 1)
  finishFirstLookup?.([address(PUBLIC_V4, 4)])
  assert.equal(await first, await concurrent)

  now += 29_999
  assert.deepEqual(await resolver.resolve('rebind.example'), [address(PUBLIC_V4, 4)])
  assert.equal(lookup.mock.calls.length, 1)

  now += 20_000
  assert.deepEqual(await resolver.resolve('rebind.example'), [address(PUBLIC_V4, 4)])
  assert.equal(lookup.mock.calls.length, 1)

  now += 30_001
  await assert.rejects(resolver.resolve('rebind.example'), /non-public/i)
  assert.equal(lookup.mock.calls.length, 2)

  assert.deepEqual(await resolver.resolve('rebind.example'), [address(PUBLIC_V6, 6)])
  assert.equal(lookup.mock.calls.length, 3)
})

test('bounds one-shot hostname pins and evicts the least recently used entry', async () => {
  let now = 0
  const lookup = vi.fn(async () => [address(PUBLIC_V4, 4)])

  const resolver = createLinkTitlePinnedResolver({
    isPublicAddress,
    lookup,
    maxEntries: 2,
    now: () => now,
    ttlMs: 30_000
  })

  await resolver.resolve('one.example')
  now += 1
  await resolver.resolve('two.example')
  now += 1
  await resolver.resolve('one.example')
  now += 1
  await resolver.resolve('three.example')
  now += 1
  await resolver.resolve('two.example')

  assert.deepEqual(lookup.mock.calls, [['one.example'], ['two.example'], ['three.example'], ['two.example']])
})

test('prunes expired one-shot pins before admitting new hostnames', async () => {
  let now = 0
  const lookup = vi.fn(async () => [address(PUBLIC_V4, 4)])

  const resolver = createLinkTitlePinnedResolver({
    isPublicAddress,
    lookup,
    maxEntries: 2,
    now: () => now,
    ttlMs: 10
  })

  await resolver.resolve('expired.example')
  now = 11
  await resolver.resolve('fresh.example')
  await resolver.resolve('expired.example')

  assert.deepEqual(lookup.mock.calls, [['expired.example'], ['fresh.example'], ['expired.example']])
})

test('clear invalidates approved pins immediately', async () => {
  const lookup = vi
    .fn<(hostname: string) => Promise<readonly LinkTitleAddress[]>>()
    .mockResolvedValueOnce([address(PUBLIC_V4, 4)])
    .mockResolvedValueOnce([address(PUBLIC_V6, 6)])

  const resolver = createLinkTitlePinnedResolver({ isPublicAddress, lookup, now: () => 0, ttlMs: 30_000 })

  assert.deepEqual(await resolver.resolve('example.com'), [address(PUBLIC_V4, 4)])
  resolver.clear()
  assert.deepEqual(await resolver.resolve('example.com'), [address(PUBLIC_V6, 6)])
  assert.equal(lookup.mock.calls.length, 2)
})
