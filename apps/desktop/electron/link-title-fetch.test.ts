import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createLinkTitleFetcher } from './link-title-fetch'

function makeDependencies() {
  return {
    admitUrl: vi.fn((value: string) => new URL(value).href),
    cache: {
      get: vi.fn(() => undefined),
      has: vi.fn(() => false)
    },
    cacheKey: vi.fn(() => 'example.com/'),
    fetchWithCurl: vi.fn(async () => ''),
    fetchWithRenderer: vi.fn(async () => 'Rendered title'),
    inflight: {
      delete: vi.fn(() => true),
      get: vi.fn(() => undefined),
      set: vi.fn()
    },
    normalizeTitle: vi.fn((value: string) => value.trim()),
    storeCachedTitle: vi.fn()
  }
}

test('blocked URLs return before cache or network title resolution', async () => {
  const deps = makeDependencies()
  deps.admitUrl.mockReturnValue(null)
  const fetchLinkTitle = createLinkTitleFetcher(deps)

  assert.equal(await fetchLinkTitle(' http://127 '), '')
  assert.deepEqual(deps.admitUrl.mock.calls, [['http://127']])
  assert.equal(deps.cacheKey.mock.calls.length, 0)
  assert.equal(deps.cache.has.mock.calls.length, 0)
  assert.equal(deps.cache.get.mock.calls.length, 0)
  assert.equal(deps.inflight.get.mock.calls.length, 0)
  assert.equal(deps.fetchWithCurl.mock.calls.length, 0)
  assert.equal(deps.fetchWithRenderer.mock.calls.length, 0)
  assert.equal(deps.storeCachedTitle.mock.calls.length, 0)
})

test('public URLs use the curl result before falling back to the renderer', async () => {
  const deps = makeDependencies()
  const fetchLinkTitle = createLinkTitleFetcher(deps)
  const pending = fetchLinkTitle(' https://example.com ')

  assert.equal(await pending, 'Rendered title')
  assert.deepEqual(deps.cacheKey.mock.calls, [['https://example.com/']])
  assert.deepEqual(deps.fetchWithCurl.mock.calls, [['https://example.com/']])
  assert.deepEqual(deps.fetchWithRenderer.mock.calls, [['https://example.com/']])
  assert.ok(deps.fetchWithCurl.mock.invocationCallOrder[0] < deps.fetchWithRenderer.mock.invocationCallOrder[0])
  assert.deepEqual(deps.storeCachedTitle.mock.calls, [['example.com/', 'Rendered title']])
  assert.equal(deps.inflight.set.mock.calls.length, 1)
  assert.equal(deps.inflight.set.mock.calls[0][0], 'example.com/')
  assert.equal(deps.inflight.set.mock.calls[0][1], pending)
  assert.deepEqual(deps.inflight.delete.mock.calls, [['example.com/']])
})

test('uses the admitted canonical URL instead of the caller representation', async () => {
  const deps = makeDependencies()
  const canonical = 'http://example.com/@127.0.0.1/'
  deps.admitUrl.mockReturnValue(canonical)
  deps.fetchWithCurl.mockResolvedValue('Curl title')
  const fetchLinkTitle = createLinkTitleFetcher(deps)

  assert.equal(await fetchLinkTitle('http://example.com\\@127.0.0.1/'), 'Curl title')
  assert.deepEqual(deps.cacheKey.mock.calls, [[canonical]])
  assert.deepEqual(deps.fetchWithCurl.mock.calls, [[canonical]])
  assert.equal(deps.fetchWithRenderer.mock.calls.length, 0)
})
