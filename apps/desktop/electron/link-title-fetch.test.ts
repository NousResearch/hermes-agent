import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createLinkTitleFetcher } from './link-title-fetch'

function makeDependencies() {
  return {
    admitUrl: vi.fn(() => true),
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
  deps.admitUrl.mockReturnValue(false)
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
  assert.deepEqual(deps.fetchWithCurl.mock.calls, [['https://example.com']])
  assert.deepEqual(deps.fetchWithRenderer.mock.calls, [['https://example.com']])
  assert.ok(deps.fetchWithCurl.mock.invocationCallOrder[0] < deps.fetchWithRenderer.mock.invocationCallOrder[0])
  assert.deepEqual(deps.storeCachedTitle.mock.calls, [['example.com/', 'Rendered title']])
  assert.equal(deps.inflight.set.mock.calls.length, 1)
  assert.equal(deps.inflight.set.mock.calls[0][0], 'example.com/')
  assert.equal(deps.inflight.set.mock.calls[0][1], pending)
  assert.deepEqual(deps.inflight.delete.mock.calls, [['example.com/']])
})
