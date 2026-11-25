import { afterEach, expect, it, vi } from 'vitest'

import { searchProvider } from './skill-catalog'

const scope = { connectionId: 'search-test', profile: 'alice' }

afterEach(() => vi.restoreAllMocks())

it('serializes searches across queries and releases the queue after a real rejection', async () => {
  let reject!: (error: Error) => void
  const api = vi.fn().mockImplementationOnce(() => new Promise((_, fail) => { reject = fail })).mockResolvedValue({ results: [] })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  const first = searchProvider(scope, 'one', 'old query', null).catch(error => error)
  const second = searchProvider(scope, 'two', 'new query', null)
  await vi.waitFor(() => expect(api).toHaveBeenCalled())
  const callsBeforeFailure = api.mock.calls.length
  reject(new Error('429 Catalog search is busy'))
  expect(await first).toEqual(new Error('429 Catalog search is busy'))
  await second
  expect(callsBeforeFailure).toBe(1)
  expect(api).toHaveBeenCalledTimes(2)
})

it('keeps independent connection and profile search queues', async () => {
  const releases: (() => void)[] = []
  const api = vi.fn().mockImplementation(() => new Promise(resolve => { releases.push(() => resolve({ results: [] })) }))
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  const pending = [scope, { ...scope, profile: 'bob' }, { ...scope, connectionId: 'other' }].map(pin => searchProvider(pin, 'one', '', null))
  await vi.waitFor(() => expect(api).toHaveBeenCalledTimes(3))
  releases.forEach(release => release())
  await Promise.all(pending)
})
