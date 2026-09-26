import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { bindConfigReadOrigin, getHermesConfigRecord } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $connection } from '@/store/session'

import {
  HERMES_CONFIG_KEY,
  hermesConfigCacheWriter,
  setHermesConfigCache,
  useHermesConfigRecord
} from './use-config-record'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getHermesConfigRecord: vi.fn()
}))

afterEach(() => {
  cleanup()
  queryClient.clear()
  $connection.set(null)
  vi.clearAllMocks()
})

const wrapper = ({ children }: { children: React.ReactNode }) =>
  createElement(QueryClientProvider, { client: queryClient }, children)

it('updates the write origin when a refetch replaces the displayed record', async () => {
  const first = { display: { theme: 'dark' } }
  const second = { display: { theme: 'light' } }
  bindConfigReadOrigin(first, { connectionId: 'connection-a', profile: 'worker' })
  bindConfigReadOrigin(second, { connectionId: 'connection-b', profile: 'worker' })
  vi.mocked(getHermesConfigRecord).mockResolvedValueOnce(first).mockResolvedValueOnce(second)

  const { result } = renderHook(() => useHermesConfigRecord(), { wrapper })

  // Before the first GET resolves the scope must be `undefined` (not `null`):
  // profileScoped(null) drops the active profile and targets the PRIMARY.
  expect(result.current.data).toBeUndefined()
  expect(result.current.writeScope).toBeUndefined()
  expect(result.current.writeScope).not.toBeNull()

  await waitFor(() => expect(result.current.data).toBe(first))
  expect(result.current.writeScope).toEqual({ connectionId: 'connection-a', profile: 'worker' })

  await queryClient.invalidateQueries({ queryKey: HERMES_CONFIG_KEY })

  await waitFor(() => expect(result.current.data).toEqual(second))
  await waitFor(() => expect(result.current.writeScope).toEqual({ connectionId: 'connection-b', profile: 'worker' }))
})

function useGateway(connectionId: string) {
  $connection.set({ connectionId, mode: 'remote' } as never)
}

it('does not share one config record across two gateways', async () => {
  const laptop = { display: { theme: 'dark' } }
  const devbox = { display: { theme: 'light' } }
  let releaseDevbox: (record: typeof devbox) => void = () => undefined
  const devboxFetch = new Promise<typeof devbox>(resolve => {
    releaseDevbox = resolve
  })

  vi.mocked(getHermesConfigRecord).mockImplementation(() =>
    $connection.get()?.connectionId === 'devbox' ? devboxFetch : Promise.resolve(laptop)
  )

  useGateway('laptop')
  const { result } = renderHook(() => useHermesConfigRecord(), { wrapper })

  await waitFor(() => expect(result.current.data).toEqual(laptop))

  await act(async () => {
    useGateway('devbox')
  })

  // A settings save paints from this cache. The previous machine's record must
  // not still be the displayed one after the switch, or the save PUTs it onto
  // the other machine's config.yaml.
  expect(result.current.data).not.toEqual(laptop)

  await act(async () => {
    releaseDevbox(devbox)
  })
  await waitFor(() => expect(result.current.data).toEqual(devbox))

  const cached = queryClient.getQueriesData({ queryKey: HERMES_CONFIG_KEY }).map(([, data]) => data)

  expect(cached).toContainEqual(laptop)
  expect(cached).toContainEqual(devbox)
})

it('a settings cache write after switching gateways does not replace the other gateway record', () => {
  const laptop = { display: { theme: 'dark' } }
  const devbox = { display: { theme: 'light' } }
  // Config settings memoize the writer on the profile name. Both gateways are
  // on `default`, so a captured unscoped key would let the second save replace
  // the first machine's record.
  const writer = hermesConfigCacheWriter('default')

  useGateway('laptop')
  setHermesConfigCache(laptop)
  writer({ agent: { model: 'laptop-model' } })

  useGateway('devbox')
  setHermesConfigCache(devbox)
  writer({ agent: { model: 'devbox-model' } })

  const cached = queryClient.getQueriesData({ queryKey: HERMES_CONFIG_KEY }).map(([, data]) => data)

  expect(cached).toContainEqual(laptop)
  expect(cached).toContainEqual(devbox)
  expect(cached).toContainEqual({ agent: { model: 'laptop-model' } })
  expect(cached).toContainEqual({ agent: { model: 'devbox-model' } })
})
