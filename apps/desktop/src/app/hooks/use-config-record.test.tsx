import { QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/connections', async () => {
  const { atom } = await import('nanostores')

  return { $activeConnectionId: atom<null | string>(null) }
})
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return { $activeGatewayRoute: atom('default') }
})

const apiMocks = vi.hoisted(() => ({ getHermesConfigRecord: vi.fn() }))

vi.mock('@/hermes', () => ({ getHermesConfigRecord: apiMocks.getHermesConfigRecord }))

import { queryClient } from '@/lib/query-client'
import { $activeConnectionId as $mockedConnectionId } from '@/store/connections'
import { $activeGatewayRoute } from '@/store/gateway'

import { HERMES_CONFIG_KEY, hermesConfigCacheWriter, hermesConfigKey, setHermesConfigCache, useHermesConfigRecord } from './use-config-record'

// The real store exports a computed (read-only) atom; the mock is a plain writable one.
const $activeConnectionId = $mockedConnectionId as WritableAtom<null | string>

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

const record = (marker: string) => ({ marker }) as unknown as Record<string, unknown>

beforeEach(() => {
  $activeConnectionId.set(null)
  $activeGatewayRoute.set('default')
  queryClient.clear()
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('hermesConfigKey', () => {
  it('scopes the ambient record to the active gateway connection and profile', () => {
    expect(hermesConfigKey()).toEqual([...HERMES_CONFIG_KEY, 'primary::default'])

    $activeConnectionId.set('devbox')
    expect(hermesConfigKey()).toEqual([...HERMES_CONFIG_KEY, 'devbox::default'])

    $activeGatewayRoute.set('work')
    expect(hermesConfigKey()).toEqual([...HERMES_CONFIG_KEY, 'devbox::work'])
  })

  it('lands an explicit pin on the active source on the ambient row', () => {
    $activeConnectionId.set('devbox')

    expect(hermesConfigKey({ connectionId: 'devbox', profile: 'default' })).toEqual(hermesConfigKey())
    expect(hermesConfigKey({ connectionId: 'local', profile: 'default' })).not.toEqual(hermesConfigKey())
  })
})

describe('setHermesConfigCache', () => {
  it('writes the row of the gateway active at write time, even through a memoized writer', () => {
    $activeConnectionId.set('local')
    const writer = hermesConfigCacheWriter()
    setHermesConfigCache(record('laptop'))

    $activeConnectionId.set('devbox')
    expect(queryClient.getQueryData(hermesConfigKey())).toBeUndefined()
    writer(record('remote'))
    expect(queryClient.getQueryData(hermesConfigKey())).toEqual(record('remote'))

    $activeConnectionId.set('local')
    expect(queryClient.getQueryData(hermesConfigKey())).toEqual(record('laptop'))
  })
})

describe('useHermesConfigRecord', () => {
  it('never serves the previous gateway record after a switch', async () => {
    apiMocks.getHermesConfigRecord.mockImplementation(async () => record($activeConnectionId.get() ?? 'primary'))
    $activeConnectionId.set('local')
    const paints: Array<[null | string, unknown]> = []

    const { result } = renderHook(
      () => {
        const query = useHermesConfigRecord()
        paints.push([$activeConnectionId.get(), query.data])

        return query
      },
      { wrapper }
    )

    await waitFor(() => expect(result.current.data).toEqual(record('local')))

    // Switch the active gateway: the ambient key moves with the connection in
    // the same render.
    await act(async () => {
      $activeConnectionId.set('devbox')
    })

    await waitFor(() => expect(result.current.data).toEqual(record('devbox')))
    // The laptop record was never painted while the remote was active — that
    // stale paint is what a settings save then PUT onto the other machine.
    const remotePaints = paints.filter(([connectionId]) => connectionId === 'devbox').map(([, data]) => data)
    expect(remotePaints.length).toBeGreaterThan(0)
    expect(remotePaints).not.toContainEqual(record('local'))
    // Each source keeps its own row.
    expect(queryClient.getQueryData([...HERMES_CONFIG_KEY, 'devbox::default'])).toEqual(record('devbox'))
    expect(queryClient.getQueryData([...HERMES_CONFIG_KEY, 'local::default'])).toEqual(record('local'))
  })
})
