import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import * as api from '@/api/client'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'

import { useSimplifiedSessionSearch } from './use-simplified-session-search'

const hit = { session_id: 'same', model: null, role: null, session_started: null, snippet: 'found', source: null }
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  $fleetRoster.set(null)
  $profiles.set([])
})

it('searches every exact owner and keeps unloaded hits attached to their gateway and profile', async () => {
  $apiRequestScope.set({ connectionId: 'alpha', profile: 'default' })
  $profiles.set([{ name: 'research', is_default: false }] as typeof $profiles.value)
  $fleetRoster.set({
    sources: [],
    agents: [
      {
        connectionId: 'beta',
        profile: 'research',
        connectionLabel: 'Beta',
        connectionKind: 'remote',
        handle: 'research-beta'
      }
    ]
  })
  const search = vi.spyOn(api, 'hermesApi').mockResolvedValue({ results: [hit] })
  const { result } = renderHook(() => useSimplifiedSessionSearch('found'))
  await waitFor(() => expect(result.current.rows).toHaveLength(2))
  expect(search).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'alpha', profile: 'research' }))
  expect(search).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'beta', profile: 'research' }))
  expect(result.current.rows.map(row => [row.connection_id, row.profile])).toEqual([
    ['alpha', 'research'],
    ['beta', 'research']
  ])
})

it('reports a failed search instead of presenting it as an empty successful result', async () => {
  $profiles.set([{ name: 'research', is_default: false }] as typeof $profiles.value)
  vi.spyOn(api, 'hermesApi').mockRejectedValue(new Error('offline'))
  const { result } = renderHook(() => useSimplifiedSessionSearch('found'))
  await waitFor(() => expect(result.current.failed).toBe(true))
  expect(result.current.pending).toBe(false)
})
