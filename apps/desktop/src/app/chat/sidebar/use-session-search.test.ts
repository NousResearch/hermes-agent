import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { searchSessions, type SessionSearchResponse } from '@/hermes'
import { invalidateSessionTags } from '@/store/session-tags'

import { useSessionSearch } from './use-session-search'

vi.mock('@/hermes', () => ({ searchSessions: vi.fn() }))
vi.mock('@/store/session', () => ({ $connection: atom({ connectionId: 'remote', profile: 'p' }) }))
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('ignores an in-flight search response superseded by tag invalidation', async () => {
  vi.useFakeTimers()
  let resolveOld!: (value: SessionSearchResponse) => void
  vi.mocked(searchSessions)
    .mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
    .mockResolvedValueOnce({ results: [] })
  const { result } = renderHook(() => useSessionSearch('needle'))
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  act(() => invalidateSessionTags())
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  await act(async () => {
    resolveOld({ results: [{ session_id: 'stale', snippet: 'needle', tags: ['a'], model: null, role: null, source: null, session_started: null }] })
  })
  expect(searchSessions).toHaveBeenCalledTimes(2)
  expect(result.current.serverMatches).toEqual([])
  expect(result.current.searchPending).toBe(false)
})
