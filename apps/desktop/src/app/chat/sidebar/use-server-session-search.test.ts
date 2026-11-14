// A failed search request is "could not search", never "no matches": the hook
// used to swallow the error and keep the previous query's hits under the new
// query, which the sidebar then rendered as results (or as "No sessions match").
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const searchSessions = vi.fn()

vi.mock('@/hermes', () => ({ searchSessions: (...args: unknown[]) => searchSessions(...args) }))
vi.mock('@/store/profile', () => ({ ALL_PROFILES: '__all__', normalizeProfileKey: (key: string) => key }))

import { useServerSessionSearch } from './use-server-session-search'

describe('useServerSessionSearch', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    searchSessions.mockReset()
  })

  it('drops the previous hits and flags the failure when a request fails', async () => {
    searchSessions.mockResolvedValueOnce({ results: [{ session_id: 'old-hit', snippet: 'nimbus' }] })
    searchSessions.mockRejectedValueOnce(new Error('503 Search failed'))

    const { rerender, result } = renderHook(({ q }) => useServerSessionSearch(q, 'default'), {
      initialProps: { q: 'nimbus' }
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    expect(result.current.serverMatches.map(m => m.session_id)).toEqual(['old-hit'])
    expect(result.current.searchFailed).toBe(false)

    rerender({ q: 'zephyr' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    expect(result.current.serverMatches).toEqual([])
    expect(result.current.searchFailed).toBe(true)
    expect(result.current.searchPending).toBe(false)
  })
})
