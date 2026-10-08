// The sidebar's FTS search must cover the profiles the sidebar SHOWS, not
// just the primary backend's launch profile. Under "All profiles" the hook
// fans one scoped request per profile (searchAllProfileSessions) — the FTS
// endpoint is single-store per call, so an unscoped request silently returns
// only the default profile's hits and a named profile's sessions vanish from
// search results.
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ALL_PROFILES } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { useServerSessionSearch } from './use-server-session-search'

vi.mock('@/hermes', () => ({
  searchAllProfileSessions: vi.fn(),
  searchSessions: vi.fn()
}))

// The hook only reads the ALL_PROFILES sentinel and normalizeProfileKey from
// the profile store; mock just those so the heavy store import graph (gateway,
// sessions, oauth…) never loads and the test isolates the search fan-out.
vi.mock('@/store/profile', () => ({
  ALL_PROFILES: '__all__',
  normalizeProfileKey: (name: string | null | undefined) => (name ?? '').trim() || 'default'
}))

const { searchAllProfileSessions, searchSessions } = await import('@/hermes')

function profile(name: string): ProfileInfo {
  return {
    display_name: undefined,
    has_env: false,
    is_default: name === 'default',
    model: null,
    name,
    path: `/profiles/${name}`,
    provider: null,
    skill_count: 0
  }
}

const hit = (owner: string, id: string) => ({
  last_active: null,
  lineage_root: null,
  model: null,
  profile: owner,
  session_id: id,
  session_started: 1000,
  snippet: id,
  source: null
})

// Stable reference across renders — the real caller passes `useStore($profiles)`,
// an atom value that keeps its identity. A fresh array per render would churn
// the hook's memo and re-fire the debounce effect, so the test models reality.
const PROFILES = [profile('default'), profile('tommy')]
const DEFAULT_ONLY = [profile('default')]

describe('useServerSessionSearch profile coverage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('fans one scoped request per profile under All profiles', async () => {
    vi.mocked(searchAllProfileSessions).mockResolvedValue({
      results: [hit('default', 'd1'), hit('tommy', 't1')]
    })

    const { result, unmount } = renderHook(() => useServerSessionSearch('deploy', ALL_PROFILES, PROFILES))

    expect(result.current.searchPending).toBe(true)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    expect(searchAllProfileSessions).toHaveBeenCalledTimes(1)
    expect(searchAllProfileSessions).toHaveBeenCalledWith('deploy', ['default', 'tommy'])
    expect(searchSessions).not.toHaveBeenCalled()
    expect(result.current.serverMatches.map(h => h.session_id)).toEqual(['d1', 't1'])
    expect(result.current.searchPending).toBe(false)

    unmount()
  })

  it('sends a single scoped request for a concrete profile scope', async () => {
    vi.mocked(searchSessions).mockResolvedValue({ results: [hit('tommy', 't1')] })

    const { result, unmount } = renderHook(() => useServerSessionSearch('deploy', 'tommy', PROFILES))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    expect(searchSessions).toHaveBeenCalledWith('deploy', 'tommy')
    expect(searchAllProfileSessions).not.toHaveBeenCalled()
    expect(result.current.serverMatches.map(h => h.session_id)).toEqual(['t1'])

    unmount()
  })

  it('sends nothing for an empty query', () => {
    const { unmount } = renderHook(() => useServerSessionSearch('', ALL_PROFILES, DEFAULT_ONLY))

    act(() => {
      vi.advanceTimersByTime(200)
    })

    expect(searchAllProfileSessions).not.toHaveBeenCalled()
    expect(searchSessions).not.toHaveBeenCalled()

    unmount()
  })

  it('cancels the debounced request when the query clears before it fires', () => {
    const { rerender, unmount } = renderHook(
      ({ q }: { q: string }) => useServerSessionSearch(q, ALL_PROFILES, PROFILES),
      { initialProps: { q: 'deploy' } }
    )

    // Clear the query inside the debounce window — the pending search dies.
    rerender({ q: '' })

    act(() => {
      vi.advanceTimersByTime(200)
    })

    expect(searchAllProfileSessions).not.toHaveBeenCalled()

    unmount()
  })
})
