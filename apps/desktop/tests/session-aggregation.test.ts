import { describe, expect, it } from 'vitest'

import { mergeProfileSessionPages } from '../electron/session-aggregation'

const params = () => new URLSearchParams({ limit: '30', offset: '0', order: 'activity' })

describe('mergeProfileSessionPages', () => {
  it('reports a primary aggregation failure instead of returning an empty success', async () => {
    const result = await mergeProfileSessionPages({
      fetchPrimary: async () => {
        throw new Error('primary unavailable')
      },
      fetchRemote: async () => ({ sessions: [], total: 0 }),
      remoteProfiles: [],
      searchParams: params()
    })

    expect(result.sessions).toEqual([])
    expect(result.total).toBe(0)
    expect(result.errors).toEqual([{ error: 'primary unavailable', profile: 'default' }])
  })

  it('reports failed remote profiles while removing their stale rows and totals', async () => {
    const result = await mergeProfileSessionPages({
      fetchPrimary: async () => ({
        errors: [
          { error: 'stale work error', profile: 'work' },
          { error: 'archive locked', profile: 'archive' }
        ],
        profile_totals: { default: 1, work: 2 },
        sessions: [
          { id: 'local', last_active: 3, profile: 'default' },
          { id: 'stale-work', last_active: 2, profile: 'work' }
        ],
        total: 3
      }),
      fetchRemote: async profile => {
        throw new Error(`${profile} unavailable`)
      },
      remoteProfiles: ['work'],
      searchParams: params()
    })

    expect(result.sessions).toEqual([{ id: 'local', last_active: 3, profile: 'default' }])
    expect(result.total).toBe(1)
    expect(result.profile_totals).toEqual({ default: 1 })
    expect(result.errors).toEqual([
      { error: 'archive locked', profile: 'archive' },
      { error: 'work unavailable', profile: 'work' }
    ])
  })

  it('replaces stale remote rows and preserves backend-reported errors', async () => {
    const result = await mergeProfileSessionPages({
      fetchPrimary: async () => ({
        errors: [
          { error: 'stale work error', profile: 'work' },
          { error: 'archive locked', profile: 'archive' }
        ],
        profile_totals: { default: 1, work: 1 },
        sessions: [
          { id: 'local', last_active: 2, profile: 'default' },
          { id: 'stale-work', last_active: 1, profile: 'work' }
        ],
        total: 2
      }),
      fetchRemote: async () => ({ sessions: [{ id: 'live-work', last_active: 4, profile: 'work' }], total: 1 }),
      remoteProfiles: ['work'],
      searchParams: params()
    })

    expect(result.sessions).toEqual([
      { id: 'live-work', last_active: 4, profile: 'work' },
      { id: 'local', last_active: 2, profile: 'default' }
    ])
    expect(result.total).toBe(2)
    expect(result.errors).toEqual([{ error: 'archive locked', profile: 'archive' }])
  })
})
