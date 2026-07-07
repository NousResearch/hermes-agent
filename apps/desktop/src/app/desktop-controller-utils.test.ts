import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { pickMostRecentSessionId, sameCronSignature } from './desktop-controller-utils'

const session = (id: string, title: string | null): SessionInfo => ({ id, title }) as SessionInfo

const recent = (id: string, over: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, archived: false, is_active: false, last_active: 0, ...over }) as SessionInfo

describe('sameCronSignature', () => {
  it('is false when the lengths differ', () => {
    expect(sameCronSignature([session('a', 't')], [])).toBe(false)
  })

  it('is true when ids and titles match in order', () => {
    const a = [session('a', 'one'), session('b', 'two')]
    const b = [session('a', 'one'), session('b', 'two')]
    expect(sameCronSignature(a, b)).toBe(true)
  })

  it('is false when a title changed', () => {
    const a = [session('a', 'one')]
    const b = [session('a', 'renamed')]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  it('is false when order differs', () => {
    const a = [session('a', 't'), session('b', 't')]
    const b = [session('b', 't'), session('a', 't')]
    expect(sameCronSignature(a, b)).toBe(false)
  })
})

describe('pickMostRecentSessionId', () => {
  it('returns null for an empty list', () => {
    expect(pickMostRecentSessionId([], null)).toBeNull()
  })

  it('picks the newest session by last_active', () => {
    const sessions = [recent('old', { last_active: 100 }), recent('new', { last_active: 300 }), recent('mid', { last_active: 200 })]
    expect(pickMostRecentSessionId(sessions, null)).toBe('new')
  })

  it('prefers an active session over a newer inactive one', () => {
    const sessions = [recent('live', { is_active: true, last_active: 100 }), recent('newer-but-ended', { last_active: 900 })]
    expect(pickMostRecentSessionId(sessions, null)).toBe('live')
  })

  it('skips archived sessions', () => {
    const sessions = [recent('archived-new', { archived: true, last_active: 900 }), recent('plain-old', { last_active: 100 })]
    expect(pickMostRecentSessionId(sessions, null)).toBe('plain-old')
  })

  it('excludes the stale session by stored id and by lineage root', () => {
    const sessions = [
      recent('dead', { last_active: 900 }),
      recent('dead-tip', { _lineage_root_id: 'dead', last_active: 800 }),
      recent('alive', { last_active: 100 })
    ]

    expect(pickMostRecentSessionId(sessions, 'dead')).toBe('alive')
  })

  it('returns null when everything is excluded or archived', () => {
    const sessions = [recent('dead', { last_active: 900 }), recent('gone', { archived: true, last_active: 800 })]
    expect(pickMostRecentSessionId(sessions, 'dead')).toBeNull()
  })
})
