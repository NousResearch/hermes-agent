import { describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { runtimeRouteContext, sessionRouteContext } from './session-route-context'
import { recordSessionEventScope } from './session-states'

// `runtimeSessionOwner` is typed to admit null; the real map never stores it, so a test has to
// force it. Everything else in the module is genuine.
const forced = vi.hoisted(() => ({ owner: undefined as null | undefined, active: false }))

vi.mock('./session-states', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    runtimeSessionOwner: (id: null | string | undefined) =>
      forced.active ? forced.owner : (actual.runtimeSessionOwner as (id: unknown) => unknown)(id)
  }
})

const row = (over: Partial<SessionInfo>): SessionInfo => ({ id: 'tip', profile: 'default', ...over }) as SessionInfo
const here = { connectionId: null, profile: 'default' }

describe('session route context', () => {
  it('gives a compressed conversation one durable id and every id it has answered to', () => {
    const context = sessionRouteContext(row({ _lineage_ids: ['root', 'mid', 'tip'], _lineage_root_id: 'root' }), here)

    expect(context.sessionId).toBe('root')
    expect([...context.lineageIds].sort()).toEqual(['mid', 'root', 'tip'])
  })

  it('is reachable by plugin REST only when the row’s route is the active route', () => {
    // Primary pool rows are bare; `local` and "no id" are the same primary pool.
    expect(sessionRouteContext(row({}), here).ambient).toBe(true)
    expect(sessionRouteContext(row({}), { connectionId: 'local', profile: 'default' }).ambient).toBe(true)
    expect(sessionRouteContext(row({ connection_id: 'local' }), here).ambient).toBe(true)
    expect(sessionRouteContext(row({ profile: 'research' }), here).ambient).toBe(false)
    expect(sessionRouteContext(row({ connection_id: 'spark' }), here).ambient).toBe(false)

    // A REMOTE primary is registry-pinned: its rows are tagged with its own id, so they match the
    // active descriptor — a bare row there would be the primary pool, not the active remote.
    const remotePrimary = { connectionId: 'spark', profile: 'default' }

    expect(sessionRouteContext(row({ connection_id: 'spark' }), remotePrimary).ambient).toBe(true)
    expect(sessionRouteContext(row({}), remotePrimary).ambient).toBe(false)
    // Same profile name on another connection is another backend.
    expect(sessionRouteContext(row({ connection_id: 'vps' }), remotePrimary).ambient).toBe(false)
  })

  it('resolves a composer’s runtime session through the listed row, and has none for a draft', () => {
    const sessions = [row({ _lineage_ids: ['root', 'tip'], _lineage_root_id: 'root', profile: 'research' })]

    expect(runtimeRouteContext('root', sessions, here, 'rt-1')).toMatchObject({
      ambient: false,
      profile: 'research',
      sessionId: 'root'
    })
    expect(runtimeRouteContext(null, sessions, here, 'rt-1')).toBeNull()
  })

  it('takes the connection its events proved for a bare row, and never throws without an owner', () => {
    recordSessionEventScope({ connectionId: 'spark', profile: 'default', session_id: 'rt-spark' })
    const remote = { connectionId: 'spark', profile: 'default' }

    // A bare optimistic row, but the runtime's own events came from `spark`.
    expect(runtimeRouteContext('tip', [row({})], remote, 'rt-spark')?.ambient).toBe(true)
    // No proof and a bare row under that active remote: fail closed.
    expect(runtimeRouteContext('tip', [row({})], remote, 'rt-unknown')?.ambient).toBe(false)

    // Unlisted session: owner null/undefined falls back to the ACTIVE route, never throws.
    forced.active = true
    forced.owner = null

    try {
      expect(runtimeRouteContext('fresh', [], here, 'rt-null')).toMatchObject({ ambient: true, profile: 'default' })
      expect(runtimeRouteContext('fresh', [row({ id: 'other' })], here, null)).toMatchObject({ connectionId: '' })
    } finally {
      forced.active = false
    }
  })
})
