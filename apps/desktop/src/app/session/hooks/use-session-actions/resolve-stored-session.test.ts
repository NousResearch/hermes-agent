import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { getSession } from '@/hermes'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $removedSessionIds, tombstoneSessions, untombstoneSessions } from '@/store/projects'
import { $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { resolveSessionProfile, resolveStoredSession } from './utils'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getSession: vi.fn()
}))

const mockGetSession = vi.mocked(getSession)

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

const profiles = (...names: string[]) => names.map(name => ({ name }) as never)

describe('resolveStoredSession profile ownership', () => {
  beforeEach(() => {
    $cronSessions.set([])
    $messagingSessions.set([])
    $removedSessionIds.set(new Set())
    $sessions.set([])
    $profiles.set(profiles('default', 'meta'))
    $activeGatewayProfile.set('meta')
    mockGetSession.mockReset()
  })

  afterEach(() => {
    $cronSessions.set([])
    $messagingSessions.set([])
    $removedSessionIds.set(new Set())
    $sessions.set([])
    $profiles.set([])
    $activeGatewayProfile.set('default')
  })

  it('returns a cached row that carries an owning profile', async () => {
    $sessions.set([session({ id: 's1', profile: 'default' })])

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('default')
    expect(mockGetSession).not.toHaveBeenCalled()
  })

  it.each([
    ['cron', $cronSessions],
    ['messaging', $messagingSessions]
  ])('resolves a %s sidebar row without duplicating it into regular sessions', async (_source, store) => {
    store.set([session({ id: 's1', profile: 'default' })])

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('default')
    expect(mockGetSession).not.toHaveBeenCalled()
    expect($sessions.get()).toEqual([])
  })

  it('treats a profile-less cache hit as unresolved when multiple profiles exist', async () => {
    $sessions.set([session({ id: 's1' })])
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 's1', profile: 'default' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('default')
    // rung 2 (active profile) then rung 3 (stamped cross-profile probe)
    expect(mockGetSession).toHaveBeenNthCalledWith(1, 's1', 'meta')
    expect(mockGetSession).toHaveBeenNthCalledWith(2, 's1', 'default')
  })

  it('scopes the first by-id lookup so a miss does not skip the active profile', async () => {
    $activeGatewayProfile.set('brain')
    $profiles.set(profiles('default', 'brain'))
    mockGetSession.mockImplementation(async (id, profile) => {
      if (profile === 'brain') {
        return session({ id, profile: 'brain' })
      }

      throw new Error('404: Session not found')
    })

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('brain')
    expect(mockGetSession).toHaveBeenCalledWith('s1', 'brain')
    expect(mockGetSession).not.toHaveBeenCalledWith('s1')
    expect(mockGetSession).not.toHaveBeenCalledWith('s1', 'default')
  })

  it('accepts a profile-less cache hit for single-profile users', async () => {
    $profiles.set(profiles('default'))
    $sessions.set([session({ id: 's1' })])

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.id).toBe('s1')
    expect(mockGetSession).not.toHaveBeenCalled()
  })

  it('stamps the active profile on a bare by-id hit from an older backend', async () => {
    mockGetSession.mockResolvedValueOnce(session({ id: 's1' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('meta')
    expect(mockGetSession).toHaveBeenCalledWith('s1', 'meta')
    // the upserted cache row is owned too, so the next hit short-circuits
    expect($sessions.get().find(s => s.id === 's1')?.profile).toBe('meta')
  })

  it('probed desktop profile overrides a remote backend answering as its own "default"', async () => {
    // Per-profile remote override: Electron strips the desktop alias before
    // forwarding, so the standalone backend stamps its backend-local root.
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 's1', profile: 'default' }))
    $activeGatewayProfile.set('default')
    $profiles.set(profiles('default', 'meta'))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('meta')
    expect($sessions.get().find(s => s.id === 's1')?.profile).toBe('meta')
  })

  it('stamps the probed profile on a scoped hit from an older backend that omits it', async () => {
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 's1' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('default')
    // the cached row is owned too — no unowned row is ever re-cached
    expect($sessions.get().find(s => s.id === 's1')?.profile).toBe('default')
  })

  it('does not recache a by-id row while its session is tombstoned', async () => {
    let resolveRequest!: (value: SessionInfo) => void
    mockGetSession.mockReturnValueOnce(
      new Promise<SessionInfo>(resolve => {
        resolveRequest = resolve
      })
    )

    const pending = resolveStoredSession('s1')
    tombstoneSessions(['s1'])
    resolveRequest(session({ archived: false, id: 's1' }))

    await expect(pending).resolves.toMatchObject({ id: 's1' })
    expect($sessions.get()).toEqual([])
    untombstoneSessions(['s1'])
  })

  it('does not recache a stale by-id row when its tombstone clears before the response', async () => {
    let resolveRequest!: (value: SessionInfo) => void
    mockGetSession.mockReturnValueOnce(
      new Promise<SessionInfo>(resolve => {
        resolveRequest = resolve
      })
    )
    tombstoneSessions(['s1'])

    const pending = resolveStoredSession('s1')
    untombstoneSessions(['s1'])
    resolveRequest(session({ archived: false, id: 's1' }))

    await expect(pending).resolves.toMatchObject({ id: 's1' })
    expect($sessions.get()).toEqual([])
  })

  it('does not recache a stale by-id row after an in-flight tombstone ABA cycle', async () => {
    let resolveRequest!: (value: SessionInfo) => void
    mockGetSession.mockReturnValueOnce(
      new Promise<SessionInfo>(resolve => {
        resolveRequest = resolve
      })
    )

    const pending = resolveStoredSession('s1')
    tombstoneSessions(['s1'])
    untombstoneSessions(['s1'])
    expect($removedSessionIds.get()).toEqual(new Set())
    resolveRequest(session({ archived: false, id: 's1' }))

    await expect(pending).resolves.toMatchObject({ id: 's1' })
    expect($sessions.get()).toEqual([])
  })

  it('does not recache a stale cross-profile by-id row after an in-flight tombstone ABA cycle', async () => {
    let resolveProbe!: (value: SessionInfo) => void
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockReturnValueOnce(
      new Promise<SessionInfo>(resolve => {
        resolveProbe = resolve
      })
    )

    const pending = resolveStoredSession('s1')
    await vi.waitFor(() => expect(mockGetSession).toHaveBeenCalledTimes(2))
    tombstoneSessions(['s1'])
    untombstoneSessions(['s1'])
    resolveProbe(session({ archived: false, id: 's1' }))

    await expect(pending).resolves.toMatchObject({ id: 's1', profile: 'default' })
    expect($sessions.get()).toEqual([])
  })

  it('returns an archived by-id row for explicit resume without adding it to sidebar caches', async () => {
    mockGetSession.mockResolvedValueOnce(session({ archived: true, id: 's1' }))

    await expect(resolveStoredSession('s1')).resolves.toMatchObject({ id: 's1' })
    expect($sessions.get()).toEqual([])
  })

  it('recaches a later by-id row after a completed archive rollback', async () => {
    tombstoneSessions(['s1'])
    untombstoneSessions(['s1'])
    mockGetSession.mockResolvedValueOnce(session({ archived: false, id: 's1' }))

    await expect(resolveStoredSession('s1')).resolves.toMatchObject({ id: 's1' })
    expect($sessions.get()).toEqual([expect.objectContaining({ id: 's1', profile: 'meta' })])
  })

  it('resolveSessionProfile routes a default-profile session from a non-default gateway', async () => {
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 's1', profile: 'default' }))

    await expect(resolveSessionProfile('s1')).resolves.toBe('default')
  })
})
