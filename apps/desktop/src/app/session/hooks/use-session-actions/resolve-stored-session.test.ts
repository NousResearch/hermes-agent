import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { getSession } from '@/hermes'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import {
  dropListedSession,
  findListedSession,
  resolveSessionMutationProfile,
  resolveSessionProfile,
  resolveStoredSession,
  restoreListedSession
} from './utils'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getSession: vi.fn()
}))

const mockGetSession = vi.mocked(getSession)

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

const profiles = (...names: string[]) => names.map(name => ({ name }) as never)

describe('resolveStoredSession profile ownership', () => {
  beforeEach(() => {
    $sessions.set([])
    $messagingSessions.set([])
    $cronSessions.set([])
    $profiles.set(profiles('default', 'meta'))
    $activeGatewayProfile.set('meta')
    mockGetSession.mockReset()
  })

  afterEach(() => {
    $sessions.set([])
    $messagingSessions.set([])
    $cronSessions.set([])
    $profiles.set([])
    $activeGatewayProfile.set('default')
  })

  it('returns a cached row that carries an owning profile', async () => {
    $sessions.set([session({ id: 's1', profile: 'default' })])

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('default')
    expect(mockGetSession).not.toHaveBeenCalled()
  })

  it('reads owning profile from the messaging sidebar slice (#78836)', async () => {
    $messagingSessions.set([session({ id: 'tg-1', profile: 'winefox', source: 'telegram' })])

    const resolved = await resolveStoredSession('tg-1')

    expect(resolved?.profile).toBe('winefox')
    expect(mockGetSession).not.toHaveBeenCalled()
  })

  it('reads owning profile from the cron sidebar slice', async () => {
    $cronSessions.set([session({ id: 'cron-1', profile: 'worker', source: 'cron' })])

    const resolved = await resolveStoredSession('cron-1')

    expect(resolved?.profile).toBe('worker')
    expect(mockGetSession).not.toHaveBeenCalled()
  })

  it('treats a profile-less cache hit as unresolved when multiple profiles exist', async () => {
    $sessions.set([session({ id: 's1' })])
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 's1', profile: 'default' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.profile).toBe('default')
    // rung 2 (bare) then rung 3 (stamped cross-profile probe)
    expect(mockGetSession).toHaveBeenNthCalledWith(1, 's1')
    expect(mockGetSession).toHaveBeenNthCalledWith(2, 's1', 'default')
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

  it('resolveSessionProfile routes a default-profile session from a non-default gateway', async () => {
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 's1', profile: 'default' }))

    await expect(resolveSessionProfile('s1')).resolves.toBe('default')
  })
})

describe('findListedSession / drop / restore across sidebar slices', () => {
  beforeEach(() => {
    $sessions.set([])
    $messagingSessions.set([])
    $cronSessions.set([])
  })

  afterEach(() => {
    $sessions.set([])
    $messagingSessions.set([])
    $cronSessions.set([])
  })

  it('finds a messaging-platform session outside $sessions', () => {
    $messagingSessions.set([session({ id: 'qq-1', profile: 'winefox', source: 'qqbot' })])

    expect(findListedSession('qq-1')).toEqual({
      session: expect.objectContaining({ id: 'qq-1', profile: 'winefox' }),
      slice: 'messaging'
    })
  })

  it('drops a messaging row from its own slice on optimistic delete', () => {
    $messagingSessions.set([session({ id: 'tg-1', profile: 'winefox', source: 'telegram' })])
    $sessions.set([session({ id: 'desk-1', profile: 'default', source: 'desktop' })])

    dropListedSession('tg-1')

    expect($messagingSessions.get()).toEqual([])
    expect($sessions.get().map(s => s.id)).toEqual(['desk-1'])
  })

  it('restores a messaging row to the messaging slice on delete failure', () => {
    const row = session({ id: 'tg-1', profile: 'winefox', source: 'telegram' })

    restoreListedSession(row, 'messaging')

    expect($messagingSessions.get()).toEqual([row])
    expect($sessions.get()).toEqual([])
  })
})

describe('resolveSessionMutationProfile', () => {
  beforeEach(() => {
    $sessions.set([])
    $messagingSessions.set([])
    $cronSessions.set([])
    $profiles.set(profiles('default', 'winefox'))
    $activeGatewayProfile.set('default')
    mockGetSession.mockReset()
  })

  afterEach(() => {
    $sessions.set([])
    $messagingSessions.set([])
    $cronSessions.set([])
    $profiles.set([])
    $activeGatewayProfile.set('default')
  })

  it('prefers the listed row profile without probing', async () => {
    const listed = session({ id: 'tg-1', profile: 'winefox', source: 'telegram' })

    await expect(resolveSessionMutationProfile('tg-1', listed)).resolves.toBe('winefox')
    expect(mockGetSession).not.toHaveBeenCalled()
  })

  it('falls back to the ownership ladder when the listed row has no profile', async () => {
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 'tg-1', profile: 'winefox', source: 'telegram' }))

    await expect(resolveSessionMutationProfile('tg-1', session({ id: 'tg-1', source: 'telegram' }))).resolves.toBe(
      'winefox'
    )
  })

  it('resolves an uncached messaging id via the cross-profile ladder', async () => {
    mockGetSession.mockRejectedValueOnce(new Error('404: Session not found'))
    mockGetSession.mockResolvedValueOnce(session({ id: 'tg-1', profile: 'winefox', source: 'telegram' }))

    await expect(resolveSessionMutationProfile('tg-1')).resolves.toBe('winefox')
  })
})
