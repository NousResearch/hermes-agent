import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { getSession } from '@/hermes'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import { $cronSessions, $messagingSessions, $sessions, $unlistedSessionOwnerRows } from '@/store/session'
import { $removedSessionIds, tombstoneSessions, untombstoneSessions } from '@/store/session-removal'
import type { SessionInfo } from '@/types/hermes'

import { cachedSessionRow, resolveStoredSession } from './utils'

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
    $sessions.set([])
    $projectTree.set([])
    $removedSessionIds.set(new Set())
    $profiles.set(profiles('default', 'meta'))
    $activeGatewayProfile.set('meta')
    $unlistedSessionOwnerRows.set([])
    mockGetSession.mockReset()
  })

  afterEach(() => {
    $cronSessions.set([])
    $messagingSessions.set([])
    $sessions.set([])
    $projectTree.set([])
    $removedSessionIds.set(new Set())
    $profiles.set([])
    $activeGatewayProfile.set('default')
    $unlistedSessionOwnerRows.set([])
  })

  it('preserves omitted rich metadata from the exact owner while direct values and explicit clears win', async () => {
    for (const clearPreview of [false, true]) {
      $sessions.set([])
      let release!: (value: SessionInfo) => void
      mockGetSession.mockReturnValueOnce(
        new Promise<SessionInfo>(resolve => {
          release = resolve
        })
      )
      const pending = resolveStoredSession('s1', {
        connectionId: 'remote-1',
        profile: 'meta',
        targetProfile: 'meta'
      } as never)
      expect(mockGetSession).toHaveBeenLastCalledWith('s1', { connectionId: 'remote-1', profile: 'meta' })
      $sessions.set([
        session({
          id: 's1',
          connection_id: ' remote-1 ',
          profile: ' meta ',
          title: 'Listed title',
          preview: 'E2E_DELEGATE_CARD_A',
          last_active: 123,
          message_count: 18,
          archived: true,
          source: 'desktop'
        })
      ])
      const direct = session({
        id: 's1',
        title: null,
        message_count: 0,
        archived: false,
        source: 'cli',
        ...(clearPreview ? { preview: null } : {})
      })
      release(direct)
      expect(await pending).toBe(direct)
      expect($sessions.get()).toHaveLength(1)
      expect($sessions.get()[0]).toMatchObject({
        id: 's1',
        connection_id: 'remote-1',
        profile: 'meta',
        preview: clearPreview ? null : 'E2E_DELEGATE_CARD_A',
        last_active: 123,
        title: null,
        message_count: 0,
        archived: false,
        source: 'cli'
      })
      // Cache enrichment must not change what the direct resolver returns.
      expect(direct.last_active).toBeUndefined()
      if (!clearPreview) expect(direct.preview).toBeUndefined()
    }
  })

  it('does not borrow rich metadata across canonical IDs or explicit owners', async () => {
    for (const [listed, connectionId] of [
      [session({ id: 's1', connection_id: 'foreign', profile: 'meta' }), 'remote-1'],
      [session({ id: 's1', connection_id: 'remote-1', profile: 'foreign' }), 'remote-1'],
      [session({ id: 's1', profile: 'meta' }), 'remote-1'],
      [session({ id: 's1', connection_id: 'remote-1', profile: 'meta' }), ''],
      [session({ id: 'other', connection_id: 'remote-1', profile: 'meta' }), 'remote-1']
    ] as const) {
      $sessions.set([])
      let release!: (value: SessionInfo) => void
      mockGetSession.mockReturnValueOnce(
        new Promise<SessionInfo>(resolve => {
          release = resolve
        })
      )
      const pending = resolveStoredSession('s1', { connectionId, profile: 'meta', targetProfile: 'meta' } as never)
      $sessions.set([{ ...listed, preview: 'Foreign preview', last_active: 456 }])
      release(session({ id: 's1', title: 'Direct title', message_count: 2, archived: false }))
      const resolved = await pending
      const cached = $sessions.get().find(row => row.id === 's1')!
      expect(cached.preview).toBeUndefined()
      expect(cached.last_active).toBeUndefined()
      expect(cached).toMatchObject({ id: 's1', title: 'Direct title', connection_id: connectionId, profile: 'meta' })
      expect(resolved?.preview).toBeUndefined()
    }
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

  it('routes a moved resolve into its current slice instead of duplicating it', async () => {
    // Cross-room /resume rewrote the row to source='matrix' (#113827): the
    // stale regular-sessions copy must go and the row must land in messaging.
    $sessions.set([session({ id: 's1' })])
    mockGetSession.mockResolvedValueOnce(session({ id: 's1', profile: 'meta', source: 'matrix' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.source).toBe('matrix')
    expect($sessions.get()).toEqual([])
    expect($messagingSessions.get().map(s => s.id)).toEqual(['s1'])
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

  it('parks a hidden by-id hit off-list — canonical Bot Chats never get a sidebar row (#113273)', async () => {
    // Opening a bot's chat resolves it by id; the backend row carries
    // hidden=true. Listed, it would paint a Sessions row the refresh
    // keep-list then holds open indefinitely.
    mockGetSession.mockResolvedValueOnce(session({ hidden: true, id: 's1' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.hidden).toBe(true)
    expect($sessions.get()).toEqual([])
    // owner resolution still finds the row on the off-list stub atom
    expect($unlistedSessionOwnerRows.get().find(s => s.id === 's1')?.hidden).toBe(true)
  })

  it('returns an internal delegate child without promoting it into regular sessions', async () => {
    mockGetSession.mockResolvedValueOnce(session({ id: 'child', is_internal_child: true }))

    const resolved = await resolveStoredSession('child')

    expect(resolved).toMatchObject({ id: 'child', is_internal_child: true, profile: 'meta' })
    expect($sessions.get()).toEqual([])
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

  it('does not recache a by-id row while its session is tombstoned (#85163)', async () => {
    // The archive row click's bubbled resume raced the tombstone: the by-id
    // resolve started just before the archive, and its response must not
    // re-insert the row the archive optimistically evicted.
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

  it('does not recache an archived by-id row', async () => {
    // The direct lookup can also observe the archive itself: the backend row
    // already carries archived=true while the tombstone is still settling.
    mockGetSession.mockResolvedValueOnce(session({ archived: true, id: 's1' }))

    const resolved = await resolveStoredSession('s1')

    expect(resolved?.archived).toBe(true)
    expect($sessions.get()).toEqual([])
  })

  it('does not recache a stale by-id row when its tombstone clears before the response', async () => {
    // A failed archive rolls the row back (untombstone) while the by-id
    // request is still in flight: the response raced the doomed row even
    // though membership looks untouched.
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
    // Tombstone added AND removed while the request was in flight: membership
    // is back to empty, but the generation moved, so the response is stale.
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

  it('does not recache a stale cross-profile by-id row after an in-flight tombstone cycle', async () => {
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

  it('does not recache a stale owner-routed by-id row after an in-flight tombstone cycle', async () => {
    let resolveRequest!: (value: SessionInfo) => void
    mockGetSession.mockReturnValueOnce(
      new Promise<SessionInfo>(resolve => {
        resolveRequest = resolve
      })
    )

    const pending = resolveStoredSession('s1', {
      connectionId: 'remote-1',
      profile: 'meta',
      targetProfile: 'meta'
    } as never)

    tombstoneSessions(['s1'])
    untombstoneSessions(['s1'])

    resolveRequest(session({ archived: false, id: 's1' }))

    await expect(pending).resolves.toMatchObject({ connection_id: 'remote-1', id: 's1', profile: 'meta' })
    expect(mockGetSession).toHaveBeenCalledWith('s1', { connectionId: 'remote-1', profile: 'meta' })
    expect($sessions.get()).toEqual([])
  })
})

describe('cachedSessionRow owner preference', () => {
  const projectNode = (sessions: SessionInfo[], preview: SessionInfo[] = []) =>
    ({
      previewSessions: preview,
      repos: [{ groups: [{ sessions }] }]
    }) as never

  beforeEach(() => {
    $cronSessions.set([])
    $messagingSessions.set([])
    $sessions.set([])
    $projectTree.set([])
    mockGetSession.mockReset()
  })

  afterEach(() => {
    $cronSessions.set([])
    $messagingSessions.set([])
    $sessions.set([])
    $projectTree.set([])
  })

  it('prefers a self-describing project-tree row over an ownerless Recents duplicate', () => {
    // The same conversation, listed twice: a legacy Recents row with no owner
    // and the profile-scoped project-tree row the gateway stamped. Picking the
    // Recents copy throws away the only routing information there is, and the
    // branch then creates its child on whichever backend is active.
    $sessions.set([session({ cwd: '/wrong', id: 's1' })])
    $projectTree.set([projectNode([session({ connection_id: 'pandora', cwd: '/right', id: 's1', profile: 'work' })])])

    expect(cachedSessionRow('s1')).toMatchObject({ connection_id: 'pandora', cwd: '/right', profile: 'work' })
  })

  it('finds a project-tree preview row when the session is in no other list', () => {
    $projectTree.set([projectNode([], [session({ connection_id: 'rigremote', id: 's1', profile: 'default' })])])

    expect(cachedSessionRow('s1')).toMatchObject({ connection_id: 'rigremote' })
  })

  it('keeps the plain Recents row when nothing carries an owner', () => {
    $sessions.set([session({ cwd: '/only', id: 's1' })])

    expect(cachedSessionRow('s1')).toMatchObject({ cwd: '/only' })
    expect(cachedSessionRow('missing')).toBeUndefined()
  })
})
