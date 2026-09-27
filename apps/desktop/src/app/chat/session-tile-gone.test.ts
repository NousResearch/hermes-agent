import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSession } from '@/hermes'
import { probeStoredSession } from '@/app/session/hooks/use-session-actions/utils'
import { setApiRequestConnection } from '@/api/client'
import { stashSessionDraft, clearSessionDraft, takeSessionDraft } from '@/store/composer'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $gatewayState, $sessions } from '@/store/session'
import { $sessionTiles, openSessionTile, reopenLastClosedTile, closeSessionTile } from '@/store/session-states'

import { startUnrestoredTileTitleBackfill } from './session-tile'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof import('@/hermes')>()),
  getSession: vi.fn()
}))

const get = vi.mocked(getSession)
let stop: (() => void) | undefined

beforeEach(() => {
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  $profiles.set([{ name: 'default' }, { name: 'writer' }] as never)
  $connection.set({ connectionId: 'local', mode: 'local' } as never)
  $sessions.set([])
  $sessionTiles.set([])
  get.mockReset()
})

afterEach(() => {
  stop?.()
  $gatewayState.set('idle')
  $sessionTiles.set([])
  $sessions.set([])
  $profiles.set([])
  $connection.set(null)
  $gatewaySwitching.set(false)
  clearSessionDraft('deleted-chat')
  window.localStorage.clear()
})

describe('restored dead tile backfill', () => {
  it('does not resurrect a dead tile ahead of a user-closed tab on reopen', async () => {
    openSessionTile('user-closed')
    closeSessionTile('user-closed')
    openSessionTile('deleted-chat')
    get.mockRejectedValue(new Error('404: Session not found'))
    stop = startUnrestoredTileTitleBackfill()
    $gatewayState.set('open')
    await vi.waitFor(() => expect($sessionTiles.get()).toEqual([]))
    reopenLastClosedTile()
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['user-closed'])
  })

  it('releases scope subscriptions on cancellation even while lookup is pending', async () => {
    openSessionTile('deleted-chat')
    const released = vi.fn()
    const listen = $connection.listen.bind($connection)
    const spy = vi.spyOn($connection, 'listen').mockImplementation(listener => {
      const off = listen(listener)
      return () => { released(); off() }
    })
    let resolve!: (result: { status: 'gone' }) => void
    const pending = new Promise<{ status: 'gone' }>(yes => { resolve = yes })
    try {
      stop = startUnrestoredTileTitleBackfill(() => pending)
      $gatewayState.set('open')
      expect(spy).toHaveBeenCalledTimes(1)
      stop()
      expect(released).toHaveBeenCalledTimes(1)
      resolve({ status: 'gone' })
      await pending
      await new Promise(yes => setTimeout(yes, 0))
      expect(released).toHaveBeenCalledTimes(1)
      expect($sessionTiles.get()).toHaveLength(1)
    } finally {
      resolve({ status: 'gone' })
      spy.mockRestore()
    }
  })

  it('persists each dead tile removal without dropping a surviving owned tile', async () => {
    openSessionTile('gone-one')
    openSessionTile('gone-two')
    openSessionTile('survivor', 'right', undefined, undefined, {
      workspaceMode: 'sessions', ownerRoute: { connectionId: 'local', profile: 'writer' }
    })
    get.mockImplementation(async (id: string) => {
      if (id === 'survivor') return { id, title: 'Still here' } as never
      throw new Error('404: Session not found')
    })
    stop = startUnrestoredTileTitleBackfill()
    $gatewayState.set('open')
    await vi.waitFor(() => expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['survivor']))
    const persisted = window.localStorage.getItem('hermes.desktop.sessionTiles.v2') ?? ''
    expect(persisted).toContain('survivor')
    expect(persisted).not.toContain('gone-one')
    expect(persisted).not.toContain('gone-two')
    expect($sessionTiles.get()[0].ownerRoute).toEqual({ connectionId: 'local', profile: 'writer' })
  })
  it('uses the real REST scope ladder and preserves caller-selected ownership', async () => {
    const actual = await vi.importActual<typeof import('@/hermes')>('@/hermes')
    const previous = window.hermesDesktop
    const api = vi.fn(async (_request: { path: string }) => { throw new Error('404: {"detail":"Session not found"}') })
    window.hermesDesktop = { ...previous, api } as never
    setApiRequestConnection('local')
    get.mockImplementation(actual.getSession)
    try {
      expect(await probeStoredSession('deleted-chat')).toEqual({ status: 'gone' })
      expect(api.mock.calls.map(([request]) => (request as { path: string }).path)).toEqual([
        '/api/sessions/deleted-chat?profile=default', '/api/sessions/deleted-chat?profile=writer'
      ])
      api.mockClear()
      expect(await probeStoredSession('deleted-chat', { connectionId: 'remote', profile: 'alias', targetProfile: 'actual' })).toEqual({ status: 'gone' })
      expect(api).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ connectionId: 'remote', profile: 'actual', path: '/api/sessions/deleted-chat?profile=actual' }))
    } finally {
      window.hermesDesktop = previous
      setApiRequestConnection(null)
    }
  })

  it.each(['500: Session not found', '404: endpoint unavailable', 'network disconnected'])(
    'keeps inconclusive misses: %s',
    async message => {
      openSessionTile('deleted-chat')
      get.mockRejectedValueOnce(new Error(message)).mockRejectedValue(new Error('404: Session not found'))
      stop = startUnrestoredTileTitleBackfill()
      $gatewayState.set('open')
      await vi.waitFor(() => expect(get).toHaveBeenCalledTimes(2))
      expect($sessionTiles.get()).toHaveLength(1)
    }
  )

  it('preserves a working cross-profile fallback and fills its title', async () => {
    openSessionTile('deleted-chat')
    get
      .mockRejectedValueOnce(new Error('network disconnected'))
      .mockResolvedValueOnce({ id: 'deleted-chat', title: 'Found' } as never)
    stop = startUnrestoredTileTitleBackfill()
    $gatewayState.set('open')
    await vi.waitFor(() => expect($sessions.get()[0]?.title).toBe('Found'))
    expect($sessionTiles.get()).toHaveLength(1)
  })

  it.each(['draft', 'switch', 'profile ABA', 'cancel', 'bound', 'no inventory', 'other backend'])(
    'preserves tiles when unsafe: %s',
    async reason => {
      openSessionTile(
        'deleted-chat',
        'right',
        undefined,
        undefined,
        reason === 'other backend' ? { workspaceMode: 'sessions', ownerRoute: { connectionId: 'remote', profile: 'writer' } } : undefined
      )
      if (reason === 'draft') stashSessionDraft('deleted-chat', 'keep my words', [])
      if (reason === 'switch') $gatewaySwitching.set(true)
      if (reason === 'no inventory') $profiles.set([])
      let reject!: (error: Error) => void
      get.mockRejectedValue(new Error('404: Session not found')).mockImplementationOnce(
        () =>
          new Promise((_resolve, no) => {
            reject = no
          })
      )
      stop = startUnrestoredTileTitleBackfill()
      $gatewayState.set('open')
      if (reason === 'profile ABA') {
        $activeGatewayProfile.set('writer')
        $activeGatewayProfile.set('default')
      }
      if (reason === 'cancel') stop()
      if (reason === 'bound') $sessionTiles.set($sessionTiles.get().map(tile => ({ ...tile, runtimeId: 'live' })))
      reject(new Error('404: Session not found'))
      await vi.waitFor(() =>
        expect(get).toHaveBeenCalledTimes(reason === 'other backend' || reason === 'no inventory' ? 1 : 2)
      )
      expect($sessionTiles.get()).toHaveLength(1)
      if (reason === 'draft') expect(takeSessionDraft('deleted-chat').text).toBe('keep my words')
    }
  )

  it('persistently closes an empty tile after every profile confirms the session is gone', async () => {
    openSessionTile('deleted-chat')
    get.mockRejectedValue(new Error('404: Session not found'))
    stop = startUnrestoredTileTitleBackfill()
    $gatewayState.set('open')

    await vi.waitFor(() => expect(get).toHaveBeenCalledTimes(2))
    await vi.waitFor(() => expect($sessionTiles.get()).toEqual([]))
    expect(window.localStorage.getItem('hermes.desktop.sessionTiles.v2') ?? '').not.toContain('deleted-chat')
    expect(get.mock.calls.map(call => call[1])).toEqual(['default', 'writer'])
  })
})
