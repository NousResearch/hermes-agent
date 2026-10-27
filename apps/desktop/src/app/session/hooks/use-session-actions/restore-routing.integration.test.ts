import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from '@/api/client'
import { getLatestSessionMessages, getSession, getSessionMessages } from '@/api/sessions'
import { fetchHistoryWindow } from '@/app/chat/history-window'
import { fetchTimelineIndex } from '@/components/assistant-ui/thread/timeline-index'
import type { HermesApiRequest } from '@/global'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint
} from '@/store/session'
import { knownOwnerForSession } from '@/store/session-states'
import type { ProfileInfo } from '@/types/hermes'

import { probeStoredSession } from './utils'

const remoteOwner = { connectionId: 'route-test-ssh', profile: 'desktop-alias', targetProfile: 'backend-work' }
const localOwner = { connectionId: 'local', profile: 'default' }
const missing = () => new Error('404: Session not found')
const profiles = (...names: string[]) => names.map(name => ({ name }) as ProfileInfo)

// Keep the production REST scope and owner resolver. Only the preload transport
// is substituted: every assertion observes the request Electron would route.
function setupRestoreRouting() {
  const api = vi.fn<(request: HermesApiRequest) => Promise<unknown>>()

  beforeEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $unlistedSessionOwnerRows.set([])
    $projectTree.set([])
    $activeGatewayProfile.set('default')
    $profiles.set(profiles('default', 'memory-canary'))
    setApiRequestLocalMode(true)
    setApiRequestConnection('local')
    setApiRequestProfile('default')
    api.mockReset()
    api.mockRejectedValue(missing())
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  })

  afterEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $unlistedSessionOwnerRows.set([])
    $projectTree.set([])
    $profiles.set([])
    $activeGatewayProfile.set('default')
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    setApiRequestLocalMode(false)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  return api
}

describe('restore routing across the real REST scope boundary', () => {
  const api = setupRestoreRouting()

  it('probes a known remote owner once with its backend profile and fails closed', async () => {
    setSessionOwnerHint('restore-id', remoteOwner)
    expect(knownOwnerForSession('restore-id')).toEqual(remoteOwner)

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'gone' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'route-test-ssh', profile: 'backend-work', path: '/api/sessions/restore-id?profile=backend-work' }
    ])
  })

  it('repeated background failures never probe unrelated SSH/profile candidates', async () => {
    setSessionOwnerHint('restore-id', localOwner)
    api.mockRejectedValue(new Error('backend unavailable'))

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' },
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })

  it('preserves the captured local inventory across an ambient switch and a late foreign hint', async () => {
    api.mockImplementationOnce(async () => {
      setApiRequestConnection('route-test-ssh')
      $activeGatewayProfile.set('remote-only')
      $profiles.set(profiles('remote-only', 'unrelated'))
      setSessionOwnerHint('restore-id', remoteOwner)
      throw missing()
    })
    api.mockResolvedValueOnce({ id: 'restore-id' })

    await expect(probeStoredSession('restore-id')).resolves.toMatchObject({
      status: 'found',
      session: { id: 'restore-id', connection_id: 'local', profile: 'memory-canary' }
    })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' },
      { connectionId: 'local', profile: 'memory-canary', path: '/api/sessions/restore-id?profile=memory-canary' }
    ])
  })

  it.each([false, true])('abandons a legacy restore after a source switch (late hint: %s)', async lateHint => {
    setApiRequestConnection(null)
    api.mockImplementationOnce(async () => {
      setApiRequestConnection('route-test-ssh')
      $profiles.set(profiles('remote-only'))

      if (lateHint) {
        setSessionOwnerHint('restore-id', remoteOwner)
      }

      throw missing()
    })

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })

  it('does not adopt a legacy result after its source changes while awaiting', async () => {
    setApiRequestConnection(null)
    api.mockImplementationOnce(async () => {
      setApiRequestConnection('route-test-ssh')

      return { id: 'restore-id' }
    })

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect($sessions.get()).toEqual([])
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })

  it('abandons a legacy sweep when a late exact owner appears without an ambient switch', async () => {
    setApiRequestConnection(null)
    api.mockImplementationOnce(async () => {
      setSessionOwnerHint('restore-id', remoteOwner)
      throw missing()
    })

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })

  it('keeps stable legacy profile probes untagged for per-profile remote overrides', async () => {
    setApiRequestConnection(null)
    api.mockRejectedValueOnce(missing())
    api.mockResolvedValueOnce({ id: 'restore-id' })

    await expect(probeStoredSession('restore-id')).resolves.toMatchObject({
      status: 'found',
      session: { id: 'restore-id', profile: 'memory-canary' }
    })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { profile: 'default', path: '/api/sessions/restore-id?profile=default' },
      { profile: 'memory-canary', path: '/api/sessions/restore-id?profile=memory-canary' }
    ])
  })

  it('stamps a successful exact owner with its Desktop identity, not the backend alias', async () => {
    setSessionOwnerHint('restore-id', remoteOwner)
    api.mockResolvedValue({ id: 'restore-id', profile: 'backend-work' })

    await expect(probeStoredSession('restore-id')).resolves.toEqual({
      status: 'found',
      session: { id: 'restore-id', connection_id: 'route-test-ssh', profile: 'desktop-alias' }
    })
    expect(api.mock.calls[0][0]).toMatchObject({ connectionId: 'route-test-ssh', profile: 'backend-work' })
  })

  it('an explicit restore tuple takes precedence over a foreign owner hint', async () => {
    setSessionOwnerHint('restore-id', remoteOwner)
    api.mockResolvedValue({ id: 'restore-id' })

    await expect(probeStoredSession('restore-id', localOwner)).resolves.toEqual({
      status: 'found',
      session: { id: 'restore-id', connection_id: 'local', profile: 'default' }
    })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })

  it.each(['local', 'route-test-other'])(
    'preserves explicit %s REST tuples despite a foreign hint',
    async connectionId => {
      setSessionOwnerHint('restore-id', remoteOwner)
      api.mockResolvedValue({ id: 'restore-id' })
      await getSession('restore-id', { connectionId, profile: 'memory-canary' })
      expect(api.mock.calls[0][0]).toEqual({
        connectionId,
        profile: 'memory-canary',
        path: '/api/sessions/restore-id?profile=memory-canary'
      })
    }
  )

  it.each(['local', 'route-test-ssh'])('keeps an exact owner tuple intact with %s ambient', async ambient => {
    setSessionOwnerHint('restore-id', remoteOwner)
    setApiRequestConnection(ambient)
    api.mockResolvedValue({ id: 'restore-id' })

    await getSession('restore-id', 'memory-canary')
    await getSessionMessages('restore-id', 'memory-canary')
    expect(
      api.mock.calls.map(([request]) => ({
        connectionId: request.connectionId,
        profile: request.profile,
        path: request.path
      }))
    ).toEqual([
      {
        connectionId: 'route-test-ssh',
        profile: 'backend-work',
        path: '/api/sessions/restore-id?profile=backend-work'
      },
      {
        connectionId: 'route-test-ssh',
        profile: 'backend-work',
        path: '/api/sessions/restore-id/messages?profile=backend-work'
      }
    ])
  })

  it('an empty starting inventory stays inconclusive even if another source appears mid-probe', async () => {
    $profiles.set([])
    api.mockImplementationOnce(async () => {
      $profiles.set(profiles('foreign'))
      throw missing()
    })
    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })
})

describe('restore route stability and sibling reads', () => {
  const api = setupRestoreRouting()

  it.each(['success', 'failure'] as const)('abandons a legacy inventory change after %s', async outcome => {
    setApiRequestConnection(null)
    api.mockImplementationOnce(async () => {
      $profiles.set(profiles('foreign'))

      if (outcome === 'failure') {
        throw missing()
      }

      return { id: 'restore-id' }
    })

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api).toHaveBeenCalledTimes(1)
    expect($sessions.get()).toEqual([])
  })

  it.each(['success', 'failure'] as const)('abandons a legacy ambient-mode change after %s', async outcome => {
    setApiRequestConnection(null)
    api.mockImplementationOnce(async () => {
      setApiRequestLocalMode(false)

      if (outcome === 'failure') {
        throw missing()
      }

      return { id: 'restore-id' }
    })

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api).toHaveBeenCalledTimes(1)
    expect($sessions.get()).toEqual([])
  })

  it.each(['success', 'failure'] as const)(
    'rejects a late legacy owner on the remaining profile after %s',
    async outcome => {
      setApiRequestConnection(null)
      api.mockRejectedValueOnce(missing())
      api.mockImplementationOnce(async () => {
        setSessionOwnerHint('restore-id', remoteOwner)

        if (outcome === 'failure') {
          throw missing()
        }

        return { id: 'restore-id' }
      })

      await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
      expect(api.mock.calls.map(([request]) => request)).toEqual([
        { profile: 'default', path: '/api/sessions/restore-id?profile=default' },
        { profile: 'memory-canary', path: '/api/sessions/restore-id?profile=memory-canary' }
      ])
      expect($sessions.get()).toEqual([])
    }
  )

  it.each([
    ['404: Session not found', 'gone'],
    ['404: Not Found', 'inconclusive'],
    ['500: Session not found', 'inconclusive'],
    ['backend unavailable', 'inconclusive']
  ])('classifies an exact owner failure %s as %s without fallback', async (message, status) => {
    setSessionOwnerHint('restore-id', remoteOwner)
    api.mockRejectedValue(new Error(message))

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status })
    expect(api).toHaveBeenCalledTimes(1)
    expect(api.mock.calls[0][0]).toEqual({
      connectionId: 'route-test-ssh',
      profile: 'backend-work',
      path: '/api/sessions/restore-id?profile=backend-work'
    })
  })

  it('keeps a mixed-failure unknown-owner sweep inconclusive', async () => {
    api.mockRejectedValueOnce(new Error('backend unavailable'))

    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'inconclusive' })
    expect(api).toHaveBeenCalledTimes(2)
  })

  it('declares an unknown owner gone only after the captured inventory misses', async () => {
    await expect(probeStoredSession('restore-id')).resolves.toEqual({ status: 'gone' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' },
      { connectionId: 'local', profile: 'memory-canary', path: '/api/sessions/restore-id?profile=memory-canary' }
    ])
  })

  it('honors an explicit failed restore tuple without falling back to its foreign hint', async () => {
    setSessionOwnerHint('restore-id', remoteOwner)

    await expect(probeStoredSession('restore-id', localOwner)).resolves.toEqual({ status: 'gone' })
    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { connectionId: 'local', profile: 'default', path: '/api/sessions/restore-id?profile=default' }
    ])
  })

  it.each(['local', 'route-test-ssh'])(
    'routes all transcript read siblings through the owner with %s ambient',
    async ambient => {
      const id = `transcript-${ambient}`
      setSessionOwnerHint(id, remoteOwner)
      setApiRequestConnection(ambient)
      api.mockResolvedValue({
        messages: [],
        entries: [],
        pagination: {
          order: 'latest',
          limit: 120,
          offset: 0,
          has_older: false,
          has_newer: false,
          has_more: false,
          next_cursor: null
        }
      })

      await getLatestSessionMessages(id, 'memory-canary')
      await fetchTimelineIndex(id, 'memory-canary')
      await fetchHistoryWindow(id, 7, 'memory-canary', new AbortController().signal)

      expect(
        api.mock.calls.map(([request]) => ({
          connectionId: request.connectionId,
          profile: request.profile,
          path: request.path
        }))
      ).toEqual([
        {
          connectionId: 'route-test-ssh',
          profile: 'backend-work',
          path: `/api/sessions/${id}/messages?profile=backend-work&limit=120&order=latest&include_compacted=true`
        },
        {
          connectionId: 'route-test-ssh',
          profile: 'backend-work',
          path: `/api/sessions/${id}/timeline?limit=500&profile=backend-work`
        },
        {
          connectionId: 'route-test-ssh',
          profile: 'backend-work',
          path: `/api/sessions/${id}/messages/around?row_id=7&limit=120&profile=backend-work`
        }
      ])
    }
  )
})
