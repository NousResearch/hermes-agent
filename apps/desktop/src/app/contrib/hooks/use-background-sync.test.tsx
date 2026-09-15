import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $liveSessions, clearLiveSessions } from '@/store/live-sessions'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $activeSessionId } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { $sessionTiles } from '@/store/session-states'

import { useBackgroundSync } from './use-background-sync'

const noop = () => undefined
const requestGateway = async () => ({ sessions: [] })

function render(
  activeGatewayProfile: string,
  activeConnectionId: string,
  refreshSessions: () => Promise<void>,
  gatewayRequest = requestGateway
) {
  return renderHook(
    ({ connectionId, profile }: { connectionId: string; profile: string }) => {
      useBackgroundSync({
        activeConnectionId: connectionId,
        activeGatewayProfile: profile,
        activeIsMessaging: false,
        activeSessionId: null,
        activeStoredSessionId: null,
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript: noop,
        refreshCronJobs: noop,
        refreshCurrentModel: noop,
        refreshHermesConfig: noop,
        refreshMessagingSessions: noop,
        refreshSessions,
        requestGateway: gatewayRequest
      })
    },
    { initialProps: { connectionId: activeConnectionId, profile: activeGatewayProfile } }
  )
}

describe('useBackgroundSync profile-scoped session refresh', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $activeSessionId.set(null)
    $changeEventsAvailable.set(false)
    $cronChangeTick.set(0)
    $sessionsChangeTick.set(0)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('coalesces change ticks while the live status request is pending', async () => {
    $changeEventsAvailable.set(true)
    let release!: (value: { sessions: [] }) => void

    const pending = new Promise<{ sessions: [] }>(resolve => {
      release = resolve
    })

    const request = vi.fn(() => pending)
    render('default', 'local', async () => undefined, request)
    await act(async () => undefined)

    for (let tick = 1; tick <= 8; tick += 1) {
      await act(async () => {
        $sessionsChangeTick.set(tick)
      })
    }

    expect(request).toHaveBeenCalledTimes(1)
    await act(async () => {
      release({ sessions: [] })
    })
    expect(request).toHaveBeenCalledTimes(2)
  })

  it('refreshes the session list after the active gateway profile changes', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    const hook = render('default', 'local', refreshSessions)

    await act(async () => undefined)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
    refreshSessions.mockClear()

    hook.rerender({ connectionId: 'local', profile: 'nova' })

    await act(async () => undefined)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('refreshes the session list when the backend changes but the profile name does not', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    const hook = render('default', 'work', refreshSessions)

    await act(async () => undefined)
    refreshSessions.mockClear()

    hook.rerender({ connectionId: 'homelab', profile: 'default' })

    await act(async () => undefined)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })
})

/**
 * The same `session.active_list` poll also feeds the sidebar's live group
 * (#50799): a session created over the gateway by another client has no DB row
 * until its first prompt persists one, so the stored slices can never show it.
 * The poll response must land in `$liveSessions` — stamped with the poll's
 * connection and profile so owner resolution routes follow-up RPCs correctly.
 */
describe('useBackgroundSync — live-session reconcile feeding the poll', () => {
  const liveResponse = () => ({
    sessions: [
      {
        id: 'rt-foreign',
        last_active: 2_000,
        message_count: 0,
        model: 'zyphra/qwen',
        preview: 'first prompt',
        session_key: 'sess-foreign',
        source: 'cli',
        started_at: 1_000,
        status: 'working',
        title: 'Foreign chat'
      }
    ]
  })

  beforeEach(() => {
    vi.useFakeTimers()
    $activeSessionId.set(null)
    $changeEventsAvailable.set(false)
    $cronChangeTick.set(0)
    $sessionsChangeTick.set(0)
    clearLiveSessions()
    $liveSessions.set([])
    $removedSessionIds.set(new Set())
    $sessionTiles.set([])
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    clearLiveSessions()
    $liveSessions.set([])
  })

  it('populates $liveSessions from the poll, stamped with the poll route', async () => {
    const request = vi.fn(async () => liveResponse())
    render('workops', 'homelab', async () => undefined, request)

    await act(async () => undefined)

    const rows = $liveSessions.get()

    expect(rows).toHaveLength(1)
    expect(rows[0].id).toBe('sess-foreign')
    expect(rows[0].connection_id).toBe('homelab')
    expect(rows[0].profile).toBe('workops')
  })

  it('clears the live group when the gateway reports nothing live', async () => {
    let response: { sessions?: unknown[] } = liveResponse()
    const request = vi.fn(async () => response)
    render('default', 'local', async () => undefined, request)

    await act(async () => undefined)
    expect($liveSessions.get()).toHaveLength(1)

    // The session's first prompt persisted its DB row and the gateway reaped
    // the live entry — an authoritative empty snapshot clears the group (the
    // DB-backed list shows it now).
    response = { sessions: [] }
    await act(async () => {
      $sessionsChangeTick.set(1)
    })

    expect($liveSessions.get()).toEqual([])
  })

  it('leaves $liveSessions untouched when the poll fails (older gateway)', async () => {
    let failing = false

    const request = vi.fn(async () => {
      if (failing) {
        throw new Error('method not found')
      }

      return liveResponse()
    })

    render('default', 'local', async () => undefined, request)

    await act(async () => undefined)
    expect($liveSessions.get()).toHaveLength(1)

    failing = true
    await act(async () => {
      $sessionsChangeTick.set(1)
    })

    // A failed request is NO INFORMATION (the store's contract): the previous
    // snapshot survives until a good one re-asserts or clears it.
    expect($liveSessions.get()).toHaveLength(1)
  })
})
