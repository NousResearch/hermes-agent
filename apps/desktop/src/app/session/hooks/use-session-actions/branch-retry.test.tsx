import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { $notifications, dismissNotification } from '@/store/notifications'
import { $messages, setCurrentCwd, setMessages } from '@/store/session'
import type { ClientSessionState } from '../../../types'

import { useSessionActions } from '.'

// A backend restart mid-`session.create` silently drops the branch RPC
// (#65410) -- requestGateway rejects. The fix offers a one-click retry
// action on the resulting error notification instead of a dead-end toast.
async function actRender(ui: React.ReactElement) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(ui)
  })

  return result!
}

interface HarnessHandle {
  branchCurrentSession: () => Promise<boolean>
}

function Harness({
  onReady,
  requestGateway
}: {
  onReady: (handle: HarnessHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const activeSessionIdRef: MutableRefObject<string | null> = useRef('rt-active')
  const selectedStoredSessionIdRef: MutableRefObject<string | null> = useRef('stored-active')
  const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = useRef(new Map())
  const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = useRef(new Map())
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)

  const actions = useSessionActions({
    activeSessionId: 'rt-active',
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    navigate: () => undefined,
    requestGateway,
    resetViewSync: () => undefined,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: 'stored-active',
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: () => undefined,
    updateSessionState: (_sessionId, updater) => updater({} as ClientSessionState)
  })

  useEffect(() => {
    onReady({
      branchCurrentSession: (...args: Parameters<typeof actions.branchCurrentSession>) =>
        act(async () => actions.branchCurrentSession(...args)) as Promise<boolean>
    })
  }, [actions.branchCurrentSession, onReady])

  return null
}

describe('forkBranch retry action (#65410)', () => {
  beforeEach(() => {
    setMessages([
      {
        id: 'm1',
        role: 'user',
        parts: [textPart('branch me')]
      } as never
    ])
    setCurrentCwd('/repo')
    $notifications.set([])
  })

  afterEach(() => {
    cleanup()
    $messages.set([])
    $notifications.set([])
  })

  it('offers a retry action that re-issues session.create with the same args after a transport failure', async () => {
    const calls: Array<Record<string, unknown> | undefined> = []
    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method !== 'session.create') return {} as never

      calls.push(params)

      if (calls.length === 1) {
        throw new Error('backend restart mid-request')
      }

      return {
        session_id: 'rt-branch',
        stored_session_id: 'stored-branch',
        info: {}
      } as never
    })

    let handle!: HarnessHandle
    await actRender(<Harness onReady={h => (handle = h)} requestGateway={requestGateway} />)

    const firstResult = await handle.branchCurrentSession()

    expect(firstResult).toBe(false)
    expect(calls).toHaveLength(1)

    const notification = $notifications.get()[0]

    expect(notification).toBeDefined()
    expect(notification!.kind).toBe('error')
    expect(notification!.action).toBeDefined()
    expect(notification!.action!.label).toBe('Retry')

    // Simulate the user clicking the toast's retry button.
    await act(async () => {
      notification!.action!.onClick()
    })

    expect(calls).toHaveLength(2)
    // The retry carries the identical branch payload (same messages/cwd) --
    // no re-derivation, no chance of drifting from what the user branched.
    expect(calls[1]).toEqual(calls[0])

    dismissNotification(notification!.id)
  })
})
