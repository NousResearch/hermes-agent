import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import * as completionSound from '@/lib/completion-sound'
import { NATIVE_NOTIFICATION_KINDS, setNativeNotifyEnabled, setNativeNotifyKind } from '@/store/native-notifications'
import * as petStore from '@/store/pet'
import { setActiveSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-origin'
let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates: Map<string, ClientSessionState>
const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())
  sessionStates = sessionStateByRuntimeIdRef.current

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

describe('turn-origin event reconciliation', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('reconciles origin from session.info, message.start, and message.complete', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { running: true, turn_origin: 'notification' },
        session_id: SID,
        type: 'session.info'
      })
    )
    expect(sessionStates.get(SID)?.turnOrigin).toBe('notification')

    act(() =>
      handleEvent!({
        payload: { turn_origin: 'goal' },
        session_id: SID,
        type: 'message.start'
      })
    )
    expect(sessionStates.get(SID)?.turnOrigin).toBe('goal')

    act(() =>
      handleEvent!({
        payload: { text: 'done', turn_origin: 'user' },
        session_id: SID,
        type: 'message.complete'
      })
    )
    expect(sessionStates.get(SID)?.turnOrigin).toBe('user')

    act(() =>
      handleEvent!({
        payload: { running: false, turn_origin: null },
        session_id: SID,
        type: 'session.info'
      })
    )
    expect(sessionStates.get(SID)?.turnOrigin).toBeNull()
  })

  it('ignores an older session snapshot and completion after a newer turn starts', async () => {
    const playSound = vi.spyOn(completionSound, 'playCompletionSound').mockImplementation(() => undefined)
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { turn_generation: 8, turn_origin: 'notification' },
        session_id: SID,
        type: 'message.start'
      })
    )

    act(() =>
      handleEvent!({
        payload: { running: false, turn_generation: 7, turn_origin: null },
        session_id: SID,
        type: 'session.info'
      })
    )

    act(() =>
      handleEvent!({
        payload: { text: 'stale answer', turn_generation: 7, turn_origin: 'user' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect(sessionStates.get(SID)).toMatchObject({
      awaitingResponse: true,
      busy: true,
      turnGeneration: 8,
      turnOrigin: 'notification'
    })
    expect(playSound).not.toHaveBeenCalled()
  })

  it('restores a notification turn during reconnect and settles the same generation', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { running: true, turn_generation: 12, turn_origin: 'notification' },
        session_id: SID,
        type: 'session.info'
      })
    )

    expect(sessionStates.get(SID)).toMatchObject({
      busy: true,
      turnGeneration: 12,
      turnOrigin: 'notification'
    })

    act(() =>
      handleEvent!({
        payload: { text: 'background answer', turn_generation: 12, turn_origin: 'notification' },
        session_id: SID,
        type: 'message.complete'
      })
    )
    act(() =>
      handleEvent!({
        payload: { running: false, turn_generation: 12, turn_origin: null },
        session_id: SID,
        type: 'session.info'
      })
    )

    expect(sessionStates.get(SID)).toMatchObject({ busy: false, turnGeneration: 12, turnOrigin: null })
  })

  it('suppresses every completion feedback surface for an interrupted notification turn', async () => {
    const playSound = vi.spyOn(completionSound, 'playCompletionSound').mockImplementation(() => undefined)
    const celebrate = vi.spyOn(petStore, 'flashPetActivity').mockImplementation(() => undefined)
    const markUnread = vi.spyOn(petStore, 'markPetUnread').mockImplementation(() => undefined)
    const notifyNative = vi.fn().mockResolvedValue(true)
    desktopWindow.hermesDesktop = { notify: notifyNative } as unknown as Window['hermesDesktop']
    setNativeNotifyEnabled(true)

    for (const kind of NATIVE_NOTIFICATION_KINDS) {
      setNativeNotifyKind(kind, true)
    }

    setActiveSessionId(SID)
    Object.defineProperty(document, 'hidden', { configurable: true, value: false })
    Object.defineProperty(document, 'hasFocus', { configurable: true, value: () => false })

    await mountStream()
    sessionStates.set(SID, {
      ...createClientSessionState(),
      busy: true,
      interrupted: true,
      turnOrigin: 'notification'
    })

    act(() =>
      handleEvent!({
        payload: { status: 'interrupted', text: '', turn_origin: 'notification' },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect(playSound).not.toHaveBeenCalled()
    expect(celebrate).not.toHaveBeenCalled()
    expect(markUnread).not.toHaveBeenCalled()
    expect(notifyNative).not.toHaveBeenCalled()
    expect(sessionStates.get(SID)).toMatchObject({ busy: false, awaitingResponse: false })
  })

  afterEach(() => {
    setActiveSessionId(null)

    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    } else {
      delete desktopWindow.hermesDesktop
    }
  })
})
