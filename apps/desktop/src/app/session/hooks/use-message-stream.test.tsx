import { act, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $currentModel,
  $currentProvider,
  $turnStartedAt,
  $workingSessionIds,
  setCurrentModel,
  setCurrentProvider,
  setTurnStartedAt
} from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../types'
import { useMessageStream } from './use-message-stream'

vi.mock('@/lib/completion-sound', () => ({ playCompletionSound: vi.fn() }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/session-sync', () => ({ broadcastSessionsChanged: vi.fn() }))
vi.mock('@/store/composer-status', () => ({ refreshBackgroundProcesses: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))
vi.mock('@/store/onboarding', () => ({ requestDesktopOnboarding: vi.fn() }))
vi.mock('@/store/updates', () => ({ reportBackendContract: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/buffer', () => ({ readActiveTerminal: vi.fn(() => '') }))

interface HarnessProps {
  activeSessionIdRef: MutableRefObject<string | null>
  onReady: (handleGatewayEvent: (event: RpcEvent) => void) => void
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
}

function Harness({ activeSessionIdRef, onReady, sessionStateByRuntimeIdRef }: HarnessProps) {
  const { handleGatewayEvent } = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: { invalidateQueries: vi.fn() } as never,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const previous = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)
      const next = updater({ ...previous, messages: previous.messages })
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      return next
    }
  })

  onReady(handleGatewayEvent)
  return null
}

describe('useMessageStream — background session.info running state', () => {
  beforeEach(() => {
    $workingSessionIds.set([])
    setTurnStartedAt(null)
    setCurrentModel('foreground-model')
    setCurrentProvider('foreground-provider')
  })

  afterEach(() => {
    vi.useRealTimers()
    $workingSessionIds.set([])
    setTurnStartedAt(null)
    setCurrentModel('')
    setCurrentProvider('')
    vi.clearAllMocks()
  })

  it('updates a background session running flag without mirroring its metadata to the foreground view', () => {
    let handleGatewayEvent!: (event: RpcEvent) => void
    const sessionStateByRuntimeIdRef = { current: new Map<string, ClientSessionState>() }

    render(
      <Harness
        activeSessionIdRef={{ current: 'foreground-runtime' }}
        onReady={handler => (handleGatewayEvent = handler)}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    handleGatewayEvent({
      type: 'session.info',
      session_id: 'background-runtime',
      payload: {
        model: 'anthropic/claude-opus-4.8',
        provider: 'anthropic',
        running: true
      }
    } as RpcEvent)

    const background = sessionStateByRuntimeIdRef.current.get('background-runtime')
    expect(background?.busy).toBe(true)
    expect(background?.turnStartedAt).toEqual(expect.any(Number))

    // Foreground-only statusbar/model atoms must not be contaminated by a background chat.
    expect($currentModel.get()).toBe('foreground-model')
    expect($currentProvider.get()).toBe('foreground-provider')
    expect($turnStartedAt.get()).toBeNull()
  })

  it('does not show a missing-response error when running false races before message.complete', () => {
    vi.useFakeTimers()
    let handleGatewayEvent!: (event: RpcEvent) => void
    const sessionStateByRuntimeIdRef = { current: new Map<string, ClientSessionState>() }

    render(
      <Harness
        activeSessionIdRef={{ current: 'foreground-runtime' }}
        onReady={handler => (handleGatewayEvent = handler)}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    act(() => {
      handleGatewayEvent({ type: 'message.start', session_id: 'foreground-runtime', payload: {} } as RpcEvent)
      handleGatewayEvent({
        type: 'session.info',
        session_id: 'foreground-runtime',
        payload: { running: false }
      } as RpcEvent)
    })

    let state = sessionStateByRuntimeIdRef.current.get('foreground-runtime')
    expect(state?.busy).toBe(false)
    expect(state?.awaitingResponse).toBe(true)
    expect(state?.messages.some(message => message.error?.includes('without delivering a response event'))).toBe(false)

    act(() => {
      handleGatewayEvent({
        type: 'message.complete',
        session_id: 'foreground-runtime',
        payload: { text: 'ok from gpt-5.5' }
      } as RpcEvent)
      vi.advanceTimersByTime(2_000)
    })

    state = sessionStateByRuntimeIdRef.current.get('foreground-runtime')
    expect(state?.busy).toBe(false)
    expect(state?.awaitingResponse).toBe(false)
    expect(state?.messages.some(message => message.error?.includes('without delivering a response event'))).toBe(false)
  })

  it('shows the missing-response recovery error only after the settle grace expires', () => {
    vi.useFakeTimers()
    let handleGatewayEvent!: (event: RpcEvent) => void
    const sessionStateByRuntimeIdRef = { current: new Map<string, ClientSessionState>() }

    render(
      <Harness
        activeSessionIdRef={{ current: 'foreground-runtime' }}
        onReady={handler => (handleGatewayEvent = handler)}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    act(() => {
      handleGatewayEvent({ type: 'message.start', session_id: 'foreground-runtime', payload: {} } as RpcEvent)
      handleGatewayEvent({
        type: 'session.info',
        session_id: 'foreground-runtime',
        payload: { running: false }
      } as RpcEvent)
    })

    let state = sessionStateByRuntimeIdRef.current.get('foreground-runtime')
    expect(state?.messages.some(message => message.error?.includes('without delivering a response event'))).toBe(false)

    act(() => {
      vi.advanceTimersByTime(1_500)
    })

    state = sessionStateByRuntimeIdRef.current.get('foreground-runtime')
    expect(state?.busy).toBe(false)
    expect(state?.awaitingResponse).toBe(false)
    expect(state?.messages.some(message => message.error?.includes('without delivering a response event'))).toBe(true)
  })
})
