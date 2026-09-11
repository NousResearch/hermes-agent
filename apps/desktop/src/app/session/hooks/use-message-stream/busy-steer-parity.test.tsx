import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $busyInputConfig, busyInputOwnerKey } from '@/store/busy-input-mode'
import { clearQueuedPrompts, getQueuedPrompts } from '@/store/composer-queue'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, setGatewayState } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import type { RpcEvent } from '@/types/hermes'

import { STREAM_DELTA_FLUSH_MS } from './utils'

import { useMessageStream } from './index'

const SID = 'steer-order-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let busySend: ((text: string) => Promise<boolean>) | null = null
let slash: ((text: string) => Promise<void>) | null = null
let states: Map<string, ClientSessionState>

/** Scripted command acknowledgement; the production hooks own stream handling. */
const requestGatewayMock = vi.fn(async (method: string): Promise<unknown> =>
  method === 'command.dispatch' ? { type: 'exec', output: 'Steer queued: inspect reconnect' } : {}
)

const requestGateway = requestGatewayMock as unknown as <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number
) => Promise<T>

function Harness() {
  const activeSessionIdRef = useRef<null | string>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())
  const busyRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const selectedStoredSessionIdRef = useRef<null | string>(SID)

  const updateSessionState = (sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
    const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
    const next = updater(current)
    sessionStateByRuntimeIdRef.current.set(sessionId, next)

    return next
  }

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  const actions = usePromptActions({
    activeSessionId: SID,
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef,
    createBackendSessionForSend: async () => SID,
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => null,
    getRouteToken: () => 'token',
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: () => undefined,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState
  })

  const { submitText, executeSlashCommand } = actions

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
    busySend = text => submitText(text, { busyInput: true, attachments: [], sessionId: SID, composerScope: SID })
    slash = executeSlashCommand
    states = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent, submitText, executeSlashCommand])

  return null
}

async function mountHarness() {
  vi.useFakeTimers()
  render(<Harness />)
  await act(async () => {
    await Promise.resolve()
  })
}

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

const emit = (event: RpcEvent) => act(() => handleEvent?.(event))

describe('configured Send shares explicit steering behaviour', () => {
  beforeEach(() => {
    states = new Map()
    setGatewayState('open')
    $busyInputConfig.set({
      connection: $connection.get(),
      owner: busyInputOwnerKey($connection.get()?.connectionId, $activeGatewayProfile.get()),
      mode: 'steer'
    })
    requestGatewayMock.mockClear()
  })
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    $busyInputConfig.set(null)
  })

  it.each(['interrupt', 'queue'] as const)('retains configured %s through normal submit', async mode => {
    $busyInputConfig.set({ ...$busyInputConfig.get()!, mode })
    await mountHarness()
    $sessionStates.set({ [SID]: { ...createClientSessionState(), busy: true, storedSessionId: SID } })
    requestGatewayMock.mockClear()

    if (mode === 'interrupt') {
      requestGatewayMock.mockResolvedValueOnce({ status: 'redirected' })
    }

    await act(async () => {
      expect(await busySend!('follow this direction')).toBe(true)
    })

    if (mode === 'interrupt') {
      expect(requestGatewayMock).toHaveBeenCalledExactlyOnceWith('session.redirect', {
        session_id: SID,
        text: 'follow this direction'
      })
    } else {
      expect(requestGatewayMock).not.toHaveBeenCalled()
      expect(getQueuedPrompts(SID)).toEqual([expect.objectContaining({ text: 'follow this direction' })])
    }

    clearQueuedPrompts(SID)
    $sessionStates.set({})
  })

  it.each(['automatic', 'explicit'] as const)('%s steering leaves the active stream and tools intact', async entry => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'reading the first file' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({
      payload: { args: { path: 'file.ts' }, name: 'read_file', tool_id: 'tool-1' },
      session_id: SID,
      type: 'tool.start'
    })
    const before = states.get(SID)!
    const activeMessage = before.messages.find(message => message.id === before.streamId)
    expect(activeMessage).toBeDefined()
    requestGatewayMock.mockClear()

    await act(async () => {
      if (entry === 'automatic') {
        expect(await busySend!('inspect reconnect')).toBe(true)
      } else {
        await slash!('/steer inspect reconnect')
      }
    })

    expect(requestGatewayMock).toHaveBeenCalledExactlyOnceWith('command.dispatch', {
      session_id: SID,
      name: 'steer',
      arg: 'inspect reconnect'
    })
    const after = states.get(SID)!
    expect(after.streamId).toBe(before.streamId)
    expect(after.busy).toBe(before.busy)
    expect(after.turnStartedAt).toBe(before.turnStartedAt)
    expect(after.messages.find(message => message.id === before.streamId)).toBe(activeMessage)
    expect(after.messages.filter(message => message.role === 'system').map(chatMessageText)).toEqual([
      'slash:/steer\nSteer queued: inspect reconnect'
    ])
    expect(after.messages.filter(message => message.role === 'user')).toHaveLength(0)

    emit({
      payload: { name: 'read_file', result: 'read completed', tool_id: 'tool-1' },
      session_id: SID,
      type: 'tool.complete'
    })
    emit({ payload: { text: ' and checking reconnect next' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({
      payload: { text: 'reading the first file and checking reconnect next' },
      session_id: SID,
      type: 'message.complete'
    })
    const settled = states.get(SID)!
    expect(settled.messages.some(message => chatMessageText(message).includes('checking reconnect next'))).toBe(true)
    expect(settled.messages.every(message => !message.pending)).toBe(true)
  })
})
