import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $currentUsage, $localDeviceName, $sessionActivityStatus } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'

let handleEvent: (event: RpcEvent) => void = () => undefined

function MessageStreamHarness({ activeSessionId = 'session-1' }: { activeSessionId?: string }) {
  const activeSessionIdRef = useRef<string | null>(activeSessionId)
  const queryClientRef = useRef(new QueryClient())
  const statesRef = useRef(new Map<string, ClientSessionState>())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    updateSessionState: (sessionId, updater) => {
      const previous = statesRef.current.get(sessionId) ?? createClientSessionState(null)
      const next = updater(previous)
      statesRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

describe('useMessageStream token usage events', () => {
  beforeEach(() => {
    handleEvent = () => undefined
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
  })

  afterEach(() => {
    cleanup()
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
    vi.restoreAllMocks()
  })

  it('updates current usage from token.usage before message.complete', () => {
    render(<MessageStreamHarness />)

    act(() =>
      handleEvent({
        payload: {},
        session_id: 'session-1',
        type: 'message.start'
      } as RpcEvent)
    )

    act(() =>
      handleEvent({
        payload: {
          context_length: 131_072,
          context_pct: 49.9,
          context_tokens: 65_432,
          input_tokens: 1_200,
          output_tokens: 34,
          total_tokens: 1_234
        },
        session_id: 'session-1',
        type: 'token.usage'
      } as RpcEvent)
    )

    expect($currentUsage.get()).toMatchObject({
      context_max: 131_072,
      context_percent: 49.9,
      context_used: 65_432,
      input: 1_200,
      output: 34,
      total: 1_234
    })
  })

  it('ignores token.usage events for inactive sessions', () => {
    $currentUsage.set({ calls: 1, input: 10, output: 5, total: 15 })
    render(<MessageStreamHarness />)

    act(() =>
      handleEvent({
        payload: {
          context_length: 131_072,
          context_tokens: 70_000,
          input_tokens: 70_000,
          total_tokens: 70_000
        },
        session_id: 'session-2',
        type: 'token.usage'
      } as RpcEvent)
    )

    expect($currentUsage.get()).toEqual({ calls: 1, input: 10, output: 5, total: 15 })
  })
})

describe('useMessageStream status.update events', () => {
  beforeEach(() => {
    handleEvent = () => undefined
    $sessionActivityStatus.set(null)
  })

  afterEach(() => {
    cleanup()
    $sessionActivityStatus.set(null)
    vi.restoreAllMocks()
  })

  const statusEvent = (text: string, kind = 'lifecycle', sessionId = 'session-1') =>
    ({
      payload: { kind, text },
      session_id: sessionId,
      type: 'status.update'
    }) as RpcEvent

  it('surfaces lifecycle statuses for the active session', () => {
    render(<MessageStreamHarness />)

    act(() => handleEvent(statusEvent('📦 Preflight compression: ~90,000 tokens. This may take a moment.')))

    expect($sessionActivityStatus.get()).toEqual({
      kind: 'lifecycle',
      text: '📦 Preflight compression: ~90,000 tokens. This may take a moment.'
    })
  })

  it('ignores statuses from inactive sessions and unknown kinds', () => {
    render(<MessageStreamHarness />)

    act(() => handleEvent(statusEvent('background noise', 'lifecycle', 'session-2')))
    expect($sessionActivityStatus.get()).toBeNull()

    act(() => handleEvent(statusEvent('voice things', 'voice')))
    expect($sessionActivityStatus.get()).toBeNull()
  })

  it('clears on ready kind and on stream activity', () => {
    render(<MessageStreamHarness />)

    act(() => handleEvent(statusEvent('⠋ compressing 120 messages', 'compressing')))
    expect($sessionActivityStatus.get()).not.toBeNull()

    act(() => handleEvent(statusEvent('ready', 'ready')))
    expect($sessionActivityStatus.get()).toBeNull()

    act(() => handleEvent(statusEvent('📦 Compression complete: ~90,000 → ~30,000 tokens.')))
    expect($sessionActivityStatus.get()).not.toBeNull()

    act(() =>
      handleEvent({
        payload: { text: 'hello' },
        session_id: 'session-1',
        type: 'message.delta'
      } as RpcEvent)
    )
    expect($sessionActivityStatus.get()).toBeNull()
  })
})

describe('useMessageStream gateway.ready device identity', () => {
  beforeEach(() => {
    handleEvent = () => undefined
    $localDeviceName.set('')
  })

  afterEach(() => {
    cleanup()
    $localDeviceName.set('')
    vi.restoreAllMocks()
  })

  const ready = (deviceName?: string) =>
    ({ payload: deviceName === undefined ? {} : { device_name: deviceName }, type: 'gateway.ready' }) as RpcEvent

  it('captures the first ready frame as this device (first-wins)', () => {
    render(<MessageStreamHarness />)

    act(() => handleEvent(ready('ko-mac')))
    expect($localDeviceName.get()).toBe('ko-mac')

    // A later ready frame — e.g. a REMOTE backend connecting — must not
    // overwrite this device's identity with the peer's name.
    act(() => handleEvent(ready('ko-win11')))
    expect($localDeviceName.get()).toBe('ko-mac')
  })

  it('ignores ready frames without a usable device name', () => {
    render(<MessageStreamHarness />)

    act(() => handleEvent(ready()))
    act(() => handleEvent(ready('   ')))
    expect($localDeviceName.get()).toBe('')

    act(() => handleEvent(ready('ko-mac')))
    expect($localDeviceName.get()).toBe('ko-mac')
  })
})
