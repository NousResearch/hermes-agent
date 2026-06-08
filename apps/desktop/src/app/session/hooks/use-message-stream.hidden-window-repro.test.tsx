import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $messages, $turnStartedAt, setAwaitingResponse, setBusy, setMessages, setTurnStartedAt } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'
import { useSessionStateCache } from './use-session-state-cache'

type StreamApi = ReturnType<typeof useMessageStream>

const SESSION_ID = 'runtime-session-1'
const STORED_SESSION_ID = 'stored-session-1'

interface HarnessProps {
  onReady: (api: StreamApi) => void
}

function Harness({ onReady }: HarnessProps) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const { activeSessionIdRef, updateSessionState } = useSessionStateCache({
    activeSessionId: SESSION_ID,
    busyRef,
    selectedStoredSessionId: STORED_SESSION_ID,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const api = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: async () => undefined,
    queryClient: { invalidateQueries: async () => undefined } as never,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined,
    updateSessionState
  })

  onReady(api)

  return null
}

function event(type: string, payload?: Record<string, unknown>): RpcEvent {
  return {
    type,
    session_id: SESSION_ID,
    payload
  }
}

describe('useMessageStream hidden-window repro', () => {
  let originalHiddenDescriptor: PropertyDescriptor | undefined
  let isDocumentHidden: boolean
  let queuedFrames: Array<{ callback: FrameRequestCallback; id: number }>
  let cancelledFrames: Set<number>
  let nextFrameHandle: number
  let now: number

  beforeEach(() => {
    cleanup()
    setMessages([])
    setBusy(false)
    setAwaitingResponse(false)
    setTurnStartedAt(null)
    isDocumentHidden = false
    queuedFrames = []
    cancelledFrames = new Set()
    nextFrameHandle = 1
    now = 100

    originalHiddenDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, 'hidden')

    Object.defineProperty(document, 'hidden', {
      configurable: true,
      get: () => isDocumentHidden
    })

    vi.spyOn(performance, 'now').mockImplementation(() => now)
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback: FrameRequestCallback) => {
      const id = nextFrameHandle++
      queuedFrames.push({ callback, id })

      return id
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((id: number) => {
      cancelledFrames.add(id)
      queuedFrames = queuedFrames.filter(frame => frame.id !== id)
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setMessages([])
    setBusy(false)
    setAwaitingResponse(false)
    setTurnStartedAt(null)

    if (originalHiddenDescriptor) {
      Object.defineProperty(Document.prototype, 'hidden', originalHiddenDescriptor)
    } else {
      delete (document as Document & { hidden?: boolean }).hidden
    }
  })

  async function flushAnimationFrame() {
    await act(async () => {
      const callbacks = [...queuedFrames]
      queuedFrames = []
      now += 16

      for (const { callback, id } of callbacks) {
        if (!cancelledFrames.has(id)) {
          callback(now)
        }
      }
    })
  }

  it('keeps the visible transcript empty until completion when view-sync rAF is paused', async () => {
    let api: StreamApi | null = null

    render(<Harness onReady={ready => (api = ready)} />)

    isDocumentHidden = true

    await act(async () => {
      api!.handleGatewayEvent(event('message.start'))
      api!.handleGatewayEvent(event('message.delta', { text: 'thinking out loud' }))
      api!.handleGatewayEvent(event('reasoning.delta', { text: 'planning' }))
    })

    // The turn is alive, but the shared transcript never sees the queued state
    // because both the delta flush and the session->view publish are waiting on
    // a requestAnimationFrame callback that never fires in this repro harness.
    expect($turnStartedAt.get()).not.toBeNull()
    expect($messages.get()).toEqual([])

    await act(async () => {
      api!.handleGatewayEvent(event('message.complete', { text: 'final answer' }))
    })

    // Completion finalizes the session state immediately, but the visible
    // transcript is still blocked behind the stalled hidden-window rAF.
    expect($messages.get()).toEqual([])

    await flushAnimationFrame()

    expect($messages.get()).toHaveLength(1)
    expect($messages.get()[0]?.role).toBe('assistant')
    expect($messages.get()[0]?.pending).toBe(false)
    expect($messages.get()[0]?.parts).toEqual([
      { type: 'reasoning', text: 'planning' },
      { type: 'text', text: 'final answer' }
    ])
  })

  it('can require a second foreground frame after returning, even before the next bubble completes', async () => {
    let api: StreamApi | null = null

    render(<Harness onReady={ready => (api = ready)} />)

    await act(async () => {
      api!.handleGatewayEvent(event('message.start'))
    })

    await flushAnimationFrame()
    expect($messages.get()).toEqual([])

    isDocumentHidden = true

    await act(async () => {
      api!.handleGatewayEvent(event('message.delta', { text: 'draft answer' }))
      api!.handleGatewayEvent(event('reasoning.delta', { text: 'thinking' }))
    })

    expect($messages.get()).toEqual([])

    isDocumentHidden = false

    await flushAnimationFrame()

    // First foreground frame only drains the queued stream deltas into the
    // per-session cache. Publishing that cache into the visible transcript is
    // itself staged behind the NEXT frame.
    expect($messages.get()).toEqual([])

    await flushAnimationFrame()

    expect($messages.get()).toHaveLength(1)
    expect($messages.get()[0]?.role).toBe('assistant')
    expect($messages.get()[0]?.pending).toBe(true)
    expect($messages.get()[0]?.parts).toEqual([
      { type: 'text', text: 'draft answer' },
      { type: 'reasoning', text: 'thinking' }
    ])
  })

  it('does not need another gateway event once the queued foreground publish frame runs', async () => {
    let api: StreamApi | null = null

    render(<Harness onReady={ready => (api = ready)} />)

    await act(async () => {
      api!.handleGatewayEvent(event('message.start'))
    })

    await flushAnimationFrame()

    isDocumentHidden = true

    await act(async () => {
      api!.handleGatewayEvent(event('message.delta', { text: 'draft answer' }))
    })

    isDocumentHidden = false

    await flushAnimationFrame()
    expect($messages.get()).toEqual([])

    await flushAnimationFrame()

    expect($messages.get()).toHaveLength(1)
    expect($messages.get()[0]?.parts).toEqual([{ type: 'text', text: 'draft answer' }])
  })

  it('publishes hidden-window progress via timer fallback even if no animation frame runs', async () => {
    let api: StreamApi | null = null

    render(<Harness onReady={ready => (api = ready)} />)

    isDocumentHidden = true

    await act(async () => {
      api!.handleGatewayEvent(event('message.start'))
      api!.handleGatewayEvent(event('message.delta', { text: 'draft answer' }))
      api!.handleGatewayEvent(event('reasoning.delta', { text: 'thinking' }))
    })

    expect($messages.get()).toEqual([])

    await waitFor(() => {
      expect($messages.get()).toHaveLength(1)
    })

    expect($messages.get()[0]?.pending).toBe(true)
    expect($messages.get()[0]?.parts).toEqual([
      { type: 'text', text: 'draft answer' },
      { type: 'reasoning', text: 'thinking' }
    ])
  })
})
