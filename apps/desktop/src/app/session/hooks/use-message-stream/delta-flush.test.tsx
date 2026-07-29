import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'

import { useMessageStream } from './index'

const SID = 'session-1'
let appendAssistantDelta: ((sessionId: string, delta: string) => void) | null = null
let states: Map<string, ClientSessionState>
type UpdateSessionState = (
  sessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState,
  storedSessionId?: string | null
) => ClientSessionState
let updateSessionState: ReturnType<typeof vi.fn<UpdateSessionState>>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(states)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  useEffect(() => {
    appendAssistantDelta = stream.appendAssistantDelta
  }, [stream.appendAssistantDelta])

  return null
}

function mountStream() {
  render(<Harness />)
  expect(appendAssistantDelta).not.toBeNull()
}

function assistantText() {
  const message = states.get(SID)?.messages.at(-1)
  const part = message?.parts.at(-1)

  return part?.type === 'text' ? part.text : ''
}

describe('useMessageStream delta flush scheduling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    appendAssistantDelta = null
    states = new Map()
    updateSessionState = vi.fn((sessionId: string, updater: (state: ClientSessionState) => ClientSessionState) => {
      const next = updater(states.get(sessionId) ?? createClientSessionState())
      states.set(sessionId, next)

      return next
    })
    vi.spyOn(performance, 'now').mockReturnValue(100)
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('flushes streaming text on a bounded timer while the window is unfocused', async () => {
    mountStream()

    act(() => appendAssistantDelta!(SID, 'still streaming'))

    expect(window.requestAnimationFrame).not.toHaveBeenCalled()
    expect(assistantText()).toBe('')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(assistantText()).toBe('still streaming')
  })

  it('cancels the pending timer on unmount and flushes exactly once', async () => {
    vi.mocked(performance.now).mockReturnValue(0)
    mountStream()

    act(() => appendAssistantDelta!(SID, 'final delta'))
    expect(vi.getTimerCount()).toBe(1)

    cleanup()

    expect(vi.getTimerCount()).toBe(0)
    expect(assistantText()).toBe('final delta')
    const updatesAfterUnmount = updateSessionState.mock.calls.length

    await vi.advanceTimersByTimeAsync(100)

    expect(updateSessionState).toHaveBeenCalledTimes(updatesAfterUnmount)
    expect(window.requestAnimationFrame).not.toHaveBeenCalled()
  })
})
