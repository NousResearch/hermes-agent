import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { type MutableRefObject, useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { useSessionStateCache } from '../use-session-state-cache'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

import { useMessageStream } from './index'

const SID = 'session-1'
let stream: MessageStreamHarness
let states: Map<string, ClientSessionState>
type UpdateSessionState = (
  sessionId: string,
  updater: (state: ClientSessionState) => ClientSessionState,
  storedSessionId?: string | null
) => ClientSessionState
let updateSessionState: ReturnType<typeof vi.fn<UpdateSessionState>>

/** The shared harness, but writing through a spy so the unmount test can count
 *  the writes that happen after teardown. */
function mountStream() {
  stream = renderMessageStream(SID, { states, updateSessionState })
}

const assistantText = () => stream.text()

describe('useMessageStream delta flush scheduling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
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

    act(() => stream.appendDelta(SID, 'still streaming'))

    expect(window.requestAnimationFrame).not.toHaveBeenCalled()
    expect(assistantText()).toBe('')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(assistantText()).toBe('still streaming')
  })

  it('keeps distinct reasoning item ids separate when they flush together', async () => {
    mountStream()
    act(() =>
      stream.handleEvent({
        type: 'reasoning.delta',
        session_id: SID,
        payload: { reasoning_id: 'reasoning-1', text: 'Inspecting' }
      })
    )
    act(() =>
      stream.handleEvent({
        type: 'reasoning.delta',
        session_id: SID,
        payload: { reasoning_id: 'reasoning-2', text: 'Comparing' }
      })
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const reasoning =
      states
        .get(SID)
        ?.messages.at(-1)
        ?.parts.filter(part => part.type === 'reasoning') ?? []
    expect(reasoning.map(part => part.text)).toEqual(['Inspecting', 'Comparing'])
    expect(reasoning.map(part => part.sourceId)).toEqual(['reasoning-1', 'reasoning-2'])
  })

  it('settles a reasoning source before later reuse so causal order survives', async () => {
    mountStream()
    act(() =>
      stream.handleEvent({
        type: 'reasoning.delta',
        session_id: SID,
        payload: { reasoning_id: 'reasoning-1', text: 'First', timestamp: 1 }
      })
    )
    act(() =>
      stream.handleEvent({
        type: 'reasoning.end',
        session_id: SID,
        payload: { reasoning_id: 'reasoning-1', text: '', timestamp: 2 }
      })
    )
    act(() =>
      stream.handleEvent({ type: 'message.delta', session_id: SID, payload: { text: 'Between', timestamp: 3 } })
    )
    act(() =>
      stream.handleEvent({
        type: 'reasoning.delta',
        session_id: SID,
        payload: { reasoning_id: 'reasoning-1', text: 'Later', timestamp: 4 }
      })
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    const parts = states.get(SID)?.messages.at(-1)?.parts ?? []
    expect(parts.map(part => part.type)).toEqual(['reasoning', 'text', 'reasoning'])
    expect(parts.map(part => ('text' in part ? part.text : undefined))).toEqual(['First', 'Between', 'Later'])
    expect(parts[0]).toMatchObject({ sourceId: 'reasoning-1', completedAt: 2 })
    expect(parts[2]).toMatchObject({ sourceId: 'reasoning-1', timestamp: 4 })
  })

  it('flushes queued text immediately when a hidden window becomes visible', () => {
    vi.mocked(performance.now).mockReturnValue(0)
    mountStream()

    act(() => stream.appendDelta(SID, 'caught up on focus'))
    expect(assistantText()).toBe('')
    expect(vi.getTimerCount()).toBe(1)

    Object.defineProperty(globalThis.document, 'visibilityState', {
      configurable: true,
      value: 'visible'
    })

    act(() => globalThis.document.dispatchEvent(new Event('visibilitychange')))

    expect(assistantText()).toBe('caught up on focus')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('flushes queued text on focus when visibility remains visible', () => {
    vi.mocked(performance.now).mockReturnValue(0)
    Object.defineProperty(globalThis.document, 'visibilityState', {
      configurable: true,
      value: 'visible'
    })
    mountStream()

    act(() => stream.appendDelta(SID, 'focused without visibility change'))
    expect(assistantText()).toBe('')
    expect(vi.getTimerCount()).toBe(1)

    act(() => globalThis.window.dispatchEvent(new Event('focus')))

    expect(assistantText()).toBe('focused without visibility change')
  })

  it('cancels the pending timer on unmount and flushes exactly once', async () => {
    vi.mocked(performance.now).mockReturnValue(0)
    mountStream()

    act(() => stream.appendDelta(SID, 'final delta'))
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

describe('useMessageStream composed with the real useSessionStateCache', () => {
  // The tests above mock updateSessionState, so they validate the adaptive
  // arithmetic but not the production ordering contract: runFlush's
  // measurement rAF must be registered AFTER the view-sync rAF that the real
  // updateSessionState schedules inside syncSessionStateToView, so the
  // measured frame cost includes the deferred $messages commit it adapts to.
  let cache: ReturnType<typeof useSessionStateCache> | null = null
  let published: ChatMessage[]
  let appendAssistantDelta: ((sessionId: string, delta: string) => void) | null = null

  function ComposedHarness() {
    const busyRef: MutableRefObject<boolean> = { current: false }
    const queryClientRef = useRef(new QueryClient())

    const sessionCache = useSessionStateCache({
      activeSessionId: SID,
      busyRef,
      selectedStoredSessionId: null,
      setAwaitingResponse: () => undefined,
      setBusy: () => undefined,
      setMessages: messages => {
        published = messages
      }
    })

    const stream = useMessageStream({
      activeSessionIdRef: sessionCache.activeSessionIdRef,
      hydrateFromStoredSession: vi.fn(async () => undefined),
      queryClient: queryClientRef.current,
      refreshHermesConfig: vi.fn(async () => undefined),
      refreshSessions: vi.fn(async () => undefined),
      sessionStateByRuntimeIdRef: sessionCache.sessionStateByRuntimeIdRef,
      updateSessionState: sessionCache.updateSessionState
    })

    useEffect(() => {
      appendAssistantDelta = stream.appendAssistantDelta
      cache = sessionCache
    }, [stream.appendAssistantDelta, sessionCache])

    return null
  }

  function cachedText() {
    const message = cache?.sessionStateByRuntimeIdRef.current.get(SID)?.messages.at(-1)
    const part = message?.parts.at(-1)

    return part?.type === 'text' ? part.text : ''
  }

  function publishedText() {
    const part = published.at(-1)?.parts.at(-1)

    return part?.type === 'text' ? part.text : ''
  }

  beforeEach(() => {
    vi.useFakeTimers()
    appendAssistantDelta = null
    cache = null
    published = []
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

  it('measures the frame cost through the real view-sync rAF and adapts the next gap', async () => {
    let now = 1000
    vi.mocked(performance.now).mockImplementation(() => now)
    const rafCallbacks: FrameRequestCallback[] = []
    vi.mocked(window.requestAnimationFrame).mockImplementation(cb => {
      rafCallbacks.push(cb)

      return rafCallbacks.length
    })

    render(<ComposedHarness />)
    expect(appendAssistantDelta).not.toBeNull()

    // Mid-turn state: busy keeps the view sync on the deferred rAF path
    // (terminal/needing-input states flush synchronously instead).
    act(() => {
      cache!.updateSessionState(SID, state => ({ ...state, busy: true }))
    })
    expect(rafCallbacks).toHaveLength(1)
    // Drain the seed's own view-sync rAF so the flush below starts clean.
    act(() => rafCallbacks.shift()!(now))

    act(() => appendAssistantDelta!(SID, 'first'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    // The store write landed synchronously, but the $messages publish is
    // deferred: exactly two rAF callbacks are pending — first the cache's
    // view-sync, then runFlush's measurement.
    expect(cachedText()).toBe('first')
    expect(publishedText()).toBe('')
    expect(rafCallbacks).toHaveLength(2)

    // Draining the FIRST registered callback must be what publishes the
    // deferred commit; that identity is the ordering contract. It runs until
    // 60ms into the frame (React commit + Streamdown re-parse).
    now = 1100
    act(() => rafCallbacks[0](1040))
    expect(publishedText()).toBe('first')

    // The measurement callback closes the same frame: 60ms of in-frame work,
    // so the next adaptive floor is 3x = 180ms.
    act(() => rafCallbacks[1](1040))

    act(() => appendAssistantDelta!(SID, 'second'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(79)
    })

    expect(cachedText()).toBe('first')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })

    expect(cachedText()).toBe('firstsecond')
  })

  it('keeps the write-cost fallback when the parked renderer never fires rAF', async () => {
    let now = 1000
    vi.mocked(performance.now).mockImplementation(() => now)
    // Parked renderer: rAF callbacks are accepted but never run.
    vi.mocked(window.requestAnimationFrame).mockImplementation(() => 1)

    render(<ComposedHarness />)

    act(() => {
      cache!.updateSessionState(SID, state => ({ ...state, busy: true }))
    })

    act(() => appendAssistantDelta!(SID, 'first'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(cachedText()).toBe('first')

    // 100ms later (well past the 33ms floor): the next flush is immediate.
    now = 1100
    act(() => appendAssistantDelta!(SID, 'second'))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(cachedText()).toBe('firstsecond')
  })
})
