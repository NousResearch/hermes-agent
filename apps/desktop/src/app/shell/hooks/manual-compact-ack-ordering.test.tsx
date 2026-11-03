import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { CompressSessionOutcome } from '@/app/session/compress-session'
import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $compactingSessions, takeCompressDeferred } from '@/store/compaction'
import { $notifications, clearNotifications } from '@/store/notifications'

import { _resetManualCompactingForTests, useCompactConversation } from './use-compact-conversation'

const originalRuntime = 'runtime-ack-original'
const recoveredRuntime = 'runtime-ack-recovered'
const otherRuntime = 'runtime-ack-other'
const endings = ['compacted', 'ready', 'error'] as const
const cases = endings.flatMap(kind => [originalRuntime, recoveredRuntime].map(runtime => ({ kind, runtime })))

beforeEach(() => {
  _resetManualCompactingForTests()
  $compactingSessions.set({})
  clearNotifications()

  for (const id of [originalRuntime, recoveredRuntime, otherRuntime]) {
    takeCompressDeferred(id)
  }
})

afterEach(() => {
  cleanup()
  _resetManualCompactingForTests()
  $compactingSessions.set({})
  clearNotifications()
  vi.restoreAllMocks()
})

it.each(cases)(
  'does not rearm after $kind precedes the pending acknowledgement on $runtime',
  async ({ kind, runtime }) => {
    let finish: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          finish = resolve
        })
    )

    const stream = renderMessageStream(runtime, {
      states: new Map([[runtime, { ...createClientSessionState(), storedSessionId: 'bot-chat-root' }]])
    })

    const { result, rerender } = renderHook(
      ({ id }) => useCompactConversation({ busy: false, compress, sessionId: id, storedSessionId: 'bot-chat-root' }),
      { initialProps: { id: originalRuntime } }
    )

    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())
    rerender({ id: runtime })
    expect(result.current.pending).toBe(true)

    await act(async () => {
      stream.handleEvent({ type: 'status.update', session_id: runtime, payload: { kind: 'compressing' } })

      if (kind === 'error') {
        stream.handleEvent({
          type: 'error',
          session_id: runtime,
          payload: { message: 'compression failed: host stopped' }
        })
        stream.handleEvent({ type: 'status.update', session_id: runtime, payload: { kind: 'ready' } })
      } else {
        stream.handleEvent({
          type: 'status.update',
          session_id: runtime,
          payload: { kind, text: kind === 'compacted' ? 'Context compacted' : '' }
        })
      }

      finish({ kind: 'deferred', message: 'still running in the background', sessionId: runtime })
    })

    expect(result.current.pending).toBe(false)
    expect($compactingSessions.get()[runtime]).toBeUndefined()
    expect(takeCompressDeferred(runtime)).toBe(false)
    expect($notifications.get().some(notice => notice.kind === 'success')).toBe(kind === 'compacted')
    expect($notifications.get().some(notice => notice.message === 'still running in the background')).toBe(false)
    expect(compress).toHaveBeenCalledTimes(1)
  }
)

it('does not mistake another conversation ending for this request completing', async () => {
  let finish: (outcome: CompressSessionOutcome) => void = () => undefined

  const compress = vi.fn(
    () =>
      new Promise<CompressSessionOutcome>(resolve => {
        finish = resolve
      })
  )

  const stream = renderMessageStream(originalRuntime)

  const { result } = renderHook(() =>
    useCompactConversation({
      busy: false,
      compress,
      sessionId: originalRuntime,
      storedSessionId: 'bot-chat-root'
    })
  )

  act(() => result.current.onRequest())
  act(() => result.current.onConfirm())
  await act(async () => {
    stream.handleEvent({ type: 'status.update', session_id: otherRuntime, payload: { kind: 'compacted' } })
    finish({ kind: 'deferred', message: 'still running in the background', sessionId: originalRuntime })
  })

  expect(result.current.pending).toBe(true)
  expect($compactingSessions.get()[originalRuntime]).toBe(true)
  await act(async () => {
    stream.handleEvent({ type: 'status.update', session_id: originalRuntime, payload: { kind: 'compacted' } })
  })
  expect(result.current.pending).toBe(false)
  expect(takeCompressDeferred(originalRuntime)).toBe(false)
  expect($notifications.get().filter(notice => notice.kind === 'success')).toHaveLength(1)
})

it.each(cases)('settles $kind after acknowledgement on its resolved $runtime', async ({ kind, runtime }) => {
  const compress = vi.fn().mockResolvedValue({
    kind: 'deferred',
    message: 'still running in the background',
    sessionId: runtime
  } satisfies CompressSessionOutcome)

  const stream = renderMessageStream(runtime)

  const { result, rerender } = renderHook(
    ({ id }) => useCompactConversation({ busy: false, compress, sessionId: id, storedSessionId: 'bot-chat-root' }),
    { initialProps: { id: originalRuntime } }
  )

  act(() => result.current.onRequest())
  await act(async () => result.current.onConfirm())
  rerender({ id: runtime })
  expect(result.current.pending).toBe(true)
  expect($notifications.get().find(notice => notice.id === `session-compress:${runtime}`)?.kind).toBe('info')

  await act(async () => {
    if (kind === 'error') {
      stream.handleEvent({
        type: 'error',
        session_id: runtime,
        payload: { message: 'compression failed: host stopped' }
      })
      stream.handleEvent({ type: 'status.update', session_id: runtime, payload: { kind: 'ready' } })
    } else {
      stream.handleEvent({ type: 'status.update', session_id: runtime, payload: { kind } })
    }
  })

  expect(result.current.pending).toBe(false)
  expect(takeCompressDeferred(runtime)).toBe(false)
  expect($notifications.get().some(notice => notice.kind === 'success')).toBe(kind !== 'error')

  if (runtime !== originalRuntime) {
    expect($notifications.get().some(notice => notice.id === `session-compress:${originalRuntime}`)).toBe(false)
  }
})

it('ignores a same-session ready edge from before this compaction starts', async () => {
  let finish: (outcome: CompressSessionOutcome) => void = () => undefined

  const compress = vi.fn(
    () =>
      new Promise<CompressSessionOutcome>(resolve => {
        finish = resolve
      })
  )

  const stream = renderMessageStream(originalRuntime)

  const { result } = renderHook(() =>
    useCompactConversation({
      busy: false,
      compress,
      sessionId: originalRuntime,
      storedSessionId: 'bot-chat-root'
    })
  )

  act(() => result.current.onRequest())
  act(() => result.current.onConfirm())
  await act(async () => {
    stream.handleEvent({ type: 'status.update', session_id: originalRuntime, payload: { kind: 'ready' } })
    finish({ kind: 'deferred', message: 'still running in the background', sessionId: originalRuntime })
  })

  expect(result.current.pending).toBe(true)
  expect($notifications.get().some(notice => notice.kind === 'success')).toBe(false)
  await act(async () => {
    stream.handleEvent({ type: 'status.update', session_id: originalRuntime, payload: { kind: 'compressing' } })
    stream.handleEvent({ type: 'status.update', session_id: originalRuntime, payload: { kind: 'compacted' } })
  })
  expect(result.current.pending).toBe(false)
  expect($notifications.get().filter(notice => notice.kind === 'success')).toHaveLength(1)
})
