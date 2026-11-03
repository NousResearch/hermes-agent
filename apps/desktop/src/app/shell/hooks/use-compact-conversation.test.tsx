import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type CompressSessionOutcome } from '@/app/session/compress-session'
import { handleStatusEvent } from '@/app/session/hooks/use-message-stream/gateway-event/status'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { reconcileSessionCompacting, takeCompressDeferred } from '@/store/compaction'
import { $notifications, clearNotifications } from '@/store/notifications'

import { _resetManualCompactingForTests, useCompactConversation } from './use-compact-conversation'

const done = (message = 'compressed 12 → 2 messages'): CompressSessionOutcome => ({
  aborted: false,
  kind: 'done',
  message,
  result: {},
  sessionId: 'runtime-a'
})

function mount(
  compress: ((sessionId: string) => Promise<CompressSessionOutcome>) | undefined,
  sessionId = 'runtime-a',
  storedSessionId: null | string = 'stored-a'
) {
  return renderHook(
    ({ busy, id, stored }: { busy: boolean; id: null | string; stored?: null | string }) =>
      useCompactConversation({ busy, compress, sessionId: id, storedSessionId: stored }),
    { initialProps: { busy: false, id: sessionId, stored: storedSessionId } }
  )
}

beforeEach(() => {
  _resetManualCompactingForTests()
  clearNotifications()
})

afterEach(() => {
  _resetManualCompactingForTests()
  reconcileSessionCompacting('runtime-a', 'terminal')
  takeCompressDeferred('runtime-a')
  vi.restoreAllMocks()
})

describe('useCompactConversation', () => {
  it('does not rearm after the host completion arrives before the pending response', async () => {
    let settle!: (outcome: CompressSessionOutcome) => void
    const compress = vi.fn(() => new Promise<CompressSessionOutcome>(resolve => { settle = resolve }))
    const { result } = mount(compress)
    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())
    const payload = { kind: 'compacted', text: '✓ Context compression complete' } as GatewayEventContext['payload']

    const ctx = {
      deps: {
        compactedTurnRef: { current: new Set() }, flushQueuedDeltas: vi.fn(),
        sessionStateByRuntimeIdRef: { current: new Map() }, updateSessionState: vi.fn()
      },
      event: { type: 'status.update', session_id: 'runtime-a', payload }, payload,
      sessionId: 'runtime-a', isActiveEvent: false, occurredAt: 1
    } as unknown as GatewayEventContext

    act(() => handleStatusEvent(ctx))
    await act(async () => settle({ kind: 'deferred', message: 'pending', sessionId: 'runtime-a' }))
    expect(result.current.pending).toBe(false)
    expect(takeCompressDeferred('runtime-a')).toBe(false)
  })
  it('asks first, then compacts the chat it was opened on', async () => {
    const compress = vi.fn().mockResolvedValue(done())
    const { result } = mount(compress)

    expect(result.current.confirming).toBe(false)

    act(() => result.current.onRequest())
    expect(result.current.confirming).toBe(true)
    expect(compress).not.toHaveBeenCalled()

    await act(async () => result.current.onConfirm())

    expect(compress).toHaveBeenCalledTimes(1)
    expect(compress).toHaveBeenCalledWith('runtime-a')
    expect(result.current.pending).toBe(false)
  })

  it('keeps compressing the captured chat after the user navigates away', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    const { result, rerender } = mount(compress)

    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())
    expect(compress).toHaveBeenCalledWith('runtime-a')

    // Navigating away drops the QUESTION for the chat in view, but the running
    // claim belongs to runtime-a and must keep running.
    rerender({ busy: false, id: 'runtime-b', stored: 'stored-b' })
    expect(result.current.confirming).toBe(false)
    expect(result.current.pending).toBe(false)

    await act(async () => settle(done('compressed 40 → 4 messages')))

    expect(compress).toHaveBeenCalledTimes(1)
    expect(compress).toHaveBeenCalledWith('runtime-a')
  })

  it('shows a running claim for the chat that owns it, not the one in view', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    const { result, rerender } = mount(compress, 'runtime-a')

    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())

    rerender({ busy: false, id: 'runtime-a', stored: 'stored-a' })
    expect(result.current.pending).toBe(true)

    rerender({ busy: false, id: 'runtime-b', stored: 'stored-b' })
    expect(result.current.pending).toBe(false)

    await act(async () => settle(done()))
  })

  it('ignores a second submit instead of firing a second summarise call', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    const { result } = mount(compress)

    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())
    act(() => result.current.onConfirm())
    act(() => result.current.onRequest())

    expect(compress).toHaveBeenCalledTimes(1)
    expect(result.current.confirming).toBe(false)

    await act(async () => settle(done()))
  })

  it('keeps the running claim through a stale-runtime re-mint of the captured chat', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    const { result, rerender } = mount(compress, 'runtime-a', 'stored-a')

    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())
    expect(result.current.pending).toBe(true)

    // The captured runtime died mid-flight and recovery re-minted its id under
    // the same durable chat. The claim belongs to the CHAT, so the panel that
    // now renders the live id must still show the running state instead of
    // re-enabling the button over an in-flight summarise call.
    rerender({ busy: false, id: 'runtime-a-reminted', stored: 'stored-a' })

    expect(result.current.pending).toBe(true)
    expect(result.current.confirming).toBe(false)

    // A neighbouring chat never inherits it.
    rerender({ busy: false, id: 'runtime-b', stored: 'stored-b' })
    expect(result.current.pending).toBe(false)

    rerender({ busy: false, id: 'runtime-a-reminted', stored: 'stored-a' })

    await act(async () => settle(done()))
    expect(result.current.pending).toBe(false)
  })

  it('never paints a late failure from the captured chat under the one on screen', async () => {
    let fail: (err: unknown) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>((_resolve, reject) => {
          fail = reject
        })
    )

    const { result, rerender } = mount(compress, 'runtime-a', 'stored-a')

    act(() => result.current.onRequest())
    act(() => result.current.onConfirm())

    // The user moved to another chat while the summarise was out; the refusal
    // lands afterwards. It belongs to the chat that was compressed, never to
    // the chat whose popover happens to be open.
    rerender({ busy: false, id: 'runtime-b', stored: 'stored-b' })

    await act(async () => fail(new Error('session busy: compression')))

    expect(result.current.error).toBeNull()

    // The chat that actually failed keeps its own error.
    rerender({ busy: false, id: 'runtime-a', stored: 'stored-a' })
    expect(result.current.error).toBe('session busy: compression')
  })

  it('guards a same-frame double submit synchronously, before React commits', async () => {
    let settle: (outcome: CompressSessionOutcome) => void = () => undefined

    const compress = vi.fn(
      () =>
        new Promise<CompressSessionOutcome>(resolve => {
          settle = resolve
        })
    )

    const { result } = mount(compress)

    act(() => result.current.onRequest())

    // A double-click on a slow frame runs the SAME rendered handler twice, so
    // neither call observes the other's state write. The claim must be taken
    // and tested against the store itself, not the render snapshot.
    const confirm = result.current.onConfirm

    act(() => {
      confirm()
      confirm()
    })

    expect(compress).toHaveBeenCalledTimes(1)

    await act(async () => settle(done()))
  })

  it('cancels without compressing', () => {
    const compress = vi.fn().mockResolvedValue(done())
    const { result } = mount(compress)

    act(() => result.current.onRequest())
    act(() => result.current.onCancel())

    expect(result.current.confirming).toBe(false)
    expect(compress).not.toHaveBeenCalled()
  })

  it('does not offer the action mid-turn, and says why', () => {
    const compress = vi.fn().mockResolvedValue(done())
    const busyMount = renderHook(() => useCompactConversation({ busy: true, compress, sessionId: 'runtime-a' }))

    act(() => busyMount.result.current.onRequest())

    expect(busyMount.result.current.busy).toBe(true)
    expect(busyMount.result.current.confirming).toBe(false)
    expect(compress).not.toHaveBeenCalled()
  })

  it('surfaces a backend refusal inline and in a notice', async () => {
    const compress = vi.fn().mockRejectedValue(new Error('session busy: compression'))
    const { result } = mount(compress)

    act(() => result.current.onRequest())
    await act(async () => result.current.onConfirm())

    expect(result.current.error).toBe('session busy: compression')
    expect(result.current.pending).toBe(false)
    expect($notifications.get().some(entry => entry.kind === 'error')).toBe(true)
  })

  it('reports a skipped compression as information, never as success', async () => {
    const compress = vi.fn().mockResolvedValue({
      kind: 'skipped',
      message: 'another process holds the compression lock for this session',
      result: {},
      sessionId: 'runtime-a'
    } satisfies CompressSessionOutcome)

    const { result } = mount(compress)

    act(() => result.current.onRequest())
    await act(async () => result.current.onConfirm())

    const notices = $notifications.get()

    expect(notices.some(entry => entry.kind === 'success')).toBe(false)
    expect(notices.map(entry => entry.message).join('\n')).toContain(
      'another process holds the compression lock for this session'
    )
    expect(result.current.error).toBeNull()
    expect(result.current.pending).toBe(false)
  })

  it('reports a deferred compression as still running, not as done', async () => {
    const compress = vi.fn().mockResolvedValue({
      kind: 'deferred',
      message: 'compression still running in the background',
      sessionId: 'runtime-a'
    } satisfies CompressSessionOutcome)

    const { result } = mount(compress)

    act(() => result.current.onRequest())
    await act(async () => result.current.onConfirm())

    expect(result.current.error).toBeNull()
    expect(result.current.pending).toBe(true)
    expect(
      $notifications
        .get()
        .map(entry => entry.message)
        .join('\n')
    ).toContain('still running in the background')

    act(() => reconcileSessionCompacting('runtime-a', 'terminal'))
    expect(result.current.pending).toBe(false)
  })
})
