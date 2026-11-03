import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { CompressSessionOutcome } from '@/app/session/compress-session'
import { $compactingSessions, reconcileSessionCompacting, takeCompressDeferred } from '@/store/compaction'
import { clearNotifications } from '@/store/notifications'

import { _resetManualCompactingForTests, useCompactConversation } from './use-compact-conversation'

beforeEach(() => {
  _resetManualCompactingForTests()
  clearNotifications()
  $compactingSessions.set({})
  takeCompressDeferred('runtime-a')
})

afterEach(() => {
  cleanup()
  _resetManualCompactingForTests()
})

it('refuses confirmation if a reply started after the question opened', async () => {
  const compress = vi.fn().mockResolvedValue({
    aborted: false,
    kind: 'done',
    message: 'compacted',
    result: {},
    sessionId: 'runtime-a'
  } satisfies CompressSessionOutcome)

  const { result, rerender } = renderHook(
    ({ busy }) => useCompactConversation({ busy, compress, sessionId: 'runtime-a', storedSessionId: 'stored-a' }),
    { initialProps: { busy: false } }
  )

  act(() => result.current.onRequest())
  expect(result.current.confirming).toBe(true)
  rerender({ busy: true })
  await act(async () => result.current.onConfirm())

  expect(compress).not.toHaveBeenCalled()
  expect(result.current.pending).toBe(false)
})

it('keeps deferred compaction guarded until the existing completion edge clears it', async () => {
  const compress = vi.fn().mockResolvedValue({
    kind: 'deferred',
    message: 'still compacting',
    sessionId: 'runtime-a'
  } satisfies CompressSessionOutcome)

  const { result } = renderHook(() =>
    useCompactConversation({ busy: false, compress, sessionId: 'runtime-a', storedSessionId: 'stored-a' })
  )

  act(() => result.current.onRequest())
  await act(async () => result.current.onConfirm())

  expect(result.current.pending).toBe(true)
  act(() => result.current.onRequest())
  await act(async () => result.current.onConfirm())
  expect(compress).toHaveBeenCalledTimes(1)
  expect(takeCompressDeferred('runtime-a')).toBe(true)

  act(() => reconcileSessionCompacting('runtime-a', 'terminal'))
  expect(result.current.pending).toBe(false)
})
