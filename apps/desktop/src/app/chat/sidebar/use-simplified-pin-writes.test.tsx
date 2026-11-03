import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as sessionsApi from '@/api/sessions'
import { makeSessionInfo } from '@/test/session-info'

import { simplifiedPins } from './simplified-session-model'
import { useSimplifiedPinWrites } from './use-simplified-pin-writes'

const row = makeSessionInfo({ id: 'chat', profile: 'research', pinned: false })

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it('shows a pin immediately, ignores repeat clicks during the write, and fences stale refreshes', async () => {
  let resolve!: (value: unknown) => void

  const write = vi.spyOn(sessionsApi, 'mutateSessionHttp').mockImplementation(
    () =>
      new Promise(done => {
        resolve = done
      })
  )

  const refresh = vi.fn()
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', refresh))
  let request!: Promise<void>
  act(() => {
    request = result.current.setPinned(row, true)
  })
  expect(simplifiedPins([row], [], new Set(), result.current.value)).toEqual([row])
  expect(result.current.pending(row)).toBe(true)
  await act(async () => {
    await result.current.setPinned(row, true)
  })
  expect(write).toHaveBeenCalledTimes(1)
  expect(write).toHaveBeenCalledWith(
    'chat',
    'PATCH',
    { pinned: true },
    { connectionId: 'gateway-a', profile: 'research' }
  )
  await act(async () => {
    resolve({ ok: true })
    await request
  })
  expect(refresh).toHaveBeenCalledOnce()
  expect(result.current.pending(row)).toBe(false)
  expect(simplifiedPins([{ ...row }], [], new Set(), result.current.value)).toHaveLength(1)
})

it('keeps a rapid unpin from being undone by a late pin page and expires the write guard', async () => {
  vi.useFakeTimers()
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  await act(async () => {
    await result.current.setPinned(row, true)
  })
  await act(async () => {
    await result.current.setPinned(row, false)
  })
  expect(simplifiedPins([{ ...row, pinned: true }], ['chat'], new Set(['chat']), result.current.value)).toEqual([])
  act(() => vi.advanceTimersByTime(10_000))
  expect(result.current.value(row)).toBeUndefined()
})

it('rolls back a rejected write and permits retry', async () => {
  const failure = new Error('write failed')

  const write = vi
    .spyOn(sessionsApi, 'mutateSessionHttp')
    .mockRejectedValueOnce(failure)
    .mockResolvedValue({ ok: true })

  const refresh = vi.fn()
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', refresh))
  await act(async () => {
    await expect(result.current.setPinned(row, true)).rejects.toThrow('write failed')
  })
  expect(result.current.value(row)).toBeUndefined()
  expect(result.current.pending(row)).toBe(false)
  expect(refresh).not.toHaveBeenCalled()
  await act(async () => {
    await result.current.setPinned(row, true)
  })
  expect(write).toHaveBeenCalledTimes(2)
  expect(result.current.value(row)).toBe(true)
})

it('keeps owner identity across gateway changes and does not roll back an acknowledged write on refresh failure', async () => {
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const refresh = vi.fn().mockRejectedValue(new Error('refresh failed'))

  const { result, rerender } = renderHook(({ owner }) => useSimplifiedPinWrites(owner, refresh), {
    initialProps: { owner: 'gateway-a' }
  })

  await act(async () => {
    await expect(result.current.setPinned(row, true)).rejects.toThrow('refresh failed')
  })
  expect(result.current.value(row)).toBe(true)
  expect(result.current.value({ ...row, profile: 'other' })).toBeUndefined()
  rerender({ owner: 'gateway-b' })
  expect(result.current.value(row)).toBeUndefined()
  expect(result.current.value({ ...row, connection_id: 'gateway-a' })).toBe(true)
})

it('restores the last acknowledged pin if a quick unpin fails before the list catches up', async () => {
  const write = vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  await act(async () => {
    await result.current.setPinned(row, true)
  })
  write.mockRejectedValueOnce(new Error('unpin failed'))
  await act(async () => {
    await expect(result.current.setPinned(row, false)).rejects.toThrow('unpin failed')
  })
  expect(result.current.value(row)).toBe(true)
  expect(simplifiedPins([row], [], new Set(), result.current.value)).toHaveLength(1)
})
