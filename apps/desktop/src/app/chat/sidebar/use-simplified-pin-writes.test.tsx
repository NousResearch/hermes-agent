import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as sessionsApi from '@/api/sessions'
import * as iaMetrics from '@/store/ia-metrics'
import { $sessionPinWriteIntent } from '@/store/session-pin-sync'
import { $removedSessionIds, tombstoneSessions, untombstoneSessions } from '@/store/session-removal'
import { makeSessionInfo } from '@/test/session-info'

import { simplifiedPins } from './simplified-session-model'
import { $simplifiedPinnedRows, useSimplifiedPinWrites } from './use-simplified-pin-writes'

const row = makeSessionInfo({ id: 'chat', connection_id: 'gateway-a', profile: 'research', pinned: false })

afterEach(() => {
  cleanup()
  $simplifiedPinnedRows.set(new Map())
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it('shows a pin immediately, ignores repeat clicks during the write, and fences stale refreshes', async () => {
  const record = vi.fn()
  vi.spyOn(iaMetrics, 'captureIaRecorder').mockReturnValue(record)
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
  expect(record).not.toHaveBeenCalled()
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
  expect(record.mock.calls).toEqual([['ia.pin.pinned']])
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
  const record = vi.fn()
  vi.spyOn(iaMetrics, 'captureIaRecorder').mockReturnValue(record)
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const refresh = vi.fn().mockRejectedValue(new Error('refresh failed'))

  const { result, rerender } = renderHook(({ owner }) => useSimplifiedPinWrites(owner, refresh), {
    initialProps: { owner: 'gateway-a' }
  })

  await act(async () => {
    await expect(result.current.setPinned(row, true)).rejects.toThrow('refresh failed')
  })
  expect(record.mock.calls).toEqual([['ia.pin.pinned']])
  expect(result.current.value(row)).toBe(true)
  expect(result.current.value({ ...row, profile: 'other' })).toBeUndefined()
  rerender({ owner: 'gateway-b' })
  expect(result.current.value(row)).toBe(true)
  expect(result.current.value({ ...row, connection_id: 'gateway-b' })).toBeUndefined()
})

it('does not assign the foreground registry owner to a legacy untagged row', async () => {
  const write = vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const legacy = { ...row, connection_id: undefined }
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-b', vi.fn()))
  await act(async () => { await result.current.setPinned(legacy, true) })
  expect(write).toHaveBeenCalledWith('chat', 'PATCH', { pinned: true }, {
    connectionId: undefined, profile: 'research'
  })
  expect(result.current.value({ ...legacy, connection_id: 'gateway-b' })).toBeUndefined()
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

it('keeps a pin attached to the same conversation after compression replaces its session id', async () => {
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  await act(async () => {
    await result.current.setPinned(row, true)
  })
  const compressed = { ...row, id: 'new-tip', _lineage_root_id: row.id }
  expect(result.current.value(compressed)).toBe(true)
  expect(result.current.value({ ...compressed, profile: 'other' })).toBeUndefined()
  expect(result.current.rows[0].connection_id).toBe('gateway-a')
})

it('retains a confirmed remote search pin after the guard expires and the sidebar remounts', async () => {
  vi.useFakeTimers()
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const first = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  const remote = { ...row, connection_id: 'gateway-b' }
  await act(async () => {
    await first.result.current.setPinned(remote, true)
  })
  act(() => vi.advanceTimersByTime(10_001))
  expect(first.result.current.value(remote)).toBeUndefined()
  expect(first.result.current.rows).toContainEqual(
    expect.objectContaining({ connection_id: 'gateway-b', pinned: true })
  )
  first.unmount()
  const second = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  expect(simplifiedPins(second.result.current.rows, [], new Set())).toContainEqual(
    expect.objectContaining({ id: row.id, pinned: true })
  )
})

it('lets a newer header intent supersede a sidebar guard only for the same owner', async () => {
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  await act(async () => {
    await result.current.setPinned(row, true)
  })
  act(() => $sessionPinWriteIntent.set({ id: row.id, profile: 'research', connectionId: 'gateway-b', value: false }))
  expect(result.current.value(row)).toBe(true)
  act(() => $sessionPinWriteIntent.set({ id: row.id, profile: 'research', connectionId: 'gateway-a', value: false }))
  expect(result.current.value(row)).toBeUndefined()
  expect(result.current.rows).toEqual([])
})

it('keeps retained pins out of removed rows while allowing a failed removal to roll back', async () => {
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockResolvedValue({ ok: true })
  const remote = { ...row, id: 'removed-remote-pin', connection_id: 'gateway-b' }
  const { result } = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  await act(async () => {
    await result.current.setPinned(remote, true)
  })
  act(() => tombstoneSessions([remote.id]))
  expect(result.current.rows).toEqual([])
  act(() => untombstoneSessions([remote.id]))
  expect(result.current.rows.length).toBeGreaterThan(0)
  act(() => tombstoneSessions([remote.id]))
  // The project tree prunes its membership after the server has confirmed removal.
  act(() => $removedSessionIds.set(new Set()))
  expect(result.current.rows).toEqual([])
})

it('does not resurrect an old pin when a newer shared write occurs after sidebar unmount', async () => {
  let resolve!: (value: unknown) => void
  vi.spyOn(sessionsApi, 'mutateSessionHttp').mockImplementation(
    () =>
      new Promise(done => {
        resolve = done
      })
  )
  const mounted = renderHook(() => useSimplifiedPinWrites('gateway-a', vi.fn()))
  let request!: Promise<void>
  act(() => {
    request = mounted.result.current.setPinned(row, true)
  })
  mounted.unmount()
  $sessionPinWriteIntent.set({ id: row.id, connectionId: 'gateway-a', profile: 'research', value: false })
  await act(async () => {
    resolve({ ok: true })
    await request
  })
  expect($simplifiedPinnedRows.get().size).toBe(0)
})
