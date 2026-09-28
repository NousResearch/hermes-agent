import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { api, view } = vi.hoisted(() => ({
  api: vi.fn(),
  view: {
    $storedId: { get: () => 'stored-a', listen: () => () => {} },
    $runtimeId: { get: () => 'runtime-a', listen: () => () => {} },
    $messages: undefined as unknown
  }
}))

vi.mock('@/api/client', () => ({ capabilityScoped: (scope: object) => scope, hermesApi: api }))
vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => view }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/session', () => ({ $connection: atom({ mode: 'local' }), getSessionOwnerHint: () => undefined }))
vi.mock('@/store/transcript-tail', () => ({ transcriptTailState: () => undefined }))

const page = (ids: number[], more = false) => ({
  entries: ids.map(id => ({ row_id: id, preview: `Prompt ${id}` })),
  pagination: { next_cursor: more ? ids.at(-1) : null, has_more: more }
})

beforeEach(() => {
  vi.resetModules()
  vi.useFakeTimers()
  api.mockReset()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('timeline history index', () => {
  it('follows prompts persisted after the complete index was read', async () => {
    const $messages = atom<Array<{ id: string; role: string; rowId?: number }>>([
      { id: 'u1', role: 'user', rowId: 1 },
      { id: 'u2', role: 'user', rowId: 2 }
    ])

    view.$messages = $messages
    api.mockResolvedValueOnce(page([1, 2])).mockResolvedValueOnce(page([1, 2, 3]))
    const { useTimelineHistory } = await import('./use-timeline-history')
    const { result } = renderHook(() => useTimelineHistory())

    await act(async () => {
      vi.advanceTimersByTime(200)
      await Promise.resolve()
    })
    expect(result.current.complete).toBe(true)
    expect(result.current.entries?.map(entry => entry.rowId)).toEqual([1, 2])
    expect(api).toHaveBeenCalledTimes(1)

    // A new turn lands in the live tail: the rail's index pages forward once,
    // from its own cursor, without a timer.
    await act(async () => {
      $messages.set([...$messages.get(), { id: 'a2', role: 'assistant' }, { id: 'u3', role: 'user', rowId: 3 }])
      await Promise.resolve()
    })
    expect(api).toHaveBeenCalledTimes(2)
    expect(result.current.entries?.map(entry => entry.rowId)).toEqual([1, 2, 3])

    // Streaming into the newest turn moves nothing: no further request.
    await act(async () => {
      $messages.set([...$messages.get(), { id: 'a3', role: 'assistant' }])
      await Promise.resolve()
    })
    expect(api).toHaveBeenCalledTimes(2)
  })
})
