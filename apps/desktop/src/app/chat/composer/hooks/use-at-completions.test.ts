import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { useAtCompletions } from './use-at-completions'

function gatewayStub(latencyMs = 40) {
  const calls: string[] = []

  const gateway = {
    request: vi.fn(async (_method: string, params: { word: string }) => {
      calls.push(params.word)
      await new Promise(r => setTimeout(r, latencyMs))

      return { items: [{ text: `@folder:${params.word.slice(1)}x/`, display: 'x/', meta: 'dir' }] }
    })
  }

  return { calls, gateway }
}

function setup(latencyMs = 40) {
  const { calls, gateway } = gatewayStub(latencyMs)

  const { result } = renderHook(() => useAtCompletions({ gateway: gateway as never, sessionId: 's1', cwd: '/repo' }))

  return { calls, result }
}

/** Type a burst of keystrokes `gapMs` apart, like a person. */
async function type(
  result: { current: { adapter: { search?: (q: string) => unknown } } },
  queries: string[],
  gapMs: number
) {
  for (const q of queries) {
    act(() => {
      result.current.adapter.search?.(q)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(gapMs)
    })
  }
}

describe('PERF: @ path completions are cached and skip the debounce', () => {
  it('serves a repeated query with no round trip and no spinner', async () => {
    vi.useFakeTimers()
    queryClient.clear()

    const { calls, result } = setup()

    // First visit to `apps/` pays the round trip.
    await type(result, ['apps/'], 0)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    const afterFirst = calls.length

    expect(afterFirst).toBe(1)

    // Walk away and come back — Tab in, Backspace out, retype. Every one of
    // these used to be a fresh git ls-files + rank on the backend.
    await type(result, ['apps/desktop/', 'apps/', 'apps/desktop/', 'apps/'], 0)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    // Two distinct paths, so exactly two round trips total — the repeats are free.
    expect(calls.length).toBe(2)
    expect(result.current.loading).toBe(false)

    vi.useRealTimers()
  })

  it('a cached query paints without waiting out the debounce', async () => {
    vi.useFakeTimers()
    queryClient.clear()

    const { calls, result } = setup()

    await type(result, ['apps/'], 0)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })

    expect(calls.length).toBe(1)

    // Re-ask for the cached query and advance by far less than the 60ms
    // debounce. A cached answer resolves in a microtask, so it must paint
    // without the timer and without ever flipping the spinner on.
    act(() => {
      result.current.adapter.search?.('apps/')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })

    expect(result.current.loading).toBe(false)
    expect(calls.length).toBe(1)

    vi.useRealTimers()
  })
})
