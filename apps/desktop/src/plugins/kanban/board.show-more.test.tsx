/**
 * The Done lane's "show more" footer must visibly take the click: it locks
 * (and spins) from the click until the bigger board has arrived, and a second
 * click while loading does nothing.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $doneLimit, DONE_PAGE } from './api'
import { useShowMoreDone } from './board'

const KEY = ['kanban', 'board', 'local', 'ops', false] as const

afterEach(() => {
  $doneLimit.set(DONE_PAGE)
  vi.restoreAllMocks()
})

function setup() {
  const client = new QueryClient()

  let settle: () => void = () => {}

  const invalidate = vi.spyOn(client, 'invalidateQueries').mockImplementation(
    () =>
      new Promise<void>(resolve => {
        settle = resolve
      })
  )

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  const hook = renderHook(() => useShowMoreDone(KEY, DONE_PAGE), { wrapper })

  return { hook, invalidate, settle: () => settle() }
}

describe('done lane show more', () => {
  it('locks from the click until the refetch settles', async () => {
    const { hook, invalidate, settle } = setup()

    expect(hook.result.current.loading).toBe(false)
    act(() => hook.result.current.onMore())
    expect(hook.result.current.loading).toBe(true)
    expect($doneLimit.get()).toBe(DONE_PAGE * 2)
    expect(invalidate).toHaveBeenCalledWith({ queryKey: KEY })

    // A second click while loading neither stacks a request nor skips a page.
    act(() => hook.result.current.onMore())
    expect(invalidate).toHaveBeenCalledTimes(1)
    expect($doneLimit.get()).toBe(DONE_PAGE * 2)

    await act(async () => settle())
    expect(hook.result.current.loading).toBe(false)
  })

  it('unlocks when the refetch fails', async () => {
    const client = new QueryClient()
    vi.spyOn(client, 'invalidateQueries').mockRejectedValue(new Error('offline'))

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )

    const { result } = renderHook(() => useShowMoreDone(KEY, DONE_PAGE), { wrapper })

    await act(async () => result.current.onMore())
    expect(result.current.loading).toBe(false)
  })
})
