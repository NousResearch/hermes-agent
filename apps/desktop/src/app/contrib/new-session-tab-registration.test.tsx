/**
 * The tab-strip "+" registration is churn-proof (#154651): the atom never
 * tears down while the component is mounted, no matter how often the action's
 * identity changes — host.newChat's bots door refuses while the atom is null,
 * so a re-registration window told an up-to-date desktop to update itself.
 */

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $newSessionTabAction } from '@/components/pane-shell/tree/store'

import { useNewSessionTabActionRegistration } from './new-session-tab-registration'

afterEach(() => {
  cleanup()
  $newSessionTabAction.set(null)
})

describe('useNewSessionTabActionRegistration', () => {
  it('never writes null across action identity churn', () => {
    const first = vi.fn()
    const second = vi.fn()
    let stillMounted = true
    const nullWrites: number[] = []
    const unlisten = $newSessionTabAction.listen(value => {
      if (value === null && stillMounted) {
        nullWrites.push(nullWrites.length)
      }
    })

    const { rerender, unmount } = renderHook(({ action }) => useNewSessionTabActionRegistration(action), {
      initialProps: { action: first }
    })

    rerender({ action: second })
    rerender({ action: vi.fn() })
    rerender({ action: second })

    stillMounted = false
    unmount()
    unlisten()

    // While mounted the atom is never torn down — only the unmount writes
    // null. host.newChat's bots door refuses while it is null.
    expect(nullWrites).toEqual([])
  })

  it('always dispatches the latest action', () => {
    const first = vi.fn()
    const second = vi.fn()

    const { rerender } = renderHook(({ action }) => useNewSessionTabActionRegistration(action), {
      initialProps: { action: first }
    })

    act(() => {
      $newSessionTabAction.get()?.()
    })

    expect(first).toHaveBeenCalledTimes(1)
    expect(second).not.toHaveBeenCalled()

    rerender({ action: second })

    act(() => {
      $newSessionTabAction.get()?.()
    })

    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(1)
  })

  it('clears the atom on unmount', () => {
    const { unmount } = renderHook(() => useNewSessionTabActionRegistration(vi.fn()))

    expect($newSessionTabAction.get()).not.toBeNull()

    unmount()

    expect($newSessionTabAction.get()).toBeNull()
  })
})
