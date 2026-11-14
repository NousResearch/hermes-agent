/**
 * The tab-strip "+" registration is mount-stable (#154651): no
 * teardown/re-register churn across action identity changes. The old
 * identity-keyed effect wrote null on every churn — the strip glyph reads the
 * atom and cycled hidden→visible with it, and correctness leaned on the
 * teardown and re-register staying one synchronous step. Null writes are
 * confined to unmount (where the sdk's host.newChat gate is meaningful) and
 * StrictMode's dev-only mount replay.
 */

import { act, cleanup, renderHook } from '@testing-library/react'
import { StrictMode } from 'react'
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
    // null, and host.newChat's bots door refuses while the atom is null.
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

  it('ends registered after StrictMode replays the mount, dispatching the latest action', () => {
    const second = vi.fn()

    const { rerender } = renderHook(({ action }) => useNewSessionTabActionRegistration(action), {
      initialProps: { action: vi.fn() },
      wrapper: ({ children }) => <StrictMode>{children}</StrictMode>
    })

    // The dev replay writes null once between the replayed unmount and mount;
    // the end state must be a live registration, not the replayed cleanup.
    rerender({ action: second })

    act(() => {
      $newSessionTabAction.get()?.()
    })

    expect(second).toHaveBeenCalledTimes(1)
  })
})
