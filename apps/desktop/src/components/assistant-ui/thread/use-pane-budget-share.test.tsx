import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $mountedTranscriptPanes, usePaneBudgetShare } from './list'

afterEach(() => {
  cleanup()
  $mountedTranscriptPanes.set(0)
})

describe('usePaneBudgetShare', () => {
  it('tracks this panes share of the render budget across mounts', () => {
    $mountedTranscriptPanes.set(1)

    const first = renderHook(({ hidden }: { hidden: boolean }) => usePaneBudgetShare(hidden), {
      initialProps: { hidden: false }
    })

    expect(first.result.current).toBe(600)

    const second = renderHook(({ hidden }: { hidden: boolean }) => usePaneBudgetShare(hidden), {
      initialProps: { hidden: false }
    })

    // A second pane mounts: both shares halve, with no other input.
    act(() => {
      $mountedTranscriptPanes.set(2)
    })
    expect(first.result.current).toBe(300)
    expect(second.result.current).toBe(300)

    // It unmounts: the survivor recovers the full budget.
    second.unmount()
    act(() => {
      $mountedTranscriptPanes.set(1)
    })
    expect(first.result.current).toBe(600)
  })

  it('never re-renders a hidden pane for mount churn', () => {
    $mountedTranscriptPanes.set(1)
    let renders = 0

    const hidden = renderHook(() => {
      renders += 1

      return usePaneBudgetShare(true)
    })

    expect(hidden.result.current).toBe(40)
    const settled = renders

    act(() => {
      $mountedTranscriptPanes.set(2)
    })
    act(() => {
      $mountedTranscriptPanes.set(5)
    })

    expect(hidden.result.current).toBe(40)
    expect(renders).toBe(settled)
  })
})
