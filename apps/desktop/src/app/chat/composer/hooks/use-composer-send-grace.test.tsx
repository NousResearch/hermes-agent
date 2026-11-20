import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useComposerSendGrace } from './use-composer-send-grace'

afterEach(cleanup)

// The two lifecycle guarantees the composer relies on, pinned against the real
// hook rather than a mirror of it:
//
//  1. a zero-length window commits EXACTLY ONCE. `hold()` used to return false
//     after committing on the spot, and every caller read false as "you did not
//     commit, I must" — so commitHeldEnter, the pause release and the idle send
//     submitted the same draft a second time (or drained/steered the composer
//     they had just emptied).
//
//  2. a waiting window belongs to the session that armed it. ChatBar stays
//     mounted across a session swap and only `commitRef` is repointed, so a
//     timer left running would submit the NEXT session's draft.
describe('useComposerSendGrace — a zero-length window commits exactly once', () => {
  it('reports the commit, so a caller does not submit the draft again', () => {
    const commit = vi.fn()

    const { result } = renderHook(() =>
      useComposerSendGrace({ graceMs: 0, onCommit: commit, ownerKey: 'session-a' })
    )

    let held: boolean | undefined

    act(() => {
      held = result.current.hold()
    })

    expect(commit).toHaveBeenCalledTimes(1)
    expect(held).toBe(true)
  })

  it('does not double-submit under the caller pattern that reads the return', () => {
    const commit = vi.fn()

    const { result } = renderHook(() =>
      useComposerSendGrace({ graceMs: 0, onCommit: commit, ownerKey: 'session-a' })
    )

    act(() => {
      // The composer's shape: `if (hold()) haptic; else submitDraft()`.
      if (!result.current.hold()) {
        commit()
      }
    })

    expect(commit).toHaveBeenCalledTimes(1)
  })
})

describe('useComposerSendGrace — the window belongs to the session that armed it', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('cancels a pending window when the owner changes, so it cannot submit the next session', () => {
    const first = vi.fn()
    const second = vi.fn()

    const { result, rerender } = renderHook(
      ({ ownerKey, onCommit }: { ownerKey: string | null; onCommit: () => void }) =>
        useComposerSendGrace({ graceMs: 900, onCommit, ownerKey }),
      { initialProps: { ownerKey: 'session-a', onCommit: first } }
    )

    act(() => {
      result.current.hold()
    })

    expect(result.current.holding).toBe(true)

    // The user switches sessions. ChatBar stays mounted, so without the owner
    // key the timer below would resolve `commitRef` to session B's send.
    rerender({ ownerKey: 'session-b', onCommit: second })

    expect(result.current.holding).toBe(false)

    act(() => {
      vi.advanceTimersByTime(900)
    })

    expect(first).not.toHaveBeenCalled()
    expect(second).not.toHaveBeenCalled()
  })

  it('still commits after the window when the owner does not change', () => {
    const commit = vi.fn()

    const { result } = renderHook(() =>
      useComposerSendGrace({ graceMs: 900, onCommit: commit, ownerKey: 'session-a' })
    )

    act(() => {
      result.current.hold()
    })

    expect(commit).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(900)
    })

    expect(commit).toHaveBeenCalledTimes(1)
  })
})
