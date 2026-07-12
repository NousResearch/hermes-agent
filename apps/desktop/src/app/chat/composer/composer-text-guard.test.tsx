// @vitest-environment jsdom
import { act, cleanup, render } from '@testing-library/react'
import { useCallback, useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

afterEach(cleanup)
beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

// Regression repro for #49903: on desktop v0.17.0 the composer threw an
// uncaught `Error: Composer is not available` at startup and the input went
// unresponsive. The throw comes from @assistant-ui/core's composer-runtime —
// every *mutator* (setText/send/…) does `if (!core) throw new Error("Composer
// is not available")` when the thread's composer core isn't bound yet. Unlike
// the read path (`s.composer.text`, which is null-safe: `runtime?.text ?? ""`),
// the mutators have no graceful fallback. ChatBar's mount-time effects (draft
// restore, clearDraft, external inserts) push text via `aui.composer().setText`
// before the core binds, and the popout refactor (#49488) widened that window,
// so the throw surfaced as an uncaught error that wedged the input.
//
// The fix keeps the latest desired draft on file and retries `setText` on
// animation frames until the composer core binds. That preserves the no-crash
// startup behavior while also preventing a stale runtime draft from leaking
// into the next session after a scope swap.

interface FakeComposer {
  setText: (value: string) => void
}

// Mirror of index.tsx's `useAui()` composer surface: composer() returns a
// runtime whose setText throws exactly like @assistant-ui/core when unbound.
function makeFakeAui(bound: { current: boolean }, applied: string[]) {
  const composer: FakeComposer = {
    setText(value: string) {
      if (!bound.current) {
        throw new Error('Composer is not available')
      }

      applied.push(value)
    }
  }

  return { composer: () => composer }
}

function Harness({
  writes,
  bound,
  applied,
  onError
}: {
  applied: string[]
  bound: { current: boolean }
  onError: (err: unknown) => void
  writes: string[]
}) {
  const aui = useRef(makeFakeAui(bound, applied)).current
  const pendingComposerTextRef = useRef<string | null>(null)
  const pendingComposerTextRafRef = useRef<number | undefined>(undefined)

  const flushPendingComposerText = useCallback(() => {
    const pending = pendingComposerTextRef.current

    if (pending === null) {
      return true
    }

    try {
      aui.composer().setText(pending)
      pendingComposerTextRef.current = null

      return true
    } catch {
      return false
    }
  }, [aui])

  const schedulePendingComposerTextFlush = useCallback(() => {
    if (pendingComposerTextRafRef.current !== undefined) {
      return
    }

    const retry = () => {
      pendingComposerTextRafRef.current = undefined

      if (pendingComposerTextRef.current === null) {
        return
      }

      if (!flushPendingComposerText()) {
        pendingComposerTextRafRef.current = window.requestAnimationFrame(retry)
      }
    }

    pendingComposerTextRafRef.current = window.requestAnimationFrame(retry)
  }, [flushPendingComposerText])

  const setComposerText = useCallback(
    (value: string) => {
      pendingComposerTextRef.current = value

      if (!flushPendingComposerText()) {
        schedulePendingComposerTextFlush()
      }
    },
    [flushPendingComposerText, schedulePendingComposerTextFlush]
  )

  useEffect(() => {
    for (const value of writes) {
      try {
        setComposerText(value)
      } catch (err) {
        onError(err)
      }
    }
  }, [onError, setComposerText, writes])

  useEffect(
    () => () => {
      if (pendingComposerTextRafRef.current !== undefined) {
        window.cancelAnimationFrame(pendingComposerTextRafRef.current)
      }
    },
    []
  )

  return null
}

describe('setComposerText guard (#49903)', () => {
  it('swallows the unbound-core throw at startup instead of crashing the renderer', () => {
    const applied: string[] = []
    const bound = { current: false }
    const onError = vi.fn()

    expect(() => render(<Harness applied={applied} bound={bound} onError={onError} writes={['restored draft']} />)).not.toThrow()

    // The guard absorbed the throw — nothing escaped to the renderer, and no
    // assistant-ui write landed (core was unbound).
    expect(onError).not.toHaveBeenCalled()
    expect(applied).toEqual([])
  })

  it('writes through to the composer once the core is bound', () => {
    const applied: string[] = []
    const bound = { current: true }
    const onError = vi.fn()

    act(() => {
      render(<Harness applied={applied} bound={bound} onError={onError} writes={['restored draft']} />)
    })

    expect(onError).not.toHaveBeenCalled()
    expect(applied).toEqual(['restored draft'])
  })

  it('replays the latest pending draft once the core binds', () => {
    const applied: string[] = []
    const bound = { current: false }

    render(<Harness applied={applied} bound={bound} onError={vi.fn()} writes={['restored draft']} />)

    act(() => {
      vi.advanceTimersByTime(16)
    })
    expect(applied).toEqual([])

    bound.current = true
    act(() => {
      vi.advanceTimersByTime(16)
    })

    expect(applied).toEqual(['restored draft'])
  })

  it('flushes only the newest pending value across an unbound session swap', () => {
    const applied: string[] = []
    const bound = { current: false }

    render(<Harness applied={applied} bound={bound} onError={vi.fn()} writes={['previous session draft', '']} />)

    act(() => {
      vi.advanceTimersByTime(16)
    })
    expect(applied).toEqual([])

    bound.current = true
    act(() => {
      vi.advanceTimersByTime(16)
    })

    expect(applied).toEqual([''])
  })
})
