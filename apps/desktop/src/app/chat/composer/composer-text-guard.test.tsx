// @vitest-environment jsdom
import { act, cleanup, render } from '@testing-library/react'
import { useCallback, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(cleanup)

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
// The fix wraps every `aui.composer().setText` call in a `setComposerText`
// helper that swallows the unbound-core throw — the contentEditable DOM +
// draftRef already hold the text and the draft⇄editor sync re-applies it once
// the core attaches, so nothing is lost. This Harness mirrors that helper
// faithfully (same try/catch shape) over a fake `aui` whose composer can be
// toggled bound/unbound, the way the assistant-ui runtime behaves across mount.

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
  bound,
  applied,
  onError
}: {
  applied: string[]
  bound: { current: boolean }
  onError: (err: unknown) => void
}) {
  const aui = useRef(makeFakeAui(bound, applied)).current

  // Verbatim mirror of the production `setComposerText` helper in index.tsx.
  const setComposerText = useCallback(
    (value: string) => {
      try {
        aui.composer().setText(value)
      } catch {
        // Composer core not bound yet — swallow so the input stays usable.
      }
    },
    [aui]
  )

  // A draft-restore-on-mount that fires while the core may still be unbound,
  // exactly like loadIntoComposer/clearDraft do on startup.
  try {
    setComposerText('restored draft')
  } catch (err) {
    onError(err)
  }

  return null
}

describe('setComposerText guard (#49903)', () => {
  it('swallows the unbound-core throw at startup instead of crashing the renderer', () => {
    const applied: string[] = []
    const bound = { current: false }
    const onError = vi.fn()

    expect(() => render(<Harness applied={applied} bound={bound} onError={onError} />)).not.toThrow()

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
      render(<Harness applied={applied} bound={bound} onError={onError} />)
    })

    expect(onError).not.toHaveBeenCalled()
    expect(applied).toEqual(['restored draft'])
  })
})
