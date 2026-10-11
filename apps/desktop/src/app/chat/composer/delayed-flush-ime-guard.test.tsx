import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(cleanup)

// Faithful mirror of index.tsx's coalesced editor→draft flush wiring: the
// composingRef flag, the composition-skip in handleEditorInput, and
// scheduleFlushEditorToDraft's rAF callback (including the mid-composition
// guard under test). `onFlush` stands in for flushEditorToDraft's body — the
// DOM-normalizing pass plus the state write — so calling it inside an active
// composition is exactly the regression we assert against.
function Harness({ onFlush }: { onFlush: () => void }) {
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const flushRafRef = useRef<number | undefined>(undefined)

  const flushEditorToDraft = () => {
    if (flushRafRef.current !== undefined) {
      window.cancelAnimationFrame(flushRafRef.current)
      flushRafRef.current = undefined
    }

    onFlush()
  }

  const scheduleFlushEditorToDraft = () => {
    if (flushRafRef.current !== undefined) {
      return
    }

    flushRafRef.current = window.requestAnimationFrame(() => {
      flushRafRef.current = undefined

      if (composingRef.current) {
        return
      }

      flushEditorToDraft()
    })
  }

  return (
    <div
      contentEditable
      data-testid="editor"
      onCompositionEnd={() => {
        composingRef.current = false
        flushEditorToDraft()
      }}
      onCompositionStart={() => {
        composingRef.current = true
      }}
      onInput={() => {
        if (composingRef.current) {
          return
        }

        scheduleFlushEditorToDraft()
      }}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

describe('composer coalesced flush — delayed rAF vs IME composition (#136460)', () => {
  it('drops a delayed rAF flush that lands inside a fresh composition instead of aborting it', async () => {
    const rafCallbacks: FrameRequestCallback[] = []
    const cancelAnimationFrame = vi.spyOn(window, 'cancelAnimationFrame')

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      rafCallbacks.push(callback)

      return rafCallbacks.length
    })

    try {
      const onFlush = vi.fn()
      const { getByTestId } = render(<Harness onFlush={onFlush} />)
      const editor = getByTestId('editor')

      // The previous IME composition commits: compositionend flushes
      // synchronously (flush #1), then Chromium's trailing input event queues
      // the coalesced flush on a rAF that a busy renderer postpones.
      await act(async () => {
        fireEvent.compositionStart(editor)
        fireEvent.compositionEnd(editor)
        expect(onFlush).toHaveBeenCalledTimes(1)

        fireEvent.input(editor)
      })

      expect(rafCallbacks).toHaveLength(1)

      // The user starts the next syllable before the delayed frame runs.
      await act(async () => {
        fireEvent.compositionStart(editor)
        rafCallbacks[0](0)
      })

      // The delayed flush must not run mid-preedit: normalizing the DOM there
      // aborts the composition, which leaks the preedit into the draft as
      // literal Latin letters.
      expect(onFlush).toHaveBeenCalledTimes(1)
      expect(cancelAnimationFrame).not.toHaveBeenCalled()

      // The dropped run re-arms: the next input after compositionend can
      // queue a fresh rAF.
      await act(async () => {
        fireEvent.compositionEnd(editor)
        expect(onFlush).toHaveBeenCalledTimes(2)

        fireEvent.input(editor)
      })

      expect(rafCallbacks).toHaveLength(2)
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('flushes on the delayed frame when no composition is active', async () => {
    const rafCallbacks: FrameRequestCallback[] = []

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      rafCallbacks.push(callback)

      return rafCallbacks.length
    })

    try {
      const onFlush = vi.fn()
      const { getByTestId } = render(<Harness onFlush={onFlush} />)
      const editor = getByTestId('editor')

      await act(async () => {
        fireEvent.input(editor)
        expect(onFlush).not.toHaveBeenCalled()

        rafCallbacks[0](0)
      })

      expect(onFlush).toHaveBeenCalledTimes(1)
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('never queues a flush from input events fired mid-composition', async () => {
    const rafCallbacks: FrameRequestCallback[] = []

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      rafCallbacks.push(callback)

      return rafCallbacks.length
    })

    try {
      const onFlush = vi.fn()
      const { getByTestId } = render(<Harness onFlush={onFlush} />)
      const editor = getByTestId('editor')

      await act(async () => {
        fireEvent.compositionStart(editor)
        fireEvent.input(editor)
        fireEvent.input(editor)
      })

      expect(rafCallbacks).toHaveLength(0)

      // compositionend still flushes the committed text synchronously.
      await act(async () => {
        fireEvent.compositionEnd(editor)
      })

      expect(onFlush).toHaveBeenCalledTimes(1)
    } finally {
      vi.restoreAllMocks()
    }
  })
})
