import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerPopoutGestures } from './use-popout-drag'

function GestureHarness({ onPopOut }: { onPopOut: () => void }) {
  const composerRef = useRef<HTMLFormElement>(null)

  const { onPointerDown } = useComposerPopoutGestures({
    composerRef,
    onDock: vi.fn(),
    onPopOut,
    poppedOut: false,
    position: { bottom: 24, right: 24 }
  })

  return (
    <form data-testid="composer" onPointerDown={onPointerDown} ref={composerRef}>
      <div data-slot="composer-drag-region" data-testid="drag-region" />
      <div data-slot="composer-surface" data-testid="surface">
        <div contentEditable data-slot="composer-rich-input">
          <span data-testid="editable-text">select me</span>
        </div>
      </div>
    </form>
  )
}

function dragUp(target: Element) {
  fireEvent.pointerDown(target, { button: 0, clientX: 100, clientY: 100, pointerId: 7 })
  fireEvent.pointerMove(window, { clientX: 100, clientY: 60, pointerId: 7 })
  fireEvent.pointerUp(window, { clientX: 100, clientY: 60, pointerId: 7 })
}

afterEach(cleanup)

describe('useComposerPopoutGestures', () => {
  it('never peels the composer out when a text-selection drag starts in the rich editor', () => {
    const onPopOut = vi.fn()
    render(<GestureHarness onPopOut={onPopOut} />)

    dragUp(screen.getByTestId('editable-text'))

    expect(onPopOut).not.toHaveBeenCalled()
  })

  it('does not arm a dock peel from the composer surface', () => {
    const onPopOut = vi.fn()
    render(<GestureHarness onPopOut={onPopOut} />)

    dragUp(screen.getByTestId('surface'))

    expect(onPopOut).not.toHaveBeenCalled()
  })

  it('still peels out from the dedicated drag region', () => {
    const onPopOut = vi.fn()
    render(<GestureHarness onPopOut={onPopOut} />)

    dragUp(screen.getByTestId('drag-region'))

    expect(onPopOut).toHaveBeenCalledOnce()
  })
})
