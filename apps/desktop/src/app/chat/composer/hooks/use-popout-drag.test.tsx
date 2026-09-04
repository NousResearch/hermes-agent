import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ComposerDragRegion } from '../composer-drag-region'

import { useComposerPopoutGestures } from './use-popout-drag'

function GestureHarness({ onPopOut }: { onPopOut: () => void }) {
  const composerRef = useRef<HTMLFormElement>(null)

  const { onPointerDown } = useComposerPopoutGestures({
    composerRef,
    groupId: 'test-zone',
    onDock: vi.fn(),
    onPopOut,
    poppedOut: false,
    position: { bottom: 24, right: 24 }
  })

  return (
    <form data-slot="composer-root" data-testid="composer" onPointerDown={onPointerDown} ref={composerRef}>
      <ComposerDragRegion dragging={false} />
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

    dragUp(window.document.querySelector('[data-drag-edge]')!)

    expect(onPopOut).toHaveBeenCalledOnce()
  })

  it('peels from the exposed composer-root margin when the visual frame is pointer-transparent', () => {
    const onPopOut = vi.fn()
    render(<GestureHarness onPopOut={onPopOut} />)

    dragUp(screen.getByTestId('composer'))

    expect(onPopOut).toHaveBeenCalledOnce()
  })

  it('keeps the drag affordance pointer-active only on the five-pixel ring', () => {
    render(<GestureHarness onPopOut={vi.fn()} />)

    const region = window.document.querySelector('[data-slot="composer-drag-region"]') as HTMLElement
    const edges = region.querySelectorAll('[data-drag-edge]')

    expect(region.classList.contains('pointer-events-none')).toBe(true)
    expect(edges).toHaveLength(4)

    for (const edge of edges) {
      expect(edge.classList.contains('pointer-events-auto')).toBe(true)
      expect(edge.classList.contains('cursor-grab')).toBe(true)
    }
  })

  it('switches the exposed ring to a grabbing cursor without activating its frame', () => {
    const { rerender } = render(<ComposerDragRegion dragging={false} />)

    rerender(<ComposerDragRegion dragging />)

    const region = window.document.querySelector('[data-slot="composer-drag-region"]') as HTMLElement

    expect(region.hasAttribute('data-dragging')).toBe(true)
    expect(region.classList.contains('pointer-events-none')).toBe(true)

    for (const edge of region.querySelectorAll('[data-drag-edge]')) {
      expect(edge.classList.contains('cursor-grabbing')).toBe(true)
    }
  })
})
