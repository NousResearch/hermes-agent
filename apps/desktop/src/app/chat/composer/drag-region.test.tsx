// @vitest-environment jsdom
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ComposerDragRegion } from './drag-region'

afterEach(cleanup)

describe('ComposerDragRegion', () => {
  it('lets a raised edge reach the root gesture and double-click handlers', () => {
    const onGesturePointerDown = vi.fn()
    const onToggle = vi.fn()
    const onSurfaceControl = vi.fn()

    const { container, getByRole } = render(
      <div onPointerDown={onGesturePointerDown}>
        <ComposerDragRegion dragging={false} onDoubleClick={onToggle} />
        <button onClick={onSurfaceControl} type="button">
          Surface control
        </button>
      </div>
    )

    const dragRegion = container.querySelector<HTMLElement>('[data-slot="composer-drag-region"]')
    const topEdge = container.querySelector<HTMLElement>('[data-slot="composer-drag-hit-target-top"]')

    expect(dragRegion).not.toBeNull()
    expect(topEdge).not.toBeNull()
    expect(dragRegion?.className).toContain('pointer-events-none')
    expect(dragRegion?.className).toContain('z-5')
    expect(topEdge?.className).toContain('pointer-events-auto')

    fireEvent.pointerDown(topEdge!)
    fireEvent.doubleClick(topEdge!)
    fireEvent.click(getByRole('button', { name: 'Surface control' }))

    expect(onGesturePointerDown).toHaveBeenCalledTimes(1)
    expect(onToggle).toHaveBeenCalledTimes(1)
    expect(onSurfaceControl).toHaveBeenCalledTimes(1)
  })

  it('keeps the drag cursor on every narrow edge', () => {
    const { container } = render(<ComposerDragRegion dragging onDoubleClick={vi.fn()} />)
    const edges = container.querySelectorAll<HTMLElement>('[data-slot^="composer-drag-hit-target-"]')

    expect(edges).toHaveLength(3)

    for (const edge of edges) {
      expect(edge.className).toContain('pointer-events-auto')
      expect(edge.className).toContain('cursor-grabbing')
    }
  })
})
