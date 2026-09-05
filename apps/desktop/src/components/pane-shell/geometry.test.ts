import { describe, expect, it, vi } from 'vitest'

import { beginSashDrag, endSashDrag, isSashDragging, onSashDragEnd } from './geometry'

describe('sash drag lifecycle', () => {
  it('notifies subscribers once when the outermost drag ends', () => {
    const ended = vi.fn()
    const unsubscribe = onSashDragEnd(ended)

    beginSashDrag()
    beginSashDrag()
    expect(isSashDragging()).toBe(true)

    endSashDrag()
    expect(isSashDragging()).toBe(true)
    expect(ended).not.toHaveBeenCalled()

    endSashDrag()
    expect(isSashDragging()).toBe(false)
    expect(ended).toHaveBeenCalledOnce()

    endSashDrag()
    expect(ended).toHaveBeenCalledOnce()

    unsubscribe()
    beginSashDrag()
    endSashDrag()
    expect(ended).toHaveBeenCalledOnce()
  })
})
