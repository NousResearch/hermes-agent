import { describe, expect, it, vi } from 'vitest'

import { projectTreeViewportSize } from './tree'

describe('projectTreeViewportSize', () => {
  it('uses ResizeObserver contentRect without forcing another layout read', () => {
    const element = document.createElement('div')
    const getBoundingClientRect = vi.spyOn(element, 'getBoundingClientRect')
    const contentRect = { height: 480, width: 320 } as DOMRectReadOnly

    expect(
      projectTreeViewportSize([{ contentRect, target: element } as unknown as ResizeObserverEntry], element)
    ).toEqual({
      height: 480,
      width: 320
    })
    expect(getBoundingClientRect).not.toHaveBeenCalled()
  })

  it('falls back to a rect when ResizeObserver is unavailable', () => {
    const element = document.createElement('div')
    vi.spyOn(element, 'getBoundingClientRect').mockReturnValue({
      height: 240,
      width: 160
    } as DOMRect)

    expect(projectTreeViewportSize([], element)).toEqual({ height: 240, width: 160 })
  })
})
