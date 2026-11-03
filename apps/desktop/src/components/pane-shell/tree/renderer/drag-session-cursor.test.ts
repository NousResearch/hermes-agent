import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startDragSession } from './drag-session'

// Tabs show the plain arrow (a hand cursor moves the hotspot off the arrow tip
// and presses land on the ✕). The grabbing hand belongs to the DRAG alone: it
// must appear only once the gesture engages, win over every element cursor
// while engaged, and disappear on release.

const press = (el: HTMLElement, x: number, y: number) =>
  ({ button: 0, clientX: x, clientY: y, currentTarget: el, pointerId: 1 }) as unknown as ReactPointerEvent<HTMLElement>

const fire = (type: string, x: number, y: number) =>
  window.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: x, clientY: y }))

const dragCursor = () => document.documentElement.dataset.dragCursor

const spec = (onTap = vi.fn()) => ({
  onCommit: vi.fn(),
  onEngage: vi.fn(),
  onTap,
  resolveMove: () => ({ groupId: 'g', pos: 'center' }) as never
})

beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    cb(0)

    return 1
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  delete document.documentElement.dataset.dragCursor
})

describe('drag cursor', () => {
  it('a plain click on a tab never shows the grabbing hand', () => {
    const onTap = vi.fn()
    startDragSession(press(document.createElement('div'), 10, 10), spec(onTap))

    fire('pointermove', 11, 11)
    expect(dragCursor()).toBeUndefined()

    fire('pointerup', 11, 11)
    expect(onTap).toHaveBeenCalledOnce()
    expect(dragCursor()).toBeUndefined()
  })

  it('an engaged drag shows the grabbing hand app-wide until release', () => {
    startDragSession(press(document.createElement('div'), 10, 10), spec())

    fire('pointermove', 40, 10)
    expect(dragCursor()).toBe('grabbing')

    fire('pointerup', 40, 10)
    expect(dragCursor()).toBeUndefined()
  })
})
