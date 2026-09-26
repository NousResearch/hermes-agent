import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { Profiler } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $layoutTree } from './store'
import { $zoneEditorOpen, ZoneEditor } from './zone-editor'

// Split preview follows the pointer before a click. Pointermove outruns a frame,
// and a repeated event in the same snap cell must not write again. Shift flips
// the line on the next frame while the pointer is still in the canvas — including
// while the button is still down — and the grid is not written until that click
// commits.

type Frame = { cb: FrameRequestCallback; id: number }

let queued: Frame[] = []
let seq = 1
let commits = 0

function rect(): DOMRect {
  return {
    bottom: 1000,
    height: 1000,
    left: 0,
    right: 1000,
    toJSON: () => ({}),
    top: 0,
    width: 1000,
    x: 0,
    y: 0
  } as DOMRect
}

function installFrames() {
  queued = []
  seq = 1
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    const id = seq++
    queued.push({ cb, id })

    return id
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => {
    queued = queued.filter(frame => frame.id !== id)
  })
}

function flushFrames() {
  const batch = queued.splice(0, queued.length)

  act(() => {
    for (const frame of batch) {
      frame.cb(0)
    }
  })
}

function zoneLabels(canvas: HTMLElement) {
  return [...canvas.querySelectorAll('span')]
    .map(node => node.textContent ?? '')
    .filter(text => /^zone \d+$/.test(text))
}

function previewLine(canvas: HTMLElement): HTMLElement | null {
  return (
    [...canvas.querySelectorAll<HTMLElement>(':scope > div')].find(el => {
      return el.className.includes('pointer-events-none') && (el.style.width === '2px' || el.style.height === '2px')
    }) ?? null
  )
}

function mount() {
  $zoneEditorOpen.set(true)

  const view = render(
    <Profiler
      id="zone-editor"
      onRender={() => {
        commits += 1
      }}
    >
      <ZoneEditor />
    </Profiler>
  )

  const canvas = view.container.querySelector('.cursor-crosshair') as HTMLDivElement
  canvas.getBoundingClientRect = () => rect()

  return canvas
}

beforeEach(() => {
  commits = 0
  $layoutTree.set(null)
  $zoneEditorOpen.set(false)
  installFrames()
})

afterEach(() => {
  act(() => {
    $zoneEditorOpen.set(false)
  })
  cleanup()
  vi.unstubAllGlobals()
})

describe('zone editor split preview', () => {
  it('coalesces pointer moves to one preview update per frame', () => {
    const canvas = mount()
    const layout = $layoutTree.get()
    const painted = commits

    fireEvent.pointerMove(canvas, { clientX: 500, clientY: 400 })
    fireEvent.pointerMove(canvas, { clientX: 450, clientY: 420 })
    fireEvent.pointerMove(canvas, { clientX: 480, clientY: 410 })

    expect(previewLine(canvas)).toBeNull()
    expect(commits).toBe(painted)
    expect(queued).toHaveLength(1)
    expect(zoneLabels(canvas)).toHaveLength(3)
    expect($layoutTree.get()).toBe(layout)

    flushFrames()

    expect(commits).toBe(painted + 1)
    expect(previewLine(canvas)?.style.left).toBe('48%')
    expect(previewLine(canvas)?.style.width).toBe('2px')
    expect(zoneLabels(canvas)).toHaveLength(3)
    expect($layoutTree.get()).toBe(layout)
  })

  it('does not write when later pointer events snap to the same preview', () => {
    const canvas = mount()

    fireEvent.pointerMove(canvas, { clientX: 500, clientY: 500 })
    flushFrames()
    expect(previewLine(canvas)?.style.left).toBe('50%')

    const painted = commits

    fireEvent.pointerMove(canvas, { clientX: 501, clientY: 500 })
    fireEvent.pointerMove(canvas, { clientX: 502, clientY: 499 })
    flushFrames()
    fireEvent.pointerMove(canvas, { clientX: 500, clientY: 501 })
    flushFrames()

    expect(commits).toBe(painted)
    expect(previewLine(canvas)?.style.left).toBe('50%')
    expect(previewLine(canvas)?.style.width).toBe('2px')
    expect(zoneLabels(canvas)).toHaveLength(3)
  })

  it('recomputes orientation on Shift while the pointer remains in the canvas', () => {
    const canvas = mount()
    const layout = $layoutTree.get()

    fireEvent.pointerMove(canvas, { clientX: 500, clientY: 500 })
    flushFrames()
    expect(previewLine(canvas)?.style.width).toBe('2px')

    fireEvent.keyDown(window, { key: 'Shift' })

    // Shift state may render immediately; the line waits for the next frame.
    expect(previewLine(canvas)?.style.width).toBe('2px')
    expect(zoneLabels(canvas)).toHaveLength(3)
    const afterKey = commits

    flushFrames()

    expect(previewLine(canvas)?.style.height).toBe('2px')
    expect(commits).toBe(afterKey + 1)
    expect(previewLine(canvas)?.style.top).toBe('50%')
    expect(zoneLabels(canvas)).toHaveLength(3)
    expect($layoutTree.get()).toBe(layout)

    fireEvent.keyUp(window, { key: 'Shift' })
    flushFrames()

    expect(previewLine(canvas)?.style.width).toBe('2px')
    expect(previewLine(canvas)?.style.left).toBe('50%')
    expect(zoneLabels(canvas)).toHaveLength(3)
  })

  it('recomputes orientation on Shift while the pointer is still down and commits the grid only on release', () => {
    const canvas = mount()
    const layout = $layoutTree.get()

    fireEvent.pointerMove(canvas, { clientX: 500, clientY: 500 })
    flushFrames()

    fireEvent.pointerDown(canvas, { button: 0, clientX: 500, clientY: 500 })
    fireEvent.keyDown(window, { key: 'Shift' })
    flushFrames()

    expect(previewLine(canvas)?.style.height).toBe('2px')
    expect(zoneLabels(canvas)).toHaveLength(3)
    expect($layoutTree.get()).toBe(layout)

    fireEvent.pointerUp(window, { button: 0, clientX: 500, clientY: 500 })

    expect(zoneLabels(canvas)).toHaveLength(4)
    expect(
      [...canvas.querySelectorAll<HTMLElement>(':scope > div')].some(
        el => /^zone \d+$/.test(el.textContent ?? '') && el.style.height === '50%'
      )
    ).toBe(true)
    expect($layoutTree.get()).toBe(layout)
  })
})
