import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it } from 'vitest'

import { useKanbanViewportGeometry } from './responsive'

let root: null | Root = null
let container: HTMLDivElement | null = null

function resizeWindow(width: number) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width, writable: true })

  act(() => window.dispatchEvent(new Event('resize')))
}

function Harness() {
  const geometry = useKanbanViewportGeometry()

  return (
    <main data-layout={geometry.desktop ? 'desktop' : 'mobile'} data-testid="viewport">
      <header style={{ display: geometry.desktop ? 'flex' : 'grid', width: geometry.viewportWidth }}>
        <button type="button">Settings</button>
        <button type="button">New task</button>
        <input aria-label="Filter cards" />
      </header>
      <div style={{ display: 'flex', overflowX: 'auto', scrollSnapType: geometry.desktop ? 'none' : 'x mandatory' }}>
        <section style={{ flexShrink: 0, scrollSnapAlign: geometry.desktop ? 'none' : 'start', width: geometry.laneWidth }} />
        <section style={{ flexShrink: 0, scrollSnapAlign: geometry.desktop ? 'none' : 'start', width: geometry.laneWidth }} />
      </div>
      <aside style={{ maxWidth: '100%', width: geometry.drawerWidth }} />
    </main>
  )
}

function render(width: number) {
  resizeWindow(width)
  container = globalThis.document.createElement('div')
  globalThis.document.body.append(container)
  root = createRoot(container)
  act(() => root!.render(<Harness />))

  return container
}

afterEach(() => {
  if (root) {
    act(() => root!.unmount())
  }

  container?.remove()
  root = null
  container = null
})

describe('Kanban responsive geometry', () => {
  for (const width of [320, 375, 390, 430]) {
    it(`keeps the toolbar, snapping lanes, and drawer contained at ${width}px`, () => {
      const view = render(width)
      const header = view.querySelector<HTMLElement>('header')!
      const strip = view.querySelector<HTMLElement>('header + div')!
      const lanes = [...strip.querySelectorAll<HTMLElement>('section')]
      const drawer = view.querySelector<HTMLElement>('aside')!

      expect(view.querySelector('main')?.dataset.layout).toBe('mobile')
      expect(header.style.width).toBe(`${width}px`)
      expect(header.querySelectorAll('button')).toHaveLength(2)
      expect(header.querySelector('input')).toBeTruthy()
      expect(strip.style.overflowX).toBe('auto')
      expect(strip.style.scrollSnapType).toBe('x mandatory')
      expect(lanes.every(lane => lane.style.width === `${width - 32}px`)).toBe(true)
      expect(lanes.every(lane => lane.style.scrollSnapAlign === 'start')).toBe(true)
      expect(drawer.style.width).toBe(`${width}px`)
      expect(Number.parseFloat(drawer.style.width)).toBeLessThanOrEqual(width)
    })
  }

  it('preserves fixed lanes and a side drawer on the desktop control path', () => {
    const view = render(1024)
    const strip = view.querySelector<HTMLElement>('header + div')!
    const lane = strip.querySelector<HTMLElement>('section')!
    const drawer = view.querySelector<HTMLElement>('aside')!

    expect(view.querySelector('main')?.dataset.layout).toBe('desktop')
    expect(strip.style.scrollSnapType).toBe('none')
    expect(lane.style.width).toBe('256px')
    expect(lane.style.scrollSnapAlign).toBe('none')
    expect(drawer.style.width).toBe('416px')
  })

  it('updates live when the viewport crosses the desktop breakpoint', () => {
    const view = render(430)

    expect(view.querySelector('main')?.dataset.layout).toBe('mobile')
    resizeWindow(768)
    expect(view.querySelector('main')?.dataset.layout).toBe('desktop')
    expect(view.querySelector<HTMLElement>('aside')?.style.width).toBe('416px')
  })
})
