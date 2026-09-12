import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { AppContextMenu } from '@/app/context-menu/app-context-menu'
import { $contextMenu } from '@/app/context-menu/store'
import type { StarmapGraph } from '@/types/hermes'

import { TILT } from './constants'
import { fitViewport } from './geometry'
import { buildSimulation } from './simulation'
import { StarMap } from './star-map'

const WIDTH = 800
const HEIGHT = 600

const graph: StarmapGraph = {
  clusters: [{ category: 'memory', count: 1 }],
  edges: [],
  memory: [],
  nodes: [
    {
      category: 'memory',
      createdBy: null,
      id: 'memory:context-menu-regression',
      kind: 'memory',
      label: 'Remember the context menu',
      pinned: false,
      state: 'active',
      timestamp: null,
      useCount: 1
    }
  ],
  stats: {}
}

class TestResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}

  disconnect() {}

  observe(target: Element) {
    Object.defineProperties(target, {
      clientHeight: { configurable: true, value: HEIGHT },
      clientWidth: { configurable: true, value: WIDTH }
    })
    this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }

  unobserve() {}
}

function nodeScreenPoint(): { x: number; y: number } {
  const built = buildSimulation(graph, () => undefined)
  const node = built.nodes[0]!
  const viewport = fitViewport(WIDTH, HEIGHT, built.rings.at(-1)!.r)

  built.sim.stop()

  return {
    x: node.x * viewport.k + viewport.x,
    y: node.y * viewport.k * TILT + viewport.y
  }
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
})

afterEach(() => {
  $contextMenu.set(null)
  cleanup()
  vi.unstubAllGlobals()
})

describe('Star Map context menu coordination', () => {
  it('opens node actions without opening the app shell menu (#109284)', () => {
    const { container } = render(
      <MemoryRouter>
        <AppContextMenu />
        <StarMap graph={graph} />
      </MemoryRouter>
    )

    const canvas = container.querySelector('canvas')!
    const { x, y } = nodeScreenPoint()

    expect(fireEvent.contextMenu(canvas, { button: 2, clientX: x, clientY: y })).toBe(false)
    expect(screen.getByRole('button', { name: 'Edit memory…' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Delete memory' })).toBeTruthy()
    expect($contextMenu.get()).toBeNull()
    expect(screen.queryByText('Settings')).toBeNull()
  })

  it('keeps the app shell menu on an unmarked outside target', async () => {
    render(
      <MemoryRouter>
        <AppContextMenu />
        <button type="button">Outside Star Map</button>
      </MemoryRouter>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Outside Star Map' }))

    expect($contextMenu.get()?.kind).toBe('dom')
    expect(await screen.findByText('Settings')).toBeTruthy()
  })
})
