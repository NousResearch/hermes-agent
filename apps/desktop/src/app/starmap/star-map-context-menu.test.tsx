import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AppContextMenu } from '@/app/context-menu/app-context-menu'
import { $contextMenu } from '@/app/context-menu/store'
import type { StarmapGraph } from '@/types/hermes'

import { StarMap } from './star-map'

class TestResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}

  disconnect() {}

  observe(target: Element) {
    Object.defineProperties(target, {
      clientHeight: { configurable: true, value: 300 },
      clientWidth: { configurable: true, value: 400 }
    })
    this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }

  unobserve() {}
}

const graph: StarmapGraph = {
  clusters: [],
  edges: [],
  memory: [],
  nodes: [
    {
      category: 'coding',
      createdBy: null,
      id: 'skill-a',
      kind: 'skill',
      label: 'Skill A',
      pinned: false,
      state: 'active',
      useCount: 1
    }
  ],
  stats: {}
}

vi.mock('./simulation', () => ({
  buildSimulation: (input: StarmapGraph) => {
    const nodes = input.nodes.map(node => ({
      ...node,
      outerRingIndex: 0,
      rec: 1,
      tr: 0,
      vx: 0,
      vy: 0,
      x: 0,
      y: 0
    }))

    return {
      byId: new Map(nodes.map(node => [node.id, node])),
      links: [],
      nodes,
      rings: [{ label: null, r: 200, ratio: 1 }],
      sim: { stop: vi.fn() }
    }
  }
}))

afterEach(() => {
  $contextMenu.set(null)
  cleanup()
  vi.restoreAllMocks()
})

describe('StarMap context menu ownership', () => {
  it('keeps the canvas gesture so a node can open its Edit/Delete menu', async () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)

    const { container } = render(
      <MemoryRouter>
        <AppContextMenu />
        <StarMap graph={graph} />
      </MemoryRouter>
    )

    const canvas = container.querySelector('canvas')!

    await waitFor(() => expect(canvas.style.width).toBe('400px'))
    fireEvent.contextMenu(canvas, { clientX: 200, clientY: 165 })

    expect(await screen.findByText('Edit skill…')).toBeTruthy()
    expect(screen.getByText('Archive skill')).toBeTruthy()
    expect($contextMenu.get()).toBeNull()
  })
})
