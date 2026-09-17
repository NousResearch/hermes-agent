import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneBody } from './pane-body'

let root: null | Root = null
let container: HTMLDivElement | null = null

function render(ui: ReactNode) {
  if (!container) {
    container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    root = createRoot(container)
  }

  act(() => {
    root!.render(ui)
  })
}

afterEach(() => {
  if (root) {
    act(() => root!.unmount())
  }

  container?.remove()
  root = null
  container = null
  vi.restoreAllMocks()
})

describe('PaneBody', () => {
  it('caches the last visible viewport while the zone is collapsed', () => {
    let observed: null | ResizeObserverCallback = null
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          observed = callback
        }
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    )

    render(
      <PaneBody hidden={false}>
        <div data-guest />
      </PaneBody>
    )
    const body = container!.firstElementChild as HTMLElement
    act(() => {
      observed?.(
        [{ contentRect: { width: 640, height: 480 } } as ResizeObserverEntry],
        {} as ResizeObserver
      )
    })

    render(
      <PaneBody hidden>
        <div data-guest />
      </PaneBody>
    )
    expect(body.style.width).toBe('640px')
    expect(body.style.height).toBe('480px')
    expect(body.style.visibility).toBe('hidden')
    expect(container!.querySelector('[data-guest]')).not.toBeNull()
  })
})
