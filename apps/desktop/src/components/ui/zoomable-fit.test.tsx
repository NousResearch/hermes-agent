import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Zoomable } from './zoomable'

// jsdom computes no layout: the fit path reads clientWidth/clientHeight off
// the stage and scrollWidth/scrollHeight off the content wrapper, so plain
// DOM properties can stand in for real geometry. The fake observer mirrors
// browser timing — entries are delivered after layout, not during mount.
class FakeResizeObserver {
  private static instances: FakeResizeObserver[] = []
  private targets = new Set<Element>()

  constructor(private readonly cb: ResizeObserverCallback) {
    FakeResizeObserver.instances.push(this)
  }

  observe(target: Element) {
    this.targets.add(target)
  }

  unobserve(target: Element) {
    this.targets.delete(target)
  }

  disconnect() {
    this.targets.clear()
  }

  static deliver() {
    for (const instance of FakeResizeObserver.instances) {
      for (const target of instance.targets) {
        instance.cb([], instance as unknown as ResizeObserver)
      }
    }
  }

  static reset() {
    FakeResizeObserver.instances = []
  }
}

function installFakeObserver() {
  vi.stubGlobal('ResizeObserver', FakeResizeObserver)
}

// The content wrapper carries the transform style; the stage is the pan/zoom
// surface two levels up.
function findNodes() {
  const overlay = screen.getByTestId('overlay')
  const content = overlay.parentElement as HTMLElement
  const stage = content.parentElement?.parentElement as HTMLElement

  return { content, stage }
}

function fakeLayout(stage: HTMLElement, content: HTMLElement, w = 4758, h = 1671) {
  Object.defineProperty(stage, 'clientWidth', { configurable: true, value: 1000 })
  Object.defineProperty(stage, 'clientHeight', { configurable: true, value: 600 })
  Object.defineProperty(content, 'scrollWidth', { configurable: true, value: w })
  Object.defineProperty(content, 'scrollHeight', { configurable: true, value: h })
}

afterEach(() => {
  cleanup()
  FakeResizeObserver.reset()
  vi.unstubAllGlobals()
})

describe('Zoomable overlay fit', () => {
  it('fits oversized content into the stage when the overlay opens', () => {
    installFakeObserver()
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    fireEvent.click(screen.getByTitle('Open diagram'))

    const { content, stage } = findNodes()
    fakeLayout(stage, content)
    act(() => {
      FakeResizeObserver.deliver()
    })

    // 1000/4758 ≈ 0.2102 — below the old MIN_SCALE floor of 0.25, so this
    // also pins the lower clamp.
    expect(content.style.transform).toBe('translate(0px, 0px) scale(0.2101723413198823)')
  })

  it('does not scale content that already fits beyond 100%', () => {
    installFakeObserver()
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    fireEvent.click(screen.getByTitle('Open diagram'))

    const { content, stage } = findNodes()
    fakeLayout(stage, content, 800, 400)
    act(() => {
      FakeResizeObserver.deliver()
    })

    expect(content.style.transform).toBe('translate(0px, 0px) scale(1)')
  })

  it('keeps zero-size content at identity until real geometry exists', () => {
    installFakeObserver()
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    fireEvent.click(screen.getByTitle('Open diagram'))

    const { content, stage } = findNodes()
    fakeLayout(stage, content, 0, 0)
    act(() => {
      FakeResizeObserver.deliver()
    })

    // Mermaid renders async: on first layout the content has no size yet and
    // the fit must not collapse the transform to a zero scale.
    expect(content.style.transform).toBe('translate(0px, 0px) scale(1)')
  })

  it('returns to the fitted view from the toolbar reset button', () => {
    installFakeObserver()
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    fireEvent.click(screen.getByTitle('Open diagram'))

    const { content, stage } = findNodes()
    fakeLayout(stage, content)
    act(() => {
      FakeResizeObserver.deliver()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(content.style.transform).not.toBe('translate(0px, 0px) scale(0.2101723413198823)')

    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
    expect(content.style.transform).toBe('translate(0px, 0px) scale(0.2101723413198823)')

    // After reset the view is the fitted one again, so a later geometry
    // change must re-fit rather than leave the reset scale behind.
    fakeLayout(stage, content, 2379, 835)
    act(() => {
      FakeResizeObserver.deliver()
    })

    expect(content.style.transform).toBe('translate(0px, 0px) scale(0.4203446826397646)')
  })

  it('keeps the fitted view tracking later geometry changes', () => {
    installFakeObserver()
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    fireEvent.click(screen.getByTitle('Open diagram'))

    const { content, stage } = findNodes()

    // Mermaid SVG lands after the dialog opened: a second delivery with the
    // real geometry must re-fit instead of leaving the identity transform.
    fakeLayout(stage, content)
    act(() => {
      FakeResizeObserver.deliver()
    })
    fakeLayout(stage, content, 2379, 835)
    act(() => {
      FakeResizeObserver.deliver()
    })

    expect(content.style.transform).toBe('translate(0px, 0px) scale(0.4203446826397646)')
  })
})
