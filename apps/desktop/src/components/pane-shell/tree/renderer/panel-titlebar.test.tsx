// @vitest-environment jsdom
import { act, cleanup, render } from '@testing-library/react'
import { type RefObject, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TITLEBAR_CHROME_CHANGED_EVENT } from '@/app/shell/titlebar'

import { usePanelTitlebar } from './panel-titlebar'

type ROCallback = (entries: ResizeObserverEntry[], observer: ResizeObserver) => void

interface ObservedInstance {
  callback: ROCallback
  observed: Element[]
}

/** A ResizeObserver we can drive by hand: records targets, fires on demand. */
const roInstances: ObservedInstance[] = []

class ControllableResizeObserver {
  callback: ROCallback
  observed: Element[] = []

  constructor(callback: ROCallback) {
    this.callback = callback
    roInstances.push(this)
  }

  observe(target: Element) {
    this.observed.push(target)
  }

  unobserve(target: Element) {
    this.observed = this.observed.filter(element => element !== target)
  }

  disconnect() {
    this.observed = []
  }
}

/** Live widths so a board switch can widen the kanban chip mid-test. */
const widths = { left: 90 }

function stubRects() {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (
    this: HTMLElement
  ) {
    if (this.dataset.panel === 'sessions') {
      return { left: 0, right: 800, width: 800 } as DOMRect
    }

    if (this.dataset.titlebarCluster === 'left') {
      return { left: 98, right: 98 + widths.left, width: widths.left } as DOMRect
    }

    if (this.dataset.titlebarCluster === 'right') {
      return { left: 1200, right: 1300, width: 100 } as DOMRect
    }

    return { left: 0, right: 0, width: 0 } as DOMRect
  })
}

function Harness({ branch }: { branch: 'chat' | 'kanban' }) {
  const ref = useRef<HTMLDivElement>(null)
  usePanelTitlebar(ref as unknown as RefObject<HTMLElement | null>, true, false)

  return (
    <>
      <div data-panel="sessions" ref={ref} />
      {/* A new DOM node per branch — like the real chat→kanban band swap. */}
      <div data-branch={branch} data-titlebar-cluster="left" />
      <div data-titlebar-cluster="right" />
    </>
  )
}

const reservation = () =>
  document.querySelector('[data-panel="sessions"]')?.getAttribute('style') ?? ''

beforeEach(() => {
  roInstances.length = 0
  widths.left = 90
  vi.stubGlobal('ResizeObserver', ControllableResizeObserver)
  stubRects()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('usePanelTitlebar across a branch flip', () => {
  it('tracks the new cluster after the band swaps and keeps tracking its resizes', () => {
    const { rerender } = render(<Harness branch="chat" />)

    // Arrival: (98 + 90) + 12 = 200px reservation from the chat band.
    expect(reservation()).toContain('--panel-titlebar-left: 200px')

    // Board switch widens the kanban chip in place; TitlebarControls
    // announces the band change exactly as on a real navigation.
    widths.left = 200
    rerender(<Harness branch="kanban" />)
    act(() => {
      window.dispatchEvent(new window.Event(TITLEBAR_CHROME_CHANGED_EVENT))
    })

    expect(reservation()).toContain('--panel-titlebar-left: 310px')

    // The hook re-queried the cluster set on the chrome change — the kanban
    // node (not the unmounted chat one) is what's observed now.
    const kanbanLeft = document.querySelector('[data-branch="kanban"]')
    const hookObserver = roInstances.find(instance => instance.observed.includes(kanbanLeft!))
    expect(hookObserver).toBeDefined()

    // Subsequent drift: a longer board name widens the same chip with no
    // navigation. The observed resize must move the reservation with it.
    widths.left = 260
    act(() => {
      hookObserver!.callback([], hookObserver as unknown as ResizeObserver)
    })

    expect(reservation()).toContain('--panel-titlebar-left: 370px')
  })
})
