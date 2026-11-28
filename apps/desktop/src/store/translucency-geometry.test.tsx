// @vitest-environment jsdom
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'MacIntel' })
})

import { group, split } from '@/components/pane-shell/tree/model'
import { LayoutTreeRoot } from '@/components/pane-shell/tree/renderer'
import {
  $collapsedTreeSides,
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  mirrorLayoutTree
} from '@/components/pane-shell/tree/store'
import { Sidebar, SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { stubMenuDomApis } from '@/test/jsdom'

import {
  $translucencyBook,
  GLASS_SUPPORTED,
  setTranslucency,
  setTranslucencyMode,
  setTranslucencyScope
} from './translucency'

// Real tree + Sidebar lifecycles; jsdom supplies no layout/RO delivery. The
// browser regression separately exercises actual CSS layout and native RO.
const observers = new Set<ResizeObserverProbe>()

class ResizeObserverProbe {
  targets = new Set<Element>()
  constructor(readonly callback: ResizeObserverCallback) {
    observers.add(this)
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
}

const disposers: Array<() => void> = []
let frames: Map<number, FrameRequestCallback>
let nextFrame: number
let railLeft: number
let railWidth: number
let railHeight: number
let rectReads: number

const flush = async () => {
  await act(async () => {
    await Promise.resolve()
    const pending = [...frames.values()]
    frames.clear()
    pending.forEach(callback => callback(0))
    await Promise.resolve()
  })
}

const interval = () => {
  const root = window.document.documentElement
  const physical = ['--glass-rail-left', '--glass-rail-right'].map(name => root.style.getPropertyValue(name))

  // Describe the OLD painter too, so red proves stale/wrong paint rather
  // than merely requiring the new custom-property names to exist.
  if (physical.every(value => value === '')) {
    const edge = root.style.getPropertyValue('--glass-rail-edge')

    if (edge) {
      return root.dir === 'rtl'
        ? [`${window.innerWidth - parseFloat(edge)}px`, `${window.innerWidth}px`]
        : ['0px', edge]
    }
  }

  return physical
}

function registerPanes() {
  disposers.push(
    registry.register({
      area: 'panes',
      id: 'sessions',
      title: 'Sessions',
      data: { placement: 'left', width: '160px', lifecycleKeepAlive: true },
      render: () => <Sidebar collapsible="none">Sessions</Sidebar>
    }),
    registry.register({
      area: 'panes',
      id: 'chat',
      title: 'Chat',
      data: { placement: 'main' },
      render: () => <div data-chat-surface />
    }),
    registry.register({
      area: 'panes',
      id: 'files',
      title: 'Files',
      data: { placement: 'right', width: '160px' },
      render: () => <div />
    })
  )
}

const initialTree = () =>
  split('row', [group(['sessions'], { id: 'sessions-zone' }), group(['chat']), group(['files'])])

function mountShell(titlebar = false) {
  return render(
    <SidebarProvider>
      <LayoutTreeRoot titlebar={titlebar} />
    </SidebarProvider>
  )
}

beforeEach(() => {
  frames = new Map()
  nextFrame = 0
  railLeft = 0
  railWidth = 160
  railHeight = 600
  rectReads = 0
  observers.clear()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frames.set(++nextFrame, callback)

    return nextFrame
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
  vi.stubGlobal('ResizeObserver', ResizeObserverProbe)
  stubMenuDomApis()
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 900 })
  window.document.documentElement.dir = 'ltr'
  $hiddenTreePanes.set(new Set())
  $dismissedPanes.set(new Set())
  $collapsedTreeSides.set(new Set())
  $narrowViewport.set(false)
  $layoutTree.set(null)
  setTranslucencyMode('clear')
  $translucencyBook.set({ mode: 'glass', base: { intensity: 29, fade: 0, scope: 'sidebar' }, light: {}, dark: {} })
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.slot === 'sidebar') {
      rectReads += 1

      return new DOMRect(railLeft, 0, railWidth, railHeight)
    }

    return new DOMRect()
  })
})

afterEach(() => {
  setTranslucencyMode('clear')
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
  window.document.documentElement.removeAttribute('dir')
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('sidebar glass at the mounted layout seam', () => {
  it.each([false, true])('discovers cold mounts and same-size connected moves (titlebar=%s)', async titlebar => {
    expect(GLASS_SUPPORTED).toBe(true)
    registerPanes()
    mountShell(titlebar)
    await flush()
    expect(interval()).toEqual(['0px', '0px'])

    // Glass is already enabled before the Sessions contribution mounts.
    act(() => $layoutTree.set(initialTree()))
    await flush()
    expect(interval()).toEqual(['0px', '160px'])
    const rail = window.document.querySelector('[data-slot="sidebar"]')!
    const book = $translucencyBook.get()

    // The real flip action commits a new tree, but does not resize/remount the
    // retained rail. Deliberately deliver NO ResizeObserver or window event.
    railLeft = 740
    act(mirrorLayoutTree)
    await flush()
    expect(window.document.querySelector('[data-slot="sidebar"]')).toBe(rail)
    expect(rail.isConnected).toBe(true)
    expect(interval()).toEqual(['740px', '900px'])

    // A non-mirror rearrangement and RTL use the same physical coordinates.
    railLeft = 230
    window.document.documentElement.dir = 'rtl'
    act(() => $layoutTree.set(split('row', [group(['files']), group(['sessions']), group(['chat'])])))
    await flush()
    expect(interval()).toEqual(['230px', '390px'])
    expect($translucencyBook.get()).toBe(book)
  })

  it('moves a warm connected rail without an observer, resize, or settings change', async () => {
    registerPanes()
    $layoutTree.set(initialTree())
    mountShell()
    // Establish the old implementation's only reliable initial measurement.
    setTranslucencyScope('window')
    setTranslucencyScope('sidebar')
    await flush()
    expect(interval()).toEqual(['0px', '160px'])
    const book = $translucencyBook.get()
    const rail = window.document.querySelector('[data-slot="sidebar"]')!
    railLeft = 740
    act(mirrorLayoutTree)
    await flush()
    expect(rail.isConnected).toBe(true)
    expect(window.document.querySelector('[data-slot="sidebar"]')).toBe(rail)
    expect(interval()).toEqual(['740px', '900px'])
    expect($translucencyBook.get()).toBe(book)
  })

  it('empties hidden keep-alive rails and rediscovers restore, remount, and zero-size states', async () => {
    registerPanes()
    $layoutTree.set(initialTree())
    mountShell()
    await flush()
    const original = window.document.querySelector('[data-slot="sidebar"]')!
    expect(interval()).toEqual(['0px', '160px'])

    act(() => $hiddenTreePanes.set(new Set(['sessions'])))
    await flush()
    expect(window.document.querySelector('[data-slot="sidebar"]')).toBe(original)
    expect(original.closest('[data-pane-hidden]')).not.toBeNull()
    expect(interval()).toEqual(['0px', '0px'])

    railLeft = 300
    act(() => $hiddenTreePanes.set(new Set()))
    await flush()
    expect(interval()).toEqual(['300px', '460px'])

    // Removing a contribution is destruction; restoring it is a cold mount.
    act(() => disposers[0]())
    await flush()
    expect(interval()).toEqual(['0px', '0px'])
    act(() => {
      disposers.push(
        registry.register({
          area: 'panes',
          id: 'sessions',
          title: 'Sessions',
          data: { placement: 'left', width: '160px', lifecycleKeepAlive: true },
          render: () => <Sidebar collapsible="none">Sessions</Sidebar>
        })
      )
    })
    await flush()
    expect(window.document.querySelector('[data-slot="sidebar"]')).not.toBe(original)
    expect(interval()).toEqual(['300px', '460px'])

    railWidth = 0
    window.dispatchEvent(new Event('resize'))
    await flush()
    expect(interval()).toEqual(['0px', '0px'])
    railWidth = 160
    railHeight = 0
    window.dispatchEvent(new Event('resize'))
    await flush()
    expect(interval()).toEqual(['0px', '0px'])
  })

  it('tracks size and neighboring-track reflow, clamps to the viewport, and cleans up outside sidebar scope', async () => {
    registerPanes()
    $layoutTree.set(initialTree())
    mountShell()
    await flush()
    const rail = window.document.querySelector('[data-slot="sidebar"]')!
    const track = window.document.querySelector('[data-tree-split]')!.firstElementChild!

    const notifyResize = (target: Element) => {
      for (const observer of observers) {
        if (observer.targets.has(target)) {
          observer.callback([{ target } as ResizeObserverEntry], observer as unknown as ResizeObserver)
        }
      }
    }

    railLeft = 120
    // During sash preview the store is not written and the rail may retain
    // its own size. A neighboring split track supplies the size notification.
    notifyResize(track)
    await flush()
    expect(interval()).toEqual(['120px', '280px'])
    railLeft = -30
    railWidth = 200
    notifyResize(rail)
    await flush()
    expect(interval()).toEqual(['0px', '170px'])
    railLeft = 850
    notifyResize(rail)
    await flush()
    expect(interval()).toEqual(['850px', '900px'])

    rectReads = 0

    for (const tint of [30, 31, 32]) {
      act(() => setTranslucency(tint))
    }

    await flush()
    expect(rectReads).toBe(0)

    // No-op notifications must not dirty :root's style on every sash frame.
    const write = vi.spyOn(window.document.documentElement.style, 'setProperty')
    notifyResize(rail)
    await flush()
    expect(write).not.toHaveBeenCalled()

    setTranslucencyScope('window')
    expect(interval()).toEqual(['', ''])
    rectReads = 0
    notifyResize(rail)
    window.dispatchEvent(new Event('resize'))
    await flush()
    expect(rectReads).toBe(0)

    setTranslucencyScope('sidebar')
    await flush()
    expect(interval()).toEqual(['850px', '900px'])
    // A scheduled update must not resurrect geometry after switching modes.
    window.dispatchEvent(new Event('resize'))
    setTranslucencyMode('clear')
    await flush()
    expect(interval()).toEqual(['', ''])
  })
})
