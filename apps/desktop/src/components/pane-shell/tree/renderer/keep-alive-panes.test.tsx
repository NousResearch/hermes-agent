import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { createElement, useEffect } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { stubMenuDomApis } from '@/test/jsdom'

import { PANE_TOGGLE_REVEAL_EVENT } from '../..'
import { usePaneGroup, usePaneLifecycle, usePaneVisible } from '../../pane-visibility'
import { group, type LayoutNode, split } from '../model'
import {
  $activeTreeGroup,
  $collapsedTreeSides,
  $dismissedPanes,
  $hiddenTreePanes,
  $hoveredTreeGroup,
  $layoutTree,
  $narrowViewport,
  $treePaneEpochs,
  reloadTreePane
} from '../store'

import { snapshotZones } from './drag-session'

import { LayoutTreeRoot } from '.'

// A native guest has a lifetime beyond React. DOM identity alone misses a
// detach/reparent (even moveBefore destroys Electron's webview guest).
const disconnected: HTMLElement[] = []

class LiveGuest extends HTMLElement {
  disconnectedCallback() {
    disconnected.push(this)
  }
}
customElements.define('pane-live-guest', LiveGuest)

// jsdom has no layout. Deliver real observer notifications to the production
// shared observer; the browser harness separately checks CSS anchor geometry.
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

function resize(target: HTMLElement, width: number, height: number) {
  const size = { inlineSize: width, blockSize: height }

  const entry: ResizeObserverEntry = {
    target,
    contentRect: new DOMRectReadOnly(0, 0, width, height),
    borderBoxSize: [size],
    contentBoxSize: [size],
    devicePixelContentBoxSize: [size]
  }

  for (const observer of observers) {
    if (observer.targets.has(target)) {
      observer.callback([entry], observer)
    }
  }
}

const disposers: (() => void)[] = []
const mounts = new Map<string, number>()
const unmounts = new Map<string, number>()
const renders = new Map<string, number>()

function registerPane(id: string, keepAlive = true, data: Record<string, unknown> = {}) {
  function Probe() {
    const groupId = usePaneGroup()
    const visible = usePaneVisible()
    const lifecycle = usePaneLifecycle()
    renders.set(id, (renders.get(id) ?? 0) + 1)
    useEffect(() => {
      mounts.set(id, (mounts.get(id) ?? 0) + 1)

      return () => {
        unmounts.set(id, (unmounts.get(id) ?? 0) + 1)
      }
    }, [])

    return createElement(
      'pane-live-guest',
      {
        'data-guest': id,
        'data-group': groupId,
        'data-visible': String(visible),
        'data-lifecycle': lifecycle
      },
      <input defaultValue="original" />
    )
  }

  const dispose = registry.register({
    area: 'panes',
    id,
    title: id,
    data: { ...data, lifecycleKeepAlive: keepAlive },
    render: Probe
  })

  disposers.push(dispose)

  return dispose
}

function setTree(tree: LayoutNode | null) {
  act(() => $layoutTree.set(tree))
}

const guest = (id = 'live') => document.querySelector<HTMLElement>(`[data-guest="${id}"]`)

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverProbe)
  stubMenuDomApis()
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  $hiddenTreePanes.set(new Set())
  $dismissedPanes.set(new Set())
  $collapsedTreeSides.set(new Set())
  $treePaneEpochs.set({})
  $narrowViewport.set(false)
  disconnected.length = 0
  mounts.clear()
  unmounts.clear()
  renders.clear()
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
  vi.unstubAllGlobals()
})

it('keeps a live body continuously connected across replacement IDs, ancestry and missing placement', () => {
  registerPane('live')
  registerPane('plain', false)
  setTree(group(['live'], { id: 'original' }))
  render(<LayoutTreeRoot />)
  const page = guest()!
  const input = page.querySelector('input')!
  input.value = 'unsaved page state'

  setTree(
    split('row', [
      group(['plain'], { id: 'chat' }),
      split('column', [group(['live'], { id: 'replacement' }), group([], { id: 'empty' })])
    ])
  )
  expect(guest()).toBe(page)
  expect(page.dataset.group).toBe('replacement')
  expect(page.dataset.visible).toBe('true')
  expect(input.value).toBe('unsaved page state')

  setTree(group(['plain'], { id: 'other-layout' }))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('false')
  expect(page.dataset.lifecycle).toBe('hot-hidden')
  expect(page.closest('[data-pane-hidden]')?.hasAttribute('inert')).toBe(true)

  setTree(group(['live'], { id: 'minimized-replacement', minimized: true }))
  expect(guest()).toBe(page)
  expect(page.dataset.group).toBe('minimized-replacement')
  expect(page.dataset.visible).toBe('false')

  setTree(null)
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('false')

  setTree(group(['live'], { id: 'restored' }))
  expect(guest()).toBe(page)
  expect(page.dataset.group).toBe('restored')
  expect(page.closest('[data-pane-hidden]')).toBeNull()
  expect(mounts.get('live')).toBe(1)
  expect(unmounts.get('live') ?? 0).toBe(0)
  expect(disconnected).not.toContain(page)
})

it('destroys removed contributions and does not revive their old activation on re-registration', () => {
  const remove = registerPane('live')
  registerPane('plain', false)
  setTree(group(['live', 'plain'], { id: 'zone' }))
  render(<LayoutTreeRoot />)
  const page = guest()!
  setTree(group(['live', 'plain'], { active: 'plain', id: 'zone' }))
  act(remove)
  expect(guest()).toBeNull()
  expect(unmounts.get('live')).toBe(1)
  expect(disconnected).toContain(page)
  act(() => {
    registerPane('live')
  })
  expect(guest()).toBeNull()
  setTree(group(['live', 'plain'], { active: 'live', id: 'zone' }))
  expect(guest()).not.toBe(page)
  expect(mounts.get('live')).toBe(2)
})

it('gates hidden side guests without mounting never-activated tabs or disconnecting live ones', () => {
  registerPane('live')
  registerPane('background')
  disposers.push(
    registry.register({
      area: 'panes',
      id: 'workspace',
      data: { placement: 'main' },
      render: () => <div />
    })
  )
  setTree(split('row', [group(['workspace']), group(['live', 'background'], { id: 'side' })]))
  $collapsedTreeSides.set(new Set(['right']))
  render(<LayoutTreeRoot />)
  expect(guest()).toBeNull()
  act(() => $collapsedTreeSides.set(new Set()))
  const page = guest()!
  expect(page).not.toBeNull()
  expect(guest('background')).toBeNull()
  act(() => $collapsedTreeSides.set(new Set(['right'])))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('false')
  expect(page.closest('[data-pane-hidden]')?.hasAttribute('inert')).toBe(true)
  act(() => $collapsedTreeSides.set(new Set()))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('true')
  expect(disconnected).not.toContain(page)
})

it('routes DOM group targeting to the new zone without adding duplicate drop zones', () => {
  registerPane('live')
  registerPane('plain', false)
  setTree(group(['live'], { id: 'old-zone' }))
  render(<LayoutTreeRoot />)
  const page = guest()!
  setTree(split('row', [group(['plain'], { id: 'other' }), group(['live'], { id: 'new-zone', tabStrip: 'always' })]))
  expect(page.closest('[data-tree-group]')?.getAttribute('data-tree-group')).toBe('new-zone')
  expect(page.closest('[data-zone-header]')).not.toBeNull()
  fireEvent.pointerDown(page)
  expect($activeTreeGroup.get()).toBe('new-zone')
  fireEvent.pointerOver(page)
  expect($hoveredTreeGroup.get()).toBe('new-zone')
  expect(
    snapshotZones()
      .map(zone => zone.id)
      .sort()
  ).toEqual(['new-zone', 'other'])
})

it('reloads only the explicit epoch target, even while it is hidden', () => {
  registerPane('live')
  registerPane('other')
  setTree(split('row', [group(['live']), group(['other'])]))
  render(<LayoutTreeRoot />)
  const page = guest()!
  const other = guest('other')!
  const host = page.parentElement
  act(() => $hiddenTreePanes.set(new Set(['live'])))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('false')
  act(() => reloadTreePane('live'))
  expect(guest()).not.toBe(page)
  expect(guest()!.parentElement).toBe(host)
  expect(guest('other')).toBe(other)
  expect(mounts.get('live')).toBe(2)
  expect(unmounts.get('live')).toBe(1)
  expect(disconnected).toContain(page)
  expect(disconnected).not.toContain(other)
})

it('keeps live tabs outside the bounded cache while ordinary tabs still park', () => {
  registerPane('live')

  for (const id of ['a', 'b', 'c', 'd']) {
    registerPane(id, false)
  }

  const panes = ['live', 'a', 'b', 'c', 'd']
  setTree(group(panes, { id: 'zone' }))
  render(<LayoutTreeRoot />)
  const page = guest()!

  for (const active of ['a', 'b', 'c', 'd']) {
    setTree(group(panes, { id: 'zone', active }))
  }

  expect(guest()).toBe(page)
  expect(page.dataset.lifecycle).toBe('hot-hidden')
  expect(guest('a')).toBeNull()
  expect(guest('d')).not.toBeNull()
  setTree(group(panes, { id: 'zone', active: 'd', minimized: true }))
  expect(guest()).toBe(page)
  expect(guest('d')).toBeNull()
  expect(disconnected).not.toContain(page)
})

it('remembers the last visible viewport without rerendering a live guest on resize', () => {
  registerPane('live')
  setTree(group(['live'], { id: 'zone' }))
  render(<LayoutTreeRoot />)
  const page = guest()!
  const host = page.closest<HTMLElement>('[data-pane-host]')!
  const priorRenders = renders.get('live')
  act(() => resize(host, 640, 480))
  expect(renders.get('live')).toBe(priorRenders)
  expect(host.style.getPropertyValue('--pane-kept-width')).toBe('640px')
  expect(host.style.getPropertyValue('--pane-kept-height')).toBe('480px')
  setTree(group([], { id: 'absent' }))
  act(() => resize(host, 0, 0))
  expect(host.style.getPropertyValue('--pane-kept-width')).toBe('640px')
  expect(host.style.getPropertyValue('--pane-kept-height')).toBe('480px')
  expect(host.style.width).toBe('var(--pane-kept-width, 0px)')
  expect(host.style.height).toBe('var(--pane-kept-height, 0px)')
  expect(guest()).toBe(page)
})

it('uses the same live guest when a collapsible pane moves into the narrow overlay', () => {
  registerPane('live', true, { collapsible: true, placement: 'right' })
  registerPane('workspace', false, { placement: 'main' })
  setTree(split('row', [group(['workspace']), group(['live'], { id: 'side' })]))
  render(<LayoutTreeRoot />)
  const page = guest()!
  act(() => $narrowViewport.set(true))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('false')
  fireEvent(window, new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: 'live', mode: 'open' } }))
  expect(document.querySelectorAll('[data-guest="live"]')).toHaveLength(1)
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('true')
  fireEvent(window, new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: 'live', mode: 'close' } }))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('false')
  // A hover reveal must treat the overlay chrome and its sibling guest as
  // one hover boundary, but leaving BOTH still dismisses it.
  fireEvent.mouseEnter(document.querySelector('.absolute.inset-y-0.z-30')!)
  const chrome = document.querySelector('[data-narrow-overlay]')!
  const host = page.closest('[data-pane-host]')!
  fireEvent.mouseLeave(chrome, { relatedTarget: host })
  expect(page.dataset.visible).toBe('true')
  fireEvent.mouseLeave(host, { relatedTarget: chrome })
  expect(page.dataset.visible).toBe('true')
  fireEvent.mouseLeave(host, { relatedTarget: document.body })
  expect(page.dataset.visible).toBe('false')
  act(() => $narrowViewport.set(false))
  expect(guest()).toBe(page)
  expect(page.dataset.visible).toBe('true')
  expect(disconnected).not.toContain(page)
})
