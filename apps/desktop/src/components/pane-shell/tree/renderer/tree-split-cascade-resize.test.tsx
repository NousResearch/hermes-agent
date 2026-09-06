import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'

import { group, split, type SplitNode } from '../model'
import { $hiddenTreePanes, $layoutTree, markCollapsePane, setTreeGroupMinimized } from '../store'

import { TreeSplit } from './tree-split'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const disposers: (() => void)[] = []

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  vi.stubGlobal('requestAnimationFrame', () => 1)
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
})

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())
  $paneStates.set({})

  disposers.push(
    registry.register({ area: 'panes', data: { placement: 'main' }, id: 'chat', render: () => null, title: 'Chat' }),
    registry.register({
      area: 'panes',
      data: { placement: 'main', width: '100px' },
      id: 'cron',
      render: () => null,
      title: 'Cron'
    }),
    registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'browser',
      render: () => null,
      title: 'Browser'
    })
  )
})

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  $paneStates.set({})
  disposers.splice(0).forEach(dispose => dispose())
})

function rect(width: number): DOMRect {
  return {
    bottom: 600,
    height: 600,
    left: 0,
    right: width,
    toJSON: () => ({}),
    top: 0,
    width,
    x: 0,
    y: 0
  } as DOMRect
}

function setWidth(element: HTMLElement, width: number) {
  Object.defineProperty(element, 'getBoundingClientRect', { configurable: true, value: () => rect(width) })
}

function setHeight(element: HTMLElement, height: number) {
  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({ ...rect(1000), bottom: height, height })
  })
}

function row(): SplitNode {
  const tree = $layoutTree.get()

  if (!tree || tree.type !== 'split') {
    throw new Error('expected root row split')
  }

  return tree
}

describe('TreeSplit cascading expansion', () => {
  it('grows Browser through Cron into Chat after Cron reaches its minimum', () => {
    const tree = split(
      'row',
      [
        group(['chat'], { id: 'chat-zone' }),
        group(['cron'], { id: 'cron-zone' }),
        group(['browser'], { id: 'browser-zone' })
      ],
      [5, 1, 2],
      'root-row'
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, cron, browser] = [...container.children] as HTMLElement[]
    setWidth(container, 800)
    setWidth(chat, 500)
    setWidth(cron, 100)
    setWidth(browser, 200)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="cron-zone"]')!, 100)

    const browserSash = document.querySelectorAll('[role="separator"]')[1]!
    fireEvent.pointerDown(browserSash, { button: 0, clientX: 600, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 300, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 300, pointerId: 1, pointerType: 'mouse' })

    // Browser's 300px requested growth first takes Cron from 100px to its
    // 80px floor, then takes the remaining 280px from Chat. The browser gets
    // every released pixel instead of stopping at Cron's local floor.
    expect($paneStates.get().cron?.widthOverride).toBe(80)
    expect(row().weights).toEqual([2.2, 1, 5])
  })
  it('commits a regular cascade when an unrelated tool rail is already minimized', () => {
    markCollapsePane('terminal')
    disposers.push(
      registry.register({
        area: 'panes',
        data: { maxWidth: '600px', minWidth: '160px', placement: 'right', width: '200px' },
        id: 'browser',
        render: () => null,
        title: 'Browser'
      }),
      registry.register({
        area: 'panes',
        data: { placement: 'bottom' },
        id: 'terminal',
        render: () => null,
        title: 'Terminal'
      })
    )

    const tree = split(
      'row',
      [
        group(['chat'], { id: 'chat-zone' }),
        group(['cron'], { id: 'cron-zone' }),
        group(['browser'], { id: 'browser-zone' }),
        group(['terminal'], { id: 'terminal-zone' })
      ],
      [5, 1, 2, 0.28],
      'root-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { open: true, widthOverride: 200 } })
    setTreeGroupMinimized('terminal-zone', true)

    render(<TreeSplit node={row()} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, cron, browser, terminal] = [...container.children] as HTMLElement[]
    setWidth(container, 828)
    setWidth(chat, 500)
    setWidth(cron, 100)
    setWidth(browser, 200)
    setWidth(terminal, 28)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="cron-zone"]')!, 100)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!, 200)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="terminal-zone"]')!, 28)

    const browserSash = document.querySelectorAll('[role="separator"]')[1]!
    fireEvent.pointerDown(browserSash, { button: 0, clientX: 600, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 300, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 300, pointerId: 1, pointerType: 'mouse' })

    expect($paneStates.get().cron?.widthOverride).toBe(80)
    expect($paneStates.get().browser?.widthOverride).toBe(500)
    expect(row().weights[0]).toBeCloseTo(2.2)
    expect(row().children[3]).toMatchObject({ id: 'terminal-zone', minimized: true })
  })
  it('folding a tool zone at its floor leaves no drag preview pinned on the flex sibling', () => {
    markCollapsePane('terminal')
    disposers.push(
      registry.register({
        area: 'panes',
        data: { height: '200px', placement: 'bottom' },
        id: 'terminal',
        render: () => null,
        title: 'Terminal'
      })
    )

    const tree = split(
      'column',
      [group(['chat'], { id: 'chat-zone' }), group(['terminal'], { id: 'terminal-zone' })],
      [1, 1],
      'root-column'
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-column"]')!
    const [chat, terminal] = [...container.children] as HTMLElement[]
    setHeight(container, 800)
    setHeight(chat, 600)
    setHeight(terminal, 200)
    setHeight(document.querySelector<HTMLElement>('[data-tree-group="terminal-zone"]')!, 200)

    const chatFlex = chat.style.flex
    const terminalSash = document.querySelectorAll('[role="separator"]')[0]!
    fireEvent.pointerDown(terminalSash, { button: 0, clientY: 600, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientY: 790, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientY: 790, pointerId: 1, pointerType: 'mouse' })

    // The zone folds to its rail (no sliver persisted) and the chat wrapper —
    // which the commit does not re-render — is back on React's own flex, not
    // the `0 1 <px>` pin the gesture previewed.
    expect(row().children[1]).toMatchObject({ id: 'terminal-zone', minimized: true })
    expect($paneStates.get().terminal?.heightOverride).toBeUndefined()
    expect(chat.style.flex).toBe(chatFlex)
  })
})
