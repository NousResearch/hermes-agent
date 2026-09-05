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
    registry.register({ area: 'panes', data: { placement: 'main', width: '100px' }, id: 'cron', render: () => null, title: 'Cron' }),
    registry.register({ area: 'panes', data: { placement: 'main' }, id: 'browser', render: () => null, title: 'Browser' })
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

  it('skips a locked middle zone and continues cascading to the outer donor', () => {
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
    $paneStates.set({ cron: { open: true, widthLocked: true, widthOverride: 100 } })

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

    expect($paneStates.get().cron).toMatchObject({ widthLocked: true, widthOverride: 100 })
    expect(row().weights).toEqual([2, 1, 5])
  })

  it('keeps the return direction local after a cascading drag reverses', () => {
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'main' }, id: 'notes', render: () => null, title: 'Notes' }),
      registry.register({ area: 'panes', data: { placement: 'main' }, id: 'preview', render: () => null, title: 'Preview' })
    )

    const tree = split(
      'row',
      [
        group(['chat'], { id: 'chat-zone' }),
        group(['browser'], { id: 'browser-zone' }),
        group(['notes'], { id: 'notes-zone' }),
        group(['preview'], { id: 'preview-zone' })
      ],
      [3, 2, 2, 1],
      'root-row'
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, browser, notes, preview] = [...container.children] as HTMLElement[]
    setWidth(container, 800)
    setWidth(chat, 300)
    setWidth(browser, 200)
    setWidth(notes, 200)
    setWidth(preview, 100)

    const middleSash = document.querySelectorAll('[role="separator"]')[1]!
    fireEvent.pointerDown(middleSash, { button: 0, clientX: 500, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 250, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 800, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 800, pointerId: 1, pointerType: 'mouse' })

    // The first (leftward) motion cascades through Browser into Chat. Returning
    // right expands Browser from Notes only; Preview was never involved.
    expect(row().weights).toEqual([3, 3.2, 0.8, 1])
  })

  it('does not let a tiny opposite false-start disable a later forward cascade', () => {
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
    // A tiny initial wobble grows Cron locally. The real movement then grows
    // Browser leftward and must still cascade through Cron into Chat.
    fireEvent.pointerMove(window, { clientX: 605, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 300, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 300, pointerId: 1, pointerType: 'mouse' })

    expect($paneStates.get().cron?.widthOverride).toBe(80)
    expect(row().weights).toEqual([2.2, 1, 5])
  })

  it('does not leak past a locked immediate donor in the local tool-panel path', () => {
    markCollapsePane('tool')
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'main' }, id: 'tool', render: () => null, title: 'Tool' }),
      registry.register({ area: 'panes', data: { placement: 'main' }, id: 'notes', render: () => null, title: 'Notes' })
    )

    const tree = split(
      'row',
      [group(['tool'], { id: 'tool-zone' }), group(['browser'], { id: 'browser-zone' }), group(['notes'], { id: 'notes-zone' })],
      [1, 1, 6],
      'root-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { open: true, widthLocked: true, widthOverride: 100 } })

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [tool, browser, notes] = [...container.children] as HTMLElement[]
    setWidth(container, 800)
    setWidth(tool, 100)
    setWidth(browser, 100)
    setWidth(notes, 600)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!, 100)

    const notesSash = document.querySelectorAll('[role="separator"]')[1]!
    fireEvent.pointerDown(notesSash, { button: 0, clientX: 200, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 0, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 0, pointerId: 1, pointerType: 'mouse' })

    expect(row().weights).toEqual([1, 1, 6])
    expect($paneStates.get().browser).toMatchObject({ widthLocked: true, widthOverride: 100 })
  })

  it('restores a collapsed tool column with an 80px usable width floor', () => {
    markCollapsePane('terminal')
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    const tree = split(
      'row',
      [group(['chat'], { id: 'chat-zone' }), group(['terminal'], { id: 'terminal-zone' })],
      [5, 0.01],
      'root-row'
    )

    $layoutTree.set(tree)

    const view = render(<TreeSplit node={tree} root rootRow />)
    setTreeGroupMinimized('terminal-zone', true)
    view.rerender(<TreeSplit node={$layoutTree.get() as SplitNode} root rootRow />)
    setTreeGroupMinimized('terminal-zone', false)
    view.rerender(<TreeSplit node={$layoutTree.get() as SplitNode} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const terminalColumn = container.children[1] as HTMLElement

    expect(terminalColumn.style.minWidth).toBe('80px')
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

  it('does not let an unrelated collapsible rail disable a regular seam cascade', () => {
    markCollapsePane('bot')
    disposers.push(registry.register({ area: 'panes', data: { placement: 'main' }, id: 'bot', render: () => null, title: 'Bot' }))

    const tree = split(
      'row',
      [
        group(['bot'], { id: 'bot-zone' }),
        group(['chat'], { id: 'chat-zone' }),
        group(['cron'], { id: 'cron-zone' }),
        group(['browser'], { id: 'browser-zone' })
      ],
      [1, 5, 1, 2],
      'root-row'
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [bot, chat, cron, browser] = [...container.children] as HTMLElement[]
    setWidth(container, 900)
    setWidth(bot, 100)
    setWidth(chat, 500)
    setWidth(cron, 100)
    setWidth(browser, 200)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="cron-zone"]')!, 100)

    const browserSash = document.querySelectorAll('[role="separator"]')[2]!
    fireEvent.pointerDown(browserSash, { button: 0, clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 400, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 400, pointerId: 1, pointerType: 'mouse' })

    // Browser takes Cron to its floor, then continues through Chat. The Bot
    // rail is unrelated to this seam and must not suppress that cascade.
    expect($paneStates.get().cron?.widthOverride).toBe(80)
    expect(row().weights).toEqual([1, 2.2, 1, 5])
  })
})
