import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'

import { $layoutEditMode } from '../../edit-mode'
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
  $layoutEditMode.set(false)
  disposers.splice(0).forEach(dispose => dispose())
})

function rect(width: number, height = 600): DOMRect {
  return {
    bottom: height,
    height,
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
  Object.defineProperty(element, 'getBoundingClientRect', { configurable: true, value: () => rect(800, height) })
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

  it('does not move a shared column boundary owned by a locked descendant', () => {
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'right', width: '150px' }, id: 'review', render: () => null, title: 'Review' }),
      registry.register({ area: 'panes', data: { placement: 'right', width: '150px' }, id: 'files', render: () => null, title: 'Files' }),
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    const topRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'top-rail'
    )

    const rightRail = split('column', [topRail, group(['terminal'], { id: 'terminal-zone' })], [1, 1], 'right-rail')
    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), rightRail], [5, 3], 'root-row')

    $layoutTree.set(tree)
    $paneStates.set({ terminal: { open: true, widthLocked: true, widthOverride: 300 } })

    render(<TreeSplit node={tree} root rootRow />)

    const container = globalThis.document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, rightRailElement] = [...container.children] as HTMLElement[]
    setWidth(container, 800)
    setWidth(chat, 500)
    setWidth(rightRailElement, 300)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="review-zone"]')!, 150)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!, 150)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="terminal-zone"]')!, 300)

    const rootSash = rightRailElement.querySelector<HTMLElement>(':scope > [role="separator"]')!
    fireEvent.pointerDown(rootSash, { button: 0, clientX: 500, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 400, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 400, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerDown(rootSash, { button: 0, clientX: 500, pointerId: 2, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 600, pointerId: 2, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 600, pointerId: 2, pointerType: 'mouse' })
    fireEvent.doubleClick(rootSash)

    expect($paneStates.get().terminal).toMatchObject({ widthLocked: true, widthOverride: 300 })
    expect($paneStates.get().review?.widthOverride).toBeUndefined()
    expect($paneStates.get().files?.widthOverride).toBeUndefined()
    expect(row().weights).toEqual([5, 3])
  })

  it('routes an outer seam into an unlocked sibling beside a local lock', () => {
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'right', width: '300px' }, id: 'review', render: () => null, title: 'Review' }),
      registry.register({ area: 'panes', data: { placement: 'right', width: '200px' }, id: 'files', render: () => null, title: 'Files' }),
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    const topRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'top-rail'
    )
    const rightRail = split('column', [topRail, group(['terminal'], { id: 'terminal-zone' })], [1, 1], 'right-rail')
    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), rightRail], [1, 1], 'root-row')

    $layoutTree.set(tree)
    $paneStates.set({
      files: { open: true, widthOverride: 200 },
      review: { open: true, widthLocked: true, widthOverride: 300 }
    })

    render(<TreeSplit node={tree} root rootRow />)

    const container = globalThis.document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, rightRailElement] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 500)
    setWidth(rightRailElement, 500)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="review-zone"]')!, 300)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!, 200)

    const rootSash = rightRailElement.querySelector<HTMLElement>(':scope > [role="separator"]')!
    fireEvent.pointerDown(rootSash, { button: 0, clientX: 500, pointerId: 3, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 560, pointerId: 3, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 560, pointerId: 3, pointerType: 'mouse' })

    expect($paneStates.get().review).toMatchObject({ widthLocked: true, widthOverride: 300 })
    expect($paneStates.get().files).toMatchObject({ widthOverride: 140 })
  })

  it('never stretches a locked uncapped track as the all-fixed absorber', () => {
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'right', width: '200px' }, id: 'review', render: () => null, title: 'Review' }),
      registry.register({ area: 'panes', data: { placement: 'right', width: '200px' }, id: 'files', render: () => null, title: 'Files' })
    )

    const tree = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'fixed-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({
      files: { open: true, widthLocked: true, widthOverride: 200 },
      review: { open: true, widthOverride: 200 }
    })

    render(<TreeSplit node={tree} root rootRow />)

    const container = globalThis.document.querySelector<HTMLElement>('[data-tree-split="fixed-row"]')!
    const [review, files] = [...container.children] as HTMLElement[]

    expect(review.style.flex).toBe('1 1 200px')
    expect(files.style.flex).toBe('0 1 200px')
  })

  it('does not move a directly height-locked Terminal boundary', () => {
    disposers.push(
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    const tree = split(
      'column',
      [group(['browser'], { id: 'browser-zone' }), group(['terminal'], { id: 'terminal-zone' })],
      [3, 2],
      'right-column'
    )

    $layoutTree.set(tree)
    $paneStates.set({ terminal: { heightLocked: true, heightOverride: 260, open: true } })

    render(<TreeSplit node={tree} root />)

    const container = globalThis.document.querySelector<HTMLElement>('[data-tree-split="right-column"]')!
    const [browser, terminal] = [...container.children] as HTMLElement[]
    setHeight(container, 600)
    setHeight(browser, 340)
    setHeight(terminal, 260)
    setHeight(globalThis.document.querySelector<HTMLElement>('[data-tree-group="terminal-zone"]')!, 260)

    const sash = terminal.querySelector<HTMLElement>(':scope > [role="separator"]')!
    fireEvent.pointerDown(sash, { button: 0, clientY: 340, pointerId: 4, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientY: 240, pointerId: 4, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientY: 240, pointerId: 4, pointerType: 'mouse' })
    fireEvent.pointerDown(sash, { button: 0, clientY: 340, pointerId: 5, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientY: 440, pointerId: 5, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientY: 440, pointerId: 5, pointerType: 'mouse' })

    expect($paneStates.get().terminal).toMatchObject({ heightLocked: true, heightOverride: 260 })
    expect($paneStates.get().browser?.heightOverride).toBeUndefined()
    expect(row().weights).toEqual([3, 2])
  })

  it('fills a locked shared column with unlocked capped siblings', () => {
    disposers.push(
      registry.register({
        area: 'panes',
        data: { maxWidth: '320px', placement: 'right', width: '160px' },
        id: 'review',
        render: () => null,
        title: 'Review'
      }),
      registry.register({
        area: 'panes',
        data: { maxWidth: '320px', placement: 'right', width: '160px' },
        id: 'files',
        render: () => null,
        title: 'Files'
      }),
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    const topRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'top-rail'
    )

    const rightRail = split('column', [topRail, group(['terminal'], { id: 'terminal-zone' })], [1, 1], 'right-rail')
    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), rightRail], [5, 3], 'root-row')

    $layoutTree.set(tree)
    $paneStates.set({ terminal: { open: true, widthLocked: true, widthOverride: 420 } })

    render(<TreeSplit node={tree} root rootRow />)

    const topRailElement = globalThis.document.querySelector<HTMLElement>('[data-tree-split="top-rail"]')!
    const [review, files] = [...topRailElement.children] as HTMLElement[]

    expect(review.style.flex).toBe('0 1 160px')
    expect(files.style.flex).toBe('1 1 160px')
    expect(files.style.maxWidth).toBe('')
    expect($paneStates.get().review).toBeUndefined()
    expect($paneStates.get().files).toBeUndefined()

    setWidth(topRailElement, 420)
    setWidth(review, 160)
    setWidth(files, 260)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="review-zone"]')!, 160)
    setWidth(globalThis.document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!, 260)

    const innerSash = files.querySelector<HTMLElement>(':scope > [role="separator"]')!
    fireEvent.pointerDown(innerSash, { button: 0, clientX: 160, pointerId: 3, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 200, pointerId: 3, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 200, pointerId: 3, pointerType: 'mouse' })

    expect($paneStates.get().review?.widthOverride).toBe(200)
    expect($paneStates.get().files?.widthOverride).toBe(220)
    expect($paneStates.get().terminal).toMatchObject({ widthLocked: true, widthOverride: 420 })
  })

  it('fills when the locked shared split is itself the layout root', () => {
    disposers.push(
      registry.register({
        area: 'panes',
        data: { maxWidth: '320px', placement: 'right', width: '160px' },
        id: 'review',
        render: () => null,
        title: 'Review'
      }),
      registry.register({
        area: 'panes',
        data: { maxWidth: '320px', placement: 'right', width: '160px' },
        id: 'files',
        render: () => null,
        title: 'Files'
      }),
      registry.register({ area: 'panes', data: { placement: 'bottom' }, id: 'terminal', render: () => null, title: 'Terminal' })
    )

    const topRail = split(
      'row',
      [group(['review'], { id: 'review-zone' }), group(['files'], { id: 'files-zone' })],
      [1, 1],
      'top-rail'
    )

    const rightRail = split('column', [topRail, group(['terminal'], { id: 'terminal-zone' })], [1, 1], 'right-rail')

    $layoutTree.set(rightRail)
    $paneStates.set({ terminal: { open: true, widthLocked: true, widthOverride: 420 } })

    render(<TreeSplit node={rightRail} root />)

    const topRailElement = globalThis.document.querySelector<HTMLElement>('[data-tree-split="top-rail"]')!
    const [, files] = [...topRailElement.children] as HTMLElement[]

    expect(files.style.flex).toBe('1 1 160px')
    expect(files.style.maxWidth).toBe('')
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

describe('TreeSplit sash in edit mode', () => {
  it('raises the sash above the edit veil so dividers stay grabbable', () => {
    const tree = split('row', [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })], [1, 1], 'root-row')
    $layoutTree.set(tree)

    // Edit mode OFF: the sash sits at its normal z-20.
    render(<TreeSplit node={tree} root rootRow />)
    const sash = document.querySelectorAll('[role="separator"]')[0]!
    expect(sash.className).toContain('z-20')
    expect(sash.className).not.toContain('z-[60]')
    cleanup()

    // Edit mode ON: the veil paints z-50 over the pane body, so the sash must
    // climb to z-60 to stay reachable while arranging.
    $layoutEditMode.set(true)
    render(<TreeSplit node={tree} root rootRow />)
    const editSash = document.querySelectorAll('[role="separator"]')[0]!
    expect(editSash.className).toContain('z-[60]')
    expect(editSash.className).not.toContain('z-20')
  })
})
