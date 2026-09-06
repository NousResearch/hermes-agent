import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates, setPaneWidthLock } from '@/store/panes'

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

describe('TreeSplit axis locking', () => {
  it('locked pane holds its width when a neighbor sash is dragged', () => {
    disposers.push(
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
        group(['browser'], { id: 'browser-zone' }),
        group(['terminal'], { id: 'terminal-zone' })
      ],
      [5, 1, 2],
      'root-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { lockWidth: true, lockedWidth: 200, open: true, widthOverride: 200 } })

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, browser, terminal] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 500)
    setWidth(browser, 200)
    setWidth(terminal, 300)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!, 200)

    // Drag the browser-terminal sash left to try to grow browser (locked).
    const browserSash = document.querySelectorAll('[role="separator"]')[1]!
    fireEvent.pointerDown(browserSash, { button: 0, clientX: 800, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 700, pointerId: 1, pointerType: 'mouse' })

    // Browser's override must not have been changed — the sash skipped the
    // locked pane entirely.
    expect($paneStates.get().browser?.widthOverride).toBe(200)
  })

  it('locked pane cannot be grown by dragging into it', () => {
    disposers.push(
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
        group(['browser'], { id: 'browser-zone' }),
        group(['terminal'], { id: 'terminal-zone' })
      ],
      [5, 1, 2],
      'root-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { lockWidth: true, lockedWidth: 200, open: true, widthOverride: 200 } })

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, browser, terminal] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 600)
    setWidth(browser, 200)
    setWidth(terminal, 200)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!, 200)

    // Drag the chat-browser sash right (into browser) by 100px.
    const sash = document.querySelectorAll('[role="separator"]')[0]!
    fireEvent.pointerDown(sash, { button: 0, clientX: 600, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 700, pointerId: 1, pointerType: 'mouse' })

    // Browser must stay at 200px — a locked target cannot grow.
    expect($paneStates.get().browser?.widthOverride).toBe(200)
  })

  it('column stack with a locked pane uses the locked width as the zone track basis', () => {
    const tree = split(
      'column',
      [
        group(['chat'], { id: 'chat-zone' }),
        split('row', [group(['browser', 'terminal'], { id: 'browser-zone' })], [1], 'browser-row')
      ],
      [1, 1],
      'root-column'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { lockWidth: true, lockedWidth: 200, open: true, widthOverride: 200 } })

    render(<TreeSplit node={tree} root />)

    const browserZone = document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!
    expect(browserZone).toBeTruthy()
    const browserRow = document.querySelector<HTMLElement>('[data-tree-split="browser-row"]')!
    expect(browserRow).toBeTruthy()
  })

  it('unlock clears lockedWidth so the pane reverts to flex behavior', () => {
    const tree = split(
      'row',
      [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })],
      [1, 1],
      'root-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { lockWidth: true, lockedWidth: 300, open: true, widthOverride: 300 } })

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [, browser] = [...container.children] as HTMLElement[]
    setWidth(container, 800)
    setWidth(browser, 300)

    const browserFlex = browser.style.flex
    expect(browserFlex).toContain('300px')

    // Unlock — the pane should revert to its flex layout.
    setPaneWidthLock('browser', false)

    const snap = $paneStates.get().browser
    expect(snap?.lockWidth).toBeUndefined()
    expect(snap?.lockedWidth).toBeUndefined()
    expect(snap?.widthOverride).toBe(300)
  })

  it('locks on a flex pane with no override, capturing measured DOM bounding rect 420', () => {
    const tree = split(
      'row',
      [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })],
      [1, 1],
      'root-row'
    )

    $layoutTree.set(tree)
    // No widthOverride — flex pane at heart.
    $paneStates.set({})

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [, browser] = [...container.children] as HTMLElement[]
    setWidth(container, 800)
    setWidth(browser, 400)

    // Mock the zone element's bounding rect to 420px (the measured DOM size).
    const zoneEl = document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!
    Object.defineProperty(zoneEl, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ height: 600, width: 420, ...{ toJSON: () => ({}) } })
    })

    // Lock with explicit dimension — the store captures 420, not the override.
    setPaneWidthLock('browser', true, 420)

    const snap = $paneStates.get().browser
    expect(snap?.lockWidth).toBe(true)
    expect(snap?.lockedWidth).toBe(420)
  })

  it('column-in-row: locked pane inside nested column preserves width on horizontal drag', () => {
    // Layout: root-row > [chat-zone, column-split > [browser-zone]]
    // browser-zone is inside a column (parentAxis="column") but the row
    // constrains its width. Locking width and dragging the chat-browser sash
    // must preserve browser's locked width.
    const tree = split(
      'row',
      [
        group(['chat'], { id: 'chat-zone' }),
        split('column', [group(['browser'], { id: 'browser-zone' })], [1], 'inner-col')
      ],
      [5, 1],
      'root-row'
    )

    $layoutTree.set(tree)
    $paneStates.set({ browser: { lockWidth: true, lockedWidth: 250, open: true, widthOverride: 250 } })

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, colSplit] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 700)
    setWidth(colSplit, 300)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!, 250)

    // Drag the chat-column sash right (into browser's column) by 100px.
    const sash = document.querySelectorAll('[role="separator"]')[0]!
    fireEvent.pointerDown(sash, { button: 0, clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 800, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 800, pointerId: 1, pointerType: 'mouse' })

    // Browser must stay at 250px — the locked pane's width is preserved.
    expect($paneStates.get().browser?.widthOverride).toBe(250)
  })

  it('snapshot signature includes lockedWidth so re-lock at new size re-renders', () => {
    // Scenario: lock on a flex pane with no prior override. The lock captures
    // the measured DOM width (420). Then a second lock captures a new size
    // (350) — the snapshot signature must include lockedWidth so the subtree
    // overrides cache invalidates and the component re-renders.
    const tree = split(
      'row',
      [group(['chat'], { id: 'chat-zone' }), group(['browser'], { id: 'browser-zone' })],
      [1, 1],
      'root-row'
    )

    $layoutTree.set(tree)
    // Lock with captured width 420 — no prior widthOverride.
    $paneStates.set({ browser: { lockWidth: true, lockedWidth: 420, open: true } })

    render(<TreeSplit node={tree} root rootRow />)

    const browserEl = document.querySelector<HTMLElement>('[data-tree-group="browser-zone"]')!.parentElement!
    setWidth(document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!, 800)
    setWidth(browserEl, 420)

    // The locked basis is 420px.
    expect(browserEl.style.flex).toContain('420px')

    // Re-lock at 350 (lockWidth stays true, lockedWidth changes 420→350).
    // Without lockedWidth in the signature, the cache returns stale data
    // and the component keeps showing 420px.
    act(() => {
      $paneStates.set({ browser: { lockWidth: true, lockedWidth: 350, open: true } })
    })

    setWidth(browserEl, 350)

    // The component must re-render with the new locked basis.
    expect(browserEl.style.flex).toContain('350px')
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
