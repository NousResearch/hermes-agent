import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'

import { group, split, type SplitNode } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { TreeSplit } from './tree-split'

// A minimized zone IS its 28px rail: the sash gesture skips it in preview and
// commit, so it must never OWN a seam. Folding the editor used to kill every
// seam around it — in a flat row both neighbors were disabled outright (no
// cursor, no drag), and on a nested section the outer seam stayed wired to
// the rail (drag wrote an override to an invisible width). The seam must
// pair with the first real track behind the rails instead.

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
      data: { placement: 'right', width: '300px' },
      id: 'editor',
      render: () => null,
      title: 'Editor'
    }),
    registry.register({
      area: 'panes',
      data: { maxWidth: '400px', placement: 'right', width: '200px' },
      id: 'files',
      render: () => null,
      title: 'Files'
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

function rootRow(section: SplitNode): SplitNode {
  return split('row', [group(['chat'], { id: 'chat-zone' }), section], [5, 2], 'root-row')
}

describe('TreeSplit seam ownership over a minimized edge zone', () => {
  it('resizes the sidebar BEHIND a minimized rail when the section seam is dragged', () => {
    const tree = rootRow(
      split(
        'row',
        [group(['editor'], { id: 'editor-zone', minimized: true }), group(['files'], { id: 'files-zone' })],
        [0.028, 0.2],
        'spl-right'
      )
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, section] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 772)
    setWidth(section, 228)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="editor-zone"]')!, 28)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!, 200)

    // The chat|section seam is the section wrapper's own sash — the first
    // separator in the DOM. Drag it 100px left: the section grows and the
    // pointer's pixels must land on FILES, the first non-rail zone inward.
    const seamSash = document.querySelectorAll('[role="separator"]')[0]!
    fireEvent.pointerDown(seamSash, { button: 0, clientX: 772, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 672, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 672, pointerId: 1, pointerType: 'mouse' })

    expect($paneStates.get().files?.widthOverride).toBe(300)
    // The rail owns no seam: writing a width the renderer ignores would strand
    // the gesture's pixels in an invisible override.
    expect($paneStates.get().editor?.widthOverride).toBeUndefined()
  })

  it('still gives the seam to the edge zone when it is NOT minimized', () => {
    const tree = rootRow(
      split(
        'row',
        [group(['editor'], { id: 'editor-zone' }), group(['files'], { id: 'files-zone' })],
        [0.3, 0.2],
        'spl-right'
      )
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, section] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 500)
    setWidth(section, 500)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="editor-zone"]')!, 300)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!, 200)

    const seamSash = document.querySelectorAll('[role="separator"]')[0]!
    fireEvent.pointerDown(seamSash, { button: 0, clientX: 500, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 400, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 400, pointerId: 1, pointerType: 'mouse' })

    // Existing contract (#87205): the edge zone owns the seam and takes the
    // growth; skipping must not reach past a live edge zone.
    expect($paneStates.get().editor?.widthOverride).toBe(400)
    expect($paneStates.get().files?.widthOverride).toBeUndefined()
  })

  it('keeps the tree resizable at its own seam when a flat row folds a neighbor into a rail', () => {
    const tree = split(
      'row',
      [group(['chat'], { id: 'chat-zone' }), group(['editor'], { id: 'editor-zone', minimized: true }), group(['files'], { id: 'files-zone' })],
      [5, 0.028, 2],
      'root-row'
    )

    $layoutTree.set(tree)

    render(<TreeSplit node={tree} root rootRow />)

    const container = document.querySelector<HTMLElement>('[data-tree-split="root-row"]')!
    const [chat, rail, files] = [...container.children] as HTMLElement[]
    setWidth(container, 1000)
    setWidth(chat, 672)
    setWidth(rail, 28)
    setWidth(files, 300)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="chat-zone"]')!, 672)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="editor-zone"]')!, 28)
    setWidth(document.querySelector<HTMLElement>('[data-tree-group="files-zone"]')!, 300)

    const [railSash, filesSash] = document.querySelectorAll('[role="separator"]')

    // The rail's OWN leading seam stays inert: a rail is 28px and nothing else.
    expect(railSash).toBeDefined()
    fireEvent.pointerDown(railSash!, { button: 0, clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 600, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 600, pointerId: 1, pointerType: 'mouse' })
    expect($paneStates.get().files?.widthOverride).toBeUndefined()

    // The tree's seam pairs past the rail with the chat: dragging it left
    // grows the tree from the chat's pixels; the rail rides along.
    fireEvent.pointerDown(filesSash!, { button: 0, clientX: 700, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerMove(window, { clientX: 650, pointerId: 1, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { clientX: 650, pointerId: 1, pointerType: 'mouse' })

    expect($paneStates.get().files?.widthOverride).toBe(350)
    // The rail owns none of it: pixels must never land on the invisible width.
    expect($paneStates.get().editor?.widthOverride).toBeUndefined()
    // The chat paid, so its weight shrank; the rail's remembered weight survives.
    const { weights } = $layoutTree.get() as SplitNode

    expect(weights[0]).toBeLessThan(5)
    expect(weights[1]).toBeCloseTo(0.028, 6)
  })
})
