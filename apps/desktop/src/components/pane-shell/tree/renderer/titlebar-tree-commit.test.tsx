import { useStore } from '@nanostores/react'
import { act, cleanup, render } from '@testing-library/react'
import { type ReactNode, useLayoutEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TITLEBAR_CHROME_CHANGED_EVENT } from '@/app/shell/titlebar'

import { type LayoutNode, mirrorTreeHorizontal, type SplitNode } from '../model'
import { $layoutTree } from '../store'

import { usePanelTitlebar } from './panel-titlebar'

import { LayoutTreeRoot } from './index'

// Isolate the root's commit seam, not a copy of its effect. Panels retain the
// real titlebar hook; unrelated shell services cannot accidentally heal them.
vi.mock('../store', async () => {
  const { atom } = await import('nanostores')

  return { $layoutTree: atom<LayoutNode | null>(null), trackActiveTreeGroup: () => {} }
})
vi.mock('../../edit-mode', () => ({ useLayoutEditHotkey: () => {} }))
vi.mock('../../geometry', () => ({ publishWorkspaceGeometry: () => {} }))
vi.mock('../tab-key-hint-state', () => ({ useTabKeyHints: () => {} }))
vi.mock('../zone-editor', () => ({ ZoneEditor: () => null }))
vi.mock('./edit-bar', () => ({ TreeEditBar: () => null }))
vi.mock('./floating-panes', () => ({ FloatingPanes: () => null }))
vi.mock('./narrow-overlays', () => ({ NarrowOverlays: () => null }))
vi.mock('./keep-alive-panes', () => ({ KeepAlivePanes: ({ children }: { children: ReactNode }) => children }))
vi.mock('./tree-node', () => ({ TreeNode: FixtureNode }))

const widths: Record<string, number> = { sidebar: 300, main: 900, a: 400, b: 400, c: 400 }

function FixtureNode({ node, topEdge }: { node: LayoutNode; topEdge: boolean }) {
  return node.type === 'split' ? (
    <div>
      {node.children.map(child => (
        <FixtureNode key={child.id} node={child} topEdge={topEdge} />
      ))}
    </div>
  ) : (
    <Panel enabled={topEdge} id={node.id} />
  )
}

function Panel({ enabled, id }: { enabled: boolean; id: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const below = usePanelTitlebar(ref, enabled, false)

  return <div data-below={below} data-tree-group={id} ref={ref} />
}

function tree(ids: string[]): SplitNode {
  return {
    type: 'split',
    id: 'root',
    orientation: 'row',
    weights: ids.map(id => widths[id]!),
    children: ids.map(id => ({ type: 'group', id, panes: [id], active: id }))
  }
}

function rect(left: number, width: number): DOMRect {
  return { bottom: 34, height: 34, left, right: left + width, top: 0, width, x: left, y: 0 } as DOMRect
}

const panels = () => [...globalThis.document.querySelectorAll<HTMLElement>('[data-tree-group]')]
const order = () => panels().map(panel => panel.dataset.treeGroup)

const reservation = (panel: HTMLElement, side: 'left' | 'right') =>
  panel.style.getPropertyValue(`--panel-titlebar-${side}`)

function expectSafeDragRegions() {
  for (const panel of panels()) {
    const bounds = panel.getBoundingClientRect()
    const left = bounds.left + Number.parseFloat(reservation(panel, 'left'))
    const right = bounds.right - Number.parseFloat(reservation(panel, 'right'))

    // Centers of the fixed no-drag controls at BOTH window edges. Native
    // app-region ignores their z-index; a stale reservation captures the click.
    for (const center of [110, 1140, 1180]) {
      expect(center >= left && center < right, `${panel.dataset.treeGroup} covers control at ${center}`).toBe(false)
    }
  }
}

beforeEach(() => {
  // Deliberately never deliver an RO callback or a window resize. Moving keyed
  // nodes at unchanged widths must be corrected by the root's actual commit.
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.titlebarCluster === 'left') {
      return rect(98, 90)
    }

    if (this.dataset.titlebarCluster === 'right') {
      return rect(1100, 100)
    }

    if (this.dataset.treeGroup) {
      const siblings = panels()
      const preceding = siblings.slice(0, siblings.indexOf(this))
      const left = preceding.reduce((sum, panel) => sum + widths[panel.dataset.treeGroup!]!, 0)

      return rect(left, widths[this.dataset.treeGroup]!)
    }

    return rect(0, 0)
  })
})

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function CommitProbe({ children, onCommit }: { children: ReactNode; onCommit?: () => void }) {
  const tree = useStore($layoutTree)

  // Ancestor layout effects run after the real root's layout effect, but before
  // passive effects/paint. A passive notification would leave this sample stale.
  useLayoutEffect(() => {
    onCommit?.()
  }, [tree, onCommit])

  return children
}

function fixture(titlebar = true, onCommit?: () => void) {
  return (
    <>
      <div data-titlebar-cluster="left" />
      <div data-titlebar-cluster="right" />
      <CommitProbe onCommit={onCommit}>
        <LayoutTreeRoot titlebar={titlebar} />
      </CommitProbe>
    </>
  )
}

describe('LayoutTreeRoot titlebar commit geometry', () => {
  it('refreshes both edges after repeated same-size keyed mirrors, after the DOM moves', () => {
    $layoutTree.set(tree(['sidebar', 'main']))

    const beforePaint = vi.fn(() =>
      panels().map(panel => ({
        id: panel.dataset.treeGroup,
        left: reservation(panel, 'left'),
        right: reservation(panel, 'right')
      }))
    )

    render(fixture(true, beforePaint))
    const [sidebar, main] = panels() as [HTMLElement, HTMLElement]
    const notify = vi.fn(() => order())
    window.addEventListener(TITLEBAR_CHROME_CHANGED_EVENT, notify)

    try {
      expect(reservation(sidebar, 'left')).toBe('200px')
      expect(reservation(main, 'right')).toBe('100px')
      expectSafeDragRegions()

      for (let i = 0; i < 4; i++) {
        act(() => $layoutTree.set(mirrorTreeHorizontal($layoutTree.get()!)))
        const mirrored = i % 2 === 0

        expect(panels()[0]).toBe(mirrored ? main : sidebar)
        expect(panels()[1]).toBe(mirrored ? sidebar : main)

        const expected = mirrored
          ? [
              { id: 'main', left: '200px', right: '0px' },
              { id: 'sidebar', left: '0px', right: '100px' }
            ]
          : [
              { id: 'sidebar', left: '200px', right: '0px' },
              { id: 'main', left: '0px', right: '100px' }
            ]

        expect(beforePaint.mock.results.at(-1)?.value).toEqual(expected)
        expect(notify).toHaveBeenCalledTimes(i + 1)
        expect(notify.mock.results.at(-1)?.value).toEqual(mirrored ? ['main', 'sidebar'] : ['sidebar', 'main'])
        expect(reservation(main, 'left')).toBe(mirrored ? '200px' : '0px')
        expect(reservation(sidebar, 'right')).toBe(mirrored ? '100px' : '0px')
        expectSafeDragRegions()
      }
    } finally {
      window.removeEventListener(TITLEBAR_CHROME_CHANGED_EVENT, notify)
    }
  })

  it('also refreshes an equal-size three-panel reorder that is not a mirror', () => {
    const initial = tree(['a', 'b', 'c'])
    $layoutTree.set(initial)
    render(fixture())
    const [a, b, c] = panels() as [HTMLElement, HTMLElement, HTMLElement]

    act(() =>
      $layoutTree.set({ ...initial, children: [initial.children[1]!, initial.children[2]!, initial.children[0]!] })
    )

    for (const [index, panel] of [b, c, a].entries()) {
      expect(panels()[index]).toBe(panel)
    }

    expect(reservation(b, 'left')).toBe('200px')
    expect(reservation(a, 'right')).toBe('100px')
    expect(reservation(c, 'right')).toBe('0px')
    expectSafeDragRegions()
  })

  it('does not notify for non-titlebar roots or unrelated root rerenders', () => {
    $layoutTree.set(tree(['sidebar', 'main']))
    const notify = vi.fn()
    window.addEventListener(TITLEBAR_CHROME_CHANGED_EVENT, notify)

    try {
      const { rerender } = render(fixture(false))
      act(() => $layoutTree.set(mirrorTreeHorizontal($layoutTree.get()!)))
      expect(notify).not.toHaveBeenCalled()

      rerender(fixture(true))
      expect(notify).toHaveBeenCalledTimes(1)
      expectSafeDragRegions()
      rerender(fixture(true))
      expect(notify).toHaveBeenCalledTimes(1)
    } finally {
      window.removeEventListener(TITLEBAR_CHROME_CHANGED_EVENT, notify)
    }
  })
})
