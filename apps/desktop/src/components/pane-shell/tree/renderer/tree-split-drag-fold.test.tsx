import { cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { TreeSplit } from './tree-split'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const disposers: (() => void)[] = []
const fold = vi.fn()

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  // A frame per microtask: each move previews on its own frame, as a real drag's do.
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    queueMicrotask(() => cb(0))

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
})

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())
  $paneStates.set({})
  fold.mockClear()

  // jsdom has no layout: an authored length measures as its own px value (resolveCssPx's probe).
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    return rect(Number.parseFloat(this.style.width) || 0)
  })

  disposers.push(
    registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'workspace',
      render: () => null,
      title: 'Chat'
    }),
    registry.register({
      area: 'panes',
      data: {
        dragCollapseBelow: '190px',
        maxWidth: '360px',
        minWidth: '260px',
        onDragCollapse: fold,
        placement: 'left',
        width: '300px'
      },
      id: 'sessions',
      render: () => null,
      title: 'Sessions'
    })
  )
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  $layoutTree.set(null)
  $paneStates.set({})
  disposers.splice(0).forEach(dispose => dispose())
})

function rect(width: number): DOMRect {
  return { bottom: 600, height: 600, left: 0, right: width, toJSON: () => ({}), top: 0, width, x: 0, y: 0 } as DOMRect
}

function setWidth(element: HTMLElement, width: number) {
  Object.defineProperty(element, 'getBoundingClientRect', { configurable: true, value: () => rect(width) })
}

/** The sidebar on `side` of the workspace, mounted with measured widths; returns its wrapper, the
 *  workspace's, and the seam's sash. */
function mountRow(side: 'left' | 'right') {
  const sessions = group(['sessions'], { id: 'grp-sessions' })
  const main = group(['workspace'], { id: 'grp-main' })
  const tree = split('row', side === 'left' ? [sessions, main] : [main, sessions], [1, 3], 'spl-root')

  $layoutTree.set(tree)

  const { container: host } = render(
    <MemoryRouter>
      <TreeSplit node={tree} root rootRow />
    </MemoryRouter>
  )

  const container = host.querySelector<HTMLElement>('[data-tree-split="spl-root"]')!
  const [first, second] = [...container.children] as HTMLElement[]
  const [sidebar, workspace] = side === 'left' ? [first, second] : [second, first]

  setWidth(container, 1000)
  setWidth(sidebar, 300)
  setWidth(workspace, 700)
  setWidth(host.querySelector<HTMLElement>('[data-tree-group="grp-sessions"]')!, 300)

  return { sash: second.querySelector('[role="separator"]')!, sidebar, workspace }
}

const move = async (to: number) => {
  fireEvent.pointerMove(window, { clientX: to, pointerId: 1, pointerType: 'mouse' })
  await Promise.resolve()
}

const drag = async (sash: Element, from: number, to: number) => {
  fireEvent.pointerDown(sash, { button: 0, clientX: from, pointerId: 1, pointerType: 'mouse' })
  await move(to)
}

const release = (to: number) => fireEvent.pointerUp(window, { clientX: to, pointerId: 1, pointerType: 'mouse' })

describe('a sash drag past a pane’s fold length (Figma 1054:22350)', () => {
  it('rests at the minimum short of the fold length', async () => {
    const { sash, sidebar } = mountRow('left')

    await drag(sash, 300, 220)
    expect(sidebar.style.flexBasis).toBe('260px')
    release(220)

    expect(fold).not.toHaveBeenCalled()
    expect($paneStates.get().sessions?.widthOverride).toBe(260)
  })

  it('folds the pane live as the pointer crosses, unfolds on the way back, and folds on release', async () => {
    const { sash, sidebar, workspace } = mountRow('left')
    const restingMin = sidebar.style.minWidth

    await drag(sash, 300, 100)
    // Folded: the pane gives up its clamp and its whole width to the other side of the seam.
    expect(sidebar.style.flexBasis).toBe('0px')
    expect(sidebar.style.minWidth).toBe('0px')
    expect(workspace.style.flex).toBe('0 1 1000px')

    await move(250)
    expect(sidebar.style.minWidth).toBe(restingMin)
    expect(sidebar.style.flexBasis).toBe('260px')

    await move(100)
    release(100)

    expect(fold).toHaveBeenCalledOnce()
    // No size is written, so the pane comes back at the width it had before the drag.
    expect($paneStates.get().sessions?.widthOverride).toBeUndefined()
  })

  it('keeps the width the user sized before the drag for when the pane returns', async () => {
    setPaneWidthOverride('sessions', 320)
    const { sash } = mountRow('left')

    await drag(sash, 300, 90)
    release(90)

    expect(fold).toHaveBeenCalledOnce()
    expect($paneStates.get().sessions?.widthOverride).toBe(320)
  })

  it('folds a pane on the trailing side of the seam when dragged toward it', async () => {
    const { sash } = mountRow('right')

    await drag(sash, 700, 900)
    release(900)

    expect(fold).toHaveBeenCalledOnce()
    expect($paneStates.get().sessions?.widthOverride).toBeUndefined()
  })

  it('never folds where the fold length resolves to nothing (the classic shell)', async () => {
    disposers.push(
      registry.register({
        area: 'panes',
        data: {
          dragCollapseBelow: 'var(--undefined-fold, 0px)',
          minWidth: '260px',
          onDragCollapse: fold,
          placement: 'left',
          width: '300px'
        },
        id: 'sessions',
        render: () => null,
        title: 'Sessions'
      })
    )
    const { sash } = mountRow('left')

    await drag(sash, 300, 0)
    release(0)

    expect(fold).not.toHaveBeenCalled()
    expect($paneStates.get().sessions?.widthOverride).toBe(260)
  })
})
