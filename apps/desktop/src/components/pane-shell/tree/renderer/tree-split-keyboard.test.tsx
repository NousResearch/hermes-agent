import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'
import { $shellChoice } from '@/store/shell-variant'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { TreeSplit } from './tree-split'

const observers: ((entries: unknown[]) => void)[] = []

class TestResizeObserver {
  constructor(callback: (entries: unknown[]) => void) {
    observers.push(callback)
  }

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
  observers.length = 0

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
  $shellChoice.set(null)
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

const press = (sash: Element, key: string, init: Partial<KeyboardEventInit> = {}) =>
  fireEvent.keyDown(sash, { key, ...init })

describe('the redesign’s sash is a keyboard window splitter', () => {
  it('is focusable and named, with a value in percent', () => {
    const { sash } = mountRow('left')
    act(() => observers.forEach(measure => measure([])))

    expect(sash.getAttribute('tabindex')).toBe('0')
    expect(sash.getAttribute('aria-label')).toBe('Resize panes')
    expect(sash.getAttribute('aria-orientation')).toBe('vertical')
    expect(sash.getAttribute('aria-valuemin')).toBe('0')
    expect(sash.getAttribute('aria-valuemax')).toBe('100')
    expect(sash.getAttribute('aria-valuenow')).toBe('30')
  })

  it('moves the seam a step per arrow (Shift: a larger one) and stops at the clamps on Home/End', () => {
    const { sash } = mountRow('left')

    press(sash, 'ArrowRight')
    expect($paneStates.get().sessions?.widthOverride).toBe(316)

    press(sash, 'ArrowLeft', { shiftKey: true })
    // 316 − 64 is under the 260px minimum: the seam stops at the clamp.
    expect($paneStates.get().sessions?.widthOverride).toBe(260)

    press(sash, 'End')
    expect($paneStates.get().sessions?.widthOverride).toBe(360)

    press(sash, 'Home')
    expect($paneStates.get().sessions?.widthOverride).toBe(260)
    expect(fold).not.toHaveBeenCalled()
  })

  it('folds a side that folds on a deep drag when Enter is pressed', () => {
    const { sash } = mountRow('left')

    press(sash, 'Enter')
    expect(fold).toHaveBeenCalledTimes(1)
  })

  it('ignores keys off its axis', () => {
    const { sash } = mountRow('left')

    press(sash, 'ArrowDown')
    expect($paneStates.get().sessions?.widthOverride).toBeUndefined()
  })
})

describe('the classic shell keeps main’s pointer-only seam', () => {
  it('is not focusable and takes no keys', () => {
    $shellChoice.set(false)
    const { sash } = mountRow('left')

    expect(sash.hasAttribute('tabindex')).toBe(false)
    expect(sash.hasAttribute('aria-valuenow')).toBe(false)
    press(sash, 'ArrowRight')
    expect($paneStates.get().sessions?.widthOverride).toBeUndefined()
  })
})
